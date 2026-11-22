import json
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_wire_holdout_resets import prepare_holdout_variant

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "configs/dh116_training/supported_wire_generalization_factorial_holdout_v1.json").read_bytes())
BASELINE = json.loads((ROOT / CATALOG["reference_rod"]["path"]).read_bytes())["rod"]
TABLE = dict(center_m=[.6, 0., .25], size_m=[.8, 1.2, .1])


def test_heldout_placement_changes_observed_wire_without_retargeting_either_hand():
    scene = json.loads((ROOT / "configs/dh116_tasks/flexible_power_supported_contact_v1.json").read_bytes())
    original = json.loads(json.dumps(scene))
    heldout = [case["id"] for case in CATALOG["variants"] if case["split"] == "test"]
    opts = dict(catalog=CATALOG, stations=dict(left=30, right=10), placement=(.002, -.003, .025))
    a = prepare_holdout_variant(scene, TABLE, BASELINE, variant_id=heldout[0], **opts)
    b = prepare_holdout_variant(scene, TABLE, BASELINE, variant_id=heldout[-1], **opts)
    repeat = prepare_holdout_variant(scene, TABLE, BASELINE, variant_id=heldout[0], **opts)
    same_wire_other_pose = prepare_holdout_variant(scene, TABLE, BASELINE, variant_id=heldout[0],
        **dict(opts, placement=(.002, .003, -.025)))
    assert scene == original
    assert a["reset_seed"]["reset_root_pose_w"] == b["reset_seed"]["reset_root_pose_w"]
    assert a["reset_seed"]["reset_root_pose_w"] == same_wire_other_pose["reset_seed"]["reset_root_pose_w"]
    assert a["scene"]["hands"] == scene["hands"]
    assert a["physical_initial_condition_sha256"] == repeat["physical_initial_condition_sha256"]
    assert a["physical_initial_condition_sha256"] != same_wire_other_pose["physical_initial_condition_sha256"]
    assert a["physical_initial_condition_sha256"] != b["physical_initial_condition_sha256"]
    nodes = np.array(a["scene"]["rod_nodes_w_m"])
    base_nodes = np.array(scene["rod_nodes_w_m"])
    np.testing.assert_allclose(nodes.mean(axis=0)[:2] - base_nodes.mean(axis=0)[:2], [.002, -.003])
    assert np.linalg.norm(nodes[-1] - nodes[0]) == pytest.approx(a["length_m"])
    assert np.isclose((nodes[-1] - nodes[0])[0], -a["length_m"] * np.sin(.025))
    assert not a["native_reset_verified"]
    for support in a["scene"]["support_scene"]["supports"]:
        center, size = np.array(support["center_m"]), np.array(support["size_m"])
        assert np.any((np.abs(nodes[:, :2] - center[:2]) <= size[:2] / 2).all(axis=1))
        assert center[2] - size[2] / 2 == pytest.approx(.30)
        assert center[2] + size[2] / 2 == pytest.approx(nodes[0, 2] - a["rod"]["radius_m"])


def test_heldout_placement_rejects_training_specs_and_fixtures_outside_the_table():
    scene = json.loads((ROOT / "configs/dh116_tasks/flexible_pinch_end_supported_seed_v1.json").read_bytes())["config"]
    opts = dict(catalog=CATALOG, stations=dict(left=35, right=3), placement=(0., 0., 0.))
    train = next(row["id"] for row in CATALOG["variants"] if row["split"] == "train")
    test = next(row["id"] for row in CATALOG["variants"] if row["split"] == "test")
    with pytest.raises(ValueError, match="held-out"):
        prepare_holdout_variant(scene, TABLE, BASELINE, variant_id=train, **opts)
    with pytest.raises(ValueError, match="table"):
        prepare_holdout_variant(scene, TABLE, BASELINE, variant_id=test, **dict(opts, placement=(.8, 0., 0.)))
    with pytest.raises(ValueError, match="finite"):
        prepare_holdout_variant(scene, TABLE, BASELINE, variant_id=test,
                               **dict(opts, placement=(0., np.nan, 0.)))


def test_holdout_matrix_requires_unique_shared_initial_placements():
    from scripts.prepare_dh116_wire_holdout_resets import prepare_matrix

    scene = json.loads((ROOT / "configs/dh116_tasks/flexible_power_supported_contact_v1.json").read_bytes())
    placements = [(0., 0., 0.), (.001, -.002, .02)]
    matrix = prepare_matrix(scene, TABLE, BASELINE, catalog=CATALOG, stations=dict(left=30, right=10),
                            placements=placements)
    assert len(matrix) == 8 * len(placements)
    assert len({item["physical_initial_condition_sha256"] for item in matrix}) == len(matrix)
    assert all(not item["native_reset_verified"] for item in matrix)
    for variant in {item["variant_id"] for item in matrix}:
        assert [tuple(item["placement"]) for item in matrix if item["variant_id"] == variant] == placements
    with pytest.raises(ValueError, match="duplicate physical"):
        prepare_matrix(scene, TABLE, BASELINE, catalog=CATALOG, stations=dict(left=30, right=10),
                       placements=[placements[0], (-0., -0., -0.)])


def test_clearance_keeps_rejected_trial_in_cohort():
    from scripts.prepare_dh116_wire_holdout_resets import prepare_matrix

    scene = json.loads((ROOT / "configs/dh116_tasks/flexible_power_supported_contact_v1.json").read_bytes())
    vertex_offsets = np.array([[0,0,0], [.002,0,0], [0,.002,0], [0,0,.002]])
    hulls = {("left", "finger14_link", True): vertex_offsets + [0.43, 0., .365]}
    matrix = prepare_matrix(scene, TABLE, BASELINE, catalog=CATALOG, stations=dict(left=30, right=10),
                            placements=[(0., 0., 0.)], hulls=hulls)
    assert len(matrix) == 8
    assert all(not item["cpu_clearance"]["passed"] for item in matrix)
    assert all(item["cpu_clearance"]["failures"][0]["link"] == "finger14_link" for item in matrix)
    assert all(not item["native_reset_verified"] for item in matrix)
