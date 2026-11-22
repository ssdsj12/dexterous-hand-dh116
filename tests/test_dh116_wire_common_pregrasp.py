"""Regression coverage for training-only common wire approach starts."""
from pathlib import Path
import json

import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from scripts import select_dh116_wire_common_pregrasp as selector
from test_supported_wire_holdout_resets import BASELINE, CATALOG, TABLE

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "configs/dh116_tasks/flexible_power_supported_contact_v1.json"


def test_training_clearance_selects_independent_smallest_wrist_offsets(monkeypatch):
    scene = json.loads(SCENE.read_bytes())
    original = json.loads(json.dumps(scene))
    # Both original synthetic pads enter the wire near its ends. The shortest
    # axis exit is opposite on the two sides.
    points = np.array([[0.,0.,0.], [.001,0.,0.], [0.,.001,0.], [0.,0.,.001]])
    positions = dict(left=[.433, .12, .365], right=[.426, -.12, .365])

    def hulls(side, hand, geometry):
        return {(side, "finger14_link", True): points + positions[side]}

    monkeypatch.setattr(selector, "authored_side_hulls", hulls)
    monkeypatch.setattr(selector, "thumb_clearance", lambda mapping: .01)
    result = selector.select_common_pregrasp(
        scene, TABLE, BASELINE, catalog=CATALOG, stations=dict(left=30, right=10),
        placements=[(0., 0., 0.)], wrist_candidates_m=[(0., 0., 0.), (.004, 0., 0.), (-.004, 0., 0.)],
        opening_fractions=[0.], return_reserve_m=.004, geometries=dict(left=None, right=None))
    assert scene == original
    assert len(result["training_spec_sha256"]) == 11
    assert {row["id"] for row in CATALOG["variants"] if row["split"] == "train"} == set(result["training_spec_sha256"])
    assert not result["native_reset_verified"]
    chosen = result["config"]
    assert chosen["rod_nodes_w_m"] == scene["rod_nodes_w_m"]
    assert chosen["support_scene"] == scene["support_scene"]
    for side, shift in (("left", .004), ("right", -.004)):
        assert chosen["hands"][side]["q_rad"] == scene["hands"][side]["q_rad"]
        assert chosen["hands"][side]["pose_wxyz"][1:] == scene["hands"][side]["pose_wxyz"][1:]
        assert chosen["hands"][side]["pose_wxyz"][0] - scene["hands"][side]["pose_wxyz"][0] == pytest.approx(shift)


def test_return_reserve_keeps_both_signs_inside_the_original_wrist_bounds():
    scene = json.loads(SCENE.read_bytes())
    with pytest.raises(ValueError, match="return travel"):
        selector.select_common_pregrasp(
            scene, TABLE, BASELINE, catalog=CATALOG, stations=dict(left=30, right=10),
            placements=[(0., 0., 0.)], wrist_candidates_m=[(0.,0.,0.), (-.029,0.,0.)],
            opening_fractions=[0.], return_reserve_m=.004, geometries=dict(left=None, right=None))


def test_full_open_keeps_thumb_opposition_and_materializes_exact_joint_lower_bounds():
    scene = json.loads(SCENE.read_bytes())
    seed = np.asarray(scene["hands"]["left"]["q_rad"])
    q = selector.open_flexion(seed, 1.)
    limits, _ = active_limits()
    assert q[0] == seed[0]
    np.testing.assert_array_equal(q[1:], limits[1:])
    assert q[1] > 0.
    expand(q)


def test_settled_shape_from_outside_training_is_rejected():
    from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import catalog_wire_specs

    scene = json.loads(SCENE.read_bytes())
    holdout = catalog_wire_specs(CATALOG, BASELINE)["test"][0]
    with pytest.raises(ValueError, match="settled layout is not a training specimen"):
        selector.select_common_pregrasp(
            scene, TABLE, BASELINE, catalog=CATALOG, stations=dict(left=30, right=10),
            placements=[(0.,0.,0.)], wrist_candidates_m=[(0.,0.,0.)], opening_fractions=[0.],
            return_reserve_m=.004, geometries=dict(left=None, right=None), settled_layout=holdout)


def test_opening_can_adjust_power_thumb_opposition_without_changing_its_flexion_limit():
    scene = json.loads(SCENE.read_bytes())
    seed = np.asarray(scene["hands"]["left"]["q_rad"])
    opened = selector.open_flexion(seed, 1., thumb_scale=.75)
    lower, _ = active_limits()
    assert opened[0] == pytest.approx(.75 * seed[0])
    np.testing.assert_array_equal(opened[1:], lower[1:])
    assert seed[0] == scene["hands"]["left"]["q_rad"][0]
    expand(opened)


def test_power_thumb_variation_is_not_enabled_for_pinch():
    scene = json.loads((ROOT / "configs/dh116_tasks/flexible_pinch_end_supported_seed_v1.json").read_bytes())["config"]
    with pytest.raises(ValueError, match="Power scene"):
        selector.select_common_pregrasp(
            scene, TABLE, BASELINE, catalog=CATALOG, stations=dict(left=35, right=3),
            placements=[(0.,0.,0.)], wrist_candidates_m=[(0.,0.,0.)], opening_fractions=[0.],
            return_reserve_m=.004, geometries=dict(left=None, right=None), power_left_thumb_scale=.75)
