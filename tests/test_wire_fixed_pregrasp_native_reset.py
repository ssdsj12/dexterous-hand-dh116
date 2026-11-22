"""Test the opt-in fixed-wrist native Reset preparation before Isaac startup."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state
from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import catalog_wire_specs, prepare_variant
from go2_pvcnn.control.dh116.tabletop_wire_holdout_resets import prepare_fixed_pregrasp_variant
from scripts import verify_dh116_wire_variant_isaac as verifier

ROOT = Path(__file__).resolve().parents[1]
TABLE = dict(center_m=[.6, 0., .25], size_m=[.8, 1.2, .1])
CATALOG = json.loads((ROOT / "configs/dh116_training/supported_wire_generalization_v2.json").read_bytes())
BASELINE = json.loads((ROOT / verifier.BASE_ROD).read_bytes())["rod"]
TRAIN_CASE = catalog_wire_specs(CATALOG, BASELINE)["train"][0]


def packet(scope):
    scene = json.loads((ROOT / "configs/dh116_tasks/flexible_power_supported_contact_v1.json").read_bytes())
    scene["scope"] = scope
    seed, _, _, _ = prepare_initial_state(scene, TABLE, expected_scope=scope)
    return dict(version="dh116-wire-common-pregrasp-v1", status="cpu_reset_candidate", scope=scope,
                config=scene, reset_seed_sha256=seed["sha256"], native_reset_verified=False,
                approach_verified=False, grasp_qualified=False,
                training_spec_sha256={TRAIN_CASE["id"]: TRAIN_CASE["spec_sha256"]})


def options():
    return dict(radius_m=TRAIN_CASE["rod"]["radius_m"], length_m=TRAIN_CASE["length_m"],
                stations=dict(left=35, right=3), variant_id=TRAIN_CASE["id"])


@pytest.mark.parametrize("grasp", ["pinch", "tripod", "power"])
def test_fixed_native_variant_never_retargets_the_published_wrist_pose(grasp):
    selected = packet(grasp + "_end_supported_contact_diagnostic")
    seed, q, nodes, supports, variant = prepare_fixed_pregrasp_variant(selected, TABLE, BASELINE, **options())
    assert seed["sha256"] == selected["reset_seed_sha256"]
    assert variant["grasp"] == grasp and variant["spec_sha256"] == TRAIN_CASE["spec_sha256"]
    assert variant["scene"]["hands"] == selected["config"]["hands"]
    for i, side in enumerate(("left", "right")):
        np.testing.assert_array_equal(seed["reset_root_pose_w"][i], selected["config"]["hands"][side]["pose_wxyz"])
        np.testing.assert_array_equal(q[i], selected["config"]["hands"][side]["q_rad"])
    assert np.linalg.norm(nodes[-1] - nodes[0]) == pytest.approx(TRAIN_CASE["length_m"])
    assert set(supports) == {"support_0", "support_1"}
    assert not variant["native_reset_verified"]
    if grasp == "tripod":
        legacy = options()
        legacy.pop("variant_id")
        retargeted, _, _, _, _ = prepare_variant(selected["config"], TABLE, BASELINE, **legacy)
        assert retargeted["reset_root_pose_w"] != seed["reset_root_pose_w"]


@pytest.mark.parametrize("mutated", ["scope", "seed", "train"])
def test_fixed_pregrasp_rejects_invalid_scope_seed_or_training_spec(mutated):
    selected = deepcopy(packet("power_end_supported_contact_diagnostic"))
    if mutated == "scope":
        selected["scope"] = "pinch_end_supported_contact_diagnostic"
    if mutated == "seed":
        selected["config"]["hands"]["left"]["pose_wxyz"][0] += .001
    if mutated == "train":
        selected["training_spec_sha256"] = {}
    with pytest.raises(ValueError):
        prepare_fixed_pregrasp_variant(selected, TABLE, BASELINE, **options())


def test_fixed_pregrasp_requires_a_quiet_settlement_budget_before_sources_are_opened():
    with pytest.raises(ValueError, match="quiet settlement"):
        verifier.freeze(SimpleNamespace(fixed_pregrasp=True, quiet_settlement_sdk=None))
