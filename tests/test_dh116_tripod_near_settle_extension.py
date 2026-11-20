import copy

import numpy as np
import pytest

from scripts import run_dh116_tripod_near_corridor as near
from scripts import run_dh116_tripod_near_settle_extension as extension


def test_retime_only_repeats_unchanged_right_wrist_hold():
    n = near.staged
    plan = {"actions": np.ones((62, 24)).tolist(), "physics_steps": 496,
            "policy_steps": 62, "native_ready": False}
    plan["actions"][n.APPROACH_START - 1] = [0.] * 24
    for key in ("intrinsic_knots_rad", "world_offset_knots_m"):
        plan[key] = np.arange(63 * 3).reshape(63, 3).tolist()
        plan[key][n.APPROACH_START - 1] = plan[key][n.APPROACH_START].copy()
    original = copy.deepcopy(plan)
    updated = extension.extend_reference(plan)
    assert plan == original
    assert updated["physics_steps"] == 520 and updated["policy_steps"] == 65
    actions = updated["actions"]
    cut = n.APPROACH_START
    np.testing.assert_array_equal(actions[cut:cut + 3], np.zeros((3, 24)))
    assert actions[:cut] + actions[cut + 3:] == original["actions"]
    for key in ("intrinsic_knots_rad", "world_offset_knots_m"):
        assert updated[key][:cut + 1] + updated[key][cut + 4:] == original[key]
        assert updated[key][cut + 1:cut + 4] == [original[key][cut]] * 3


def test_settle_extension_refuses_an_active_predecessor():
    n = near.staged
    plan = {"actions": np.ones((62, 24)).tolist(), "physics_steps": 496,
            "policy_steps": 62, "native_ready": False}
    for key in ("intrinsic_knots_rad", "world_offset_knots_m"):
        plan[key] = np.zeros((63, 3)).tolist()
    with pytest.raises(ValueError, match="quiet predecessor"):
        extension.extend_reference(plan)


def test_new_gate_uses_new_eight_same_instance_rows():
    n = near.staged
    rows = [{"sdk_step": step, "target": [0.,0.,0.], "q": [0.,0.,0.], "qd": [0.,0.,0.]}
            for step in range(347, 355)]
    assert n.wrist_gate(rows, [0,0,0], expected_sdk=extension.GATE_SDK)["passed"]
    rows[-1]["sdk_step"] = 330
    assert not n.wrist_gate(rows, [0,0,0], expected_sdk=extension.GATE_SDK)["passed"]
