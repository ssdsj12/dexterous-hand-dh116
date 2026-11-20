import numpy as np
import pytest

from scripts.run_dh116_tripod_staged_wrist import HOLD_WINDOWS, ROTATION_WINDOWS, staged_reference, wrist_gate


def test_rotation_and_approach_are_on_opposite_sides_of_measured_hold():
    motion = np.zeros((20, 24))
    motion[:, 6] = -.1
    motion[:, 17] = -.35
    motion[:, 22] = -.16
    old = dict(actions=motion.tolist(), q0_rad=[0.] * 6, q1_rad=[0.] * 6,
               intrinsic_knots_rad=np.linspace([0., 0., 0.], [0., -.025, 0.], 21).tolist(),
               world_offset_knots_m=np.linspace([0., 0., 0.], [0., 0., -.003], 21).tolist())
    plan = staged_reference(old)
    actions = np.asarray(plan["actions"])
    after_hold = 1 + ROTATION_WINDOWS + HOLD_WINDOWS
    np.testing.assert_array_equal(actions[:after_hold, :21], 0)
    np.testing.assert_array_equal(actions[after_hold:, 21:], 0)
    np.testing.assert_allclose(actions.sum(0), motion.sum(0))
    np.testing.assert_array_equal(actions[1 + ROTATION_WINDOWS:after_hold], 0.)


@pytest.mark.parametrize(("error", "speed", "expected"), [(1e-4, 1e-4, True), (3e-4, 1e-4, False), (1e-4, .003, False)])
def test_wrist_gate_requires_both_position_and_velocity(error, speed, expected):
    desired = np.array([.001, -.032, .003])
    rows = [{"sdk_step": s, "target": desired, "q": desired + error, "qd": [speed] * 3} for s in range(30, 38)]
    assert wrist_gate(rows, desired, expected_sdk=37)["passed"] is expected


def test_stale_wrist_readback_refuses_approach():
    desired = np.zeros(3)
    rows = [{"sdk_step": 37, "target": desired, "q": desired, "qd": desired}] * 8
    assert not wrist_gate(rows, desired, expected_sdk=37)["passed"]
