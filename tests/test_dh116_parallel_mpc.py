import numpy as np
import pytest

from go2_pvcnn.control.dh116.hand_only_env import HandOnlyState
from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.parallel_mpc import ParallelJointMPC


def batch_state(batch=4):
    wire = np.array([[0.0, 0.0], [0.8, 0.0], [0.8, 0.5], [1.0, 1.0]], dtype=np.float32)[:batch]
    return HandOnlyState(
        np.zeros((batch, 2, 7), dtype=np.float32),
        np.zeros((batch, 2, 6), dtype=np.float32),
        wire,
        np.zeros((batch, 12), dtype=np.float32),
    )


def test_plan_is_batched_bounded_and_deterministic():
    planner = ParallelJointMPC()
    s = batch_state()
    action, diagnostics = planner.plan(s, s.previous_action)
    action2, diagnostics2 = planner.plan(s, s.previous_action)
    assert action.shape == (4, 12)
    np.testing.assert_array_equal(action, action2)
    np.testing.assert_array_equal(diagnostics["phase"], diagnostics2["phase"])
    lower, upper = active_limits()
    lower12 = np.tile(lower, 2)
    upper12 = np.tile(upper, 2)
    assert np.all(action > lower12[None, :] - 1e-7)
    assert np.all(action < upper12[None, :] + 1e-7)
    assert diagnostics["phase"].tolist() == ["grasp", "grasp", "lift", "lift"]


def test_plan_rejects_wrong_previous_action_shape_and_nan():
    planner = ParallelJointMPC()
    s = batch_state(2)
    with pytest.raises(ValueError):
        planner.plan(s, np.zeros((2, 11)))
    with pytest.raises(ValueError, match="NaN"):
        planner.plan(s, np.full((2, 12), np.nan))


def test_plan_trajectory_generates_complete_batched_grasp_then_lift_from_reference():
    planner = ParallelJointMPC(smoothing=0.25)
    state = batch_state(2)
    initial = np.full((2, 12), 0.1, dtype=np.float32)

    trajectory, diagnostics = planner.plan_trajectory(
        state, grasp_steps=3, lift_steps=2, previous_action=initial,
    )

    assert trajectory.shape == (2, 5, 12)
    assert diagnostics["phase"].shape == (2, 5)
    assert diagnostics["phase"][0].tolist() == [0] * 3 + [1] * 2
    assert diagnostics["grasp_progress"][0].tolist() == pytest.approx([1 / 3, 2 / 3, 1, 1, 1])
    assert diagnostics["lift_progress"][0].tolist() == pytest.approx([0, 0, 0, 0.5, 1])
    np.testing.assert_allclose(trajectory[:, 0], planner.plan(
        HandOnlyState(state.hand_poses[:2], state.active_joints[:2],
                      np.tile([[1 / 3, 0]], (2, 1)), initial), initial)[0])
    assert np.isfinite(trajectory).all()
    lower, upper = active_limits()
    assert np.all(trajectory >= np.tile(lower, 2)[None, None, :] - 1e-7)
    assert np.all(trajectory <= np.tile(upper, 2)[None, None, :] + 1e-7)


def test_lift_phase_adds_hook_bias_only_to_thumb_and_middle_finger():
    planner = ParallelJointMPC(smoothing=0.0)
    base = batch_state(1)
    state = HandOnlyState(base.hand_poses, base.active_joints,
                          np.asarray([[1.0, 0.0]], dtype=np.float32),
                          np.zeros((1, 12), dtype=np.float32))
    grasp, _ = planner.plan(state)
    lift_state = HandOnlyState(base.hand_poses, base.active_joints,
                               np.asarray([[1.0, 1.0]], dtype=np.float32),
                               np.zeros((1, 12), dtype=np.float32))
    lifted, _ = planner.plan(lift_state)
    delta = lifted - grasp
    active = np.flatnonzero(np.abs(delta[0]) > 1e-7).tolist()
    assert active == [0, 2, 5, 6, 8, 11]
    lower, upper = active_limits()
    expected = 0.10 * np.tile(upper - lower, 2) * np.tile([1, 0, 1, 0, 0, 1], 2)
    np.testing.assert_allclose(delta[0], expected, atol=1e-7)


def test_mpc_grasp_fraction_limits_maximum_closure_from_measured_overforce():
    planner = ParallelJointMPC(smoothing=0.0, grasp_fraction=0.45)
    base = batch_state(1)
    state = HandOnlyState(base.hand_poses, base.active_joints,
                          np.asarray([[1.0, 0.0]], dtype=np.float32),
                          np.zeros((1, 12), dtype=np.float32))
    action, _ = planner.plan(state)
    lower, upper = active_limits()
    expected_hand = lower + (upper - lower) * 0.70
    np.testing.assert_allclose(action[0, :6], expected_hand, atol=1e-7)
    assert np.max(action[0]) < np.max(ParallelJointMPC(smoothing=0).plan(state)[0][0])


def test_cuda_request_fails_clearly_without_cuda():
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None or not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            ParallelJointMPC(device="cuda:0")
