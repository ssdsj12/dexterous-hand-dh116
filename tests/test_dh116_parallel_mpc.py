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


def test_cuda_request_fails_clearly_without_cuda():
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None or not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            ParallelJointMPC(device="cuda:0")
