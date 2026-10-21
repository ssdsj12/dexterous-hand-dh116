import numpy as np
import pytest

from go2_pvcnn.control.dh116.hand_only_env import HandOnlyEnv, HandOnlySpec, HandOnlyState


def state(batch=None):
    shape = () if batch is None else (batch,)
    return HandOnlyState(
        np.zeros(shape + (2, 7), dtype=np.float32),
        np.zeros(shape + (2, 6), dtype=np.float32),
        np.zeros(shape + (2,), dtype=np.float32),
        np.zeros(shape + (12,), dtype=np.float32),
    )


def test_contract_is_two_hands_and_twelve_actions():
    spec = HandOnlySpec()
    assert spec.action_dim == spec.previous_action_dim == 12
    assert state(3).batch_size == 3
    assert state().batched().hand_poses.shape == (1, 2, 7)


def test_aliases_and_previous_action_step():
    s = HandOnlyState(
        left_pose=np.zeros(7),
        right_pose=np.zeros(7),
        active_joints=np.zeros((2, 6)),
        wire_grasp=0.2,
        wire_lift=0.0,
        previous_action=np.zeros(12),
    )
    env = HandOnlyEnv()
    env.reset(s)
    nxt = env.step(np.ones(12, dtype=np.float32))
    np.testing.assert_allclose(nxt.previous_action, 1.0)
    assert nxt.left_pose.shape == (7,)


@pytest.mark.parametrize("field,value", [("active_joints", np.zeros((2, 5))), ("previous_action", np.zeros(11))])
def test_rejects_wrong_dimensions(field, value):
    kwargs = dict(
        hand_poses=np.zeros((2, 7)),
        active_joints=np.zeros((2, 6)),
        wire_features=np.zeros(2),
        previous_action=np.zeros(12),
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        HandOnlyState(**kwargs)


def test_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        HandOnlyState(np.zeros((2, 7)), np.zeros((2, 6)), np.array([np.nan, 0.0]), np.zeros(12))
