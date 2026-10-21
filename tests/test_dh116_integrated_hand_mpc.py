import numpy as np
import pytest

from go2_pvcnn.control.dh116.integrated_hand_mpc import (
    IntegratedHandMPC,
    route_phase_targets,
)
from go2_pvcnn.control.dh116.parallel_mpc import ParallelJointMPC


def inputs():
    return dict(
        arm_approach=np.ones((2, 7)),
        arm_lift=np.full((2, 7), 2.0),
        arm_frozen=np.full((2, 7), 3.0),
        hand_approach=np.ones((2, 6)),
        hand_mpc=np.full((2, 6), 4.0),
        hand_hold=np.full((2, 6), 5.0),
    )


@pytest.mark.parametrize(
    "phase,arm_value,hand_value,arm_owner,hand_owner",
    [
        ("approach", 1.0, 1.0, "panda", "approach_trajectory"),
        ("close", 3.0, 4.0, "frozen", "hand_mpc"),
        ("hold", 3.0, 5.0, "frozen", "hand_mpc_hold"),
        ("lift", 2.0, 5.0, "panda", "hand_mpc_hold"),
    ],
)
def test_phase_routing_isolates_arm_and_hand_actions(
    phase, arm_value, hand_value, arm_owner, hand_owner
):
    arms, hands, owners = route_phase_targets(phase, **inputs())
    assert arms.shape == (2, 7) and hands.shape == (2, 6)
    np.testing.assert_allclose(arms, arm_value)
    np.testing.assert_allclose(hands, hand_value)
    assert owners == {"arm": arm_owner, "hand": hand_owner}


def test_hand_mpc_returns_only_bounded_12d_active_action():
    controller = IntegratedHandMPC(ParallelJointMPC(smoothing=0.0))
    action = controller.plan(
        np.zeros((2, 7)), np.zeros((2, 6)), 0.5, np.zeros(12)
    )
    assert action.shape == (2, 6)
    assert np.isfinite(action).all()
    lower, upper = controller.planner.lower, controller.planner.upper
    assert np.all(action >= lower[None, :] - 1e-7)
    assert np.all(action <= upper[None, :] + 1e-7)


@pytest.mark.parametrize(
    "hand_poses,active_joints,progress,previous",
    [
        (np.zeros((2, 6)), np.zeros((2, 6)), 0.5, np.zeros(12)),
        (np.zeros((2, 7)), np.zeros((2, 5)), 0.5, np.zeros(12)),
        (np.zeros((2, 7)), np.zeros((2, 6)), np.nan, np.zeros(12)),
        (np.zeros((2, 7)), np.zeros((2, 6)), 0.5, np.zeros(11)),
    ],
)
def test_hand_mpc_rejects_invalid_observation_dimensions_or_values(
    hand_poses, active_joints, progress, previous
):
    controller = IntegratedHandMPC(ParallelJointMPC())
    with pytest.raises(ValueError):
        controller.plan(hand_poses, active_joints, progress, previous)


def test_route_rejects_invalid_phase_and_action_shapes():
    with pytest.raises(ValueError, match="phase"):
        route_phase_targets("approach_and_close", **inputs())
    bad = inputs()
    bad["arm_lift"] = np.zeros((2, 6))
    with pytest.raises(ValueError, match="arm_lift"):
        route_phase_targets("lift", **bad)


def test_phase_router_returns_independent_copies():
    source = inputs()
    arms, hands, _ = route_phase_targets("close", **source)
    arms[0, 0] = -100.0
    hands[0, 0] = -100.0
    assert source["arm_frozen"][0, 0] == 3.0
    assert source["hand_mpc"][0, 0] == 4.0
