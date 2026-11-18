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


def test_integrated_contact_optimizer_uses_measured_hand_geometry():
    from go2_pvcnn.control.dh116.contact_mpc import ParallelContactMPC
    from test_dh116_contact_mpc import observation

    obs = observation()
    controller = IntegratedHandMPC(ParallelContactMPC())
    action = controller.plan(np.zeros((2, 7)), obs["active_joints"].numpy(), .1,
        obs["previous_action"].numpy(), pad_positions=obs["pad_positions"],
        pad_jacobians=obs["pad_jacobians"], target_points=obs["target_points"])
    assert action.shape == (2, 6)
    assert action[0, 0] > .2 and action[0, 2] < .2
    assert float(controller.last_diagnostics["optimized_cost"]) < float(
        controller.last_diagnostics["hold_cost"])
    with pytest.raises(ValueError, match="measured"):
        controller.plan(np.zeros((2, 7)), obs["active_joints"].numpy(), .1,
                        obs["previous_action"].numpy())


def test_integrated_preload_hold_preserves_only_requested_hand_coordinates():
    from go2_pvcnn.control.dh116.contact_mpc import ParallelContactMPC
    from test_dh116_contact_mpc import observation
    obs=observation();held=np.array([True]*6+[False]*6)
    controller=IntegratedHandMPC(ParallelContactMPC())
    action=controller.plan(np.zeros((2,7)),obs['active_joints'].numpy(),.1,
        obs['previous_action'].numpy(),pad_positions=obs['pad_positions'],
        pad_jacobians=obs['pad_jacobians'],target_points=obs['target_points'],held_coordinates=held)
    np.testing.assert_array_equal(action[0],obs['previous_action'].numpy()[:6])
    assert action[1,0]>.2


def test_learned_complete_policy_uses_the_same_preload_handoff():
    from go2_pvcnn.control.dh116.complete_contact_policy import CompleteContactPolicy
    from test_dh116_contact_collection import observation
    obs=observation();obs['pad_jacobians'][:]=.01
    controller=IntegratedHandMPC(CompleteContactPolicy(role='student',control_dt=.01,rod_nodes=5))
    previous=np.full(12,.2,dtype=np.float32)
    action=controller.plan(obs['palm_poses_w'],obs['hand_q'],.1,previous,
        pad_positions=obs['pad_positions_w'],pad_jacobians=obs['pad_jacobians'],
        target_points=obs['target_points_w'],complete_observation=obs,
        held_coordinates=np.array([True]*6+[False]*6))
    np.testing.assert_array_equal(action[0],previous[:6])
    assert controller.last_diagnostics['held_coordinates'][:6].all()
