import numpy as np
import pytest

from go2_pvcnn.control.dh116.panda_lift import (
    panda_feedback_increment, panda_only_transport_increment,
)


def full_rank_fixture():
    point_jacobians = np.zeros((2, 3, 3, 7), dtype=float)
    for side in range(2):
        point_jacobians[side, :, :, :3] = 0.02 * np.eye(3)
    palm_angular_jacobians = np.zeros((2, 3, 7), dtype=float)
    velocity = np.tile([0.0, 0.0, 0.001], (2, 1))
    return point_jacobians, palm_angular_jacobians, velocity


def test_panda_lift_uses_only_seven_arm_columns_and_tracks_transport():
    J, W, velocity = full_rank_fixture()
    dt = 1e-4
    increments = panda_only_transport_increment(J, W, velocity, dt)
    assert increments.shape == (2, 7)
    for side in range(2):
        actual_velocity = np.einsum("hij,j->hi", J[side], increments[side] / dt)
        np.testing.assert_allclose(
            actual_velocity, np.tile(velocity[side], (3, 1)), atol=1e-8
        )
    assert np.max(np.abs(increments / dt)) <= 0.05 + 1e-12


@pytest.mark.parametrize("change", ["singular", "nonfinite", "bad_dt"])
def test_panda_lift_rejects_invalid_jacobians_and_timing(change):
    J, W, velocity = full_rank_fixture()
    dt = 1e-4
    if change == "singular":
        J[1] = 0.0
    elif change == "nonfinite":
        velocity[0, 0] = np.nan
    else:
        dt = 0.0
    with pytest.raises(ValueError):
        panda_only_transport_increment(J, W, velocity, dt)


def test_panda_lift_rejects_wrong_shapes_and_caps_velocity():
    J, W, velocity = full_rank_fixture()
    with pytest.raises(ValueError):
        panda_only_transport_increment(J[..., :6], W[..., :6], velocity, 1e-4)
    increments = panda_only_transport_increment(
        J, W, np.tile([0.0, 0.0, 0.01], (2, 1)), 1e-4,
        speed_limit_rad_s=0.025,
    )
    assert np.max(np.abs(increments / 1e-4)) <= 0.025 + 1e-12


def test_panda_feedback_lift_closes_measured_palm_error_and_reports_residual():
    J, W, velocity = full_rank_fixture()
    target=np.tile([.0,.0,.30], (2,1))
    actual=np.tile([.0,.0,.2999], (2,1))
    increments, reports=panda_feedback_increment(
        J,W,velocity,1e-3,palm_target_positions_w=target,
        palm_actual_positions_w=actual,position_gain=2.,speed_limit_rad_s=.1)
    assert increments.shape==(2,7)
    assert len(reports)==2
    for report in reports:
        assert report['position_error_m']==pytest.approx(.0001)
        assert report['commanded_velocity_w'][2] > .001
        assert report['residual_norm'] < 1e-8


def test_panda_feedback_lift_respects_joint_limits_and_rejects_unreachable_velocity():
    J, W, velocity = full_rank_fixture()
    target=np.zeros((2,3));actual=np.zeros((2,3))
    q=np.zeros((2,7));lower=np.full((2,7),-.001);upper=np.full((2,7),.001)
    increments,reports=panda_feedback_increment(
        J,W,np.full((2,3),.01),1e-3,palm_target_positions_w=target,
        palm_actual_positions_w=actual,joint_positions=q,joint_lower=lower,
        joint_upper=upper,speed_limit_rad_s=.05)
    assert np.all(q+increments<=upper+1e-12)
    assert np.all(q+increments>=lower-1e-12)
    bad=np.zeros((2,3,3,7));bad[:,:,0,:]=J[:,:,0,:]
    with pytest.raises(ValueError,match='singular|residual'):
        panda_feedback_increment(bad,W,velocity,1e-3,
            palm_target_positions_w=target,palm_actual_positions_w=actual)
