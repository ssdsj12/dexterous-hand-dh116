import numpy as np
import pytest

from go2_pvcnn.control.dh116.panda_lift import panda_only_transport_increment


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
