"""Check the pose path against the original wrist integration contract."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scripts import certify_dh116_tripod_opposition as route
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import compose_rotation
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT


def test_pivot_chords_use_world_translation_and_seed_intrinsic_angles():
    seed = Rotation.from_euler("XYZ", [.2, -.7, .9])
    start_angles = np.array([.03, -.06, .02])
    target = (seed * Rotation.from_euler("XYZ", start_angles)).as_quat()[[3, 0, 1, 2]]
    measured = (Rotation.from_rotvec([.01, .005, -.004]) *
                Rotation.from_quat(target[[1, 2, 3, 0]])).as_quat()
    start = np.r_[[.45, .11, .36], measured[[3, 0, 1, 2]]]
    station = np.array([.40, .20, .36])
    rotation = np.array([.08, 0., -.08])

    world, angles, commands = route.pivot_commands(start, target, start_angles, station, rotation, windows=100)

    np.testing.assert_array_equal(angles[0], start_angles)
    np.testing.assert_array_equal(world[0], start[:3])
    np.testing.assert_allclose(world[-1], station + Rotation.from_rotvec(rotation).apply(start[:3] - station))
    increments = np.diff(world, axis=0)
    np.testing.assert_allclose(commands[:, 12:15], increments * 60 / ACTION_CONTRACT["translation_rate_m_s"])
    np.testing.assert_allclose(commands[:, 18:21], np.diff(angles, axis=0) * 60 / ACTION_CONTRACT["rotation_rate_rad_s"])
    assert not commands[:, np.r_[0:12, 15:18, 21:24]].any()
    assert np.linalg.norm(commands[:, 12:15], axis=1).max() <= 1.
    assert np.linalg.norm(commands[:, 18:21], axis=1).max() <= 1.

    nominal = route.nominal_orientation(target, start_angles, start[3:], (angles[37] + angles[38]) / 2)
    expected_angle = torch.tensor((angles[37] + angles[38]) / 2, dtype=torch.float64)
    controller = compose_rotation(torch.tensor(seed.as_quat()[[3, 0, 1, 2]]), expected_angle)
    residual = Rotation.from_quat(target[[1, 2, 3, 0]]).inv() * Rotation.from_quat(measured)
    np.testing.assert_allclose(nominal.as_matrix(),
        (Rotation.from_quat(controller.numpy()[[1, 2, 3, 0]]) * residual).as_matrix(), atol=1e-12)
