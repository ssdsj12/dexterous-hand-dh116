import numpy as np
import pytest


def test_power_pose_follows_the_same_material_station_in_world_space():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_power_distance import power_pose_error

    initial = np.column_stack((np.zeros(41), np.arange(41) * .0125, np.ones(41)))
    targets = np.array([[.01, .1, .99], [.02, .3875, 1.02]])
    positions = targets.copy()
    positions[0, 2] += .003
    shift = np.array([.2, -.3, .1])
    distances, stations = power_pose_error(initial, targets, initial, positions)
    translated, shifted_stations = power_pose_error(initial, targets, initial + shift, positions + shift)
    assert stations == shifted_stations == [8, 31]
    assert distances == pytest.approx([.003, 0.])
    assert translated == pytest.approx(distances)


def test_power_pose_error_rejects_incomplete_measured_rod():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_power_distance import power_pose_error

    initial = np.zeros((41, 3))
    wrist_targets = np.zeros((2, 3))
    with pytest.raises(ValueError, match="41"):
        power_pose_error(initial, wrist_targets, initial[:40], wrist_targets)
