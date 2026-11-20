import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts import screen_dh116_pinch_wrist_relief as relief


def test_world_rotation_reduces_tangential_motion_about_the_index_witness():
    thumb = np.array([.016, 0., 0.])
    index = np.zeros(3)
    measured_tangent = np.array([.2, .1, -10.])
    axis = relief.relief_axis(thumb, index, measured_tangent)
    dex = Rotation.from_rotvec(axis * relief.ANGLE_RAD)
    np.testing.assert_allclose(dex.apply(index) + index, index)
    assert (dex.apply(thumb) - thumb) @ measured_tangent < 0.
    with pytest.raises(ValueError, match="witness"):
        relief.relief_axis(index, index, measured_tangent)


def test_correction_changes_only_left_wrist_in_last_two_windows():
    before = np.zeros((9, 24))
    before[:, [1, 2, 7, 8]] = .5
    dx, de = [.00008, 0., 0.], [0., 0., .001]
    after = relief.compile_schedule(before, dx, de)
    expected = before.copy()
    expected[-2:, 12] = .24
    expected[-2:, 20] = .06
    np.testing.assert_allclose(after, expected)
    with pytest.raises(ValueError, match="rate"):
        relief.compile_schedule(before, [.001, 0., 0.], de)


def test_world_correction_about_off_origin_pivot_preserves_joint_orientation_convention():
    pivot = np.array([.3, -.2, .5])
    xyz = np.array([.4, -.25, .5])
    anchor = Rotation.from_euler("z", .2)
    old_angles = np.array([.01, -.2, .03])
    axis = np.array([0., 0., 1.])
    dx, de = relief.target_difference(xyz, old_angles, anchor, pivot, axis)
    rotation = Rotation.from_rotvec(axis * relief.ANGLE_RAD)
    np.testing.assert_allclose(xyz + dx, pivot + rotation.apply(xyz - pivot), atol=1e-12)
    np.testing.assert_allclose((anchor * Rotation.from_euler("XYZ", old_angles + de)).as_matrix(),
                               (rotation * anchor * Rotation.from_euler("XYZ", old_angles)).as_matrix(),
                               atol=1e-12)
