import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_front_depth_camera import (
    camera_matrices, project_points, reconstruct_points, points_in_hand,
)

CONFIG = dict(position_world_m=[.43,-.60,1.10],target_world_m=[.48,0.,.43],
    world_up=[0.,0.,1.],resolution_wh=[640,480],horizontal_fov_degrees=75.,
    clipping_range_m=[.1,2.])


def test_optical_axes_form_a_right_handed_frame_and_aim_at_target():
    k, transform = camera_matrices(CONFIG)
    rotation = transform[:3,:3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    assert rotation[2,1] < 0.
    pixel, depth = project_points(np.array([CONFIG['target_world_m']]), k, transform)
    np.testing.assert_allclose(pixel, [[319.5,239.5]], atol=1e-10)
    assert depth[0] > 0.


def test_depth_reprojection_and_moving_hand_coordinates():
    k, camera = camera_matrices(CONFIG)
    points = np.array([[.48,-.2,.40],[.50,.2,.48]])
    pixel, depth = project_points(points,k,camera)
    reconstructed = reconstruct_points(pixel,depth,k,camera)
    np.testing.assert_allclose(reconstructed,points,atol=1e-12)
    hand = np.eye(4)
    hand[:3,3] = [.5,-.2,.45]
    np.testing.assert_allclose(points_in_hand(points,hand),points-hand[:3,3])
    hand[:3,3] += [.01,.02,0.]
    np.testing.assert_allclose(points_in_hand(points,hand),points-hand[:3,3])


def test_same_eye_and_target_is_rejected():
    bad = dict(CONFIG,target_world_m=CONFIG['position_world_m'])
    with pytest.raises(ValueError):
        camera_matrices(bad)


def test_nonpositive_depth_is_rejected():
    k, camera = camera_matrices(CONFIG)
    with pytest.raises(ValueError):
        reconstruct_points(np.array([[300.,200.]]),np.array([0.]),k,camera)
