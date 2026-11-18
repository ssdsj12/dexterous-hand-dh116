"""Depth optical-Z projection and measured wrist-frame CPU regressions."""
import importlib
import importlib.util

import numpy as np
import pytest

MODULE = 'go2_pvcnn.control.dh116.tabletop_depth_hand_points'


def api():
    assert importlib.util.find_spec(MODULE), 'depth-to-hand coordinate interface is missing'
    return importlib.import_module(MODULE)


def hand_poses():
    return np.array([[0., 0., 0., 1., 0., 0., 0.],
                     [1., 0., 0., 1., 0., 0., 0.]])


def test_depth_projection_uses_optical_z_filters_invalid_and_applies_object_mask():
    module = api()
    depth = np.array([[np.nan, 0., np.inf, -.5, .05],
                      [.1, .2, .3, 4., .4]])
    mask = np.zeros_like(depth, dtype=bool)
    mask[1, 0:3] = True
    camera = np.eye(4)
    camera[:3, 3] = [0., 0., 1.]
    intrinsic = np.array([[2., 0., 1.], [0., 4., 1.], [0., 0., 1.]])
    result = module.depth_to_hand_points(
        depth, intrinsic, camera, hand_poses(), object_mask=mask,
        min_depth_m=.1, max_depth_m=1.)
    np.testing.assert_array_equal(result['pixels_uv'], [[0, 1], [1, 1], [2, 1]])
    expected = np.array([[-.05, 0., .1], [0., 0., .2], [.15, 0., .3]])
    np.testing.assert_allclose(result['points_camera'], expected)
    np.testing.assert_allclose(result['points_world'], expected + [0., 0., 1.])
    np.testing.assert_allclose(result['points_in_hands'][0], expected + [0., 0., 1.])
    np.testing.assert_allclose(result['points_in_hands'][1], expected + [-1., 0., 1.])
    np.testing.assert_array_equal(result['valid_mask'], mask)
    assert np.isnan(depth[0, 0]) and np.isinf(depth[0, 2])


def test_hand_transform_uses_new_pose_each_frame_and_wxyz_inverse_rotation():
    module = api()
    points = np.array([[1., 2., 3.]])
    poses = hand_poses()
    poses[0] = [1., 0., 0., 2**-.5, 0., 0., 2**-.5]
    first = module.transform_world_points_to_hands(points, poses)
    np.testing.assert_allclose(first[0], [[2., 0., 3.]], atol=1e-14)
    poses[:, :3] += [0., 1., 0.]
    second = module.transform_world_points_to_hands(points, poses)
    np.testing.assert_allclose(second[0], [[1., 0., 3.]], atol=1e-14)
    assert not np.array_equal(first, second)


def test_real_three_keypoint_projection_matches_existing_rod_points_in_hands():
    import torch
    from go2_pvcnn.control.dh116.tabletop_multifinger_monitor import rod_points_in_hands

    module = api()
    camera = np.eye(4)
    camera[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    camera[:3, 3] = [.2, -.1, .4]
    intrinsic = np.array([[1000., 0., 250.], [0., 1000., 250.], [0., 0., 1.]])
    object_pose = np.array([[.2, -.1, 1.4, 2**-.5, 0., 0., 2**-.5]])
    poses = np.array([[.1, .2, .3, 2**-.5, 0., 0., 2**-.5],
                      [-.3, .05, .1, .5, .5, .5, .5]])
    points_camera = np.array([[0., -.2, 1.], [0., .2, 1.], [.008, 0., 1.]])
    projected = points_camera @ intrinsic.T
    pixels = np.rint(projected[:, :2] / projected[:, 2:]).astype(int)
    depth = np.full((501, 501), np.nan)
    depth[pixels[:, 1], pixels[:, 0]] = points_camera[:, 2]
    result = module.depth_to_hand_points(depth, intrinsic, camera, poses)
    expected = rod_points_in_hands(torch.tensor(object_pose), torch.tensor(poses[None])).numpy()[0]
    # Retained pixels are in row-major order: negative-Y, radius, positive-Y.
    np.testing.assert_allclose(result['points_in_hands'], expected[:, [0, 2, 1]],
                               atol=2e-14, rtol=0.)


def test_all_invalid_depth_preserves_empty_cloud_shapes():
    module = api()
    result = module.depth_to_hand_points(
        np.array([[0., np.nan]]), np.eye(3), np.eye(4), hand_poses())
    assert result['pixels_uv'].shape == (0, 2)
    assert result['points_camera'].shape == (0, 3)
    assert result['points_world'].shape == (0, 3)
    assert result['points_in_hands'].shape == (2, 0, 3)
    assert not result['valid_mask'].any()


@pytest.mark.parametrize('invalid', ['intrinsic', 'extrinsic', 'pose', 'mask', 'range'])
def test_malformed_calibration_or_pose_is_rejected(invalid):
    module = api()
    intrinsic, camera, poses = np.eye(3), np.eye(4), hand_poses()
    mask, minimum, maximum = None, .01, 10.
    if invalid == 'intrinsic':
        intrinsic[0, 0] = 0.
    elif invalid == 'extrinsic':
        camera[0, 0] = 2.
    elif invalid == 'pose':
        poses[0, 3:] = 0.
    elif invalid == 'mask':
        mask = np.ones((1, 1), dtype=bool)
    elif invalid == 'range':
        minimum, maximum = 2., 1.
    with pytest.raises(ValueError):
        module.depth_to_hand_points(np.ones((2, 2)), intrinsic, camera, poses,
                                    object_mask=mask, min_depth_m=minimum, max_depth_m=maximum)
