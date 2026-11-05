"""Depth optical-Z samples expressed in current left and right wrist link frames.

All positions and depth values use metres. Camera coordinates use the optical
convention x right, y down, z forward. T_world_camera maps those coordinates
into the world frame. Hand poses are [left, right], with xyz + wxyz quaternions,
and refer to the actual palm/body link origin used by rod_points_in_hands.
"""

import numpy as np


def transform_world_points_to_hands(points_world, hand_poses):
    """Transform [P,3] world points to [2,P,3] using this frame's wrist poses."""
    points = np.asarray(points_world, dtype=np.float64)
    poses = np.asarray(hand_poses, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('points_world must be finite [P,3] positions in metres')
    if poses.shape != (2, 7) or not np.isfinite(poses).all():
        raise ValueError('hand_poses must be finite [2,7] xyz+wxyz link poses')
    quaternions = poses[:, 3:].copy()
    scale = np.abs(quaternions).max(axis=1, keepdims=True)
    if np.any(scale == 0.):
        raise ValueError('hand pose quaternions must be nonzero')
    quaternions /= scale
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = quaternions.T
    rotations = np.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), axis=1).reshape(2, 3, 3)
    # Row vectors multiply R_world_hand to apply its inverse rotation.
    offsets = points[None] - poses[:, None, :3]
    return offsets @ rotations


def depth_to_hand_points(depth, K, T_world_camera, hand_poses, *, object_mask=None,
                         min_depth_m=.01, max_depth_m=10.):
    """Backproject retained pixels and transform them to both current hand frames.

    depth[H,W] is optical-axis Z in metres, rather than range along a ray.
    K[3,3] is a calibrated pinhole intrinsic matrix, including optional skew.
    object_mask[H,W], when supplied, must be boolean and selects visible pixels.

    Returns pixels_uv[P,2] in row-major pixel order, points_camera[P,3],
    points_world[P,3], points_in_hands[2,P,3], and valid_mask[H,W].
    An all-invalid frame produces empty point arrays and a false valid_mask.
    """
    image = np.asarray(depth, dtype=np.float64)
    intrinsic = np.asarray(K, dtype=np.float64)
    camera = np.asarray(T_world_camera, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError('depth must be an [H,W] optical-Z image in metres')
    if (not np.isfinite(min_depth_m) or not np.isfinite(max_depth_m)
            or not 0. <= min_depth_m < max_depth_m):
        raise ValueError('depth limits must satisfy 0 <= min_depth_m < max_depth_m')
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0. or intrinsic[1, 1] <= 0.
            or not np.allclose(intrinsic[2], [0., 0., 1.], atol=1e-12, rtol=0.)
            or not np.isclose(intrinsic[1, 0], 0., atol=1e-12, rtol=0.)):
        raise ValueError('K must be a finite pinhole matrix with positive focal lengths')
    if (camera.shape != (4, 4) or not np.isfinite(camera).all()
            or not np.allclose(camera[3], [0., 0., 0., 1.], atol=1e-12, rtol=0.)):
        raise ValueError('T_world_camera must be a finite rigid [4,4] transform')
    rotation = camera[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0.)):
        raise ValueError('T_world_camera must contain a proper orthonormal rotation')

    valid = np.isfinite(image) & (image > 0.) & (image >= min_depth_m) & (image <= max_depth_m)
    if object_mask is not None:
        mask = np.asarray(object_mask)
        if mask.shape != image.shape or mask.dtype != np.bool_:
            raise ValueError('object_mask must be boolean with the same [H,W] shape as depth')
        valid &= mask
    rows, columns = np.nonzero(valid)
    pixels = np.column_stack((columns, rows))
    homogeneous_pixels = np.column_stack((pixels, np.ones(len(pixels))))
    rays = homogeneous_pixels @ np.linalg.inv(intrinsic).T
    points_camera = rays * image[valid, None]
    points_world = points_camera @ rotation.T + camera[:3, 3]
    points_in_hands = transform_world_points_to_hands(points_world, hand_poses)
    return dict(
        pixels_uv=pixels, points_camera=points_camera, points_world=points_world,
        points_in_hands=points_in_hands, valid_mask=valid,
    )
