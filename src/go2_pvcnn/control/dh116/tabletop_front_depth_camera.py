"""Explicit frame geometry for the fixed front depth camera.

The optical frame is right/down/forward and depth is optical-axis Z in meters.
This module does not acquire images or supply simulator truth to a policy.
"""
import numpy as np


def camera_matrices(config):
    eye = np.asarray(config['position_world_m'], dtype=float)
    target = np.asarray(config['target_world_m'], dtype=float)
    up = np.asarray(config['world_up'], dtype=float)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (eye,target,up)):
        raise ValueError('finite xyz camera position, target and up required')
    forward = target-eye
    if np.linalg.norm(forward) < 1e-9:
        raise ValueError('camera position and target must differ')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward,up)
    if np.linalg.norm(right) < 1e-9:
        raise ValueError('camera direction and up must not be parallel')
    right /= np.linalg.norm(right)
    down = np.cross(forward,right)
    transform = np.eye(4)
    transform[:3,:3] = np.column_stack((right,down,forward))
    transform[:3,3] = eye
    width,height = config['resolution_wh']
    fov = float(config['horizontal_fov_degrees'])
    if width < 1 or height < 1 or not 0 < fov < 180:
        raise ValueError('positive image dimensions and horizontal FOV in (0,180) required')
    focal = width/(2*np.tan(np.deg2rad(fov)/2))
    intrinsics = np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1.]])
    return intrinsics,transform


def project_points(points_world,intrinsics,world_from_camera):
    points = np.asarray(points_world,dtype=float)
    local = (points-world_from_camera[:3,3]) @ world_from_camera[:3,:3]
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(local).all() or (local[:,2]<=0).any():
        raise ValueError('finite visible points Nx3 in front of the camera required')
    pixels = local @ intrinsics.T
    return pixels[:,:2]/pixels[:,2,None],local[:,2]


def reconstruct_points(pixels,depth,intrinsics,world_from_camera):
    pixels = np.asarray(pixels,dtype=float)
    depth = np.asarray(depth,dtype=float)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or depth.shape != (len(pixels),)
            or not np.isfinite(pixels).all() or not np.isfinite(depth).all() or (depth<=0).any()):
        raise ValueError('finite pixels Nx2 and positive optical depth N required')
    rays = np.column_stack((pixels,np.ones(len(pixels)))) @ np.linalg.inv(intrinsics).T
    local = rays*depth[:,None]
    return local @ world_from_camera[:3,:3].T + world_from_camera[:3,3]


def points_in_hand(points_world,world_from_hand):
    points = np.asarray(points_world,dtype=float)
    transform = np.asarray(world_from_hand,dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4,4):
        raise ValueError('points Nx3 and a homogeneous hand transform 4x4 required')
    if not np.isfinite(points).all() or not np.isfinite(transform).all():
        raise ValueError('finite points and hand transform required')
    homogeneous = np.column_stack((points,np.ones(len(points))))
    return np.linalg.solve(transform,homogeneous.T).T[:,:3]
