"""Vectorized measured geometry, with wxyz link poses and exact hull vertices."""
import numpy as np


def normalized_convex(points):
    """Cook a well-scaled hull, then restore identical geometry by mesh scale."""
    points = np.asarray(points, dtype=float)
    center = (points.max(axis=0)+points.min(axis=0))/2
    scale = np.ptp(points, axis=0)/2
    if np.any(scale <= 0) or not np.isfinite(points).all():
        raise ValueError('finite nondegenerate convex vertices required')
    return (points-center)/scale, center, scale


def rotated_points(poses, points):
    poses, points = np.asarray(poses), np.asarray(points)
    xyz = poses[..., None, 4:]
    cross = 2*np.cross(xyz, points)
    return points + poses[..., None, 3:4]*cross + np.cross(xyz, cross) + poses[..., None, :3]


def rotated_heights(poses, points):
    """Exact z component of the existing quaternion transform, for every vertex.

    Avoid allocating two unused coordinates and cross-product temporaries for
    thousands of environments. Float64 matches the original hull arithmetic.
    No hull subsampling, bounding sphere or quaternion normalization is used.
    """
    poses, points = np.asarray(poses,dtype=float), np.asarray(points,dtype=float)
    w,x,y,z = np.moveaxis(poses[...,3:],-1,0)
    row = np.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),axis=-1)
    return np.einsum('...i,vi->...v',row,points)+poses[...,2,None]


def collision_clearance(poses, shapes):
    heights = np.stack([rotated_heights(poses[:, hand, body], points).min(axis=-1)
                        for hand, body, points in shapes], axis=1)
    per_hand = np.stack([heights[:, [i for i, s in enumerate(shapes) if s[0] == h]].min(axis=1)
                         for h in range(2)], axis=1)
    return per_hand-.30, heights.argmin(axis=1)
