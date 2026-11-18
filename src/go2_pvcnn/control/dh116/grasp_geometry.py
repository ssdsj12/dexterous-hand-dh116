"""Cylinder-contact candidates, not collision or dynamic grasp qualification."""
import numpy as np


def _contacts(points, normals, radius):
    p = np.asarray(points, dtype=float)
    n = np.asarray(normals, dtype=float)
    if (p.shape != (2, 3) or n.shape != (2, 3)
            or not np.isfinite(p).all() or not np.isfinite(n).all()
            or not np.isfinite(radius) or radius <= 0
            or not np.allclose(np.linalg.norm(n, axis=1), 1., atol=1e-8, rtol=0)):
        raise ValueError('expected two finite points, unit normals and positive radius')
    return p, n, p + radius * n


def cylinder_residual(points, normals, radius, normal_dot):
    """Tangency at two possibly different material coordinates, plus opposition."""
    _, n, c = _contacts(points, normals, radius)
    if not np.isfinite(normal_dot) or not -1 <= normal_dot <= 1:
        raise ValueError('invalid normal opposition')
    delta = c[1] - c[0]
    return np.r_[n @ delta / radius, n[0] @ n[1] - normal_dot]


def align_grasp_to_wire(points, normals, radius, center, axis, roll):
    """Rigid transform from candidate frame to a chosen straight wire segment.

    This cannot change hand geometry or make a non-tangent candidate tangent.
    Roll zero places the summed outward pad normal toward world +Z when possible.
    """
    _, n, c = _contacts(points, normals, radius)
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if (center.shape != (3,) or axis.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(axis).all()
            or not np.isfinite(roll) or np.linalg.norm(axis) < 1e-12):
        raise ValueError('invalid wire frame')
    delta = c[1] - c[0]
    length = np.linalg.norm(delta)
    if length < 1e-8 or np.max(np.abs(n @ delta)) > 1e-7:
        raise ValueError('contacts do not define a tangent cylinder axis')
    source_x = delta / length
    source_y = n.sum(axis=0)
    source_y -= source_x * (source_x @ source_y)
    if np.linalg.norm(source_y) < 1e-8:
        source_y = n[0] - source_x * (source_x @ n[0])
    source_y /= np.linalg.norm(source_y)
    source = np.column_stack((source_x, source_y, np.cross(source_x, source_y)))
    target_x = axis / np.linalg.norm(axis)
    up = np.array([0., 0., 1.])
    if abs(up @ target_x) > .99:
        up = np.array([1., 0., 0.])
    y = up - target_x * (up @ target_x)
    y /= np.linalg.norm(y)
    target_y = np.cos(roll) * y + np.sin(roll) * np.cross(target_x, y)
    target = np.column_stack((target_x, target_y, np.cross(target_x, target_y)))
    result = np.eye(4)
    result[:3, :3] = target @ source.T
    result[:3, 3] = center - result[:3, :3] @ c.mean(axis=0)
    return result
