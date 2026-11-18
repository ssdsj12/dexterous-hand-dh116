"""DH116 surface coordinates, in metres, with column-vector transforms.

Pure geometry only: neither a selected pad calibration nor hardware evidence.
Triangle winding defines a normal but does not establish which side is outward.
"""
import numpy as np
from .linear_coupling import JOINTS, fold_jacobian


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result


def _unit(vector, name, minimum=1e-12):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length <= minimum:
        raise ValueError(f'{name} is degenerate or overflowed')
    return vector / length


def surface_reference(triangle, weights):
    """Return body-local surface point and winding normal; no mesh identity claim."""
    vertices = _array(triangle, (3, 3), 'triangle')
    weights = _array(weights, (3,), 'weights')
    if np.any(weights < 0.) or abs(float(weights.sum()) - 1.) > 1e-9:
        raise ValueError('weights must be nonnegative and sum to one')
    with np.errstate(over='ignore', invalid='ignore'):
        cross = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        normal = _unit(cross, 'triangle', minimum=1e-14)
        point = weights @ vertices
    return _array(point, (3,), 'point'), normal


def palm_frame(origin, outward, wrist_to_middle):
    """Base-link-from-palm frame: +Z outward, +Y wrist-to-middle projected."""
    origin = _array(origin, (3,), 'origin')
    outward = _array(outward, (3,), 'outward')
    direction = _array(wrist_to_middle, (3,), 'wrist_to_middle')
    with np.errstate(over='ignore', invalid='ignore'):
        z = _unit(outward, 'outward')
        y = _unit(direction - z * np.dot(z, direction), 'projected direction')
        x = np.cross(y, z)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x, y, z))
    result[:3, 3] = origin
    return result


def transform_reference(transform, point, normal):
    """Transform a point/normal through a proper rigid transform, no scale/shear."""
    transform = _array(transform, (4, 4), 'transform')
    point = _array(point, (3,), 'point')
    normal = _array(normal, (3,), 'normal')
    rotation = transform[:3, :3]
    with np.errstate(over='ignore', invalid='ignore'):
        gram = rotation.T @ rotation
        determinant = np.linalg.det(rotation)
        normal_length = np.linalg.norm(normal)
    if (not np.allclose(transform[3], [0., 0., 0., 1.], rtol=0., atol=1e-8)
            or not np.allclose(gram, np.eye(3), rtol=0., atol=1e-8)
            or not np.isfinite(determinant) or abs(determinant - 1.) > 1e-8):
        raise ValueError('transform must be a proper rigid transform')
    if not np.isfinite(normal_length) or abs(normal_length - 1.) > 1e-8:
        raise ValueError('normal must be unit length')
    with np.errstate(over='ignore', invalid='ignore'):
        transformed_point = rotation @ point + transform[:3, 3]
        transformed_normal = rotation @ normal
    return (_array(transformed_point, (3,), 'transformed point'),
            _array(transformed_normal, (3,), 'transformed normal'))


def point_jacobian(spatial_jacobian, offset_world, joint_names, side):
    """Shift world [linear; angular] rows, then reduce 18 columns to 13.

    offset_world is FROM the supplied Jacobian's reference origin TO the
    desired point. Caller must identify that origin (COM or link) explicitly;
    this pure function cannot infer it. Output: Panda1..7, then DH116 ACTIVE.
    """
    spatial = _array(spatial_jacobian, (6, 18), 'spatial_jacobian')
    offset = _array(offset_world, (3,), 'offset_world')
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    arm_names = tuple(f'{side}_panda_joint{i}' for i in range(1, 8))
    names = tuple(joint_names)
    if (len(names) != 18 or any(not isinstance(n, str) for n in names)
            or len(set(names)) != 18 or set(names) != set(arm_names + JOINTS)):
        raise ValueError('expected exactly named Panda7 and DH11611 joints')
    with np.errstate(over='ignore', invalid='ignore'):
        shifted = spatial[:3] + np.cross(spatial[3:].T, offset).T
        _array(shifted, (3, 18), 'shifted Jacobian')
        arms = shifted[:, [names.index(n) for n in arm_names]]
        hand = shifted[:, [names.index(n) for n in JOINTS]]
        result = np.concatenate((arms, fold_jacobian(hand, JOINTS)), axis=1)
    return _array(result, (3, 13), 'point Jacobian')
