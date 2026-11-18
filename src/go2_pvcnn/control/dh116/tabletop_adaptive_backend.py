"""Pure bounds and measured pad geometry for adaptive grasp actions."""
import numpy as np

from .tabletop_direct import joint_target


PALM_LIMIT_M = np.array([.003, .003, .004])
PALM_RATE_M_S = .002
STIFFNESS_MIN = 10.
STIFFNESS_MAX = 20.
STIFFNESS_RATE = 10.
DESCENT_CLEARANCE_M = .0021


def _array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name} must have finite shape {shape}')
    return value


def adaptive_step(*, palm_offset, stiffness, palm_action, grip_action,
                  hand_clearance, dt, palm_motion_allowed, parameters=None):
    """Rate-limit absolute seed-relative targets without advancing physics.

    Missing optional actions hold the current state. The palm gate freezes only
    palm motion: grip stiffness remains policy-controlled, including reductions.
    Descent consumes only that hand's measured all-collision clearance budget;
    upward and horizontal motion retain their independent rate limits.
    """
    from .tabletop_load_parameters import LEGACY_LOAD_PARAMETERS
    parameters = parameters or LEGACY_LOAD_PARAMETERS
    maximum, rate = parameters.stiffness_max, parameters.stiffness_rate
    palm = _array(palm_offset, (2, 3), 'palm offset')
    gains = _array(stiffness, (2,), 'stiffness')
    clearance = _array(hand_clearance, (2,), 'hand clearance')
    if (not np.isscalar(dt) or not np.isfinite(dt) or not 0 < dt <= 1/60
            or not isinstance(palm_motion_allowed, (bool, np.bool_))):
        raise ValueError('a positive substep at most 1/60 s and boolean palm gate required')
    if (np.any(np.abs(palm) > PALM_LIMIT_M+1e-9)
            or np.any(gains < STIFFNESS_MIN) or np.any(gains > maximum)):
        raise ValueError('adaptive state exceeds palm or stiffness bounds')
    palm_target = palm.copy()
    if palm_action is not None:
        action = _array(palm_action, (2, 3), 'palm action')
        if np.any(np.abs(action) > 1.):
            raise ValueError('palm action must be within [-1,1]')
        palm_target = action*PALM_LIMIT_M
    gain_target = gains.copy()
    if grip_action is not None:
        action = _array(grip_action, (2,), 'grip action')
        if np.any(np.abs(action) > 1.):
            raise ValueError('grip action must be within [-1,1]')
        gain_target = STIFFNESS_MIN+(action+1.)*.5*(maximum-STIFFNESS_MIN)
    next_palm = palm.copy()
    if palm_motion_allowed:
        movement = np.clip(palm_target-palm, -PALM_RATE_M_S*dt, PALM_RATE_M_S*dt)
        movement[:, 2] = np.maximum(movement[:, 2], -np.maximum(clearance-DESCENT_CLEARANCE_M, 0.))
        next_palm = np.clip(palm+movement, -PALM_LIMIT_M, PALM_LIMIT_M)
    next_gains = gains+np.clip(gain_target-gains, -rate*dt, rate*dt)
    return next_palm, np.clip(next_gains, STIFFNESS_MIN, maximum)


def pad_relative_positions(link_poses, local_points, wire_center):
    """Transform calibrated pad points using actual wxyz link poses.

    Ordering is left thumb/index, then right thumb/index. The output is in world
    axes relative to the measured wire center, so vector-scene offsets cancel.
    """
    poses = _array(link_poses, (4, 7), 'link poses')
    points = _array(local_points, (4, 3), 'local pad points')
    center = _array(wire_center, (3,), 'wire center')
    norms = np.linalg.norm(poses[:, 3:], axis=1)
    if not np.allclose(norms, 1., atol=1e-4, rtol=0.):
        raise ValueError('measured pad-link quaternion must be unit length')
    quat = poses[:, 3:]/norms[:, None]
    xyz = quat[:, 1:]
    cross = 2.*np.cross(xyz, points)
    rotated = points+quat[:, :1]*cross+np.cross(xyz, cross)
    return poses[:, :3]+rotated-center


def select_grasp_command(default, seed):
    """Use an explicit calibrated target center when supplied by the seed."""
    center = default if seed is None else seed.get('grasp_command', default)
    return joint_target(center, np.zeros(12)).copy()
