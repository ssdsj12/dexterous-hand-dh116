"""Direct finger targets and balanced measured contact shaping."""
import numpy as np

from .linear_coupling import active_limits, ACTIVE, JOINTS, MIMIC


def joint_target(center, action, radius=.08):
    center, action = np.asarray(center, dtype=float), np.asarray(action, dtype=float)
    if (center.shape != (12,) or action.shape != (12,)
            or not np.isfinite(np.r_[center, action, radius]).all()
            or np.any(np.abs(action) > 1.) or not 0 < radius <= .15):
        raise ValueError('finite 12D center/action, bounded action and radius in (0,.15] required')
    lo, hi = (np.tile(v, 2) for v in active_limits())
    if np.any(center < lo) or np.any(center > hi):
        raise ValueError('target center violates coupled joint limits')
    return np.clip(center + radius*action, lo, hi)


def coupling_recovery_target(target, measured, errors, warning=.004):
    """Relieve coupling strain using the follower-consistent master angle.

    e = q_f - ratio*q_m - offset, hence q_m + e/ratio = (q_f-offset)/ratio.
    Returning to q_m alone leaves the coupling constraint displaced. The caller
    still applies the normal command rate limit; no joint state is assigned.
    """
    target, measured, errors = (np.asarray(v, dtype=float) for v in (target, measured, errors))
    if (target.shape != (12,) or measured.shape != (12,) or errors.shape != (10,)
            or not np.isfinite(np.r_[target, measured, errors, warning]).all()
            or warning <= 0):
        raise ValueError('finite 12D target/state, 10D errors and positive warning required')
    result = target.copy()
    lo, hi = active_limits()
    for i in np.flatnonzero(np.abs(errors) >= warning):
        _, master, ratio, _ = MIMIC[i % 5]
        joint = ACTIVE.index(master)
        index = (i // 5)*6 + joint
        result[index] = np.clip(measured[index] + errors[i]/ratio, lo[joint], hi[joint])
    return result


def contact_quality(forces):
    forces = np.asarray(forces, dtype=float)
    if forces.shape != (4,) or not np.isfinite(forces).all():
        raise ValueError('four finite pad forces required')
    loads = np.clip(forces/.1, 0., 1.)
    pairs = np.minimum(loads[::2], loads[1::2])
    return float(.1*loads.sum() + .5*pairs.sum() + 2.*loads.min())


def adjust_palms(current, forces, clearance, phase, dt, *, allowed):
    current, forces = np.asarray(current, dtype=float), np.asarray(forces, dtype=float)
    if (current.shape != (2,) or forces.shape != (4,) or dt <= 0
            or not np.isfinite(np.r_[current, forces, clearance, dt]).all()):
        raise ValueError('finite palm state, four forces and positive dt required')
    if phase != 'close' or not allowed:
        return current.copy()
    result = current.copy()
    step = min(.001*dt, max(0., clearance-.0021))
    for i in range(2):
        if not np.any(forces[2*i:2*i+2] >= .1):
            result[i] = max(-.002, result[i]-step)
    return result


def snapshot_state(row, layout):
    """Reconstruct a measured close seed, not solver/contact checkpoint state."""
    if row.get('phase') != 'close' or len(row.get('hands', [])) != 2:
        raise ValueError('measured two-hand close row required')
    roots, positions, velocities, commands = [], [], [], []
    low, high = active_limits()
    for hand in row['hands']:
        named = hand['joints_rad']
        q = np.array([named[n] for n in JOINTS], dtype=float)
        vmap = dict(zip(named, hand['joint_vel_rad_s']))
        qd = np.array([vmap[n] for n in JOINTS], dtype=float)
        active = np.array([named[n] for n in ACTIVE], dtype=float)
        errors = [named[f]-r*named[m]-o for f,m,r,o in MIMIC]
        root = np.asarray(hand['root_pose'], dtype=float)
        if (root.shape != (7,) or not np.isfinite(np.r_[root,q,qd]).all()
                or abs(np.linalg.norm(root[3:])-1) > 1e-4
                or np.any(active < low-.02) or np.any(active > high+.02)
                or max(abs(e) for e in errors) > .02):
            raise ValueError('invalid close seed joints/root')
        roots.append(root);positions.append(q);velocities.append(qd);commands.extend(active)
    roots = np.array(roots)
    original = np.array([layout['hand_roots'][s] for s in ('left','right')])
    delta = roots[:,:3]-original[:,:3]
    if (not np.allclose(delta[0],delta[1],atol=1e-6,rtol=0)
            or not np.allclose(roots[:,3:],original[:,3:],atol=1e-6,rtol=0)
            or abs(delta[0,2]-row['root_offset_m']) > 1e-6):
        raise ValueError('close seed differs from common translated palm layout')
    quat, velocity = np.asarray(row['object_quat']), np.asarray(row['object_velocity'])
    relative = np.asarray(row['object_relative'])
    if (quat.shape != (4,) or velocity.shape != (6,) or relative.shape != (3,)
            or not np.isfinite(np.r_[quat,velocity,relative]).all()
            or abs(np.linalg.norm(quat)-1) > 1e-4):
        raise ValueError('invalid close seed object state')
    return dict(roots=roots, joint_positions=np.array(positions),
                joint_velocities=np.array(velocities), initial_command=np.array(commands),
                root_xy=delta[0,:2], root_offset=float(delta[0,2]),
                wire_pose=np.r_[roots[:,:3].mean(axis=0)+relative,quat], wire_velocity=velocity)
