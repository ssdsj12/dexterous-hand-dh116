"""Pure contracts for two-hand residual control and measured PPO observations.

Nothing here advances physics or invents contact. A nominal reference must be
advanced independently by the caller, then frozen through hold and lift.
"""
import numpy as np

from .linear_coupling import active_limits
from .tabletop_task import REWARD_PARAMETERS


PHASES = ('approach', 'close', 'hold', 'lift')
STAGES = ('approach', 'close', 'lift', 'full')
OBSERVATION_VERSION = 'dh116-tabletop-v4'
OBSERVATION_SIZE = 104


def _vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return array


def _scalar(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != () or not np.isfinite(array):
        raise ValueError(f'{name} must be a finite scalar')
    return float(array)


def bounded_residual_target(reference, residual, scale=.1):
    """Apply a bounded 12D action to an independent nominal reference.

The two six-joint targets are clipped to the affine-coupled DH116 limits.
The caller applies the separate 0.3 rad/s command rate limit. Passing the last
executed command back as ``reference`` would defeat the no-drift contract.
"""
    reference = _vector(reference, 12, 'reference')
    residual = _vector(residual, 12, 'residual')
    scale = _scalar(scale, 'scale')
    if np.any(np.abs(residual) > 1.) or not 0. <= scale <= .1:
        raise ValueError('residual must lie in [-1,1] and scale in [0,0.1] rad')
    low, high = (np.tile(bound, 2) for bound in active_limits())
    if np.any(reference < low) or np.any(reference > high):
        raise ValueError('nominal reference must satisfy coupled joint limits')
    return np.clip(reference+scale*residual, low, high)


def track_root_xy(current, object_position, spawn_position, phase, dt, *, allowed=True):
    """Bound scripted planar tracking; freeze the palm frame after close."""
    current = _vector(current, 2, 'current')
    desired = _vector(object_position, 3, 'object_position')[:2] - _vector(spawn_position, 3, 'spawn_position')[:2]
    dt = _scalar(dt, 'dt')
    if dt <= 0 or phase not in PHASES:
        raise ValueError('positive dt and known phase required')
    return current + np.clip(desired-current, -.01*dt, .01*dt) if allowed and phase in ('approach', 'close') else current.copy()


def moving_goal_error(roots, goals, object_position, spawn_position):
    """Measured root distance to the object-translated target, including lag."""
    roots, goals = np.asarray(roots, dtype=float), np.asarray(goals, dtype=float)
    if roots.shape != (2, 3) or goals.shape != (2, 3) or not np.isfinite(np.r_[roots, goals]).all():
        raise ValueError('two finite root/goal positions required')
    delta = _vector(object_position, 3, 'object_position') - _vector(spawn_position, 3, 'spawn_position')
    translated = goals.copy()
    translated[:, :2] += delta[:2]
    return float(np.max(np.linalg.norm(roots-translated, axis=1)))


def prepared_approach_reference(grasp):
    """Open thumb/index by at most 0.04 rad, preserving the calibrated posture."""
    grasp = _vector(grasp, 6, 'grasp')
    low, high = active_limits()
    if np.any(grasp < low) or np.any(grasp > high):
        raise ValueError('grasp violates coupled limits')
    result = grasp.copy()
    result[:3] = np.maximum(low[:3], result[:3]-.04)
    return result


def curriculum_complete(stage, task):
    """Stage completion is independent of the original full grasp gate."""
    if stage not in STAGES:
        raise ValueError(f'unknown curriculum stage: {stage}')
    if task.phase not in PHASES:
        raise ValueError(f'unknown task phase: {task.phase}')
    if task.failure:
        return False
    if task.grasp_mode == 'object_stability' and stage != 'approach':
        return bool(task.success)
    if stage == 'approach':
        return task.phase != 'approach'
    if stage == 'close':
        return task.phase == 'lift'
    return bool(task.success)


def build_observation(*, q, qd, previous, reference, forces, task, offset,
                      close_progress, descent_stop, clearance, object_relative,
                      object_quat, object_velocity, grip_relative, filtered_residual=None, mimic_errors=None):
    """Return the versioned 104D float32 observation without hiding latch state.

Order: q12, qd12, previous12, reference12, forces4, phase onehot4,
elapsed/contact/success timers3, offset1, close_progress1, descent-stop
onehot3 (None, pad_contact, table_margin), clearance1, object_relative3,
object_quat4 (caller convention), object linear/angular velocity6,
grip_relative3 and a grip-reference-valid bit1, measured mimic errors10, filtered residual radians12.

Positions/targets use affine-coupled limits mapped to [-1,1]; qd uses .3
rad/s. Forces use 30 N; timers use the 15 s timeout; distances use .05 m.
Object linear velocity uses .1 m/s, angular velocity 1 rad/s. Quaternion,
close progress and onehot bits are unscaled. Residual filter state uses .1 rad.
Values are not clipped, so
unsafe measurements remain visible. None grip reference is zero plus bit 0.
"""
    q, qd, previous, reference = (
        _vector(value, 12, name) for name, value in
        (('q', q), ('qd', qd), ('previous', previous), ('reference', reference)))
    forces = _vector(forces, 4, 'forces')
    object_relative = _vector(object_relative, 3, 'object_relative')
    object_quat = _vector(object_quat, 4, 'object_quat')
    object_velocity = _vector(object_velocity, 6, 'object_velocity')
    mimic_errors = (np.zeros(10) if mimic_errors is None else
                    _vector(mimic_errors, 10, 'mimic_errors'))
    valid_grip = grip_relative is not None
    grip_relative = (_vector(grip_relative, 3, 'grip_relative') if valid_grip
                     else np.zeros(3))
    filtered_residual = (np.zeros(12) if filtered_residual is None else
                         _vector(filtered_residual, 12, 'filtered_residual'))
    if task.phase not in PHASES:
        raise ValueError(f'unknown task phase: {task.phase}')
    stop_reasons = (None, 'pad_contact', 'table_margin')
    if descent_stop not in stop_reasons:
        raise ValueError(f'unknown descent stop: {descent_stop}')
    times = np.array([_scalar(getattr(task, name), name)
                      for name in ('elapsed', 'contact_time', 'success_time')])
    if np.any(times < 0.):
        raise ValueError('task times must be nonnegative')
    offset = _scalar(offset, 'offset')
    close_progress = _scalar(close_progress, 'close_progress')
    clearance = _scalar(clearance, 'clearance')
    low, high = (np.tile(bound, 2) for bound in active_limits())

    def joints(value):
        return 2.*(value-low)/(high-low)-1.

    result = np.concatenate((joints(q), qd/.3, joints(previous), joints(reference),
        forces/30., np.eye(4)[PHASES.index(task.phase)], times/15.,
        [offset/.05, close_progress], np.eye(3)[stop_reasons.index(descent_stop)],
        [clearance/.05], object_relative/.05, object_quat,
        object_velocity/np.array([.1, .1, .1, 1., 1., 1.]),
        grip_relative/.05, [float(valid_grip)], mimic_errors/.02, filtered_residual/.1))
    # Casting extremely large finite sensor values must not emit infinities.
    if (result.shape != (OBSERVATION_SIZE,) or not np.isfinite(result).all()
            or np.any(np.abs(result) > np.finfo(np.float32).max)):
        raise ValueError('observation cannot be represented as finite float32')
    return result.astype(np.float32)


def _metrics(values, name):
    try:
        result = {key: _scalar(values[key], f'{name}.{key}') for key in (
            'approach_error_m', 'clearance_m', 'object_bottom_m',
            'object_speed_m_s', 'slip_m', 'max_force_n')}
        result['pad_forces_n'] = _vector(values['pad_forces_n'], 4, f'{name}.pad_forces_n')
    except (KeyError, TypeError) as exc:
        raise ValueError(f'{name} requires complete measured task metrics') from exc
    return result


def progress_reward(*, phase, previous_metrics, metrics, action, reference,
                    previous, dt, success_event, failed, stage_event=False):
    """Reward measured change; idle poses receive only nonpositive terms.

Progress terms are differences of bounded potentials, without another dt
factor. Tracking/safety/smoothness/time costs retain per-second scaling.
The caller emits the grasp (+25), failure (-25) and stage (+5) events once.
"""
    if phase not in PHASES:
        raise ValueError(f'unknown task phase: {phase}')
    action, reference, previous = (
        _vector(value, 12, name) for name, value in
        (('action', action), ('reference', reference), ('previous', previous)))
    dt = _scalar(dt, 'dt')
    if dt <= 0.:
        raise ValueError('positive dt required')
    before = _metrics(previous_metrics, 'previous_metrics')
    now = _metrics(metrics, 'metrics')

    def approach_potential(value):
        return float(np.exp(-max(0., value['approach_error_m'])/.025))

    def contact_potential(value):
        return float(np.mean(np.clip(value['pad_forces_n'], 0., 1.)))

    def lift_potential(value):
        return float(np.clip((value['object_bottom_m']-.30)/.03, 0., 1.))

    grasped = bool(np.all(now['pad_forces_n'] >= .1)
                   and np.all(before['pad_forces_n'] >= .1))
    terms = dict(
        approach=REWARD_PARAMETERS['approach']*(approach_potential(now)-approach_potential(before))
                 if phase == 'approach' else 0.,
        contact=REWARD_PARAMETERS['contact']*(contact_potential(now)-contact_potential(before))
                if phase in ('close', 'hold') else 0.,
        lift=REWARD_PARAMETERS['lift']*(lift_potential(now)-lift_potential(before))
             if phase == 'lift' and grasped else 0.)
    # Match the old reward's bounded tracking scale with its maximum at zero.
    tracking_error = float(np.mean(((action-reference)/.15)**2))
    costs = dict(mpc_tracking=1.-np.exp(-tracking_error),
                 slip=min((now['slip_m']/.005)**2, 25.),
                 smoothness=min(float(np.mean(((action-previous)/.1)**2)), 25.),
                 force=min(max(0., now['max_force_n']-15.)**2/225., 25.), time=1.)
    for name, value in costs.items():
        weight = -REWARD_PARAMETERS[name] if name == 'mpc_tracking' else REWARD_PARAMETERS[name]
        terms[name] = float(value*weight*dt)
    terms['success'] = REWARD_PARAMETERS['success'] if success_event and not failed else 0.
    terms['failure'] = REWARD_PARAMETERS['failure'] if failed else 0.
    terms['stage'] = 5. if stage_event and not failed else 0.
    return float(sum(terms.values())), terms
