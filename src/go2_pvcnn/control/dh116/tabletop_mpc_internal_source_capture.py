"""Named simulator task state readback for future 371D MPC captures."""

import math

import torch


VERSION = 'dh116-mpc-internal-state-source-v1'
SCOPE = 'native_simulator_internal_value_parity_only'
_TASK = {
    'reference': (12,), 'elapsed': (), 'contact_time': (), 'success_time': (),
    'close_progress': (), 'grip': (3,), 'grip_valid': (), 'filtered_residual': (12,),
    'f_offset': (2,), 'f_filtered': (2,), 'f_target': (2,), 'f_boost': (2,),
    'f_requested': (2,), 'f_last_slip': (), 'contact_reward_clock': (),
    'grasp_ready_awarded': (), 'force_scoring_started': (), 'rotation_targets': (2, 3),
}
_TASK_BOOL = {'grip_valid', 'grasp_ready_awarded', 'force_scoring_started'}
_ACCUMULATORS = {
    '_sum': (20,), '_elapsed': (), '_depth': (), '_drift': (), '_hold': (),
    '_start_z': (2,), '_end_z': (2,), '_window_good': (), '_valid': (), '_open': (), '_ok': (),
}
_ACCUMULATOR_BOOL = {'_window_good', '_valid', '_open', '_ok'}


def _read(owner, name, shape, *, boolean=False):
    value = getattr(owner, name, None)
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
            or (value.dtype != torch.bool if boolean else not value.is_floating_point())):
        raise ValueError(f'missing or incompatible native internal source {name}')
    return value.detach().to(device='cpu', copy=True)


def capture_internal_sources(task, environments, *, after_reset):
    """Read actual named task state. This observer never asks for observe()."""
    feedback = getattr(task, 'surface_feedback', None)
    if (getattr(task, 'surface_force_sharing', None) is not True
            or getattr(task, 'free_wrist_grasp', None) is not True
            or not getattr(task, 'contact_reward_window', None)
            or getattr(task, 'force_reward_start', None) in (None, 'episode_start')
            or feedback is None or type(after_reset) is not bool):
        raise ValueError('canonical 371D native surface/free-wrist task and feedback required')
    stop = getattr(task, 'descent_stop', None)
    if type(stop) is not int or not 0 <= stop < 3 or type(getattr(feedback, '_started', None)) is not bool:
        raise ValueError('invalid native internal state or feedback lifecycle')
    maximum = task.parameters.max_force_n
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError('invalid native force-controller normalizer')
    raw_task = {name: _read(task, name, (environments, *tail), boolean=name in _TASK_BOOL)
                for name, tail in _TASK.items()}
    bounds = {key: _read(task, key, (12,)) for key in ('low', 'high')}
    if not (bounds['high'] > bounds['low']).all():
        raise ValueError('invalid native joint bounds for reference normalization')
    accumulators = {name: _read(feedback, name, (environments, *tail),
                               boolean=name in _ACCUMULATOR_BOOL)
                    for name, tail in _ACCUMULATORS.items()}
    return dict(version=VERSION, after_reset=after_reset, descent_stop=stop,
                raw_task=raw_task, bounds=bounds, force_max_n=maximum,
                feedback_started=feedback._started, feedback_accumulators=accumulators)


def internal_source_masks(observations, source):
    """Prove controller values and separately prove an original reset sentinel."""
    if source['version'] != VERSION or observations.ndim != 2 or observations.shape[1] != 371:
        raise ValueError('native 371D internal evidence required')
    m = source['raw_task']
    low, high = (source['bounds'][key] for key in ('low', 'high'))
    n = len(observations)
    stop = torch.nn.functional.one_hot(torch.full((n,), source['descent_stop']), 3)
    values = {
        (36, 48): 2 * (m['reference'] - low) / (high - low) - 1,
        (52, 55): torch.stack([m[key] for key in ('elapsed', 'contact_time', 'success_time')], 1) / 15.,
        (56, 57): m['close_progress'][:, None],
        (57, 60): stop,
        (74, 77): m['grip'] / .05,
        (77, 78): m['grip_valid'][:, None],
        (88, 100): m['filtered_residual'] / .1,
        (131, 132): m['f_last_slip'][:, None] / .02,
        (144, 145): m['contact_reward_clock'][:, None],
        (145, 146): m['grasp_ready_awarded'][:, None],
        (146, 147): m['force_scoring_started'][:, None],
        (185, 191): m['rotation_targets'].flatten(1) / (math.pi / 2),
    }
    mask = torch.zeros_like(observations, dtype=torch.bool)
    for (start, stop), candidate in values.items():
        candidate = candidate.to(observations)
        if candidate.shape != (n, stop - start):
            raise ValueError('native internal source batch differs from captured 371D')
        mask[:, start:stop] = torch.isfinite(candidate) & candidate.eq(observations[:, start:stop])
    mask[:, 74:77] &= m['grip_valid'].to(mask.device)[:, None]
    # A zero observation is insufficient. Require the original whole-env reset
    # boundary and the independently captured cleared accumulator state.
    reset = torch.zeros_like(mask)
    if source['after_reset'] and source['feedback_started'] is False:
        accum = source['feedback_accumulators']
        clear = ~(accum['_valid'] | accum['_open'] | accum['_ok'] | accum['_window_good'])
        for key in ('_elapsed', '_sum', '_depth', '_drift', '_hold', '_start_z', '_end_z'):
            clear &= accum[key].reshape(n, -1).eq(0).all(1)
        reset[:, 367:371] = clear.to(reset.device)[:, None] & observations[:, 367:371].eq(0)
    return dict(internal_state_value_valid=mask, reset_feedback_initialization_value_valid=reset)


def select_internal_row(source, index):
    return dict(source, raw_task={name: value[index].clone() for name, value in source['raw_task'].items()},
        feedback_accumulators={name: value[index].clone()
                               for name, value in source['feedback_accumulators'].items()})
