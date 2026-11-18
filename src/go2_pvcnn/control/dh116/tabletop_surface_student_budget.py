"""Pure cumulative update budgets for a surface student's diagnostic pilot.

The caller invokes this helper only in explicit or checkpoint-inherited strict
native student validation mode. It selects teacher-initialized or
demonstration-assisted students and excludes evaluation. This helper neither
enables strict mode nor changes checkpoints or default/free PPO runs.
"""


def _count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def diagnostic_budget(saved, requested_max_iterations, requested_timesteps,
                      episodes_per_env, num_envs, rollout_steps, admitted, limit=64):
    """Return already-used diagnostic updates after validating the requested run.

    Iteration and timestep targets are absolute. An explicit saved diagnostic
    counter takes priority over the historical iteration fallback. The caller
    increments and persists the returned count after completed pilot updates.
    An admitted student has no pilot limit; all supplied counts remain checked.
    Strict-mode selection and inheritance belong to the caller, not this API.
    """
    if saved is not None and not isinstance(saved, dict):
        raise ValueError('saved checkpoint must be a dictionary or None')
    if type(admitted) is not bool:
        raise ValueError('admitted must be a validated boolean')
    _count(limit, 'diagnostic update limit')
    _count(num_envs, 'num_envs', 1)
    _count(rollout_steps, 'rollout_steps', 1)
    checkpoint = {} if saved is None else saved
    start_iteration = _count(checkpoint.get('iteration', 0), 'saved iteration')
    start_timesteps = _count(checkpoint.get('timesteps', 0), 'saved timesteps')
    used = _count(checkpoint.get('surface_student_diagnostic_updates', start_iteration),
                  'surface_student_diagnostic_updates')
    if requested_max_iterations is not None:
        _count(requested_max_iterations, 'requested_max_iterations')
    if requested_timesteps is not None:
        _count(requested_timesteps, 'requested_timesteps')
    if episodes_per_env is not None:
        _count(episodes_per_env, 'episodes_per_env', 1)
    if admitted:
        return used
    if episodes_per_env is not None:
        raise ValueError('unadmitted student episode budgets cannot bound diagnostic updates')
    if requested_max_iterations is not None:
        requested_delta = max(0, requested_max_iterations - start_iteration)
    elif requested_timesteps is not None:
        remaining = max(0, requested_timesteps - start_timesteps)
        transitions_per_update = num_envs * rollout_steps
        requested_delta = (remaining + transitions_per_update - 1) // transitions_per_update
    else:
        raise ValueError('unadmitted student requires a finite diagnostic update budget')
    if used + requested_delta > limit:
        raise ValueError(f'diagnostic update budget exceeds limit {limit}: '
                         f'{used} used + {requested_delta} requested')
    return used
