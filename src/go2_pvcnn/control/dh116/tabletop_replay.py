"""Replay recorded actions through normal measured reset/step transitions.

Logs supply actions and the close reset height only. Recorded contacts, joint
states, rewards and success flags never replace simulator measurements.
"""
from collections.abc import Mapping

import numpy as np

from .tabletop_training import PHYSICAL_FAILURES


def replay_episode(env, rows, *, seed=42, extend_zero_steps=0):
    """Replay a close trajectory, optionally continuing with zero actions.

    ``frames_used`` includes source and extension steps. ``max_mimic`` is the
    peak absolute error in reset/control-step measurements, not a guaranteed
    physics-substep peak. Exhausting either action budget leaves a partial
    episode; only the environment's actual terminal success can qualify it.
    The caller owns flushing, reporting and simulation lifecycle.
    """
    if (isinstance(extend_zero_steps, (bool, np.bool_))
            or not isinstance(extend_zero_steps, (int, np.integer))
            or extend_zero_steps < 0):
        raise ValueError('zero extension must be a nonnegative integer')
    if getattr(env, 'reset_phase', None) != 'close':
        raise ValueError('replay requires an environment configured for close reset')
    try:
        source = list(rows)
    except TypeError as exc:
        raise ValueError('nonempty close trajectory required') from exc
    if not source or not isinstance(source[0], Mapping) or source[0].get('phase') != 'close':
        raise ValueError('nonempty trajectory starting in close required')
    try:
        offset = np.asarray(source[0]['root_offset_m'], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('finite positive scalar reset offset required') from exc
    if offset.shape != () or not np.isfinite(offset) or offset <= 0:
        raise ValueError('finite positive scalar reset offset required')
    actions = []
    for row in source:
        if not isinstance(row, Mapping) or row.get('phase') not in ('close', 'hold', 'lift'):
            raise ValueError('replay supports close/hold/lift logs only')
        try:
            action = np.asarray(row['residual_action'], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('finite 12D action in [-1,1] required') from exc
        if action.shape != (12,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('finite 12D action in [-1,1] required')
        actions.append(action.copy())

    # Validate the whole input before the first physics mutation. In close,
    # the shared root offset remains constant through the first logged frame.
    env.initial_offset = float(offset)
    env.reset_jitter = 0.
    observation, info = env.reset(seed=seed)
    frames = source_frames = extension_frames = 0
    terminated = truncated = False
    max_mimic = 0.

    def finite_measurement(obs, reward=0.):
        nonlocal max_mimic
        errors = np.asarray(env.measurement.get('mimic_errors', []), dtype=float)
        finite = bool(np.isfinite(obs).all() and np.isfinite(reward)
                      and np.isfinite(errors).all())
        if errors.size and np.isfinite(errors).all():
            max_mimic = max(max_mimic, float(np.max(np.abs(errors))))
        return finite

    finite = finite_measurement(observation)
    for index in range(len(actions) + extend_zero_steps):
        if not finite:
            break
        from_source = index < len(actions)
        action = actions[index] if from_source else np.zeros(12)
        observation, reward, terminated, truncated, info = env.step(action)
        frames += 1
        source_frames += int(from_source)
        extension_frames += int(not from_source)
        finite = finite_measurement(observation, reward)
        if terminated or truncated:
            break

    failure = info.get('failure')
    safety = info.get('safety_failure')
    if failure in PHYSICAL_FAILURES:
        safety = failure
    if not finite:
        safety = failure = 'nonfinite_state'
    partial = not bool(terminated) or bool(truncated)
    success = bool(terminated and not truncated and finite and not failure
                   and not safety and info.get('is_success', False))
    return dict(frames_used=frames, source_frames_used=source_frames,
                zero_extension_steps=extension_frames,
                source_exhausted=source_frames == len(actions),
                partial=partial, terminated=bool(terminated), truncated=bool(truncated),
                finite=finite, safety_failure=safety, is_success=success,
                failure=failure, max_mimic=max_mimic,
                reset_offset_m=float(offset), effective_reset_jitter_m=0., seed=seed)
