"""Pure action/observation contracts for measured palm and stiffness control."""
import numpy as np

ACTION_SIZE = 20
OBSERVATION_SIZE = 125
PALM_LIMITS_M = np.array([.003, .003, .004])
STIFFNESS_MIN = 10.
STIFFNESS_MAX = 20.


def split_action(action):
    """Return normalized joint, per-hand XYZ and per-hand stiffness commands."""
    try:
        action = np.asarray(action, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('adaptive action must contain 20 finite normalized values') from exc
    if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all() or np.any(abs(action) > 1.):
        raise ValueError('adaptive action must contain 20 finite values in [-1,1]')
    return action[:12].copy(), action[12:18].reshape(2, 3).copy(), action[18:].copy()


def measured_state(measurement):
    """Validate real backend measurements; never substitute absent channels."""
    fields = (('adaptive_palm_offset', (2, 3)), ('grip_stiffness', (2,)), ('pad_relative', (4, 3)))
    values = []
    for name, shape in fields:
        try:
            value = np.asarray(measurement[name], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'measured {name} is required') from exc
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'measured {name} must be finite with shape {shape}')
        values.append(value)
    return tuple(values)


def observation_tail(measurement):
    palm, stiffness, pads = measured_state(measurement)
    try:
        bottom = np.asarray(measurement['bottom'], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('measured absolute object bottom is required') from exc
    if bottom.shape != () or not np.isfinite(bottom):
        raise ValueError('measured absolute object bottom must be a finite scalar')
    with np.errstate(over='ignore', invalid='ignore'):
        tail = np.r_[(palm/PALM_LIMITS_M).ravel(),
            (stiffness-STIFFNESS_MIN)/(STIFFNESS_MAX-STIFFNESS_MIN),
            (float(bottom)-.30)/.03, pads.ravel()/.05]
    if not np.isfinite(tail).all() or np.any(abs(tail) > np.finfo(np.float32).max):
        raise ValueError('adaptive measurements must produce a finite float32 observation')
    return tail.astype(np.float32)
