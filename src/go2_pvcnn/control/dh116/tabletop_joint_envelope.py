"""Explicit per-joint action bounds, independent of controller rate limiting."""
import numpy as np
from .linear_coupling import active_limits


def bounds(profile):
    if profile.get('version') != 'dh116-joint-envelope-v1':
        raise ValueError('unknown joint envelope version')
    arrays = tuple(np.asarray(profile.get(k), dtype=float)
                   for k in ('lower_rad', 'center_rad', 'upper_rad'))
    if any(a.shape != (12,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('joint envelope requires three finite 12D arrays')
    low, center, high = arrays
    legal_low, legal_high = (np.tile(a, 2) for a in active_limits())
    if (np.any(low < legal_low) or np.any(high > legal_high)
            or np.any(low > center) or np.any(center > high)):
        raise ValueError('joint envelope violates coupled limits or ordering')
    return arrays


def targets(profile, action):
    low, center, high = bounds(profile)
    action = np.asarray(action, dtype=float)
    if (action.ndim not in (1, 2) or action.shape[-1] != 12
            or not np.isfinite(action).all() or np.any(abs(action) > 1.)):
        raise ValueError('finite bounded 12D joint action required')
    return center + action * np.where(action >= 0., high-center, center-low)


def clip_targets(profile, command):
    low, _, high = bounds(profile)
    command = np.asarray(command, dtype=float)
    if (command.ndim not in (1, 2) or command.shape[-1] != 12
            or not np.isfinite(command).all()):
        raise ValueError('finite 12D joint command required')
    return np.clip(command, low, high)
