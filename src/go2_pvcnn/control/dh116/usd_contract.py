"""USD boundary helpers for the DH116 URDF-linear simulation approximation."""
import numpy as np
from .linear_coupling import active_limits, expand


def axis_vector(token):
    if token not in ('X', 'Y', 'Z'):
        raise ValueError(f'invalid USD joint axis: {token!r}')
    return np.eye(3)[('X', 'Y', 'Z').index(token)]


def full_limits_degrees():
    """Image of the feasible active box; all source mimic ratios are positive."""
    lo, hi = active_limits()
    return np.rad2deg(expand(lo)), np.rad2deg(expand(hi))
