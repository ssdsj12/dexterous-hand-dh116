"""Coordinate-neutral DH116 angle interpolation; not a calibrated hand model.

All angles are radians. Derivatives are rad/rad. This module does not infer
joint signs, zero offsets, actuator limits, or authorize physical motion.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator


class AngleTable:
    """Own a PCHIP and its analytic derivative with strict measured bounds."""

    def __init__(self, x_rad, y_rad):
        x = np.array(x_rad, dtype=np.float64, copy=True)
        y = np.array(y_rad, dtype=np.float64, copy=True)
        if (x.ndim != 1 or y.shape != x.shape or x.size < 2
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(np.diff(x) <= 0)):
            raise ValueError('expected finite strictly increasing paired samples')
        self._domain = (float(x[0]), float(x[-1]))
        self._value = PchipInterpolator(x, y, extrapolate=False)
        self._derivative = self._value.derivative()

    def evaluate(self, q_rad):
        """Return (angle, derivative), preserving scalar/array query shape.

        Reject the entire request if any query is invalid. No clamp,
        extrapolation, coordinate transform, or mutable last-safe cache.
        """
        q = np.asarray(q_rad, dtype=np.float64)
        lo, hi = self._domain
        if not np.isfinite(q).all() or np.any(q < lo) or np.any(q > hi):
            raise ValueError('query outside measured table domain')
        value, derivative = self._value(q), self._derivative(q)
        if not np.isfinite(value).all() or not np.isfinite(derivative).all():
            raise ValueError('non-finite interpolation result')
        return value, derivative
