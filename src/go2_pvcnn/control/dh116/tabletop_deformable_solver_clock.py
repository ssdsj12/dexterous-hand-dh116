"""Validate RopeBatch's actual float32 accumulator without masking skipped ticks."""
import math

import numpy as np


DT = 1 / 480


def solver_clock_matches(before, after, *, steps):
    """Compare SDK-observed solver seconds to the native repeated float sum.

    A float64 backend or CPU fixture may agree with the real-valued sum instead.
    Match either representation within a fraction of one step. The caller still
    verifies actual SDK and integer rod ticks independently.
    """
    if (type(steps) is not int or not 1 <= steps <= 1_000_000
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   or value > np.finfo(np.float32).max / 2 for value in (before, after))
            or after <= before):
        return False
    rounded = np.float32(before)
    delta = np.float32(DT)
    if float(np.spacing(np.float32(after))) >= DT / 4:
        return False
    for _ in range(steps):
        rounded += delta
    # The existing short-clock comparison used 1e-8. Permit a float32
    # rounding of the observed origin/result, never an entire physical step.
    tolerance = max(1e-8, float(np.spacing(np.float32(after))))
    exact = before + steps * DT
    return min(abs(after - exact), abs(after - float(rounded))) <= tolerance
