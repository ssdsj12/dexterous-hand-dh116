"""Measured failure classification for a bounded grasp/lift retry.

This module returns a supervisor action only.  It never changes a force,
teleports a robot, releases a suspended wire, or turns a failed run into PASS.
"""
from __future__ import annotations

import numpy as np


def diagnose_lift(*, safety_ok, contact_valid, tracking_error_m, slip_m,
                  support_force_n, required_support_force_n, station_rise_m,
                  min_surface_clearance_m):
    if not isinstance(safety_ok, (bool, np.bool_)) or not isinstance(contact_valid, (bool, np.bool_)):
        raise ValueError("safety_ok and contact_valid must be boolean")
    rises = np.asarray(station_rise_m, dtype=float)
    scalars = np.asarray([tracking_error_m, slip_m, support_force_n,
                          required_support_force_n, min_surface_clearance_m], dtype=float)
    if (rises.shape != (2,) or not np.isfinite(rises).all() or np.any(rises < 0)
            or not np.isfinite(scalars).all() or np.any(scalars < 0)
            or required_support_force_n <= 0):
        raise ValueError("finite nonnegative lift observations required")
    if not safety_ok:
        return {"failure_class": "SAFETY_ABORT", "action": "STOP"}
    if tracking_error_m > .01:
        return {"failure_class": "TRACKING_FAIL", "action": "STOP"}
    if not contact_valid:
        return {"failure_class": "CONTACT_LOST", "action": "SAFE_LOWER"}
    if slip_m > .002:
        return {"failure_class": "SLIP", "action": "INCREASE_PRELOAD"}
    if support_force_n < required_support_force_n:
        return {"failure_class": "SUPPORT_INSUFFICIENT", "action": "REPLAN_GRASP"}
    if min_surface_clearance_m < .002 and np.max(rises) >= .03:
        return {"failure_class": "PARTIAL_LIFT", "action": "ADJUST_LIFT"}
    return {"failure_class": "OK", "action": "CONTINUE"}


__all__ = ["diagnose_lift"]
