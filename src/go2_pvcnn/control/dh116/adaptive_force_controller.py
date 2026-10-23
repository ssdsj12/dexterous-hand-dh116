"""Rate-limited simulation force targets for the DH116 contact loop.

The controller consumes measured contact loads and returns the next commanded
targets.  It never edits a measured force, applies an external wrench, or
declares a grasp.  When a bounded preload increase cannot establish support,
the caller must stop and re-plan the grasp.
"""
from __future__ import annotations

import numpy as np


def allocate_preload(points, normals, com, *, mass_kg, mu, max_load_n,
                     gravity_m_s2=9.81, uncertainty=0.1,
                     safety_factor=1.15, minimum_load_n=0.05):
    """Screen a four-contact preload from measured physical parameters.

    This is a quasistatic *candidate* allocator.  It calls the conservative
    Coulomb-cone LP and returns no target when the support problem is
    infeasible.  The returned targets are commands for the rate-limited
    controller; measured reactions remain exclusively backend observations and
    ``grasp_qualified`` is always false here.
    """
    p = np.asarray(points, dtype=float)
    n = np.asarray(normals, dtype=float)
    c = np.asarray(com, dtype=float)
    cap = np.asarray(max_load_n, dtype=float)
    scalars = np.asarray([mass_kg, mu, gravity_m_s2, uncertainty,
                          safety_factor, minimum_load_n], dtype=float)
    if (p.shape != (4, 3) or n.shape != (4, 3) or c.shape != (3,)
            or cap.shape not in ((), (4,)) or not np.isfinite(p).all()
            or not np.isfinite(n).all() or not np.isfinite(c).all()
            or not np.isfinite(cap).all() or not np.isfinite(scalars).all()
            or mass_kg <= 0 or mu < 0 or gravity_m_s2 <= 0
            or not 0 <= uncertainty < 1 or safety_factor < 1
            or minimum_load_n < 0 or np.any(cap <= 0)):
        raise ValueError("invalid physical preload inputs")
    caps = np.full(4, float(cap)) if cap.shape == () else cap.copy()
    from .grip_loads import solve_support
    # The LP accepts one common cap.  Using the smallest per-contact cap is a
    # conservative screen; the returned command is clipped per contact below.
    weight_n = float(mass_kg * gravity_m_s2 * (1.0 + uncertainty))
    mu_eff = float(mu * (1.0 - uncertainty))
    support = solve_support(p, n, c, weight_n=weight_n, mu=mu_eff,
                            max_load_n=float(np.min(caps)))
    result = dict(feasible=bool(support["feasible"]),
                  grasp_qualified=False, mass_kg=float(mass_kg),
                  mu=float(mu), effective_mu=mu_eff, weight_n=weight_n,
                  support=support, target_loads_n=None)
    if not support["feasible"]:
        return result
    normal = np.asarray(support["normal_loads_n"], dtype=float)
    target = np.clip(np.maximum(minimum_load_n, safety_factor * normal),
                     0., caps)
    result["target_loads_n"] = target.tolist()
    return result


class AdaptiveForceController:
    """Small bounded target-load adapter for a measured contact loop."""

    def __init__(self, initial_loads_n, *, max_loads_n, min_loads_n=None,
                 rate_limit_n_s=1.0, preload_step_n=0.1,
                 slip_threshold_m=0.002, minimum_follow_ratio=0.5):
        initial = self._loads(initial_loads_n, "initial_loads_n")
        maximum = self._loads(max_loads_n, "max_loads_n")
        minimum = (np.zeros(6) if min_loads_n is None
                   else self._loads(min_loads_n, "min_loads_n"))
        scalars = np.asarray([rate_limit_n_s, preload_step_n, slip_threshold_m,
                              minimum_follow_ratio], dtype=float)
        if (not np.isfinite(scalars).all() or rate_limit_n_s <= 0
                or preload_step_n <= 0 or slip_threshold_m < 0
                or not 0 <= minimum_follow_ratio <= 1):
            raise ValueError("invalid adaptive force controller limits")
        if np.any(minimum < 0) or np.any(minimum > maximum) or np.any(initial < minimum) or np.any(initial > maximum):
            raise ValueError("adaptive force targets must lie within limits")
        self.minimum_loads_n = minimum
        self.max_loads_n = maximum
        self.target_loads_n = initial.copy()
        self.rate_limit_n_s = float(rate_limit_n_s)
        self.preload_step_n = float(preload_step_n)
        self.slip_threshold_m = float(slip_threshold_m)
        self.minimum_follow_ratio = float(minimum_follow_ratio)
        self.last_result = None

    @staticmethod
    def _loads(value, name):
        loads = np.asarray(value, dtype=float)
        if loads.shape != (6,) or not np.isfinite(loads).all():
            raise ValueError(f"{name} must contain six finite values")
        return loads.copy()

    def reset(self, loads_n=None):
        if loads_n is not None:
            loads = self._loads(loads_n, "loads_n")
            if np.any(loads < self.minimum_loads_n) or np.any(loads > self.max_loads_n):
                raise ValueError("reset targets must lie within limits")
            self.target_loads_n = loads
        self.last_result = None

    def update(self, measured_loads_n, dt, *, support_feasible=True,
               slip_m=0.0, follow_ratio=1.0, requested_targets_n=None):
        measured = self._loads(measured_loads_n, "measured_loads_n")
        if (not np.isfinite(dt) or dt <= 0 or not isinstance(support_feasible, (bool, np.bool_))
                or not np.isfinite([slip_m, follow_ratio]).all() or slip_m < 0
                or follow_ratio < 0):
            raise ValueError("invalid adaptive force observation")
        requested = self.target_loads_n.copy() if requested_targets_n is None else self._loads(
            requested_targets_n, "requested_targets_n")
        requested = np.clip(requested, self.minimum_loads_n, self.max_loads_n)
        needs_support = (not bool(support_feasible) or slip_m > self.slip_threshold_m
                         or follow_ratio < self.minimum_follow_ratio)
        reason = "tracking"
        desired = requested.copy()
        if needs_support:
            desired = np.minimum(self.max_loads_n, desired + self.preload_step_n)
            reason = "support_insufficient" if not support_feasible else (
                "slip_detected" if slip_m > self.slip_threshold_m else "weak_follow")
        step = self.rate_limit_n_s * float(dt)
        next_target = self.target_loads_n + np.clip(desired - self.target_loads_n, -step, step)
        next_target = np.clip(next_target, self.minimum_loads_n, self.max_loads_n)
        at_cap = bool(np.any(next_target >= self.max_loads_n - 1e-12))
        replan = bool(needs_support and at_cap and np.any(measured < next_target))
        if replan:
            reason = "support_insufficient_at_force_cap"
        self.target_loads_n = next_target
        self.last_result = dict(target_loads_n=next_target.tolist(),
                                measured_loads_n=measured.tolist(),
                                requested_targets_n=requested.tolist(),
                                support_feasible=bool(support_feasible),
                                slip_m=float(slip_m), follow_ratio=float(follow_ratio),
                                replan_required=replan, reason=reason,
                                commands_measured_force=False)
        return dict(self.last_result)


__all__ = ["AdaptiveForceController", "allocate_preload"]
