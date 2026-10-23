"""Dense, MPC-aligned rewards for the DH116 hand-only rigid-wire probe."""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_hand_only_reward(
    *, phase: str, action: Any, mpc_action: Any, previous_action: Any,
    contact_force_n: float, wire_lift_m: float, lift_reference_m: float,
    physics_valid: bool = True, lift_completion_event: bool = False,
) -> tuple[float, dict[str, float]]:
    """Return a scalar reward and additive terms; safety qualification stays separate."""
    if phase not in {"grasp", "lift"}:
        raise ValueError("phase must be grasp or lift")
    arrays = [np.asarray(x, dtype=np.float64) for x in (action, mpc_action, previous_action)]
    if any(x.shape != (12,) for x in arrays):
        raise ValueError("action, mpc_action, and previous_action must each have 12 values")
    scalars = np.asarray([contact_force_n, wire_lift_m, lift_reference_m], dtype=np.float64)
    if not all(np.isfinite(x).all() for x in arrays) or not np.isfinite(scalars).all():
        raise ValueError("reward inputs must be finite")
    if contact_force_n < 0 or lift_reference_m < 0:
        raise ValueError("contact force and lift reference must be non-negative")

    executed, reference, previous = arrays
    tracking_error = float(np.mean(((executed - reference) / 0.15) ** 2))
    delta_error = float(np.mean(((executed - previous) / 0.10) ** 2))
    contact = 0.5 * float(np.clip(1.0 - abs(contact_force_n - 5.0) / 5.0, 0.0, 1.0))
    force_penalty = -0.05 * max(0.0, (contact_force_n - 15.0) / 15.0) ** 2
    lift_tracking = (float(np.exp(-abs(wire_lift_m - lift_reference_m) / 0.01))
                     if phase == "lift" else 0.0)
    lift_complete = float(lift_completion_event and phase == "lift" and physics_valid and contact_force_n >= 1.0
                          and wire_lift_m >= 0.02)
    terms = {
        "mpc_tracking": float(np.exp(-tracking_error)),
        "action_smoothness": -0.05 * delta_error,
        "contact": contact,
        "force_penalty": force_penalty,
        "lift_tracking": lift_tracking,
        "lift_complete": lift_complete,
    }
    return float(sum(terms.values())), terms


__all__ = ["compute_hand_only_reward"]
