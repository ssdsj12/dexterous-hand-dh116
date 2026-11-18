"""A pinned candidate-3 hand configuration for fresh direct-Close MPC runs.

The lock is deliberately local: it does not claim that the offline pose is
physically safe.  It only prevents an MPC episode from silently switching to
an unrelated hand configuration while the candidate is being validated.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .linear_coupling import ACTIVE, active_limits
from .pose_artifact import load_active_pose


@dataclass(frozen=True)
class PoseLock:
    source_path: str
    source_sha256: str
    states: dict[str, np.ndarray]
    tolerance_rad: float

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(f"{side}_{name}" for side in ("left", "right") for name in ACTIVE)

    @property
    def active_reference(self) -> np.ndarray:
        return np.concatenate([self.states[side][7:] for side in ("left", "right")]).copy()

    def active_bounds(self, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        reference = self.active_reference
        if (lo.shape != (12,) or hi.shape != (12,) or not np.isfinite(lo).all()
                or not np.isfinite(hi).all() or np.any(lo >= hi)):
            raise ValueError("finite 12D active limits required")
        bounded_lo = np.maximum(lo, reference - self.tolerance_rad)
        bounded_hi = np.minimum(hi, reference + self.tolerance_rad)
        if np.any(bounded_lo >= bounded_hi):
            raise ValueError("candidate pose neighborhood does not intersect active limits")
        return bounded_lo, bounded_hi


def load_pose_lock(path, sha256, asset_sha256, current_layer_pins, *, tolerance_rad=.12) -> PoseLock:
    """Load a reviewed static pose and turn it into one MPC reference.

    ``load_active_pose`` performs the schema, asset, layer, finite-value and
    DH116 coupling checks.  This wrapper adds a bounded local neighborhood and
    records the exact source digest for trajectory evidence.
    """
    path = Path(path)
    if not isinstance(sha256, str) or hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError("candidate pose SHA mismatch")
    if not np.isfinite(tolerance_rad) or tolerance_rad <= 0 or tolerance_rad > 1.0:
        raise ValueError("candidate pose tolerance must be in (0,1] radians")
    states = load_active_pose(path, sha256, asset_sha256, current_layer_pins)
    # Validate the reference against the authoritative active limits before a
    # controller ever receives it.
    lower, upper = active_limits()
    reference = np.concatenate([states[side][7:] for side in ("left", "right")])
    if np.any(reference < np.tile(lower, 2)) or np.any(reference > np.tile(upper, 2)):
        raise ValueError("candidate pose active joints outside DH116 limits")
    return PoseLock(str(path), sha256, {s: q.copy() for s, q in states.items()}, float(tolerance_rad))


__all__ = ["PoseLock", "load_pose_lock"]
