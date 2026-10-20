"""Simulation speed candidates, not manufacturer limits or grasp evidence."""
from dataclasses import dataclass, astuple
import numpy as np


@dataclass(frozen=True)
class GraspSpeedProfile:
    far_m_s: float = .02
    near_m_s: float = .001
    loaded_m_s: float = .001
    acceleration_m_s2: float = .05
    slowdown_gap_m: float = .002
    lift_m_s: float = .01

    def __post_init__(self):
        if (not np.isfinite(astuple(self)).all()
                or not 0 < self.near_m_s <= self.far_m_s <= .05
                or not 0 < self.loaded_m_s <= .002
                or not 0 < self.acceleration_m_s2 <= 1.
                or not 0 < self.slowdown_gap_m <= .02
                or not 0 < self.lift_m_s <= .02):
            raise ValueError('invalid simulation grasp speed profile')


class ContactSpeedGovernor:
    def __init__(self, profile):
        if not isinstance(profile, GraspSpeedProfile):
            raise ValueError('explicit simulation speed profile required')
        self.profile = profile
        self.last_time = None
        self.last_speed = np.zeros((2, 2))

    def update(self, time_s, *, force_speeds, loads, valid, gaps, active):
        force, load, gap = (np.asarray(a, dtype=float) for a in (force_speeds, loads, gaps))
        valid, active = np.asarray(valid), np.asarray(active)
        if (any(a.shape != (2, 2) for a in (force, load, gap, valid, active))
                or any(not np.isfinite(a).all() for a in (force, load, gap))
                or np.any(load < 0) or valid.dtype.kind != 'b' or active.dtype.kind != 'b'
                or not np.isfinite(time_s) or time_s < 0
                or (self.last_time is not None and not 0 < time_s - self.last_time <= .01 + 1e-12)):
            raise ValueError('invalid measured speed governor inputs')
        dt = 0. if self.last_time is None else float(time_s - self.last_time)
        p = self.profile
        envelope = np.minimum(p.far_m_s, np.sqrt(p.near_m_s**2 +
            2 * p.acceleration_m_s2 * np.maximum(gap - p.slowdown_gap_m, 0)))
        speed = np.minimum(envelope, self.last_speed + p.acceleration_m_s2 * dt)
        speed = np.maximum(speed, 0.)
        feedback = valid | (load > 0) | (gap <= 0) | (force <= 0)
        # Contact, retreat and channel stop take priority over a smooth ramp.
        speed = np.where(feedback, np.clip(force, -p.loaded_m_s, p.loaded_m_s), speed)
        speed = np.where(active, speed, 0.)
        self.last_time = float(time_s)
        self.last_speed = speed.copy()
        return speed
