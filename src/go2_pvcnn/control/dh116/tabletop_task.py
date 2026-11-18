"""Measured tabletop phase gates and time-scaled MPC residual rewards."""
from dataclasses import dataclass

import numpy as np


REWARD_PARAMETERS = dict(approach=2.0, mpc_tracking=1.0, contact=2.0,
                         lift=5.0, slip=-1.0, smoothness=-0.05,
                         force=-0.1, time=-0.1, success=25.0, failure=-25.0)


def approach_joint_reference(grasp, offset):
    """Smooth open-to-grasp reference over a 40 mm approach, in radians."""
    from .linear_coupling import active_limits, expand
    grasp = np.asarray(grasp, dtype=float)
    expand(grasp)
    if not np.isfinite(offset):
        raise ValueError('finite approach offset required')
    low, _ = active_limits()
    opened = low + .05*(grasp-low)
    progress = float(np.clip(1.-offset/.04, 0., 1.))
    blend = progress*progress*(3.-2.*progress)
    return opened + blend*(grasp-opened)


def guarded_descent(offset, dt, clearance_m, pad_forces_n):
    """Limit translation using measured clearance; contact stops descent.

    This is a conservative supervisor, not a contact solver or grasp gate.
    The caller latches contact stops to avoid restarting on sensor chatter.
    """
    forces = np.asarray(pad_forces_n, dtype=float)
    if (forces.shape != (4,) or not np.isfinite(forces).all()
            or not np.isfinite([offset, dt, clearance_m]).all() or dt <= 0):
        raise ValueError('finite four-pad measurements and positive dt required')
    if np.any(forces >= .1):
        return offset
    step = min(.02*dt, max(0., clearance_m-.002), max(0., offset))
    return offset-step


def advance_root_offset(offset, phase, dt):
    """Advance the hand-only root reference at the physics rate (meters)."""
    if not np.isfinite([offset, dt]).all() or dt <= 0:
        raise ValueError('finite offset and positive dt required')
    if phase == 'approach':
        return max(0., offset-.02*dt)
    if phase == 'lift':
        return min(.05, offset+.01*dt)
    return offset


def joint_safety_failure(positions, limits, mimic_errors):
    """Simulation guards in radians, not hardware qualification thresholds."""
    q, bounds, errors = [np.asarray(v, dtype=float) for v in (positions, limits, mimic_errors)]
    if bounds.shape != (q.size, 2) or q.ndim != 1 or errors.ndim != 1:
        raise ValueError('expected joint vector, Nx2 limits and mimic error vector')
    if not all(np.isfinite(v).all() for v in (q, bounds, errors)):
        return 'nonfinite_joint_state'
    if np.any(q < bounds[:, 0] - .02) or np.any(q > bounds[:, 1] + .02):
        return 'joint_limit'
    if np.any(np.abs(errors) > .02):
        return 'mimic_error'
    return None


def baseline_allows_training(baseline):
    """Minimum gate only; repeatability qualification is still required."""
    return baseline.get('is_success') is True and baseline.get('failure') is None


@dataclass
class TabletopTask:
    phase: str = 'approach'
    elapsed: float = 0.0
    contact_time: float = 0.0
    success_time: float = 0.0
    success: bool = False
    failure: str | None = None
    grasp_mode: str = 'four_pad'
    force_feedback: bool = False

    def update(self, dt, *, approach_error_m, pad_forces_n, clearance_m,
               object_bottom_m, object_speed_m_s, slip_m, max_force_n,
               approach_contact_ready=False, approach_blocked=False, approach_pregrasp_ready=False, grasp_ready=False):
        if self.success or self.failure:
            return False
        forces = np.asarray(pad_forces_n, dtype=float)
        if forces.shape != (4,):
            self.failure = 'nonfinite_state'
            return False
        values = np.r_[dt, approach_error_m, forces, clearance_m,
                       object_bottom_m, object_speed_m_s, slip_m, max_force_n]
        if not np.isfinite(values).all() or dt <= 0:
            self.failure = 'nonfinite_state'
            return False
        self.elapsed += dt
        contact = bool(np.all(forces >= 0.1))
        if clearance_m < -0.001:
            self.failure = 'table_penetration'
        elif max_force_n > 30:
            self.failure = 'excess_force'
        elif object_bottom_m < 0.29:
            self.failure = 'object_dropped'
        elif self.elapsed > 15:
            self.failure = 'timeout'
        if self.failure:
            return False
        if self.grasp_mode == 'object_stability' and self.phase != 'approach':
            if self.force_feedback and self.phase in ('close', 'hold'):
                if self.elapsed > 6.:
                    self.failure = 'grasp_not_established'
                    return False
                if not grasp_ready:
                    self.phase, self.contact_time = 'close', 0.
                else:
                    self.contact_time += dt
                    dwell = .2 if self.phase == 'close' else .3
                    if self.contact_time >= dwell:
                        self.phase = 'hold' if self.phase == 'close' else 'lift'
                        self.contact_time = 0.
                return False
            # Training attempts a lift without a prescribed contact topology.
            # A timed phase transition is not a grasp qualification.
            if self.phase in ('close', 'hold'):
                self.contact_time += dt
                if self.contact_time >= (2. if self.phase == 'close' else .3):
                    self.phase = 'hold' if self.phase == 'close' else 'lift'
                    self.contact_time = 0.
            elif self.phase == 'lift':
                ready = object_bottom_m >= .33 and object_speed_m_s < .03 and slip_m <= .02
                self.success_time = self.success_time + dt if ready else 0.
                if self.success_time >= .5:
                    self.success = True
                    return True
            return False
        if self.phase == 'approach':
            if approach_blocked:
                self.failure = 'approach_blocked'
                return False
            measured_stop = (approach_contact_ready and clearance_m >= 0.002
                             and bool(np.any(forces >= 0.1)))
            prepared_stop = (approach_pregrasp_ready and approach_error_m <= .005
                             and clearance_m >= .002)
            if approach_error_m <= 0.002 or measured_stop or prepared_stop:
                self.phase = 'close'
        elif self.phase == 'close':
            self.contact_time = self.contact_time + dt if contact else 0.0
            if self.contact_time >= 0.2:
                self.phase = 'hold'
                self.contact_time = 0.0
        elif self.phase == 'hold':
            if not contact:
                self.phase, self.contact_time = 'close', 0.0
            else:
                self.contact_time += dt
                if self.contact_time >= 0.5 and slip_m <= 0.005:
                    self.phase = 'lift'
        elif self.phase == 'lift':
            if not contact or slip_m > 0.01:
                self.failure = 'contact_lost'
                return False
            ready = object_bottom_m >= 0.33 and object_speed_m_s < 0.03
            self.success_time = self.success_time + dt if ready else 0.0
            if self.success_time >= 0.5:
                self.success = True
                return True
        return False


def task_reward(*, phase, action, reference, previous, dt, success_event, failed,
                approach_error_m, pad_forces_n, clearance_m, object_bottom_m,
                object_speed_m_s, slip_m, max_force_n):
    """Dense terms are per second; terminal bonuses are emitted once by caller."""
    action, reference, previous = [np.asarray(x, dtype=float) for x in (action, reference, previous)]
    values = np.r_[action, reference, previous, dt, approach_error_m, pad_forces_n,
                   clearance_m, object_bottom_m, object_speed_m_s, slip_m, max_force_n]
    if any(x.shape != (12,) for x in (action, reference, previous)) or not np.isfinite(values).all() or dt <= 0:
        raise ValueError('finite 12D actions and positive dt required')
    contact = np.asarray(pad_forces_n)
    grasped = contact.shape == (4,) and bool(np.all(contact >= 0.1))
    raw = dict(approach=np.exp(-approach_error_m / 0.025),
               mpc_tracking=np.exp(-np.mean(((action-reference)/0.15)**2)),
               contact=float(np.mean(np.clip(contact / 1.0, 0, 1))) if phase != 'approach' else 0.0,
               lift=float(np.clip((object_bottom_m-0.30)/0.03, 0, 1)) if phase == 'lift' and grasped else 0.0,
               slip=min((slip_m/0.005)**2, 25),
               smoothness=min(float(np.mean(((action-previous)/0.1)**2)), 25),
               force=min(max(0, max_force_n-15)**2 / 225, 25), time=1.0)
    terms = {name: float(value * REWARD_PARAMETERS[name] * dt) for name, value in raw.items()}
    terms['success'] = REWARD_PARAMETERS['success'] if success_event and not failed else 0.0
    terms['failure'] = REWARD_PARAMETERS['failure'] if failed else 0.0
    return sum(terms.values()), terms
