"""Measured touch-first sequencing; simulation targets, no attachment authority.

Rows are left/right hands; columns thumb/index/middle. Outputs are desired pad-normal
velocities, not joint commands or evidence that grasping has succeeded.
"""
import numpy as np


def bounded_handoff(command, measured, dt):
    """Rate-limited transition to a captured actual active pose, not nominal end."""
    q=np.asarray(command,dtype=float); actual=np.asarray(measured,dtype=float)
    if (q.shape!=(2,13) or actual.shape!=(2,13)
            or not np.isfinite(q).all() or not np.isfinite(actual).all()
            or not np.isfinite(dt) or not 0<dt<=1e-4):
        raise ValueError('invalid measured handoff or timestep')
    cap=np.r_[np.full(7,.05),np.full(6,.02)]*dt
    remaining=actual-q
    ready=bool(np.all(np.abs(remaining)<=cap))
    return (actual.copy() if ready else q+np.clip(remaining,-cap,cap)),ready


def _time_delta(time_s, previous):
    if not np.isfinite(time_s) or time_s < 0:
        raise ValueError('invalid physical evidence time')
    if previous is not None and not 0 < time_s - previous <= .01 + 1e-12:
        raise ValueError('physical evidence must advance continuously')
    return 0. if previous is None else float(time_s - previous)


class ApproachReadiness:
    """Call on approach entry, including after restore; no imported dwell."""

    def __init__(self):
        self.started = None
        self.last_time = None
        self.ready_since = None

    def update(self, time_s, *, position_errors_m, normal_errors_rad, joint_speeds_rad_s):
        position = np.asarray(position_errors_m, dtype=float)
        angles = np.asarray(normal_errors_rad, dtype=float)
        speeds = np.asarray(joint_speeds_rad_s, dtype=float)
        _time_delta(time_s, self.last_time)
        if (position.shape not in ((4,), (6,)) or angles.shape != position.shape or speeds.shape != (2, 18)
                or any(not np.isfinite(a).all() for a in (position, angles, speeds))
                or np.any(position < 0) or np.any(angles < 0)):
            raise ValueError('finite actual position, normal and joint speeds required')
        if self.started is None:
            self.started = float(time_s)
        self.last_time = float(time_s)
        qualified = (np.all(position <= .002) and np.all(angles <= np.deg2rad(5.))
                     and np.all(np.abs(speeds) <= .02))
        if not qualified:
            self.ready_since = None
        elif self.ready_since is None:
            self.ready_since = float(time_s)
        ready = self.ready_since is not None and time_s - self.ready_since >= .05 - 1e-12
        if not ready and time_s - self.started >= 10. - 1e-12:
            raise TimeoutError('approach did not reach measured low-speed readiness within 10 s')
        return bool(ready)


class TouchFirst:
    """Each hand touches, opposes, then ramps preload independently.

Caller supplies valid contacts from actual load/region/separation checks.
ready_for_hold only means ramp complete; the separate grasp evidence gate
must still qualify actual six-channel preload, duration, slip and lift.
"""

    def __init__(self, target_loads_n, *, speed_profile=None):
        target = np.asarray(target_loads_n, dtype=float)
        if (target.shape not in ((2, 2), (2, 3)) or not np.isfinite(target).all()
                or np.any(target < .05) or np.any(target > 3.)):
            raise ValueError('touch-first simulation targets must be (2,2) or (2,3) in [.05,3] N')
        self.channels = target.shape[1]
        self.final_targets = target.copy()
        self.targets = np.full(target.shape, .05)
        self.phases = ['touch', 'touch']
        self.since = [None, None]
        self.last_time = None
        from .speed_profile import ContactSpeedGovernor
        self.speed_governor = None if speed_profile is None else ContactSpeedGovernor(speed_profile)

    def update(self, time_s, loads_n, valid_contacts, *, surface_gaps_m=None,
               controller_target_loads_n=None):
        loads = np.asarray(loads_n, dtype=float)
        valid = np.asarray(valid_contacts)
        dt = _time_delta(time_s, self.last_time)
        if (loads.shape != (2, self.channels) or valid.shape != (2, self.channels) or valid.dtype.kind != 'b'
                or not np.isfinite(loads).all() or np.any(loads < 0)):
            raise ValueError('finite compressive loads and boolean measured contacts required')
        controller_targets=(None if controller_target_loads_n is None else
                            np.asarray(controller_target_loads_n,dtype=float))
        if controller_targets is not None and (controller_targets.shape!=(2,self.channels)
                or not np.isfinite(controller_targets).all() or np.any(controller_targets<.05)
                or np.any(controller_targets>3.)):
            raise ValueError('finite external controller load targets in [.05,3] N required')
        if self.speed_governor is not None:
            gaps = np.asarray(surface_gaps_m, dtype=float)
            if gaps.shape != (2, self.channels) or not np.isfinite(gaps).all():
                raise ValueError('finite measured surface gaps required for staged speed')
        self.last_time = float(time_s)
        for hand in range(2):
            if controller_targets is not None:self.targets[hand]=controller_targets[hand]
            thumb = bool(valid[hand, 0] and loads[hand, 0] >= .025)
            pinch = thumb and bool(valid[hand, 1] and loads[hand, 1] >= .025)
            support = self.channels == 3 and bool(valid[hand, 2] and loads[hand, 2] >= .025)
            phase = self.phases[hand]
            if phase != 'touch' and not thumb:
                self.phases[hand] = 'touch'
                self.since[hand] = None
            elif phase == 'preload' and not pinch:
                self.phases[hand] = 'oppose'
                self.since[hand] = None
            phase = self.phases[hand]
            if phase in ('touch', 'oppose'):
                if controller_targets is None:self.targets[hand] = .05
                qualified = thumb if phase == 'touch' else pinch
                if not qualified:
                    self.since[hand] = None
                elif self.since[hand] is None:
                    self.since[hand] = float(time_s)
                elif time_s - self.since[hand] >= .05 - 1e-12:
                    self.phases[hand] = 'oppose' if phase == 'touch' else 'preload'
                    self.since[hand] = float(time_s) if phase == 'touch' and pinch else None
            else:
                if controller_targets is None:
                    self.targets[hand] = np.minimum(self.final_targets[hand], self.targets[hand] + dt)
        error = self.targets - loads
        speed = np.clip(.01 * error, -.002, .002)
        speed[np.abs(error) <= .005] = 0.
        for hand, phase in enumerate(self.phases):
            if phase == 'touch':
                speed[hand, 1] = 0.
                if self.channels == 3: speed[hand, 2] = 0.
            elif phase == 'oppose':
                # The middle finger remains parked until the thumb/index
                # opposition is measured; then it is allowed to hook under
                # the wire during preload.
                if self.channels == 3: speed[hand, 2] = 0.
        if self.speed_governor is not None:
            active = np.ones((2, self.channels), dtype=bool)
            for hand, phase in enumerate(self.phases):
                if phase == 'touch':
                    active[hand, 1] = False
                    if self.channels == 3: active[hand, 2] = False
                elif phase == 'oppose':
                    if self.channels == 3: active[hand, 2] = False
            speed = self.speed_governor.update(time_s, force_speeds=speed, loads=loads,
                valid=valid, gaps=gaps, active=active)
        return dict(phases=self.phases[:], target_loads_n=self.targets.copy(),
                    preload_target_source='touch_first_ramp' if controller_targets is None else 'controller',
                    normal_speeds_m_s=speed,
                    ready_for_hold=bool(self.phases == ['preload', 'preload']
                                        and np.all(self.targets >= self.final_targets)))
