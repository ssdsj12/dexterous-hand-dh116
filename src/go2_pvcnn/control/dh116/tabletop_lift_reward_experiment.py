"""Opt-in reward factors; simulator diagnostics never become Student inputs.

R0 adds no terms to the unchanged support_deficit objective. Other profiles
are explicit experiments with fixed coefficients, not an automatic curriculum.
Native success remains independent. Positive rewards use completed native
control windows. Descent cost uses a separate live physical-substep ledger,
so a failed terminal sample is charged before reset, at its actual physical dt.
"""
import math
from numbers import Real

import torch

VERSION = 'dh116-lift-quality-reward-experiment-v2'
HOLD = 'lift_quality_height_hold'
UP = 'lift_quality_upward'
DROP = 'lift_quality_drop'
FACTORS = {'R0': (), 'R1': (HOLD,), 'R2': (UP,), 'R3': (DROP,),
           'R4': (HOLD, UP), 'R5': (HOLD, UP, DROP)}
PROFILES = tuple(FACTORS)
WEIGHTS = {HOLD: 1., UP: .5, DROP: .5}
MIN_HEIGHT_M, TARGET_HEIGHT_M = .003, .03
REFERENCE_SPEED_M_S, QUALITY_SPEED_LIMIT_M_S = .005, .03
SLIP_LIMIT_M, DESCENT_DEADBAND_M = .0005, .0005


def experiment_contract(profile):
    if profile not in FACTORS:
        raise ValueError('unknown isolated lift reward experiment')
    return dict(version=VERSION, profile=profile, base_reward_profile='support_deficit',
        enabled_terms=list(FACTORS[profile]),
        weights={name: WEIGHTS[name] for name in FACTORS[profile]},
        reference='actual_measured_physical_episode_reset_bottom',
        minimum_height_m=MIN_HEIGHT_M, target_height_m=TARGET_HEIGHT_M,
        reference_speed_m_s=REFERENCE_SPEED_M_S,
        quality_speed_limit_m_s=QUALITY_SPEED_LIMIT_M_S,
        slip_limit_m=SLIP_LIMIT_M, descent_deadband_m=DESCENT_DEADBAND_M,
        quality='completed_native_window;opposed_both_hands;observable_fixed_anchor;'
                'support_at_least_runtime_weight;finite_measured_velocity;no_failure',
        quality_weight='clip(1-norm(actual_object_linear_velocity)/0.03,0,1)',
        upward='positive_height_rate_of_uncredited_qualified_high_water;no_cyclic_credit',
        descent='new_drop_beyond_0.5mm_in_contiguous_descending_segment;'
                'no_contact_requirement_for_valid_measured_drop',
        descent_validity='independent_preconsume_finite_live_physical_substeps;'
                         'terminal_sample_included;inactive_frozen;invalid_live_breaks_bridge',
        accumulation='positive_rate_times_completed_native_window_seconds;'
                     'descent_rate_times_actual_live_physics_substep_seconds;no_window_double_count',
        success='unchanged_30mm_5s_0.5mm',
        actor_feedback='none;3844D_actor_371D_privileged_critic_unchanged')


class LiftRewardExperiment:
    def __init__(self, profile, reference_bottom):
        if (profile not in FACTORS or not isinstance(reference_bottom, torch.Tensor)
                or reference_bottom.ndim != 1 or len(reference_bottom) < 1
                or reference_bottom.dtype not in (torch.float32, torch.float64)
                or not bool(torch.isfinite(reference_bottom).all())):
            raise ValueError('profile and finite floating measured reset bottom[N] required')
        self.profile = profile
        self.reference = reference_bottom.detach().clone()
        self._prev_h = torch.zeros_like(self.reference)
        self._prev_valid = torch.ones_like(self.reference, dtype=torch.bool)
        self._qualified_peak = torch.zeros_like(self.reference)
        self._descent_anchor = torch.zeros_like(self.reference)
        self._pending_reset_bottom = torch.zeros_like(self.reference)
        self._pending_reset = torch.zeros_like(self.reference, dtype=torch.bool)
        self._physical_window_valid = torch.ones_like(self.reference, dtype=torch.bool)
        self._drop_prev_h = torch.zeros_like(self.reference)
        self._drop_prev_valid = torch.ones_like(self.reference, dtype=torch.bool)
        self._drop_descent_anchor = torch.zeros_like(self.reference)

    def _reset_values(self, ids, bottom):
        ids = torch.as_tensor(ids, device=self.reference.device)
        if ids.dtype == torch.bool:
            if ids.shape != self.reference.shape:
                raise ValueError('boolean reset mask must match environment count')
            ids = ids.nonzero(as_tuple=False).flatten()
        elif ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('reset requires integer indices or boolean mask')
        if bool(((ids < 0) | (ids >= len(self.reference))).any()):
            raise ValueError('reset index out of range')
        if (not isinstance(bottom, torch.Tensor) or bottom.ndim != 1
                or bottom.dtype != self.reference.dtype or bottom.device != self.reference.device):
            raise ValueError('measured reset bottom must match dtype/device')
        value = bottom[ids] if bottom.shape == self.reference.shape else bottom
        if value.shape != ids.shape or not bool(torch.isfinite(value).all()):
            raise ValueError('finite measured reset bottom for each selected environment required')
        return ids, value

    @torch.no_grad()
    def capture_physical_reset(self, ids, bottom):
        """Own the actual reset measurement before inactive tail physics changes it."""
        ids, value = self._reset_values(ids, bottom)
        self._pending_reset_bottom[ids] = value
        self._pending_reset[ids] = True

    @torch.no_grad()
    def reset(self, ids, bottom):
        ids, value = self._reset_values(ids, bottom)
        value = torch.where(self._pending_reset[ids], self._pending_reset_bottom[ids], value)
        self.reference[ids] = value
        self._prev_h[ids] = self._qualified_peak[ids] = self._descent_anchor[ids] = 0.
        self._prev_valid[ids] = True
        self._physical_window_valid[ids] = True
        self._drop_prev_h[ids] = self._drop_descent_anchor[ids] = 0.
        self._drop_prev_valid[ids] = True
        self._pending_reset[ids] = False
        self._pending_reset_bottom[ids] = 0.

    def _check(self, value, tail, name, boolean=False):
        dtype = torch.bool if boolean else self.reference.dtype
        if (not isinstance(value, torch.Tensor) or value.shape != (len(self.reference), *tail)
                or value.dtype != dtype or value.device != self.reference.device):
            raise ValueError('matching shape/dtype/device required: '+name)

    @torch.no_grad()
    def drop_physics_substep(self, *, bottom, object_velocity, live, measurement_valid, dt):
        """Charge each finite live descent once, including the terminal sample.

        Inactive reset-tail samples have no authority over the last live state.
        An invalid live sample clears continuity without bridging its distance.
        This ledger never advances positive/control-window observer state.
        """
        if (isinstance(dt, bool) or not isinstance(dt, Real)
                or not math.isfinite(dt) or dt <= 0):
            raise ValueError('finite positive physical substep dt required')
        self._check(bottom, (), 'bottom')
        self._check(object_velocity, (6,), 'object_velocity')
        self._check(live, (), 'live', True)
        self._check(measurement_valid, (), 'measurement_valid', True)
        if DROP not in FACTORS[self.profile]:
            return torch.zeros_like(bottom)
        physical = (live & measurement_valid & torch.isfinite(bottom)
                    & torch.isfinite(object_velocity).all(-1))
        h = torch.where(physical, bottom-self.reference, 0.)
        rising = (h >= self._drop_prev_h) | ~self._drop_prev_valid
        anchor = torch.where(rising, h, torch.maximum(self._drop_descent_anchor, self._drop_prev_h))
        before = (anchor-self._drop_prev_h-DESCENT_DEADBAND_M).clamp_min(0.)
        after = (anchor-h-DESCENT_DEADBAND_M).clamp_min(0.)
        distance = (after-before).clamp_min(0.)
        rate = (distance/(REFERENCE_SPEED_M_S*dt)).clamp(0., 1.)
        cost = torch.where(physical & self._drop_prev_valid, -WEIGHTS[DROP]*rate*dt, 0.)
        self._drop_prev_h.copy_(torch.where(live, h, self._drop_prev_h))
        self._drop_prev_valid.copy_(torch.where(live, physical, self._drop_prev_valid))
        self._drop_descent_anchor.copy_(torch.where(live,
            torch.where(physical, anchor, 0.), self._drop_descent_anchor))
        return cost

    @torch.no_grad()
    def update(self, *, bottom, opposed_contact, support_n, weight_n, drift_m,
               slip_valid, object_velocity, measurement_valid, failed, window_good,
               dt, window_completed):
        if (isinstance(dt, bool) or not isinstance(dt, Real)
                or not math.isfinite(dt) or dt <= 0 or type(window_completed) is not bool):
            raise ValueError('finite positive native window dt and boolean completion required')
        for name, value in (('bottom', bottom), ('support', support_n),
                            ('weight', weight_n), ('drift', drift_m)):
            self._check(value, (), name)
        self._check(object_velocity, (6,), 'object_velocity')
        self._check(opposed_contact, (2,), 'opposed_contact', True)
        for name, value in (('slip_valid', slip_valid), ('measurement_valid', measurement_valid),
                            ('failed', failed), ('window_good', window_good)):
            self._check(value, (), name, True)
        result = {name: torch.zeros_like(bottom) for name in FACTORS[self.profile]}
        if not window_completed or not result:
            return result
        physical = (measurement_valid & torch.isfinite(bottom)
                    & torch.isfinite(object_velocity).all(-1))
        h = torch.where(physical, bottom - self.reference, 0.)
        speed = object_velocity[:, :3].norm(dim=-1)
        eligible = (physical & ~failed & window_good & opposed_contact.all(-1)
            & slip_valid & torch.isfinite(drift_m) & (drift_m >= 0.) & (drift_m <= SLIP_LIMIT_M)
            & torch.isfinite(support_n) & torch.isfinite(weight_n)
            & (weight_n > 0.) & (support_n >= weight_n) & (h >= MIN_HEIGHT_M)
            & (speed < QUALITY_SPEED_LIMIT_M_S))
        quality = torch.where(eligible, (1. - speed/QUALITY_SPEED_LIMIT_M_S).clamp(0., 1.), 0.)
        if HOLD in result:
            height = ((h-MIN_HEIGHT_M)/(TARGET_HEIGHT_M-MIN_HEIGHT_M)).clamp(0., 1.)
            result[HOLD] = WEIGHTS[HOLD]*height*quality*dt
        if UP in result:
            distance = torch.minimum(h-self._prev_h, h-self._qualified_peak).clamp_min(0.)
            rate = (distance/(REFERENCE_SPEED_M_S*dt)).clamp(0., 1.)
            result[UP] = torch.where(self._prev_valid, WEIGHTS[UP]*rate*quality*dt, 0.)
        rising = (h >= self._prev_h) | ~self._prev_valid
        anchor = torch.where(rising, h, torch.maximum(self._descent_anchor, self._prev_h))
        if DROP in result:
            before = (anchor-self._prev_h-DESCENT_DEADBAND_M).clamp_min(0.)
            after = (anchor-h-DESCENT_DEADBAND_M).clamp_min(0.)
            distance = (after-before).clamp_min(0.)
            rate = (distance/(REFERENCE_SPEED_M_S*dt)).clamp(0., 1.)
            result[DROP] = torch.where(physical & self._prev_valid, -WEIGHTS[DROP]*rate*dt, 0.)
        self._qualified_peak.copy_(torch.where(quality > 0., torch.maximum(self._qualified_peak, h),
                                                self._qualified_peak))
        self._descent_anchor.copy_(torch.where(physical, anchor, 0.))
        self._prev_h.copy_(h)
        self._prev_valid.copy_(physical)
        return result


def install_lift_reward_experiment(task, profile):
    """Opt in after the original R0 observer exists; never add Actor fields."""
    if profile not in FACTORS:
        raise ValueError('unknown isolated lift reward experiment')
    if profile == 'R0':
        return None
    if (getattr(task, 'surface_reward_profile', None) != 'support_deficit'
            or not hasattr(task, 'surface_reward_ablation')
            or not hasattr(task, 'surface_early_hold')):
        raise ValueError('lift factors require the unchanged native support_deficit base')
    observer = LiftRewardExperiment(profile, task.m['bottom'])
    task.surface_reward_ablation.lift_quality_experiment = observer
    for name in FACTORS[profile]:
        task.terms[name] = torch.zeros_like(task.reward)
    return observer


def add_lift_experiment_reward(task, pieces, observer, tracker, valid, support, weight, *,
                               physical_dt, physical_live):
    """Use completed windows for positives and actual live substeps for descent."""
    if tracker is None:
        raise ValueError('lift experiment requires the native early-window observer')
    m = task.m
    drop = observer.drop_physics_substep(bottom=m['bottom'],
        object_velocity=m['object_velocity'], live=physical_live,
        measurement_valid=valid, dt=physical_dt)
    # Hold's native-valid mask uses task.active, which excludes real terminal
    # failures. Keep measurement continuity independent of success eligibility,
    # including finite/live samples before the final control-window boundary.
    observer._physical_window_valid &= (valid & torch.isfinite(m['bottom'])
        & torch.isfinite(m['object_velocity']).all(-1))
    if not tracker.window_completed:
        pieces.update({name: torch.zeros_like(task.reward) for name in FACTORS[observer.profile]})
        if DROP in pieces: pieces[DROP] = drop
        return
    physical_window_valid = observer._physical_window_valid.clone()
    observer._physical_window_valid.fill_(True)
    from .tabletop_surface_force_sharing import surface_targets
    paired = surface_targets(m['effective_pad_loads_n'], m['effective_pad_vectors_w'],
                             valid=m['effective_pad_valid'])['paired']
    pieces.update(observer.update(bottom=m['bottom'], opposed_contact=paired,
        support_n=support, weight_n=weight, drift_m=tracker.reward_drift_m,
        slip_valid=tracker.slip_observable, object_velocity=m['object_velocity'],
        measurement_valid=physical_window_valid,
        failed=task.failure != 0, window_good=tracker.window_good,
        dt=tracker.period, window_completed=True))
    # The physical ledger already charged this window's descent incrementally.
    if DROP in pieces: pieces[DROP] = drop
