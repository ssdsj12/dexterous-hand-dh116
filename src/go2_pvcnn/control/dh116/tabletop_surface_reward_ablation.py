"""Isolated, reward-only objectives for the native DH116 Student."""
import math

import torch

LIFT = 'surface_lift_progress'
HOLD = 'surface_early_hold_rate'
MILESTONE = 'surface_hold_milestone'
BREAK = 'surface_contact_break'
SLIP = 'surface_slip_window'
MOTION = 'surface_relative_motion'
SUPPORT = 'surface_support_deficit'
SMOOTH = 'surface_executed_smoothness'

SINGLE_TERMS = dict(lift=LIFT, early_hold=HOLD, milestone=MILESTONE, contact_break=BREAK,
                    slip=SLIP, relative_motion=MOTION, support=SUPPORT, smoothness=SMOOTH)
CUMULATIVE_TERMS = {
    'lift_progress': (LIFT,),
    'early_hold': (LIFT, HOLD, MILESTONE),
    'contact_break': (LIFT, HOLD, MILESTONE, BREAK),
    'slip_relative': (LIFT, HOLD, MILESTONE, BREAK, SLIP, MOTION),
    'support_deficit': (LIFT, HOLD, MILESTONE, BREAK, SLIP, MOTION, SUPPORT),
}
STAGE_TERMS = {'baseline': (), 'v3_reference': (),
               **{f'v3_{factor}_only': (term,) for factor, term in SINGLE_TERMS.items()},
               **CUMULATIVE_TERMS}
PROFILES = tuple(STAGE_TERMS)
MILESTONE_SECONDS = (.1, .3, .5, 1., 2., 5.)
MILESTONE_POINTS = (1., 2., 2., 3., 4., 5.)
CONTROL_PERIOD_S = 1 / 60
COEFFICIENTS = {LIFT: 8., HOLD: 2., MILESTONE: MILESTONE_POINTS, BREAK: -4.,
                SLIP: -24., MOTION: -2., SUPPORT: -1.5, SMOOTH: -1.}


def disabled_legacy_terms(profile):
    terms = STAGE_TERMS[profile]
    return tuple(old for term, old in ((LIFT, 'dex_lift'), (LIFT, 'lift_milestone'), (SLIP, SLIP))
                 if term in terms)


def ablation_contract(profile):
    if profile not in STAGE_TERMS:
        raise ValueError('unknown staged reward profile')
    terms = STAGE_TERMS[profile]
    kind = ('legacy' if profile == 'baseline' else 'reference' if profile == 'v3_reference'
            else 'cumulative' if profile in CUMULATIVE_TERMS else 'one_factor')
    return dict(version='surface_stable_hold_reward_v4', profile=profile, kind=kind,
        reference_profile='v3_reference', enabled_terms=list(terms),
        disabled_legacy_terms=list(disabled_legacy_terms(profile)),
        weights={name: list(COEFFICIENTS[name]) if name == MILESTONE else COEFFICIENTS[name] for name in terms},
        height='episode_bottom_reference;0.5mm_deadband;30mm_scale;qualified_high_water',
        early_hold='qualified_3mm_window;10_points_per_episode',
        milestones=dict(seconds=list(MILESTONE_SECONDS), points=list(MILESTONE_POINTS)),
        slip='fresh_60Hz_native_fixed_anchor_drift;3mm_airborne;500um_scale;quadratic_cap4',
        contact_break='two_bad_native_60Hz_windows;100ms_rearm;max2_per_episode',
        relative_motion='native_pad_point_tangent_speed;2mm_per_s_scale;quadratic_cap4',
        support_deficit='signed_attributable_native_Fz/runtime_weight;3mm_airborne;clip_deficit_0_1',
        smoothness='measured_increment_change+0.1*squared_applied_command_residual;rate_normalized;component_cap1',
        success='unchanged_30mm_5s_0.5mm', teacher_servo=False)


class RewardAblation:
    def __init__(self, profile, bottom):
        if profile not in STAGE_TERMS:
            raise ValueError('unknown staged reward profile')
        self.terms = STAGE_TERMS[profile]
        self.reference = bottom.clone()
        self.best_ratio = torch.zeros_like(bottom)
        self.hold_paid = torch.zeros_like(bottom)
        self.milestone_paid = torch.zeros(len(bottom), len(MILESTONE_SECONDS), dtype=torch.bool, device=bottom.device)
        self.armed = torch.zeros_like(bottom, dtype=torch.bool)
        self.invalid_windows = torch.zeros_like(bottom, dtype=torch.long)
        self.break_events = torch.zeros_like(self.invalid_windows)

    def reset(self, ids, bottom):
        self.reference[ids] = bottom[ids]
        self.best_ratio[ids] = 0.
        self.hold_paid[ids] = 0.
        self.milestone_paid[ids] = False
        self.armed[ids] = False
        self.invalid_windows[ids] = 0
        self.break_events[ids] = 0

    @torch.no_grad()
    def update(self, *, bottom, qualified, early_window_good, early_duration, drift,
               relative_speed, support, weight, valid, dt, window_completed,
               window_native_valid, slip_observable, smoothness, milestone_valid=None):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('positive physical dt required')
        valid = valid & torch.isfinite(bottom) & torch.isfinite(self.reference)
        qualified = qualified & valid
        lift = bottom - self.reference
        airborne = lift >= .003
        early = valid & window_completed & window_native_valid & early_window_good
        result = {}
        if LIFT in self.terms:
            ratio = ((lift - .0005) / .03).clamp(0, 1)
            best = torch.where(qualified, torch.maximum(self.best_ratio, ratio), self.best_ratio)
            result[LIFT] = COEFFICIENTS[LIFT] * (best - self.best_ratio)
            self.best_ratio.copy_(best)
        if HOLD in self.terms:
            credit = torch.where(early, (10. - self.hold_paid).clamp(0., COEFFICIENTS[HOLD] * CONTROL_PERIOD_S), 0.)
            self.hold_paid.add_(credit)
            result[HOLD] = credit
        if MILESTONE in self.terms:
            levels = bottom.new_tensor(MILESTONE_SECONDS)
            reached = (early_duration[:, None] >= levels - 1e-12) & torch.isfinite(early_duration[:, None])
            milestone_window = early
            if milestone_valid is not None:
                milestone_window = (milestone_valid & torch.isfinite(bottom) & torch.isfinite(self.reference)
                                    & window_completed & window_native_valid & early_window_good)
            pulse = reached & ~self.milestone_paid & milestone_window[:, None]
            result[MILESTONE] = (pulse * bottom.new_tensor(MILESTONE_POINTS)).sum(-1)
            self.milestone_paid |= pulse
        if BREAK in self.terms:
            result[BREAK] = torch.zeros_like(bottom)
            if window_completed:
                native = valid & window_native_valid
                self.invalid_windows = torch.where(native & ~early & self.armed, self.invalid_windows + 1, 0)
                broke = self.armed & (self.invalid_windows >= 2)
                paid = broke & (self.break_events < 2)
                result[BREAK] = torch.where(paid, COEFFICIENTS[BREAK], 0.).to(bottom)
                self.break_events.add_(paid.long())
                self.armed = (self.armed & ~broke) | (native & early & (early_duration >= .1 - 1e-12))
        if SLIP in self.terms:
            observable = valid & airborne & window_completed & window_native_valid & slip_observable
            result[SLIP] = torch.where(observable & torch.isfinite(drift) & (drift >= 0.),
                COEFFICIENTS[SLIP] * (drift / .0005).square().clamp_max(4.) * CONTROL_PERIOD_S, 0.)
        if MOTION in self.terms:
            result[MOTION] = torch.where(qualified & airborne & torch.isfinite(relative_speed),
                COEFFICIENTS[MOTION] * (relative_speed / .002).square().clamp_max(4.) * dt, 0.)
        if SUPPORT in self.terms:
            measured = torch.isfinite(weight) & (weight > 0.) & torch.isfinite(support)
            deficit = ((weight - support) / torch.where(measured, weight, 1.)).clamp(0., 1.)
            result[SUPPORT] = torch.where(qualified & airborne & measured, COEFFICIENTS[SUPPORT] * deficit * dt, 0.)
        if SMOOTH in self.terms:
            result[SMOOTH] = torch.where(valid & torch.isfinite(smoothness),
                                         COEFFICIENTS[SMOOTH] * smoothness * dt, 0.)
        return result


def enable(task):
    from .tabletop_surface_success import SurfaceLiftSuccess
    terms = STAGE_TERMS[task.surface_reward_profile]
    task.surface_reward_ablation = RewardAblation(task.surface_reward_profile, task.m['bottom'])
    if any(name in terms for name in (HOLD, MILESTONE, BREAK, SLIP)):
        task.surface_early_hold = SurfaceLiftSuccess(task.n, task.device, task.m['bottom'],
            dtype=task.command.dtype, minimum_lift_m=.003, reward_diagnostics=True)
    if SMOOTH in terms:
        task.surface_action_motion = ExecutedMotion(task.actions)
    if MOTION in terms:
        from .tabletop_effective_pad import DISTAL_LINKS
        task.surface_pad_body_indices = [
            [hand.body_names.index(name) for name in DISTAL_LINKS] for hand in task.surface_world.hands]
    for name in terms:
        task.terms[name] = torch.zeros_like(task.reward)


class ExecutedMotion:
    """Read actual movement and compare projected and requested 24D rates."""
    def __init__(self, actions):
        from .tabletop_surface_contract import ACTION_CONTRACT
        self.rate = actions.new_tensor([ACTION_CONTRACT['joint_rate_rad_s']] * 12
            + [ACTION_CONTRACT['translation_rate_m_s']] * 6 + [ACTION_CONTRACT['rotation_rate_rad_s']] * 6)
        self.applied = torch.zeros_like(actions)
        self.previous = torch.zeros_like(actions)
        self.ready = torch.zeros_like(actions[:, 0], dtype=torch.bool)

    def reset(self, ids):
        self.ready[ids] = False

    def command(self, target_delta, dt):
        self.applied = target_delta / (self.rate * dt)

    def measure(self, previous, current, nominal, valid, dt):
        measured = (current - previous) / (self.rate * dt)
        valid = valid & torch.isfinite(measured).all(-1)
        motion = (measured - self.previous).square().clamp_max(1.).mean(-1)
        projection = (self.applied - nominal).square().clamp_max(1.).mean(-1)
        self.previous.copy_(torch.where(valid[:, None], measured, 0.))
        result = torch.where(self.ready & valid, motion, 0.) + .1 * projection
        self.ready.copy_(valid)
        return result


def executed_position(m):
    return torch.cat((m['q'], m['wrist_pose'][..., :3].flatten(1), m['wrist_joint_angles'].flatten(1)), -1)


def target_position(task, world):
    return torch.cat((task.command, world.palms.flatten(1), world.wrist_angles.flatten(1)), -1)


def accepted_support_n(measurement, whole_hand_lift_force):
    from .tabletop_surface_exploration import support_components
    return support_components(measurement, whole_hand_lift_force)[1].sum(-1)


def _relative_tangent_speed(task):
    m = task.m
    points = m['effective_pad_points_w']
    center = m['object_relative'] + m['wrist_pose'][..., :3].mean(1)
    obj_v, obj_w = m['object_velocity'].split(3, dim=-1)
    obj_contact = obj_v[:, None, None] + torch.linalg.cross(
        obj_w[:, None, None], points - center[:, None, None], dim=-1)
    hands = task.surface_world.hands
    indices = task.surface_pad_body_indices
    centers = torch.stack([h.data.body_com_pos_w[:, ids] for h, ids in zip(hands, indices)], 1).to(points)
    velocity = torch.stack([h.data.body_com_vel_w[:, ids] for h, ids in zip(hands, indices)], 1).to(points)
    linear, angular = velocity.split(3, dim=-1)
    relative = obj_contact - linear - torch.linalg.cross(angular, points - centers, dim=-1)
    vectors = m['effective_pad_vectors_w']
    normals = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tangent = relative - (relative * normals).sum(-1, keepdim=True) * normals
    weights = torch.where(m['effective_pad_valid'], m['effective_pad_loads_n'], 0.)
    return torch.where(weights > 0., tangent.norm(dim=-1) * weights, 0.).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1e-8)


def apply_reward(task, pieces, valid, dt, hand_lift_force_n, object_weight_n, *, milestone_valid=None):
    from .tabletop_surface_exploration import qualified_pad_contact
    reward = task.surface_reward_ablation
    if not reward.terms:
        return
    tracker = getattr(task, 'surface_early_hold', None)
    bottom = task.m['bottom']
    zero = torch.zeros_like(bottom)
    # Keep the legacy milestone observer live for the 351D observation prefix.
    if LIFT in reward.terms and 'lift_milestone' in pieces:
        pieces['lift_milestone'] = zero
    no = torch.zeros_like(valid)
    support, weight = zero, zero
    if SUPPORT in reward.terms:
        if hand_lift_force_n is None or object_weight_n is None:
            raise ValueError('native support and object weight required')
        support, weight = accepted_support_n(task.m, hand_lift_force_n), object_weight_n
    terms = reward.update(bottom=bottom, qualified=qualified_pad_contact(task),
        early_window_good=tracker.window_good if tracker is not None else no,
        early_duration=tracker.duration if tracker is not None else zero,
        drift=tracker.reward_drift_m if tracker is not None else zero,
        window_completed=tracker is not None and tracker.window_completed,
        window_native_valid=tracker.window_native_valid if tracker is not None else no,
        slip_observable=tracker.slip_observable if tracker is not None else no,
        relative_speed=_relative_tangent_speed(task) if MOTION in reward.terms else zero,
        support=support, weight=weight, valid=valid, dt=dt,
        smoothness=task.surface_motion_cost if SMOOTH in reward.terms else zero,
        milestone_valid=milestone_valid)
    pieces.update(terms)
