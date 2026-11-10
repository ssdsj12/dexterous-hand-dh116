"""Opt-in bilateral fixed-wrist proposals, with original full nonlinear guards.

FK/Jacobians and the first-order response below propose geometry only. They
never set a measured contact, grant a budget, or qualify moving-wrist control.
"""
from types import SimpleNamespace
import math

import numpy as np
import torch

from .tabletop_tripod_candidate_planner import (
    TripodCandidatePlanner, ACTIVE_IDS, DIGITS, JOINT_GROUPS, DT, RATE,
    _command_path_bounds, _finite, _logical_command, _substeps,
    decode_velocity_action24,
)
from .tabletop_tripod_nonlinear_geometry import audit_pad_region
from .tabletop_tripod_pressure_hold import TripodPressureHold


HAND_DIGIT_ACTIVE_INDICES = (((0, 1), (2,), (3,)), ((6, 7), (8,), (9,)))


def recovery_scope(styles, recovery_hands='right'):
    if recovery_hands not in ('right', 'left', 'both'):
        raise ValueError("recovery_hands must be 'right', 'left' or 'both'")
    if not isinstance(styles, torch.Tensor) or styles.ndim != 1 or styles.dtype != torch.long:
        raise ValueError('actual mixed reset style identity required')
    enabled = torch.zeros(len(styles), 2, 5, dtype=torch.bool, device=styles.device)
    fingers = torch.zeros(len(styles), 12, dtype=torch.bool, device=styles.device)
    hands = (0, 1) if recovery_hands == 'both' else (0,) if recovery_hands == 'left' else (1,)
    for h in hands:
        enabled[styles == 1, h, :3] = True
        for group in HAND_DIGIT_ACTIVE_INDICES[h]:
            for j in group: fingers[styles == 1, j] = True
    return enabled, fingers


class BilateralTripodCandidatePlanner(TripodCandidatePlanner):
    """Independent hand FK seeds; original 34-hull nonlinear final certificate."""
    def _audit_path(self, snapshots, full_qd, current, snapshot_identity, bearing_mask,
                    bearing_provenance, commands, deltas, action24=None, logical_commands=None):
        result = super()._audit_path(snapshots, full_qd, current, snapshot_identity,
            bearing_mask, bearing_provenance, commands, deltas, action24,
            logical_commands=logical_commands)
        before = np.concatenate((_logical_command(current).astype(np.float32)[None], commands[:-1]))
        result['bilateral_approach_command_m'] = np.asarray([
            sum((_command_path_bounds(model, snapshots[h], a[h], b[h])
                 for a, b in zip(before, commands)), np.zeros(3))
            for h, model in enumerate(self.geometry.models)])
        return result

    def _hand_regions(self, hand, snapshot, delta):
        model = self.geometry.models[hand]; pred = model.predict(snapshot, delta)
        rows = []
        for digit in DIGITS:
            patch = model.patches[digit]
            shape = next(s for s in model.shapes if s['body'] == patch['body'])
            rows.append(audit_pad_region(shape['planes'], patch, pred['poses'][patch['body']],
                self.geometry.scene['rod'], contact_distance_m=self.geometry.thresholds.contact_distance_m,
                normal_cosine_min=self.geometry.thresholds.normal_cosine_min))
        return rows

    def plan(self, *, snapshots, full_qd, current_command, missing_hands,
             remaining_approach_m, snapshot_identity, bearing_mask, bearing_provenance,
             substeps=8, wrist_motion_requested=False, cooked_live_qualified=False):
        if wrist_motion_requested:
            raise ValueError('fixed wrist recovery only; moving-wrist cooked/live qualification remains closed')
        n = _substeps(substeps)
        missing = np.asarray(missing_hands)
        if missing.shape != (2, 3) or missing.dtype != np.bool_:
            raise ValueError('independent hand x digit missing mask required')
        remaining = _finite(remaining_approach_m, (2, 3), 'remaining per-hand approach budget')
        if (remaining < 0).any() or (remaining > .001+1e-12).any():
            raise ValueError('original remaining per-episode approach budget required')
        bearing = np.asarray(bearing_mask.detach().cpu() if hasattr(bearing_mask, 'detach') else bearing_mask)
        if bearing.shape != (2, 5) or bearing.dtype != np.bool_:
            raise ValueError('actual bilateral bearing mask required')
        full, _ = self._snapshot(snapshots, full_qd, snapshot_identity)
        command = _logical_command(current_command)
        baseline = dict(accepted=False, action24=np.zeros(24, np.float32), budget_consumed_m=0,
            native_contact_verified=False, native_candidate_eligible=False,
            evidence_kind='geometric_proposal_not_native_contact',
            objective=dict(bearing_weight=10, regularization=1e-8),
            post_native_guard_reaudit_required=True, moving_wrist_enabled=False)
        eligible = missing & ~bearing[:, :3] & (remaining > 0)
        if not eligible.any(): return dict(baseline, reason='NO_ELIGIBLE_RECOVERY_BUDGET')
        actual = self.geometry.audit_state(snapshots, np.zeros(6), left_delta=np.zeros(6),
            bearing_mask=bearing_mask, bearing_provenance=bearing_provenance)
        if not actual['motion_safe'] or not actual['audit_complete']:
            return dict(baseline, reason='ACTUAL_SOURCE_GEOMETRY_UNCERTIFIED', audit=actual,
                rejection_reasons=actual['rejection_reasons'])
        zero = np.zeros(24, np.float32)
        base = decode_velocity_action24(command, full[:, ACTIVE_IDS], zero, self.low, self.high,
            substeps=n, response_per_control=self.response)
        action = zero.copy(); initial_rows = []
        for h, model in enumerate(self.geometry.models):
            delta = base['predicted_deltas'][-1, h]
            rows = self._hand_regions(h, snapshots[h], delta); initial_rows.append(rows)
            free = sorted({j for d, group in enumerate(JOINT_GROUPS) if eligible[h, d] for j in group})
            if not free: continue
            pred = model.predict(snapshots[h], delta)
            Jm, desired, Jb = [], [], []
            for d, digit in enumerate(DIGITS):
                w = rows[d].get('witness'); patch = model.patches[digit]; pose = pred['poses'][patch['body']]
                point = (np.asarray(w['point_w_m']) if w is not None else
                    pose[:3, :3] @ np.asarray(patch['center_local_m']) + pose[:3, 3])
                local = pose[:3, :3].T @ (point-pose[:3, 3]); jac = []
                for j in free:
                    points = []
                    for sign in (-1, 1):
                        trial = delta.copy(); trial[j] += sign*1e-5
                        p = model.predict(snapshots[h], trial)['poses'][patch['body']]
                        points.append(p[:3, :3] @ local + p[:3, 3])
                    jac.append((points[1]-points[0])/2e-5)
                jac = np.asarray(jac).T
                if bearing[h, d]: Jb.append(jac)
                if eligible[h, d] and w is not None and abs(w['distance_m']) <= .010:
                    target = np.asarray(w['object_surface_point_w_m'])-point
                    target *= min(1., min(remaining[h,d], .001*DT*n)/max(np.linalg.norm(target), 1e-30))
                    Jm.append(jac); desired.append(target)
            if not Jm: continue
            A = np.concatenate(Jm); b = np.concatenate(desired)
            bearing_jac = np.concatenate(Jb) if Jb else np.zeros((0, len(free)))
            # The measured material-point bearing penalty is 10 times the
            # squared movement error; inactive bearing joints are never replaced.
            lhs = A.T @ A + 10*bearing_jac.T @ bearing_jac + np.eye(len(free))*1e-8
            dq = np.linalg.solve(lhs, A.T @ b)
            # First-order response only seeds the velocity; final nonlinear
            # sweep, limits and encoded command path remain authoritative.
            alpha = max(self.response, 1e-9)
            action[6*h+np.asarray(free)] = np.clip(dq/(RATE*DT*n*alpha), -1., 1.).astype(np.float32)
        if not np.any(action): return dict(baseline, reason='NO_FK_APPROACH_AUTHORITY')
        allowed = np.where(eligible, np.minimum(remaining, .001*DT*n), 0.)
        step_allowed = np.where(eligible, np.minimum(remaining, .001*DT), 0.)
        for _ in range(24):
            path = decode_velocity_action24(command, full[:,ACTIVE_IDS], action, self.low, self.high,
                substeps=n, response_per_control=self.response)
            previous = np.concatenate((command.astype(np.float32)[None], path['command_targets'][:-1]))
            costs = np.asarray([[_command_path_bounds(model, snapshots[h], a[h], b[h])
                for h, model in enumerate(self.geometry.models)] for a,b in zip(previous,path['command_targets'])])
            if (costs.sum(0) <= allowed+1e-15).all() and (costs <= step_allowed+1e-15).all(): break
            # Conservative hull-vertex arc bounds can be much larger than
            # the local witness displacement. Scale to that actual bound in
            # one step, then redecode including float32 SDK rounding.
            charged=costs.sum(0)
            ratios=[allowed[h,d]/charged[h,d] for h in range(2) for d in range(3) if charged[h,d]>0]
            ratios += [step_allowed[h,d]/costs[k,h,d] for k in range(n) for h in range(2)
                for d in range(3) if costs[k,h,d]>0]
            action[:12] *= np.float32(min(.8,.95*min(ratios,default=0.)))
        else: return dict(baseline, reason='NO_ENCODED_BUDGET_SAFE_CANDIDATE')
        before = after = 0.
        for h in range(2):
            rows = self._hand_regions(h, snapshots[h], path['predicted_deltas'][-1,h])
            for d in range(3):
                if eligible[h,d]: before += self._score(initial_rows[h][d]); after += self._score(rows[d])
        if not math.isfinite(after) or not after < before:
            return dict(baseline, reason='NO_CERTIFIED_IMPROVING_CANDIDATE')
        audit = self._audit_path(snapshots, full_qd, command, snapshot_identity, bearing_mask,
            bearing_provenance, path['command_targets'], path['predicted_deltas'], action,
            logical_commands=path['logical_command_targets'])
        return dict(audit, objective=baseline['objective'], post_native_guard_reaudit_required=True,
            moving_wrist_enabled=False, reason='BOUNDED_BILATERAL_GEOMETRIC_PROPOSAL'
                if audit['accepted'] else 'NONLINEAR_GEOMETRY_REJECTED')


class _LeftHandPlannerView:
    """Run the unchanged right pressure law on left data, auditing original order."""
    def __init__(self, planner):
        self.actual = planner; self.low = planner.low[::-1].copy(); self.high = planner.high[::-1].copy()
        self.response = planner.response
        self.geometry = SimpleNamespace(models=planner.geometry.models[::-1], scene=planner.geometry.scene,
            audit_state=self._audit_state)

    @staticmethod
    def _context(ctx):
        ctx = dict(ctx)
        for key in ('snapshots', 'full_qd', 'current_command', 'bearing_mask'):
            if key in ctx:
                value=ctx[key]
                if hasattr(value,'detach'):value=value.detach().cpu().numpy()
                ctx[key] = value[::-1].copy() if isinstance(value, np.ndarray) else list(value)[::-1]
        return ctx

    def _snapshot(self, snapshots, full_qd, identity):
        full, calibration = self.actual._snapshot(snapshots[::-1], full_qd[::-1], identity)
        return full[::-1], calibration

    def _audit_state(self, snapshots, right_delta, left_delta, **ctx):
        ctx = self._context(ctx)
        return self.actual.geometry.audit_state(snapshots[::-1], left_delta, left_delta=right_delta, **ctx)

    def audit_action(self, *, action24, **ctx):
        action = np.asarray(action24).copy(); action[:12] = action[:12].reshape(2,6)[::-1].reshape(12)
        return self.actual.audit_action(action24=action, **self._context(ctx))


class HandTripodPressureHold:
    """Fresh measured bearing pressure per hand, without recovery ledger writes."""
    def __init__(self, planner, hand):
        if hand not in (0,1): raise ValueError('hand must be 0 or 1')
        self.hand = hand; self.planner = planner
        self.hold = TripodPressureHold(planner if hand == 1 else _LeftHandPlannerView(planner))

    def plan(self, **ctx):
        if self.hand == 1: return self.hold.plan(**ctx)
        ctx = _LeftHandPlannerView._context(ctx)
        for key in ('effective_pad_loads_n','effective_pad_vectors_w','effective_pad_points_w_m',
                    'effective_pad_valid','effective_pad_depth_m','rejected_contact_loads_n'):
            value = ctx[key]
            if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
            ctx[key] = np.asarray(value)[::-1].copy()
        a = np.asarray(ctx['upstream_pressure_action24'].detach().cpu() if hasattr(ctx['upstream_pressure_action24'],'detach')
            else ctx['upstream_pressure_action24']).copy()
        a[:12] = a[:12].reshape(2,6)[::-1].reshape(12); ctx['upstream_pressure_action24'] = a
        out = self.hold.plan(**ctx); action = out['action24'].copy()
        action[:12] = action[:12].reshape(2,6)[::-1].reshape(12); out['action24'] = action
        for row in out['per_pad']: row['hand'] = 0
        return out


class TripodTeacherLift:
    """Finite teacher-only phase candidate; original success observer owns HOLD.

    LOAD_TRANSFER begins from contact, without an mg prerequisite. The output
    is a velocity request; original transport/collision/terminal guards follow.
    """
    ACQUIRE, STABILIZE_CONTACT, LOAD_TRANSFER, LIFT, AIRBORNE_HOLD, TERMINATED = range(6)
    def __init__(self, n, *, enabled=False, source='teacher', device='cpu', max_duration_s=60.):
        if enabled and source != 'teacher': raise ValueError('privileged lift candidate requires teacher source')
        if type(n) is not int or n < 1 or not math.isfinite(max_duration_s) or max_duration_s <= 0:
            raise ValueError('finite teacher lift duration and positive environment count required')
        self.enabled = enabled; self.n = n; self.max_duration_s = max_duration_s
        self.phase = torch.zeros(n, device=device, dtype=torch.long)
        self.elapsed = torch.zeros(n, device=device, dtype=torch.float64)
        self.dwell = torch.zeros_like(self.elapsed); self.transfer = torch.zeros_like(self.elapsed)
        self.lift_time = torch.zeros_like(self.elapsed)

    def step(self, *, dt_s, real_stable_contact, tracking_safe, object_following, airborne,
             original_hold_qualified, mandatory_unload, terminal, object_net_lift_m=None):
        masks = (real_stable_contact,tracking_safe,object_following,airborne,original_hold_qualified,mandatory_unload,terminal)
        if (type(dt_s) not in (int,float) or not math.isfinite(dt_s) or dt_s <= 0
                or any(not isinstance(m,torch.Tensor) or m.shape!=(self.n,) or m.dtype!=torch.bool
                    or m.device!=self.phase.device for m in masks)):
            raise ValueError('finite dt and per-environment real boolean teacher feedback required')
        # Unknown height never qualifies HOLD. This is the unchanged original
        # SurfaceLiftSuccess default 30mm gate, not a relaxed task success.
        if object_net_lift_m is None:
            height_reached=torch.zeros_like(airborne)
        else:
            if (not isinstance(object_net_lift_m,torch.Tensor) or object_net_lift_m.shape!=(self.n,)
                    or object_net_lift_m.device!=self.phase.device or not object_net_lift_m.is_floating_point()
                    or not bool(torch.isfinite(object_net_lift_m).all())):
                raise ValueError('actual finite native net object lift [N] required')
            height_reached=object_net_lift_m>=.03
        speed = torch.zeros_like(self.elapsed)
        if self.enabled:
            self.elapsed += dt_s
            fail = terminal | (self.elapsed >= self.max_duration_s)
            self.phase[fail] = self.TERMINATED
            safe = real_stable_contact & tracking_safe & ~mandatory_unload & ~fail & (self.phase!=self.TERMINATED)
            self.phase[(self.phase==self.ACQUIRE)&safe] = self.STABILIZE_CONTACT
            stabilizing = self.phase==self.STABILIZE_CONTACT
            self.dwell = torch.where(stabilizing & safe,self.dwell+dt_s,torch.zeros_like(self.dwell))
            self.phase[stabilizing & (self.dwell >= .2-1e-12)] = self.LOAD_TRANSFER
            # Loss during transfer/lift withdraws motion; it cannot silently
            # restart acquisition or replenish its per-episode budgets.
            running = (self.phase==self.LOAD_TRANSFER)|(self.phase==self.LIFT)
            # The object cannot follow an unloading motion before any motion
            # is requested. Check following after the first bounded step;
            # contact/tracking remain required at entry, without an mg gate.
            following_required=(self.phase==self.LIFT)|((self.phase==self.LOAD_TRANSFER)&(self.transfer>0))
            self.phase[running & (~real_stable_contact|~tracking_safe|(following_required&~object_following))] = self.TERMINATED
            transfer = (self.phase==self.LOAD_TRANSFER)&safe
            remaining = (.002-self.transfer).clamp_min(0.)
            speed[transfer] = torch.minimum(remaining[transfer]/dt_s, speed.new_full((int(transfer.sum()),),.0005))
            self.transfer += speed*dt_s
            lifted = transfer & (self.transfer >= .002-1e-12)
            self.phase[lifted] = self.LIFT
            lift = (self.phase==self.LIFT)&safe&~lifted
            self.lift_time[lift] += dt_s
            speed[lift] = (.001+.002*(self.lift_time[lift]/2.).clamp(0.,1.)).clamp_max(.003)
            self.phase[(self.phase==self.LIFT)&airborne&height_reached&safe] = self.AIRBORNE_HOLD
            speed[(self.phase==self.AIRBORNE_HOLD)|mandatory_unload|terminal|(self.phase==self.TERMINATED)] = 0.
        return dict(phase=self.phase.clone(),common_ascent_m_s=speed,mandatory_unload=mandatory_unload.clone(),
            original_hold_qualified=original_hold_qualified.clone(),counts_as_student_success=False,
            original_30mm_5s_success_observer_required=True,moving_wrist_geometry_qualification_required=True)
