"""Experimental training-only pressure-guided local MPC for tripod acquisition.

Pinch uses the existing absolute-reference expert unchanged. Tripod references
come from actual measured pad pressure and points, with the original Teacher
wrist actions. The common wrist transport is only a kinematic reference; no
object motion, contact force, or friction dynamics are predicted here.
"""
from __future__ import annotations

import torch

from .tabletop_surface_mpc import SurfaceGripMPC
from .tabletop_teacher_pressure_reference import pressure_reference_targets
from .tabletop_vision_teacher_absolute_mpc import AbsoluteTeacherReferenceMPC, _AbsoluteNativeSurfaceMPC
from .tabletop_vision_teacher_mpc import _json_value


PRESSURE_GOAL_SOURCE_SHA256 = 'fea884d24928026d006669fcef8af1565eb12ad25e0c35130035a8bc0378274b'
PRESSURE_GOALS_N = (1.5, .75, .75, 0., 0.)
PINCH = 'pinch_thumb_index'
TRIPOD = 'tripod_thumb_index_middle'


class _PressureNativeSurfaceMPC(_AbsoluteNativeSurfaceMPC):
    def __init__(self, env, radius, tripod):
        super().__init__(env, radius)
        self.tripod = tripod
        self.pressure_planner = SurfaceGripMPC()
        self.source_horizon = None
        self.pressure_info = {}
        self.pressure_reasons = {}
        self.stats.update(activation='pinch_absolute_3mm_opposition;tripod_any_valid_bearing_pad_before_lift',
            model='local_pad_kinematics_with_simulated_pressure_reference',
            pressure_direction_scope='relative_to_common_commanded_wrist_transport',
            force_prediction=False, force_precision_gate=False)

    def _prepare_pressure(self):
        m = self.t.m
        loads = m['effective_pad_loads_n']
        valid = m['effective_pad_valid']
        vectors = m['effective_pad_vectors_w']
        bearing = valid & (loads > .05)
        norm = vectors.norm(dim=-1)
        normal_ok = torch.isfinite(vectors).all(-1) & torch.isfinite(norm) & (norm > 0)
        unknown = ((loads > .05) & (~valid | ~normal_ok)).any(-1)
        rejected = m.get('rejected_contact_loads_n')
        if not isinstance(rejected, torch.Tensor) or rejected.shape != (self.w.n, 2):
            rejected = loads.new_full((self.w.n, 2), float('nan'))
        bad_hand = (~torch.isfinite(loads).all(-1) | (loads < 0).any(-1)
                    | ~torch.isfinite(rejected) | (rejected < 0))
        self.unknown_hands = unknown | bad_hand
        self.rejected_hands = torch.isfinite(rejected) & (rejected > .05)
        self.bad_pressure_rows = bad_hand.any(-1)
        # Two thumb joints, then one joint for each remaining digit.
        self.hold_joints = bearing[..., [0, 0, 1, 2, 3, 4]]
        self.hold_joints |= (self.unknown_hands | self.rejected_hands)[..., None]
        self.bearing = bearing

    @torch.no_grad()
    def refine(self, nominal):
        self.pressure_info = {}
        self.pressure_reasons = {}
        self._prepare_pressure()
        fallback = nominal.clone()
        fingers = self.source_horizon[:, 0, :12].to(nominal).reshape(-1, 2, 6)
        fingers = torch.where(self.hold_joints, 0., fingers)
        fingers = torch.where(self.bad_pressure_rows[:, None, None], 0., fingers)
        fallback[self.tripod, :12] = fingers.flatten(1)[self.tripod]
        fallback[self.tripod & ~self.alive, :12] = 0.
        self.pressure_fallback_actions = fallback.clone()
        return super().refine(fallback)

    def _refine_ids(self, ids, nominal, result, failed):
        # Pressure's measured lead bounds differ per environment. Keep each solve
        # independent while preserving the original batched pinch solve exactly.
        pinch = ids[~self.tripod[ids]]
        if pinch.numel():
            super()._refine_ids(pinch, nominal, result, failed)
        for index in ids[self.tripod[ids]].unbind(0):
            super()._refine_ids(index.reshape(1), nominal, result, failed)

    def _activation_mask(self, paired, bottom, reference, ids, valid):
        if not bool(self.tripod[ids].any()):
            return super()._activation_mask(paired, bottom, reference, ids, valid)
        return ((valid & ~self.unknown_hands[ids, :, None]).any((1, 2))
                & ~self.bad_pressure_rows[ids] & (self.bad_predictions[ids] < 3))

    def _nominal_trajectory(self, ids, command, actions, lower, upper):
        if not bool(self.tripod[ids].any()):
            return super()._nominal_trajectory(ids, command, actions, lower, upper)
        source = self.source_horizon[ids, :, :12].to(command).reshape(-1, self.horizon, 2, 6)
        source = torch.where(self.hold_joints[ids, None], 0., source)
        result = []; current = command.clone()
        for action in source.unbind(1):
            current = (current + action * .005).maximum(lower).minimum(upper)
            result.append(current)
        return torch.stack(result, 1)

    def _model_points(self, ids, hand, material, measured):
        return measured if bool(self.tripod[ids].any()) else super()._model_points(ids, hand, material, measured)

    def _target_trajectory(self, ids, points, targets, wrist_motion, target_points, valid):
        if not bool(self.tripod[ids].any()):
            return super()._target_trajectory(ids, points, targets, wrist_motion, target_points, valid)
        m = self.t.m
        loads = m['effective_pad_loads_n'][ids].to(points)
        vectors = m['effective_pad_vectors_w'][ids].to(points)
        goals = points.new_tensor(PRESSURE_GOALS_N).expand_as(loads)
        depth = m.get('effective_pad_depth_m')
        depth = depth[ids].to(points) if isinstance(depth, torch.Tensor) else None
        measured_valid = valid & ~self.unknown_hands[ids, :, None]
        detail = pressure_reference_targets(points, vectors, loads, measured_valid, goals,
            horizon=self.horizon, effective_depth_m=depth)
        rejected = self.rejected_hands[ids]
        relief = detail['eligible'] & rejected[..., None]
        speed = torch.where(relief, -.001, detail['speed_m_s'])
        time = torch.arange(1, self.horizon + 1, device=points.device, dtype=points.dtype) / 60
        pressure_targets = points[:, None] + time[None, :, None, None, None] * (
            detail['inward'] * speed[..., None])[:, None]
        overloaded = detail['eligible'] & ((loads > goals) | rejected[..., None])
        if depth is not None:
            overloaded |= detail['eligible'] & (depth > .0006)
        index = int(ids[0])
        self.pressure_info[index] = dict(inward=detail['inward'], protected=overloaded,
            speed_m_s=speed, eligible=detail['eligible'], target_loads_n=goals,
            reference_targets_m=pressure_targets, common_wrist_motion_m=wrist_motion)
        # Pressure is a relative finger correction carried by the unchanged
        # commanded wrist transport. This transport is not measured object motion.
        return pressure_targets + wrist_motion, detail['weights'][:, 0]

    def _plan(self, ids, **kwargs):
        if not bool(self.tripod[ids].any()):
            return super()._plan(ids, **kwargs)
        index = int(ids[0]); q = kwargs['q'][0]; command = kwargs['command'][0]
        low = kwargs['lower'].maximum(q - .1)
        high = kwargs['upper'].minimum(q + .1)
        # An unknown bearing normal is never an invitation to replay compression.
        low = torch.where(self.unknown_hands[index, :, None], low.maximum(command), low)
        high = torch.where(self.unknown_hands[index, :, None], high.minimum(command), high)
        kwargs = dict(kwargs, lower=low, upper=high)
        target, diag = self.pressure_planner.plan(**kwargs)
        detail = self.pressure_info[index]
        relative = diag['predicted_points'] - kwargs['points'][:, None] - kwargs['wrist_motion']
        normal = (relative * detail['inward'][:, None]).sum(-1)
        detail['predicted_relative_normal_m'] = normal
        if bool(((normal > 1e-10) & detail['protected'][:, None]).any()):
            self.pressure_reasons[index] = 'pressure_direction_rejected'
            raise ValueError('pressure proposal predicts inward motion at overloaded pad')
        self.pressure_reasons[index] = 'optimized_pressure_reference'
        return target, diag


class PressureTeacherReferenceMPC(AbsoluteTeacherReferenceMPC):
    """Mixed-style experimental expert; canonical action and privilege boundaries stay 24D."""
    def __init__(self, env, source_sha, reference_actions, *, grasp_styles, trust_radius_rad=.001):
        if (not isinstance(grasp_styles, (list, tuple)) or len(grasp_styles) != env.world.n
                or any(style not in (PINCH, TRIPOD) for style in grasp_styles)):
            raise ValueError('one supported grasp style per native environment required')
        super().__init__(env, source_sha, reference_actions, trust_radius_rad=trust_radius_rad)
        self.grasp_styles = tuple(grasp_styles)
        tripod = torch.tensor([style == TRIPOD for style in grasp_styles],
            device=env.task.command.device, dtype=torch.bool)
        self.native = _PressureNativeSurfaceMPC(env, self.trust_radius_rad, tripod)

    @torch.no_grad()
    def refine(self, reference_actions, *, context=None, alive=None):
        tick = context.get('tick') if isinstance(context, dict) else None
        if type(tick) is int and tick >= 0:
            horizon = self.reference_actions.new_zeros(self.env.world.n, self.native.horizon, 24)
            count = min(self.native.horizon, max(0, self.reference_actions.shape[1] - tick))
            horizon[:, :count] = self.reference_actions[:, tick:tick + count]
            self.native.source_horizon = horizon
        issued, diagnostics = super().refine(reference_actions, context=context, alive=alive)
        for index, row in enumerate(diagnostics['per_env']):
            row['grasp_style'] = self.grasp_styles[index]
            if not bool(self.native.tripod[index]):
                row['profile'] = 'absolute_reference'
                continue
            row.update(profile='pressure_reference', pressure_reference_radius_rad=None,
                pressure_fallback_actions=_json_value(self.native.pressure_fallback_actions[index]),
                pressure_unknown_normal_hands=_json_value(self.native.unknown_hands[index]),
                pressure_rejected_contact_hands=_json_value(self.native.rejected_hands[index]))
            detail = self.native.pressure_info.get(index)
            if detail is not None:
                for source, destination in (('speed_m_s', 'pressure_speed_m_s'),
                        ('eligible', 'pressure_eligible'), ('target_loads_n', 'pressure_target_loads_n'),
                        ('reference_targets_m', 'pressure_reference_targets_m'),
                        ('predicted_relative_normal_m', 'pressure_predicted_relative_normal_m')):
                    if source in detail:
                        row[destination] = _json_value(detail[source][0])
            if row['reason'] not in ('inactive_episode', 'invalid_absolute_command'):
                if index in self.native.pressure_reasons:
                    row['reason'] = self.native.pressure_reasons[index]
                elif bool(self.native.bad_pressure_rows[index]):
                    row['reason'] = 'invalid_pressure_measurement_hold'
                elif bool(self.native.unknown_hands[index].all()):
                    row['reason'] = 'unknown_contact_normal_hold'
                elif not bool(self.native.bearing[index].any()):
                    row['reason'] = 'no_measured_bearing_contact'
                else:
                    row['reason'] = 'pressure_model_fallback_hold'
        diagnostics.update(version='dh116-vision-teacher-pressure-reference-mpc-v1',
            profile='pressure_reference', pressure_goal_source_sha256=PRESSURE_GOAL_SOURCE_SHA256,
            pressure_goal_scope='historically_audited_privileged_simulation_only_not_hardware_calibration',
            pressure_target_loads_n=list(PRESSURE_GOALS_N), pressure_reference_radius_rad=None,
            pressure_command_bounds='original_joint_limits_intersect_measured_q_plus_minus_0.1_and_0.005_slew',
            pressure_bounds_scope='accepted_local_solves;infeasible_hold_fallback_still_subject_to_native_guards',
            pressure_fallback='hold_bearing_or_unknown_hand_fingers;source_velocity_for_missing_digits',
            pressure_prediction='local_pad_kinematics_not_force_or_object_dynamics',
            pressure_direction_admission='overloaded_pad_no_inward_displacement_relative_to_common_wrist_transport',
            pressure_transport_limit='commanded_common_wrist_transport_is_not_measured_object_motion',
            force_prediction=False, force_precision_gate=False, training_expert_privileged_only=True)
        return issued, diagnostics
