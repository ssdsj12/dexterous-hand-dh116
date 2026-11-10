"""Opt-in CAD acquisition references beside actual-contact pressure MPC.

Only twelve finger commands are optimized. Teacher wrist actions are unchanged.
CAD points/normals and the current finite-cylinder pose are privileged geometry,
never native contact measurements. No force or object dynamics are predicted.
"""
from __future__ import annotations

import math

import torch

from .tabletop_effective_pad import DISTAL_LINKS, build_pad_atlas
from .tabletop_effective_pad_runtime import _rotation
from .tabletop_teacher_geometric_acquisition import geometric_acquisition_targets
from .tabletop_vision_teacher_pressure_mpc import (
    PressureTeacherReferenceMPC, _PressureNativeSurfaceMPC, PRESSURE_GOALS_N,
)
from .tabletop_vision_teacher_mpc import _json_value


STOP_CODES = dict(none=0, distance_budget=1, request_time_budget=2,
                  no_progress=3, model_infeasible=4, terminal_or_reset=5)


class _AcquisitionNativeSurfaceMPC(_PressureNativeSurfaceMPC):
    def __init__(self, env, radius, tripod, atlas, cylinder_radius_m, cylinder_length_m):
        super().__init__(env, radius, tripod)
        self.cylinder_radius_m = cylinder_radius_m
        self.cylinder_length_m = cylinder_length_m
        command = env.task.command
        rows = [atlas['sides'][side][name] for side in ('left', 'right') for name in DISTAL_LINKS]
        if any(row.get('geometry_only') is not True for row in rows):
            raise ValueError('explicit geometry_only CAD atlas required')
        self.local_centers = command.new_tensor([row['center_local_m'] for row in rows]).reshape(2,5,3)
        self.local_outward = command.new_tensor([row['reference_outward_normal_local'] for row in rows]).reshape(2,5,3)
        if (not torch.isfinite(self.local_centers).all() or not torch.isfinite(self.local_outward).all()
                or (self.local_outward.norm(dim=-1) <= 0).any()):
            raise ValueError('finite CAD centers and nonzero outward normals required')
        self.requested_approach = command.new_zeros(self.w.n,2,5)
        self.request_time = torch.zeros_like(self.requested_approach)
        self.no_progress = torch.zeros_like(self.requested_approach)
        self.best_gap = torch.full_like(self.requested_approach, float('inf'))
        self.prior_request = torch.zeros_like(self.requested_approach, dtype=torch.bool)
        self.stop_code = torch.zeros_like(self.requested_approach, dtype=torch.long)
        self.geometry = {}
        self.dt = 1/60  # The existing canonical eight 480-Hz substeps, not wall time.
        self.stats.update(activation='pinch_absolute_3mm_opposition;tripod_measured_bearing_or_eligible_CAD_acquisition',
            model='local_point_kinematics_with_measured_pressure_and_missing_pad_CAD_references')

    def _sensor_validity(self):
        """Read cached native flags only; never sample or mutate the world."""
        sources = []
        bad = torch.zeros(self.w.n, device=self.t.command.device, dtype=torch.bool)
        def checked(value, name, boolean):
            types = (torch.bool,) if boolean else (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            if (not isinstance(value, torch.Tensor) or value.shape != (self.w.n,)
                    or value.device != bad.device or value.dtype not in types):
                raise ValueError(name+' has invalid shape/dtype/device')
            sources.append(name)
            return value if boolean else value != 0
        if hasattr(self.w, 'pad_measurements'):
            packet = self.w.pad_measurements
            if isinstance(packet, dict) and 'effective_pad_bad_env' in packet:
                bad |= checked(packet['effective_pad_bad_env'], 'world.pad_measurements.effective_pad_bad_env', True)
            else:
                bad[:] = True; sources.append('missing_world_pad_validity')
        elif 'effective_pad_bad_env' in self.t.m:
            bad |= checked(self.t.m['effective_pad_bad_env'], 'task.m.effective_pad_bad_env', True)
        elif hasattr(self.w, 'adapter_fault_bits') or getattr(self.w, 'continuous_motion', False):
            bad[:] = True; sources.append('missing_native_pad_validity')
        else:
            sources.append('cpu_fixture_without_native_validity')
        if hasattr(self.w, 'adapter_fault_bits'):
            bad |= checked(self.w.adapter_fault_bits, 'world.adapter_fault_bits', False)
        self.geometry_sensor_invalid = bad
        self.geometry_validity_sources = sources

    def reset(self, ids):
        super().reset(ids)
        if hasattr(self, 'stop_code'):
            # One cumulative attempt per source episode. Native reset never
            # replenishes budgets; the outer absolute wrapper forbids restart.
            self.stop_code[ids] = STOP_CODES['terminal_or_reset']
            self.prior_request[ids] = False

    def _prepare_pressure(self):
        super()._prepare_pressure()
        self._sensor_validity()
        invalid_tripod = self.geometry_sensor_invalid & self.tripod
        self.bad_pressure_rows |= invalid_tripod
        self.unknown_hands |= invalid_tripod[:,None]
        self.hold_joints |= invalid_tripod[:,None,None]
        self.geometry_prediction_current = torch.zeros_like(self.tripod)
        self.geometry_predicted_gap_m = self.t.command.new_zeros(self.w.n,self.horizon,2,5)
        self.geometry_models = {}
        m = self.t.m
        poses = torch.stack([hand.data.body_link_pose_w[:, bids]
                             for hand, bids in zip(self.w.hands, self.bodies)],1).to(self.t.command)
        rotations = _rotation(poses)
        self.cad_points = (rotations @ self.local_centers[None,...,None]).squeeze(-1) + poses[...,:3]
        self.cad_outward = (rotations @ self.local_outward[None,...,None]).squeeze(-1)
        quaternion_norm = poses[...,3:].norm(dim=-1)
        pose_valid = torch.isfinite(poses).all(-1) & torch.isfinite(quaternion_norm) & (quaternion_norm > 0)
        self.cad_points = torch.where(pose_valid[...,None], self.cad_points, float('nan'))
        self.cad_outward = torch.where(pose_valid[...,None], self.cad_outward, float('nan'))
        # Same measured world center reconstruction as the existing native MPC.
        center = m['object_relative'] + m['wrist_pose'][...,:3].mean(1)
        self.object_pose = torch.cat((center,m['object_quat']),-1).to(self.t.command)
        remaining = (.001 - self.requested_approach).clamp_min(0.)
        info = geometric_acquisition_targets(self.cad_points, self.cad_outward,
            self.object_pose, m['effective_pad_loads_n'].to(self.t.command), m['effective_pad_valid'],
            cylinder_radius_m=self.cylinder_radius_m, cylinder_length_m=self.cylinder_length_m,
            horizon=self.horizon, dt=self.dt, remaining_distance_m=remaining)
        gap = info['gap_m']
        # Compare completed physical movement following the previous request.
        # Bearing intervals pause this accounting and never reset its budget.
        current_geometry_valid = (pose_valid & torch.isfinite(self.object_pose).all(-1)[:,None,None]
            & (self.object_pose[:,3:].norm(dim=-1)>0)[:,None,None]
            & ~self.geometry_sensor_invalid[:,None,None])
        # Contact acquired after the prior request may already be bearing.
        # Its valid positive CAD gap still measures actual completed progress;
        # eligibility controls new requests, not observation of the last one.
        progressed = (self.prior_request & current_geometry_valid
                      & ~self.unknown_hands[...,None] & ~self.bad_pressure_rows[:,None,None]
                      & torch.isfinite(gap) & (gap > 0) & (gap < self.best_gap - 1e-6))
        self.no_progress = torch.where(progressed, 0., self.no_progress)
        self.no_progress += (self.prior_request & ~progressed).to(gap) * self.dt
        self.best_gap = torch.where(progressed, gap, self.best_gap)
        for mask, reason in ((self.requested_approach >= .001-1e-12, 'distance_budget'),
                             (self.request_time >= 1.-1e-12, 'request_time_budget'),
                             (self.no_progress >= .5-1e-12, 'no_progress')):
            self.stop_code = torch.where((self.stop_code == 0) & mask, STOP_CODES[reason], self.stop_code)
        goals = self.t.command.new_tensor(PRESSURE_GOALS_N)
        loads = m['effective_pad_loads_n']
        overloaded = ((loads > goals) & (goals > 0) & self.bearing).any(-1)
        depth = m.get('effective_pad_depth_m')
        if isinstance(depth, torch.Tensor): overloaded |= ((depth > .0006) & self.bearing).any(-1)
        self.geometry_blocked_hands = (self.unknown_hands | self.rejected_hands | overloaded
                                       | self.bearing[...,3:].any(-1))
        eligible = (info['geometry_eligible'] & self.tripod[:,None,None] & self.alive[:,None,None]
                    & ~self.geometry_blocked_hands[...,None] & (self.stop_code == 0))
        info['geometry_eligible'] = eligible
        info['weights'] = info['weights'] * eligible[:,None]
        info['requested_distance_m'] = torch.where(eligible, info['requested_distance_m'], 0.)
        # A new request segment starts from its current positive, missing-pad
        # gap. Bearing/invalid intervals never clear accumulated no-progress
        # time, and their negative/placeholder gaps never become its baseline.
        first_request = eligible & ~self.prior_request
        self.best_gap = torch.where(first_request, gap, self.best_gap)
        requested = info['requested_distance_m'] > 0
        self.requested_approach += info['requested_distance_m']
        self.request_time += requested.to(gap) * self.dt
        self.prior_request = requested.clone()
        self.geometry = info
        self.geometry_model_infeasible = torch.zeros_like(eligible)
        # The new profile owns missing tripod digits even if CAD is currently
        # ineligible. Never fall back to a blind source closing velocity.
        required = torch.zeros_like(self.bearing); required[...,:3] = True
        self.geometry_owned = required & ~self.bearing & self.tripod[:,None,None]
        self.hold_joints |= self.geometry_owned[..., [0,0,1,2,3,4]]

    def _activation_mask(self, paired, bottom, reference, ids, valid):
        original = super()._activation_mask(paired,bottom,reference,ids,valid)
        geometric = self.geometry['geometry_eligible'][ids].any((1,2)) & (self.bad_predictions[ids] < 3)
        return original | geometric

    def _model_points(self, ids, hand, material, measured):
        original = super()._model_points(ids,hand,material,measured)
        active = self.geometry['geometry_eligible'][ids,hand]
        return torch.where(active[...,None], self.cad_points[ids,hand], original)

    def _target_trajectory(self, ids, points, targets, wrist_motion, target_points, valid):
        targets, weights = super()._target_trajectory(ids,points,targets,wrist_motion,target_points,valid)
        mask = self.geometry['geometry_eligible'][ids]
        targets = torch.where(mask[:,None,...,None], self.geometry['target_points'][ids], targets)
        weights = torch.where(mask, torch.ones_like(weights), weights)
        return targets, weights

    def _predicted_gap(self, points, ids):
        pose = self.object_pose[ids]
        axis = _rotation(pose)[..., :, 1]
        delta = points - pose[:,None,None,None,:3]
        axial = (delta * axis[:,None,None,None]).sum(-1)
        radial = delta - axial[...,None] * axis[:,None,None,None]
        length = radial.norm(dim=-1)
        gap = length - self.cylinder_radius_m
        facing = (-radial / length.clamp_min(1e-30)[...,None]
                  * self.cad_outward[ids,None]).sum(-1) / self.cad_outward[ids].norm(dim=-1)[:,None].clamp_min(1e-30)
        side = axial.abs() < self.cylinder_length_m/2
        return gap, facing, side

    def _plan(self, ids, **kwargs):
        if not bool(self.tripod[ids].any()): return super()._plan(ids, **kwargs)
        weights = kwargs['weights'].clone()
        # At most six geometry targets can be removed. Recheck every retained
        # target after each solve because objectives can share kinematic DOFs.
        for _ in range(7):
            mask = self.geometry['geometry_eligible'][ids]
            held = (self.geometry_owned[ids] & ~mask)[..., [0,0,1,2,3,4]][0]
            command = kwargs['command'][0]
            lower = torch.where(held, kwargs['lower'].maximum(command), kwargs['lower'])
            upper = torch.where(held, kwargs['upper'].minimum(command), kwargs['upper'])
            target, diag = super()._plan(ids, **dict(kwargs,weights=weights,lower=lower,upper=upper))
            gap, facing, side = self._predicted_gap(diag['predicted_points'], ids)
            original = self.geometry['gap_m'][ids,None]
            good = (torch.isfinite(gap) & (gap < original-1e-10) & (gap >= -1e-10)
                    & torch.isfinite(facing) & (facing >= .8) & side)
            failed = mask & ~good.all(1)
            if not bool(failed.any()):
                self.geometry_predicted_gap_m[ids] = gap
                self.geometry_prediction_current[ids] = True
                self.geometry_models[int(ids[0])] = kwargs
                return target, diag
            self.geometry_model_infeasible[ids] |= failed
            self.stop_code[ids] = torch.where(failed, STOP_CODES['model_infeasible'], self.stop_code[ids])
            self.prior_request[ids] &= ~failed
            self.geometry['geometry_eligible'][ids] &= ~failed
            weights = torch.where(failed, 0., weights)
        raise ValueError('geometry admission failed to converge within six target removals')


class AcquisitionTeacherReferenceMPC(PressureTeacherReferenceMPC):
    """Explicit finite-cylinder training expert; original profiles remain intact.

    Geometry dimensions must come from the caller's verified native object
    configuration. An injected atlas is useful for independent CPU replay;
    normal runtime builds its atlas from actual source collision hulls.
    """
    def __init__(self, env, source_sha, reference_actions, *, grasp_styles,
                 cylinder_radius_m, cylinder_length_m, pad_atlas=None, trust_radius_rad=.001):
        for name,value in (('cylinder_radius_m',cylinder_radius_m),('cylinder_length_m',cylinder_length_m)):
            if type(value) not in (float,int) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be explicitly finite and positive')
        super().__init__(env,source_sha,reference_actions,grasp_styles=grasp_styles,trust_radius_rad=trust_radius_rad)
        atlas = build_pad_atlas(env.world) if pad_atlas is None else pad_atlas
        self.native = _AcquisitionNativeSurfaceMPC(env,self.trust_radius_rad,self.native.tripod,
            atlas,float(cylinder_radius_m),float(cylinder_length_m))

    @torch.no_grad()
    def refine(self, reference_actions, *, context=None, alive=None):
        issued, diagnostics = super().refine(reference_actions,context=context,alive=alive)
        native = self.native
        for index,row in enumerate(diagnostics['per_env']):
            if not bool(native.tripod[index]): continue
            current = bool(native.geometry_prediction_current[index] & native.pending[index]
                           & native.geometry['geometry_eligible'][index].any())
            if current:
                # The first reported point prediction follows the actual
                # encoded command, including its floating-point conversion.
                model = native.geometry_models[index]
                dq = native.last_prediction[index:index+1] - model['q']
                predicted = model['points'] + torch.einsum('nhdcj,nhj->nhdc',model['jacobian'],dq)
                predicted += model['wrist_motion'][:,0]
                ids = torch.tensor([index],device=native.tripod.device)
                native.geometry_predicted_gap_m[index,0] = native._predicted_gap(predicted[:,None],ids)[0][0,0]
            row.update(profile='pressure_geometry_acquire',
                geometry_eligible=_json_value(native.geometry['geometry_eligible'][index]),
                geometry_gap_m=_json_value(native.geometry['gap_m'][index]),
                geometry_facing_cos=_json_value(native.geometry['facing_cos'][index]),
                geometry_reason_code=_json_value(native.geometry['reason_code'][index]),
                geometry_reference_points_w_m=_json_value(native.cad_points[index]),
                geometry_reference_outward_w=_json_value(native.cad_outward[index]),
                geometry_target_points_w_m=_json_value(native.geometry['target_points'][index]),
                geometry_requested_step_m=_json_value(native.geometry['requested_distance_m'][index]),
                geometry_requested_approach_m=_json_value(native.requested_approach[index]),
                geometry_request_time_s=_json_value(native.request_time[index]),
                geometry_no_progress_s=_json_value(native.no_progress[index]),
                geometry_stop_code=_json_value(native.stop_code[index]),
                geometry_blocked_hands=_json_value(native.geometry_blocked_hands[index]),
                geometry_model_infeasible=_json_value(native.geometry_model_infeasible[index]),
                geometry_sensor_invalid=bool(native.geometry_sensor_invalid[index]),
                geometry_prediction_current=current,
                geometry_predicted_gap_valid=_json_value(native.geometry['geometry_eligible'][index]
                    & current),
                geometry_predicted_gap_m=_json_value(native.geometry_predicted_gap_m[index]) if current else None)
            if row['reason'] not in ('inactive_episode','invalid_absolute_command'):
                if bool(native.geometry_sensor_invalid[index]): row['reason'] = 'invalid_geometry_sensor_hold'
                elif current and bool(native.geometry['geometry_eligible'][index].any()):
                    row['reason'] = 'optimized_pressure_geometry_acquisition'
                elif bool(native.geometry_model_infeasible[index].any()): row['reason'] = 'geometry_model_infeasible_hold_or_pressure_relief'
        diagnostics.update(version='dh116-vision-teacher-geometric-acquisition-mpc-v1',
            profile='pressure_geometry_acquire', geometry_points_kind='CAD_FK_geometry_only_not_measured_contact',
            geometry_reason_codes=native.geometry['reason_codes'],geometry_stop_codes=STOP_CODES.copy(),
            geometry_validity_sources=native.geometry_validity_sources,
            geometry_dimensions_m=dict(radius=native.cylinder_radius_m,length=native.cylinder_length_m),
            geometry_request_limits=dict(speed_m_s=.001,cumulative_distance_m=.001,
                cumulative_request_time_s=1.,no_progress_request_time_s=.5,progress_resolution_m=1e-6),
            geometry_budget_policy='one cumulative attempt per pad per episode; bearing intervals pause, never replenish',
            geometry_target_frame='current_measured_object_sidewall_world_frame;no_commanded_wrist_transport_added',
            geometry_prediction='local_point_J_with_fixed_current_CAD_normal_and_object_pose;no_force_or_object_dynamics',
            pressure_fallback='hold_missing_required_tripod_digits;original_pressure_relief_for_measured_bearing;source_velocity_only_other_digits',
            wrist_intent='bitwise_unchanged_teacher_reference',force_prediction=False)
        return issued, diagnostics
