"""Opt-in constrained geometry expert, separate from frozen acquisition v1.

Only the twelve finger commands are optimized. The inherited CAD eligibility,
pressure references, Teacher wrist actions, budgets and permanent rejection
policy are unchanged. A feasible local trajectory is not a contact prediction.
"""
from __future__ import annotations

import torch

from .linear_coupling import active_limits
from .tabletop_effective_pad import build_pad_atlas
from .tabletop_geometric_constrained_mpc import GeometricConstrainedMPC
from .tabletop_vision_teacher_acquisition_mpc import (
    AcquisitionTeacherReferenceMPC, _AcquisitionNativeSurfaceMPC, STOP_CODES,
)
from .tabletop_vision_teacher_mpc import _json_value


class _RecordedGeometricConstrainedMPC(GeometricConstrainedMPC):
    """Preserve all inputs and full-horizon outputs for independent replay."""
    def __init__(self):
        super().__init__()
        self.calls = []
        self.last_inputs = None

    def plan(self, **kwargs):
        context = self.geometry_context
        self.last_inputs = kwargs
        if context is None:
            return super().plan(**kwargs)
        record = dict(planner_inputs=_json_value(kwargs),
            geometry_context=_json_value(context),
            effective_geometry_mask=_json_value(context['geometry_mask'] & (kwargs['weights'] > 0)))
        try:
            target, diagnostics = super().plan(**kwargs)
        except (ValueError, RuntimeError) as error:
            record['error'] = str(error)
            self.calls.append(record)
            raise
        record.update(returned_first_command_rad=_json_value(target),
            planner_diagnostics=_json_value(diagnostics))
        self.calls.append(record)
        return target, diagnostics


class _ConstrainedAcquisitionNativeSurfaceMPC(_AcquisitionNativeSurfaceMPC):
    def __init__(self, env, radius, tripod, atlas, cylinder_radius_m, cylinder_length_m):
        super().__init__(env,radius,tripod,atlas,cylinder_radius_m,cylinder_length_m)
        self.pressure_planner = _RecordedGeometricConstrainedMPC()
        self.constraint_calls = {}
        self.encoded_checks = {}
        self.stats['geometry_optimizer'] = 'bounded_scipy_constrained_feasibility_repair'

    def _prepare_pressure(self):
        super()._prepare_pressure()
        self.constraint_calls = {}
        self.encoded_checks = {}

    def _check_encoded_first(self, ids, target, diagnostics, inputs):
        """Check the command that the original float action encoder will issue."""
        index = int(ids[0])
        command = inputs['command']; q = inputs['q']
        action = ((target-command)/.005).to(self.source_horizon.dtype).clamp(-1,1)
        encoded_command = command + action.to(command)*.005
        delta = .5*(encoded_command-command)
        predicted = inputs['points'] + torch.einsum('nhdcj,nhj->nhdc',inputs['jacobian'],delta)
        predicted += inputs['wrist_motion'][:,0]
        gap,facing,side = self._predicted_gap(predicted[:,None],ids)
        gap=gap[:,0];facing=facing[:,0];side=side[:,0]
        mask = self.geometry['geometry_eligible'][ids]
        # Exact old admission is retained; the new planner itself requests
        # nonnegative gaps. No outer numerical tolerance widens any limit.
        geometry_good = (torch.isfinite(gap) & (gap >= 0.)
            & (gap < self.geometry['gap_m'][ids]-1e-10)
            & torch.isfinite(facing) & (facing >= .8) & side)
        detail = self.pressure_info[index]
        relative = predicted-inputs['points']-inputs['wrist_motion'][:,0]
        normal = (relative*detail['inward']).sum(-1)
        protected = detail['protected']
        physical_low,physical_high=(command.new_tensor(value) for value in active_limits())
        low=inputs['lower'].maximum(physical_low).maximum(q[0]-.1)
        high=inputs['upper'].minimum(physical_high).minimum(q[0]+.1)
        trajectory=diagnostics.get('trajectory')
        checks=dict(finite=bool(torch.isfinite(encoded_command).all() & torch.isfinite(predicted).all()),
            planned_first_matches_returned=bool(isinstance(trajectory,torch.Tensor)
                and trajectory.shape[:2]==(len(ids),self.horizon)
                and torch.equal(target,trajectory[:,0])),
            physical_and_lead_bounds=bool(((encoded_command>=low)&(encoded_command<=high)).all()),
            canonical_slew=bool(torch.isfinite(action).all() & (action.abs()<=1.).all()),
            geometry=bool((geometry_good|~mask).all()),
            protected_pressure=bool((torch.isfinite(normal)&((normal<=1e-10)|~protected)).all()))
        passed=all(checks.values())
        self.encoded_checks[index]=dict(passed=passed,checks=checks,
            command_rad=_json_value(encoded_command[0]),action_12=_json_value(action.flatten(1)[0]),
            predicted_joints_rad=_json_value((q+delta)[0]),
            predicted_points_w_m=_json_value(predicted[0]),
            gap_m=_json_value(gap[0]),facing_cos=_json_value(facing[0]),sidewall=_json_value(side[0]),
            geometry_mask=_json_value(mask[0]),pressure_protected=_json_value(protected[0]),
            pressure_relative_normal_m=_json_value(normal[0]))
        if not passed:
            failed=mask&~geometry_good
            self.geometry_model_infeasible[ids] |= failed
            self.stop_code[ids]=torch.where(failed,STOP_CODES['model_infeasible'],self.stop_code[ids])
            self.prior_request[ids] &= ~failed
            self.geometry['geometry_eligible'][ids] &= ~failed
            self.pressure_reasons[index]='constrained_encoded_first_step_rejected'
            raise ValueError('constrained encoded first command failed original bounds or exact admission')

    def _plan(self, ids, **kwargs):
        if not bool(self.tripod[ids].any()) or not bool(self.geometry['geometry_eligible'][ids].any()):
            return super()._plan(ids,**kwargs)
        index=int(ids[0])
        detail=self.pressure_info[index]
        context=dict(geometry_mask=self.geometry['geometry_eligible'][ids].clone(),
            object_pose=self.object_pose[ids],cad_outward=self.cad_outward[ids],
            cylinder_radius_m=self.cylinder_radius_m,cylinder_length_m=self.cylinder_length_m,
            pressure_inward=detail['inward'],pressure_protected=detail['protected'])
        planner=self.pressure_planner
        planner.calls=[]
        planner.set_geometry_context(context)
        try:
            # Inherited acquisition owns the unchanged seven-solve removal and
            # permanent code4 policy. Removed pads have weight0 on re-solves.
            target,diagnostics=super()._plan(ids,**kwargs)
            self._check_encoded_first(ids,target,diagnostics,planner.last_inputs)
            return target,diagnostics
        finally:
            self.constraint_calls[index]=planner.calls
            planner.set_geometry_context(None)


class ConstrainedAcquisitionTeacherReferenceMPC(AcquisitionTeacherReferenceMPC):
    """Explicit training-expert profile; historical constructors are unchanged."""
    def __init__(self, env, source_sha, reference_actions, *, grasp_styles,
                 cylinder_radius_m, cylinder_length_m, pad_atlas=None,trust_radius_rad=.001):
        atlas=build_pad_atlas(env.world) if pad_atlas is None else pad_atlas
        super().__init__(env,source_sha,reference_actions,grasp_styles=grasp_styles,
            cylinder_radius_m=cylinder_radius_m,cylinder_length_m=cylinder_length_m,
            pad_atlas=atlas,trust_radius_rad=trust_radius_rad)
        self.native=_ConstrainedAcquisitionNativeSurfaceMPC(env,self.trust_radius_rad,self.native.tripod,
            atlas,float(cylinder_radius_m),float(cylinder_length_m))

    @torch.no_grad()
    def refine(self, reference_actions, *, context=None,alive=None):
        issued,diagnostics=super().refine(reference_actions,context=context,alive=alive)
        for index,row in enumerate(diagnostics['per_env']):
            if not bool(self.native.tripod[index]):continue
            check=self.native.encoded_checks.get(index)
            row.update(profile='pressure_geometry_constrained',
                constrained_solver_calls=self.native.constraint_calls.get(index,[]),
                constrained_encoded_first_check=check)
            if check is not None and not check['passed'] and row['reason'] not in ('inactive_episode','invalid_absolute_command'):
                row['reason']='constrained_encoded_first_step_rejected'
        diagnostics.update(version='dh116-vision-teacher-geometric-constrained-mpc-v1',
            profile='pressure_geometry_constrained',
            geometry_solver_scope='CPU bounded constrained local point kinematics;12 finger targets;H4',
            geometry_cost_interpretation='feasibility repair may cost more than rejected soft candidate;cost is not physical success',
            geometry_fallback_policy='unchanged acquisition_v1 permanent code4 and cumulative budgets',
            geometry_replay_provenance='all solver calls include full inputs,geometry context,H trajectory,status and costs',
            geometry_encoded_action_validation='first command re-encoded to source action dtype then exact geometry/pressure/bounds checked',
            force_prediction=False)
        return issued,diagnostics
