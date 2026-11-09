"""Opt-in bounded stage-A recovery, isolated from PPO and published experts.

Actual SDK evidence establishes contact. Geometry only proposes or rejects
motion. Original control/transport/physical guards stay authoritative. A failed
stage never enables subsequent experiments or Student data qualification.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from .tabletop_tripod_contact_recovery import TripodContactRecovery
from .tabletop_tripod_support_diagnostics import TripodSupportDiagnostics, _plain
from .tabletop_tripod_native_evidence import TripodNativeEvidenceBuilder
from .tabletop_tripod_nonlinear_geometry import (GeometryThresholds,
    TripodNonlinearGeometry, audit_pad_region)
from .tabletop_tripod_candidate_planner import TripodCandidatePlanner
from .tabletop_tripod_bilateral_recovery import (HAND_DIGIT_ACTIVE_INDICES,
    recovery_scope, BilateralTripodCandidatePlanner, HandTripodPressureHold)

PREPARATION_TICKS = 437
DT = 1 / 480


def validate_staged_options(args):
    stage=getattr(args,'tripod_recovery_stage',None)
    hands=getattr(args,'tripod_recovery_hands','right')
    if (stage not in ('A','B') or (stage=='A' and hands!='right')
            or (stage=='B' and hands not in ('left','both'))
            or not getattr(args,'vision_teacher_source',None)
            or getattr(args,'vision_teacher_mpc',False) is not True
            or getattr(args,'vision_teacher_mpc_profile',None) != 'pressure_reference'
            or getattr(args,'vision_student',False) is not True
            or getattr(args,'vision_action_distribution',None) != 'tanh_gaussian'
            or getattr(args,'vision_camera_mode',None) != 'fixed'
            or getattr(args,'num_envs',None) not in (2,4,8,16)
            or not 438 <= getattr(args,'vision_teacher_max_control_steps',0) <= 1200
            or getattr(args,'resume',None) or getattr(args,'vision_evaluation_checkpoint',None)
            or getattr(args,'surface_mpc',False)
            or getattr(args,'vision_lift_reward_experiment','R0') != 'R0'):
        raise ValueError('fixed-wrist A/right or explicit B/left|both requires isolated fixed-camera Teacher preparation, bounded 438..1200 controls; no PPO/resume/moving stage')


def commit_submitted_targets(ledger, pending, world):
    """Called only AFTER both original hand.write_data_to_sim() return.

    Verify exact retained logical commands AND encoded SDK targets before
    charging the pre-reserved command arc length. This is not a candidate or
    measured displacement charge. Physics time is charged by the next sample.
    """
    actual=world.commands.reshape(-1,12)
    expected=pending['final_target_rad']
    live=pending.get('native_live_mask',torch.ones(len(actual),device=actual.device,dtype=torch.bool))
    if (not isinstance(live,torch.Tensor) or live.shape!=(len(actual),)
            or live.dtype!=torch.bool or live.device!=actual.device
            or bool((pending['executed_mask'].any((1,2))&~live).any())):
        raise ValueError('native submission live scope mismatch')
    if (world._physics_steps + 1 != pending['physics_step']
            or actual.shape != expected.shape or actual.dtype != expected.dtype
            or actual.device != expected.device or not torch.equal(actual[live],expected[live])
            or not torch.equal(actual[live].float(),expected[live].float())):
        raise ValueError('native submission identity/retained or encoded command mismatch')
    payload={k:v for k,v in pending.items() if k!='native_live_mask'}
    # Inactive slots may already have been physically reset while TensorTask
    # retains their terminal snapshot until the control tick ends.
    payload['final_target_rad']=actual.detach().clone()
    return ledger.commit_execution(**payload)


def cancel_inapplicable_substep_actions(task,ledger,tripod,acquisition_origins):
    """Stop held acquisition at actual CONTACT, before original target guards.

    A60Hz action spans8 physical steps. Its origin does not become pressure
    control simply because a pad establishes contact halfway through it.
    Original, explicitly pressure-origin withdrawal is untouched.
    """
    # Accept the historic helper input for old Stage A callers; runtime-owned
    # state is always explicitly [N,2,3]. No left authority is inferred.
    if acquisition_origins.shape==(ledger.n,3):
        right=acquisition_origins
        acquisition_origins=torch.zeros(ledger.n,2,3,device=right.device,dtype=right.dtype)
        acquisition_origins[:,1]=right
    if (task.actions.shape!=(ledger.n,24) or tripod.shape!=(ledger.n,)
            or tripod.dtype!=torch.bool or acquisition_origins.shape!=(ledger.n,2,3)
            or acquisition_origins.dtype!=torch.bool):
        raise ValueError('canonical per-substep action lineage required')
    scope=ledger.enabled_pad_mask[:,:,:3]&tripod[:,None,None]
    stop=acquisition_origins&scope&(ledger.state[:,:,:3]!=ledger.REACQUIRING)
    # The converse transition matters too: a held pressure action cannot become
    # unbudgeted acquisition when its actual contact disappears mid-control.
    stop|=(~acquisition_origins&scope
           &~(ledger.contact_known[:,:,:3]&ledger.measured_contact[:,:,:3]))
    for hand,groups in enumerate(HAND_DIGIT_ACTIVE_INDICES):
        for digit,group in enumerate(groups):
            for j in group:task.actions[stop[:,hand,digit],j]=0.


def retire_completed_first_episode(ledger,ids,completed):
    """A newly reset slot has no second-trial recovery authorization in A.

    Its genuine new physical episode is still recorded, but cannot acquire a
    reservation or masquerade as the completed first episode. Siblings retain
    their own state/budgets. This disables authority, never replenishes it.
    """
    ids=torch.as_tensor(ids,device=ledger.device,dtype=torch.long)
    if completed.shape!=(ledger.n,) or completed.dtype!=torch.bool or completed.device!=ledger.device:
        raise ValueError('completed first-episode authorization mask required')
    ledger.enabled_pad_mask[ids[completed[ids]]]=False


def _safe_json(value):
    tags={}; result=_plain(value,nonfinite=tags)
    return dict(value=result,nonfinite_originals=tags) if tags else result


def _thresholds():
    values=dict(contact_distance_m=.001,normal_cosine_min=.8,object_overlap_m=.0006,
        noncontact_object_clearance_m=0.,table_clearance_m=.0003,
        self_clearance_m=.0003,interhand_clearance_m=.0003,obstacle_clearance_m=.0003,
        max_command_step_rad=.005,command_lead_rad=.1)
    sources={k:'tabletop_centerline_grasp.DEFAULT_THRESHOLDS (source-geometry policy, not physics)' for k in values}
    sources.update(normal_cosine_min='original effective-pad .8 gate',
        object_overlap_m='new source-envelope policy using existing pressure retreat .0006; NOT native depth equivalence',
        table_clearance_m='original native pinch transport .0003 floor',
        max_command_step_rad='original24D .3rad/s divided by60; planner separately checks480Hz slew',
        command_lead_rad='original tabletop_free_wrist_grasp.CONTRACT .1rad')
    return GeometryThresholds(**values,sources=sources)


class TripodRecoveryRuntime:
    """One first native episode per slot; right-only stage A authority.

    A100ms settling observation is separate from REACQUIRING attempt time.
    It grants no new distance/time budget. A truly moving wrist never passes
    the fixed-wrist geometry admission test merely because its command is0.
    """
    def __init__(self,env,models,source_sha,output,*,recovery_hands='right',recovery_stage=None):
        self.env,self.w,self.t=env,env.world,env.task
        self.n,self.device=self.w.n,self.w.device
        self.models=models;self.source_sha=source_sha
        self.tick=0;self.active=False;self.pending=None;self.last_batch=None
        self.episode_ids=torch.full((self.n,),-1,device=self.device,dtype=torch.long)
        self.ended=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        styles=self.w.mixed_reset_style_ids
        if styles.shape!=(self.n,) or styles.dtype!=torch.long:
            raise ValueError('actual mixed reset identity required')
        self.tripod=styles==1
        enabled,self.finger_mask=recovery_scope(styles,recovery_hands)
        self.recovery_hands=recovery_hands
        self.stage=('A' if recovery_hands=='right' else 'B') if recovery_stage is None else recovery_stage
        if self.stage not in ('A','B') or (self.stage=='A' and recovery_hands!='right'):
            raise ValueError('legacy Stage A is right-only; bilateral fixed-wrist candidate uses explicit Stage B')
        self.ledger=TripodContactRecovery(self.n,self.device,enabled_pad_mask=enabled)
        self.builder=TripodNativeEvidenceBuilder(self.w,source_models=models)
        surfaces=['Table']+[s['name'] for s in (self.w.w.end_support_scene or {}).get('supports',[])]
        self.output=Path(output)
        self.physics=self.output.joinpath('tripod_physics_substeps.jsonl').open('x',encoding='utf-8')
        self.raw_physics=self.output.joinpath('tripod_raw_sdk_substeps.jsonl').open('x',encoding='utf-8')
        self.events=self.output.joinpath('tripod_contact_events.jsonl').open('x',encoding='utf-8')
        self.decisions=self.output.joinpath('tripod_geometry_decisions.jsonl').open('x',encoding='utf-8')
        self.fk_trace=self.output.joinpath('native_full_joint_fk_comparisons.jsonl').open('x',encoding='utf-8')
        self.diag=TripodSupportDiagnostics(self.output.name,sink=self._write_physics,
            sensor_bindings=self.builder.bindings,passive_surface_ids=surfaces)
        self.event_cursor=0;self.planners={};self.plans={};self.latest_rows={}
        self.requested_actions=torch.zeros(self.n,24,device=self.device)
        self.control_issued_actions=self.requested_actions.clone()
        self.upstream_pressure_actions=self.requested_actions.clone()
        self.pressure_controllers={};self.pressure_step_proposals=set()
        self.acquisition_origins=torch.zeros(self.n,2,3,device=self.device,dtype=torch.bool)
        self.substep_action_origin=torch.zeros(self.n,2,3,device=self.device,dtype=torch.long)
        self.last_before_step_tick=None;self.previous_fresh_mask=torch.zeros_like(self.acquisition_origins)
        self.pressure_mandatory_mask=torch.zeros_like(self.acquisition_origins)
        self.executions=0;self.fixed_start=None;self.fixed_palms=None;self.fixed_angles=None
        self.stage_outcomes={};self.geometry_blocked={};self.last_capture_step=None
        self.terminal_ledgers={};self._step_live=None
        self.fk_summary=dict(pair_count=0,body_comparisons=0,max_translation_error_m=0.,max_rotation_error_rad=0.)
        self.fixed_offset=None;self.start_positions=None;self.fixed_left_commands=None
        self.env.control_probe=self
        self.w.tripod_recovery_runtime=self;self.t.tripod_recovery_runtime=self
        self.t.tripod_recovery_stage=self.stage;self.w.capture_tripod_contact_evidence=True

    def _write_physics(self,row):
        row['experiment_phase']='fixed_wrist_'+self.stage if self.active else 'published_pressure_preparation'
        row['counts_as_student_success']=False
        self.physics.write(json.dumps(_safe_json(row),allow_nan=False,separators=(',',':'))+'\n')

    def _drain(self):
        for event in self.ledger.events[self.event_cursor:]:
            self.events.write(json.dumps(_safe_json(event),allow_nan=False,separators=(',',':'))+'\n')
        self.event_cursor=len(self.ledger.events)

    def _capture(self,live,*,fatal=None,event_kind='physics'):
        false=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        substep=0 if self.last_capture_step is None else int(self.env.substep)
        return self.builder.capture(episode_ids=self.episode_ids,stage_id=self.stage,
            control_step=self.tick,control_substep=substep if event_kind=='reset' else int(self.env.substep),
            live=live,terminal=(self.t.done&live) if event_kind=='physics' else false,
            fatal=false if fatal is None else fatal,native_success=self.t.success&live,
            wrists_fixed=self.active,left_targets_fixed=self.active and self.recovery_hands=='right',
            object_external_force_applied=False,induced_disconnect=False,event_kind=event_kind)

    def after_physical_reset(self,ids,measurement):
        self._drain()
        if self.pending is not None:
            raise RuntimeError('physical reset encountered unsettled approach reservation')
        self.episode_ids[ids]+=1
        self.ledger.reset(ids,episode_ids=self.episode_ids,source_sha=self.source_sha)
        retire_completed_first_episode(self.ledger,ids,self.ended)
        self.event_cursor=len(self.ledger.events)
        batch=self._capture(torch.zeros(self.n,device=self.device,dtype=torch.bool),event_kind='reset')
        for index in ids.tolist():
            batch['packets'][index]['automatic_recovery_authority']=self.ledger.enabled_pad_mask[index].clone()
            self.diag.observe(batch['packets'][index])
        self.last_batch=batch

    def before_step(self):
        if self.pending is not None:raise RuntimeError('unsubmitted approach reservation')
        self._step_live=(~self.t.done&~self.ended).clone()
        if self.active:
            previous=getattr(self,'substep_action_origin',torch.zeros_like(self.acquisition_origins,dtype=torch.long))
            # Within the same held60Hz action, a fresh proposal is a one-physics-
            # substep permission. Revoke it before attempting a new model. At a
            # new control begin_step has installed a new upstream action.
            same_control=getattr(self,'last_before_step_tick',None)==self.tick
            self.previous_fresh_mask=(previous==2)&self.tripod[:,None,None]&same_control
            mandatory=getattr(self,'pressure_mandatory_mask',torch.zeros_like(self.acquisition_origins))
            known_bearing=self.ledger.contact_known[:,:,:3]&self.ledger.measured_contact[:,:,:3]
            if bool((self.previous_fresh_mask&mandatory&~known_bearing&~self.ended[:,None,None]).any()):
                raise RuntimeError('fresh mandatory pressure relief lost actual contact admission; stopped before SDK submission')
            for hand,groups in enumerate(HAND_DIGIT_ACTIVE_INDICES):
                for digit,group in enumerate(groups):
                    for j in group:self.t.actions[self.previous_fresh_mask[:,hand,digit],j]=0.
            self.substep_action_origin=torch.where(self.acquisition_origins,
                torch.where(self.ledger.state[:,:,:3]==self.ledger.REACQUIRING,1,3),0)
            self.substep_action_origin[self.previous_fresh_mask]=3
            cancel_inapplicable_substep_actions(self.t,self.ledger,self.tripod,self.acquisition_origins)
            self.pressure_step_proposals=set()
            self._fresh_pressure_substep()
        self.last_before_step_tick=self.tick
        self.requested_actions=self.t.actions.clone()

    def _fresh_pressure_substep(self):
        from .tabletop_tripod_pressure_hold import TripodPressureHold
        m=self.w.pad_measurements
        for index in (self.tripod&~self.ended).nonzero().flatten().tolist():
            if not self._settled(index):
                mandatory=getattr(self,'pressure_mandatory_mask',torch.zeros_like(self.acquisition_origins))
                if bool((self.previous_fresh_mask[index]&mandatory[index]
                         &self.ledger.measured_contact[index,:,:3]).any()):
                    raise RuntimeError('fresh mandatory pressure relief lost model admission; stage A stopped before SDK submission')
                continue
            for hand,groups in enumerate(HAND_DIGIT_ACTIVE_INDICES):
                bearing=(self.ledger.contact_known[index,hand,:3]&self.ledger.measured_contact[index,hand,:3]
                    &self.ledger.enabled_pad_mask[index,hand,:3])
                if not bool(bearing.any()):continue
                planner=self._planner(index)
                key=index if getattr(self,'recovery_hands','right')=='right' else (index,hand)
                if key not in self.pressure_controllers:
                    self.pressure_controllers[key]=(TripodPressureHold(planner) if hand==1 else HandTripodPressureHold(planner,hand))
                plan=self.pressure_controllers[key].plan(**self._contexts(index),
                    current_command=self.t.command[index].reshape(2,6),
                    effective_pad_loads_n=m['effective_pad_loads_n'][index],
                    effective_pad_vectors_w=m['effective_pad_vectors_w'][index],
                    effective_pad_points_w_m=m['effective_pad_points_w'][index],
                    effective_pad_valid=m['effective_pad_valid'][index],
                    effective_pad_depth_m=m['effective_pad_depth_m'][index],
                    rejected_contact_loads_n=m['rejected_contact_loads_n'][index],
                    upstream_pressure_action24=self.upstream_pressure_actions[index],substeps=1)
                self.decisions.write(json.dumps(_safe_json(dict(event_kind='fresh_pressure_proposal',hand=hand,
                    env_id=index,episode_id=int(self.episode_ids[index]),physics_step=self.w._physics_steps,
                    control_tick=self.tick,control_substep=int(self.env.substep),plan=plan)),
                    allow_nan=False,separators=(',',':'))+'\n')
                if any(r['relief_blocked'] for r in plan['per_pad']):
                    raise RuntimeError('mandatory pressure relief not certified; stage stopped before SDK submission')
                if not hasattr(self,'pressure_mandatory_mask'):
                    self.pressure_mandatory_mask=torch.zeros_like(self.acquisition_origins)
                self.pressure_mandatory_mask[index,hand]=torch.tensor(
                    [r['mandatory_relief'] for r in plan['per_pad']],device=self.device,dtype=torch.bool)
                for digit,group in enumerate(groups):
                    if bool(bearing[digit]):
                        self.t.actions[index,list(group)]=torch.as_tensor(
                            plan['action24'][list(group)],device=self.device,dtype=self.t.actions.dtype)
                        self.substep_action_origin[index,hand,digit]=2 if plan['accepted'] else 3
                if plan['accepted']:self.pressure_step_proposals.add(index)
                if plan['reason']=='ACTUAL_SOURCE_GEOMETRY_UNCERTIFIED':
                    self.geometry_blocked[index]=plan.get('rejection_reasons',['PRESSURE_SOURCE_GEOMETRY_UNCERTIFIED'])

    def _contexts(self,index):
        p=self.last_batch['packets'][index];snap=self.last_batch['geometry_snapshots'][index]
        if any(s is None for s in snap):raise ValueError('incomplete actual SDK source frame snapshot')
        k=p['actual_kinematics']
        identity=dict(env_id=index,episode_id=int(self.episode_ids[index]),
            physics_step=p['tensor_world_physics_steps'],source_snapshot_id=k['snapshot_identity'])
        qd=np.asarray([h['full_joint_velocities_rad_s'] for h in k['hands']])
        for h,s in enumerate(snap):
            s['measurement_identity']=dict(identity);s['full_qd']=qd[h].copy()
        bearing=self.w.pad_measurements['effective_pad_valid'][index]&(
            self.w.pad_measurements['effective_pad_loads_n'][index]>.05)
        provenance=dict(source='same-clock original native effective-pad reducer',
            physics_step=identity['physics_step'],snapshot_identity=identity['source_snapshot_id'],
            native_accepted_contacts=True)
        return dict(snapshots=snap,full_qd=qd,snapshot_identity=identity,
            bearing_mask=bearing,bearing_provenance=provenance)

    def _scene(self,index):
        w=self.w;origin=w.origins[index].detach().cpu().numpy();layout=w.w.layout
        obstacles=[]
        for item in (w.w.end_support_scene or {}).get('supports',[]):
            c=np.asarray(item['center_m'])+origin;s=np.asarray(item['size_m'])/2
            vertices=np.asarray([c+s*np.asarray(sign) for sign in itertools.product((-1,1),repeat=3)])
            obstacles.append(dict(id=item['name'],vertices_w_m=vertices))
        table_top=float(layout['table_center'][2]+layout['table_size'][2]/2+origin[2])
        return dict(rod=dict(pose_w=torch.cat((w.wp[index],w.wq[index])).detach().cpu().numpy(),
                radius_m=float(w.w.args.solid_cylinder_radius_m),length_m=.5),
            table_top_z_m=table_top,obstacles=obstacles,obstacles_complete=True,
            provenance='original GpuWorld layout/end_support_scene plus actual clone origin; original scene verification required by runner')

    def _planner(self,index):
        if index not in self.planners:
            limits=self.w.limits[index].detach().cpu().numpy()
            full=np.asarray([limits[h,self.builder.joint_ids[h]] for h in range(2)])
            geometry=TripodNonlinearGeometry(*self.models,self._scene(index),_thresholds())
            planner_type=TripodCandidatePlanner if getattr(self,'recovery_hands','right')=='right' else BilateralTripodCandidatePlanner
            self.planners[index]=planner_type(geometry,
                active_lower=self.t.low.reshape(2,6),active_upper=self.t.high.reshape(2,6),
                full_lower=full[...,0],full_upper=full[...,1],max_candidates=1,
                limit_sources=dict(active='unchanged task.low/high',full='actual TensorWorld original SDK limits by JOINTS',
                    lead='original .1rad',command='original .3rad/s per480Hz',
                    response='existing MPC .5/control approximation; not native dynamics validation'))
        else:self.planners[index].geometry.scene['rod']=self._scene(index)['rod']
        return self.planners[index]

    def start_recovery(self):
        if self.active or self.tick!=PREPARATION_TICKS or self.last_batch is None:
            raise ValueError('stage A begins once after the exact published pressure preparation prefix')
        self.active=True;self.fixed_start=float(self.w.w.sim.current_time)
        self.fixed_palms=self.t.m['adaptive_palm_offset'].clone()
        self.fixed_angles=self.t.m['wrist_joint_angles'].clone()
        self.fixed_offset=self.t.offset.clone();self.start_positions=self.w.roots.clone()
        self.fixed_left_commands=self.t.command[:,:6].clone()
        self.fixed_commands=self.t.command.clone()
        # A new stage reference does not reset physics, physical episode IDs,
        # any recovery budget, native success or reset-based net lift.
        for index in self.tripod.nonzero().flatten().tolist():
            self.diag.begin_stage(self.last_batch['packets'][index])
        self.physics.flush();self.events.flush()

    def _settled(self,index):
        if float(self.w.w.sim.current_time)-self.fixed_start<.1:return False
        velocity=self.t.m.get('wrist_velocity')
        return (isinstance(velocity,torch.Tensor) and bool(torch.isfinite(velocity[index]).all())
            and bool((velocity[index,:,:3].norm(dim=-1)<=.0001).all())
            and bool((velocity[index,:,3:].norm(dim=-1)<=.001).all())
            and bool(torch.isfinite(self.w.wv[index]).all())
            and bool(self.w.wv[index,:3].norm()<=.0001)
            and bool(self.w.wv[index,3:].norm()<=.001)
            and bool(((self.w.roots[index,:,:3]-self.start_positions[index,:,:3]).norm(dim=-1)<=.0005).all()))

    def refine(self,pressure_actions):
        issued=pressure_actions.clone();self.plans={};self.acquisition_origins.zero_()
        self.upstream_pressure_actions=pressure_actions.clone()
        if not self.active:
            self.control_issued_actions=issued.clone();self.requested_actions=issued.clone();return issued
        hands=getattr(self,'recovery_hands','right')
        if hands=='right':issued[self.tripod,:6]=0.
        elif hands=='left':issued[self.tripod,6:12]=0.
        issued[self.tripod,12:]=0.
        eligible=torch.zeros_like(self.ledger.enabled_pad_mask)
        for index in (self.tripod&~self.ended).nonzero().flatten().tolist():
            # Existing bearing-pad pressure actions, including withdrawal, are
            # retained. Only the missing digit's source velocity is replaced.
            missing=(self.ledger.contact_known[index,:,:3]&~self.ledger.measured_contact[index,:,:3]
                &self.ledger.enabled_pad_mask[index,:,:3])
            for hand,groups in enumerate(HAND_DIGIT_ACTIVE_INDICES):
                for digit,group in enumerate(groups):
                    if bool(missing[hand,digit]):issued[index,list(group)]=0.
            if not self._settled(index):continue
            if not bool(missing.any()):continue
            planner=self._planner(index);context=self._contexts(index)
            kwargs=(dict(missing_right=missing[1].detach().cpu().numpy(),
                remaining_approach_m=(.001-self.ledger.executed_approach_m[index,1,:3]).clamp_min(0.))
                if hands=='right' else dict(missing_hands=missing.detach().cpu().numpy(),
                    remaining_approach_m=(.001-self.ledger.executed_approach_m[index,:,:3]).clamp_min(0.)))
            plan=planner.plan(**context,current_command=self.t.command[index].reshape(2,6),**kwargs,substeps=8)
            self.plans[index]=plan
            self.decisions.write(json.dumps(_safe_json(dict(env_id=index,episode_id=int(self.episode_ids[index]),
                tick=self.tick,plan=plan)),allow_nan=False,separators=(',',':'))+'\n')
            if plan['accepted']:
                for hand,groups in enumerate(HAND_DIGIT_ACTIVE_INDICES):
                    for digit,group in enumerate(groups):
                        if bool(missing[hand,digit]):
                            issued[index,list(group)]=torch.as_tensor(plan['action24'][list(group)],device=self.device)
                            eligible[index,hand,digit]=True
                            self.acquisition_origins[index,hand,digit]=True
            elif plan['reason']=='ACTUAL_SOURCE_GEOMETRY_UNCERTIFIED':
                self.geometry_blocked[index]=plan['rejection_reasons']
        self.ledger.request_retry(episode_ids=self.episode_ids,control_tick=self.tick,
            observation_revision=self.w._physics_steps,geometry_eligible=eligible)
        self.control_issued_actions=issued.clone();self.requested_actions=issued.clone();return issued

    def filter_post_guard_targets(self,packet):
        if not self.active:return packet['governed_target_rad']
        result=packet['governed_target_rad'].clone();request=torch.zeros_like(self.ledger.executed_approach_m)
        selected=self.tripod&packet['live']&~self.ended
        hands=getattr(self,'recovery_hands','right')
        if hands=='right' and self.fixed_left_commands is not None and not torch.equal(
                result[selected,:6],self.fixed_left_commands[selected]):
            raise RuntimeError('stage A left target changed by native guard; fixed-target experiment stopped')
        if hands=='left' and not torch.equal(result[selected,6:12],self.fixed_commands[selected,6:12]):
            raise RuntimeError('fixed non-recovery right target changed by native guard')
        for index in (self.tripod&packet['live']&~self.ended).nonzero().flatten().tolist():
            mask=packet['finger_mask'][index]
            # An exhausted/stopped/unknown missing pad receives no automatic
            # approach. Original bearing pressure cannot be changed by mask.
            active=(self.ledger.state[index,:,:3]==self.ledger.REACQUIRING)&self.ledger.enabled_pad_mask[index,:,:3]
            for hand,groups in enumerate(HAND_DIGIT_ACTIVE_INDICES):
                for digit,group in enumerate(groups):
                    if not bool(active[hand,digit]):
                        for j in group:
                            if bool(mask[j]):result[index,j]=packet['command_rad'][index,j]
            changed=mask&(result[index]!=packet['command_rad'][index])
            pressure_proposal=index in self.pressure_step_proposals
            if not bool(changed.any()) and not pressure_proposal:continue
            planner=self._planner(index)
            audit=planner.audit_commands(**self._contexts(index),
                current_command=packet['command_rad'][index].reshape(2,6),
                final_logical_command=result[index].reshape(2,6),final_next_command=result[index].reshape(2,6).float())
            if 'bilateral_approach_command_m' in audit:
                amounts=torch.as_tensor(audit['bilateral_approach_command_m'],device=self.device,dtype=torch.float64)
            else:
                amounts=torch.zeros(2,3,device=self.device,dtype=torch.float64)
                amounts[1]=torch.as_tensor(audit['approach_command_m'],device=self.device,dtype=torch.float64)
            amounts=torch.where(active,amounts,0.)
            room=torch.minimum((.001-self.ledger.executed_approach_m[index,:,:3]).clamp_min(0.),
                amounts.new_full((2,3),.001*DT))
            approved=bool(audit['accepted']) and bool((amounts<=room+1e-15).all())
            if pressure_proposal and not approved:
                raise RuntimeError('combined post-native-guard pressure/acquisition target not certified; no SDK submission')
            if approved:request[index,:,:3]=amounts
            else:result[index,mask]=packet['command_rad'][index,mask]
            self.decisions.write(json.dumps(_safe_json(dict(env_id=index,episode_id=int(self.episode_ids[index]),
                tick=self.tick,physics_step=self.w._physics_steps,post_guard_approved=approved,audit=audit)),
                allow_nan=False,separators=(',',':'))+'\n')
        if bool((request>0).any()):
            ticket=self.ledger.reserve_approach(episode_ids=self.episode_ids,control_tick=self.tick,dt_s=DT,
                requested_approach_m=request)
            if bool((request>ticket['allowed_distance_m']+1e-15).any()):
                self.ledger.cancel_reservation(ticket['reservation_id'])
                raise RuntimeError('post-guard command exceeds conserved reservation; no submission allowed')
            approved_actions=self.requested_actions.clone()
            delta=(result-packet['command_rad'])/(.3*DT)
            if bool((delta.abs()>1.+1e-10).any()):raise RuntimeError('original target slew mismatch')
            approved_actions[:,:12]=delta.clamp(-1.,1.).to(approved_actions)
            self.pending=dict(episode_ids=self.episode_ids.clone(),reservation_id=ticket['reservation_id'],
                execution_id=f'sdk-{self.w._physics_steps+1}',physics_step=self.w._physics_steps+1,
                approved_approach_m=request.clone(),executed_mask=request>0,
                requested_action24=self.requested_actions.clone(),final_target_rad=result.clone(),
                approved_action24=approved_actions,native_live_mask=packet['live'].clone())
        return result

    def prepare_transport(self,world,task,allowed,live):
        if not self.active:return
        selected=self.tripod&live
        if not torch.equal(task.offset[selected],self.fixed_offset[selected]):
            raise RuntimeError('stage A common lift offset changed')
        # Drive targets only. No SDK measured pose/velocity, object or forces
        # are written here. Original transport/clearance checks follow this.
        world.palms[selected]=self.fixed_palms[selected]
        task.palm_targets[selected]=self.fixed_palms[selected]
        world.wrist_angles[selected]=self.fixed_angles[selected]
        task.rotation_targets[selected]=self.fixed_angles[selected]

    def fixed_stage_qualified(self):
        """Stage B needs real stable bilateral contact AND current geometry.

        Diagnostic contact alone never admits a source-uncertified state. This
        checks the unchanged complete geometry certificate at actual snapshots,
        with zero wrist/joint increment; it grants no moving-wrist permission.
        Legacy Stage A retains its original diagnostic decision unchanged.
        """
        selected=self.tripod.nonzero().flatten().tolist()
        if not self.active or not selected:return False
        if not all(self.latest_rows.get(i,{}).get('stage_gate',{}).get('pass',False) for i in selected):return False
        if self.stage=='A':return True
        qualified=True
        for index in selected:
            row=self.latest_rows[index]
            if bool(self.ended[index]) or not self._settled(index) or row.get('six_pad_contiguous_s',0.)<.2:
                qualified=False;continue
            context=self._contexts(index);planner=self._planner(index)
            audit=planner.geometry.audit_state(context['snapshots'],np.zeros(6),left_delta=np.zeros(6),
                bearing_mask=context['bearing_mask'],bearing_provenance=context['bearing_provenance'])
            accepted=bool(audit['motion_safe'] and audit['audit_complete'])
            if not accepted:
                self.geometry_blocked[index]=audit['rejection_reasons'];qualified=False
            self.decisions.write(json.dumps(_safe_json(dict(event_kind='fixed_B_contact_geometry_qualification',
                env_id=index,episode_id=int(self.episode_ids[index]),physics_step=self.w._physics_steps,
                tick=self.tick,accepted=accepted,audit=audit,moving_wrist_permission=False)),
                allow_nan=False,separators=(',',':'))+'\n')
        return qualified

    def after_native_submission(self,world,task,live):
        if self.pending is not None:
            commit_submitted_targets(self.ledger,self.pending,world)
            self.executions+=1;self.pending=None;self._drain()

    def _finish(self,live,fatal=None):
        if self.last_capture_step==self.w._physics_steps:return
        batch=self._capture(live,fatal=fatal)
        # Persist owned raw SDK evidence before any stateful reducer can fail.
        # A fatal reducer/guard cannot replace or erase this native substep.
        self.raw_physics.write(json.dumps(_safe_json(batch['packets']),allow_nan=False,separators=(',',':'))+'\n')
        self.raw_physics.flush()
        if self.tick in (0,1,50,100,200,300,436) or self.active:
            self._compare_native_fk(batch,live)
        measurement=torch.tensor([p['measurement_valid'] for p in batch['packets']],device=self.device)
        shape=(self.n,2,5);gap=torch.full(shape,float('nan'),device=self.device,dtype=torch.float64)
        good=torch.zeros(shape,device=self.device,dtype=torch.bool)
        signature=torch.full((*shape,6),float('nan'),device=self.device,dtype=torch.float64)
        if self.active:
            for index in (self.tripod&live&~self.ended).nonzero().flatten().tolist():
                for hand,model in enumerate(self.models):
                    if not bool(self.ledger.enabled_pad_mask[index,hand,:3].any()):continue
                    snap=batch['geometry_snapshots'][index][hand]
                    if snap is None:continue
                    rod=self._scene(index)['rod']
                    for digit,name in enumerate(('thumb','index','middle')):
                        patch=model.patches[name];s=next(s for s in model.shapes if s['body']==patch['body'])
                        row=audit_pad_region(s['planes'],patch,snap['poses'][patch['body']],rod,
                            contact_distance_m=.001,normal_cosine_min=.8);witness=row.get('witness')
                        if row['complete'] and witness is not None:
                            gap[index,hand,digit]=max(0.,float(witness['distance_m']));good[index,hand,digit]=True
                            signature[index,hand,digit]=torch.as_tensor(np.r_[witness['point_w_m'],witness['normal_w']],device=self.device)
        m=self.w.pad_measurements;terminated=live&self.t.done
        if fatal is not None:terminated|=fatal
        self.ledger.observe_substep(episode_ids=self.episode_ids,physics_step=self.w._physics_steps,
            dt_s=DT,live=live&~self.ended,terminal=terminated,measurement_valid=measurement,
            pad_valid=m['effective_pad_valid'],loads_n=m['effective_pad_loads_n'],
            vectors_w=m['effective_pad_vectors_w'],depth_m=m['effective_pad_depth_m'],
            rejected_loads_n=m['rejected_contact_loads_n'],measured_gap_m=gap,geometry_valid=good,
            measured_geometry_signature=signature)
        for index,p in enumerate(batch['packets']):
            if bool(live[index]&~self.ended[index]):
                p['recovery_ledger']=dict(state=self.ledger.state[index],
                    executed_approach_m=self.ledger.executed_approach_m[index],
                    attempt_time_s=self.ledger.attempt_time_s[index],no_progress_s=self.ledger.no_progress_s[index],
                    measured_region_gap_m=self.ledger.measured_gap_m[index],
                    measured_region_witness_and_normal_w=self.ledger.geometry_signature[index],
                    geometry_valid=self.ledger.geometry_valid[index],
                    best_region_gap_m=self.ledger.best_gap_m[index],
                    round_id=self.ledger.round_id[index],round_active=self.ledger.round_active[index],
                    round_stopped=self.ledger.round_stopped[index],
                    contact_known=self.ledger.contact_known[index],measured_contact=self.ledger.measured_contact[index],
                    stable_time_s=self.ledger.stable_time_s[index],
                    reserved_approach_m=self.ledger.reserved_approach_m[index],
                    distance_exhausted=self.ledger.distance_exhausted[index],
                    time_exhausted=self.ledger.time_exhausted[index],
                    blocked_reason=self.ledger.blocked_reason[index])
                p['requested_action24_this_substep']=self.requested_actions[index]
                p['control_issued_action24']=self.control_issued_actions[index]
                p['acquisition_origin_hand_digits']=self.acquisition_origins[index]
                p['action_origin_hand_this_substep']=self.substep_action_origin[index]
                p['acquisition_origin_right_digits']=self.acquisition_origins[index,1]
                p['action_origin_right_this_substep']=self.substep_action_origin[index,1]
                p['action_origin_codes']={'legacy_pressure_or_hold':0,'acquisition':1,'fresh_pressure':2,'suppressed_hold':3}
                row=self.diag.observe(p)
                if row is not None:self.latest_rows[index]=row
                if bool(terminated[index]):
                    self.terminal_ledgers[index]=dict(episode_id=int(self.episode_ids[index]),
                        attempt_time_s=_plain(self.ledger.attempt_time_s[index]),
                        approved_approach_m=_plain(self.ledger.executed_approach_m[index]))
        self.ended|=terminated;self.last_batch=batch;self._drain()
        self.last_capture_step=self.w._physics_steps

    def after_step(self,live,previous_phase):self._finish(live)

    def _compare_native_fk(self,batch,live):
        if self.last_batch is None:return
        for index in (live&~self.ended).nonzero().flatten().tolist():
            old=self.last_batch['packets'][index];new=batch['packets'][index]
            if (old['episode_id']!=new['episode_id'] or old['tensor_world_physics_steps']+1!=new['tensor_world_physics_steps']):continue
            for hand,model in enumerate(self.models):
                previous=self.last_batch['geometry_snapshots'][index][hand]
                actual=batch['geometry_snapshots'][index][hand]
                if previous is None or actual is None:continue
                prediction=model.predict_full_delta(previous,actual['full_q']-previous['full_q'],
                    root_pose_next=actual['poses'][model.root])
                rows=[]
                for body in sorted(model.bodies):
                    p,a=prediction['poses'][body],actual['poses'][body]
                    translation=float(np.linalg.norm(p[:3,3]-a[:3,3]))
                    rotation=float(np.arccos(np.clip((np.trace(p[:3,:3].T@a[:3,:3])-1)/2,-1.,1.)))
                    rows.append(dict(body=body,translation_error_m=translation,rotation_error_rad=rotation))
                self.fk_trace.write(json.dumps(dict(env_id=index,episode_id=new['episode_id'],hand=hand,
                    from_physics_step=old['tensor_world_physics_steps'],to_physics_step=new['tensor_world_physics_steps'],
                    measured_full_joint_delta_rad=(actual['full_q']-previous['full_q']).tolist(),
                    actual_next_base_used=True,ideal_mimic_projection_used=False,
                    meaning='actual consecutive SDK frames versus full-measured-joint source FK; not native dynamics prediction',
                    bodies=rows),allow_nan=False,separators=(',',':'))+'\n')
                self.fk_summary['pair_count']+=1;self.fk_summary['body_comparisons']+=len(rows)
                self.fk_summary['max_translation_error_m']=max(self.fk_summary['max_translation_error_m'],max(r['translation_error_m'] for r in rows))
                self.fk_summary['max_rotation_error_rad']=max(self.fk_summary['max_rotation_error_rad'],max(r['rotation_error_rad'] for r in rows))

    def on_physics_failure(self,mask,measurement,context):
        self.decisions.write(json.dumps(_safe_json(dict(event_kind='native_physics_failure',context=context,
            mask=mask,physical_episode_ids=self.episode_ids,physics_step=self.w._physics_steps,
            control_tick=self.tick,control_substep=int(self.env.substep),
            reset_substeps=self.env.reset_substep,pending_reset=self.env.pending,measurement=measurement)),
            allow_nan=False,separators=(',',':'))+'\n')
        live=(~self.t.done|mask) if self._step_live is None else self._step_live|mask
        self._finish(live,fatal=mask)
        self.physics.flush();self.events.flush();self.decisions.flush()

    def close(self):
        if self.pending is not None:
            self.ledger.cancel_reservation(self.pending['reservation_id']);self.pending=None
        self._drain()
        for stream in (self.physics,self.raw_physics,self.events,self.decisions,self.fk_trace):stream.flush();stream.close()
        return dict(stage=self.stage,recovery_hands=self.recovery_hands,native_diagnostic_summary=self.diag.summary(),
            geometry_blocked=self.geometry_blocked,confirmed_approach_submissions=self.executions,
            actual_native_fk_comparison=self.fk_summary,
            per_environment={str(i):dict(episode_id=self.terminal_ledgers.get(i,{}).get('episode_id',int(self.episode_ids[i])),
                ended=bool(self.ended[i]),grasp_style='tripod' if bool(self.tripod[i]) else 'pinch',
                last_gate=self.latest_rows.get(i,{}).get('stage_gate'),
                attempt_time_s=self.terminal_ledgers.get(i,{}).get('attempt_time_s',_plain(self.ledger.attempt_time_s[i])),
                approved_approach_m=self.terminal_ledgers.get(i,{}).get('approved_approach_m',_plain(self.ledger.executed_approach_m[i]))) for i in range(self.n)},
            counts_as_student_success=False,moving_wrist_executed=False,
            stage_A_executed=self.stage=='A' and self.active,
            stage_B_executed=self.stage=='B' and self.active,
            stage_C_executed=False,stage_D_executed=False,stage_E_executed=False,
            stages_B_C_D_E_executed=False,
            unresolved='source certificates conservative; prediction dynamics unvalidated; passive surface force coverage not connected')
