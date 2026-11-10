"""Unwired CPU H8 command hypotheses using the original shared command kernels.

Twelve named targets per hand are commands, never a 17D physical response.
Feedback is explicitly frozen or externally forecast; no SDK write/step occurs.
"""
from collections.abc import Mapping
from dataclasses import asdict,dataclass
from types import MappingProxyType

import numpy as np
import torch

from .linear_coupling import ACTIVE,MIMIC
from .tabletop_palm_mount import JOINT_NAMES,ANGULAR_JOINT_NAMES,local_translation
from .tabletop_command_kernels import (finger_pre_guard,surface_final_guard,wrist_requested_targets,
    palm_transport,per_hand_safe_z)
from .tabletop_free_wrist_grasp import CONTRACT,compose_rotation
from .tabletop_surface_contract import ACTION_CONTRACT
from . import tabletop_vision_command_shadow as SHADOW
from .tabletop_bimanual_mpc_articulation_measurements import _binding
from .tabletop_bimanual_mpc_dynamics import _seed,_array,canonical_candidate_digest,PHYSICS_PERIOD_S,PHYSICS_SUBSTEPS
from .tabletop_bimanual_mpc_model_descriptor import _freeze,_thaw
from .tabletop_bimanual_mpc_mounted_kinematics import descriptor_digest
from .tabletop_bimanual_mpc_state import SampleClock

VERSION='dh116-h8-command-diagnostic-v1'
SIDES=('left','right')
DIRECT=ACTIVE+JOINT_NAMES+ANGULAR_JOINT_NAMES
_STATE=dict(joint_command_rad=(12,),palm_command_offset_m=(2,3),wrist_command_angle_rad=(2,3))
_CURRENT=dict(root_seed_pose=(2,7),prior_root_target_pose=(2,7),clearance_m=(2,),task_offset_m=(),
    clearance_floor_m=(),live=(),allowed=(),joint_lower=(12,),joint_upper=(12,),teacher_force_controlled_digits=(2,5))
_FEEDBACK=dict(q_sdk17=(2,17),qd_sdk17=(2,17),measured_root_pose=(2,7),anchor_root_pose=(2,7),
    **{k:v for k,v in _CURRENT.items() if k!='prior_root_target_pose'})
_BOOL={'live','allowed','teacher_force_controlled_digits'}
_UNITS=dict(joint_command_rad='rad',palm_command_offset_m='m',wrist_command_angle_rad='rad',
    root_seed_pose='world_xyz_wxyz',prior_root_target_pose='world_xyz_wxyz',clearance_m='m',
    task_offset_m='m',clearance_floor_m='m',live='boolean',allowed='boolean',joint_lower='rad',
    joint_upper='rad',teacher_force_controlled_digits='boolean')
_QUALIFICATION=dict(execution_qualified=False,planning_eligible=False,action_response_qualified=False,
    dynamics_qualified=False,complete_teacher_equivalence=False,native_attested=False,
    contact_model_qualified=False,hardware_measurement_qualified=False)


def _text(value,name):
    if not isinstance(value,str) or not value.strip():raise ValueError(name+' requires an explicit source')


def _own(value,shape,name,boolean=False):
    if isinstance(value,torch.Tensor):
        if value.device.type!='cpu':raise ValueError('CPU diagnostic accepts owned CPU inputs only')
        value=value.detach().numpy()
    return _array(value,shape,name,boolean=boolean)


def _tensor(value,dtype):
    result=torch.tensor(_thaw(value),dtype=dtype)[None]
    if dtype!=torch.bool and not bool(torch.isfinite(result).all()):
        raise ValueError('finite feedback/command is not representable in original logical dtype')
    return result


def _pose(value,name):
    a=np.asarray(value)
    if not np.allclose(np.linalg.norm(a[...,3:],axis=-1),1.,atol=1e-5,rtol=0):
        raise ValueError(name+' requires actual unit wxyz quaternions')


def _owned_record(record,shape,source):
    try:
        dtype=np.dtype(record.get('raw_dtype','').removeprefix('torch.'))
        raw_shape=record.get('raw_shape')
        valid=(record.get('available') is True and record.get('missing_reason') is None
            and isinstance(raw_shape,(tuple,list)) and all(type(n) is int for n in raw_shape)
            and tuple(raw_shape)==shape and np.asarray(record.get('value')).shape==shape
            and dtype.kind=='f' and dtype.itemsize in (4,8) and record.get('source')==source
            and isinstance(record.get('source_device'),str) and bool(record['source_device'].strip()))
    except (TypeError,ValueError,AttributeError):valid=False
    if not valid:raise ValueError('actual owned getter source/shape/dtype unavailable: '+source)


def _feedback_units(sample):
    position=[['m' if t==1 else 'rad' for t in sample.descriptor['hands'][s]['binding']['dof_types']] for s in SIDES]
    velocity=[[u+'/s' for u in hand] for hand in position]
    return dict(q_sdk17=position,qd_sdk17=velocity,measured_root_pose='world_xyz_wxyz',
        anchor_root_pose='world_xyz_wxyz',**{k:_UNITS[k] for k in _FEEDBACK if k in _UNITS})


def freeze_feedback_models(sample,*,actions,source):
    """Explicitly declare each future field frozen; no future actual clock."""
    _seed(sample,sample.clock);_text(source,'freeze assumption')
    digest=canonical_candidate_digest(actions);units=_feedback_units(sample)
    return {name:dict(mode='freeze_current',source=source,start_clock=asdict(sample.clock),
        descriptor_sha256=descriptor_digest(sample),candidate_digest=digest,candidate_independent=True,
        valid_mask=[True]*PHYSICS_SUBSTEPS,values=None,unit=units[name]) for name in _FEEDBACK}


def _current(sample,clock,state,feedback,receipts,dtype):
    q,qd,_,bindings,_=_seed(sample,clock)
    if not isinstance(state,Mapping) or set(state)!=set(_STATE) or not isinstance(feedback,Mapping) or set(feedback)!=set(_CURRENT):
        raise ValueError('complete exact current command/feedback schema required; no defaults')
    if not isinstance(receipts,Mapping) or set(receipts)!=set(_STATE)|set(_CURRENT):
        raise ValueError('every current command/feedback field requires its source receipt')
    for key in receipts:
        r=receipts[key]
        if (not isinstance(r,Mapping) or r.get('valid') is not True or r.get('unit')!=_UNITS[key]
                or r.get('descriptor_sha256')!=descriptor_digest(sample)):
            raise ValueError('current field receipt/source/unit unavailable: '+key)
        try:matched=SampleClock(**r['clock'])==clock
        except (TypeError,ValueError,KeyError):matched=False
        if not matched:raise ValueError('current field receipt clock differs: '+key)
        _text(r.get('source'),key)
    for key,value in state.items():
        if isinstance(value,torch.Tensor) and value.dtype!=dtype:raise ValueError('logical command dtype differs from actual cache input')
        if isinstance(value,np.ndarray) and value.dtype!=np.dtype(str(dtype).removeprefix('torch.')):
            raise ValueError('logical command dtype differs from actual NumPy cache input')
    state={k:_own(state[k],shape,k) for k,shape in _STATE.items()}
    now={k:_own(feedback[k],shape,k,k in _BOOL) for k,shape in _CURRENT.items()}
    if now['live'] is not True:raise ValueError('live current command seed required')
    for key in ('root_seed_pose','prior_root_target_pose'):_pose(now[key],key)
    mappings={};bases=[];anchors=[];bounds=[]
    for side in SIDES:
        h=sample.descriptor['hands'][side];b=h['binding'];raw=sample.raw['hands'][side]['articulation_input']
        if _thaw(b)!=_binding(_thaw(raw['metadata']),side=side,clock=clock,hand_names=_thaw(raw['hand_names'])):
            raise ValueError('actual named SDK binding differs from owned metadata')
        names=list(b['dof_names']);indices=[names.index(n) for n in DIRECT]
        if [b['dof_types'][i] for i in indices]!=[0]*6+[1]*3+[0]*3:
            raise ValueError('named active/mount coordinate types differ from actual contract')
        permutation=list(b['hand_joint_to_dof'])
        if any(type(i) is not int for i in permutation) or sorted(permutation)!=list(range(17)):
            raise ValueError('actual hand/SDK row mapping is not a complete unique integer permutation')
        mappings[side]=dict(joint_names=DIRECT,sdk_dof_indices=indices,
            hand_joint_indices=[permutation.index(i) for i in indices],units=['rad']*6+['m']*3+['rad']*3,
            articulation_path=b['articulation_path'],selected_view_index=b['selected_view_index'])
        read=h['readbacks']['get_link_transforms'];n=raw['metadata']['count'];index=b['selected_view_index']
        _owned_record(read,(n,23,7),f'hand[{side}].root_physx_view.get_link_transforms()')
        poses=_own(read.get('selected'),(23,7),side+' actual link poses')
        if (read.get('available') is not True or read.get('selected_valid') is not True
                or not np.array_equal(np.asarray(read.get('value'))[index],np.asarray(poses))):
            raise ValueError('actual base/anchor selected SDK row unavailable')
        root=h['graph']['root_link_row'];links=list(b['link_names'])
        if type(root) is not int or links[root]!='wrist_anchor' or links.count('base_link')!=1:
            raise ValueError('actual mounted anchor and base_link binding required')
        # SDK link poses are xyzw; original world/mount kernels consume wxyz.
        for target,row in ((anchors,root),(bases,links.index('base_link'))):
            p=poses[row];target.append((*p[:3],p[6],*p[3:6]))
        r=raw['getters'].get('get_dof_limits',{})
        _owned_record(r,(n,17,2),f'hand[{side}].root_physx_view.get_dof_limits()')
        try:
            batch=np.asarray(r.get('value'))
            if batch.shape!=(n,17,2):raise ValueError('actual limit layout differs')
            limit=np.asarray(_own(batch[index],(17,2),side+' actual selected limits'))
        except (TypeError,ValueError):raise ValueError('current actual SDK limits unavailable') from None
        if (r.get('available') is not True or r.get('unit')!='mixed_coordinate_lower_upper'
                or r.get('source')!=f'hand[{side}].root_physx_view.get_dof_limits()'
                or np.any(limit[:,0]>limit[:,1])):raise ValueError('current actual SDK limit source/interval unavailable')
        bounds.extend(limit[indices[:6]].tolist())
        mappings[side]['sdk_limits']=limit[indices].tolist()
    now.update(q_sdk17=_freeze([q[s] for s in SIDES]),qd_sdk17=_freeze([qd[s] for s in SIDES]),
        measured_root_pose=_freeze(bases),anchor_root_pose=_freeze(anchors))
    _pose(bases,'current actual base');_pose(anchors,'current actual anchor')
    _limits(now,np.asarray(bounds))
    return _freeze(state),_freeze(now),_freeze(mappings),bindings,np.asarray(bounds)


def _limits(row,bounds):
    lo=np.asarray(row['joint_lower']);hi=np.asarray(row['joint_upper'])
    if np.any(lo>hi) or np.any(lo<bounds[:,0]) or np.any(hi>bounds[:,1]):
        raise ValueError('current/forecast guard limits must fit the actual SDK named limits')
    if row['clearance_floor_m']<0:raise ValueError('clearance floor must be nonnegative')


def _models(sample,actions,now,models,bounds):
    digest=canonical_candidate_digest(actions);units=_feedback_units(sample)
    if not isinstance(models,Mapping) or set(models)!=set(_FEEDBACK):
        raise ValueError('each future feedback field requires an explicit freeze or external forecast')
    values={};owned={}
    for name,shape in _FEEDBACK.items():
        r=models[name]
        if not isinstance(r,Mapping):raise ValueError('feedback model record required')
        try:clock=SampleClock(**r['start_clock'])
        except (KeyError,TypeError,ValueError):raise ValueError('feedback source clock unavailable') from None
        if (clock!=sample.clock or r.get('descriptor_sha256')!=descriptor_digest(sample)
                or r.get('candidate_digest')!=digest or type(r.get('candidate_independent')) is not bool):
            raise ValueError('feedback clock/source/candidate binding differs')
        _text(r.get('source'),'future '+name)
        if _thaw(r.get('unit'))!=units[name]:raise ValueError('future feedback per-SDK coordinate unit differs: '+name)
        mask=_own(r.get('valid_mask'),(PHYSICS_SUBSTEPS,),name+' validity',True)
        if not all(mask):raise ValueError('unavailable future feedback is never zero-filled')
        if r.get('mode')=='freeze_current':
            if r.get('values') is not None or r['candidate_independent'] is not True:
                raise ValueError('freeze_current must retain the exact current value independently of candidate')
            series=(now[name],)*PHYSICS_SUBSTEPS
        elif r.get('mode')=='external_forecast':
            series=_own(r.get('values'),(PHYSICS_SUBSTEPS,)+shape,name+' external forecast',name in _BOOL)
            if series[0]!=now[name]:raise ValueError('first pre-substep forecast must equal the sourced current feedback')
        else:raise ValueError('future feedback must be freeze_current or external_forecast')
        if name.endswith('_pose'):
            _pose(series,name)
        values[name]=series;owned[name]=_thaw(r)
    rows=[{key:values[key][k] for key in _FEEDBACK} for k in range(PHYSICS_SUBSTEPS)]
    for row in rows:_limits(row,bounds)
    return rows,_freeze(owned)


def _profile(profile,scope,sample):
    if not isinstance(profile,Mapping):raise ValueError('recorded controller profile required')
    p=_thaw(profile);callbacks=p.get('callbacks')
    if (not isinstance(callbacks,list) or any(not isinstance(x,str) or not x.strip() for x in callbacks)
            or len(set(callbacks))!=len(callbacks)
            or any(type(p.get(k)) is not bool for k in ('surface_policy_owned_targets','pinch_force_hold','redistribution'))):
        raise ValueError('complete explicit controller callback list/flags required')
    try:clock=SampleClock(**p['source_clock'])
    except (KeyError,TypeError,ValueError):raise ValueError('controller profile current source clock unavailable') from None
    if clock!=sample.clock or p.get('descriptor_sha256')!=descriptor_digest(sample):
        raise ValueError('controller profile current clock/source binding differs')
    _text(p.get('source'),'current controller profile')
    if (p.get('version')!='dh116-conditional-command-v1' or type(p.get('physics_substeps')) is not int
            or p['physics_substeps']!=8 or p.get('action_contract')!=ACTION_CONTRACT
            or p.get('free_wrist_contract')!=CONTRACT or p.get('action_contract_sha256')!=SHADOW.action_contract_sha256()):
        raise ValueError('original controller action/guard/substep contract differs')
    if p.get('controller_source_sha256')!=SHADOW.controller_source_sha256():raise ValueError('controller source SHA differs')
    if scope=='supported_conditional_shadow':SHADOW._validate_contract(p)
    elif scope!='kernel_subchain_diagnostic':raise ValueError('unsupported command diagnostic scope')
    return p


def _kernel_context(row,mappings,profile,prior,dtype):
    active=[];qd=[];errors=[]
    for i,side in enumerate(SIDES):
        # The full SDK rows remain independent, including all five followers.
        m=mappings[side];names=m['joint_names'];positions=row['q_sdk17'][i];velocities=row['qd_sdk17'][i]
        active.extend(positions[j] for j in m['sdk_dof_indices'][:6]);qd.extend(velocities[j] for j in m['sdk_dof_indices'][:6])
        dofs=m['all_dof_names']
        errors.extend(positions[dofs.index(f)]-ratio*positions[dofs.index(master)]-offset for f,master,ratio,offset in MIMIC)
    ctx={k:_tensor(row[k],torch.bool if k in _BOOL else dtype) for k in _CURRENT if k!='prior_root_target_pose'}
    ctx.update(measured_q_rad=_tensor(active,dtype),measured_qd_rad_s=_tensor(qd,dtype),mimic_errors=_tensor(errors,dtype),
        measured_root_pose=_tensor(row['measured_root_pose'],dtype),prior_root_target_pose=prior,
        dt_s=torch.tensor([PHYSICS_PERIOD_S],dtype=dtype),source_valid=torch.tensor([True]),
        active_controller_flags=torch.tensor([[profile['surface_policy_owned_targets'],profile['pinch_force_hold']]],dtype=torch.bool))
    return ctx


def _subchain(state,action,ctx,profile,pre):
    """Only composition of shared kernels; no callback or unload emulation."""
    rates=dict(joint_rate_rad_s=ACTION_CONTRACT['joint_rate_rad_s'],wrist_rate_m_s=ACTION_CONTRACT['translation_rate_m_s'],
        rotation_rate_rad_s=ACTION_CONTRACT['rotation_rate_rad_s'])
    dt=float(ctx['dt_s'][0])
    requested,target=surface_final_guard(state['joint_command_rad'],pre,ctx['joint_lower'],ctx['joint_upper'],dt)
    palms,angles=wrist_requested_targets(state['palm_command_offset_m'],state['wrist_command_angle_rad'],action,rates,dt)
    floor=float(ctx['clearance_floor_m'][0])
    palms,preprojection,reasons=palm_transport(state['palm_command_offset_m'],palms,ctx['clearance_m'],floor,
        CONTRACT['wrist_rate_m_s'],dt,action.new_tensor(CONTRACT['palm_lower_m']),action.new_tensor(CONTRACT['palm_upper_m']),
        ctx['allowed'],ctx['live'],vector_norm=True)
    z,safe=per_hand_safe_z(palms,ctx['measured_root_pose'],ctx['root_seed_pose'],ctx['task_offset_m'],ctx['clearance_m'],floor)
    reasons|=(z-palms[:,:,2]).abs().gt(1e-12).long()*2
    palms=torch.cat((palms[:,:,:2],torch.where(ctx['live'][:,None],z,palms[:,:,2])[:,:,None]),-1)
    angles=torch.where((ctx['allowed']&ctx['live'])[:,None,None],angles,state['wrist_command_angle_rad'])
    command=torch.where(ctx['live'][:,None],target,state['joint_command_rad'])
    seed=ctx['root_seed_pose'];position=seed[:,:,:3]+palms
    position=torch.cat((position[:,:,:2],(position[:,:,2]+ctx['task_offset_m'][:,None])[:,:,None]),-1)
    root=torch.cat((position,compose_rotation(seed[:,:,3:],angles)),-1)
    root=torch.where(ctx['live'][:,None,None],root,ctx['prior_root_target_pose'])
    return dict(zip(SHADOW.STATE_KEYS,(command,palms,angles))),dict(root_pose=root,requested_finger=requested,
        pre_projection_palm=preprojection,safe_z_m=safe,projection_reasons=reasons)


def _copy_tensors(tree):
    return _freeze({k:v.detach().cpu().numpy().tolist()[0] for k,v in tree.items()})


@dataclass(frozen=True)
class CommandDiagnosticStep:
    substep_index:int
    control_index:int
    control_endpoint:bool
    prediction_offset_s:float
    command_state:object
    diagnostics:object
    named_sdk_float32:object
    kernel_context:object
    feedback_hypotheses:object
    kind:str='conditional_command_hypothesis'


@dataclass(frozen=True)
class CommandDiagnostic:
    source_clock:SampleClock
    candidate_digest:str
    initial_current:object
    steps:tuple
    receipt:object
    qualification:object
    version:str=VERSION


def unfold_command_h8(sample,*,current_clock,actions,command_state,current_feedback,source_receipts,
                      feedback_models,controller_contract,scope,logical_dtype='float64'):
    """Unfold 64 conditional command intervals; never execute or infer response."""
    if logical_dtype not in ('float32','float64'):raise ValueError('explicit original logical float32/float64 dtype required')
    dtype=getattr(torch,logical_dtype);digest=canonical_candidate_digest(actions)
    state,now,mappings,bindings,bounds=_current(sample,current_clock,command_state,current_feedback,source_receipts,dtype)
    profile=_profile(controller_contract,scope,sample);rows,models=_models(sample,actions,now,feedback_models,bounds)
    mapping=_thaw(mappings)
    for side in SIDES:mapping[side]['all_dof_names']=list(sample.descriptor['hands'][side]['binding']['dof_names'])
    initial=_freeze(dict(q_sdk17=now['q_sdk17'],qd_sdk17=now['qd_sdk17'],command_state=state,
        feedback=now,source_bindings=bindings,source_receipts=source_receipts,
        command_state_kind='current_reported_target_cache_not_actual_q',clock=asdict(current_clock)))
    state={k:_tensor(v,dtype) for k,v in state.items()};prior=_tensor(now['prior_root_target_pose'],dtype);steps=[]
    candidate=torch.tensor(_thaw(_own(actions,(8,24),'canonical candidate')),dtype=dtype)
    rates=dict(joint_rate_rad_s=ACTION_CONTRACT['joint_rate_rad_s'])
    for k,row in enumerate(rows):
        ctx=_kernel_context(row,mapping,profile,prior,dtype);action=candidate[k//8][None];dt=float(ctx['dt_s'][0])
        raw=state['joint_command_rad']+action[:,:12]*ACTION_CONTRACT['joint_rate_rad_s']*dt
        pre=finger_pre_guard(state['joint_command_rad'],action,ctx['measured_q_rad'],ctx['mimic_errors'],
            ctx['joint_lower'],ctx['joint_upper'],dt,rates,
            policy_owned=profile['surface_policy_owned_targets'],pinch_force_hold=profile['pinch_force_hold'],
            teacher_force_controlled_digits=ctx['teacher_force_controlled_digits'])
        if scope=='supported_conditional_shadow':state,diagnostics=SHADOW.shadow_step(state,action,ctx,profile)
        else:state,diagnostics=_subchain(state,action,ctx,profile,pre)
        direct={}
        for i,side in enumerate(SIDES):
            local=local_translation(diagnostics['root_pose'][:,i],_tensor(row['anchor_root_pose'][i],dtype))
            values=torch.cat((state['joint_command_rad'][:,6*i:6*i+6],local,state['wrist_command_angle_rad'][:,i]),-1).float()
            if not bool(torch.isfinite(values).all()):raise ValueError('SDK float32 command encoding overflow')
            direct[side]=dict(_thaw(mappings[side]),values=values[0].tolist())
            direct[side]['within_sdk_limits']=[lo<=v<=hi for v,(lo,hi) in zip(direct[side]['values'],direct[side]['sdk_limits'])]
        prior=diagnostics['root_pose']
        saved={key:value for key,value in diagnostics.items() if isinstance(value,torch.Tensor)}
        saved.update(raw_increment_finger=raw,pre_final_guard_finger=pre)
        steps.append(CommandDiagnosticStep(k+1,k//8,(k+1)%8==0,(k+1)*PHYSICS_PERIOD_S,_copy_tensors(state),
            _copy_tensors(saved),_freeze(direct),_copy_tensors(ctx),_freeze(row)))
    receipt=_freeze(dict(scope=scope,controller_callbacks=profile['callbacks'],controller_profile=profile,
        controller_source_sha256=profile['controller_source_sha256'],descriptor_sha256=descriptor_digest(sample),
        source_clock=asdict(current_clock),candidate_digest=digest,logical_dtype=logical_dtype,feedback_models=models,
        complete_teacher_equivalence=False,mandatory_release_modeled=False,uncovered_feedback=['pressure','contact','callbacks'],
        targets_are_actual_response=False,full17_response_generated_from_targets=False,
        future_rows_are_actual_measurements=False,controller_profile_is_caller_reported=True,
        direct_targets_per_hand=12,passive_followers_targeted=False,original_paths_modified=False,
        sdk_link_pose_order_assumption='xyz_xyzw converted to original world/mount wxyz',
        native_cache_pose_alias_parity_verified=False))
    return CommandDiagnostic(current_clock,digest,initial,tuple(steps),receipt,MappingProxyType(dict(_QUALIFICATION)))
