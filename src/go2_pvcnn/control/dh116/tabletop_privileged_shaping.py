"""Typed R1/R2 candidate arithmetic over measured SI inputs.

This module imports no simulator and confers no source qualification. The
native measurement factory and geometry/force source gates remain independent.
No function defines success, changes a controller or treats missing force as 0.
"""
from copy import deepcopy
from dataclasses import dataclass,field
import json
import math
from pathlib import Path

import torch

VERSION='dh116-privileged-shaping-v1'
GEOMETRY_TERMS=('geometry_pad_distance','geometry_coordination','geometry_arc_position','geometry_contact_area')
SUPPORT_TERMS=('hand_support_balance','opposing_pad_balance','contact_recovery',
    'normal_overload','tangential_overload','force_target_delta')
TERM_NAMES=GEOMETRY_TERMS+SUPPORT_TERMS
_CANDIDATES=dict(geometry_pad_distance=1.,geometry_coordination=.5,geometry_arc_position=.5,geometry_contact_area=.25,
    hand_support_balance=-.25,opposing_pad_balance=-.25,contact_recovery=.5,
    normal_overload=-.1,tangential_overload=-.1,force_target_delta=-.05)
_ALIASES={
    'geometry_coordination':('grasp_acquisition',),
    'geometry_pad_distance':('grasp_acquisition',),
    'geometry_contact_area':('grasp_acquisition',),
    'normal_overload':('force_excess','load_excess','surface_load_excess'),
    'tangential_overload':('tangential_excess',),
    'force_target_delta':('force_target_change',),
}


@dataclass(frozen=True)
class MeasuredTensor:
    value:torch.Tensor
    valid:torch.Tensor
    unit:str
    source:dict|str
    role:str='measured'


@dataclass(frozen=True)
class GeometryMeasurements:
    pad_position_m:MeasuredTensor
    target_pad_position_m:MeasuredTensor
    grasp_position_m:MeasuredTensor
    target_grasp_position_m:MeasuredTensor
    arc_position_m:MeasuredTensor
    target_arc_position_m:MeasuredTensor
    contact_area_m2:MeasuredTensor
    target_contact_area_m2:MeasuredTensor


@dataclass(frozen=True)
class SupportMeasurements:
    upward_support_n:MeasuredTensor
    normal_load_n:MeasuredTensor
    tangential_load_n:MeasuredTensor
    contact_engaged:MeasuredTensor
    force_target_n:MeasuredTensor


@dataclass(frozen=True)
class ShapingContext:
    transition_valid:torch.Tensor
    same_episode:torch.Tensor
    phase_active:torch.Tensor
    phase:str


@dataclass(frozen=True)
class SupportContext:
    base:ShapingContext
    weight_n:MeasuredTensor
    required_pads:torch.Tensor
    opposing_pairs:torch.Tensor  # [N,2,2], actual digit IDs, one pair per hand
    recovery_active_before:torch.Tensor
    recovery_epoch_before:torch.Tensor
    recovery_epoch_after:torch.Tensor
    unnecessary_target_change:torch.Tensor
    normal_limit_n:MeasuredTensor
    tangential_limit_n:MeasuredTensor
    force_target_scale_n:MeasuredTensor
    expected_hand_support_share:MeasuredTensor|None=None  # geometry-derived [N,2], never implicit 50/50


@dataclass(frozen=True)
class ShapingMeasurements:
    geometry_before:GeometryMeasurements
    geometry_after:GeometryMeasurements
    context:ShapingContext
    support_before:SupportMeasurements|None=None
    support_after:SupportMeasurements|None=None
    support_context:SupportContext|None=None


@dataclass(frozen=True)
class ShapingConfig:
    enabled:bool=False
    pad_distance_scale_m:float=.02
    coordination_scale_m:float=.02
    arc_scale_m:float=.02
    coefficients:dict=field(default_factory=lambda:dict(_CANDIDATES))

    def __post_init__(self):
        if type(self.enabled) is not bool:raise ValueError('explicit shaping opt-in boolean required')
        for value in (self.pad_distance_scale_m,self.coordination_scale_m,self.arc_scale_m):
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('positive SI geometry scales required')
        if not isinstance(self.coefficients,dict) or set(self.coefficients)!=set(TERM_NAMES):
            raise ValueError('one explicit candidate coefficient per shaping term required')
        for name,value in self.coefficients.items():
            positive=name in GEOMETRY_TERMS or name=='contact_recovery'
            if type(value) not in (int,float) or not math.isfinite(value) or (value<0 if positive else value>0):
                raise ValueError('finite coefficient with audited reward/cost sign required: '+name)


def load_shaping_config(value):
    value=json.loads(Path(value).read_text(encoding='utf-8')) if isinstance(value,(str,Path)) else deepcopy(value)
    if (not isinstance(value,dict) or value.get('version')!=VERSION
            or value.get('explicit_opt_in_required') is not True or value.get('native_qualified') is not False):
        raise ValueError('explicit unqualified candidate shaping config required')
    keys=('enabled','pad_distance_scale_m','coordination_scale_m','arc_scale_m','coefficients')
    if any(key not in value for key in keys):raise ValueError('complete candidate coefficient/normalization config required')
    return ShapingConfig(**{key:value[key] for key in keys})


def _bool(value,shape,device,name):
    if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=torch.bool or value.device!=device:
        raise ValueError(name+' requires exact per-environment/source boolean shape; broadcasting is forbidden')
    return value


def _measurement(item,shape,unit,device,name,*,vector=False,nonnegative=False,positive=False,boolean=False):
    if type(item) is not MeasuredTensor:raise ValueError('typed measured input required: '+name)
    value=item.value
    if (not isinstance(value,torch.Tensor) or value.shape!=shape or value.device!=device or item.unit!=unit
            or (value.dtype!=torch.bool if boolean else not value.is_floating_point())):
        raise ValueError(name+' requires exact shape/device and SI unit '+unit+'; no whole-hand broadcast')
    valid=_bool(item.valid,shape[:-1] if vector else shape,device,name+' validity')
    if not isinstance(item.source,(str,dict)) or not item.source or not isinstance(item.role,str) or not item.role:
        raise ValueError('actual measurement source/role metadata required: '+name)
    if not boolean:
        finite=torch.isfinite(value).all(-1) if vector else torch.isfinite(value)
        if bool((valid&~finite).any()):raise ValueError('valid measured input contains NA: '+name)
        if nonnegative and bool((valid&(value<0)).any()):raise ValueError('negative magnitude/area: '+name)
        if positive and bool((valid&(value<=0)).any()):raise ValueError('positive measured weight/limit/scale required: '+name)
    return value,valid


def _context(context,n,device):
    if type(context) is not ShapingContext or not isinstance(context.phase,str) or not context.phase:
        raise ValueError('explicit physical transition/phase context required')
    return (_bool(context.transition_valid,(n,),device,'transition validity')
        &_bool(context.same_episode,(n,),device,'same episode')
        &_bool(context.phase_active,(n,),device,'physical phase'))


def _setup(config,names,existing_owners):
    if type(config) is not ShapingConfig or config.enabled is not True:raise ValueError('candidate shaping disabled; explicit opt-in required')
    config.__post_init__()  # Coefficient dictionaries cannot bypass validation by mutation.
    if not isinstance(existing_owners,(list,tuple,set,frozenset)) or any(not isinstance(value,str) for value in existing_owners):
        raise ValueError('explicit existing reward owners required')
    owners=set(existing_owners)
    for name in names:
        if owners&{name,'teacher_'+name,*_ALIASES.get(name,())}:
            raise ValueError('duplicate reward owner/overlap for '+name)


def _bilateral_mean(value,valid):
    # Each hand has equal weight. Padding/NA never becomes measured zero.
    count=valid.sum(-1)
    good=(count>0).all(-1)
    safe=torch.where(valid,value,0.)
    mean=(safe.sum(-1)/count.clamp_min(1)).mean(-1)
    return torch.where(good,mean,float('nan')),good


def _source(item):return dict(measurement=deepcopy(item.source),role=item.role,unit=item.unit,valid_mask=item.valid.detach().clone())


def expected_support_share_from_geometry(grasp_position_m,com_position_m,gravity_m_s2,*,transverse_tolerance_m=1e-5):
    """Quasistatic vertical point-support lever arms, only when applicable.

    Actual World-frame grasp/COM/gravity measurements are mandatory. Coincident
    horizontal anchors, off-line/outside COM or unavailable geometry give NA.
    Dynamics, contact moments and external supports require a qualified factory
    to establish that this deliberately narrow equilibrium model applies.
    """
    ref=grasp_position_m.value
    if not isinstance(ref,torch.Tensor) or ref.ndim!=3:raise ValueError('two actual World grasp positions [N,2,3] required')
    n,device=ref.shape[0],ref.device
    if type(transverse_tolerance_m) not in (int,float) or not math.isfinite(transverse_tolerance_m) or transverse_tolerance_m<0:
        raise ValueError('finite nonnegative geometric model tolerance in m required')
    grasp,gv=_measurement(grasp_position_m,(n,2,3),'m',device,'actual World grasp XYZ',vector=True)
    com,cv=_measurement(com_position_m,(n,3),'m',device,'actual World COM XYZ',vector=True)
    gravity,av=_measurement(gravity_m_s2,(n,3),'m/s^2',device,'actual World gravity',vector=True)
    length=gravity.norm(dim=-1)
    if bool((av&(length<=0)).any()):raise ValueError('actual nonzero gravity required for vertical support shares')
    up=-gravity/length[:,None]
    delta=grasp[:,1]-grasp[:,0]
    delta=delta-(delta*up).sum(-1)[:,None]*up
    offset=com-grasp[:,0];offset=offset-(offset*up).sum(-1)[:,None]*up
    span=delta.square().sum(-1)
    parameter=(offset*delta).sum(-1)/span
    transverse=(offset-parameter[:,None]*delta).norm(dim=-1)
    applicable=(gv.all(-1)&cv&av&torch.isfinite(parameter)&torch.isfinite(transverse)&(span>1e-12)
        &(parameter>=0)&(parameter<=1)&(transverse<=transverse_tolerance_m))
    value=torch.stack((1-parameter,parameter),-1)
    value=torch.where(applicable[:,None],value,float('nan'))
    return MeasuredTensor(value,applicable[:,None].expand(-1,2).clone(),'fraction',
        dict(algorithm='quasistatic_vertical_point_support_lever_arms',native_qualified=False,
            model_applicability='requires_independent_geometry/contact/external-support_qualification',
            grasp=_source(grasp_position_m),com=_source(com_position_m),gravity=_source(gravity_m_s2),
            transverse_tolerance_m=float(transverse_tolerance_m)),role='geometry_expected_hand_support_share')


def _trace(name,raw,normalized,mask,config,context,source,*,dt=None,raw_unit):
    coefficient=float(config.coefficients[name])
    if bool((mask&(~torch.isfinite(raw)|~torch.isfinite(normalized))).any()):
        raise ValueError('finite normalized shaping required; measurement arithmetic overflow in '+name)
    contribution=torch.where(mask,normalized*coefficient*(1. if dt is None else dt),0.)
    if not bool(torch.isfinite(contribution).all()):raise ValueError('shaping coefficient/contribution overflow in '+name)
    return dict(raw=raw.detach().clone(),normalized=normalized.detach().clone(),coefficient=coefficient,
        raw_unit=raw_unit,normalized_unit='dimensionless',
        coefficient_unit='points_per_normalized_unit' if dt is None else 'points_per_second',
        dt_scaling='potential_difference_no_dt' if dt is None and name!='force_target_delta' else
            ('executed_target_delta_no_dt' if dt is None else float(dt)),
        valid_mask=mask.detach().clone(),phase=context.phase,source=deepcopy(source),
        per_env_contribution=contribution.detach().clone(),total_contribution=contribution.detach().sum(),
        qualification='pure_arithmetic;native_measurement_factory_and_source_gate_required')


def geometry_terms(before,after,context,config,*,existing_owners=()):
    """Four potential improvements; no success/force qualification prerequisite."""
    _setup(config,GEOMETRY_TERMS,existing_owners)
    if type(before) is not GeometryMeasurements or type(after) is not GeometryMeasurements:
        raise ValueError('typed geometry measurements before/after required')
    ref=before.pad_position_m.value
    if not isinstance(ref,torch.Tensor) or ref.ndim!=4:raise ValueError('pad XYZ must be [N,2,5,3]')
    n,device=ref.shape[0],ref.device;base=_context(context,n,device)
    states=[]
    for state in (before,after):
        values={}
        for name in ('pad_position_m','target_pad_position_m'):
            values[name]=_measurement(getattr(state,name),(n,2,5,3),'m',device,name,vector=True)
        for name in ('grasp_position_m','target_grasp_position_m'):
            values[name]=_measurement(getattr(state,name),(n,2,3),'m',device,name,vector=True)
        for name in ('arc_position_m','target_arc_position_m'):
            values[name]=_measurement(getattr(state,name),(n,2),'m',device,name,nonnegative=True)
        values['contact_area_m2']=_measurement(state.contact_area_m2,(n,2,5),'m^2',device,'contact area',nonnegative=True)
        values['target_contact_area_m2']=_measurement(state.target_contact_area_m2,(n,2,5),'m^2',device,'target contact area',positive=True)
        states.append(values)
    old,new=states;terms={}
    for name,field,target,scale in (
        ('geometry_pad_distance','pad_position_m','target_pad_position_m',config.pad_distance_scale_m),
        ('geometry_arc_position','arc_position_m','target_arc_position_m',config.arc_scale_m)):
        valid=old[field][1]&old[target][1]&new[field][1]&new[target][1]
        a,b=old[field][0]-old[target][0],new[field][0]-new[target][0]
        a,b=(a.norm(dim=-1),b.norm(dim=-1)) if field=='pad_position_m' else (a.abs(),b.abs())
        if field=='arc_position_m':a,b,valid=a[...,None],b[...,None],valid[...,None]
        raw_before,good=_bilateral_mean(a,valid);raw_after,_=_bilateral_mean(b,valid)
        terms[name]=_trace(name,raw_after-raw_before,(raw_before-raw_after)/scale,base&good,config,context,
            dict(before=_source(getattr(before,field)),after=_source(getattr(after,field)),
                target_before=_source(getattr(before,target)),target_after=_source(getattr(after,target))),raw_unit='m')
    name='geometry_coordination';field='grasp_position_m';target='target_grasp_position_m'
    valid=(old[field][1]&old[target][1]&new[field][1]&new[target][1]).all(-1)
    errors=[]
    for state in (old,new):
        residual=state[field][0]-state[target][0]
        errors.append((residual[:,1]-residual[:,0]).norm(dim=-1))
    terms[name]=_trace(name,errors[1]-errors[0],(errors[0]-errors[1])/config.coordination_scale_m,base&valid,
        config,context,dict(before=_source(before.grasp_position_m),after=_source(after.grasp_position_m),
            target_before=_source(before.target_grasp_position_m),target_after=_source(after.target_grasp_position_m)),raw_unit='m')
    name='geometry_contact_area';field='contact_area_m2';target='target_contact_area_m2'
    valid=old[field][1]&old[target][1]&new[field][1]&new[target][1]
    raw_before,good=_bilateral_mean(old[field][0],valid);raw_after,_=_bilateral_mean(new[field][0],valid)
    phi=[]
    for state in (old,new):
        # Known measured absence may be valid area=0; unknown contact is NA/false.
        area=(state[field][0]/state[target][0]).clamp(0.,1.)
        potential,_=_bilateral_mean(area,valid);phi.append(potential)
    terms[name]=_trace(name,raw_after-raw_before,phi[1]-phi[0],base&good,config,context,
        dict(before=_source(before.contact_area_m2),after=_source(after.contact_area_m2),
            target_before=_source(before.target_contact_area_m2),target_after=_source(after.target_contact_area_m2)),raw_unit='m^2')
    return {name:terms[name] for name in GEOMETRY_TERMS}


def support_terms(before,after,context,config,*,dt,existing_owners=()):
    """Five R2 factor classes; normal/tangential overload have separate traces."""
    _setup(config,SUPPORT_TERMS,existing_owners)
    if type(dt) not in (int,float) or not math.isfinite(dt) or dt<=0:raise ValueError('positive actual physical dt required')
    if type(before) is not SupportMeasurements or type(after) is not SupportMeasurements or type(context) is not SupportContext:
        raise ValueError('typed per-digit support measurements/context required')
    ref=before.upward_support_n.value
    if not isinstance(ref,torch.Tensor) or ref.ndim!=3:raise ValueError('actual per-digit support must be [N,2,5]')
    n,device=ref.shape[0],ref.device;base=_context(context.base,n,device);shape=(n,2,5)
    values=[]
    for state in (before,after):
        row={}
        for name in ('upward_support_n','normal_load_n','tangential_load_n','force_target_n'):
            row[name]=_measurement(getattr(state,name),shape,'N',device,name,nonnegative=name!='upward_support_n')
        row['contact_engaged']=_measurement(state.contact_engaged,shape,'bool',device,'contact engagement',boolean=True)
        if state.force_target_n.role!='executed_force_target':raise ValueError('actual executed force targets required; requested targets cannot substitute')
        values.append(row)
    old,new=values
    weight,weight_valid=_measurement(context.weight_n,(n,),'N',device,'actual object weight',positive=True)
    normal_limit,nl_valid=_measurement(context.normal_limit_n,shape,'N',device,'normal load limit',positive=True)
    tangent_limit,tl_valid=_measurement(context.tangential_limit_n,shape,'N',device,'tangential load limit',positive=True)
    target_scale,ts_valid=_measurement(context.force_target_scale_n,shape,'N',device,'executed target scale',positive=True)
    required=_bool(context.required_pads,shape,device,'required physical pads')
    unnecessary=_bool(context.unnecessary_target_change,shape,device,'unnecessary executed target change')
    recovery=_bool(context.recovery_active_before,(n,),device,'actual contact-break recovery window')
    for epoch in (context.recovery_epoch_before,context.recovery_epoch_after):
        if not isinstance(epoch,torch.Tensor) or epoch.shape!=(n,) or epoch.device!=device or epoch.dtype not in (torch.int32,torch.int64):
            raise ValueError('actual per-environment recovery epoch integer required')
    pairs=context.opposing_pairs
    if (not isinstance(pairs,torch.Tensor) or pairs.shape!=(n,2,2) or pairs.device!=device
            or pairs.dtype not in (torch.int32,torch.int64) or bool(((pairs<0)|(pairs>=5)).any())
            or bool((pairs[:,:,0]==pairs[:,:,1]).any())):raise ValueError('two distinct actual opposing digit IDs per hand required')
    terms={};name='hand_support_balance'
    hand=new['upward_support_n'][0].sum(-1)
    if context.expected_hand_support_share is None:
        raw=weight.new_full((n,),float('nan'));valid=torch.zeros(n,device=device,dtype=torch.bool)
        share_source=dict(unavailable_reason='geometry-derived expected hand support share is NA;no_equal_split_fallback')
    else:
        share,share_valid=_measurement(context.expected_hand_support_share,(n,2),'fraction',device,'expected hand support share',nonnegative=True)
        if bool((share_valid&(share>1)).any()):raise ValueError('expected support share must be a legal fraction in [0,1]')
        complete=share_valid.all(-1)
        if bool((complete&~torch.isclose(share.sum(-1),torch.ones_like(weight),rtol=1e-6,atol=1e-6)).any()):
            raise ValueError('expected support shares must sum to one')
        # Compare distribution at observed total support, so this cost does not
        # duplicate the separately owned total signed-support deficit.
        raw=(hand-hand.sum(-1)[:,None]*share).abs().sum(-1)
        raw=torch.where(complete,raw,float('nan'))
        valid=new['upward_support_n'][1].all((1,2))&weight_valid&complete
        share_source=_source(context.expected_hand_support_share)
    terms[name]=_trace(name,raw,raw/weight,base&valid,config,context.base,
        dict(after=_source(after.upward_support_n),weight=_source(context.weight_n),expected_share=share_source),dt=dt,raw_unit='N')
    name='opposing_pad_balance'
    a,b=pairs[:,:,0:1].long(),pairs[:,:,1:2].long()
    normal=new['normal_load_n'][0]
    first,second=normal.gather(-1,a).squeeze(-1),normal.gather(-1,b).squeeze(-1)
    selected=new['normal_load_n'][1]&new['contact_engaged'][1]&new['contact_engaged'][0]
    valid=(selected.gather(-1,a)&selected.gather(-1,b)).squeeze(-1)&((first+second)>0)
    raw=(first-second).abs().mean(-1);normalized=((first-second).abs()/(first+second)).mean(-1)
    terms[name]=_trace(name,raw,normalized,base&valid.all(-1),config,context.base,
        dict(after=_source(after.normal_load_n),contact=_source(after.contact_engaged),actual_digit_pairs=pairs.detach().clone()),dt=dt,raw_unit='N')
    name='contact_recovery'
    selected=required&old['contact_engaged'][1]&new['contact_engaged'][1]
    # Every required channel must be observed, and both hands must have a target.
    complete=(~required|selected).all((1,2))&required.any(-1).all(-1)
    old_fraction,_=_bilateral_mean(old['contact_engaged'][0].float(),selected)
    new_fraction,_=_bilateral_mean(new['contact_engaged'][0].float(),selected)
    progress=new_fraction-old_fraction
    valid=base&complete&recovery&(context.recovery_epoch_before==context.recovery_epoch_after)
    terms[name]=_trace(name,progress,progress,valid,config,context.base,
        dict(before=_source(before.contact_engaged),after=_source(after.contact_engaged),
            required_pads=required.detach().clone(),recovery_active_before=recovery.detach().clone(),
            recovery_epoch_before=context.recovery_epoch_before.detach().clone(),
            recovery_epoch_after=context.recovery_epoch_after.detach().clone()),raw_unit='contact_fraction')
    for name,field,limit,limit_valid,limit_source in (
        ('normal_overload','normal_load_n',normal_limit,nl_valid,context.normal_limit_n),
        ('tangential_overload','tangential_load_n',tangent_limit,tl_valid,context.tangential_limit_n)):
        selected=new[field][1]&limit_valid
        excess=(new[field][0]-limit).clamp_min(0.)
        raw,good=_bilateral_mean(excess,selected);normalized,_=_bilateral_mean(excess/limit,selected)
        terms[name]=_trace(name,raw,normalized,base&good,config,context.base,
            dict(after=_source(getattr(after,field)),limit=_source(limit_source)),dt=dt,raw_unit='N')
    name='force_target_delta'
    selected=old['force_target_n'][1]&new['force_target_n'][1]&ts_valid
    delta=(new['force_target_n'][0]-old['force_target_n'][0]).abs()
    # A known necessary change has zero cost; it is not a missing source. Keep
    # both hands' source masks so unilateral unnecessary jitter is still charged.
    delta=torch.where(unnecessary,delta,0.)
    raw,good=_bilateral_mean(delta,selected);normalized,_=_bilateral_mean(delta/target_scale,selected)
    terms[name]=_trace(name,raw,normalized,base&good,config,context.base,
        dict(before=_source(before.force_target_n),after=_source(after.force_target_n),scale=_source(context.force_target_scale_n),
            unnecessary_target_change=unnecessary.detach().clone()),raw_unit='N')
    return {name:terms[name] for name in SUPPORT_TERMS}
