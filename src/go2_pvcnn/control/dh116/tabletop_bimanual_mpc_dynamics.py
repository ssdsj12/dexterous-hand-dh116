"""Pure hands-only H8 kinematic extrapolation hypotheses, default unwired.

Eight 60 Hz controls contain 64 original 480 Hz integration intervals. Each
explicit full17 velocity row describes (t_k,t_k+dt] and produces q_{k+1}; every
poststep uses the complete independently sourced left/right mounted FK tree.
Candidates identify hypotheses; no command/target, drive, force, contact,
RNEA, SDK call, solver, physical prediction or execution is supplied here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict,dataclass
import math
from numbers import Real
from types import MappingProxyType

import numpy as np

from .tabletop_bimanual_mpc_state import SampleClock
from .tabletop_bimanual_mpc_objective import HORIZON,CONTROL_PERIOD_S,candidate_digest
from .tabletop_bimanual_mpc_model_descriptor import (
    MountedModelDescriptorSample,VERSION as DESCRIPTOR_VERSION,_digest,_freeze,_thaw)
from .tabletop_bimanual_mpc_mounted_kinematics import MountedTreeKinematics,descriptor_digest

VERSION='dh116-h8-mounted-kinematic-rollout-v1'
PHYSICS_PERIOD_S=1/480
SUBSTEPS_PER_CONTROL=8
PHYSICS_SUBSTEPS=HORIZON*SUBSTEPS_PER_CONTROL
SIDES=('left','right')
_QUALIFICATION=dict(native_geometry_calibrated=False,action_response_qualified=False,
    dynamics_qualified=False,candidate_future_dynamics_qualified=False,contact_model_qualified=False,
    constraints_qualified=False,solver_qualified=False,execution_qualified=False,planning_eligible=False,
    native_attested=False,native_success_qualification_issued=False,new_expert_qualification_issued=False,
    hardware_measurement_qualified=False)


def _text(value,name):
    if not isinstance(value,str) or not value.strip():raise ValueError(name+' must explicitly name its source/assumption')


def _array(value,shape,name,*,boolean=False,nullable=False):
    def scalar(v):
        if isinstance(v,np.ndarray):
            if v.dtype.kind not in ('b' if boolean else 'fiu'):raise ValueError(name+' has unsupported dtype')
            return scalar(v.tolist())
        if isinstance(v,(tuple,list)):
            for x in v:scalar(x)
        elif boolean:
            if type(v) is not bool:raise ValueError(name+' requires actual boolean validity')
        elif v is None and nullable:pass
        elif isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v):
            raise ValueError(name+' requires finite real values; NA is never zero-filled')
    scalar(value)
    try:a=np.asarray(value,dtype=object if nullable else None)
    except (TypeError,ValueError) as exc:raise ValueError(name+' has invalid shape') from exc
    if a.shape!=shape:raise ValueError(f'{name} requires exact shape {shape}')
    return _freeze(a.tolist())


def canonical_candidate_digest(actions):
    owned=_array(actions,(HORIZON,24),'canonical24 candidate')
    if any(abs(v)>1 for row in owned for v in row):raise ValueError('canonical24 candidate must be bounded [-1,1]; no clipping')
    return candidate_digest(owned)


def _seed(sample,current_clock):
    if (not isinstance(sample,MountedModelDescriptorSample) or sample.structure_valid is not True
            or not isinstance(current_clock,SampleClock) or sample.clock!=current_clock
            or current_clock.terminal or current_clock.reset):
        raise ValueError('same-clock live structure-valid descriptor required')
    if _thaw(sample.descriptor.get('clock'))!=asdict(current_clock):raise ValueError('descriptor clock differs from current actual sample')
    if sample.descriptor.get('version')!=DESCRIPTOR_VERSION or sample.raw.get('version')!=DESCRIPTOR_VERSION:
        raise ValueError('current descriptor schema version is unsupported')
    brackets=sample.raw.get('read_brackets')
    if not isinstance(brackets,(tuple,list)) or not brackets:
        raise ValueError('current descriptor read brackets unavailable')
    for receipt in brackets:
        if not isinstance(receipt,Mapping) or 'after_error' in receipt:
            raise ValueError('current descriptor read bracket failed its post-read clock check')
        for edge in ('before','after'):
            clocks=receipt.get(edge)
            if not isinstance(clocks,(tuple,list)):
                raise ValueError('current descriptor read bracket clock unavailable')
            own=[c for c in clocks if isinstance(c,Mapping) and c.get('env_id')==current_clock.env_id]
            try:
                if len(own)!=1 or SampleClock(**own[0])!=current_clock:
                    raise ValueError('current descriptor read bracket clock differs from current actual sample')
            except (TypeError,ValueError) as exc:
                raise ValueError('current descriptor read bracket requires one exact current env clock') from exc
    if any(v is not False for v in sample.qualification.values()):raise ValueError('descriptor cannot grant kinematic rollout qualification')
    q={};qd={};units={};bindings={};poses={}
    for side in SIDES:
        hand=sample.descriptor['hands'][side];binding=hand['binding'];raw=sample.raw['hands'][side]
        if (_thaw(raw['articulation_input']['binding'])!=_thaw(binding)
                or _thaw(raw['source_before'])!=_thaw(hand['source_identity'])
                or _thaw(raw['source_after'])!=_thaw(hand['source_identity'])
                or _thaw(raw['articulation_input']['source_identity'])!=_thaw(hand['source_identity'])
                or any(raw['articulation_input']['source_brackets'].get(k,{}).get('available') is not True
                    or _thaw(raw['articulation_input']['source_brackets'][k].get('value'))!=_thaw(hand['source_identity'])
                    for k in ('before','after'))):
            raise ValueError('current descriptor binding/source is inconsistent')
        if any(hand['source_identity'].get(k)!=1. or isinstance(hand['source_identity'].get(k),bool)
                for k in ('meters_per_unit','kilograms_per_unit')):
            raise ValueError('current actual SI source required; no stage-unit conversion')
        types=binding['dof_types']
        if len(types)!=17 or any(type(t) is not int or t not in (0,1) for t in types):raise ValueError('actual full17 DOF types required')
        units[side]=tuple('m/s' if t==1 else 'rad/s' for t in types)
        index=binding['selected_view_index'];n=raw['articulation_input']['metadata']['count']
        if binding.get('available') is not True or type(index) is not int or type(n) is not int or not 0<=index<n:
            raise ValueError('current actual SDK row binding unavailable')
        for name,getter in (('q','get_dof_positions'),('qd','get_dof_velocities')):
            receipt=hand['measurement_receipts'].get(name)
            if (not isinstance(receipt,Mapping) or receipt.get('original_field_valid') is not True
                    or receipt.get('kind')!='measured' or _thaw(receipt.get('clock'))!=asdict(current_clock)
                    or receipt.get('timestamp_s')!=current_clock.world_time_s or receipt.get('raw_getter')!=getter
                    or receipt.get('unit')!=('mixed_coordinate' if name=='q' else 'mixed_coordinate/s')
                    or receipt.get('frame')!='backend_generalized_or_link_coordinates'
                    or receipt.get('source')!=f'hand[{side}].root_physx_view.{getter}()'):
                raise ValueError('current actual full17 '+name+' receipt unavailable or wrong clock')
            _text(receipt.get('source'),name+' measurement source')
            value=_array(hand['measurements'].get(name),(17,),'current actual '+side+'.'+name)
            record=raw['articulation_input']['getters'].get(getter)
            data=np.asarray(record.get('value')) if isinstance(record,Mapping) else None
            if (not isinstance(record,Mapping) or record.get('available') is not True or data.shape!=(n,17)
                    or record.get('source')!=receipt['source'] or record.get('unit')!=receipt['unit']
                    or not np.array_equal(np.asarray(value),data[index])):
                raise ValueError('current actual full17 '+name+' does not match owned SDK row')
            (q if name=='q' else qd)[side]=value
        bindings[side]=dict(binding=_thaw(binding),source_identity=_thaw(hand['source_identity']))
        # Canonical owned-value comparison accepts identical unavailable NaN
        # padding in other batch slots, without filling or selecting that row.
        if _digest(hand['readbacks']['get_link_transforms'])!=_digest(raw['readbacks']['get_link_transforms']):
            raise ValueError('current root-link readback differs from owned raw')
        poses[side]=_thaw(hand['readbacks']['get_link_transforms'])
    return q,qd,units,_freeze(bindings),_freeze(poses)


@dataclass(frozen=True)
class Full17VelocityForecast:
    start_clock: SampleClock
    descriptor_sha256: str
    candidate_digest: str
    source_bindings: object
    velocity_units: object
    values: object
    valid_mask: object
    model_source: str
    model_kind: str
    candidate_independent: bool
    data_valid: bool=True
    missing_reason: str | None=None
    version: str=VERSION

    def __post_init__(self):
        if not isinstance(self.start_clock,SampleClock):raise ValueError('forecast requires its actual source clock')
        for name in ('descriptor_sha256','candidate_digest'):
            v=getattr(self,name)
            if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):raise ValueError('forecast '+name+' required')
        _text(self.model_source,'forecast model_source')
        if type(self.candidate_independent) is not bool or type(self.data_valid) is not bool:raise ValueError('forecast flags must be actual booleans')
        if (self.model_kind not in ('measured_constant_velocity','external_candidate_conditioned','external_exogenous')
                or (self.model_kind=='external_candidate_conditioned')==self.candidate_independent):
            raise ValueError('forecast model kind/candidate independence declaration disagrees')
        object.__setattr__(self,'values',_array(self.values,(PHYSICS_SUBSTEPS,2,17),'full17 interval velocity forecast',nullable=True))
        object.__setattr__(self,'valid_mask',_array(self.valid_mask,(PHYSICS_SUBSTEPS,2,17),'full17 forecast validity',boolean=True))
        for key in ('source_bindings','velocity_units'):object.__setattr__(self,key,_freeze(_thaw(getattr(self,key))))
        if not self.data_valid:_text(self.missing_reason,'forecast missing_reason')
        elif self.missing_reason is not None:raise ValueError('valid forecast cannot claim missing response data')


def make_velocity_forecast(sample,*,actions,values,valid_mask,model_source,model_kind,candidate_independent,
                           data_valid=True,missing_reason=None):
    _,_,units,bindings,_=_seed(sample,sample.clock)
    return Full17VelocityForecast(sample.clock,descriptor_digest(sample),canonical_candidate_digest(actions),
        bindings,units,values,valid_mask,model_source,model_kind,candidate_independent,data_valid,missing_reason)


def measured_constant_velocity_forecast(sample,*,actions,model_source):
    _,qd,_,_,_=_seed(sample,sample.clock)
    values=[[qd[side] for side in SIDES] for _ in range(PHYSICS_SUBSTEPS)]
    return make_velocity_forecast(sample,actions=actions,values=values,valid_mask=np.ones((PHYSICS_SUBSTEPS,2,17),dtype=bool),
        model_source=model_source,model_kind='measured_constant_velocity',candidate_independent=True)


@dataclass(frozen=True)
class KinematicForecastStep:
    substep_index: int
    control_index: int
    within_control_substep: int
    control_endpoint: bool
    prediction_offset_s: float
    q_sdk17: object
    interval_qd_sdk17: object
    fk: object
    kind: str='kinematic_forecast_hypothesis'


@dataclass(frozen=True)
class KinematicRollout:
    source_clock: SampleClock
    candidate_digest: str
    initial_current_actual: object
    initial_geometry_hypothesis: object
    states: tuple
    forecast_receipt: object
    qualification: object
    horizon: int=HORIZON
    substeps_per_control: int=SUBSTEPS_PER_CONTROL
    control_period_s: float=CONTROL_PERIOD_S
    physics_period_s: float=PHYSICS_PERIOD_S
    kind: str='kinematic_rollout_hypothesis'


def rollout_kinematic_h8(sample,*,current_clock,actions,coordinate_hypotheses,forecast):
    q0,qd0,units,bindings,poses=_seed(sample,current_clock);digest=canonical_candidate_digest(actions)
    if (not isinstance(forecast,Full17VelocityForecast) or forecast.version!=VERSION
            or forecast.start_clock!=current_clock or forecast.descriptor_sha256!=descriptor_digest(sample)
            or forecast.candidate_digest!=digest):raise ValueError('forecast clock/descriptor/candidate binding mismatch')
    if _thaw(forecast.source_bindings)!=_thaw(bindings):raise ValueError('forecast source binding mismatch')
    if _thaw(forecast.velocity_units)!=_thaw(units):raise ValueError('forecast per-row velocity unit mismatch')
    if (forecast.data_valid is not True or any(v is not True for row in forecast.valid_mask for hand in row for v in hand)
            or any(v is None for row in forecast.values for hand in row for v in hand)):
        raise ValueError('complete valid full17 response required; unavailable passive/drive values are never filled')
    if forecast.model_kind=='measured_constant_velocity' and any(tuple(row[i])!=tuple(qd0[side])
            for row in forecast.values for i,side in enumerate(SIDES)):
        raise ValueError('measured constant-velocity assumption must repeat the actual full17 qd; no relabelled response')
    if not isinstance(coordinate_hypotheses,Mapping) or set(coordinate_hypotheses)!=set(SIDES):raise ValueError('independent left/right coordinate hypotheses required')
    trees={side:MountedTreeKinematics.from_descriptor(sample,side=side,coordinate_hypothesis=coordinate_hypotheses[side]) for side in SIDES}
    initial_geometry=MappingProxyType({side:trees[side].fk(q0[side]) for side in SIDES})
    initial=_freeze(dict(kind='current_reported_actual_readback',clock=asdict(current_clock),q_sdk17=q0,qd_sdk17=qd0,
        sdk_link_transform_raw=poses,source_bindings=bindings,source_scope='simulation readback provenance; not hardware or future measurement'))
    q={side:np.array(q0[side],dtype=float,copy=True) for side in SIDES};states=[]
    for k in range(PHYSICS_SUBSTEPS):
        velocity={side:forecast.values[k][i] for i,side in enumerate(SIDES)}
        for side in SIDES:
            q[side]=q[side]+np.asarray(velocity[side],dtype=float)*PHYSICS_PERIOD_S
            if not np.isfinite(q[side]).all():raise ValueError('kinematic integration overflow; no finite trajectory')
        fk=MappingProxyType({side:trees[side].fk(q[side]) for side in SIDES})
        states.append(KinematicForecastStep(k+1,k//SUBSTEPS_PER_CONTROL,k%SUBSTEPS_PER_CONTROL+1,
            (k+1)%SUBSTEPS_PER_CONTROL==0,(k+1)*PHYSICS_PERIOD_S,_freeze({side:q[side].tolist() for side in SIDES}),_freeze(velocity),fk))
    receipt=_freeze(dict(version=VERSION,source_clock=asdict(current_clock),descriptor_sha256=descriptor_digest(sample),candidate_digest=digest,
        model_source=forecast.model_source,model_kind=forecast.model_kind,candidate_independent=forecast.candidate_independent,
        data_valid=forecast.data_valid,velocity_units=forecast.velocity_units,source_bindings=forecast.source_bindings,
        interval_convention='row k applies to (t_k,t_k+1/480] and yields poststep k+1',
        candidate_binding_is_not_action_response_validation=True,conditioning_claim_is_declaration_only=True,
        action_response_qualified=False,commands_predicted=False,targets_used_as_actual=False,affine_mimic_applied=False,
        root_link_pose_assumed_constant=True,future_values_are_measured=False))
    return KinematicRollout(current_clock,digest,initial,initial_geometry,tuple(states),receipt,MappingProxyType(dict(_QUALIFICATION)))
