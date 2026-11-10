"""Unwired full17 local force-drive response hypotheses, pure NumPy arithmetic.

Current M is held unchanged for 64 intervals; no symmetrization, armature,
perfect mimic, contact model or implicit C/G sign is added. Explicit full17
targets are not the twelve commands written by the original controller.
This is neither a PhysX oracle nor a calibrated physical/execution model.
"""
from collections.abc import Mapping
from dataclasses import asdict,dataclass
from types import MappingProxyType

import numpy as np

from .linear_coupling import ACTIVE
from .tabletop_bimanual_mpc_articulation_measurements import ArticulationMeasurementSample,_GETTERS,_WRIST
from .tabletop_bimanual_mpc_dynamics import (
    _array,_seed,_text,_QUALIFICATION,canonical_candidate_digest,PHYSICS_PERIOD_S,
    PHYSICS_SUBSTEPS,SUBSTEPS_PER_CONTROL,SIDES)
from .tabletop_bimanual_mpc_model_descriptor import _digest,_freeze,_thaw
from .tabletop_bimanual_mpc_mounted_kinematics import descriptor_digest
from .tabletop_bimanual_mpc_state import SampleClock,StateField

VERSION='dh116-local-force-drive-response-v1'
DIRECT=ACTIVE+_WRIST
RESIDUAL_ATOL=1e-10
RESIDUAL_RTOL=1e-10
_FIELDS={name:(getter,unit) for name,getter,_,unit in _GETTERS}
_PARAMETERS=dict(mass_matrix='generalized_mass_matrix_raw',stiffness='stiffness',damping='damping',
    effort_caps='drive_effort_cap',drive_types='drive_type')


def _numeric(value,shape,name):return np.asarray(_array(value,shape,name),dtype=float)


def _parameter_arrays(mass_matrix,stiffness,damping,effort_caps,drive_types):
    m=_numeric(mass_matrix,(17,17),'raw current full17 M')
    k=_numeric(stiffness,(17,),'full17 stiffness');d=_numeric(damping,(17,),'full17 damping')
    cap=_numeric(effort_caps,(17,),'full17 raw drive cap');drive=_numeric(drive_types,(17,),'full17 drive types')
    if not np.isin(drive,(0,1)).all():raise ValueError('only explicit none/force drive types supported; acceleration drive is unsupported')
    if any(np.any(x<0) for x in (k,d,cap)):raise ValueError('nonnegative stiffness/damping/drive caps required')
    try:np.linalg.solve(m,np.ones(17))
    except np.linalg.LinAlgError as exc:raise ValueError('singular raw M; no identity/armature fallback') from exc
    # Known NONE/zero-cap drives have exactly zero effort, not missing state.
    effective=(drive==1)&(cap>0)
    return m,np.where(effective,k,0.),np.where(effective,d,0.),cap


@dataclass(frozen=True)
class ForceDriveStep:
    q_sdk17:object
    qd_sdk17:object
    drive_effort_hypothesis:object
    equation_residual:object
    iterations:int
    actual_motor_measurement:bool=False


def implicit_force_drive_step(*,q,qd,mass_matrix,stiffness,damping,effort_caps,drive_types,
                              position_targets,velocity_targets,generalized_load,max_iterations=64):
    """Solve M*(v_next-v)/dt=clip(K*(r-q_next)+D*(vr-v_next),cap)+load.

    q_next=q+dt*v_next, dt=1/480. The cap clips only this hypothetical drive;
    explicit load is not relabelled as motor effort. Residual tolerances are
    row-wise arithmetic tolerances in N/N*m, never a physical error tube.
    Arbitrary supplied M is used verbatim; solvability is not an SPD claim.
    """
    if type(max_iterations) is not int or max_iterations<1:raise ValueError('positive integer solver iteration budget required')
    m,k,d,cap=_parameter_arrays(mass_matrix,stiffness,damping,effort_caps,drive_types)
    q=_numeric(q,(17,),'full17 q');v=_numeric(qd,(17,),'full17 qd')
    r=_numeric(position_targets,(17,),'full17 position targets');vr=_numeric(velocity_targets,(17,),'full17 velocity targets')
    load=_numeric(generalized_load,(17,),'explicit generalized load')
    dt=PHYSICS_PERIOD_S
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            a=k*(r-q)+d*vr;b=d+dt*k
            x=np.linalg.solve(m+np.diag(dt*b),m@v+dt*(a+load))
            def evaluate(value):
                raw=a-b*value;effort=np.clip(raw,-cap,cap)
                acceleration=(value-v)/dt
                residual=m@acceleration-effort-load
                scale=RESIDUAL_ATOL+RESIDUAL_RTOL*(np.abs(m)@np.abs(acceleration)+np.abs(effort)+np.abs(load))
                return raw,effort,residual,float(np.max(np.abs(residual)/scale))
            for iteration in range(1,max_iterations+1):
                raw,effort,residual,merit=evaluate(x)
                if merit<=1:
                    position=q+dt*x
                    if not np.isfinite(position).all():raise ValueError('response integration overflow')
                    return ForceDriveStep(_freeze(position.tolist()),_freeze(x.tolist()),
                        _freeze(effort.tolist()),_freeze(residual.tolist()),iteration)
                free=(raw>-cap)&(raw<cap)
                correction=np.linalg.solve(m/dt+np.diag(b*free),-residual)
                for backtrack in range(25):
                    trial=x+correction*(.5**backtrack)
                    if evaluate(trial)[3]<merit:
                        x=trial;break
                else:raise ValueError('implicit capped drive solver residual did not converge')
    except (np.linalg.LinAlgError,FloatingPointError,OverflowError) as exc:
        raise ValueError('implicit force-drive arithmetic failed; no complete response') from exc
    raise ValueError('implicit capped drive solver did not converge within iteration budget')


def _brackets_current(brackets,clock):
    if not isinstance(brackets,(tuple,list)) or not brackets:raise ValueError('current articulation read brackets unavailable')
    for receipt in brackets:
        if not isinstance(receipt,Mapping) or 'after_error' in receipt:raise ValueError('current articulation read bracket failed')
        for edge in ('before','after'):
            rows=receipt.get(edge)
            if not isinstance(rows,(tuple,list)):raise ValueError('current articulation clock unavailable')
            own=[r for r in rows if isinstance(r,Mapping) and r.get('env_id')==clock.env_id]
            try:
                if len(own)!=1 or set(own[0])!=set(asdict(clock)) or SampleClock(**own[0])!=clock:
                    raise ValueError('complete current articulation clock receipt differs')
            except (ValueError,TypeError) as exc:raise ValueError('one exact current env articulation clock required') from exc


def _current_field(sample,side,name,shape):
    field=sample.fields.get(side+'.'+name);getter,unit=_FIELDS[name];clock=sample.clock
    if (not isinstance(field,StateField) or field.valid is not True or field.kind!='measured'
            or field.clock!=clock or field.timestamp_s!=clock.world_time_s or field.unit!=unit
            or field.frame!='backend_generalized_or_link_coordinates'
            or field.source!=f'hand[{side}].root_physx_view.{getter}()'):
        raise ValueError('current valid measured '+side+'.'+name+' required')
    hand=sample.raw['hands'][side];index=hand['binding']['selected_view_index'];n=hand['metadata']['count']
    record=hand['getters'].get(getter)
    if (not isinstance(record,Mapping) or record.get('available') is not True
            or record.get('source')!=field.source or record.get('unit')!=unit):raise ValueError('current raw getter unavailable: '+name)
    # Only the selected environment must be finite; other batch padding may be NA.
    values=np.asarray(record.get('value'))
    value=_numeric(field.value,shape,'current selected '+name)
    if values.shape!=(n,)+shape or not np.array_equal(value,values[index]):raise ValueError('current DTO differs from owned selected SDK row: '+name)
    return _freeze(value.tolist())


@dataclass(frozen=True)
class Full17TargetHypothesis:
    source_clock:SampleClock
    descriptor_sha256:str
    candidate_digest:str
    source_bindings:object
    position_units:object
    velocity_units:object
    direct_sdk_dof_indices:object
    position_targets:object
    velocity_targets:object
    source:str
    interval_assumption:str
    version:str=VERSION

    def __post_init__(self):
        if not isinstance(self.source_clock,SampleClock):raise ValueError('target source clock required')
        for key in ('descriptor_sha256','candidate_digest'):
            text=getattr(self,key)
            if not isinstance(text,str) or len(text)!=64 or any(c not in '0123456789abcdef' for c in text):raise ValueError('target identity digest required')
        for key in ('source','interval_assumption'):_text(getattr(self,key),key)
        for key in ('position_targets','velocity_targets'):
            object.__setattr__(self,key,_array(getattr(self,key),(PHYSICS_SUBSTEPS,2,17),'explicit full17 '+key))
        for key in ('source_bindings','position_units','velocity_units','direct_sdk_dof_indices'):
            object.__setattr__(self,key,_freeze(_thaw(getattr(self,key))))


def make_target_hypothesis(sample,*,actions,position_targets,velocity_targets,source,interval_assumption):
    _,_,units,bindings,_=_seed(sample,sample.clock)
    position={side:tuple('m' if v=='m/s' else 'rad' for v in units[side]) for side in SIDES}
    direct={side:tuple(list(bindings[side]['binding']['dof_names']).index(n) for n in DIRECT) for side in SIDES}
    return Full17TargetHypothesis(sample.clock,descriptor_digest(sample),canonical_candidate_digest(actions),bindings,
        position,units,direct,position_targets,velocity_targets,source,interval_assumption)


@dataclass(frozen=True)
class LocalResponseStep:
    substep_index:int
    control_index:int
    within_control_substep:int
    control_endpoint:bool
    prediction_offset_s:float
    q_sdk17:object
    qd_sdk17:object
    drive_effort_hypothesis:object
    equation_residual:object
    kind:str='local_force_drive_response_hypothesis'


@dataclass(frozen=True)
class LocalResponseRollout:
    source_clock:SampleClock
    candidate_digest:str
    initial_current:object
    states:tuple
    receipt:object
    qualification:object
    error_tube:None=None


@dataclass(frozen=True,init=False)
class LocalHandResponse:
    source_clock:SampleClock
    descriptor_sha256:str
    source_bindings:object
    velocity_units:object
    current_parameters:object
    initial_current:object
    model_source:str

    @classmethod
    def from_current(cls,sample,articulation_sample,*,current_clock,model_source):
        return cls(sample,articulation_sample,current_clock=current_clock,model_source=model_source)

    def __init__(self,sample,articulation_sample,*,current_clock,model_source):
        q,qd,units,bindings,_=_seed(sample,current_clock);_text(model_source,'model_source')
        a=articulation_sample
        if not isinstance(a,ArticulationMeasurementSample) or a.clock!=current_clock:
            raise ValueError('same-clock actual articulation sample required')
        if any(v is not False for v in a.qualification.values()):raise ValueError('current readback cannot grant response qualification')
        _brackets_current(a.raw.get('read_brackets'),current_clock)
        params={};current_targets={};current_velocity_targets={}
        for side in SIDES:
            if _digest(a.raw['hands'][side])!=_digest(sample.raw['hands'][side]['articulation_input']):
                raise ValueError('articulation source/raw snapshot differs from current descriptor input')
            params[side]={key:_current_field(a,side,name,(17,17) if key=='mass_matrix' else (17,)) for key,name in _PARAMETERS.items()}
            _parameter_arrays(**params[side])
            current_targets[side]=_current_field(a,side,'backend_position_target',(17,))
            current_velocity_targets[side]=_current_field(a,side,'backend_velocity_target',(17,))
        initial=_freeze(dict(kind='current_reported_actual_readback',clock=asdict(current_clock),q_sdk17=q,qd_sdk17=qd,
            backend_position_targets=current_targets,backend_velocity_targets=current_velocity_targets,
            articulation_input_sha256=_digest(a.raw)))
        for key,value in dict(source_clock=current_clock,descriptor_sha256=descriptor_digest(sample),source_bindings=bindings,
                velocity_units=units,current_parameters=_freeze(params),initial_current=initial,model_source=model_source).items():
            object.__setattr__(self,key,value)

    def predict_h8(self,*,actions,targets,generalized_load,load_assumption,compensation_assumption,contact_assumption):
        digest=canonical_candidate_digest(actions)
        if (not isinstance(targets,Full17TargetHypothesis) or targets.version!=VERSION
                or targets.source_clock!=self.source_clock or targets.descriptor_sha256!=self.descriptor_sha256
                or targets.candidate_digest!=digest):raise ValueError('target clock/descriptor/candidate identity differs')
        if _thaw(targets.source_bindings)!=_thaw(self.source_bindings):raise ValueError('target source binding differs')
        position={side:tuple('m' if v=='m/s' else 'rad' for v in self.velocity_units[side]) for side in SIDES}
        direct={side:tuple(list(self.source_bindings[side]['binding']['dof_names']).index(n) for n in DIRECT) for side in SIDES}
        if (_thaw(targets.position_units)!=_thaw(position) or _thaw(targets.velocity_units)!=_thaw(self.velocity_units)
                or _thaw(targets.direct_sdk_dof_indices)!=_thaw(direct)):raise ValueError('target per-row units/direct command mapping differs')
        for value,name in ((load_assumption,'load_assumption'),(compensation_assumption,'compensation_assumption'),(contact_assumption,'contact_assumption')):_text(value,name)
        load=_array(generalized_load,(PHYSICS_SUBSTEPS,2,17),'explicit full17 generalized-load hypothesis')
        q=_thaw(self.initial_current['q_sdk17']);v=_thaw(self.initial_current['qd_sdk17']);states=[]
        for k in range(PHYSICS_SUBSTEPS):
            effort={};residual={}
            for i,side in enumerate(SIDES):
                step=implicit_force_drive_step(q=q[side],qd=v[side],**self.current_parameters[side],
                    position_targets=targets.position_targets[k][i],velocity_targets=targets.velocity_targets[k][i],generalized_load=load[k][i])
                q[side]=step.q_sdk17;v[side]=step.qd_sdk17;effort[side]=step.drive_effort_hypothesis;residual[side]=step.equation_residual
            states.append(LocalResponseStep(k+1,k//SUBSTEPS_PER_CONTROL,k%SUBSTEPS_PER_CONTROL+1,
                (k+1)%SUBSTEPS_PER_CONTROL==0,(k+1)*PHYSICS_PERIOD_S,_freeze(q),_freeze(v),_freeze(effort),_freeze(residual)))
        receipt=_freeze(dict(version=VERSION,model_source=self.model_source,descriptor_sha256=self.descriptor_sha256,
            target_source=targets.source,target_interval_assumption=targets.interval_assumption,
            load_assumption=load_assumption,compensation_assumption=compensation_assumption,contact_assumption=contact_assumption,
            source_bindings=self.source_bindings,position_units=position,velocity_units=self.velocity_units,
            direct_sdk_dof_indices=direct,full_target_rows_per_hand=17,direct_command_rows_per_hand=12,
            mass_matrix='exact current raw matrix held constant; no symmetrization or armature added',
            generalized_load=load,current_C_G_and_explicit_input_automatically_applied=False,
            cap_scope='raw readback cap applied only to hypothetical force drive; not a measured motor/shared feedforward budget',
            residual_atol=RESIDUAL_ATOL,residual_rtol=RESIDUAL_RTOL,calibrated_error_tube=None,
            future_values_are_measured=False,targets_are_actual_q=False,perfect_mimic_applied=False,
            target_timing_native_calibrated=False,current_model_physics_calibrated=False))
        return LocalResponseRollout(self.source_clock,digest,self.initial_current,tuple(states),receipt,MappingProxyType(dict(_QUALIFICATION)))
