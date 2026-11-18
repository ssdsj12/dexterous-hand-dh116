"""Unwired H8 constraint arithmetic with mandatory uncertified physical gaps.

This report does not produce a feasible action. Full17 box/velocity arithmetic
uses a declared kinematic rollout and frozen current SDK limits. Its interval
coverage is a piecewise-linear joint-box check, never a collision certificate.
No setters, solver, SDK reads, contact inference or fallback hold live here.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType

import numpy as np

from . import tabletop_bimanual_mpc_dynamics as D
from .tabletop_bimanual_mpc_articulation_measurements import ArticulationMeasurementSample
from .tabletop_bimanual_mpc_model_descriptor import _digest, _freeze, _thaw
from .tabletop_bimanual_mpc_mounted_kinematics import descriptor_digest
from .tabletop_bimanual_mpc_state import SampleClock, StateField


VERSION='dh116-h8-constraint-report-v1'
REQUIRED_CONSTRAINTS=(
    'full17_position_bounds','full17_velocity_cap_frozen',
    'command_position_bounds','command_slew','command_lead','original_guard_equivalence',
    'moving_wrist_workspace','continuous_collision','future_drive_effort',
    'contact_state','object_dynamics','bilateral_support','action_response')
_QUALIFICATION=dict(constraints_qualified=False,solver_qualified=False,execution_qualified=False,
    planning_eligible=False,action_response_qualified=False,dynamics_qualified=False,
    contact_model_qualified=False,native_attested=False,native_success_qualification_issued=False,
    new_expert_qualification_issued=False,hardware_measurement_qualified=False)
_GAPS={
    'command_position_bounds':'same-candidate typed command diagnostic not supplied',
    'command_slew':'same-candidate typed command diagnostic not supplied',
    'command_lead':'same-candidate typed command diagnostic not supplied',
    'original_guard_equivalence':'complete original controller callbacks and mandatory release are not certified',
    'moving_wrist_workspace':'complete moving-wrist workspace evidence is not supplied',
    'continuous_collision':'complete 64-interval moving-hand/object/table collision certificates are not supplied',
    'future_drive_effort':'future drive, effort and motor response are not certified by current readbacks',
    'contact_state':'same-horizon actual contact response is not certified',
    'object_dynamics':'same-horizon object dynamics are not certified',
    'bilateral_support':'same-horizon bilateral object-side support is not certified'}


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    status: str
    valid: bool
    source: object
    min_margin_by_unit: object
    witness_by_unit: object
    coverage: object
    missing_reason: str | None=None
    assumption: str | None=None
    required: bool=True


@dataclass(frozen=True)
class H8ConstraintReport:
    source_clock: SampleClock
    candidate_digest: str
    descriptor_sha256: str
    source_bindings: object
    forecast_receipt: object
    command_receipt: object
    constraints: object
    mathematical_checks_passed: bool
    qualification: object
    required_constraint_names: tuple=REQUIRED_CONSTRAINTS
    feasible: bool=False
    action: object=None
    version: str=VERSION


def _same_clock(value,clock):
    try:
        return (isinstance(value,Mapping) and set(value)==set(asdict(clock))
            and SampleClock(**value)==clock)
    except (TypeError,ValueError):return False


def _brackets(raw,clock,*,required_getters=True):
    rows=raw.get('read_brackets')
    if not isinstance(rows,(list,tuple)) or not rows:raise ValueError('articulation getter clock brackets unavailable')
    names=set()
    for r in rows:
        if not isinstance(r,Mapping) or 'after_error' in r:raise ValueError('articulation getter clock bracket failed')
        names.add(r.get('name'))
        for edge in ('before','after'):
            clocks=r.get(edge)
            if not isinstance(clocks,(list,tuple)):raise ValueError('articulation getter clock bracket unavailable')
            own=[c for c in clocks if isinstance(c,Mapping) and c.get('env_id')==clock.env_id]
            if len(own)!=1 or not _same_clock(own[0],clock):raise ValueError('articulation getter clock bracket mismatch')
    if required_getters and not {s+'.'+g for s in D.SIDES for g in ('get_dof_positions','get_dof_velocities','get_dof_limits','get_dof_max_velocities')}<=names:
        raise ValueError('required articulation getter clock bracket absent')


def _owned(record,shape,source,unit):
    try:
        dtype=np.dtype(record.get('raw_dtype','').removeprefix('torch.'))
        layout=record.get('raw_shape')
        ok=(record.get('available') is True and record.get('missing_reason') is None
            and isinstance(layout,(list,tuple)) and all(type(x) is int and x>=0 for x in layout)
            and tuple(layout)==shape and np.asarray(record.get('value')).shape==shape
            and dtype.kind=='f' and dtype.itemsize in (4,8)
            and isinstance(record.get('source_device'),str) and bool(record['source_device'].strip())
            and record.get('source')==source and record.get('unit')==unit)
    except (AttributeError,TypeError,ValueError):ok=False
    if not ok:raise ValueError('owned SDK getter shape/dtype/source/unit mismatch: '+source)


def _field(sample,side,name,getter,shape,unit):
    field=sample.fields.get(side+'.'+name)
    if field is None:return None,'current actual '+side+'.'+name+' field unavailable'
    if (not isinstance(field,StateField) or field.clock!=sample.clock
            or field.env_id!=sample.clock.env_id or field.episode_uid!=sample.clock.episode_uid
            or field.source!=f'hand[{side}].root_physx_view.{getter}()' or field.unit!=unit
            or field.frame!='backend_generalized_or_link_coordinates'):
        raise ValueError('current actual field clock/source/unit mismatch: '+side+'.'+name)
    if not field.valid:return None,field.missing_reason
    if field.kind!='measured' or field.timestamp_s!=sample.clock.world_time_s:
        raise ValueError('current actual sampled field receipt mismatch: '+side+'.'+name)
    hand=sample.raw['hands'][side];n=hand['metadata']['count'];i=hand['binding']['selected_view_index']
    record=hand['getters'].get(getter)
    _owned(record,(n,)+shape,field.source,unit)
    value=D._array(field.value,shape,side+'.'+name)
    if not np.array_equal(np.asarray(value),np.asarray(record['value'])[i]):
        raise ValueError('current actual field differs from owned selected SDK row: '+side+'.'+name)
    return np.asarray(value,dtype=float),None


def _current(sample,articulation,current_clock):
    q,qd,units,bindings,_=D._seed(sample,current_clock)
    if not _same_clock(sample.descriptor.get('clock'),current_clock):raise ValueError('descriptor source clock mismatch')
    for side in D.SIDES:
        for name in ('q','qd'):
            if not _same_clock(sample.descriptor['hands'][side]['measurement_receipts'][name].get('clock'),current_clock):
                raise ValueError('descriptor measurement receipt clock mismatch')
    _brackets(sample.raw,current_clock,required_getters=False)
    if (not isinstance(articulation,ArticulationMeasurementSample) or articulation.clock!=current_clock
            or any(v is not False for v in articulation.qualification.values())):
        raise ValueError('same-clock unqualified actual articulation sample required')
    _brackets(articulation.raw,current_clock)
    limits={};caps={};reasons={}
    for side in D.SIDES:
        hand=articulation.raw['hands'][side]
        if _digest(hand)!=_digest(sample.raw['hands'][side]['articulation_input']):
            raise ValueError('articulation source/raw differs from descriptor original input')
        if (_digest(hand['binding'])!=_digest(bindings[side]['binding'])
                or _digest(hand['source_identity'])!=_digest(bindings[side]['source_identity'])):
            raise ValueError('articulation original binding/source differs from canonical descriptor identity')
        if hand.get('unit_layout',{}).get('si_stage_units_verified') is not True:
            raise ValueError('actual SI stage-unit identity required; no normalization')
        for name,getter,expected,unit in (
            ('q','get_dof_positions',q[side],'mixed_coordinate'),
            ('qd','get_dof_velocities',qd[side],'mixed_coordinate/s')):
            value,reason=_field(articulation,side,name,getter,(17,),unit)
            if reason or not np.array_equal(value,np.asarray(expected)):
                raise ValueError('current full17 seed differs from actual articulation field')
        limits[side],reason=_field(articulation,side,'limits','get_dof_limits',(17,2),'mixed_coordinate_lower_upper')
        if reason:reasons[side+'.limits']=reason
        elif np.any(limits[side][:,0]>limits[side][:,1]):raise ValueError('actual joint lower limit exceeds upper limit')
        caps[side],reason=_field(articulation,side,'velocity_cap','get_dof_max_velocities',(17,),'mixed_coordinate/s')
        if reason:reasons[side+'.velocity_cap']=reason
        elif np.any(caps[side]<0):raise ValueError('actual velocity cap is negative')
    return q,qd,units,bindings,limits,caps,reasons


def _rollout(sample,current_clock,actions,rollout,q0,qd0,units,bindings):
    digest=D.canonical_candidate_digest(actions)
    if (not isinstance(rollout,D.KinematicRollout) or rollout.source_clock!=current_clock
            or rollout.candidate_digest!=digest or rollout.horizon!=8 or type(rollout.horizon) is not int
            or rollout.substeps_per_control!=8 or type(rollout.substeps_per_control) is not int
            or rollout.control_period_s!=D.CONTROL_PERIOD_S or rollout.physics_period_s!=D.PHYSICS_PERIOD_S
            or rollout.kind!='kinematic_rollout_hypothesis' or len(rollout.states)!=64
            or any(v is not False for v in rollout.qualification.values())):
        raise ValueError('complete same-clock/candidate H8 kinematic rollout required')
    r=rollout.forecast_receipt
    if not _same_clock(r.get('source_clock'),current_clock):raise ValueError('rollout forecast source clock mismatch')
    if (r.get('version')!=D.VERSION
            or r.get('descriptor_sha256')!=descriptor_digest(sample) or r.get('candidate_digest')!=digest
            or _digest(r.get('source_bindings'))!=_digest(bindings) or _thaw(r.get('velocity_units'))!=_thaw(units)
            or r.get('data_valid') is not True or not isinstance(r.get('model_source'),str) or not r['model_source'].strip()
            or r.get('model_kind') not in ('measured_constant_velocity','external_candidate_conditioned','external_exogenous')
            or type(r.get('candidate_independent')) is not bool
            or (r['model_kind']=='external_candidate_conditioned')==r['candidate_independent']):
        raise ValueError('rollout forecast source/descriptor/units/response receipt mismatch')
    for flag,expected in dict(action_response_qualified=False,commands_predicted=False,targets_used_as_actual=False,
            future_values_are_measured=False,candidate_binding_is_not_action_response_validation=True,
            conditioning_claim_is_declaration_only=True).items():
        if r.get(flag) is not expected:raise ValueError('rollout may not relabel hypotheses as qualified physical responses')
    initial=rollout.initial_current_actual
    if not _same_clock(initial.get('clock'),current_clock):raise ValueError('rollout initial actual source clock mismatch')
    if (initial.get('kind')!='current_reported_actual_readback'
            or _digest(initial.get('source_bindings'))!=_digest(bindings)):
        raise ValueError('rollout current actual source receipt mismatch')
    for side in D.SIDES:
        for name,values in (('q_sdk17',q0),('qd_sdk17',qd0)):
            if not np.array_equal(D._array(initial.get(name,{}).get(side),(17,),name),values[side]):
                raise ValueError('rollout initial current value mismatch')
    previous={s:np.asarray(q0[s]) for s in D.SIDES};states=[];velocities=[]
    for k,step in enumerate(rollout.states):
        if (not isinstance(step,D.KinematicForecastStep) or type(step.substep_index) is not int or step.substep_index!=k+1
                or type(step.control_index) is not int or step.control_index!=k//8
                or type(step.within_control_substep) is not int or step.within_control_substep!=k%8+1
                or type(step.control_endpoint) is not bool or step.control_endpoint!=((k+1)%8==0)
                or step.prediction_offset_s!=(k+1)*D.PHYSICS_PERIOD_S or step.kind!='kinematic_forecast_hypothesis'):
            raise ValueError('rollout must cover all 64 ordered poststates/intervals and eight endpoints')
        row={};velocity={}
        for side in D.SIDES:
            row[side]=np.asarray(D._array(step.q_sdk17[side],(17,),'rollout q'),dtype=float)
            velocity[side]=np.asarray(D._array(step.interval_qd_sdk17[side],(17,),'rollout interval qd'),dtype=float)
            if not np.array_equal(row[side],previous[side]+velocity[side]*D.PHYSICS_PERIOD_S):
                raise ValueError('rollout interval does not integrate its declared full17 velocity')
            if r['model_kind']=='measured_constant_velocity' and not np.array_equal(velocity[side],qd0[side]):
                raise ValueError('candidate-independent measured CV must repeat current actual qd')
        states.append(row);velocities.append(velocity);previous=row
    return digest,states,velocities


def _na(name,reason,source='required evidence not supplied',status='NA'):
    return ConstraintResult(name,status,False,_freeze(source),_freeze({}),_freeze({}),_freeze({}),reason)


def _arithmetic(name,rows,limits,bindings,clock,*,velocity=False):
    margins={};witness={}
    for k,row in enumerate(rows):
        for side in D.SIDES:
            binding=bindings[side]['binding']
            for j,value in enumerate(row[side]):
                unit=('m' if binding['dof_types'][j]==1 else 'rad')+('/s' if velocity else '')
                margin=float(limits[side][j]-abs(value) if velocity else min(value-limits[side][j,0],limits[side][j,1]-value))
                if unit not in margins or margin<margins[unit]:
                    substep=k+1 if velocity else k
                    margins[unit]=margin
                    witness[unit]=dict(clock=asdict(clock),side=side,dof_index=j,dof_name=binding['dof_names'][j],
                        selected_view_index=binding['selected_view_index'],articulation_path=binding['articulation_path'],
                        substep_index=substep,control_index=max(substep-1,0)//8,
                        interval_start_substep=max(substep-1,0),interval_end_substep=substep,
                        value=float(value),margin=margin,unit=unit)
    source={s:f'hand[{s}].root_physx_view.'+('get_dof_max_velocities()' if velocity else 'get_dof_limits()') for s in D.SIDES}
    assumption=('current SDK velocity caps frozen over H8; no future drive certification' if velocity else
        'current SDK limits frozen over H8; linear interpolation within each declared joint interval only')
    return ConstraintResult(name,'PASS' if all(v>=0 for v in margins.values()) else 'VIOLATION',True,_freeze(source),
        _freeze(margins),_freeze(witness),_freeze(dict(poststates=64,intervals=64,initial_state_checked=not velocity,
            values_checked=len(rows)*2*17,continuous_collision_certified=False)),assumption=assumption)


def _command(sample,actions,trace,q0,states,limits,bindings,reasons):
    # Reuse the actual bridge and its original shared kernels, not a second
    # implementation of controller arithmetic or a caller's executed=True bit.
    from . import tabletop_bimanual_mpc_command_bridge as B
    from .tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S
    from .tabletop_free_wrist_grasp import CONTRACT
    from .linear_coupling import ACTIVE
    if (not isinstance(trace,B.CommandDiagnostic) or trace.version!=B.VERSION or trace.source_clock!=sample.clock
            or trace.candidate_digest!=D.canonical_candidate_digest(actions) or len(trace.steps)!=64
            or any(v is not False for v in trace.qualification.values())):
        raise ValueError('same-clock/candidate typed original command diagnostic required')
    initial=trace.initial_current;r=trace.receipt
    try:
        replay=B.unfold_command_h8(sample,current_clock=sample.clock,actions=actions,
            command_state=initial['command_state'],current_feedback={k:initial['feedback'][k] for k in B._CURRENT},
            source_receipts=initial['source_receipts'],feedback_models=r['feedback_models'],
            controller_contract=r['controller_profile'],scope=r['scope'],logical_dtype=r['logical_dtype'])
        if (_digest(initial)!=_digest(replay.initial_current) or _digest(r)!=_digest(replay.receipt)
                or _digest(trace.qualification)!=_digest(replay.qualification)
                or any(not isinstance(x,B.CommandDiagnosticStep) or _digest(vars(x))!=_digest(vars(y))
                    for x,y in zip(trace.steps,replay.steps))):
            raise ValueError('saved command diagnostic differs from original bridge replay')
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('complete original command diagnostic schema required') from exc
    previous=np.asarray(initial['command_state']['joint_command_rad'],dtype=float)
    best={n:({}, {}) for n in ('command_position_bounds','command_slew','command_lead')}
    def retain(name,unit,margin,k,side,j,value,scope):
        margins,witness=best[name]
        if unit not in margins or margin<margins[unit]:
            b=bindings[side]['binding'];margins[unit]=float(margin)
            witness[unit]=dict(clock=asdict(sample.clock),substep_index=k+1,interval_start_substep=k,interval_end_substep=k+1,
                control_index=k//8,side=side,dof_index=j,dof_name=b['dof_names'][j],
                selected_view_index=b['selected_view_index'],articulation_path=b['articulation_path'],
                value=float(value),margin=float(margin),unit=unit,scope=scope)
    for k,step in enumerate(trace.steps):
        pre=q0 if k==0 else states[k-1]
        if not np.array_equal(np.asarray(step.feedback_hypotheses['q_sdk17']),np.asarray([pre[s] for s in D.SIDES])):
            raise ValueError('command feedback pre-state differs from the joint rollout hypothesis')
        command=np.asarray(step.command_state['joint_command_rad'],dtype=float)
        for h,side in enumerate(D.SIDES):
            b=bindings[side]['binding'];direct=step.named_sdk_float32[side]
            if limits[side] is not None:
                for j,value in zip(direct['sdk_dof_indices'],direct['values']):
                    lo,hi=limits[side][j];unit='m' if b['dof_types'][j]==1 else 'rad'
                    retain('command_position_bounds',unit,min(value-lo,hi-value),k,side,j,value,'encoded SDK float32 target')
            for d,name in enumerate(ACTIVE):
                i=6*h+d;j=list(b['dof_names']).index(name);value=command[i]
                lo=step.feedback_hypotheses['joint_lower'][i];hi=step.feedback_hypotheses['joint_upper'][i]
                retain('command_position_bounds','rad',min(value-lo,hi-value),k,side,j,value,'logical active-finger guard bounds')
                retain('command_slew','rad/s',ACTIVE_VELOCITY_LIMIT_RAD_S-abs(value-previous[i])/D.PHYSICS_PERIOD_S,
                    k,side,j,abs(value-previous[i])/D.PHYSICS_PERIOD_S,'logical active-finger command rate')
                retain('command_lead','rad',CONTRACT['joint_target_lead_rad']-abs(value-pre[side][j]),
                    k,side,j,value,'logical active-finger target relative to interval pre-state q')
        previous=command
    source={
        'command_position_bounds':dict(sdk_limits='current actual get_dof_limits() selected SDK row',
            active_limits='original bridge sourced joint_lower/joint_upper feedback'),
        'command_slew':dict(rate_rad_s=ACTIVE_VELOCITY_LIMIT_RAD_S,
            interface='tabletop_velocity_limits.ACTIVE_VELOCITY_LIMIT_RAD_S'),
        'command_lead':dict(lead_rad=CONTRACT['joint_target_lead_rad'],
            interface="tabletop_free_wrist_grasp.CONTRACT['joint_target_lead_rad']",contract_version=CONTRACT['version'])}
    result={}
    for name,(margins,witness) in best.items():
        missing=[s+'.limits: '+reasons[s+'.limits'] for s in D.SIDES if s+'.limits' in reasons]
        if name=='command_position_bounds' and missing:
            result[name]=_na(name,'; '.join(missing),source=source[name]);continue
        result[name]=ConstraintResult(name,'PASS' if all(v>=0 for v in margins.values()) else 'VIOLATION',True,
            _freeze(source[name]),_freeze(margins),_freeze(witness),_freeze(dict(intervals=64,poststates=64,
                scope='logical active-finger command; not actual qd' if name=='command_slew' else 'conditional command hypothesis',
                sdk_float32_slew_certified=False,executed_guard_certified=False)),
            assumption='original bridge command arithmetic only; no actual response or complete controller equivalence')
    result['original_guard_equivalence']=_na('original_guard_equivalence',_GAPS['original_guard_equivalence'],status='UNCERTIFIED')
    receipt=_freeze(dict(replayed_from_original_bridge=True,bridge_version=B.VERSION,bridge_receipt=r,
        initial_source_receipts=initial['source_receipts'],same_rollout_pre_q_checked=True,
        executed_guard_certified=False,sdk_float32_slew_certified=False))
    return result,receipt


def evaluate_h8_constraints(descriptor_sample,articulation_sample,*,current_clock,actions,rollout,command_trace=None):
    """Return required arithmetic and gaps; never an executable/feasible action.

    The registry is fixed. Missing honest readbacks produce NA, whereas torn or
    relabelled source/clock/DTO data raises. No caller can turn required gaps off.
    """
    q,qd,units,bindings,limits,caps,reasons=_current(descriptor_sample,articulation_sample,current_clock)
    digest,states,velocities=_rollout(descriptor_sample,current_clock,actions,rollout,q,qd,units,bindings)
    results={name:_na(name,reason) for name,reason in _GAPS.items()}
    for name,data,rows,velocity,suffix in (
            ('full17_position_bounds',limits,[{s:np.asarray(q[s]) for s in D.SIDES}]+states,False,'.limits'),
            ('full17_velocity_cap_frozen',caps,velocities,True,'.velocity_cap')):
        missing=[s+suffix+': '+reasons[s+suffix] for s in D.SIDES if s+suffix in reasons]
        source={s:f'hand[{s}].root_physx_view.'+('get_dof_max_velocities()' if velocity else 'get_dof_limits()') for s in D.SIDES}
        results[name]=(_na(name,'; '.join(missing),source=source) if missing else
            _arithmetic(name,rows,data,bindings,current_clock,velocity=velocity))
    independent=rollout.forecast_receipt['candidate_independent']
    results['action_response']=_na('action_response',
        'candidate-independent forecast cannot establish action response' if independent else
        'candidate conditioning is a declaration; future drive/action response is not certified',
        source=rollout.forecast_receipt['model_source'],status='NA' if independent else 'UNCERTIFIED')
    command_receipt=None
    math_names=['full17_position_bounds','full17_velocity_cap_frozen']
    if command_trace is not None:
        command_results,command_receipt=_command(descriptor_sample,actions,command_trace,q,states,limits,bindings,reasons)
        results.update(command_results);math_names+=['command_position_bounds','command_slew','command_lead']
    mathematical=all(results[n].status=='PASS' for n in math_names)
    return H8ConstraintReport(current_clock,digest,descriptor_digest(descriptor_sample),_freeze(_thaw(bindings)),
        _freeze(_thaw(rollout.forecast_receipt)),command_receipt,MappingProxyType({n:results[n] for n in REQUIRED_CONSTRAINTS}),
        mathematical,_freeze(_QUALIFICATION))
