"""CPU evidence boundaries for the unwired H8 joint-constraint report."""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib
from pathlib import Path

import numpy as np
import pytest

from test_bimanual_mpc_model_descriptor import fixture
from test_bimanual_mpc_mounted_kinematics import hypothesis
from go2_pvcnn.control.dh116 import tabletop_bimanual_mpc_dynamics as D
from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_model_descriptor import _freeze, _thaw


def api():
    path=Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_constraints.py'
    assert path.is_file(), 'H8 constraints report is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_constraints')


def context(*, velocity=None, configure=None, cv=False):
    f=fixture()
    for h in f.hands.values():
        v=h.root_physx_view;old_name=v.shared_metatype.link_names[-1]
        v.shared_metatype.link_names[-1]='base_link'
        v.shared_metatype.link_indices={n:i for i,n in enumerate(v.shared_metatype.link_names)}
        v.link_paths=[p.rsplit('/',1)[0]+'/base_link' if p.endswith('/'+old_name) else p for p in v.link_paths]
        h.body_names=list(reversed(v.shared_metatype.link_names))
        a=h.root_physx_view.arrays
        for name in ('get_dof_positions','get_dof_velocities','get_dof_position_targets','get_dof_velocity_targets'):
            a[name][:]=0
        a['get_dof_limits'][:]=[-1.,1.]
        a['get_dof_max_velocities'][:]=1.
    if configure:configure(f)
    old=f.old.__class__(hands=f.hands,clock_reader=lambda i:f.now[i],source_reader=f.p.source_reader,enabled=True)
    a=old.capture_batch(clocks=f.now)
    s=f.p.capture_batch(clocks=f.now,articulation_samples=a)[0]
    k=importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_mounted_kinematics')
    h={side:hypothesis(k,s,side) for side in ('left','right')};u=np.zeros((8,24))
    if cv:forecast=D.measured_constant_velocity_forecast(s,actions=u,model_source='CPU actual qd frozen, candidate-independent')
    else:
        forecast=D.make_velocity_forecast(s,actions=u,values=np.zeros((64,2,17)) if velocity is None else velocity,
            valid_mask=np.ones((64,2,17),dtype=bool),model_source='CPU declared future full17 only; unqualified',
            model_kind='external_candidate_conditioned',candidate_independent=False)
    r=D.rollout_kinematic_h8(s,current_clock=s.clock,actions=u,coordinate_hypotheses=h,forecast=forecast)
    return s,a[0],u,r


def command_trace(ctx,*,initial_finger=0.,actions=None,scope='supported_conditional_shadow'):
    from go2_pvcnn.control.dh116 import tabletop_bimanual_mpc_command_bridge as B
    from go2_pvcnn.control.dh116 import tabletop_vision_command_shadow as S
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_mounted_kinematics import descriptor_digest
    sample,_,u,_=ctx
    if actions is not None:u=actions
    state=dict(joint_command_rad=[initial_finger]*12,palm_command_offset_m=[[0.]*3]*2,wrist_command_angle_rad=[[0.]*3]*2)
    pose=[[0.,0.,0.,1.,0.,0.,0.]]*2
    current=dict(root_seed_pose=pose,prior_root_target_pose=pose,clearance_m=[.05]*2,
        task_offset_m=0.,clearance_floor_m=.0021,live=True,allowed=True,joint_lower=[-1.]*12,
        joint_upper=[1.]*12,teacher_force_controlled_digits=[[False]*5]*2)
    receipts={k:dict(clock=asdict(sample.clock),source='CPU recorded cache/context '+k,unit=B._UNITS[k],valid=True,
        descriptor_sha256=descriptor_digest(sample)) for k in (*state,*current)}
    profile=dict(version='dh116-conditional-command-v1',profile='surface_free_wrist_v1',
        action_contract=deepcopy(ACTION_CONTRACT),free_wrist_contract=deepcopy(CONTRACT),
        surface_policy_owned_targets=True,pinch_force_hold=False,redistribution=False,callbacks=[],physics_substeps=8,
        action_contract_sha256=S.action_contract_sha256(),controller_source_sha256=S.controller_source_sha256(),
        source_clock=asdict(sample.clock),descriptor_sha256=descriptor_digest(sample),source='CPU recorded controller contract')
    if scope=='kernel_subchain_diagnostic':profile.update(profile='pressure_teacher_recorded_profile',callbacks=['actual_cpu_callback_name'])
    return B.unfold_command_h8(sample,current_clock=sample.clock,actions=u,command_state=state,current_feedback=current,
        source_receipts=receipts,feedback_models=B.freeze_feedback_models(sample,actions=u,source='explicit frozen current feedback'),
        controller_contract=profile,scope=scope)


def evaluate(m,ctx,**kwargs):
    s,a,u,r=ctx
    return m.evaluate_h8_constraints(s,a,current_clock=s.clock,actions=u,rollout=r,**kwargs)


def test_complete_joint_arithmetic_remains_uncertified_with_fixed_required_gaps_and_no_action():
    m=api();out=evaluate(m,context())
    assert out.mathematical_checks_passed is True
    assert out.constraints['full17_position_bounds'].status=='PASS'
    assert out.constraints['full17_velocity_cap_frozen'].status=='PASS'
    assert out.required_constraint_names==m.REQUIRED_CONSTRAINTS
    for name in ('future_drive_effort','contact_state','object_dynamics','bilateral_support','continuous_collision','command_slew'):
        assert out.constraints[name].required is True and out.constraints[name].status in ('NA','UNCERTIFIED')
        assert not out.constraints[name].valid and out.constraints[name].missing_reason
    assert out.feasible is False and out.action is None and not any(out.qualification.values())
    with pytest.raises(TypeError):out.constraints['contact_state']=out.constraints['full17_position_bounds']


def test_all_64_intervals_catch_transient_passive_violation_when_final_q_returns_inside():
    v=np.zeros((64,2,17));v[31,1,6]=.96;v[32,1,6]=-.96
    def configure(f):f.hands['right'].root_physx_view.arrays['get_dof_limits'][1,6]=[-.001,.001]
    m=api();ctx=context(velocity=v,configure=configure);out=evaluate(m,ctx)
    t=out.constraints['full17_position_bounds']
    assert ctx[-1].states[-1].q_sdk17['right'][6]==0.
    assert t.status=='VIOLATION' and not out.mathematical_checks_passed
    assert t.coverage['poststates']==64 and t.coverage['intervals']==64
    witness=t.witness_by_unit['rad']
    assert witness['substep_index']==32 and witness['side']=='right' and witness['dof_index']==6
    assert witness['clock']==_freeze(asdict(ctx[0].clock))
    assert witness['interval_start_substep']==31 and witness['interval_end_substep']==32
    assert t.min_margin_by_unit['rad']==pytest.approx(-.001)


def test_current_selected_sdk_row_and_mixed_units_are_never_aggregated_or_env_index_guessed():
    def configure(f):
        for h in f.hands.values():
            a=h.root_physx_view.arrays
            a['get_dof_limits'][0]=np.nan;a['get_dof_max_velocities'][0]=np.nan
            a['get_dof_limits'][1,0]=[-.02,.02] # wrist yaw is rotational in this reversed layout
            a['get_dof_limits'][1,3]=[-.003,.003] # wrist z is translational
    m=api();ctx=context(configure=configure);out=evaluate(m,ctx)
    p=out.constraints['full17_position_bounds'];v=out.constraints['full17_velocity_cap_frozen']
    assert p.status==v.status=='PASS'
    assert set(p.min_margin_by_unit)=={'m','rad'} and set(v.min_margin_by_unit)=={'m/s','rad/s'}
    assert p.min_margin_by_unit['m']==pytest.approx(.003)
    assert p.min_margin_by_unit['rad']==pytest.approx(.02)
    assert p.witness_by_unit['m']['selected_view_index']==1


def test_velocity_cap_is_a_frozen_current_readback_hypothesis_and_covers_last_interval():
    v=np.zeros((64,2,17));v[-1,0,3]=.75
    def configure(f):f.hands['left'].root_physx_view.arrays['get_dof_max_velocities'][1,3]=.5
    m=api();out=evaluate(m,context(velocity=v,configure=configure));t=out.constraints['full17_velocity_cap_frozen']
    assert t.status=='VIOLATION' and t.min_margin_by_unit['m/s']==pytest.approx(-.25)
    assert t.witness_by_unit['m/s']['substep_index']==64
    assert t.coverage['intervals']==64 and t.assumption=='current SDK velocity caps frozen over H8; no future drive certification'


def test_honest_unavailable_limits_remain_na_without_falling_back_to_raw_or_default_box():
    m=api();ctx=context();s,a,u,r=ctx;fields=dict(a.fields);field=fields['left.limits']
    fields['left.limits']=replace(field,value=None,valid=False,kind='unavailable',timestamp_s=None,missing_reason='actual limits unavailable')
    out=evaluate(m,(s,replace(a,fields=fields),u,r));t=out.constraints['full17_position_bounds']
    assert t.status=='NA' and not t.valid and not t.min_margin_by_unit and 'unavailable' in t.missing_reason
    assert t.source['left']=='hand[left].root_physx_view.get_dof_limits()'
    assert not out.mathematical_checks_passed and out.action is None


@pytest.mark.parametrize('fault',['candidate','clock','source','initial_q','state_recurrence','interval_order','endpoint','qualification'])
def test_rollout_binding_or_timing_forgery_is_rejected_instead_of_reporting_math_pass(fault):
    m=api();s,a,u,r=context()
    if fault=='candidate':u=u.copy();u[7,23]=.2
    elif fault=='clock':r=replace(r,source_clock=replace(s.clock,episode_uid='other'))
    elif fault=='source':
        rec=_thaw(r.forecast_receipt);rec['source_bindings']['left']['source_identity']['stage_identity']='other'
        r=replace(r,forecast_receipt=_freeze(rec))
    elif fault=='initial_q':
        initial=_thaw(r.initial_current_actual);initial['q_sdk17']['left'][0]=.2;r=replace(r,initial_current_actual=_freeze(initial))
    elif fault=='qualification':r=replace(r,qualification={'planning_eligible':True})
    else:
        steps=list(r.states);bad=steps[40]
        if fault=='state_recurrence':
            q=_thaw(bad.q_sdk17);q['left'][5]=.1;bad=replace(bad,q_sdk17=_freeze(q))
        elif fault=='interval_order':bad=replace(bad,substep_index=40)
        else:bad=replace(bad,control_endpoint=True)
        steps[40]=bad;r=replace(r,states=tuple(steps))
    with pytest.raises(ValueError):evaluate(m,(s,a,u,r))


@pytest.mark.parametrize('fault',['clock','dtype','unit','selected_value','source_bracket'])
def test_articulation_limits_require_actual_same_owned_selected_row_and_source(fault):
    m=api();s,a,u,r=context();raw=deepcopy(a.raw);fields=dict(a.fields)
    if fault=='clock':a=replace(a,clock=replace(a.clock,reset_generation=1))
    elif fault=='unit':fields['left.limits']=replace(fields['left.limits'],unit='deg')
    elif fault=='selected_value':fields['left.limits']=replace(fields['left.limits'],value=[[-2.,2.]]*17)
    elif fault=='dtype':raw['hands']['left']['getters']['get_dof_limits']['raw_dtype']='unknown'
    else:raw['hands']['left']['source_brackets']['after']['value']['stage_identity']='drifted'
    a=replace(a,raw=raw,fields=fields)
    with pytest.raises(ValueError):evaluate(m,(s,a,u,r))


def test_candidate_independent_cv_cannot_get_action_response_or_joint_feasibility_credit():
    m=api();out=evaluate(m,context(cv=True));t=out.constraints['action_response']
    assert out.mathematical_checks_passed and t.status=='NA' and not t.valid
    assert 'candidate-independent' in t.missing_reason and out.feasible is False and out.action is None


def test_caller_cannot_disable_required_constraints_or_pass_untyped_command_dictionary():
    m=api();ctx=context()
    with pytest.raises(TypeError):evaluate(m,ctx,required=())
    with pytest.raises(ValueError):evaluate(m,ctx,command_trace={'guards_executed':True,'action_safe':True})


def test_inputs_preserved_and_sources_are_owned_immutable_receipts():
    m=api();ctx=context();s,a,u,r=ctx;before=deepcopy(a.raw)
    out=evaluate(m,ctx)
    assert a.raw==before and not u.any()
    a.raw['hands']['left']['source_identity']['stage_identity']='caller mutation after report'
    assert out.source_bindings['left']['source_identity']['stage_identity']=='actual-fixture-stage'
    assert out.source_clock==s.clock and out.candidate_digest==r.candidate_digest


def test_typed_original_command_bridge_is_replayed_and_reported_without_execution_credit():
    m=api();ctx=context();trace=command_trace(ctx);out=evaluate(m,ctx,command_trace=trace)
    for name in ('command_position_bounds','command_slew','command_lead'):
        assert out.constraints[name].status=='PASS' and out.constraints[name].coverage['intervals']==64
    assert out.constraints['command_slew'].source['rate_rad_s']==.3
    assert out.constraints['command_lead'].source['lead_rad']==.1
    assert out.constraints['command_slew'].coverage['scope']=='logical active-finger command; not actual qd'
    assert out.constraints['original_guard_equivalence'].status=='UNCERTIFIED'
    assert out.command_receipt['replayed_from_original_bridge'] is True
    assert out.forecast_receipt==ctx[-1].forecast_receipt
    assert out.action is None and out.feasible is False and not any(out.qualification.values())


def test_replayed_command_slew_does_not_hide_a_valid_old_cache_outside_current_q_lead():
    m=api();ctx=context();out=evaluate(m,ctx,command_trace=command_trace(ctx,initial_finger=.6))
    assert out.constraints['command_slew'].status=='PASS'
    lead=out.constraints['command_lead']
    assert lead.status=='VIOLATION' and lead.min_margin_by_unit['rad']<-.49
    assert lead.witness_by_unit['rad']['substep_index']==1
    assert not out.mathematical_checks_passed and out.action is None


@pytest.mark.parametrize('fault',['named_target','named_index','source','initial','pre_q','late_step'])
def test_typed_command_forgery_or_different_future_q_hypothesis_cannot_get_guard_credit(fault):
    m=api();ctx=context();trace=command_trace(ctx)
    if fault=='source':
        receipt=_thaw(trace.receipt);receipt['descriptor_sha256']='0'*64;trace=replace(trace,receipt=_freeze(receipt))
    elif fault=='initial':
        initial=_thaw(trace.initial_current);initial['q_sdk17'][0][0]=.1;trace=replace(trace,initial_current=_freeze(initial))
    else:
        steps=list(trace.steps);x=steps[63]
        if fault in ('named_target','named_index'):
            direct=_thaw(x.named_sdk_float32)
            if fault=='named_target':direct['left']['values'][0]=.9
            else:direct['right']['sdk_dof_indices'][0]=direct['right']['sdk_dof_indices'][1]
            x=replace(x,named_sdk_float32=_freeze(direct))
        elif fault=='pre_q':
            feedback=_thaw(x.feedback_hypotheses);feedback['q_sdk17'][1][16]=.05;x=replace(x,feedback_hypotheses=_freeze(feedback))
        else:x=replace(x,prediction_offset_s=0.)
        steps[63]=x;trace=replace(trace,steps=tuple(steps))
    with pytest.raises(ValueError):evaluate(m,ctx,command_trace=trace)


def test_honest_command_and_rollout_future_q_mismatch_is_rejected_even_when_each_is_individually_valid():
    m=api();v=np.zeros((64,2,17));v[:,:,16]=.01;ctx=context(velocity=v)
    trace=command_trace(ctx)
    with pytest.raises(ValueError,match='pre-state'):evaluate(m,ctx,command_trace=trace)


def test_command_target_bounds_include_moving_wrist_sdk_coordinates_without_asserting_workspace_safety():
    def configure(f):f.hands['left'].root_physx_view.arrays['get_dof_limits'][1,5]=[-.0001,.0001]
    m=api();ctx=context(configure=configure);s,a,u,r=ctx;u[:,12]=1.
    # This zero-response hypothesis is still unqualified; commands never become actual q.
    r=replace(r,candidate_digest=D.canonical_candidate_digest(u),
        forecast_receipt=_freeze(dict(_thaw(r.forecast_receipt),candidate_digest=D.canonical_candidate_digest(u))))
    ctx=(s,a,u,r);trace=command_trace(ctx);out=evaluate(m,ctx,command_trace=trace)
    assert out.constraints['command_position_bounds'].status=='VIOLATION'
    w=out.constraints['command_position_bounds'].witness_by_unit['m']
    assert w['side']=='left' and w['dof_name']=='wrist_x' and w['substep_index']==64
    assert out.constraints['moving_wrist_workspace'].status=='NA' and out.action is None


def test_complete_teacher_callbacks_stay_explicitly_uncertified_after_kernel_subchain_replay():
    m=api();ctx=context();trace=command_trace(ctx,scope='kernel_subchain_diagnostic')
    out=evaluate(m,ctx,command_trace=trace)
    assert out.command_receipt['bridge_receipt']['controller_callbacks']==('actual_cpu_callback_name',)
    assert out.constraints['original_guard_equivalence'].status=='UNCERTIFIED'
    assert not out.qualification['execution_qualified'] and out.action is None


def test_command_replay_cannot_override_an_honest_na_current_limit_field_using_saved_raw():
    m=api();ctx=context();s,a,u,r=ctx;trace=command_trace(ctx);fields=dict(a.fields)
    fields['right.limits']=replace(fields['right.limits'],value=None,valid=False,kind='unavailable',
        timestamp_s=None,missing_reason='current limit qualification unavailable')
    out=evaluate(m,(s,replace(a,fields=fields),u,r),command_trace=trace)
    assert out.constraints['command_position_bounds'].status=='NA'
    assert out.constraints['full17_position_bounds'].status=='NA' and not out.mathematical_checks_passed
    assert out.constraints['command_lead'].status=='PASS' and out.action is None


@pytest.mark.parametrize('fault',['before_bool','after_bool','before_float','after_float',
    'missing_own','duplicate_own','after_error'])
def test_global_articulation_brackets_require_typed_sample_clocks_not_equal_python_dicts(fault):
    m=api();s,a,u,r=context();raw=_thaw(a.raw);receipt=raw['read_brackets'][0]
    if fault.endswith('_bool'):
        own=next(c for c in receipt[fault.split('_')[0]] if c['env_id']==s.clock.env_id)
        own['reset_generation']=False # Python False == 0; SampleClock explicitly rejects it.
    elif fault.endswith('_float'):
        own=next(c for c in receipt[fault.split('_')[0]] if c['env_id']==s.clock.env_id)
        own['sdk_step']=float(own['sdk_step'])
    elif fault=='missing_own':receipt['before']=[c for c in receipt['before'] if c['env_id']!=s.clock.env_id]
    elif fault=='duplicate_own':receipt['after'].append(deepcopy(next(c for c in receipt['after'] if c['env_id']==s.clock.env_id)))
    else:receipt['after_error']='actual bracket post-read failed'
    with pytest.raises(ValueError,match='bracket'):evaluate(m,(s,replace(a,raw=_freeze(raw)),u,r))


@pytest.mark.parametrize('target',['forecast','initial','descriptor','q_receipt','qd_receipt'])
@pytest.mark.parametrize('fault',['bool','float','missing'])
def test_every_consumed_original_clock_receipt_requires_complete_typed_sample_clock(target,fault):
    from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_mounted_kinematics import descriptor_digest
    m=api();s,a,u,r=context()
    if target=='forecast':payload=_thaw(r.forecast_receipt);clock=payload['source_clock']
    elif target=='initial':payload=_thaw(r.initial_current_actual);clock=payload['clock']
    else:
        payload=_thaw(s.descriptor)
        clock=payload['clock'] if target=='descriptor' else payload['hands']['left']['measurement_receipts'][target.split('_')[0]]['clock']
    if fault=='bool':clock['reset_generation']=False
    elif fault=='float':clock['sdk_step']=float(clock['sdk_step'])
    else:clock.pop('reset') # Constructor defaults must not hide a missing saved field.
    if target=='forecast':r=replace(r,forecast_receipt=_freeze(payload))
    elif target=='initial':r=replace(r,initial_current_actual=_freeze(payload))
    else:
        s=replace(s,descriptor=_freeze(payload))
        receipt=_thaw(r.forecast_receipt);receipt['descriptor_sha256']=descriptor_digest(s)
        r=replace(r,forecast_receipt=_freeze(receipt))
    with pytest.raises(ValueError,match='clock'):evaluate(m,(s,a,u,r))


@pytest.mark.parametrize('target',['forecast','initial'])
@pytest.mark.parametrize('replacement',[True,1.])
def test_original_binding_identity_cannot_relabel_sdk_index_as_equal_bool_or_float(target,replacement):
    m=api();s,a,u,r=context()
    payload=_thaw(r.forecast_receipt if target=='forecast' else r.initial_current_actual)
    payload['source_bindings']['left']['binding']['selected_view_index']=replacement
    r=replace(r,**{('forecast_receipt' if target=='forecast' else 'initial_current_actual'):_freeze(payload)})
    with pytest.raises(ValueError):evaluate(m,(s,a,u,r))


@pytest.mark.parametrize('fault',['binding_bool','binding_float','source_identity_int'])
def test_original_articulation_copy_must_match_canonical_descriptor_binding_and_source_exactly(fault):
    from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_mounted_kinematics import descriptor_digest
    m=api();s,a,u,r=context();raw=_thaw(s.raw);actual=_thaw(a.raw)
    for hand in (actual['hands']['left'],raw['hands']['left']['articulation_input']):
        if fault.startswith('binding_'):
            values=hand['binding']['hand_joint_to_dof'];index=values.index(1)
            values[index]=True if fault=='binding_bool' else 1.
        else:hand['source_identity']['meters_per_unit']=1
    s=replace(s,raw=_freeze(raw));a=replace(a,raw=_freeze(actual))
    receipt=_thaw(r.forecast_receipt);receipt['descriptor_sha256']=descriptor_digest(s)
    r=replace(r,forecast_receipt=_freeze(receipt))
    with pytest.raises(ValueError,match='binding|source'):evaluate(m,(s,a,u,r))
