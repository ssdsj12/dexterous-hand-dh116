"""CPU-only experimental load-transfer requests, never Native qualification."""
from dataclasses import replace
import importlib
from pathlib import Path

import pytest

from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import SampleClock,StateField


def api():
    p=Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_bimanual_load_transfer.py'
    assert p.is_file(),'independent pure load-transfer state machine is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_load_transfer')


def clock(k,env=0,uid='ep0',generation=0,**changes):
    values=dict(run_uuid='CPU-load-transfer',env_id=env,episode_uid=uid,world_step=480+k,
        world_time_s=1+k/480,sdk_step=1480+k,sdk_time_s=11+k/480,policy_step=60+k//8,
        reset_generation=generation)
    values.update(changes);return SampleClock(**values)


def feedback(m,k,*,env=0,uid='ep0',generation=0,style='pinch_thumb_index',terminal=False,**changes):
    c=clock(k,env,uid,generation,terminal=terminal)
    values=dict(approach_ready=True,bilateral_opposed=[True,True],tracking_safe=True,object_following=True,
        tilt_safe=True,slip_safe=True,support_safe=True,airborne=True,transfer_complete=False,
        recovery_ready=False,mandatory_unload=False,fatal_failure=False,original_hold_qualified=False,
        terminal_success=False,hand_upward_support=[1.,1.],table_reaction=[0.,0.,0.],
        end_support_reaction=[0.,0.,0.],net_lift_m=0.,fixed_anchor_drift_m=0.,original_hold_duration_s=0.)
    values.update(changes);fields={}
    for name,value in values.items():
        unit='N' if name in ('hand_upward_support','table_reaction','end_support_reaction') else 'm' if name in (
            'net_lift_m','fixed_anchor_drift_m') else 's' if name=='original_hold_duration_s' else 'dimensionless'
        force='ON_OBJECT' if unit=='N' else 'NA';frame='world' if unit in ('N','m') else 'not_applicable'
        basis='actual_contact' if name in ('bilateral_opposed','hand_upward_support','support_safe') else 'not_applicable'
        kind='measured' if name in ('table_reaction','end_support_reaction','net_lift_m') else 'derived'
        fields[name]=StateField(value,'CPU explicit original-measurement criterion '+name,unit,frame,
            c.world_time_s,True,c.env_id,c.episode_uid,c,kind=kind,
            dependencies=() if kind=='measured' else ('CPU actual source: '+name,),force_side=force,contact_basis=basis)
    return m.LoadTransferFeedback(c,style,fields)


def reach_transfer(m,controller,*,start=0,env=0):
    assert controller.step(feedback(m,start,env=env)).phase==m.Phase.APPROACH
    assert controller.step(feedback(m,start+1,env=env)).phase==m.Phase.CLOSE
    assert controller.step(feedback(m,start+2,env=env)).phase==m.Phase.STABILIZE
    for k in range(start+3,start+98):
        out=controller.step(feedback(m,k,env=env));assert out.phase==m.Phase.STABILIZE
        assert out.common_ascent_request_m_s==0.
    out=controller.step(feedback(m,start+98,env=env))
    assert out.phase==m.Phase.LOAD_TRANSFER
    return start+98,out


def test_default_disabled_has_no_motion_or_actor_sdk_or_success_authority():
    m=api();c=m.BimanualLoadTransfer();out=c.step(feedback(m,0))
    assert out.enabled is False and out.common_ascent_request_m_s==0.
    assert out.hold_issued is False and not any(out.qualification.values())
    assert out.actor_observation_changed is False and out.sdk_targets_issued is False


@pytest.mark.parametrize('style',['pinch_thumb_index','tripod_thumb_index_middle'])
def test_actual_96_contiguous_intervals_before_transfer_and_independent_requested_exposure_cap(style):
    m=api();c=m.BimanualLoadTransfer(enabled=True,grasp_style=style)
    # Style is explicit; the externally sourced opposition predicate owns the pad criterion.
    def f(k):return feedback(m,k,style=style)
    c.step(f(0));c.step(f(1));c.step(f(2))
    for k in range(3,98):assert c.step(f(k)).common_ascent_request_m_s==0.
    out=c.step(f(98));assert out.stable_contact_duration_s==pytest.approx(.2)
    assert out.common_ascent_request_m_s==.0005
    for k in range(99,2100):out=c.step(f(k))
    assert out.phase==m.Phase.LOAD_TRANSFER and out.common_ascent_request_m_s==0.
    assert out.transfer_requested_exposure_m==.002 and out.request_measured_hold is True
    assert out.transfer_is_measured_displacement is False


def test_external_transfer_complete_then_default_lift_and_original_terminal_success_are_distinct():
    m=api();c=m.BimanualLoadTransfer(enabled=True);k,out=reach_transfer(m,c)
    out=c.step(feedback(m,k+1,transfer_complete=True));assert out.phase==m.Phase.LIFT
    assert out.common_ascent_request_m_s==.001
    out=c.step(feedback(m,k+2,net_lift_m=.03));assert out.phase==m.Phase.AIRBORNE_HOLD
    assert out.request_measured_hold and not out.finished and not out.hold_issued
    out=c.step(feedback(m,k+3,terminal=True,net_lift_m=.03,fixed_anchor_drift_m=.0005,
        original_hold_duration_s=5.,original_hold_qualified=True,terminal_success=True))
    assert out.phase==m.Phase.AIRBORNE_HOLD and out.finished and out.original_observer_reported_success is True
    assert not any(out.qualification.values())


@pytest.mark.parametrize('changes',[dict(object_following=False),dict(bilateral_opposed=[True,False]),
    dict(tracking_safe=False),dict(slip_safe=False),dict(tilt_safe=False),dict(support_safe=False),dict(mandatory_unload=True)])
def test_motion_loss_requests_original_measured_hold_and_pressure_without_issuing_safe_hold(changes):
    m=api();c=m.BimanualLoadTransfer(enabled=True);k,before=reach_transfer(m,c)
    out=c.step(feedback(m,k+1,**changes))
    assert out.phase==m.Phase.RECOVERY and out.common_ascent_request_m_s==0.
    assert out.request_measured_hold and out.request_pressure_recovery and out.preserve_original_pressure_and_unload
    assert not out.hold_issued and out.transfer_requested_exposure_m==before.transfer_requested_exposure_m
    out=c.step(feedback(m,k+2,recovery_ready=True));assert out.phase==m.Phase.STABILIZE
    assert out.stable_contact_duration_s==0.
    assert out.transfer_requested_exposure_m==before.transfer_requested_exposure_m


@pytest.mark.parametrize('missing',['table_reaction','end_support_reaction','airborne','object_following','hand_upward_support'])
def test_na_never_becomes_zero_or_a_forward_motion_gate(missing):
    m=api();c=m.BimanualLoadTransfer(enabled=True);k,old=reach_transfer(m,c);f=feedback(m,k+1)
    fields=dict(f.fields);v=fields[missing];fields[missing]=replace(v,value=None,valid=False,kind='unavailable',
        timestamp_s=None,missing_reason='CPU source unavailable')
    out=c.step(replace(f,fields=fields))
    assert out.phase==m.Phase.RECOVERY and out.common_ascent_request_m_s==0.
    assert out.field_evidence[missing].value is None and out.field_evidence[missing].valid is False
    assert out.transfer_requested_exposure_m==old.transfer_requested_exposure_m


def test_wrong_clock_source_basis_reaction_kind_or_units_refuse_without_state_mutation():
    m=api();c=m.BimanualLoadTransfer(enabled=True);c.step(feedback(m,0));initial=c.phase
    for fault in ('clock','basis','reaction_kind','unit'):
        f=feedback(m,1);fields=dict(f.fields)
        if fault=='clock':fields['tracking_safe']=replace(fields['tracking_safe'],clock=clock(0),timestamp_s=1.)
        elif fault=='basis':fields['bilateral_opposed']=replace(fields['bilateral_opposed'],contact_basis='geometry_only')
        elif fault=='reaction_kind':fields['table_reaction']=replace(fields['table_reaction'],kind='derived',dependencies=('mass*g minus hand force',))
        else:fields['net_lift_m']=replace(fields['net_lift_m'],unit='mm')
        with pytest.raises(ValueError):c.step(replace(f,fields=fields))
        assert c.phase==initial


def test_duplicate_gap_cross_episode_and_policy_clock_do_not_accumulate_time():
    m=api();c=m.BimanualLoadTransfer(enabled=True);c.step(feedback(m,0))
    for f in (feedback(m,0),feedback(m,2),feedback(m,1,uid='other'),replace(feedback(m,1),clock=clock(1,policy_step=99))):
        with pytest.raises(ValueError):c.step(f)
    assert c.step(feedback(m,1)).phase==m.Phase.CLOSE


def test_reset_is_explicit_new_episode_and_clears_only_this_env_request_ledger():
    m=api();a=m.BimanualLoadTransfer(enabled=True);b=m.BimanualLoadTransfer(enabled=True)
    ka,oa=reach_transfer(m,a,env=0);kb,ob=reach_transfer(m,b,env=1)
    with pytest.raises(ValueError):a.reset(clock(ka+1),grasp_style='pinch_thumb_index')
    a.reset(clock(ka+1,uid='ep1',generation=1,reset=True),grasp_style='pinch_thumb_index')
    out=a.step(feedback(m,ka+2,uid='ep1',generation=1))
    assert out.phase==m.Phase.APPROACH and out.transfer_requested_exposure_m==0.
    assert b.transfer_requested_exposure_m==ob.transfer_requested_exposure_m


def test_terminal_unknown_is_not_fabricated_failure_and_first_terminal_cannot_be_overwritten():
    m=api();c=m.BimanualLoadTransfer(enabled=True);f=feedback(m,0,terminal=True)
    fields=dict(f.fields);fields['terminal_success']=replace(fields['terminal_success'],value=None,valid=False,
        timestamp_s=None,kind='unavailable',missing_reason='native outcome unavailable')
    out=c.step(replace(f,fields=fields))
    assert out.finished and out.phase==m.Phase.RECOVERY and out.original_observer_reported_success is None
    assert c.step(feedback(m,1,net_lift_m=99.,terminal_success=True)) is out
    assert out.clock.world_step==480


def test_unverified_terminal_success_cannot_lower_any_original_success_threshold():
    m=api()
    for values in (dict(net_lift_m=.029999),dict(original_hold_duration_s=4.99),dict(fixed_anchor_drift_m=.000501),dict(airborne=False)):
        c=m.BimanualLoadTransfer(enabled=True)
        args=dict(net_lift_m=.03,original_hold_duration_s=5.,fixed_anchor_drift_m=.0005,
            original_hold_qualified=True,terminal_success=True);args.update(values)
        out=c.step(feedback(m,0,terminal=True,**args))
        assert out.finished and out.phase==m.Phase.RECOVERY and not any(out.qualification.values())


def test_known_failed_terminal_or_fatal_is_failed_but_three_mm_rate_needs_independent_evidence():
    m=api();out=m.BimanualLoadTransfer(enabled=True).step(feedback(m,0,terminal=True,terminal_success=False))
    assert out.phase==m.Phase.FAILED and out.finished
    assert m.BimanualLoadTransfer(enabled=True).step(feedback(m,0,fatal_failure=True)).phase==m.Phase.FAILED
    with pytest.raises(ValueError):m.LoadTransferConfig(lift_rate_m_s=.003)
    with pytest.raises(ValueError):m.LoadTransferConfig(lift_rate_m_s=.011,lift_speed_evidence_source='CPU experiment',lift_speed_evidence_sha256='a'*64)
    cfg=m.LoadTransferConfig(lift_rate_m_s=.003,lift_speed_evidence_source='independent bounded experiment fixture',lift_speed_evidence_sha256='a'*64)
    c=m.BimanualLoadTransfer(enabled=True,config=cfg);k,out=reach_transfer(m,c)
    assert c.step(feedback(m,k+1,transfer_complete=True)).common_ascent_request_m_s==.003


def test_first_missing_or_unload_sample_requests_original_hold_without_crediting_an_interval():
    m=api()
    for fault in ('missing','unload','nonfinite'):
        c=m.BimanualLoadTransfer(enabled=True);f=feedback(m,0)
        if fault=='unload':f=feedback(m,0,mandatory_unload=True)
        elif fault=='nonfinite':f=feedback(m,0,net_lift_m=float('nan'))
        else:
            fields=dict(f.fields);fields['table_reaction']=replace(fields['table_reaction'],value=None,valid=False,
                timestamp_s=None,kind='unavailable',missing_reason='not measured')
            f=replace(f,fields=fields)
        out=c.step(f)
        assert out.phase==m.Phase.RECOVERY and out.request_measured_hold and out.request_pressure_recovery
        assert out.stable_contact_duration_s==0. and out.transfer_requested_exposure_m==0.
        assert out.hold_issued is False and out.preserve_original_pressure_and_unload


def test_retired_episode_uid_cannot_be_reused_for_a_new_request_budget():
    m=api();c=m.BimanualLoadTransfer(enabled=True);c.step(feedback(m,0))
    c.reset(clock(1,uid='ep1',generation=1,reset=True),grasp_style='pinch_thumb_index')
    c.step(feedback(m,2,uid='ep1',generation=1))
    with pytest.raises(ValueError):c.reset(clock(3,uid='ep0',generation=2,reset=True),grasp_style='pinch_thumb_index')


def test_feedback_and_decision_keep_owned_original_sources_immutable_without_action_commands():
    m=api();c=m.BimanualLoadTransfer(enabled=True);f=feedback(m,0);original=dict(f.fields)
    out=c.step(f)
    assert dict(f.fields)==original and out.field_evidence['table_reaction'] is f.fields['table_reaction']
    with pytest.raises(TypeError):f.fields['table_reaction']=None
    with pytest.raises(TypeError):out.field_evidence['table_reaction']=None
    with pytest.raises(TypeError):out.qualification['native_attested']=True
    assert not hasattr(out,'action24') and not hasattr(out,'sdk_target') and not any(out.qualification.values())


@pytest.mark.parametrize('fault',['rewind_times','step_delta_disagree','time_delta_disagree','policy_regression'])
def test_reset_rejects_same_world_time_inconsistency_atomically(fault):
    m=api();c=m.BimanualLoadTransfer(enabled=True);k,last=reach_transfer(m,c)
    bad=clock(k+1,uid='new-ep',generation=1,reset=True)
    if fault=='rewind_times':bad=replace(bad,world_time_s=0.,sdk_time_s=0.,policy_step=0)
    elif fault=='step_delta_disagree':bad=replace(bad,sdk_step=bad.sdk_step+1,sdk_time_s=bad.sdk_time_s+1/480)
    elif fault=='time_delta_disagree':bad=replace(bad,world_time_s=bad.world_time_s+1/480)
    else:bad=replace(bad,policy_step=c._clock.policy_step-1)
    before=(c.phase,c._clock,c.transfer_requested_exposure_m,c._stable_intervals,set(c._seen_episode_uids))
    with pytest.raises(ValueError):c.reset(bad,grasp_style='pinch_thumb_index')
    assert (c.phase,c._clock,c.transfer_requested_exposure_m,c._stable_intervals,set(c._seen_episode_uids))==before
    assert c.step(feedback(m,k+1)).transfer_requested_exposure_m>last.transfer_requested_exposure_m


def test_real_reset_may_span_many_or_zero_physical_steps_with_consistent_clocks():
    m=api();c=m.BimanualLoadTransfer(enabled=True);k,out=reach_transfer(m,c)
    c.reset(clock(k+17,uid='ep1',generation=1,reset=True),grasp_style='pinch_thumb_index')
    out=c.step(feedback(m,k+18,uid='ep1',generation=1));assert out.phase==m.Phase.APPROACH
    c.reset(clock(k+18,uid='ep2',generation=2,reset=True),grasp_style='pinch_thumb_index')
    out=c.step(feedback(m,k+19,uid='ep2',generation=2))
    assert out.phase==m.Phase.APPROACH and out.transfer_requested_exposure_m==0.
