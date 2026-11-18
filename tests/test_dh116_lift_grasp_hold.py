"""Lift retains the accepted grasp while feedback and safety stay active."""
import copy
import pytest
import torch

from test_dh116_coordinated_grasp import enabled
from test_dh116_tensor_contact_window import loads


def close_to_lift(n=2,feedback_offset=0.,palm_z=None):
    t=enabled(n)
    t.m=loads(t,4.)
    t.command.copy_(t.m['q'])
    t.f_target.fill_(2.);t.f_requested.fill_(2.);t.f_filtered.fill_(2.)
    t.f_offset.fill_(feedback_offset)
    t.contact_time.fill_(.2)
    t.palm_targets.fill_(.0004)  # Queued exploration request has not been reached.
    m={k:v.clone() for k,v in t.m.items()}
    m['commanded_palm_offset_m']=torch.full((n,2,3),.0001,dtype=torch.float64)
    if palm_z is not None:m['commanded_palm_offset_m'][:,:,2]=torch.tensor(palm_z,dtype=torch.float64)
    t.consume(m,~t.done,1/480,vertical_force_n=torch.ones(n),
              object_weight_n=torch.full((n,),1.1772))
    assert t.phase.tolist()==[3]*n
    return t


def test_lift_ignores_new_ppo_joint_force_and_palm_search_requests():
    a=close_to_lift();b=copy.deepcopy(a)
    low=-torch.ones((2,20),dtype=torch.float64)
    high=torch.ones((2,20),dtype=torch.float64)
    for _ in range(8):
        a.control(low,~a.done,1/480);b.control(high,~b.done,1/480)
    torch.testing.assert_close(a.command,b.command)
    torch.testing.assert_close(a.f_target,b.f_target)
    torch.testing.assert_close(a.f_target,torch.full_like(a.f_target,2.))
    torch.testing.assert_close(a.palm_targets,b.palm_targets)
    torch.testing.assert_close(a.offset,b.offset)


def test_lift_holds_applied_palm_offset_despite_finger_midpoint_motion():
    t=close_to_lift();t.m['pad_relative'][:,:,2]+=.002
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,1/480)
    torch.testing.assert_close(t.palm_targets,torch.full_like(t.palm_targets,.0001))


def test_lift_holds_closed_joint_command_when_force_matches_target():
    t=close_to_lift();before=t.command.clone();lift=t.offset.clone()
    t.control(-torch.ones((2,20),dtype=torch.float64),~t.done,1/480)
    torch.testing.assert_close(t.command,before,rtol=0,atol=1e-12)
    assert (t.offset>lift).all()


def test_close_still_explores_different_ppo_requests():
    a=enabled();b=copy.deepcopy(a)
    a.control(-torch.ones((2,20),dtype=torch.float64),~a.done,1/480)
    b.control(torch.ones((2,20),dtype=torch.float64),~b.done,1/480)
    assert not torch.equal(a.command,b.command)
    assert not torch.equal(a.f_target,b.f_target)


def test_retained_grasp_still_releases_overpressure_and_pauses_lift():
    t=close_to_lift();t.m=loads(t,32.)
    before=t.command.clone();offset=t.offset.clone()
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,1/480)
    assert t.pressure_limited.all()
    torch.testing.assert_close(t.offset,offset)
    sign=torch.where(t.direction==0,1.,t.direction.sign()).flatten(1)
    assert (((t.command-before)*sign)<=1e-12).all()
    assert (abs(t.command-before)<=.20/480+1e-12).all()


def test_nonzero_close_feedback_is_not_added_to_saved_command_twice():
    t=close_to_lift(feedback_offset=.05);before=t.command.clone()
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,1/480)
    torch.testing.assert_close(t.command,before,rtol=0,atol=1e-12)


@pytest.mark.parametrize('side',['lower','upper'])
def test_saved_grasp_outside_policy_envelope_keeps_mechanical_safety_bounds(side):
    t=close_to_lift();before=t.command.clone()
    t.envelope={};t.lower=t.low+.02;t.upper=t.high-.02
    if side=='lower':t.lower=t.command[0]+.02
    else:t.upper=t.command[0]-.02
    t.center=(t.lower+t.upper)/2
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,1/480)
    torch.testing.assert_close(t.command,before,rtol=0,atol=1e-12)
    assert (t.command>=t.low).all() and (t.command<=t.high).all()


def test_lift_feedback_can_restore_load_without_new_force_request_or_slip_boost():
    t=close_to_lift();t.m=loads(t,2.)
    t.f_filtered.fill_(1.);t.slip.fill_(.003);t.f_last_slip.zero_()
    before=t.f_offset.clone();command=t.command.clone();boost=t.f_boost.clone()
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,1/480)
    torch.testing.assert_close(t.f_offset,before)
    assert (((t.command-command).reshape(2,2,6)*t.direction)>0).any()
    torch.testing.assert_close(t.f_requested,torch.full_like(t.f_requested,2.+4./480))
    torch.testing.assert_close(t.f_target,t.f_requested)
    torch.testing.assert_close(t.f_boost,boost)


def test_hold_reset_and_recapture_affect_only_selected_environment():
    t=close_to_lift(palm_z=(-.00035,.0002));saved=t.lift_grasp_hold.command[1].clone()
    saved_palm=t.lift_grasp_hold.palm_offset[1].clone()
    m=loads(t,4.);t.reset([0],m)
    assert t.lift_grasp_hold.active.tolist()==[False,True]
    torch.testing.assert_close(t.lift_grasp_hold.command[1],saved)
    torch.testing.assert_close(t.lift_grasp_hold.palm_offset[1],saved_palm)
    t.command[0]+=.001;t.f_target[0]=3.;t.contact_time[0]=.2
    m['commanded_palm_offset_m']=torch.full((2,2,3),.0002,dtype=torch.float64)
    t.consume(m,~t.done,1/480,vertical_force_n=torch.ones(2),object_weight_n=torch.ones(2))
    assert t.lift_grasp_hold.active.all()
    torch.testing.assert_close(t.lift_grasp_hold.command[0],t.command[0])
    torch.testing.assert_close(t.lift_grasp_hold.command[1],saved)
    torch.testing.assert_close(t.lift_grasp_hold.palm_offset[1],saved_palm)
    assert t.lift_grasp_hold.force_target[:,0].tolist()==[3.,2.]
    assert t.lift_grasp_hold.palm_offset[:,0,0].tolist()==[.0002,.0001]


def test_coordinated_close_ignores_old_hold_flag_and_captures_immediate_lift():
    t=enabled();t.skip_hold=False;t.close_downward_penalty=False;t.m=loads(t,4.)
    t.contact_time.zero_()
    t.consume(t.m,~t.done,1/480,vertical_force_n=torch.ones(2),object_weight_n=torch.ones(2))
    assert (t.phase==3).all() and t.lift_grasp_hold.active.all()
    assert not t.contact_time.any()


def test_lift_hold_metadata_and_training_resume_are_exact():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_tensor_training import (
        apply_lift_grasp_hold_contract,checkpoint_contract_changes)
    from go2_pvcnn.control.dh116.tabletop_lift_grasp_hold import CONTRACT
    old={'num_envs':4096,'palm_exploration':{'lift_policy':'palm_zero_offset;task_offset_lifts_hand'}}
    new=copy.deepcopy(old)
    apply_lift_grasp_hold_contract(new,close_to_lift())
    assert new['lift_grasp_hold']==CONTRACT
    assert new['lift_grasp_hold']['lift_height_projection']=='per_hand_captured_offset_and_independent_clearance'
    assert new['palm_exploration']['lift_policy']=='hold_applied_palm_offset;task_offset_lifts_hand'
    untouched=copy.deepcopy(old);apply_lift_grasp_hold_contract(untouched,SimpleNamespace())
    assert untouched==old
    assert checkpoint_contract_changes(new,new)==[]
    for other in (old,dict(new,lift_grasp_hold=dict(CONTRACT,version='unknown'))):
        for a,b in ((other,new),(new,other)):
            with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(a,b)


@pytest.mark.parametrize('palm_z',[(.0001,.0001),(-.00035,.0002)])
@pytest.mark.parametrize('axis',['xyz','z_down'])
def test_first_lift_world_target_has_only_common_z_increment_with_carriage_lag(palm_z,axis):
    from test_dh116_tensor_world import palm_world
    w,_=palm_world();t=close_to_lift(3,palm_z=palm_z)
    t.palm_exploration_axis=axis
    w.palms.copy_(t.lift_grasp_hold.palm_offset)
    w.root_targets=w.root_seed.clone();w.root_targets[:,:,:3]+=w.palms
    w.root_targets[:,:,2]+=t.offset[:,None]
    w.roots=w.root_targets.clone();w.roots[:,:,2]-=.00005
    before=w.root_targets.clone();offset=t.offset.clone()
    t.m['adaptive_palm_offset'][:,:,2]-=.00005
    allowed=t.control(torch.ones((3,20),dtype=torch.float64),~t.done,w.dt)
    w.advance(t,allowed,~t.done)
    torch.testing.assert_close(w.root_targets[:,:,:2],before[:,:,:2],rtol=0,atol=1e-12)
    torch.testing.assert_close(w.root_targets[:,:,2]-before[:,:,2],
        (t.offset-offset)[:,None].expand(-1,2),rtol=0,atol=1e-12)
    torch.testing.assert_close(w.palms,t.lift_grasp_hold.palm_offset)


@pytest.mark.parametrize('waiting_hand',[0,1])
def test_one_ready_hand_cannot_finish_close_despite_long_dwell(waiting_hand):
    t=enabled();m=loads(t,8.)
    for key in ('hand_object_loads','object_link_loads','grouped_object_loads'):
        m[key][:,waiting_hand]*=.05
    t.contact_time.fill_(.2)
    for _ in range(240):
        t.consume(m,~t.done,1/480,vertical_force_n=torch.ones(2),object_weight_n=torch.ones(2))
    assert t.phase.tolist()==[1,1]
    assert not t.done.any()
    assert not t.lift_grasp_hold.active.any()
    torch.testing.assert_close(t.contact_time,torch.zeros_like(t.contact_time))
    # Restoring the current 1 N minimum admits lift on this substep.
    for key in ('hand_object_loads','object_link_loads','grouped_object_loads'):
        m[key][:,waiting_hand]*=5.
    t.consume(m,~t.done,1/480,vertical_force_n=torch.zeros(2),object_weight_n=torch.ones(2))
    assert t.phase.tolist()==[3,3]
    assert t.lift_grasp_hold.active.all()


def test_both_ready_hands_still_require_current_nonnegative_normal_force():
    t=enabled();m=loads(t,4.)
    t.contact_time.fill_(.2)
    for _ in range(48):
        t.consume(m,~t.done,1/480,vertical_force_n=-torch.ones(2),object_weight_n=torch.ones(2))
    assert t.phase.tolist()==[1,1]
    assert not t.lift_grasp_hold.active.any()
    torch.testing.assert_close(t.contact_time,torch.zeros_like(t.contact_time))
    t.consume(m,~t.done,1/480,vertical_force_n=torch.tensor([0.,.1]),object_weight_n=torch.ones(2))
    assert t.phase.tolist()==[3,3]
    assert t.lift_grasp_hold.active.all()


def test_rate_limited_previous_force_request_cannot_continue_ramping_in_lift():
    from dataclasses import replace
    t=close_to_lift();t.parameters=replace(t.parameters,force_rate_n_s=.5)
    t.f_requested.fill_(10.)
    for _ in range(8):t.control(torch.ones((2,20),dtype=torch.float64),~t.done,1/480)
    torch.testing.assert_close(t.f_target,torch.full_like(t.f_target,2.))
    torch.testing.assert_close(t.f_requested,t.f_target)


def test_severe_coupling_still_pauses_lift_and_slews_safely():
    t=close_to_lift();t.m['mimic_errors'][:,1]=-.013
    before=t.command.clone();offset=t.offset.clone()
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,1/480)
    assert t.paused.all()
    torch.testing.assert_close(t.offset,offset)
    assert ((t.command-before).abs()<=.20/480+1e-12).all()
    assert (t.command>=t.low).all() and (t.command<=t.high).all()


def test_inactive_lift_row_does_not_change_hold_or_control_state():
    t=close_to_lift();before=copy.deepcopy(t)
    t.m['hand_object_loads'][1]=20.
    t.control(torch.ones((2,20),dtype=torch.float64),torch.tensor([True,False]),1/480)
    for key in ('command','f_target','f_requested','f_offset','offset','palm_targets'):
        torch.testing.assert_close(getattr(t,key)[1],getattr(before,key)[1])
    for key,value in vars(t.lift_grasp_hold).items():
        torch.testing.assert_close(value[1],getattr(before.lift_grasp_hold,key)[1])


def test_failed_close_never_latches_a_grasp():
    t=enabled();t.m=loads(t,4.);t.contact_time[:]=.2
    t.m['clearance'][:]=-.002
    t.consume(t.m,~t.done,1/480,vertical_force_n=torch.ones(2),object_weight_n=torch.ones(2))
    assert t.done.all() and not t.lift_grasp_hold.active.any()


@pytest.mark.parametrize('invalid',[float('nan'),float('inf')])
def test_invalid_applied_palm_cannot_enter_lift_or_poison_saved_grasp(invalid):
    from go2_pvcnn.control.dh116.tabletop_tensor_task import FAILURE_CODES
    t=enabled();m=loads(t,4.);t.contact_time[:]=.2
    m['commanded_palm_offset_m']=torch.zeros((2,2,3),dtype=torch.float64)
    m['commanded_palm_offset_m'][0,0,2]=invalid
    t.consume(m,~t.done,1/480,vertical_force_n=torch.ones(2),object_weight_n=torch.ones(2))
    assert t.failure.tolist()==[FAILURE_CODES['nonfinite_state'],0]
    assert t.phase.tolist()==[1,3]
    assert t.lift_grasp_hold.active.tolist()==[False,True]
    assert torch.isfinite(t.lift_grasp_hold.palm_offset).all()


def test_inactive_or_legacy_row_does_not_validate_or_latch_unused_optional_palm():
    t=enabled(3);t.coordinated_control[1]=False;t.coordinated_all=False
    m=loads(t,4.);t.contact_time[:]=.2
    m['commanded_palm_offset_m']=torch.zeros((3,2,3),dtype=torch.float64)
    m['commanded_palm_offset_m'][1:]=float('nan')
    t.consume(m,torch.tensor([True,True,False]),1/480,
              vertical_force_n=torch.ones(3),object_weight_n=torch.ones(3))
    assert not t.failure.any()
    assert t.lift_grasp_hold.active.tolist()==[True,False,False]


def test_malformed_applied_palm_shape_is_rejected_before_transition():
    t=enabled();m=loads(t,4.);t.contact_time[:]=.2
    m['commanded_palm_offset_m']=torch.zeros((2,3),dtype=torch.float64)
    with pytest.raises(ValueError,match='applied palm offset'):
        t.consume(m,~t.done,1/480,vertical_force_n=torch.ones(2),object_weight_n=torch.ones(2))
    assert (t.phase==1).all() and not t.lift_grasp_hold.active.any()
