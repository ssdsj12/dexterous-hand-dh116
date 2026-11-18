import copy
import pytest
from types import SimpleNamespace
import torch
from test_dh116_policy_control import enabled
from go2_pvcnn.control.dh116.tabletop_policy_lift_hold import PolicyLiftHold, CONTRACT


def configured(n=2):
    t=enabled(n)
    t.policy_lift_hold=PolicyLiftHold(n,'cpu')
    t.last_observation=t.observe().clone()
    return t


def rewarded():
    from go2_pvcnn.control.dh116.tabletop_stable_lift_reward import StableLiftReward
    t=accepted()
    t.stable_lift_reward=StableLiftReward(t.n,t.device)
    t.terms['stable_lift_hold']=torch.zeros_like(t.reward)
    t.last_observation=t.observe().clone()
    return t


def lifted_measurement(t,height=.015):
    m={k:v.clone() for k,v in t.m.items()}
    m['bottom'][:]=.30+height
    return m


def test_stable_lift_reward_replaces_old_height_term_and_is_logged():
    t=rewarded();t.begin_step(torch.zeros((2,20)))
    t.consume(lifted_measurement(t),~t.done,.01,vertical_force_n=torch.ones(2))
    assert t.terms['lift'][0]==pytest.approx(2.5)
    assert t.terms['stable_lift_hold'][0]==pytest.approx(.02)
    before=t.terms['lift'].clone()
    t.consume(lifted_measurement(t),~t.done,.01,vertical_force_n=torch.ones(2))
    torch.testing.assert_close(t.terms['lift'],before)
    _,reward,_,extras=t.end_step()
    torch.testing.assert_close(reward,sum(extras['terms'].values()))
    assert t.observe().shape[1]==t.last_observation.shape[1]
    assert t.stable_lift_reward.observe().any()
    t.reset(torch.tensor([0]),t.m)
    assert not t.stable_lift_reward.observe()[0].any()
    assert t.stable_lift_reward.observe()[1].any()


def test_grip_loss_disables_original_ungated_height_reward():
    t=rewarded();m=lifted_measurement(t)
    for key in ('object_link_loads','grouped_object_loads','hand_object_loads'):
        m[key].zero_()
    t.consume(m,~t.done,.01,vertical_force_n=torch.ones(2))
    assert not t.terms['lift'].any()
    assert not t.terms['stable_lift_hold'].any()


def test_substep_terminal_failure_cannot_receive_stable_lift_credit():
    t=rewarded();m=lifted_measurement(t);m['clearance'][:]=-.002
    t.consume(m,~t.done,.01,vertical_force_n=torch.ones(2))
    assert (t.failure!=0).all()
    assert not t.terms['lift'].any() and not t.terms['stable_lift_hold'].any()


def accepted():
    t=configured()
    t.policy_stiffness.fill_(16.)
    t.contact_time.fill_(.2)
    m={k:v.clone() for k,v in t.m.items()}
    m['commanded_palm_offset_m']=torch.full((2,2,3),.0001,dtype=torch.float64)
    t.consume(m,~t.done,1/480,vertical_force_n=torch.ones(2))
    assert (t.phase==3).all()
    assert t.policy_lift_hold.active.all()
    return t


def test_real_accepted_transition_captures_applied_offset_and_targets_once():
    t=accepted()
    torch.testing.assert_close(t.policy_lift_hold.command,t.command)
    torch.testing.assert_close(t.policy_lift_hold.palm_offset,
                               torch.full_like(t.palm_targets,.0001))
    saved=t.policy_lift_hold.command.clone()
    t.consume(t.m,~t.done,1/480,vertical_force_n=torch.ones(2))
    torch.testing.assert_close(t.policy_lift_hold.command,saved)


def test_lift_random_policy_cannot_open_grasp_or_reduce_captured_stiffness():
    a=accepted(); b=copy.deepcopy(a)
    a.control(-torch.ones((2,20),dtype=torch.float64),~a.done,1/480)
    b.control(torch.ones((2,20),dtype=torch.float64),~b.done,1/480)
    torch.testing.assert_close(a.command,b.command)
    torch.testing.assert_close(a.policy_stiffness,b.policy_stiffness)
    torch.testing.assert_close(a.policy_stiffness,torch.full_like(a.policy_stiffness,16.))


def test_close_policy_remains_in_control_before_capture():
    a=configured();b=copy.deepcopy(a)
    a.control(-torch.ones((2,20),dtype=torch.float64),~a.done,1/480)
    b.control(torch.ones((2,20),dtype=torch.float64),~b.done,1/480)
    assert not a.policy_lift_hold.active.any()
    assert not torch.equal(a.command,b.command)


def test_held_grasp_does_not_disable_overpressure_release():
    t=accepted(); before=t.command.clone()
    t.m['hand_object_loads'].fill_(32.)
    allowed=t.control(torch.ones((2,20),dtype=torch.float64),~t.done,1/480)
    assert not allowed.any() and t.pressure_limited.all()
    sign=torch.where(t.direction==0,1.,t.direction.sign()).flatten(1)
    assert (((t.command-before)*sign)<=1e-12).all()


def test_selected_reset_clears_only_one_grasp_and_observation_remains_consistent():
    t=accepted(); saved=t.policy_lift_hold.command[1].clone()
    old_size=t.last_observation.shape[-1]
    t.reset([0],t.m)
    assert t.policy_lift_hold.active.tolist()==[False,True]
    torch.testing.assert_close(t.policy_lift_hold.command[1],saved)
    assert t.observe().shape[-1]==old_size
    assert torch.isfinite(t.observe()).all()


def test_world_recovery_cancels_pending_upward_drive_at_measured_offset():
    from test_dh116_tensor_world import palm_world
    w,t=palm_world();w.device='cpu'
    del t.coordinated_control
    t.policy_control=True;t.phase=torch.full((3,),3)
    t.policy_stiffness=w.gains.clone()
    t.palm_targets=torch.full((3,2,3),.002,dtype=torch.float64)
    w.palms.copy_(t.palm_targets)
    actual=torch.full_like(w.palms,.001)
    t.m={'adaptive_palm_offset':actual}
    t.paused=torch.tensor([True,False,False])
    t.policy_lift_hold=SimpleNamespace(active=torch.ones(3,dtype=torch.bool),paused=t.paused.clone())
    w.roots[:,:,:3]=w.root_seed[:,:,:3]+actual
    w.advance(t,torch.zeros(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool))
    torch.testing.assert_close(w.palms[0],actual[0])
    torch.testing.assert_close(w.palms[1],torch.full_like(w.palms[1],.002))
