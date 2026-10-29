import torch
from test_phase_force_integration import rewarded


def free_task():
    from go2_pvcnn.control.dh116.tabletop_free_grasp_lift import enable
    t=rewarded();enable(t)
    return t


def consume(t,normal=-1.,dt=1/480):
    t.m['vertical_hand_force_n'].fill_(normal/2)
    t.consume(t.m,~t.done,dt,vertical_force_n=torch.full((2,),normal),
              lift_vertical_force_n=torch.full((2,),normal),
              hand_lift_force_n=torch.full((2,2),normal/2),object_weight_n=torch.ones(2))


def test_no_stage_feature_or_close_timeout_or_force_admission():
    base=rewarded();t=free_task()
    assert t.observe().shape[1]==base.observe().shape[1]-4
    t.elapsed.fill_(4.1);t.m['hand_object_loads'].zero_()
    t.m['object_link_loads'].zero_();t.m['grouped_object_loads'].zero_()
    t.m['contact_valid'].zero_();t.m['contact_all_below_centerline'].zero_()
    consume(t)
    assert not t.done.any() and not t.contact_time.any()
    assert t.free_grasp_lift and not hasattr(t,'policy_lift_hold')
    assert not t.grip_valid.any()
    for name in ('stage','contact_dwell','grasp_ready','close_downward'):
        assert not t.terms[name].any()


def test_upward_policy_request_allowed_without_contact_and_with_downward_force():
    t=free_task();t.m['hand_object_loads'].zero_()
    t.m['contact_valid'].zero_();t.m['contact_all_below_centerline'].zero_()
    t.m['vertical_hand_force_n'].fill_(-1.)
    a=torch.zeros((2,20),dtype=torch.float64);a[:,14]=1.;a[:,17]=1.
    assert t.control(a,~t.done,1/480).all()
    assert (t.palm_targets[:,:,2]>.04).all()


def test_force_rewards_evaluated_together_and_downward_cost_preserved():
    t=free_task();consume(t,normal=.8,dt=.01)
    assert (t.terms['grasp_upward_force']>0).all()
    assert (t.terms['lift_upward_force']>0).all()
    t=free_task();consume(t,normal=-.8,dt=.01)
    assert not t.terms['grasp_upward_force'].any()
    assert not t.terms['lift_upward_force'].any()
    torch.testing.assert_close(t.terms['downward_force'],torch.full_like(t.reward,-.004))


def test_selected_reset_keeps_continuous_mode_and_no_close_dwell():
    t=free_task();t.elapsed[:]=1.;t.reset([1],t.m)
    assert t.free_grasp_lift and t.phase.tolist()==[3,3]
    assert t.elapsed.tolist()==[1.,0.]
    assert not t.contact_time.any()


def test_bilateral_contact_tracks_grip_without_force_target_gate():
    t=free_task();t.f_requested.fill_(12.)
    consume(t,.8)
    assert t.grip_valid.all() and not t.grip_force_ready(t.m['hand_object_loads']).any()
    t.m['object_relative'][:,0]+=.001;consume(t,.8)
    torch.testing.assert_close(t.slip,torch.full_like(t.slip,.001))


def test_success_uses_actual_lift_and_bilateral_contact_not_old_admission():
    t=free_task();t.f_requested.fill_(12.)
    t.m['contact_all_below_centerline'].zero_();t.m['bottom'].fill_(.331)
    consume(t,.8,dt=.5)
    assert t.success.all()
    t=free_task();t.m['bottom'].fill_(.331);t.m['contact_valid'].zero_()
    consume(t,.8,dt=.5)
    assert not t.success.any()


def test_mechanical_failure_is_not_removed():
    t=free_task();t.m['joint_failure'][0]=1
    consume(t)
    assert t.failure[0]!=0
