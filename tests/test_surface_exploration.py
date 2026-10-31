import torch
from test_free_wrist_grasp import setup
from go2_pvcnn.control.dh116 import tabletop_surface_exploration as mode
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import control,add_feedback


def surface_task():
    t,w=setup()
    t.m['effective_pad_loads_n']=torch.zeros(2,2,5,dtype=torch.float64)
    t.m['effective_pad_vectors_w']=torch.zeros(2,2,5,3,dtype=torch.float64)
    t.m['effective_pad_points_w']=torch.zeros(2,2,5,3,dtype=torch.float64)
    t.m['effective_pad_valid']=torch.zeros(2,2,5,dtype=torch.bool)
    t.m['rejected_contact_loads_n']=torch.zeros(2,2,dtype=torch.float64)
    w.lift_reference_bottom_m=.36
    mode.enable(t,w,redistribution=False)
    return t,w


def test_new_policy_has_24_motion_actions_and_no_force_action_or_goal():
    t,w=surface_task()
    assert t.action_size==24 and t.actions.shape==(2,24)
    assert t.surface_force_sharing and not t.f_requested.any()
    c={};mode.apply_contract(c,t)
    assert c['surface_force_sharing']['force_target_range_n'] is None
    assert c['policy_command_size']==24 and c['initial_force_target_n'] is None


def test_zero_motion_does_not_request_five_newtons_or_fixed_closing():
    t,w=surface_task();middle=(t.low+t.high)/2
    t.command[:]=middle;t.m['q'][:]=middle
    control(t,torch.zeros(2,24),torch.ones(2,dtype=torch.bool),.01)
    torch.testing.assert_close(t.command,middle.expand(2,-1))
    assert not t.f_requested.any() and not t.pressure_limited.any()


def test_all_finger_joints_and_wrist_up_are_free_before_contact():
    t,w=surface_task();middle=(t.low+t.high)/2
    t.command[:]=middle;t.m['q'][:]=middle
    action=torch.zeros(2,24);action[:,:12]=.1;action[:,14]=1;action[:,17]=1
    allowed=control(t,action,torch.ones(2,dtype=torch.bool),.01)
    assert allowed.all() and (t.command>middle).all()
    assert (t.palm_targets[:,:,2]>0).all()


def test_legacy_twelve_newton_target_guard_does_not_block_new_policy():
    t,w=surface_task();middle=(t.low+t.high)/2
    t.command[:]=middle;t.m['q'][:]=middle;t.m['hand_object_loads'][:]=13.
    action=torch.zeros(2,24);action[:,0]=.1
    assert control(t,action,torch.ones(2,dtype=torch.bool),.01).all()
    assert (t.command[:,0]>middle[0]).all() and not t.pressure_limited.any()


def test_support_height_alone_never_receives_net_lift_reward():
    t,w=surface_task();t.m['bottom'].fill_(.36)
    t.m['effective_pad_loads_n'][:,:,:2]=1.
    t.m['effective_pad_vectors_w'][:,:,0,0]=1.;t.m['effective_pad_vectors_w'][:,:,1,0]=-1.
    t.m['effective_pad_valid'][:,:,:2]=True
    rates=mode.reward_rates(t)
    assert not rates['dex_lift'].any()
    t.m['bottom'].fill_(.3905)
    assert (mode.reward_rates(t)['dex_lift']>0).all()


def test_new_reward_does_not_compare_with_old_force_requests_or_thumb_index_only():
    t,w=surface_task()
    t.m['effective_pad_loads_n'][:,:,2:]=torch.tensor([2.,1.,1.])
    t.m['effective_pad_vectors_w'][:,:,2,0]=2.
    t.m['effective_pad_vectors_w'][:,:,3:,0]=-1.
    t.m['effective_pad_valid'][:,:,2:]=True
    rates=mode.reward_rates(t)
    assert not rates['dex_force'].any()
    assert not rates['surface_between_faces'].any()
    assert not rates['surface_within_faces'].any()
    assert (rates['dex_contact']>0).all()


def test_reset_retains_new_observation_shape_and_only_selected_integrators():
    t,w=surface_task();before=t.observe().shape
    t.reset(torch.tensor([0]),t.m)
    assert t.observe().shape==before and t.action_size==24
    assert t.phase.tolist()==[3,3] and not t.f_requested.any()


def test_raised_initial_height_cannot_be_credited_as_actual_lift_success():
    t,w=surface_task();t.m['bottom'].fill_(.36)
    t.m['contact_valid'].fill_(True);t.m['contact_center_relative_m'].zero_()
    t.m['effective_pad_loads_n'][:,:,:2]=1.
    t.m['effective_pad_vectors_w'][:,:,0,0]=1.
    t.m['effective_pad_vectors_w'][:,:,1,0]=-1.
    t.m['effective_pad_valid'][:,:,:2]=True
    t.consume(t.m,~t.done,1.,vertical_force_n=torch.ones(2),lift_vertical_force_n=torch.ones(2),
        hand_lift_force_n=torch.ones(2,2),object_weight_n=torch.ones(2))
    assert not t.success.any() and not t.success_time.any()


def test_real_supported_lift_requires_five_seconds_not_legacy_half_second():
    t,w=surface_task();t.m['bottom'].fill_(.3905)
    t.m['contact_valid'].fill_(True);t.m['contact_center_relative_m'].zero_()
    t.m['effective_pad_loads_n'][:,:,:2]=1.
    t.m['effective_pad_vectors_w'][:,:,0,0]=1.
    t.m['effective_pad_vectors_w'][:,:,1,0]=-1.
    t.m['effective_pad_valid'][:,:,:2]=True
    t.success_time.fill_(.49)
    t.consume(t.m,~t.done,.02,vertical_force_n=torch.ones(2),lift_vertical_force_n=torch.ones(2),
        hand_lift_force_n=torch.ones(2,2),object_weight_n=torch.ones(2))
    assert not t.success.any()


def test_reset_clears_only_selected_native_contact_cache_not_fk_geometry():
    data=dict(effective_pad_loads_n=torch.ones(2,2,5),
        effective_pad_vectors_w=torch.ones(2,2,5,3),effective_pad_points_w=torch.ones(2,2,5,3),
        effective_pad_valid=torch.ones(2,2,5,dtype=torch.bool),rejected_contact_loads_n=torch.ones(2,2),
        effective_pad_depth_m=torch.ones(2,2,5),pad_reference_points_w=torch.ones(2,2,5,3))
    data['effective_pad_body_ids']=torch.full((2,2,5),10,dtype=torch.long)
    mode.clear_reset_contacts(data,torch.tensor([0]))
    for key,value in data.items():
        assert value[1].all()
        if key=='effective_pad_body_ids':assert (value[0]==-1).all()
        elif key!='pad_reference_points_w':assert not value[0].any()
    assert data['pad_reference_points_w'].all()


@__import__('pytest').mark.parametrize('load',[.2,13.])
def test_upward_force_reward_has_no_hidden_two_to_twelve_band(load):
    t,w=surface_task();t.phase_force_reward=True
    t.m['bottom'].fill_(.36)
    t.m['effective_pad_loads_n'][:,:,:2]=load/2
    t.m['effective_pad_vectors_w'][:,:,0,0]=load/2
    t.m['effective_pad_vectors_w'][:,:,1,0]=-load/2
    t.m['effective_pad_vectors_w'][:,:,:2,2]=-load*.1
    t.m['effective_pad_valid'][:,:,:2]=True
    t.terms['grasp_upward_force']=torch.zeros_like(t.reward)
    t.terms['lift_upward_force']=torch.zeros_like(t.reward)
    t.m['hand_object_loads'].fill_(load);t.m['object_link_loads'].fill_(load)
    t.m['grouped_object_loads'].zero_();t.m['grouped_object_loads'][:,:,5]=load
    t.m['contact_valid'].fill_(True);t.m['contact_center_relative_m'].zero_()
    t.m['contact_all_below_centerline'].fill_(True);t.m['vertical_hand_force_n'].fill_(.4)
    t.consume(t.m,~t.done,.01,vertical_force_n=torch.full((2,),.8),
        lift_vertical_force_n=torch.ones(2),hand_lift_force_n=torch.full((2,2),.5),object_weight_n=torch.ones(2))
    assert (t.terms['grasp_upward_force']>0).all()
    assert (t.terms['lift_upward_force']>0).all()


def test_ordinary_collision_cannot_supply_positive_pad_support_reward():
    t,w=surface_task()
    t.m['effective_pad_loads_n'][:,:,:2]=1.
    t.m['effective_pad_vectors_w'][:,:,0,0]=1.
    t.m['effective_pad_vectors_w'][:,:,1,0]=-1.
    t.m['effective_pad_valid'][:,:,:2]=True
    t.m['rejected_contact_loads_n'].fill_(4.)
    t.m['vertical_hand_force_n'].fill_(3.)
    normal,total=mode.verified_support(t,torch.full((2,2),4.))
    assert not normal.any() and not total.any()
    assert not mode.qualified_pad_contact(t).any()


def test_whole_hand_friction_is_credited_only_with_exclusively_verified_pad_contacts():
    t,w=surface_task()
    t.m['effective_pad_loads_n'][:,:,:2]=1.
    t.m['effective_pad_vectors_w'][:,:,0,0]=1.
    t.m['effective_pad_vectors_w'][:,:,1,0]=-1.
    t.m['effective_pad_valid'][:,:,:2]=True
    t.m['vertical_hand_force_n'].fill_(0.)
    normal,total=mode.verified_support(t,torch.full((2,2),.4,dtype=torch.float64))
    assert not normal.any() and (total==.4).all()
    assert mode.qualified_pad_contact(t).all()
    t.m['effective_pad_depth_m'][0,0,0]=.001
    assert not mode.qualified_pad_contact(t)[0] and mode.qualified_pad_contact(t)[1]


def test_ordinary_bilateral_collision_cannot_create_grip_anchor_or_paid_contact():
    t,w=surface_task();t.phase_force_reward=True
    for name in ('grasp_upward_force','lift_upward_force'):t.terms[name]=torch.zeros_like(t.reward)
    t.m['bottom'].fill_(.36)
    t.m['contact_valid'].fill_(True);t.m['contact_center_relative_m'].zero_()
    t.m['hand_object_loads'].fill_(2.)
    t.m['object_link_loads'].fill_(2./t.m['object_link_loads'].shape[-1])
    t.m['grouped_object_loads']=torch.stack([t.m['object_link_loads'][...,t.link_groups==g].sum(-1) for g in range(6)],-1)
    t.m['vertical_hand_force_n'].fill_(.4)
    t.consume(t.m,~t.done,.01,vertical_force_n=torch.full((2,),.8),
        lift_vertical_force_n=torch.ones(2),hand_lift_force_n=torch.full((2,2),.5),object_weight_n=torch.ones(2))
    assert not t.grip_valid.any()
    assert not t.contact_reward_paid.any()
    assert not t.terms['contact'].any()


def test_force_balance_is_a_reward_objective_not_lift_qualification():
    t,w=surface_task()
    t.m['effective_pad_loads_n'][:,:,:2]=torch.tensor([1.8,.3])
    t.m['effective_pad_vectors_w'][:,:,0,0]=1.8
    t.m['effective_pad_vectors_w'][:,:,1,0]=-.3
    t.m['effective_pad_valid'][:,:,:2]=True
    assert not mode.allocation(t)['balanced'].any()
    assert mode.qualified_pad_contact(t).all()
    assert (mode.reward_rates(t)['surface_between_faces']<0).all()


def test_surface_policy_does_not_silently_modify_zero_motion_to_correct_follower_error():
    t,w=surface_task();middle=(t.low+t.high)/2
    t.command[:]=middle;t.m['q'][:]=middle;t.m['mimic_errors'].fill_(.03)
    control(t,torch.zeros(2,24),torch.ones(2,dtype=torch.bool),.01)
    torch.testing.assert_close(t.command,middle.expand(2,-1),atol=0,rtol=0)


def test_default_surface_mode_has_no_automatic_force_redistribution_controller():
    t,w=surface_task()
    mode.enable(t,w)
    assert t.surface_redistribution is False


def test_surface_episode_can_cover_successful_twenty_one_second_pickup():
    t,w=surface_task();t.m['bottom'].fill_(.36);t.elapsed.fill_(21.3)
    kw=dict(vertical_force_n=torch.zeros(2),lift_vertical_force_n=torch.zeros(2),
        hand_lift_force_n=torch.zeros(2,2),object_weight_n=torch.ones(2))
    t.consume(t.m,~t.done,.01,**kw)
    assert not t.failure.any()
    t.elapsed.fill_(30.01);t.consume(t.m,~t.done,.01,**kw)
    assert (t.failure!=0).all()


def test_ppo_surface_consume_can_finish_real_lift_with_unequal_digit_forces():
    t,w=surface_task();t.m['bottom'].fill_(.391)
    t.m['effective_pad_loads_n'][:,:,:2]=torch.tensor([1.8,.3])
    t.m['effective_pad_vectors_w'][:,:,0,0]=1.8
    t.m['effective_pad_vectors_w'][:,:,1,0]=-.3
    t.m['effective_pad_valid'][:,:,:2]=True
    kw=dict(vertical_force_n=torch.ones(2),lift_vertical_force_n=torch.ones(2),
        hand_lift_force_n=torch.ones(2,2),object_weight_n=torch.ones(2))
    for _ in range(301):t.consume(t.m,~t.done,1/60,**kw)
    assert t.success.all() and not t.failure.any()
