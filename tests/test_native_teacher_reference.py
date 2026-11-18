"""Counterexamples from the native teacher review, without launching Isaac."""
import importlib
import torch


def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_native_teacher_reference')


def test_contact_loss_still_seeks_after_a_grasp_was_acquired():
    p=torch.full((1,2,5),.001,dtype=torch.float64)
    result=module().closure_step(p,torch.zeros_like(p),p.bool(),p.bool(),
        torch.tensor([True]),torch.tensor([False]),torch.zeros_like(p),1/60)
    assert (result>p).all()


def test_depth_retreat_wins_over_closure_and_positive_force_correction():
    p=torch.full((1,2,5),.001,dtype=torch.float64)
    result=module().closure_step(p,torch.full_like(p,5e-6),p.bool(),p.bool(),
        torch.tensor([True]),torch.tensor([True]),torch.full_like(p,.0007),1/60)
    torch.testing.assert_close(result,p-.001/60)


def test_slip_closure_cannot_cancel_unloading_of_the_stronger_grasp_face():
    p=torch.full((1,2,5),.001,dtype=torch.float64)
    redistribution=torch.zeros_like(p);redistribution[:,:,0]=-5e-6;redistribution[:,:,1]=5e-6
    eligible=torch.zeros_like(p,dtype=torch.bool);eligible[:,:,:2]=True
    result=module().closure_step(p,redistribution,torch.zeros_like(eligible),eligible,
        torch.tensor([True]),torch.tensor([True]),torch.zeros_like(p),1/60)
    assert (result[:,:,0]<p[:,:,0]).all() and (result[:,:,1]>p[:,:,1]).all()


def test_final_lift_height_cannot_leave_positive_feedforward():
    rise=torch.tensor([.031,.03099],dtype=torch.float64)
    result,velocity=module().rise_step(rise,torch.full_like(rise,.031),
        torch.tensor([True,True]),torch.tensor([True,True]),.003,1/60)
    torch.testing.assert_close(result,torch.full_like(rise,.031))
    torch.testing.assert_close(velocity,torch.tensor([0.,.0006],dtype=torch.float64))


def test_reference_cannot_pull_away_from_an_object_still_on_its_support():
    rise=torch.tensor([.00075],dtype=torch.float64)
    result,velocity=module().rise_step(rise,torch.zeros_like(rise),
        torch.tensor([True]),torch.tensor([True]),.003,1/60)
    torch.testing.assert_close(result,rise)
    assert velocity.item()==0


def test_unpaired_or_unacquired_grasp_does_not_advance_reference():
    rise=torch.zeros(2,dtype=torch.float64)
    result,velocity=module().rise_step(rise,torch.ones_like(rise),
        torch.tensor([False,True]),torch.tensor([True,False]),.003,1/60)
    assert not result.any() and not velocity.any()


def test_one_mm_slip_cannot_be_hidden_by_losing_and_reacquiring_contact():
    initial=torch.zeros(2,2,3,3,dtype=torch.float64)
    latched=torch.tensor([True,True]);slipped=initial+.001
    anchor,new_latched=module().anchor_step(initial,latched,slipped,
        torch.tensor([True,True]),torch.tensor([True,False]),torch.tensor([False,False]))
    torch.testing.assert_close(anchor,initial)
    assert new_latched.all() and (slipped-anchor).norm(dim=-1).amin()>.0005


def test_only_reset_can_clear_an_existing_grasp_anchor():
    anchor=torch.ones(2,2,3,3,dtype=torch.float64)
    result,latched=module().anchor_step(anchor,torch.tensor([True,True]),anchor*2,
        torch.tensor([True,True]),torch.tensor([True,True]),torch.tensor([True,False]))
    assert latched.tolist()==[False,True]
    torch.testing.assert_close(result[1],anchor[1])


def test_fingers_cannot_absorb_a_requested_physical_wrist_ascent():
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import bounded_dls
    z=torch.zeros((1,2),dtype=torch.float64)
    floor=module().ascent_floor(z,z,torch.tensor([.00075]),torch.tensor([0.]),torch.tensor([True]))
    a=torch.zeros(1,2,1,12,dtype=torch.float64);a[:,:,:,0]=.1;a[:,:,:,8]=1.
    b=torch.full((1,2,1),.003,dtype=torch.float64);bias=torch.zeros(1,2,12,dtype=torch.float64)
    lo=torch.full_like(bias,-.15);hi=-lo;lo[:,:,8]=floor
    result=bounded_dls(a,b,bias,lo,hi,1e-5)
    assert (result[:,:,8]>=.003-1e-9).all()


def test_unacquired_hand_keeps_the_original_bidirectional_wrist_bounds():
    z=torch.zeros((1,2),dtype=torch.float64)
    floor=module().ascent_floor(z,z,torch.tensor([.00075]),torch.tensor([0.]),torch.tensor([False]))
    torch.testing.assert_close(floor,torch.full_like(z,-.008))


def test_two_newtons_in_total_cannot_hide_a_large_per_face_error():
    loads=torch.tensor([[[1.8,.2,0.,0.,0.],[1.,1.,0.,0.,0.]]],dtype=torch.float64)
    goal=torch.tensor([[[1.,1.,0.,0.,0.],[1.,1.,0.,0.,0.]]],dtype=torch.float64)
    ok,hand_error,digit_error=module().force_tracking(loads,goal,goal>0,2.,.2)
    assert hand_error.abs().max()==0 and not ok.item()
    assert digit_error.abs().max()>.2


def test_fixed_two_newton_goal_accepts_only_both_hands_within_point_two():
    loads=torch.tensor([[[.95,1.15,0.,0.,0.],[1.,1.,0.,0.,0.]],
                        [[1.15,1.15,0.,0.,0.],[1.,1.,0.,0.,0.]]],dtype=torch.float64)
    goal=torch.tensor([[[1.,1.,0.,0.,0.],[1.,1.,0.,0.,0.]]],dtype=torch.float64).expand_as(loads)
    ok,_,_=module().force_tracking(loads,goal,goal>0,2.,.2)
    assert ok.tolist()==[True,False]


def test_small_per_digit_errors_cannot_hide_a_large_total_error_on_one_face():
    loads=torch.tensor([[[.89,.65,.65,0.,0.],[1.,.5,.5,0.,0.]]],dtype=torch.float64)
    desired=torch.tensor([[[1.,.5,.5,0.,0.],[1.,.5,.5,0.,0.]]],dtype=torch.float64)
    faces=torch.tensor([[[.89,1.3],[1.,1.]]],dtype=torch.float64)
    ok,_,_=module().force_tracking(loads,desired,desired>0,2.,.2,face_loads=faces)
    assert not ok.item()


def test_hybrid_force_servo_opens_strong_face_without_losing_tangent_ascent():
    normal=torch.tensor([[[[1.,0.,0.],[-1.,0.,0.]]]],dtype=torch.float64)
    load=torch.tensor([[[1.8,.2]]],dtype=torch.float64)
    goal=torch.ones_like(load)
    v=module().hybrid_velocity(torch.zeros_like(normal),normal,load,goal,
        torch.zeros_like(load),torch.tensor([.003],dtype=torch.float64))
    signed=(v*normal).sum(-1)
    assert signed[0,0,0]<0 and signed[0,0,1]>0 and (v[...,2]>.0029).all()


def test_normal_position_error_does_not_override_oblique_force_axis():
    normal=torch.tensor([[[[.6,0.,.8],[-.6,0.,-.8]]]],dtype=torch.float64)
    load=torch.ones(1,1,2,dtype=torch.float64);up=torch.zeros(1,dtype=torch.float64)
    error=torch.ones_like(normal)*.001
    a=module().hybrid_velocity(error,normal,load,load,load*0,up)
    b=module().hybrid_velocity(error+1000*normal,normal,load,load,load*0,up)
    torch.testing.assert_close(a,b,rtol=0,atol=1e-12)


def test_native_depth_retreat_wins_over_fixed_force_error():
    normal=torch.tensor([[[[1.,0.,0.],[-1.,0.,0.]]]],dtype=torch.float64)
    load=torch.zeros(1,1,2,dtype=torch.float64)
    v=module().hybrid_velocity(torch.zeros_like(normal),normal,load,load+1,load+.0007,
        torch.zeros(1,dtype=torch.float64))
    torch.testing.assert_close((v*normal).sum(-1),load-.001)


def test_saturated_position_target_does_not_promise_unavailable_finger_motion():
    current=torch.tensor([.2,0.],dtype=torch.float64)
    lower=torch.tensor([0.,-.03],dtype=torch.float64)
    upper=torch.tensor([.2,.03],dtype=torch.float64)
    limits=torch.tensor([.15,.004],dtype=torch.float64)
    lo,hi=module().reference_velocity_box(current,lower,upper,limits,1/60)
    assert hi[0].item()==0 and hi[1].item()==.004
    assert (current+hi/60<=upper).all() and (current+lo/60>=lower).all()


def test_measured_tracking_envelope_restores_an_already_overdue_target():
    q=torch.tensor([.201],dtype=torch.float64)
    lo,hi=module().reference_velocity_box(q,q*0,q*0+.2,q*0+.15,1/60)
    assert hi.item()<0 and hi.item()>=-.15 and lo.item()<=hi.item()


def test_acquired_teacher_lift_continues_after_the_first_point_seven_five_mm():
    rise=torch.tensor([.00075,.031],dtype=torch.float64)
    next_rise,up=module().scheduled_ascent(rise,torch.tensor([True,True]),.006,1/60)
    torch.testing.assert_close(next_rise,torch.tensor([.00085,.031],dtype=torch.float64))
    torch.testing.assert_close(up,torch.tensor([.006,0.],dtype=torch.float64))


def test_six_newtons_means_three_per_hand_and_point_three_is_the_error_limit():
    load=torch.tensor([[[1.4,1.4,0.,0.,0.],[1.6,1.6,0.,0.,0.]],
                       [[1.1,1.5,0.,0.,0.],[1.5,1.5,0.,0.,0.]]],dtype=torch.float64)
    goal=torch.zeros_like(load);goal[:,:,:2]=1.5
    faces=load[:,:,:2]
    ok,_,_=module().force_tracking(load,goal,goal>0,3.,.3,face_loads=faces)
    assert ok.tolist()==[True,False]


def test_user_force_tolerance_uses_real_control_interval_mean_not_ema_or_peaks():
    f=torch.zeros(8,1,2,5,dtype=torch.float64);f[1::2,:,:,:2]=3.
    normals=torch.zeros_like(f)[...,None].expand(*f.shape,3).clone()
    normals[:,:,:,0,0]=-1.;normals[:,:,:,1,0]=1.
    desired=torch.zeros_like(f[0]);desired[:,:,:2]=1.5
    state=module().interval_force_state(f,f[...,None]*normals,desired,desired>0,3.,.3)
    assert state['good'].tolist()==[True]
    torch.testing.assert_close(state['mean_loads'],desired)
    assert not module().force_tracking(f[0],desired,desired>0,3.,.3)[0].item()


def test_interval_total_cannot_hide_a_missing_opposed_face():
    f=torch.zeros(8,1,2,5,dtype=torch.float64);f[:,:,:,0]=3.
    v=torch.zeros(*f.shape,3,dtype=torch.float64);v[...,0]=-f
    desired=f[0]*0;desired[:,:,:2]=1.5
    state=module().interval_force_state(f,v,desired,desired>0,3.,.3)
    assert state['good'].tolist()==[False]


def test_interval_still_checks_the_sum_on_each_multi_finger_face():
    f=torch.zeros(8,1,2,5,dtype=torch.float64);f[:,:,:,:3]=torch.tensor([1.39,.92,.92])
    v=torch.zeros(*f.shape,3,dtype=torch.float64);v[...,0]=f;v[:,:,:,0,0]*=-1
    desired=f[0]*0;desired[:,:,:3]=torch.tensor([1.5,.75,.75])
    state=module().interval_force_state(f,v,desired,desired>0,3.,.3)
    assert state['good'].tolist()==[False]


def test_reset_interval_mean_cannot_count_as_new_episode_stability():
    age=torch.tensor([0.,.15,.15],dtype=torch.float64)
    ready=torch.tensor([True,True,False]);reset=torch.tensor([True,False,False])
    actual=module().acquisition_dwell(age,ready,reset,1/60)
    torch.testing.assert_close(actual,torch.tensor([0.,.15+1/60,0.],dtype=torch.float64))


def test_in_band_oblique_contacts_transport_up_together_instead_of_unfolding_against_lift():
    normal=torch.tensor([[[[.6,0.,.8],[-.6,0.,.8]]]],dtype=torch.float64)
    load=torch.ones(1,1,2,dtype=torch.float64)
    actual=module().hybrid_velocity(torch.zeros_like(normal),normal,load,load,load*0,
        torch.tensor([.003],dtype=torch.float64))
    expected=torch.zeros_like(normal);expected[...,2]=.003
    torch.testing.assert_close(actual,expected,atol=1e-15,rtol=0)
