import pytest
import torch
from test_force_region_integration import constrained


def setup(loads):
    from go2_pvcnn.control.dh116.tabletop_manual_digit_force_balance import enable
    t=constrained();t.manual_fast_force=True;enable(t)
    t.m['grouped_object_loads'].zero_()
    t.m['grouped_object_loads'][:,:,:2]=torch.tensor(loads,dtype=torch.float64)
    t.m['hand_object_loads']=t.m['grouped_object_loads'].sum(-1)
    return t


def step(t,live=None):
    from go2_pvcnn.control.dh116.tabletop_policy_force_region import adjust
    target=t.reference.clone()
    return target,adjust(t,target,torch.full_like(t.f_target,5.),~t.done if live is None else live,.01)


def test_weak_digit_closes_independently_of_strong_digit():
    t=setup([[3.3,.2],[.1,3.3]])
    before,out=step(t)
    assert t.manual_digit_offsets[0,0,0]<0 and t.manual_digit_offsets[0,0,1]>0
    assert t.manual_digit_offsets[0,1,0]>0 and t.manual_digit_offsets[0,1,1]<0
    assert torch.equal(out[:,[3,4,5,9,10,11]],before[:,[3,4,5,9,10,11]])
    torch.testing.assert_close(t.manual_digit_targets.sum(-1),torch.full_like(t.f_target,5.))
    assert (t.manual_digit_targets.max(-1).values/t.manual_digit_targets.min(-1).values<=1.5).all()


def test_allowed_unequal_split_is_preserved_at_requested_total_force():
    t=setup([[2.2,2.8],[2.8,2.2]])
    before,out=step(t)
    torch.testing.assert_close(out,before)
    assert not t.manual_digit_offsets.any()


def test_no_contact_acquisition_and_selected_reset_keep_other_env_state():
    t=setup([[0.,0.],[0.,0.]])
    step(t,torch.tensor([True,False]))
    assert (t.manual_digit_offsets[0]>0).all()
    assert not t.manual_digit_offsets[1].any()
    saved=t.manual_digit_offsets[0].clone()
    t.reset(torch.tensor([1]),t.m)
    torch.testing.assert_close(t.manual_digit_offsets[0],saved)
    t.reset(torch.tensor([0]),t.m)
    assert not t.manual_digit_offsets.any()


def test_one_severe_follower_does_not_disable_other_finger_force_control():
    t=setup([[0.,0.],[0.,0.]])
    t.m['mimic_errors'][0,0]=.013
    step(t)
    assert t.manual_digit_offsets[0,0,0]<=0
    assert t.manual_digit_offsets[0,0,1]>0


def test_measured_pair_over_target_releases_both_digits_and_honors_budget():
    t=setup([[4.,4.],[4.,4.]])
    t.manual_digit_offsets.fill_(-.0799)
    step(t)
    assert (t.manual_digit_offsets>=-.08).all()
    assert (t.manual_digit_offsets<-.0799).all()


def test_other_finger_load_is_not_falsely_assigned_to_thumb_or_index():
    t=setup([[1.,1.],[1.,1.]])
    t.m['grouped_object_loads'][:,:,2]=2.
    t.m['hand_object_loads']=t.m['grouped_object_loads'].sum(-1)
    step(t)
    torch.testing.assert_close(t.manual_digit_targets.sum(-1),torch.full_like(t.f_target,3.))


def test_actual_fifty_percent_limit_is_distinct_from_commanded_targets():
    from go2_pvcnn.control.dh116.tabletop_manual_digit_force_balance import summary
    t=setup([[2.,3.],[1.,4.]])
    step(t)
    row=summary(t)
    assert row['balanced']==[True,False]
    assert row['measured_ratio']==[1.5,4.]
    assert row['target_load_n'][1]!=[1.,4.]


def test_separating_fingers_does_not_amplify_authored_joint_closing_motion():
    from go2_pvcnn.control.dh116.tabletop_manual_digit_force_balance import enable
    t=setup([[0.,0.],[0.,0.]])
    t.direction[:]=torch.tensor([[.156,.662,.733,0.,0.,0.],[.099,.419,.902,0.,0.,0.]])
    enable(t)
    before,out=step(t)
    maximum=(t.direction.abs()*.0012).flatten(1)
    assert ((out-before).abs()<=maximum+1e-9).all()


def test_digit_balance_stays_in_control_after_legacy_phase3_capture():
    from go2_pvcnn.control.dh116.tabletop_policy_lift_hold import PolicyLiftHold
    t=setup([[0.,0.],[0.,0.]])
    t.policy_lift_hold=PolicyLiftHold(t.n,t.device)
    t.phase[:]=3
    t.policy_lift_hold.capture(t,~t.done,t.m['adaptive_palm_offset'])
    t.policy_lift_hold.force_target.fill_(2.)
    actions=torch.zeros((t.n,20),dtype=torch.float64)
    actions[:,18:]=-.4  # 5N current manual request
    t.control(actions,~t.done,.01)
    assert (t.manual_digit_offsets>0).all()
    torch.testing.assert_close(t.f_requested,torch.full_like(t.f_requested,5.))


def test_strong_digit_backs_off_if_weak_digit_has_exhausted_microclosing_budget():
    t=setup([[1.94,3.015],[2.5,2.5]])
    t.manual_digit_offsets[0,0,0]=.08
    step(t)
    assert t.manual_digit_targets[0,0,1]<=1.94*1.4+1e-8


def test_manual_joint_target_has_no_relay_jump_across_soft_mimic_boundary():
    import copy
    def jump(smooth):
        a=setup([[1.9,3.],[.8,.7]])
        a.manual_smooth_mimic=smooth
        a.manual_digit_offsets.fill_(.06)
        a.m['q']=a.reference.clone()
        b=copy.deepcopy(a)
        a.m['mimic_errors'][0,0]=-.01199
        b.m['mimic_errors'][0,0]=-.01201
        actions=torch.zeros((a.n,20),dtype=torch.float64)
        actions[:,18:]=-.4
        a.control(actions,~a.done,1/480);b.control(actions,~b.done,1/480)
        return float((a.joint_governed_target-b.joint_governed_target).abs().max())
    assert jump(False)>.005  # Reproduce the legacy relay on identical states.
    assert jump(True)<.001


def test_contact_acquisition_latches_before_low_gain_force_regulation():
    t=setup([[0.,0.],[0.,0.]])
    t.manual_digit_force_gain=.005
    step(t)
    first=t.manual_digit_offsets.clone()
    t.m['grouped_object_loads'][:,:,:2]=.2
    t.m['hand_object_loads']=t.m['grouped_object_loads'].sum(-1)
    step(t)
    assert (t.manual_digit_offsets-first).abs().max()<.0002
    second=t.manual_digit_offsets.clone()
    t.m['grouped_object_loads'].zero_();t.m['hand_object_loads'].zero_()
    step(t)
    assert (t.manual_digit_offsets-second).abs().max()<.0002
