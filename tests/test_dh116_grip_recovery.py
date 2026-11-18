"""Lift grip loss pauses transport while bounded feedback keeps the grasp owned."""
import copy
import pytest
import torch
from test_dh116_lift_grasp_hold import close_to_lift
from go2_pvcnn.control.dh116.tabletop_contact_effort import ContactEffortFeedback

DT=1/480
ACTION=torch.zeros((2,20),dtype=torch.float64)


def tick(t,steps=1):
    for _ in range(steps):t.control(ACTION,~t.done,DT)


def test_healthy_lift_is_five_times_faster_and_wrist_can_follow():
    from go2_pvcnn.control.dh116.tabletop_palm_mount import CONTRACT
    t=close_to_lift();before=t.offset.clone();tick(t)
    torch.testing.assert_close(t.offset,before+.05*DT)
    assert CONTRACT['velocity_limit_m_s']==.05
    assert CONTRACT['linear_speed_guard_m_s']==.06 and CONTRACT['tracking_guard_m']==.003


def test_growing_relative_slip_pauses_lift_and_tightens_without_reopening():
    t=close_to_lift();before=t.command.clone();palm=t.lift_grasp_hold.palm_offset.clone();offset=t.offset.clone()
    t.slip[:]=.003;t.f_last_slip.zero_();tick(t)
    torch.testing.assert_close(t.offset,offset)
    assert t.lift_grasp_hold.recovering.all()
    assert (t.f_target>2.).all() and (t.f_target<=12.).all()
    assert ((t.command-before)*t.direction.sign().flatten(1)>=-1e-12).all()
    torch.testing.assert_close(t.palm_targets,palm)
    # One good sample cannot release the latch; no normal-Z gate is repeated.
    t.m['vertical_hand_force_n'].fill_(-1.)
    tick(t)
    torch.testing.assert_close(t.offset,offset)
    tick(t,10)
    assert not t.lift_grasp_hold.recovering.any() and (t.offset>offset).all()
    assert (t.f_target>2.).all()  # Stronger target remains after recovery.


@pytest.mark.parametrize('side',[0,1])
def test_persistent_contact_loss_pauses_but_brief_dropout_keeps_lifting(side):
    t=close_to_lift();t.m['hand_object_loads'][:,side]=0.
    tick(t,9)
    assert not t.lift_grasp_hold.recovering.any()
    before=t.offset.clone();tick(t)
    assert t.lift_grasp_hold.recovering.all()
    torch.testing.assert_close(t.offset,before)
    tick(t,1000)
    torch.testing.assert_close(t.offset,before)
    assert (t.f_target<=12.).all() and (t.f_target>2.).all()
    assert t.lift_grasp_hold.active.all()


def test_recovery_and_force_ramp_preserve_inactive_and_reset_rows():
    t=close_to_lift();before=copy.deepcopy(t.lift_grasp_hold)
    t.slip[:]=.003;t.f_last_slip.zero_()
    t.control(ACTION,torch.tensor([True,False]),DT)
    assert t.lift_grasp_hold.recovering.tolist()==[True,False]
    for name,value in vars(before).items():torch.testing.assert_close(getattr(t.lift_grasp_hold,name)[1],value[1])
    t.lift_grasp_hold.reset([0])
    assert not t.lift_grasp_hold.recovering.any()
    assert not t.lift_grasp_hold.contact_loss_s.any()


def test_recovery_target_rate_and_configured_ceiling_are_enforced():
    from dataclasses import replace
    t=close_to_lift();t.parameters=replace(t.parameters,max_force_n=5.)
    t.m['hand_object_loads'][:,0]=0.
    tick(t,10)
    before=t.f_target.clone();tick(t)
    torch.testing.assert_close(t.f_target-before,torch.full_like(before,4.*DT))
    tick(t,400)
    torch.testing.assert_close(t.f_target,torch.full_like(before,5.))
    assert t.lift_grasp_hold.recovering.all()


def test_held_force_above_old_target_band_keeps_pose_but_actual_overload_releases():
    t=close_to_lift();t.m['hand_object_loads'].fill_(6.);t.f_filtered.fill_(6.)
    before=t.command.clone();tick(t)
    assert not t.pressure_limited.any()
    torch.testing.assert_close(t.command,before)
    t.m['hand_object_loads'].fill_(16.);offset=t.offset.clone();tick(t)
    assert t.pressure_limited.all()
    torch.testing.assert_close(t.offset,offset)
    assert (((t.command-before)*t.direction.sign().flatten(1))<0).any()


def test_retained_contact_effort_survives_loss_without_windup_or_resurrection():
    c=ContactEffortFeedback(2,'cpu');target=torch.full((2,2),4.,dtype=torch.float64)
    loads=torch.ones_like(target);direction=torch.ones((2,2,6),dtype=torch.float64)
    eligible=torch.ones((2,2),dtype=torch.bool);retained=eligible.clone()
    for _ in range(100):before=c.step(target,loads,direction,eligible,DT,retained=retained)
    for _ in range(100):out=c.step(target,loads*0,direction,eligible,DT,retained=retained)
    torch.testing.assert_close(out,before,rtol=0,atol=0)
    eligible[0,0]=False
    assert not c.step(target,loads*0,direction,eligible,DT,retained=retained)[0,0].any()
    eligible[0,0]=True
    assert not c.step(target,loads*0,direction,eligible,DT,retained=retained)[0,0].any()


def test_retained_effort_mask_cannot_create_contact_history():
    c=ContactEffortFeedback(2,'cpu');target=torch.full((2,2),12.,dtype=torch.float64)
    direction=torch.ones((2,2,6),dtype=torch.float64);eligible=torch.ones((2,2),dtype=torch.bool)
    for _ in range(100):
        out=c.step(target,target*0,direction,eligible,DT,retained=eligible)
        assert not out.any() and not c.contact_acquired.any()


def test_previous_controller_and_missing_contract_are_rejected_on_resume():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_coordinated_grasp import CONTRACT
    current=dict(num_envs=2,grasp_controller=CONTRACT)
    for saved in (dict(num_envs=2),dict(current,grasp_controller=dict(CONTRACT,version='measured_motion_grasp_v2'))):
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(saved,current)
