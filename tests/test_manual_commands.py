import pytest
import torch


def commands():
    from go2_pvcnn.control.dh116.tabletop_manual_control import ManualCommands
    return ManualCommands(torch.zeros(12),-torch.ones(12),torch.zeros(12),torch.ones(12),'cpu')


def test_translation_settings_preserve_finger_pose():
    c=commands();old=c.joints.clone()
    c.set_placement(dict(palm_offsets_m=[[.0005,0.,.0008],[0.,0.,.0009]],force_n=[5.,5.]))
    torch.testing.assert_close(c.joints,old)
    assert c.palms[0,0]==pytest.approx(.0005)
    with pytest.raises(ValueError):c.set_placement(dict(palm_offsets_m=[[0,0,0]]*2,rotation=[0,0,0]))
    with pytest.raises(ValueError):c.set_placement(dict(palm_offsets_m=[[.1,0,0]]*2))


def test_auto_lift_accepts_stable_measured_force_below_requested_target_once():
    c=commands();c.auto_lift_goal=.01
    # The caller now supplies actual bilateral thumb stability + pair balance.
    for _ in range(12):assert not c.lift_when_stable(False)
    assert c.lift_when_stable(True)
    assert c.lifting and c.lift_goal==pytest.approx(.01) and c.auto_lift_goal is None
    assert not c.lift_when_stable(True)


def test_manual_force_and_joint_action_mapping():
    c=commands();c.force[:]=torch.tensor([2.,12.]);c.joints[:]=.25
    a=c.actions(1/60)
    assert a.shape==(1,20) and a.dtype==torch.float32
    assert a[0,18:].tolist()==[-1.,1.]
    torch.testing.assert_close(a[0,:12],torch.full((12,),.25))


def test_lift_height_speed_and_stop_are_directly_controlled():
    c=commands();c.lift_goal=.03;c.lift_speed=.005;c.lifting=True
    for _ in range(60):a=c.actions(1/60)
    assert c.lift_current==pytest.approx(.005)
    assert float(a[0,14])==pytest.approx(.005/.05)
    c.lifting=False;c.actions(.2)
    assert c.lift_current==pytest.approx(.005)
    c.lift_goal=0.;c.lifting=True
    c.actions(1.)
    assert c.lift_current==pytest.approx(0.)


def test_lift_button_resumes_paused_physics_and_submits_upward_motion():
    c=commands();c.running=False;c.lift_goal=.01
    c.start_lift()
    a=c.actions(1/60)
    assert c.running and c.lifting
    assert a[0,14]>0 and a[0,17]>0


def test_commands_do_not_exceed_actuator_envelope():
    c=commands();c.force[:]=100.;c.joints[:]=100.;c.palms[:]=100.
    a=c.actions(.01)
    assert torch.isfinite(a).all() and a.abs().max()<=1.


def test_manual_mode_disables_learning_deadlines_only():
    from test_force_region_integration import constrained
    t=constrained();t.manual_control=True;t.manual_lift_qualified=torch.zeros_like(t.success);t.elapsed[:]=20.
    t.consume(t.m,~t.done,.01,vertical_force_n=torch.ones(2))
    assert not t.failure.any()
    t.m['clearance'][:]=-.002
    t.consume(t.m,~t.done,.01,vertical_force_n=torch.ones(2))
    assert (t.failure!=0).all()


def test_reset_clears_unexecuted_lift_request():
    c=commands();c.lift_current=.03;c.lifting=True;c.running=True
    c.pause_after_reset()
    assert not c.running and not c.lifting and c.lift_current==0.
    assert c.actions(.01)[0,14]==0.


def test_held_manual_xyz_and_lowering_reach_control_targets():
    from test_constant_force_hold import held
    t,h,live=held();t.manual_control=True;t.m['hand_object_loads'][:]=5.
    h.common_height[:]=.01
    t.palm_targets[:]=0.;t.palm_targets[:,:,0]=.002
    _,palms,_,paused=h.targets(t,live,.01)
    assert not paused.any()
    torch.testing.assert_close(palms,t.palm_targets)
