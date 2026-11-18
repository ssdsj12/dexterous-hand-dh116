import numpy as np
import pytest

from go2_pvcnn.control.dh116.grip_load_reference import GripLoadReference


def test_preload_hold_captures_each_hand_and_releases_on_measured_load_loss():
    from go2_pvcnn.control.dh116.grip_load_reference import PreloadTargetHold
    hold=PreloadTargetHold(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]])
    gaps=np.full((2,3),-.0003);known=np.ones((2,3),dtype=bool)
    loads=np.array([[1.,1.,0.],[.8,1.,0.]])
    np.testing.assert_array_equal(hold.update(loads,gaps,known),[True]*6+[False]*6)
    loads[0,0]=.92  # Hysteresis, still above the unchanged .9 N hold gate.
    assert hold.update(loads,gaps,known)[:6].all()
    loads[0,0]=.89;loads[1,0]=1.
    np.testing.assert_array_equal(hold.update(loads,gaps,known),[False]*6+[True]*6)
    gaps[1,0]=.0001
    assert not hold.update(loads,gaps,known).any()


def test_preload_hold_does_not_import_contact_or_require_parked_middle_load():
    from go2_pvcnn.control.dh116.grip_load_reference import PreloadTargetHold
    hold=PreloadTargetHold(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]])
    loads=np.array([[1.,1.,0.],[1.,1.,0.]])
    gaps=np.array([[-.0003,-.0003,.05],[-.0003,-.0003,.05]])
    assert not hold.update(loads,gaps,np.zeros((2,3),dtype=bool)).any()
    assert hold.update(loads,gaps,np.ones((2,3),dtype=bool)).all()
    # This is only a command latch; no grasp qualification or stage is exposed.
    assert not hasattr(hold,'grasp_lift_passed')


def test_preload_hold_uses_the_existing_grasp_load_requirement_not_an_extra_force_gate():
    from go2_pvcnn.control.dh116.grip_load_reference import PreloadTargetHold
    hold=PreloadTargetHold(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]])
    known=np.ones((2,3),dtype=bool);gaps=np.full((2,3),-.0003)
    # These actual loads already meet GraspEvidenceGate's .9 * targets.
    assert hold.update(np.full((2,3),.93),gaps,known).all()
    loads=np.full((2,3),.93);loads[0,0]=.899
    np.testing.assert_array_equal(hold.update(loads,gaps,known),[False]*6+[True]*6)


def test_preload_hold_releases_on_a_real_interval_dip_even_if_decision_load_recovers():
    from go2_pvcnn.control.dh116.grip_load_reference import PreloadTargetHold
    hold=PreloadTargetHold(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]])
    gaps=np.full((2,3),-.0003);known=np.ones((2,3),dtype=bool)
    recovered=np.full((2,3),.9012758196887054)
    captured=hold.update(recovered,gaps,known)
    low=recovered.copy();low[0,0]=.8998669252160455  # Actual V21 interval2300.
    for loads in (recovered,low,recovered):hold.observe(loads,gaps,known)
    assert captured.all() and hold.held.all()  # No mid-interval motor decision.
    mask=hold.update(recovered,gaps,known)
    np.testing.assert_array_equal(mask,[False]*6+[True]*6)
    assert hold.last_window['sample_count']==3
    assert hold.last_window['minimum_loads_n'][0][0]==low[0,0]


def test_preload_feedback_consumes_the_window_without_retaining_old_load_loss():
    from go2_pvcnn.control.dh116.grip_load_reference import PreloadTargetHold
    hold=PreloadTargetHold(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]])
    gaps=np.full((2,3),-.0003);known=np.ones((2,3),dtype=bool)
    low=np.full((2,3),.93);low[0,0]=.89
    hold.observe(low,gaps,known)
    assert not hold.update(np.full((2,3),.93),gaps,known)[:6].any()
    used=hold.last_window
    assert hold.update(np.full((2,3),.93),gaps,known).all()
    assert hold.last_window['sample_count']==0
    assert used['minimum_loads_n'][0][0]==.89


@pytest.mark.parametrize('loss',('open_gap','unknown_contact'))
def test_preload_feedback_retains_contact_loss_between_decisions(loss):
    from go2_pvcnn.control.dh116.grip_load_reference import PreloadTargetHold
    hold=PreloadTargetHold(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]])
    loads=np.full((2,3),.93);gaps=np.full((2,3),-.0003);known=np.ones((2,3),dtype=bool)
    assert hold.update(loads,gaps,known).all()
    lost_gaps=gaps.copy();lost_known=known.copy()
    if loss=='open_gap':lost_gaps[0,1]=.0001
    else:lost_known[0,1]=False
    hold.observe(loads,lost_gaps,lost_known)
    np.testing.assert_array_equal(hold.update(loads,gaps,known),[False]*6+[True]*6)


def test_preload_window_uses_the_same_exact_threshold_and_ignores_parked_middle():
    from go2_pvcnn.control.dh116.grip_load_reference import PreloadTargetHold
    hold=PreloadTargetHold(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]])
    loads=np.full((2,3),.9);gaps=np.full((2,3),-.0003);known=np.ones((2,3),dtype=bool)
    loads[:,2]=0.;gaps[:,2]=.05;known[:,2]=False
    hold.observe(loads,gaps,known)
    assert hold.update(loads,gaps,known).all()
    low=loads.copy();low[1,1]=.89999
    hold.observe(low,gaps,known)
    np.testing.assert_array_equal(hold.update(loads,gaps,known),[True]*6+[False]*6)


def test_preload_window_copies_physical_inputs_and_does_not_read_future_mutations():
    from go2_pvcnn.control.dh116.grip_load_reference import PreloadTargetHold
    hold=PreloadTargetHold(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]])
    loads=np.full((2,3),.93);gaps=np.full((2,3),-.0003);known=np.ones((2,3),dtype=bool)
    hold.observe(loads,gaps,known)
    loads[0,0]=0.;gaps[0,0]=1.;known[0,0]=False
    assert hold.update(np.full((2,3),.93),np.full((2,3),-.0003),np.ones((2,3),dtype=bool)).all()
    assert hold.last_window['minimum_loads_n'][0][0]==.93
    assert hold.last_window['maximum_gaps_m'][0][0]==-.0003
    assert hold.last_window['known'][0][0] is True


def test_load_reference_builds_preload_only_after_actual_contact():
    control=GripLoadReference(np.ones((2,3)))
    a=control.update(np.zeros((2,3)),np.full((2,3),.002),.01)
    b=control.update(np.zeros((2,3)),np.full((2,3),.001),.01)
    np.testing.assert_array_equal(a,b)
    c=control.update(np.full((2,3),.01),np.full((2,3),-.00001),.01)
    assert np.all(c>a)


def test_load_reference_backs_off_excess_force_and_holds_disabled_fingers():
    control=GripLoadReference(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]])
    before=control.update(np.full((2,3),.01),np.full((2,3),-.00001),.01)
    after=control.update(np.full((2,3),2.),np.full((2,3),-.0003),.01)
    assert np.all(after[:,:2]<before[:,:2])
    np.testing.assert_array_equal(after[:,2],0.)
    for _ in range(100):last=control.update(np.zeros((2,3)),np.full((2,3),-.0001),.01)
    assert np.max(last)<=.0006


def test_loaded_initialization_does_not_command_less_closure_when_underloaded():
    control=GripLoadReference(np.ones((2,3)))
    depth=control.update(np.full((2,3),.5),np.full((2,3),-.0003),.01)
    assert np.all(depth>.0003)


def test_reached_force_backcalculates_wound_up_depth_from_actual_contact():
    control=GripLoadReference(np.ones((2,3)),initial_depth_m=np.full((2,3),.0006))
    actual=np.full((2,3),-.00035)
    depth=control.update(np.full((2,3),1.01),actual,.01)
    assert np.all(depth<=-actual)
    assert np.all(depth>.0003)
    exact=control.update(np.ones((2,3)),actual,.01)
    assert np.all(exact<=-actual)


def test_held_hand_reference_tracks_measured_depth_without_integrating_unused_force_error():
    control=GripLoadReference(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]],
        initial_depth_m=np.array([[.0006,.0006,0.],[.0006,.0006,0.]]))
    depth=control.update(np.full((2,3),.96),np.full((2,3),-.0003),.01,held_hands=[True,False])
    np.testing.assert_allclose(depth[0,:2],.0003)
    assert (depth[1,:2]>.0003).all()
    assert (depth[:,2]==0.).all()
