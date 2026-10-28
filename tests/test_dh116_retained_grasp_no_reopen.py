import pytest
import torch
from test_dh116_close_grasp_hold import held, measured, consume, DT
from test_dh116_lift_grasp_hold import close_to_lift


@pytest.mark.parametrize('phase',['close','lift'])
@pytest.mark.parametrize('side',[0,1])
def test_ordinary_force_overshoot_does_not_open_accepted_hand(phase,side):
    t=held(side) if phase=='close' else close_to_lift()
    t.m['hand_object_loads'][:,side]=2.3  # Above2N target, below2.52N pressure band.
    t.f_filtered[:,side]=2.3
    before=t.command.clone()
    t.control(-torch.ones((2,20)),~t.done,DT)
    s=slice(side*6,side*6+6)
    torch.testing.assert_close(t.command[:,s],before[:,s],rtol=0,atol=1e-12)
    assert not t.pressure_limited[:,side].any()


def test_contact_loss_keeps_pose_ownership_but_cannot_admit_lift():
    t=held();anchor=t.close_grasp_hold.command.clone()
    for _ in range(48):consume(t,measured(t,.0,2.))
    assert t.close_grasp_hold.active[:,0].all()
    assert not t.close_grasp_hold.ready[:,0].any()
    assert (t.phase==1).all()
    torch.testing.assert_close(t.close_grasp_hold.command[:,:6],anchor[:,:6])
    consume(t,measured(t,2.,2.))
    assert (t.phase==3).all()


@pytest.mark.parametrize('phase',['close','lift'])
@pytest.mark.parametrize('lag,error',[(.005,-.006),(.01,0.)])
def test_mild_projection_limits_new_closure_without_rewriting_held_pose(phase,lag,error):
    t=held() if phase=='close' else close_to_lift()
    before=t.command.clone();t.m['q'][:,2]-=lag;t.m['mimic_errors'][:,1]=error
    t.control(torch.zeros((2,20)),~t.done,DT)
    torch.testing.assert_close(t.command[:,:6],before[:,:6],rtol=0,atol=1e-12)
    assert not t.paused.any()


@pytest.mark.parametrize('cause',['load_dip','stable_slip','soft_mimic'])
def test_accepted_lift_does_not_repeat_close_admission_each_step(cause):
    t=close_to_lift();before=t.offset.clone()
    if cause=='load_dip':t.m['hand_object_loads'][:,0]=.2;t.f_filtered[:,0]=.2
    if cause=='stable_slip':t.slip[:]=.003;t.f_last_slip[:]=.003
    if cause=='soft_mimic':t.m['mimic_errors'][:,1]=-.009
    t.control(torch.zeros((2,20)),~t.done,DT)
    torch.testing.assert_close(t.offset,before+.05*DT)


def test_severe_coupling_still_pauses_and_can_correct_retained_pose():
    t=close_to_lift();before=t.command.clone();offset=t.offset.clone()
    t.m['q'][:,2]-=.005;t.m['mimic_errors'][:,1]=-.013
    t.control(torch.zeros((2,20)),~t.done,DT)
    assert (t.command[:,2]<before[:,2]).all() and t.paused.all()
    torch.testing.assert_close(t.offset,offset)
