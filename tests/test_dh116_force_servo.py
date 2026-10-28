"""Applied lift pressure correction remains available after close saturation."""
import copy
import pytest
import torch

from test_dh116_lift_grasp_hold import close_to_lift
from test_dh116_tensor_contact_window import loads

DT=1/480


def underloaded(offset=.08):
    t=close_to_lift(feedback_offset=offset)
    t.direction.zero_();t.direction[:,:,1:3]=2**-.5
    t.m=loads(t,.2);t.f_filtered.fill_(.2)
    return t


def test_exhausted_close_integral_does_not_disable_actual_lift_closing():
    t=underloaded();before=t.command.clone()
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,DT)
    delta=(t.command-before).reshape(2,2,6)
    assert (delta[:,:,1:3]>0).all()
    assert (delta.abs()<=.03*DT*t.direction.abs()+1e-12).all()
    torch.testing.assert_close(t.f_offset,torch.full_like(t.f_offset,.08))


def test_lift_servo_uses_last_applied_command_and_a_fresh_bounded_budget():
    t=underloaded();anchor=t.command.clone()
    for _ in range(1500):
        t.m['q'].copy_(t.command)
        t.control(torch.zeros((2,20),dtype=torch.float64),~t.done,DT)
    delta=(t.command-anchor).reshape(2,2,6)
    torch.testing.assert_close(delta,.08*t.direction,rtol=0,atol=1e-10)
    assert t.force_tracking_limited.all() and t.f_blocked.all()


def test_lift_actual_overload_releases_when_close_integral_already_negative_limit():
    t=underloaded(-.08);t.m=loads(t,32.);before=t.command.clone()
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,DT)
    delta=(t.command-before).reshape(2,2,6)
    assert (delta[:,:,1:3]<0).all()
    assert (delta.abs()<=.20*DT+1e-12).all()
    assert t.pressure_limited.all()


def test_lift_servo_is_independent_per_hand_and_ignores_new_policy_requests():
    a=underloaded();a.m['hand_object_loads'][:,0]=2.;a.f_filtered[:,0]=2.
    b=copy.deepcopy(a);before=a.command.clone()
    a.control(torch.ones((2,20),dtype=torch.float64),~a.done,DT)
    b.control(-torch.ones((2,20),dtype=torch.float64),~b.done,DT)
    torch.testing.assert_close(a.command,b.command)
    torch.testing.assert_close(a.command[:,:6],before[:,:6])
    assert (a.command[:,7:9]>before[:,7:9]).all()


def test_close_integrator_does_not_wind_up_against_signed_mimic_projection():
    t=underloaded();t.phase.fill_(1);t.f_offset.zero_()
    t.close_grasp_hold.reset(torch.arange(t.n))  # Exercise unaccepted close exploration.
    t.m['mimic_errors'][:,5:7]=-.007
    t.reference.copy_(t.command[0]);before=t.f_offset.clone()
    # Policy itself requests closure far beyond the follower-feasible interval.
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,DT)
    assert t.force_tracking_limited[:,1].all() and t.f_blocked[:,1].all()
    torch.testing.assert_close(t.f_offset[:,1],before[:,1])
    assert (t.joint_requested_target[:,7:9]>t.joint_governed_target[:,7:9]).all()


def test_projection_block_recovers_next_tick_without_sticking_or_blocking_other_hand():
    t=underloaded();t.m['mimic_errors'][:,5:7]=-.007
    t.command[:,7]+= .0011340595752568072
    t.command[:,8]+= .0010862480990657675
    before=t.command.clone()
    t.control(torch.zeros((2,20),dtype=torch.float64),~t.done,DT)
    assert t.f_blocked[:,1].all() and not t.f_blocked[:,0].any()
    assert (t.command[:,1:3]>before[:,1:3]).all()
    t.m['mimic_errors'].zero_();t.m['q'].copy_(t.command)
    before=t.command.clone()
    t.control(torch.zeros((2,20),dtype=torch.float64),~t.done,DT)
    assert not t.f_blocked.any()
    assert (t.command[:,7:9]>before[:,7:9]).all()


def test_zero_error_preserves_mechanical_recovery_outside_feedback_budget():
    t=underloaded();t.m['hand_object_loads'].fill_(2.);t.f_filtered.fill_(2.)
    # A prior safety recovery can move an uncalibrated finger away from the
    # accepted grasp. Feedback must not restore the old unsafe reference.
    t.command[:,3]-=.002;t.m['q'].copy_(t.command);before=t.command.clone()
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,DT)
    torch.testing.assert_close(t.command,before)


@pytest.mark.parametrize('stale_lead',[0.,.002])
def test_overpressure_severe_positive_mimic_conflict_requests_neutral_then_slews(stale_lead):
    t=underloaded();t.m=loads(t,32.)
    t.m['mimic_errors'][:,1]=.013
    t.command[:,2]=t.m['q'][:,2]+stale_lead
    before=t.command.clone()
    lift_before=t.offset.clone()
    t.control(torch.ones((2,20),dtype=torch.float64),~t.done,DT)
    torch.testing.assert_close(t.joint_governed_target[:,2],t.m['q'][:,2])
    expected=before[:,2]+(t.m['q'][:,2]-before[:,2]).clamp(-.20*DT,.20*DT)
    torch.testing.assert_close(t.command[:,2],expected)
    assert t.force_guard_conflict[:,0].all() and t.paused.all()
    assert (t.command[:,1]<before[:,1]).all() # Other joints can release.
    torch.testing.assert_close(t.offset,lift_before)


def test_governed_servo_preserves_inactive_rows_and_mechanical_lead_bounds():
    t=underloaded();before=copy.deepcopy(t)
    t.m['q'][0,7]=t.high[7]-.00001;t.command[0,7]=t.m['q'][0,7]
    t.lift_grasp_hold.command[0,7]=t.command[0,7]
    t.control(torch.ones((2,20),dtype=torch.float64),torch.tensor([True,False]),DT)
    assert (t.command[0]>=t.low).all() and (t.command[0]<=t.high).all()
    assert ((t.command[0]-t.m['q'][0]).abs()<=.008+1e-12).all()
    torch.testing.assert_close(t.command[1],before.command[1])
    torch.testing.assert_close(t.f_offset[1],before.f_offset[1])


@pytest.mark.parametrize('policy_floor',[False,True])
def test_neutral_conflict_survives_thumb_hold_and_policy_floor(policy_floor):
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    t=enabled(1);t.m=loads(t,20.)
    if policy_floor:
        t.envelope={};t.lower=t.low+.1;t.upper=t.high-.1;t.center=(t.lower+t.upper)/2
        t.m['q'][:]=t.center;t.m['q'][:,7]=t.lower[7]-.02
    else:t.m['q'][:,7]=t.low[7]+.002
    t.command.copy_(t.m['q'])
    if not policy_floor:t.command[:,7]+=.002
    t.m['mimic_errors'][:,5]=.013
    t.control(-torch.ones((1,20),dtype=torch.float64),~t.done,1/480)
    assert t.force_guard_conflict[:,1].all()
    torch.testing.assert_close(t.joint_governed_target[:,7],t.m['q'][:,7])


def test_discarded_release_does_not_later_open_an_underloaded_grasp():
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    t=enabled(1);t.m=loads(t,20.);t.command.copy_(t.m['q']);t.reference.copy_(t.command)
    t.direction.zero_();t.direction[:,:,1]=1.
    t.m['mimic_errors'][:,[0,5]]=.009
    a=-torch.ones((1,20),dtype=torch.float64);a[:,:12]=0.
    before=t.f_offset.clone()
    for _ in range(200):t.control(a,~t.done,1/480)
    torch.testing.assert_close(t.f_offset,before)
    t.m['mimic_errors'].zero_();t.m=loads(t,.2);t.f_filtered.fill_(.2)
    command=t.command.clone();t.control(a,~t.done,1/480)
    assert (t.command[:,[1,7]]>=command[:,[1,7]]).all()
