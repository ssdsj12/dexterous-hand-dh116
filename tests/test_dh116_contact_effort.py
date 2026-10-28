"""Contact load feedback regulates effort without depending on joint motion."""
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_contact_effort import ContactEffortFeedback, split_effort_budget

DT=1/480


def inputs(n=2):
    return (torch.full((n,2),10.,dtype=torch.float64),
            torch.ones((n,2),dtype=torch.float64),
            torch.tensor([0.,1.,1.,0.,0.,0.],dtype=torch.float64).expand(n,2,6).clone(),
            torch.ones((n,2),dtype=torch.bool))


def test_contact_force_deficit_increases_effort_without_any_angle_input():
    c=ContactEffortFeedback(2,'cpu');args=inputs()
    first=c.step(*args,DT);last=c.step(*args,DT)
    assert first.dtype==torch.float64 and first.shape==(2,2,6)
    assert (first[:,:,1:3]>0).all() and (last[:,:,1:3]>first[:,:,1:3]).all()
    assert (last-first).abs().max()<=.1*DT+1e-12
    assert (last[:,:,[0,3,4,5]]==0).all()


def test_positive_effort_caps_and_direction_uses_maximum_positive_component():
    c=ContactEffortFeedback(2,'cpu',effort_limit_nm=.2)
    target,loads,direction,eligible=inputs();direction[:]=torch.tensor([-1.,.5,1.,0.,0.,0.])
    for _ in range(1000):out=c.step(target,loads,direction,eligible,DT)
    torch.testing.assert_close(out[:,:,2],torch.full((2,2),.2,dtype=torch.float64))
    torch.testing.assert_close(out[:,:,1],out[:,:,2]/2)
    assert (out>=0).all() and (out<=.2).all() and (out[:,:,0]==0).all()


def test_force_overshoot_unwinds_effort_with_bounded_negative_slew():
    c=ContactEffortFeedback(2,'cpu');target,loads,direction,eligible=inputs()
    for _ in range(100):before=c.step(target,loads,direction,eligible,DT)
    after=c.step(target,torch.full_like(loads,20.),direction,eligible,DT)
    assert (after[:,:,1:3]<before[:,:,1:3]).all()
    assert (before-after).abs().max()<=.1*DT+1e-12
    for _ in range(1000):after=c.step(target,torch.full_like(loads,20.),direction,eligible,DT)
    assert after.count_nonzero()==0


def test_feedback_is_independent_per_hand_and_deadband_holds_effort():
    c=ContactEffortFeedback(2,'cpu');target,loads,direction,eligible=inputs()
    loads[0,0]=target[0,0];loads[1,1]=target[1,1]+.01
    out=c.step(target,loads,direction,eligible,DT)
    assert out[0,0].count_nonzero()==0 and out[1,1].count_nonzero()==0
    assert out[0,1,1]>0 and out[1,0,1]>0
    held=c.step(target,target+.01,direction,eligible,DT)
    torch.testing.assert_close(out,held)


@pytest.mark.parametrize('reason',['ineligible','zero_direction'])
@pytest.mark.parametrize('during_grace',[False,True])
def test_lost_eligibility_or_zero_direction_clears_effort_and_contact_history_immediately(reason,during_grace):
    c=ContactEffortFeedback(2,'cpu');target,loads,direction,eligible=inputs()
    for _ in range(100):before=c.step(target,loads,direction,eligible,DT)
    if during_grace:
        loads[0,1]=0.
        c.step(target,loads,direction,eligible,DT)
    if reason=='ineligible':eligible[0,1]=False
    else:direction[0,1]=0
    out=c.step(target,loads,direction,eligible,DT)
    assert out[0,1].count_nonzero()==0 and out[1,1,1]>=before[1,1,1]
    assert not c.contact_acquired[0,1] and c.contact_loss_s[0,1]==0.
    # Restoring permission with missing contact must not revive any history.
    eligible[0,1]=True;direction[0,1,1:3]=1;loads[0,1]=0.
    out=c.step(target,loads,direction,eligible,DT)
    assert out[0,1].count_nonzero()==0 and not c.contact_acquired[0,1]
    loads[0,1]=1;eligible[0,1]=True;direction[0,1,1:3]=1
    recovered=c.step(target,loads,direction,eligible,DT)
    assert recovered[0,1].max()<=.1*DT+1e-12


def test_reset_clears_selected_environments_only():
    c=ContactEffortFeedback(2,'cpu');target,loads,direction,eligible=inputs()
    for _ in range(10):before=c.step(target,loads,direction,eligible,DT)
    loads.zero_();c.step(target,loads,direction,eligible,DT)
    saved_loss=c.contact_loss_s[1].clone()
    c.reset([0]);out=c.step(target,target,direction,eligible,DT)
    assert out[0].count_nonzero()==0
    torch.testing.assert_close(out[1],before[1])
    assert not c.contact_loss_s.any()
    assert c.contact_acquired.all()  # Measured contact reacquired at target load.
    c.step(target,loads,direction,eligible,DT);c.reset([0])
    assert not c.contact_acquired[0].any() and not c.contact_loss_s[0].any()
    assert c.contact_acquired[1].all()
    torch.testing.assert_close(c.contact_loss_s[1],saved_loss)


@pytest.mark.parametrize('gap_steps',range(1,10))
@pytest.mark.parametrize('missing_load',[0.,.049])
def test_brief_contact_gap_holds_effort_without_integrating_and_reacquires(gap_steps,missing_load):
    c=ContactEffortFeedback(2,'cpu');target,loads,direction,eligible=inputs()
    for _ in range(100):before=c.step(target,loads,direction,eligible,DT)
    loads[0,1]=missing_load
    for _ in range(gap_steps):
        out=c.step(target,loads,direction,eligible,DT)
        torch.testing.assert_close(out[0,1],before[0,1],rtol=0,atol=0)
    assert c.contact_loss_s[0,1].item()==pytest.approx(gap_steps*DT)
    assert c.contact_acquired[0,1]
    assert c.contact_loss_s[1,1]==0. and out[1,1,1]>before[1,1,1]
    bounded,drive=split_effort_budget(out,.2)
    assert (bounded.abs()+drive<=.2+1e-12).all()
    loads[0,1]=1.
    recovered=c.step(target,loads,direction,eligible,DT)
    torch.testing.assert_close(recovered[0,1],before[0,1]+direction[0,1]*(.1*DT),rtol=0,atol=1e-12)
    assert c.contact_loss_s[0,1]==0. and c.contact_acquired[0,1]


@pytest.mark.parametrize('missing_load',[0.,.049])
def test_sustained_contact_loss_clears_effort_at_twenty_ms(missing_load):
    c=ContactEffortFeedback(2,'cpu');target,loads,direction,eligible=inputs()
    for _ in range(100):before=c.step(target,loads,direction,eligible,DT)
    loads[0,1]=missing_load
    for _ in range(9):out=c.step(target,loads,direction,eligible,DT)
    torch.testing.assert_close(out[0,1],before[0,1],rtol=0,atol=0)
    out=c.step(target,loads,direction,eligible,DT)
    assert out[0,1].count_nonzero()==0 and not c.contact_acquired[0,1]
    for _ in range(20):
        assert c.step(target,loads,direction,eligible,DT)[0,1].count_nonzero()==0
    loads[0,1]=1.
    assert c.step(target,loads,direction,eligible,DT)[0,1].max()<=.1*DT+1e-12


def test_contact_loss_exact_grace_boundary_clears_and_repeated_brief_gaps_restart_timer():
    c=ContactEffortFeedback(2,'cpu');target,loads,direction,eligible=inputs()
    for _ in range(100):before=c.step(target,loads,direction,eligible,DT)
    for _ in range(3):
        c.step(target,torch.zeros_like(loads),direction,eligible,.019)
        out=c.step(target,target,direction,eligible,DT)
        torch.testing.assert_close(out,before,rtol=0,atol=0)
        assert not c.contact_loss_s.any()
    c.step(target,torch.zeros_like(loads),direction,eligible,.01)
    assert c.step(target,torch.zeros_like(loads),direction,eligible,.01).count_nonzero()==0
    assert not c.contact_acquired.any()


def test_missing_contact_does_not_unwind_effort_when_target_is_zero():
    c=ContactEffortFeedback(2,'cpu');target,loads,direction,eligible=inputs()
    for _ in range(100):before=c.step(target,loads,direction,eligible,DT)
    for _ in range(9):
        out=c.step(torch.zeros_like(target),torch.full_like(loads,.049),direction,eligible,DT)
        torch.testing.assert_close(out,before,rtol=0,atol=0)


@pytest.mark.parametrize('load',[0.,.049])
def test_no_effort_or_grace_before_first_measured_contact(load):
    c=ContactEffortFeedback(2,'cpu');target,loads,direction,eligible=inputs()
    for _ in range(20):
        assert c.step(target,torch.full_like(loads,load),direction,eligible,DT).count_nonzero()==0
    assert not c.contact_acquired.any() and not c.contact_loss_s.any()
    assert c.step(target,torch.full_like(loads,.05),direction,eligible,DT).max()>0.


def test_contact_effort_v3_contract_records_grace_retention_and_permission_invalidation():
    from go2_pvcnn.control.dh116.tabletop_contact_effort import CONTRACT
    assert CONTRACT['version']=='contact_load_integral_effort_v3'
    assert CONTRACT['retained_contact_loss']=='hold_previously_acquired_integral_without_integration_until_permission_revoked_or_reset'
    assert CONTRACT['contact_loss_grace_s']==.02
    assert CONTRACT['inactive']=='ineligible_or_zero_direction_clear_integral_and_contact_history_immediately'


@pytest.mark.parametrize('field,value',[
    (0,torch.full((2,2),float('nan'))),(0,torch.full((2,2),-1.)),
    (1,torch.full((2,2),float('inf'))),(1,torch.full((2,2),-.1)),
    (2,torch.zeros((2,6))),(2,torch.full((2,2,6),float('nan'))),
    (3,torch.ones((2,2))),(3,torch.ones((2,),dtype=torch.bool)),
    (4,0.),(4,-.01),(4,float('nan'))])
def test_invalid_step_inputs_are_rejected_before_state_mutation(field,value):
    c=ContactEffortFeedback(2,'cpu');args=list(inputs())+[DT]
    before=c.step(*args)
    c.step(args[0],torch.zeros_like(args[1]),args[2],args[3],DT)
    args[field]=value
    states={name:getattr(c,name).clone() for name in ('integral_nm','contact_loss_s','contact_acquired')}
    with pytest.raises(ValueError):c.step(*args)
    for name,value_before in states.items():torch.testing.assert_close(getattr(c,name),value_before)
    target,_,direction,eligible=inputs()
    torch.testing.assert_close(c.step(target,target,direction,eligible,DT),before)


@pytest.mark.parametrize('n,limit',[(0,.2),(2,0.),(2,float('inf')),(2,float('nan'))])
def test_invalid_constructor_rejected(n,limit):
    with pytest.raises(ValueError):ContactEffortFeedback(n,'cpu',effort_limit_nm=limit)


def test_split_budget_bounds_sum_of_feedforward_and_any_remaining_drive():
    ff=torch.tensor([[.4,-.1,0.]],dtype=torch.float64)
    bounded,drive=split_effort_budget(ff,.2)
    torch.testing.assert_close(bounded,torch.tensor([[.2,-.1,0.]],dtype=torch.float64))
    torch.testing.assert_close(bounded.abs()+drive,torch.full_like(ff,.2))
    assert ((bounded+drive).abs()<=.2+1e-12).all()
    assert ((bounded-drive).abs()<=.2+1e-12).all()


@pytest.mark.parametrize('ff,limit',[(torch.tensor([float('nan')]),.2),(torch.ones(1),float('inf')),(torch.ones(1),0.)])
def test_split_budget_rejects_nonfinite_or_invalid_limits(ff,limit):
    with pytest.raises(ValueError):split_effort_budget(ff,limit)
