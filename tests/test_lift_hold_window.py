"""Lift success is sustained real height/grasp, independent of force setpoint error."""
import importlib
import torch

def module():return importlib.import_module('go2_pvcnn.control.dh116.tabletop_lift_hold_window')

def initial():
    age=torch.zeros(2,dtype=torch.float64)
    anchor=torch.zeros(2,2,3,3,dtype=torch.float64)
    latched=torch.zeros(2,dtype=torch.bool)
    samples=torch.zeros(8,2,2,3,3,dtype=torch.float64)
    return age,anchor,latched,samples

def test_five_seconds_of_actual_stable_lift_does_not_require_force_error_input():
    age,anchor,latched,samples=initial()
    for _ in range(300):
        age,anchor,latched,diag=module().hold_window_step(age,anchor,latched,samples,torch.ones(2,dtype=torch.bool),1/60)
    torch.testing.assert_close(age,torch.full_like(age,5.),atol=1e-12,rtol=0)
    assert diag['good'].all()

def test_settling_before_hold_does_not_permanently_prevent_later_stability():
    age,anchor,latched,samples=initial()
    eligible=torch.ones(2,dtype=torch.bool)
    age,anchor,latched,_=module().hold_window_step(age,anchor,latched,samples,eligible,1/60)
    samples[:,0,1,2,2]=.0006
    age,anchor,latched,diag=module().hold_window_step(age,anchor,latched,samples,eligible,1/60)
    assert age[0]==0 and diag['restart'].tolist()==[True,False]
    for _ in range(300):age,anchor,latched,diag=module().hold_window_step(age,anchor,latched,samples,eligible,1/60)
    assert age[0]>=5.-1e-12

def test_continuous_sliding_cannot_hide_behind_repeated_reanchoring():
    age,anchor,latched,samples=initial()
    maximum=age.clone()
    for i in range(400):
        samples.fill_(i*.0001)
        age,anchor,latched,_=module().hold_window_step(age,anchor,latched,samples,torch.ones(2,dtype=torch.bool),1/60)
        maximum=torch.maximum(maximum,age)
    assert (maximum<.1).all()

def test_a_single_raw_substep_slip_or_loss_of_height_breaks_only_that_environment():
    age,anchor,latched,samples=initial();age[:]=4.;latched[:]=True
    samples[3,0,0,0,0]=.0006
    age,anchor,latched,diag=module().hold_window_step(age,anchor,latched,samples,torch.tensor([True,False]),1/60)
    assert age.tolist()==[0.,0.] and latched.tolist()==[True,False]
    assert diag['restart'].tolist()==[True,False]

def test_hold_anchor_does_not_follow_small_accumulating_slip_each_period():
    age,anchor,latched,samples=initial();latched[:]=True
    samples[...,0]=.0002
    age,updated,_,diag=module().hold_window_step(age,anchor,latched,samples,torch.ones(2,dtype=torch.bool),1/60)
    assert torch.equal(anchor,updated) and not diag['restart'].any()
