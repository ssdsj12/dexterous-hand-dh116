import importlib
import torch
import pytest

def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_multifinger_control')

def test_balanced_targets_allow_any_three_fingers_and_conserve_total():
    eligible=torch.tensor([[[False,False,True,True,True],[True,True,False,True,False]]])
    total=torch.tensor([[3.,6.]],dtype=torch.float64)
    goal=module().balanced_targets(total,eligible)
    torch.testing.assert_close(goal,total.new_tensor([[[0.,0.,1.,1.,1.],[2.,2.,0.,2.,0.]]]))
    torch.testing.assert_close(goal.sum(-1),total)
    assert not eligible[0,0,0]

def test_balanced_targets_no_contacts_does_not_fabricate_force():
    total=torch.ones(2,2,dtype=torch.float64)
    assert module().balanced_targets(total,torch.zeros(2,2,5,dtype=torch.bool)).sum()==0

def test_multifinger_force_feedback_closes_weak_fingers_and_releases_overload():
    j=torch.zeros(1,2,5,6,dtype=torch.float64)
    for digit in range(5):j[:,:,digit,digit+1]=.02
    load=torch.tensor([[[0.,1.,3.,.5,2.],[1.,1.,1.,1.,1.]]],dtype=torch.float64)
    goal=torch.ones_like(load);active=torch.ones_like(load,dtype=torch.bool)
    lo=torch.full((1,2,6),-.3,dtype=torch.float64);hi=-lo
    v=module().force_joint_rates(j,load,goal,active,lo,hi,1/60)
    assert v[0,0,1]>0 and v[0,0,3]<0 and v[0,0,4]>0 and v[0,0,5]<0
    torch.testing.assert_close(v[0,1],torch.zeros(6,dtype=torch.float64),atol=1e-12,rtol=0)
    assert (v.abs()<=.3).all()

def test_force_feedback_uses_lever_compliance_and_actual_joint_limits():
    j=torch.zeros(1,2,5,6,dtype=torch.float64);j[:,:,0,1]=.002
    load=torch.zeros(1,2,5,dtype=torch.float64);goal=torch.ones_like(load)
    active=torch.zeros_like(load,dtype=torch.bool);active[:,:,0]=True
    lo=torch.full((1,2,6),-.01,dtype=torch.float64);hi=torch.zeros_like(lo)
    v=module().force_joint_rates(j,load,goal,active,lo,hi,1/60)
    assert (v<=0).all() and (v>=lo).all()

def test_penetrating_unloaded_digit_must_retreat_instead_of_seeking():
    j=torch.zeros(1,2,5,6,dtype=torch.float64);j[:,:,0,1]=.02
    loads=torch.zeros(1,2,5,dtype=torch.float64);goal=torch.ones_like(loads)
    active=torch.zeros_like(loads,dtype=torch.bool);active[:,:,0]=True
    bounds=torch.full((1,2,6),.3,dtype=torch.float64)
    v=module().force_joint_rates(j,loads,goal,active,-bounds,bounds,1/60,retreat=active)
    assert (v[:,:,1]<0).all()

def test_table_rate_projection_scales_descent_without_changing_direction():
    v=torch.tensor([[[.1,0.,0.,0.,0.,0.]]],dtype=torch.float64).expand(1,2,6).clone()
    j=torch.zeros(1,2,3,6,dtype=torch.float64);j[:,:,0,0]=-.02
    clearance=torch.full((1,2,3),.001,dtype=torch.float64);clearance[:,:,0]=.00031
    result=module().table_safe_rates(v,j,clearance,.01)
    torch.testing.assert_close(result[:,:,0],torch.full((1,2),.05,dtype=torch.float64))
    assert (clearance+torch.einsum('nhsk,nhk->nhs',j,result)*.01>=.0003-1e-12).all()

@pytest.mark.parametrize('fault',['nan','shape','negative_goal','bad_bounds','dt'])
def test_force_feedback_rejects_invalid_data(fault):
    j=torch.zeros(1,2,5,6,dtype=torch.float64);load=torch.zeros(1,2,5,dtype=torch.float64)
    goal=torch.ones_like(load);active=torch.ones_like(load,dtype=torch.bool)
    lo=torch.full((1,2,6),-.3,dtype=torch.float64);hi=-lo;dt=1/60
    if fault=='nan':j[0,0,0,0]=float('nan')
    elif fault=='shape':active=active[:,:,:2]
    elif fault=='negative_goal':goal[0,0,0]=-1.
    elif fault=='bad_bounds':hi=lo-1.
    else:dt=0.
    with pytest.raises(ValueError):module().force_joint_rates(j,load,goal,active,lo,hi,dt)
