import importlib
import itertools
import torch
import pytest

def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_multifinger_control')

@pytest.mark.parametrize('lever',[.0006,.05])
def test_series_pad_and_drive_compliance_corrects_same_fraction_of_force_error(lever):
    jac=torch.zeros(1,2,5,6,dtype=torch.float64)
    for digit in range(5):jac[:,:,digit,digit]=lever
    loads=torch.full((1,2,5),.5,dtype=jac.dtype);desired=loads+.05
    pad=torch.full_like(loads,1/10000)
    limits=torch.full((1,2,6),.12,dtype=jac.dtype);dt=1/60
    rates=module().force_joint_rates(jac,loads,desired,torch.ones_like(loads,dtype=torch.bool),
        -limits,limits,dt,contact_compliance_m_n=pad)
    displacement=torch.einsum('nhdk,nhk->nhd',jac,rates)*dt
    predicted_force_change=displacement/(lever**2/10+pad)
    torch.testing.assert_close(predicted_force_change,.1*(desired-loads),rtol=.001,atol=1e-8)
    assert (rates.abs()<=limits).all()

def test_series_zero_pad_compliance_preserves_legacy_rates():
    jac=torch.zeros(1,2,5,6,dtype=torch.float64)
    for digit in range(5):jac[:,:,digit,digit]=.03
    loads=torch.full((1,2,5),.5,dtype=jac.dtype);limits=torch.full((1,2,6),.12,dtype=jac.dtype)
    args=(jac,loads,loads+.1,torch.ones_like(loads,dtype=torch.bool),-limits,limits,1/60)
    old=module().force_joint_rates(*args)
    new=module().force_joint_rates(*args,contact_compliance_m_n=torch.zeros_like(loads))
    assert torch.equal(old,new)

@pytest.mark.parametrize('bad',[float('nan'),-.001,float('inf')])
def test_series_rejects_invalid_contact_compliance(bad):
    jac=torch.ones(1,2,5,6,dtype=torch.float64)*.01
    loads=torch.full((1,2,5),.5,dtype=jac.dtype);limits=torch.full((1,2,6),.12,dtype=jac.dtype)
    with pytest.raises(ValueError,match='compliance'):
        module().force_joint_rates(jac,loads,loads+.1,torch.ones_like(loads,dtype=torch.bool),
            -limits,limits,1/60,contact_compliance_m_n=torch.full_like(loads,bad))

def test_material_compliance_decodes_native_soft_stiffness_and_hard_contacts():
    properties=torch.tensor([[[1.,.8,-10000.],[1.,.8,0.],[1.,.8,.2]]],dtype=torch.float64)
    actual=module().compliance_from_physx_material(properties)
    torch.testing.assert_close(actual,properties.new_tensor([[.0001,0.,0.]]))

def test_parallel_stiffness_projection_can_be_below_one():
    normals=torch.tensor([[3**.5/2,0.,.5],[-3**.5/2,0.,.5]],dtype=torch.float64)
    force=torch.ones(2,dtype=normals.dtype);direction=normals.sum(0);direction/=direction.norm()
    factor=module().projected_contact_stiffness_count(normals,force,direction)
    torch.testing.assert_close(factor,force.new_tensor(.5))
    # Finite normal displacement gives the same summed projected spring force.
    delta=1e-5;stiffness=10000.
    force_delta=stiffness*delta*(normals@direction)
    projected=(force_delta[:,None]*normals).sum(0)@direction
    torch.testing.assert_close(projected,stiffness*delta*factor)

def test_parallel_stiffness_uses_only_loaded_points():
    normals=torch.tensor([[0.,0.,1.],[0.,0.,1.],[0.,0.,1.]],dtype=torch.float64)
    factor=module().projected_contact_stiffness_count(normals,torch.tensor([1.,0.,2.],dtype=normals.dtype),normals[0])
    torch.testing.assert_close(factor,normals.new_tensor(2.))

def test_parallel_stiffness_fallback_applies_only_without_loaded_points():
    normals=torch.zeros(0,3,dtype=torch.float64)
    factor=module().projected_contact_stiffness_count(normals,torch.zeros(0,dtype=normals.dtype),normals.new_tensor([0.,0.,1.]))
    torch.testing.assert_close(factor,normals.new_tensor(1.))

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


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('digits',[
    digits for count in range(2,6) for digits in itertools.combinations(range(5),count)
])
def test_sharing_targets_conserve_total_for_any_loaded_combination(digits,dtype):
    total=torch.tensor([[2.,3.],[2.5,2.8]],dtype=dtype)
    loads=torch.zeros(2,2,5,dtype=dtype)
    for index,digit in enumerate(digits):loads[:,:,digit]=.1*(index+1)
    gap=torch.full_like(loads,.001)
    eligible=torch.ones_like(loads,dtype=torch.bool)
    desired,sharing,seeking=module().sharing_targets(total,loads,gap,eligible)
    expected=loads>.05
    assert torch.equal(sharing,expected)
    assert torch.equal(seeking,~expected)
    torch.testing.assert_close(desired,total[:,:,None]*expected/len(digits))
    torch.testing.assert_close(desired.sum(-1),total)
    torch.testing.assert_close((desired*expected).sum(-1),total)


def test_sharing_targets_seek_new_digit_without_diluting_load_then_rebalance():
    total=torch.tensor([[2.4,3.]],dtype=torch.float64)
    loads=torch.tensor([[[0.,0.,1.2,1.2,0.],[0.,0.,1.5,0.,1.5]]],dtype=torch.float64)
    gap=torch.full_like(loads,.001)
    eligible=torch.tensor([[[False,False,True,True,True],[False,False,True,True,True]]])
    desired,sharing,seeking=module().sharing_targets(total,loads,gap,eligible)
    torch.testing.assert_close(desired,loads)
    assert torch.equal(sharing,loads>.05)
    assert torch.equal(seeking,eligible&~sharing)
    assert seeking[0,0,4] and desired[0,0,4]==0
    loads[seeking]=.051
    desired,sharing,seeking=module().sharing_targets(total,loads,gap,eligible)
    assert torch.equal(sharing,eligible)
    assert not seeking.any()
    torch.testing.assert_close(desired,total[:,:,None]*eligible/3)
    torch.testing.assert_close(desired.sum(-1),total)


def test_sharing_targets_preserve_loaded_fingers_even_when_not_eligible():
    total=torch.tensor([[2.,3.]],dtype=torch.float64)
    loads=torch.tensor([[[0.,0.,.1,.2,0.],[0.,0.,0.,0.,.2]]],dtype=torch.float64)
    gap=torch.tensor([[[.001,.002,.01,.02,.03],[.01,.02,-.003,.004,.03]]],dtype=torch.float64)
    eligible=torch.ones_like(loads,dtype=torch.bool);eligible[loads>.05]=False
    desired,sharing,seeking=module().sharing_targets(total,loads,gap,eligible)
    expected=torch.tensor([[[False,False,True,True,False],[False,False,True,False,True]]])
    assert torch.equal(sharing,expected)
    assert torch.equal(seeking,eligible)
    torch.testing.assert_close(desired,total[:,:,None]*expected/2)


def test_sharing_targets_bootstrap_nearest_two_by_absolute_gap_without_preferred_digits():
    total=torch.tensor([[2.,3.]],dtype=torch.float64)
    loads=torch.zeros(1,2,5,dtype=torch.float64)
    gap=torch.tensor([[[.001,.002,-.004,.003,.005],[.02,.03,-.004,.003,.001]]],dtype=torch.float64)
    eligible=torch.tensor([[[False,False,True,True,True],[False,False,True,True,True]]])
    desired,sharing,seeking=module().sharing_targets(total,loads,gap,eligible)
    expected=torch.tensor([[[False,False,True,True,False],[False,False,False,True,True]]])
    assert torch.equal(sharing,expected)
    assert torch.equal(seeking,eligible)
    torch.testing.assert_close(desired,total[:,:,None]*expected/2)


def test_sharing_targets_only_load_above_threshold_counts_as_bearing():
    total=torch.tensor([[2.,3.]],dtype=torch.float64)
    loads=torch.tensor([[[0.,0.,.051,.05,.051],[0.,0.,.05,.051,.051]]],dtype=torch.float64)
    eligible=torch.ones_like(loads,dtype=torch.bool)
    desired,sharing,seeking=module().sharing_targets(total,loads,torch.zeros_like(loads),eligible)
    assert torch.equal(sharing,loads>.05)
    assert torch.equal(seeking,loads<=.05)
    torch.testing.assert_close(desired,total[:,:,None]*sharing/2)


@pytest.mark.parametrize('load',[0.,.1])
@pytest.mark.parametrize('digit',range(5))
def test_sharing_targets_single_eligible_never_gets_entire_hand_budget(digit,load):
    total=torch.tensor([[2.,3.]],dtype=torch.float64)
    loads=torch.zeros(1,2,5,dtype=torch.float64);loads[:,:,digit]=load
    eligible=torch.zeros_like(loads,dtype=torch.bool);eligible[:,:,digit]=True
    desired,sharing,seeking=module().sharing_targets(total,loads,torch.zeros_like(loads),eligible)
    assert torch.equal(sharing,eligible)
    assert torch.equal(seeking,eligible&(loads<=.05))
    torch.testing.assert_close(desired,total[:,:,None]*eligible/2)


def test_sharing_targets_no_eligible_or_load_does_not_fabricate_force():
    total=torch.ones(1,2,dtype=torch.float64)*2.5
    loads=torch.zeros(1,2,5,dtype=torch.float64)
    eligible=torch.zeros_like(loads,dtype=torch.bool)
    desired,sharing,seeking=module().sharing_targets(total,loads,torch.zeros_like(loads),eligible)
    assert not desired.any() and not sharing.any() and not seeking.any()


@pytest.mark.parametrize('fault',[
    'total_shape','loads_shape','gap_shape','eligible_shape','total_int',
    'loads_dtype','gap_dtype','eligible_dtype','eligible_device',
    'total_nan','total_inf','loads_nan','loads_inf','gap_nan','gap_inf',
    'negative_total','negative_loads',
])
def test_sharing_targets_reject_invalid_data(fault):
    total=torch.ones(1,2,dtype=torch.float64)*2.5
    loads=torch.zeros(1,2,5,dtype=torch.float64);gap=torch.zeros_like(loads)
    eligible=torch.ones_like(loads,dtype=torch.bool)
    if fault=='total_shape':total=total[0]
    elif fault=='loads_shape':loads=loads[:,:,:4]
    elif fault=='gap_shape':gap=gap[:,:,:4]
    elif fault=='eligible_shape':eligible=eligible[:,:,:4]
    elif fault=='total_int':total=total.long()
    elif fault=='loads_dtype':loads=loads.float()
    elif fault=='gap_dtype':gap=gap.float()
    elif fault=='eligible_dtype':eligible=eligible.long()
    elif fault=='eligible_device':eligible=eligible.to('meta')
    elif fault=='total_nan':total[0,0]=float('nan')
    elif fault=='total_inf':total[0,0]=float('inf')
    elif fault=='loads_nan':loads[0,0,0]=float('nan')
    elif fault=='loads_inf':loads[0,0,0]=float('inf')
    elif fault=='gap_nan':gap[0,0,0]=float('nan')
    elif fault=='gap_inf':gap[0,0,0]=float('inf')
    elif fault=='negative_total':total[0,0]=-1.
    else:loads[0,0,0]=-1.
    with pytest.raises(ValueError):module().sharing_targets(total,loads,gap,eligible)

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


@pytest.mark.parametrize('retreat',[False,True])
def test_force_feedback_explicit_seek_with_zero_desired_and_retreat_priority(retreat):
    j=torch.zeros(1,2,5,6,dtype=torch.float64)
    for digit in range(5):j[:,:,digit,digit+1]=.02
    loads=torch.zeros(1,2,5,dtype=torch.float64)
    loads[:,:,2:4]=1.2;desired=loads.clone()
    active=torch.zeros_like(loads,dtype=torch.bool);active[:,:,2:]=True
    seeking=torch.zeros_like(active);seeking[:,:,4]=True
    bounds=torch.full((1,2,6),.3,dtype=torch.float64)
    rates=module().force_joint_rates(j,loads,desired,active,-bounds,bounds,1/60,
        seeking=seeking,retreat=seeking if retreat else None)
    normal_v=torch.einsum('nhdk,nhk->nhd',j,rates)
    expected=torch.zeros_like(loads);expected[:,:,4]=-.0003 if retreat else .0003
    torch.testing.assert_close(normal_v,expected,atol=1e-10,rtol=1e-6)


def test_force_feedback_explicit_seek_obeys_active_mask_and_hardware_bounds():
    j=torch.zeros(1,2,5,6,dtype=torch.float64)
    for digit in range(5):j[:,:,digit,digit+1]=.002
    loads=torch.zeros(1,2,5,dtype=torch.float64)
    active=torch.zeros_like(loads,dtype=torch.bool);active[:,:,4]=True
    bounds=torch.full((1,2,6),.005,dtype=torch.float64)
    rates=module().force_joint_rates(j,loads,loads,active,-bounds,bounds,1/60,
        seeking=torch.ones_like(active))
    assert (rates>=-bounds).all() and (rates<=bounds).all()
    assert (rates[:,:,5]>0).all()
    torch.testing.assert_close(rates[:,:,:5],torch.zeros_like(rates[:,:,:5]))


def test_force_feedback_omitted_seeking_preserves_legacy_behavior():
    j=torch.zeros(1,2,5,6,dtype=torch.float64);j[:,:,4,5]=.02
    loads=torch.zeros(1,2,5,dtype=torch.float64);desired=torch.zeros_like(loads)
    active=torch.zeros_like(loads,dtype=torch.bool);active[:,:,4]=True
    bounds=torch.full((1,2,6),.3,dtype=torch.float64)
    rates=module().force_joint_rates(j,loads,desired,active,-bounds,bounds,1/60)
    torch.testing.assert_close(rates,torch.zeros_like(rates))
    desired[:,:,4]=1.
    rates=module().force_joint_rates(j,loads,desired,active,-bounds,bounds,1/60)
    torch.testing.assert_close(rates[:,:,5]*.02,torch.full((1,2),.0003,dtype=torch.float64),
        atol=1e-10,rtol=1e-6)


@pytest.mark.parametrize('fault',['shape','dtype','device'])
def test_force_feedback_rejects_invalid_seeking_mask(fault):
    j=torch.zeros(1,2,5,6,dtype=torch.float64)
    loads=torch.zeros(1,2,5,dtype=torch.float64);active=torch.ones_like(loads,dtype=torch.bool)
    seeking=active.clone();bounds=torch.full((1,2,6),.3,dtype=torch.float64)
    if fault=='shape':seeking=seeking[:,:,:4]
    elif fault=='dtype':seeking=seeking.double()
    else:seeking=seeking.to('meta')
    with pytest.raises(ValueError):
        module().force_joint_rates(j,loads,loads,active,-bounds,bounds,1/60,seeking=seeking)

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
