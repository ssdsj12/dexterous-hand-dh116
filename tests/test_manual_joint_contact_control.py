"""Joint solution must address actual contact height and load in one step."""
import torch


def test_bound_is_solved_inside_coupled_system_not_clipped_after_solution():
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import bounded_dls
    a=torch.tensor([[[1.,1.]]],dtype=torch.float64)
    b=torch.tensor([[1.]],dtype=torch.float64)
    lo=torch.tensor([[0.,-2.]],dtype=torch.float64)
    hi=torch.tensor([[0.,2.]],dtype=torch.float64)
    v=bounded_dls(a,b,torch.zeros_like(lo),lo,hi,1e-5)
    assert v[0,0]==0 and v[0,1]>.999


def model():
    # Inputs are wrist XYZ and thumb side/flexion plus index flexion.
    j=torch.zeros((1,2,2,3,6),dtype=torch.float64)
    j[:,:,:,:,:3]=torch.eye(3,dtype=torch.float64)
    j[:,:,0,0,4]=-.07;j[:,:,0,2,4]=.05;j[:,:,0,1,3]=.04
    j[:,:,0,0,3]=-.015
    j[:,:,1,0,5]=.09;j[:,:,1,2,5]=.04
    p=torch.tensor([[[[.008,0.,.0005],[-.008,0.,.0005]],
                     [[.008,0.,.0005],[-.008,0.,.0005]]]],dtype=torch.float64)
    return j,p


def test_joint_solution_closes_weak_digit_while_lowering_both_actual_contacts():
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import solve_velocity
    j,p=model();loads=torch.tensor([[[.1,3.],[.1,3.]]],dtype=torch.float64)
    v,diag=solve_velocity(j,p,loads,torch.full((1,2),5.,dtype=torch.float64),torch.zeros((1,2,6),dtype=torch.float64))
    cart=torch.einsum('nhdck,nhk->nhdc',j,v)
    inward=-p.clone();inward[:,:,:,1]=0.;inward/=inward.norm(dim=-1,keepdim=True)
    normal_speed=(cart*inward).sum(-1)
    assert (normal_speed[:,:,0]>0).all()
    assert (normal_speed[:,:,1]<0).all()
    assert (cart[:,:,:,2]<0).all()
    assert torch.isfinite(v).all()
    assert (diag['target_load_n'].amax(-1)/diag['target_load_n'].amin(-1)<=1.4+1e-9).all()


def test_balanced_force_and_contact_height_equilibrium_has_no_motion():
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import solve_velocity
    j,p=model();p[:,:,:,2]=-.002
    v,_=solve_velocity(j,p,torch.full((1,2,2),2.5,dtype=torch.float64),torch.full((1,2),5.,dtype=torch.float64),torch.zeros((1,2,6),dtype=torch.float64))
    torch.testing.assert_close(v,torch.zeros_like(v),atol=1e-12,rtol=0)


def test_loaded_flat_pad_requires_rotation_not_vertical_slide_to_lower_tangent():
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import solve_velocity
    j,p=model()
    angular=torch.zeros_like(j)
    angular[:,:,0,1,4]=1.;angular[:,:,1,1,5]=-1.
    valid=torch.ones((1,2,2),dtype=torch.bool)
    v,diag=solve_velocity(j,p,torch.full((1,2,2),2.5,dtype=torch.float64),torch.full((1,2),5.,dtype=torch.float64),
                         torch.zeros((1,2,6),dtype=torch.float64),angular_jac=angular,contact_valid=valid)
    height_speed=torch.einsum('nhdk,nhk->nhd',diag['contact_height_jacobian'],v)
    assert (height_speed<0).all()
    assert not diag['contact_height_jacobian'][:,:,:,:3].any()
    assert (v[:,:,4:]>0).all()


def test_lift_is_a_common_velocity_and_does_not_become_extra_closing():
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import solve_velocity
    j,p=model();p[:,:,:,2]=-.002
    bias=torch.zeros((1,2,6),dtype=torch.float64);bias[:,:,2]=.001
    v,_=solve_velocity(j,p,torch.full((1,2,2),2.5,dtype=torch.float64),torch.full((1,2),5.,dtype=torch.float64),bias,lift_velocity=bias[:,:,2])
    cart=torch.einsum('nhdck,nhk->nhdc',j,v)
    torch.testing.assert_close(cart[:,:,:,2],torch.full((1,2,2),.001,dtype=torch.float64),atol=2e-5,rtol=0)


def test_shift_com_jacobian_and_fold_mimic_at_actual_contact_point():
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import point_jacobian
    # Revolute about Y: v(p)=omega cross(p-COM).
    j=torch.zeros((6,2),dtype=torch.float64);j[4,0]=1.;j[4,1]=2.
    c=torch.tensor([1.,0.,0.],dtype=torch.float64)
    fold=torch.tensor([[1.],[.5]],dtype=torch.float64)
    out=point_jacobian(j,c,torch.zeros(3,dtype=torch.float64),fold)
    torch.testing.assert_close(out[:,0],torch.tensor([0.,0.,-2.],dtype=torch.float64))


def test_controller_routes_manual_only_and_does_not_move_other_fingers():
    from types import SimpleNamespace
    from test_manual_digit_force_balance import setup
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import JointContactController
    from go2_pvcnn.control.dh116.tabletop_control_probe import joint_actions
    import pytest
    t=setup([[.1,3.],[.1,3.]])
    t.lower=t.low;t.upper=t.high;t.center=t.reference[0]
    w=SimpleNamespace(hands=[SimpleNamespace(joint_names=JOINTS)]*2,
                      palms=t.command.new_zeros((t.n,2,3)),wp=t.command.new_zeros((t.n,3)),
                      wq=t.command.new_tensor([[1.,0.,0.,0.]]).repeat(t.n,1))
    with pytest.raises(ValueError,match='manual only'):JointContactController(t,w)
    t.manual_control=True;c=JointContactController(t,w)
    j,p=model();c.contacts=lambda:(j.repeat(t.n,1,1,1,1),p.repeat(t.n,1,1,1))
    a=t.command.new_zeros((t.n,20));a[:,:12]=joint_actions(t.reference,t.lower,t.center,t.upper)
    a[:,18:]=-.4
    before=t.command.clone();t.control(a,~t.done,1/480)
    torch.testing.assert_close(t.command[:,[3,4,5,9,10,11]],before[:,[3,4,5,9,10,11]])
    assert (t.command-before).abs().max()<=.12/480+1e-12
    assert (t.command-t.reference).abs().max()<=.08
    assert c.nominal_valid.all()
    c.reset(torch.tensor([0]));assert c.nominal_valid.tolist()==[False,True]


def test_overload_release_respects_pose_budget_and_mimic_recovery_direction():
    from types import SimpleNamespace
    from test_manual_digit_force_balance import setup
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import JointContactController
    from go2_pvcnn.control.dh116.tabletop_control_probe import joint_actions
    t=setup([[5.,5.],[5.,5.]])
    t.lower=t.low;t.upper=t.high;t.center=t.reference[0];t.manual_control=True
    w=SimpleNamespace(hands=[SimpleNamespace(joint_names=JOINTS)]*2,
                      palms=t.command.new_zeros((t.n,2,3)),wp=t.command.new_zeros((t.n,3)),
                      wq=t.command.new_tensor([[1.,0.,0.,0.]]).repeat(t.n,1))
    c=JointContactController(t,w);j,p=model();c.contacts=lambda:(j.repeat(t.n,1,1,1,1),p.repeat(t.n,1,1,1))
    a=t.command.new_zeros((t.n,20));a[:,:12]=joint_actions(t.reference,t.lower,t.center,t.upper);a[:,18:]=-.4
    t.command=t.reference.clone()-.08
    # Positive follower error requires a positive master correction; the old
    # overload override always opened the joint, worsening this inconsistency.
    t.m['mimic_errors'][:,0]=.018
    before=t.command.clone();t.control(a,~t.done,1/480)
    assert (t.command-t.reference).abs().max()<=.08+1e-12
    assert (t.command[:,1]>=before[:,1]).all()


def test_total_hand_overload_includes_other_fingers_before_pose_or_lift_motion():
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import solve_velocity
    j,p=model();bias=torch.ones((1,2,6),dtype=torch.float64)*.001
    _,diag=solve_velocity(j,p,torch.full((1,2,2),4.5,dtype=torch.float64),torch.full((1,2),8.,dtype=torch.float64),bias,
                         total_hand_load=torch.full((1,2),13.,dtype=torch.float64),
                         requested_hand_force=torch.full((1,2),12.,dtype=torch.float64))
    assert diag['high_hand_load'].all()
    assert not diag['contact_height_jacobian'].any()


def test_new_pose_reference_waits_for_inconsistent_mimic_to_recover():
    from types import SimpleNamespace
    from test_manual_digit_force_balance import setup
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS
    from go2_pvcnn.control.dh116.tabletop_manual_joint_contact import JointContactController
    from go2_pvcnn.control.dh116.tabletop_control_probe import joint_actions
    t=setup([[1.,1.],[1.,1.]])
    t.lower=t.low;t.upper=t.high;t.center=t.reference[0];t.manual_control=True
    w=SimpleNamespace(hands=[SimpleNamespace(joint_names=JOINTS)]*2,
                      palms=t.command.new_zeros((t.n,2,3)),wp=t.command.new_zeros((t.n,3)),
                      wq=t.command.new_tensor([[1.,0.,0.,0.]]).repeat(t.n,1))
    c=JointContactController(t,w);j,p=model();c.contacts=lambda:(j.repeat(t.n,1,1,1,1),p.repeat(t.n,1,1,1))
    new=t.reference.clone();new[:,1]+=.10
    a=t.command.new_zeros((t.n,20));a[:,:12]=joint_actions(new,t.lower,t.center,t.upper);a[:,18:]=-.4
    t.m['mimic_errors'][:,0]=-.018
    before=t.command.clone();t.control(a,~t.done,1/480)
    assert (t.command[:,1]<=before[:,1]).all()
    assert (c.accepted_reference[:,1]==t.reference[:,1]).all()
