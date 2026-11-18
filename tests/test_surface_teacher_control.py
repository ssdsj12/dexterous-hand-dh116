import importlib
import importlib.util
import pytest
import torch


def module():
    name='go2_pvcnn.control.dh116.tabletop_surface_teacher_control'
    assert importlib.util.find_spec(name) is not None, 'surface teacher controller missing'
    return importlib.import_module(name)


def toy():
    j=torch.zeros(1,2,2,3,12,dtype=torch.float64)
    j[:,:,:,2,2]=1.
    j[:,:,0,0,6]=.05;j[:,:,1,0,7]=.05
    j[:,:,:,1,1]=1.
    normals=torch.zeros(1,2,2,3,dtype=torch.float64)
    normals[:,:,0,0]=1.;normals[:,:,1,0]=-1.
    lower=torch.full((1,2,12),-.3,dtype=torch.float64);upper=-lower
    lower[:,:,:3]=-.01;upper[:,:,:3]=.01
    lower[:,:,3:6]=-.5;upper[:,:,3:6]=.5
    return j,normals,lower,upper


def test_scheduled_ascent_does_not_require_contact_admission():
    j,n,lo,hi=toy()
    v=module().solve_surface_velocity(j,n,torch.zeros(1,2,2,dtype=torch.float64),
        torch.full((1,2),.003,dtype=torch.float64),lo,hi)
    assert (v[:,:,2]>.0029).all()


def test_finger_lift_motion_does_not_consume_the_scheduled_wrist_ascent():
    j,n,lo,hi=toy()
    j[:,:,0,2,6]=.1;j[:,:,1,2,7]=-.1
    v=module().solve_surface_velocity(j,n,torch.zeros(1,2,2,dtype=torch.float64),
        torch.full((1,2),.003,dtype=torch.float64),lo,hi)
    assert (v[:,:,2]>=.003-1e-8).all()


def test_rotation_norm_limit_cannot_reduce_wrist_translation_lower_bound():
    j,n,lo,hi=toy();j.zero_();j[:,:,:,2,2]=1.
    j[:,:,0,0,3]=.001;j[:,:,1,0,4]=-.001
    v=module().solve_surface_velocity(j,n,torch.zeros(1,2,2,dtype=torch.float64),
        torch.full((1,2),.003,dtype=torch.float64),lo,hi)
    assert (v[:,:,2]>=.003-1e-8).all()
    assert (v[:,:,3:6].norm(dim=-1)<=.5+1e-10).all()


def test_force_imbalance_opens_strong_digit_and_closes_weak_digit():
    j,n,lo,hi=toy();loads=torch.tensor([[[4.5,.5],[4.5,.5]]],dtype=torch.float64)
    v=module().solve_surface_velocity(j,n,loads,torch.zeros(1,2,dtype=torch.float64),lo,hi)
    projected=torch.einsum('nhdc,nhdck,nhk->nhd',n,j,v)
    assert (projected[:,:,0]<0).all() and (projected[:,:,1]>0).all()


def test_commands_obey_native_translation_rotation_and_joint_velocity_budgets():
    j,n,lo,hi=toy();hi[:,:,2]=.001
    v=module().solve_surface_velocity(j,n,torch.zeros(1,2,2,dtype=torch.float64),
        torch.full((1,2),.01,dtype=torch.float64),lo,hi)
    assert torch.isfinite(v).all() and (v>=lo).all() and (v<=hi).all()
    assert (v[:,:,:3].norm(dim=-1)<=.01+1e-10).all()


def test_invalid_contact_model_rejected_before_action_generation():
    j,n,lo,hi=toy();j[0,0,0,0,0]=float('nan')
    with pytest.raises(ValueError):module().solve_surface_velocity(j,n,torch.zeros(1,2,2),torch.zeros(1,2),lo,hi)


def test_contact_seed_points_follow_their_actual_phalanx_in_each_candidate():
    poses=torch.zeros(2,2,4,7,dtype=torch.float64);poses[:,:,:,3]=1.
    poses[:,:,:,0]=torch.arange(4,dtype=torch.float64)
    poses[1,:,:,2]=.5
    ids=torch.tensor([[[0,1],[2,3]],[[3,2],[1,0]]])
    local=torch.zeros(2,2,2,3,dtype=torch.float64);local[:,:,:,1]=.01
    result=module().transform_contact_seeds(poses,ids,local)
    torch.testing.assert_close(result[:,:,:,0],ids.double())
    assert (result[1,:,:,2]==.5).all() and (result[:,:,:,1]==.01).all()


def test_pinch_hold_closes_fingers_while_wrist_commands_stay_fixed():
    j,n,lo,hi=toy()
    v=module().solve_surface_velocity(j,n,torch.zeros(1,2,2,dtype=torch.float64),
        torch.full((1,2),.003,dtype=torch.float64),lo,hi,wrist_locked=True)
    assert not v[:,:,:6].any()
    normal_motion=torch.einsum('nhdc,nhdck,nhk->nhd',n,j,v)
    assert (normal_motion>0).all()


def test_pinching_allows_the_natural_finger_arc_instead_of_fixing_tip_height():
    j,n,lo,hi=toy();j[:,:,0,2,6]=.1;j[:,:,1,2,7]=.1
    v=module().solve_surface_velocity(j,n,torch.zeros(1,2,2,dtype=torch.float64),
        torch.zeros(1,2,dtype=torch.float64),lo,hi,wrist_locked=True)
    normal_motion=torch.einsum('nhdc,nhdck,nhk->nhd',n,j,v)
    torch.testing.assert_close(normal_motion,torch.full_like(normal_motion,.0015),atol=2e-6,rtol=0.)


@pytest.mark.parametrize('lever',[.003,.03])
def test_loaded_force_hold_converges_for_different_contact_levers(lever):
    j,n,lo,hi=toy();j.zero_();j[:,:,0,0,6]=lever;j[:,:,1,0,7]=-lever
    q=torch.full((1,2,2),.5*lever/10,dtype=torch.float64);history=[]
    for _ in range(180):
        loads=(10*q/lever).clamp(0.,.2/lever)
        velocity=module().solve_surface_velocity(j,n,loads,torch.zeros(1,2,dtype=torch.float64),lo,hi,
            desired=torch.ones_like(loads),wrist_locked=True)
        q+=velocity[:,:,6:8]/60
        history.append((10*q/lever).clamp(0.,.2/lever))
    stable=torch.stack(history[-60:])
    assert (stable-1.).abs().max()<.05
