"""DH116 reduced coordinates and USD-derived sampled collision geometry."""
import importlib
import importlib.util
import math
import numpy as np
import pytest
import torch
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, expand, active_limits


def module():
    name='go2_pvcnn.control.dh116.tabletop_surface_kinematics'
    assert importlib.util.find_spec(name) is not None, 'surface kinematics missing'
    return importlib.import_module(name)


def spec():
    edges=[];parent='base_link'
    for name in JOINTS:
        child=name+'_link';t0=np.eye(4);t0[0,3]=.01
        edges.append(dict(name=name,parent=parent,child=child,t0=t0.tolist(),t1=np.eye(4).tolist(),axis=[0.,0.,1.]))
        parent=child
    vertices=np.array([[x,y,z] for x in (-.001,.001) for y in (-.001,.001) for z in (-.001,.001)])
    shapes=[dict(body=parent,vertices=vertices.tolist())]
    patches=[dict(body=parent,points=[[0.,0.,0.],[.001,0.,0.],[0.,.001,0.]],normals=[[1.,0.,0.]]*3)]*2
    return dict(side='left',root='base_link',edges=edges,shapes=shapes,patches=patches)


def test_full_angles_exact_affine_followers_and_gradient():
    m=module();lo,hi=active_limits()
    q=torch.tensor((lo+hi)[None]/2,dtype=torch.float64,requires_grad=True)
    full=m.full_angles(q)
    assert full.shape==(1,11)
    np.testing.assert_allclose(full.detach().numpy()[0],expand(q.detach().numpy()[0]),atol=1e-12)
    full.sum().backward()
    expected=np.ones(6)
    for _,master,ratio,_ in MIMIC:expected[ACTIVE.index(master)]+=ratio
    np.testing.assert_allclose(q.grad.numpy()[0],expected,atol=1e-12)


def test_fk_matches_manual_chain_and_has_active_gradients():
    model=module().SurfaceHandModel(spec())
    lo,hi=active_limits();q=torch.tensor((lo+hi)[None]/2,dtype=torch.float64,requires_grad=True)
    root=torch.tensor([[.5,.2,.4,1.,0.,0.,0.]],dtype=torch.float64)
    result=model.fk(q,root)
    manual=np.eye(4);manual[:3,3]=root[0,:3].numpy()
    for edge,angle in zip(spec()['edges'],expand(q.detach().numpy()[0])):
        rot=np.eye(4);c,s=math.cos(angle),math.sin(angle);rot[:2,:2]=[[c,-s],[s,c]]
        manual=manual@np.array(edge['t0'])@rot
    torch.testing.assert_close(result[JOINTS[-1]+'_link'][0],torch.tensor(manual),atol=1e-12,rtol=0)
    result[JOINTS[-1]+'_link'][0,:3,3].sum().backward()
    assert torch.isfinite(q.grad).all() and (q.grad.abs()>1e-6).any()


def test_sampled_points_follow_body_pose_and_not_root_only():
    model=module().SurfaceHandModel(spec())
    lo,hi=active_limits();q=torch.tensor((lo+hi)[None]/2,dtype=torch.float64)
    roots=torch.tensor([[0.,0.,.4,1.,0.,0.,0.]],dtype=torch.float64)
    result=model.sample(q,roots)
    pose=model.fk(q,roots)[JOINTS[-1]+'_link']
    assert result['patches'].shape==(1,2,3,3)
    torch.testing.assert_close(result['patches'][0,0,0],pose[0,:3,3])
    assert result['collision_points'].shape[-1]==3 and result['collision_points'].shape[1]>=8
    assert torch.allclose(result['patch_normals'].norm(dim=-1),torch.ones(1,2,3,dtype=torch.float64))
    translated=roots.clone();translated[:,:3]+=torch.tensor([.1,-.2,.3])
    other=model.sample(q,translated)
    torch.testing.assert_close(other['collision_points']-result['collision_points'],
        torch.tensor([.1,-.2,.3],dtype=torch.float32).double().expand_as(other['collision_points']))


def test_actual_link_pose_sampling_retains_follower_deviation():
    model=module().SurfaceHandModel(spec());lo,hi=active_limits()
    q=torch.tensor((lo+hi)[None]/2,dtype=torch.float64)
    root=torch.tensor([[0.,0.,.4,1.,0.,0.,0.]],dtype=torch.float64)
    poses=model.fk(q,root)
    sampled=model.sample_poses(poses)
    changed={name:value.clone() for name,value in poses.items()}
    changed[JOINTS[-1]+'_link'][:,2,3]+=.002
    shifted=model.sample_poses(changed)
    torch.testing.assert_close(shifted['patches'][...,2]-sampled['patches'][...,2],
        torch.full((1,2,3),.002,dtype=torch.float64))


def test_object_inside_convex_is_detected_without_surface_point_inside_object():
    m=module();data=spec();data['shapes'][0]['body']='base_link'
    model=m.SurfaceHandModel(data)
    lo,hi=active_limits();q=torch.tensor((lo+hi)[None]/2,dtype=torch.float64)
    root=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]],dtype=torch.float64)
    points=torch.tensor([[[0.,0.,0.],[.01,0.,0.]]],dtype=torch.float64)
    depth=model.object_penetration(q,root,points)
    torch.testing.assert_close(depth,torch.tensor([.001],dtype=torch.float64))


def test_fixed_parent_child_colliders_are_excluded_from_self_penetration():
    data=spec();data['shapes']=[dict(body='base_link',vertices=data['shapes'][0]['vertices']),
        dict(body=JOINTS[0]+'_link',vertices=data['shapes'][0]['vertices'])]
    model=module().SurfaceHandModel(data)
    lo,hi=active_limits();q=torch.tensor((lo+hi)[None]/2,dtype=torch.float64)
    root=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]],dtype=torch.float64)
    assert model.self_pairs==[]
    torch.testing.assert_close(model.self_penetration(q,root),torch.zeros(1,dtype=torch.float64))


def test_nonadjacent_two_moving_joint_bodies_cannot_bypass_self_penetration():
    data=spec()
    for e in data['edges'][:2]:e['t0']=np.eye(4).tolist()
    vertices=np.array(data['shapes'][0]['vertices'])
    data['shapes']=[dict(body='base_link',vertices=(vertices*2).tolist()),
        dict(body=JOINTS[1]+'_link',vertices=vertices.tolist())]
    model=module().SurfaceHandModel(data);lo,hi=active_limits()
    q=torch.tensor((lo+hi)[None]/2,dtype=torch.float64)
    root=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]],dtype=torch.float64)
    assert model.self_pairs==[(1,0)]
    assert model.self_penetration(q,root)[0]>.0005


def test_sibling_finger_bodies_are_not_direct_neighbors():
    data=spec();index_edge=next(e for e in data['edges'] if e['name']=='finger21')
    index_edge['parent']='base_link'
    vertices=data['shapes'][0]['vertices']
    data['shapes']=[dict(body='finger11_link',vertices=vertices),dict(body='finger21_link',vertices=vertices)]
    model=module().SurfaceHandModel(data)
    assert model.self_pairs==[(1,0)]


def test_fixed_body_cluster_preserves_direct_adjacency_exclusion():
    data=spec();data['edges'].append(dict(name='tip_fixed',parent='finger11_link',child='tip',
        t0=np.eye(4).tolist(),t1=np.eye(4).tolist(),axis=None))
    vertices=data['shapes'][0]['vertices']
    data['shapes']=[dict(body='base_link',vertices=vertices),dict(body='tip',vertices=vertices)]
    model=module().SurfaceHandModel(data)
    assert model.self_pairs==[]


def test_bounded_optimization_proxy_keeps_exact_box_planes_for_validation():
    data=spec();data['shapes'][0]['body']='base_link';model=module().SurfaceHandModel(data)
    lo,hi=active_limits();q=torch.tensor((lo+hi)[None]/2,dtype=torch.float64)
    root=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]],dtype=torch.float64)
    points=torch.zeros(1,1,3,dtype=torch.float64)
    assert model.shapes[0]['probe_points'].shape[0]<=96
    torch.testing.assert_close(model.object_penetration(q,root,points,fast=True),
                               model.object_penetration(q,root,points))


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_active_is_rejected(value):
    model=module().SurfaceHandModel(spec());lo,hi=active_limits()
    q=torch.tensor((lo+hi)[None]/2,dtype=torch.float64);q[0,0]=value
    with pytest.raises(ValueError,match='finite'):
        model.fk(q,torch.tensor([[0.,0.,0.,1.,0.,0.,0.]],dtype=torch.float64))


def test_zero_quaternion_and_wrong_active_count_are_rejected():
    model=module().SurfaceHandModel(spec());q=torch.zeros(1,6,dtype=torch.float64)
    with pytest.raises(ValueError,match='quaternion'):model.fk(q,torch.zeros(1,7,dtype=torch.float64))
    with pytest.raises(ValueError):model.fk(torch.zeros(1,11),torch.ones(1,7))


@pytest.mark.parametrize('side,filename',[
    ('left','d134680ab619_DH116_left.usd'),('right','3046ad2ed855_DH116_right.usd')])
def test_actual_usd_patches_are_real_collision_surfaces(side,filename):
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'assets/fixed_dual_panda_dh116_v1/dependencies'/filename
    model=module().SurfaceHandModel.from_usd(source,side)
    assert len(model.shapes)==17
    assert all(p['body'] in {s['body'] for s in model.shapes} for p in model.patches)
    assert all(p['points'].shape==(24,3) for p in model.patches)
    for patch in model.patches:
        shape=next(s for s in model.shapes if s['body']==patch['body'])
        signed=patch['points']@shape['planes'][:,:3].T+shape['planes'][:,3]
        torch.testing.assert_close(signed.amax(-1),torch.zeros(24,dtype=torch.float64),atol=1e-8,rtol=0)
        assert torch.cdist(patch['points'],patch['points']).max()>.008, 'surface samples must cover the inner pad, not one small reference neighborhood'
