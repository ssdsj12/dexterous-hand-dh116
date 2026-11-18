"""Batched rod: independent mechanics, reference parity, and reset isolation."""
from dataclasses import replace
import importlib.util
import numpy as np
import pytest
import torch
from co_sim.gpu_cosserat import GpuRodConfig, TorchCudaCosseratRod

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

def backend(cfg, count):
    assert importlib.util.find_spec('co_sim.batched_cosserat'), 'batched CUDA backend is missing'
    from co_sim.batched_cosserat import BatchedCosserat
    return BatchedCosserat(cfg, count)

def motion(count, meshes=0):
    p = torch.zeros(count, meshes, 3, device='cuda', dtype=torch.float64)
    r = torch.eye(3, device='cuda', dtype=torch.float64).expand(count, meshes, 3, 3).clone()
    return p, r, p.clone(), p.clone()

def box():
    v = np.array([[x,y,z] for z in (.004,.04) for y in (-.03,.03) for x in (.1,.4)])
    from scipy.spatial import ConvexHull
    return v, ConvexHull(v).simplices

def config(**kwargs):
    return replace(GpuRodConfig(n_elem=5, base_length=.5, base_radius=.008,
                   youngs_modulus=2e4, dt=1e-6, damping=0.), **kwargs)

def test_independent_expected_gravity_and_subset_reset():
    cfg = config()
    rod = backend(cfg, 4)
    initial = rod.position.clone()
    rod.advance(cfg.dt, *motion(4))
    expected_v = torch.tensor(cfg.gravity, device='cuda', dtype=torch.float64)[None,:,None]*cfg.dt
    torch.testing.assert_close(rod.velocity, expected_v.expand_as(rod.velocity), atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(rod.position, initial + .5*cfg.dt*expected_v, atol=1e-13, rtol=0)
    saved = [x.clone() for x in (rod.position,rod.velocity,rod.director,rod.omega,rod.radius)]
    rod.reset(torch.tensor([1,3], device='cuda'))
    for current, before in zip((rod.position,rod.velocity,rod.director,rod.omega,rod.radius), saved):
        assert torch.equal(current[[0,2]], before[[0,2]])
    assert torch.equal(rod.position[[1,3]], initial[[1,3]])
    assert torch.count_nonzero(rod.velocity[[1,3]]) == 0

def test_perturbed_multiple_environments_match_reference_midpoint():
    cfg = config(damping=.1, contact_subsamples=3)
    rod = backend(cfg, 3)
    for env in range(3):
        rod.position[env,2,2] += (env+1)*.0002
        rod.velocity[env,1] = (env+1)*.002
        rod.omega[env,0] = (env+1)*.001
        angle = torch.linspace(-.02,.03,cfg.n_elem,device='cuda',dtype=torch.float64)*(env+1)
        r = torch.eye(3,device='cuda',dtype=torch.float64)[:,:,None].repeat(1,1,cfg.n_elem)
        r[0,0]=angle.cos();r[1,1]=angle.cos();r[0,1]=angle.sin();r[1,0]=-angle.sin()
        rod.director[env] = torch.einsum('ijn,jkn->ikn',r,rod.director[env])
    references=[]
    for env in range(3):
        ref=TorchCudaCosseratRod(cfg)
        for src,name in zip((rod.position,rod.velocity,rod.director,rod.omega),('rod_position','rod_velocity','rod_director','rod_omega')):
            getattr(ref,name).copy_(src[env])
        ref.step(4)
        references.append(ref)
    rod.advance(4*cfg.dt,*motion(3))
    for env, ref in enumerate(references):
        for value,name in zip((rod.position,rod.velocity,rod.director,rod.omega),('rod_position','rod_velocity','rod_director','rod_omega')):
            torch.testing.assert_close(value[env],getattr(ref,name),atol=2e-10,rtol=2e-9)

def test_independent_expected_table_normal_force():
    cfg=config(table_center=(.25,0.,-.02),table_size=(1.,1.,.04),gravity=(0.,0.,0.))
    rod=backend(cfg,2)
    rod.position[:,2]=.006
    rod.advance(cfg.dt,*motion(2))
    # 5 elements, 2mm penetration, 1e4 N/m, quadrature weights sum to 1/element.
    impulse=(rod.velocity*rod.node_mass[None,None,:]).sum(dim=2)
    expected=torch.tensor([0.,0.,100.*cfg.dt],device='cuda',dtype=torch.float64).expand(2,3)
    torch.testing.assert_close(impulse,expected,atol=1e-12,rtol=1e-9)

def test_mesh_contact_matches_reference_and_conserves_full_wrench():
    cfg=config(gravity=(0.,0.,0.),contact_subsamples=4)
    rod=backend(cfg,2)
    v,f=box(); rod.register_meshes([v],[f])
    rod.velocity[:,0]=.2
    rod.omega[:,2]=.1
    ref=TorchCudaCosseratRod(cfg)
    ref.rod_velocity.copy_(rod.velocity[0]);ref.rod_omega.copy_(rod.omega[0])
    # Orient closed triangles consistently for reference winding classification.
    center=v.mean(0);f=f.copy()
    for i,face in enumerate(f):
        a,b,c=v[face]
        if np.dot(np.cross(b-a,c-a),a-center)<0:f[i]=face[[0,2,1]]
    ref.register_contact_mesh('box',v,f)
    ref.step(1,contact_meshes={'box':{'position':np.zeros(3)}})
    result=rod.advance(cfg.dt,*motion(2,1))
    d=ref.last_contact_diagnostics['box']
    torch.testing.assert_close(result['linear_impulse'][0,0],d['body_force']*cfg.dt,atol=1e-11,rtol=1e-8)
    torch.testing.assert_close(result['angular_origin_impulse'][0,0],d['body_angular_origin']*cfg.dt,atol=1e-11,rtol=1e-8)
    assert result['normal_impulse'][0,0,0].abs() < 1e-12
    assert result['linear_impulse'][0,0,0].abs() > 0
    torch.testing.assert_close(result['linear_balance'],torch.zeros_like(result['linear_balance']),atol=1e-12,rtol=0)
    torch.testing.assert_close(result['angular_origin_balance'],torch.zeros_like(result['angular_origin_balance']),atol=1e-12,rtol=0)
    torch.testing.assert_close(rod.velocity[0],ref.rod_velocity,atol=2e-10,rtol=2e-9)
    torch.testing.assert_close(rod.omega[0],ref.rod_omega,atol=2e-10,rtol=2e-9)

def test_nonfinite_state_aborts():
    rod=backend(config(),2)
    rod.position[1,0,2]=float('nan')
    with pytest.raises(FloatingPointError):rod.advance(1e-6,*motion(2))

def test_inside_rotating_convex_contact_and_finite_table_edges_match_reference():
    from scipy.spatial.transform import Rotation
    cfg=config(gravity=(0.,0.,0.),contact_subsamples=3,rod_start=(0.,0.,.005),
               table_center=(.16,0.,-.02),table_size=(.3,.03,.04))
    rod=backend(cfg,2);v,f=box();rod.register_meshes([v],[f])
    poses=motion(2,1);poses[0][:,0]=torch.tensor([.001,.001,0.],device='cuda')
    poses[2][:,0]=torch.tensor([.1,-.02,.03],device='cuda')
    poses[3][:,0]=torch.tensor([.2,.1,.3],device='cuda')
    ref=TorchCudaCosseratRod(cfg)
    center=v.mean(0);f=f.copy()
    for i,face in enumerate(f):
        a,b,c=v[face]
        if np.dot(np.cross(b-a,c-a),a-center)<0:f[i]=face[[0,2,1]]
    ref.register_contact_mesh('box',v,f)
    for step in range(3):
        time=(step+.5)*cfg.dt
        pos=(poses[0][0,0]+time*poses[2][0,0]).cpu().numpy()
        rot=Rotation.from_rotvec(poses[3][0,0].cpu().numpy()*time).as_matrix()
        ref.step(1,contact_meshes={'box':dict(position=pos,rotation=rot,
                     velocity=poses[2][0,0],omega=poses[3][0,0])})
    result=rod.advance(3*cfg.dt,*poses)
    for value,name in zip((rod.position,rod.velocity,rod.director,rod.omega),('rod_position','rod_velocity','rod_director','rod_omega')):
        torch.testing.assert_close(value[0],getattr(ref,name),atol=3e-9,rtol=3e-8)
    assert result['peak_force'].min()>0

def test_many_hulls_do_not_couple_environments_and_partial_reset_clears_impulses():
    cfg=config(gravity=(0.,0.,0.),contact_subsamples=2)
    rod=backend(cfg,4);vertices,faces=box()
    rod.register_meshes([vertices]*34,[faces]*34)
    poses=motion(4,34);poses[0][:]=10.
    for env in range(4):poses[0][env,env]=0.
    result=rod.advance(cfg.dt,*poses)
    peak=result['peak_force'].clone()
    for env in range(4):
        assert peak[env,env]>0
        assert torch.count_nonzero(peak[env])==1
    before={name:value.clone() for name,value in result.items()}
    rod.reset([1])
    for name,value in result.items():
        assert torch.count_nonzero(value[1])==0
        assert torch.equal(value[[0,2,3]],before[name][[0,2,3]])


def test_bvh_complex_convex_hull_matches_independent_reference():
    from scipy.spatial import ConvexHull
    rng=np.random.default_rng(10)
    directions=rng.normal(size=(80,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    vertices=directions*np.array([.15,.03,.025])+np.array([.25,0.,.022])
    faces=ConvexHull(vertices).simplices.copy()
    for i,face in enumerate(faces):
        a,b,c=vertices[face]
        if np.dot(np.cross(b-a,c-a),a-vertices.mean(0))<0:faces[i]=face[[0,2,1]]
    cfg=config(gravity=(0.,0.,0.),contact_subsamples=10)
    rod=backend(cfg,2);rod.register_meshes([vertices],[faces]);rod.velocity[:,0]=.1
    ref=TorchCudaCosseratRod(cfg);ref.register_contact_mesh('hull',vertices,faces);ref.rod_velocity[0]=.1
    ref.step(1,contact_meshes={'hull':{}})
    result=rod.advance(cfg.dt,*motion(2,1))
    torch.testing.assert_close(result['linear_impulse'][0,0],ref.last_contact_diagnostics['hull']['body_force']*cfg.dt,atol=1e-11,rtol=1e-8)
    torch.testing.assert_close(rod.omega[0],ref.rod_omega,atol=1e-10,rtol=1e-8)

@pytest.mark.parametrize('height', [.02,.046,.05])
def test_radius_bounded_query_preserves_deep_inside_and_exterior_contacts(height):
    cfg=config(gravity=(0.,0.,0.),contact_subsamples=5,rod_start=(0.,0.,height))
    rod=backend(cfg,1);vertices,faces=box();rod.register_meshes([vertices],[faces])
    center=vertices.mean(0);faces=faces.copy()
    for i,face in enumerate(faces):
        a,b,c=vertices[face]
        if np.dot(np.cross(b-a,c-a),a-center)<0:faces[i]=face[[0,2,1]]
    ref=TorchCudaCosseratRod(cfg);ref.register_contact_mesh('box',vertices,faces)
    ref.step(1,contact_meshes={'box':{}})
    result=rod.advance(cfg.dt,*motion(1,1))
    expected=ref.last_contact_diagnostics['box']['body_force']*cfg.dt
    torch.testing.assert_close(result['linear_impulse'][0,0],expected,atol=1e-11,rtol=1e-8)
    torch.testing.assert_close(rod.velocity[0],ref.rod_velocity,atol=1e-10,rtol=1e-8)
