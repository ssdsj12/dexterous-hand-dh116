"""Cross-check adapter with the actual vendored rod dynamic/contact kernels."""
import importlib.util

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('co_sim.implicit_contact_adapter') is not None, 'adapter missing'
    from co_sim import implicit_contact_adapter
    return implicit_contact_adapter


@pytest.mark.parametrize('stretch', [1., 1.7])
def test_mobility_matches_actual_rod_acceleration_kernel(stretch):
    adapter = module()
    from elastica.rod.cosserat_rod import _update_accelerations
    mass = np.array([.2,.4,.3])
    inv_i = np.stack([np.diag([2.,3.,4.]), np.array([[3.,.2,0.],[.2,4.,0.],[0.,0.,5.]])], axis=2)
    dilatation = np.array([stretch, .9*stretch])
    mobility = adapter.rod_mobility(mass, inv_i, dilatation)
    # Every column, not only one force direction, is compared with production.
    for col in range(15):
        force = np.zeros((3,3)); torque = np.zeros((3,2))
        if col < 9:
            force[col%3,col//3] = 1.
        else:
            torque[(col-9)%3,(col-9)//3] = 1.
        accel=np.zeros((3,3)); alpha=np.zeros((3,2))
        _update_accelerations(accel, np.zeros_like(force), force, mass, alpha,
                              inv_i, np.zeros_like(torque), torque, dilatation)
        expected=np.concatenate([accel.T.ravel(), alpha.T.ravel()])
        np.testing.assert_allclose(mobility[:,col], expected, atol=1e-14)


@pytest.mark.parametrize('subsamples', [1, 4, 16])
def test_contact_jacobian_matches_actual_force_distribution(subsamples):
    adapter=module()
    from elastica._contact_functions import _calculate_contact_forces_rod_mesh
    n=3
    positions=np.array([[0.,.1,.2,.3],[0.,0.,0.,0.],[.1,.1,.1,.1]])
    rotation=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    directors=np.repeat(rotation[:,:,None], n, axis=2)
    points=np.array([[.1*(i+(s+.5)/subsamples),.003,.099]
                     for i in range(n) for s in range(subsamples)])
    b=adapter.contact_jacobians(positions,directors,points,subsamples=subsamples)
    rng=np.random.default_rng(17)
    velocity=rng.normal(size=(3,n+1))*.001
    omega=rng.normal(size=(3,n))*.01
    u=np.concatenate([velocity.T.ravel(),omega.T.ravel()])
    normals=np.tile([0.,0.,1.], (n*subsamples,1))
    relative=np.einsum('nij,j->ni',b,u)
    fn=np.maximum(0., 1e4*.0002-10.*relative[:,2])
    tangent=relative.copy(); tangent[:,2]=0.
    speed=np.linalg.norm(tangent,axis=1)
    f=(-np.minimum(1000.*speed,.5*fn)[:,None]*tangent/(speed[:,None]+1e-14)+fn[:,None]*normals)/subsamples
    expected=np.einsum('nij,ni->j',b,f)
    force=np.zeros((3,n+1)); torque=np.zeros((3,n))
    mesh_force=np.zeros((3,1)); mesh_torque=np.zeros((3,1))
    _calculate_contact_forces_rod_mesh(points, np.full(n*subsamples,.0078), normals,
        positions,velocity,np.full(n,.008),force,torque,omega,directors,
        np.zeros(3),np.zeros(3),np.zeros(3),mesh_force,mesh_torque,np.eye(3),False,
        1e4,10.,1000.,.5,subsamples)
    actual=np.concatenate([force.T.ravel(),torque.T.ravel()])
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    np.testing.assert_allclose(mesh_force[:,0],-f.sum(axis=0),atol=1e-12)
    np.testing.assert_allclose(mesh_torque[:,0],np.cross(points,-f).sum(axis=0),atol=1e-12)
    assert u @ expected == pytest.approx(np.sum(relative*f),abs=1e-12)


def test_point_velocity_includes_material_rotation_and_lever_arm():
    adapter=module()
    positions=np.array([[0.,1.],[0.,0.],[0.,0.]])
    d=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])[:,:,None]
    points=np.array([[.2,.1,0.],[.7,.1,0.]])
    b=adapter.contact_jacobians(positions,d,points,subsamples=2)
    v=np.array([[.1,.2],[.3,.4],[.5,.6]])
    omega=np.array([.2,.3,.4])
    u=np.concatenate([v.T.ravel(),omega])
    for i,t in enumerate([.25,.75]):
        arm=points[i]-((1-t)*positions[:,0]+t*positions[:,1])
        expected=(1-t)*v[:,0]+t*v[:,1]+np.cross(d[:,:,0].T@omega,arm)
        np.testing.assert_allclose(b[i]@u,expected,atol=1e-14)


def test_invalid_mass_and_stretch_rejected():
    a=module()
    with pytest.raises(ValueError):
        a.rod_mobility(np.array([0.,1.]),np.eye(3)[:,:,None],np.ones(1))
    with pytest.raises(ValueError):
        a.rod_mobility(np.ones(2),np.eye(3)[:,:,None],np.array([-1.]))


def test_invalid_geometry_rejected():
    a=module()
    with pytest.raises(ValueError):
        a.contact_jacobians(np.zeros((3,2)),np.eye(3)[:,:,None],np.zeros((2,3)),subsamples=1)
    with pytest.raises(ValueError):
        a.contact_jacobians(np.zeros((3,2)),np.zeros((3,3,1)),np.zeros((1,3)),subsamples=1)
