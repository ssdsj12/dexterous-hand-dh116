import numpy as np
import pytest
from go2_pvcnn.control.dh116.deformx_wire_runtime import (
    GraspRegions, outward_hull, paper_rod_config, reaction_at_com, link_origin_velocity)


def test_region_includes_interpolated_endpoints_and_is_fixed_material():
    p = np.array([[0., .5, 1.], [0., 0., 0.], [1., 1., 1.]])
    region = GraspRegions(p, [[.4, 0., 1.]], half_width=.1)
    q = p.copy(); q[2] = [0., 1., 0.]
    v = np.zeros_like(q); v[2] = [0., 2., 0.]
    result = region.measure(q, v, .05)
    assert result['local_surface_min_z_m'] == pytest.approx([.55])
    assert result['local_max_speed_m_s'] == pytest.approx([2.])
    assert result['global_surface_min_z_m'] == pytest.approx(-.05)
    # Later horizontal motion cannot reselect a different material segment.
    q[0] += 100
    assert region.measure(q, v, .05)['local_surface_min_z_m'] == pytest.approx([.55])


def test_origin_angular_impulse_converts_to_actual_com():
    # Force (0,4,0) at point (3,0,0), COM (1,0,0), duration .5.
    f, t = reaction_at_com([0,2,0], [0,0,6], [1,0,0], .5)
    np.testing.assert_allclose(f, [0,4,0])
    np.testing.assert_allclose(t, [0,0,8])
    np.testing.assert_allclose(link_origin_velocity([0,1,0],[0,0,2],[3,0,0],[1,0,0]),[0,5,0])


def test_hull_faces_are_outward_and_original_envelope_retained():
    from itertools import product
    p = np.array(list(product([-1.,1.],repeat=3)))
    vertices, faces = outward_hull(p)
    normals = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    assert np.all(np.einsum('ij,ij->i', normals,vertices[faces].mean(axis=1)) > 0)
    assert len(vertices) == 8


def test_paper_configuration_matches_mesh_and_explicit_contact_approximation():
    c = paper_rod_config([.4,0,.305],[.4,0,.15],[1,2,.3])
    assert (c.n_elem,c.base_length,c.base_radius,c.contact_subsamples) == (20,1,.005,10)
    assert c.youngs_modulus/(2*c.shear_modulus_ratio) == pytest.approx(200000/3)
    assert c.friction_coefficient == 1
    assert c.rod_start == (.4,-.5,.305)


def test_real_cuda_contact_routes_balanced_reaction_to_com():
    import torch
    from dataclasses import replace
    from itertools import product
    from vendor.DeformX.co_sim.moving_contact_models import MeshMotion
    from go2_pvcnn.control.dh116.deformx_wire_runtime import WireRuntime
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    config=replace(paper_rod_config([0,0,0],[0,0,-2],[1,1,.1]),n_elem=2,base_length=.02,
        rod_start=(0,-.01,0),gravity=(0,0,0),contact_subsamples=2)
    # Rod surface penetrates cube top by 1 mm; free rod, no fixed attachment.
    vertices=np.array(list(product([-.02,.02],[-.03,.03],[-.02,-.004])))
    runtime=WireRuntime(config,{'cube':vertices},[[0,0,0]],'cuda:0')
    zero=np.zeros(3)
    motion=MeshMotion(zero,np.eye(3),zero,zero,zero,zero)
    com=np.array([.1,0,0])
    state,reactions,measurement=runtime.advance({'cube':motion},{(0,0):com},{'cube':(0,0)},1e-5)
    assert measurement['contact_detected']
    assert measurement['max_contact_balance_impulse'] < 1e-10
    force,torque=reactions[(0,0)]
    assert force[2] < 0
    # COM offset of +x requires negative y torque for downward mesh force.
    assert torque[1] < 0
    assert state['rod_velocity'][2].max()>0
    np.testing.assert_allclose(torque,np.cross(-com,force),atol=1e-8)


def test_actual_element_radii_and_invalid_radius_are_measured():
    p=np.array([[0.,.5,1.],[0.,0.,0.],[1.,1.,1.]])
    region=GraspRegions(p,[[.7,0,1]],half_width=.1)
    result=region.measure(p,np.zeros_like(p),[.1,.2])
    assert result['local_surface_min_z_m']==pytest.approx([.8])
    assert result['global_surface_min_z_m']==pytest.approx(.8)
    for radius in (-1.,float('nan'),[.1,-.2]):
        with pytest.raises(ValueError,match='radius'):
            region.measure(p,np.zeros_like(p),radius)
