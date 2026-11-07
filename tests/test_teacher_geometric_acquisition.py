"""CPU tests for bounded CAD geometry references, without invented contacts."""
import importlib
import math

import pytest
import torch


def acquire(**kwargs):
    module = importlib.import_module('go2_pvcnn.control.dh116.tabletop_teacher_geometric_acquisition')
    return module.geometric_acquisition_targets(**kwargs)


def inputs(dtype=torch.float64):
    points = torch.zeros(2, 2, 5, 3, dtype=dtype)
    points[:, 0, :, 0] = .012
    points[:, 1, :, 0] = -.012
    outward = torch.zeros_like(points)
    outward[:, 0, :, 0] = -1.
    outward[:, 1, :, 0] = 1.
    pose = torch.zeros(2, 7, dtype=dtype)
    pose[:, 3] = 1.
    return dict(points=points, pad_outward=outward, object_pose=pose,
        loads=torch.zeros(2, 2, 5, dtype=dtype),
        contact_valid=torch.zeros(2, 2, 5, dtype=torch.bool),
        cylinder_radius_m=.01, cylinder_length_m=.5)


def test_signed_sidewall_targets_horizon_and_disabled_digits():
    d = inputs(); r = acquire(**d)
    assert r['geometry_eligible'][..., :3].all()
    assert not r['geometry_eligible'][..., 3:].any()
    assert r['target_points'].shape == (2, 4, 2, 5, 3)
    assert r['weights'].shape == (2, 4, 2, 5)
    assert r['weights'][..., :3].eq(1).all() and r['weights'][..., 3:].eq(0).all()
    distance = torch.arange(1, 5, dtype=torch.float64) * (.001/60)
    torch.testing.assert_close(r['target_points'][0, :, 0, 0, 0], .012-distance)
    torch.testing.assert_close(r['target_points'][0, :, 1, 0, 0], -.012+distance)
    assert r['closest_surface_points'][0, 0, 0, 0] == .01
    assert r['closest_surface_points'][0, 1, 0, 0] == -.01
    assert r['requested_distance_m'][..., :3].eq(.001/60).all()
    assert r['force_prediction'] is False


def test_translated_rotated_local_y_cylinder_and_antipodal_quaternions():
    d = inputs(); d['object_pose'][:, :3] = torch.tensor([1.,2.,3.],dtype=torch.float64)
    d['object_pose'][:, 3:] = torch.tensor([math.sqrt(.5),0.,0.,math.sqrt(.5)],dtype=torch.float64)
    d['points'][:] = torch.tensor([.9,2.012,3.],dtype=torch.float64)
    d['pad_outward'][:] = torch.tensor([0.,-1.,0.],dtype=torch.float64)
    r = acquire(**d)
    assert r['geometry_eligible'][..., :3].all()
    torch.testing.assert_close(r['closest_surface_points'][0,0,0], torch.tensor([.9,2.01,3.],dtype=torch.float64),rtol=0,atol=1e-9)
    d['object_pose'][:, 3:] *= -1
    flipped = acquire(**d)
    for k,v in r.items():
        if isinstance(v,torch.Tensor): assert torch.equal(v,flipped[k])


def test_gap_budget_speed_each_bound_and_zero_budget_reason():
    d=inputs(); budget=torch.full_like(d['loads'],1e-5);budget[0,0,0]=0.
    d['points'][0,0,1,0]=.010001
    r=acquire(**d,remaining_distance_m=budget)
    assert not r['geometry_eligible'][0,0,0]
    assert r['reason_code'][0,0,0] == r['reason_codes']['budget_exhausted']
    distance=(r['target_points']-d['points'][:,None]).norm(dim=-1)
    assert bool((distance<=budget[:,None]+1e-15).all())
    assert float(distance[0,:,0,1].max()) <= 1e-6+1e-15
    assert float(r['requested_distance_m'].max()) <= 1e-5
    small=acquire(**inputs(),max_speed_m_s=.0001,dt=.01,remaining_distance_m=.001)
    assert float(small['requested_distance_m'].max()) == pytest.approx(1e-6)


@pytest.mark.parametrize('kind', ['inside','surface','far','end','beyond_end','wrong_facing','zero_normal'])
def test_rejects_non_sidewall_or_unreachable_geometry(kind):
    d=inputs()
    if kind in ['inside','surface','far']:d['points'][0,0,0,0]={'inside':.009,'surface':.01,'far':.021}[kind]
    if kind in ['end','beyond_end']:d['points'][0,0,0,1]={'end':.25,'beyond_end':.26}[kind]
    if kind=='wrong_facing':d['pad_outward'][0,0,0,0]=1.
    if kind=='zero_normal':d['pad_outward'][0,0,0]=0.
    r=acquire(**d)
    assert not r['geometry_eligible'][0,0,0]
    assert r['geometry_eligible'][1,1,0]
    assert r['weights'][0,:,0,0].eq(0).all()
    torch.testing.assert_close(r['target_points'][0,:,0,0],d['points'][0,0,0].expand(4,3),rtol=0,atol=0)


def test_load_threshold_blocks_unknown_contact_but_subthreshold_can_acquire():
    d=inputs();d['loads'][0,0,:3]=torch.tensor([.050001,.05,0.],dtype=torch.float64);d['contact_valid'][0,0,2]=True
    r=acquire(**d)
    assert r['geometry_eligible'][0,0,:3].tolist()==[False,True,True]
    assert r['reason_code'][0,0,0]==r['reason_codes']['bearing_load']


@pytest.mark.parametrize('field', ['points','pad_outward','loads','remaining_distance_m'])
def test_nonfinite_pad_isolated_and_all_outputs_finite(field):
    d=inputs();d['remaining_distance_m']=torch.ones_like(d['loads'])*.001
    d[field][0,0,0]=float('nan')
    r=acquire(**d)
    assert not r['geometry_eligible'][0,0,0] and r['geometry_eligible'][1,1,0]
    for v in r.values():
        if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()


@pytest.mark.parametrize('kind',['nonfinite','zero_quaternion','overflow_quaternion'])
def test_invalid_object_disables_whole_environment_only(kind):
    d=inputs()
    if kind=='nonfinite':d['object_pose'][0,0]=float('nan')
    elif kind=='zero_quaternion':d['object_pose'][0,3:]=0.
    else:d['object_pose'][0,3:]=1e308
    r=acquire(**d)
    assert not r['geometry_eligible'][0].any() and r['geometry_eligible'][1,:,:3].all()
    assert r['reason_code'][0].eq(r['reason_codes']['invalid_object']).all()
    for v in r.values():
        if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_owned_outputs_leave_every_input_unchanged(dtype):
    d=inputs(dtype);d['remaining_distance_m']=torch.full_like(d['loads'],.001)
    copies={k:v.clone() for k,v in d.items() if isinstance(v,torch.Tensor)}
    r=acquire(**d)
    for v in r.values():
        if isinstance(v,torch.Tensor):v.zero_()
    for k,v in copies.items():assert torch.equal(d[k],v)


@pytest.mark.parametrize('kwargs',[
    dict(horizon=0),dict(horizon=True),dict(dt=0),dict(dt=float('nan')),
    dict(dt=1e308,horizon=4),dict(max_speed_m_s=.002),dict(max_speed_m_s=-1),
    dict(max_gap_m=.011),dict(max_gap_m=0),dict(minimum_facing_cosine=1.1),
    dict(cylinder_radius_m=0),dict(cylinder_length_m=float('inf')),
    dict(remaining_distance_m=-1),dict(remaining_distance_m=float('inf'))])
def test_illegal_parameters_raise(kwargs):
    d=inputs();d.update(kwargs)
    with pytest.raises(ValueError):acquire(**d)


@pytest.mark.parametrize('field',['pad_outward','object_pose','loads','contact_valid','remaining_distance_m'])
def test_shape_and_dtype_contracts_raise(field):
    d=inputs();d['remaining_distance_m']=torch.ones_like(d['loads'])
    d[field]=d[field][:1]
    with pytest.raises(ValueError):acquire(**d)
    d=inputs();d['remaining_distance_m']=torch.ones_like(d['loads'])
    d[field]=d[field].float()
    with pytest.raises(ValueError):acquire(**d)


def test_facing_threshold_is_cosine_of_normalized_cad_normal():
    d=inputs();d['pad_outward'][0,0,0]=torch.tensor([-.8,.6,0.],dtype=torch.float64)*5
    d['pad_outward'][0,0,1]=torch.tensor([-.79,math.sqrt(1-.79**2),0.],dtype=torch.float64)
    r=acquire(**d)
    assert r['geometry_eligible'][0,0,0] and not r['geometry_eligible'][0,0,1]
    assert r['facing_cos'][0,0,0] == pytest.approx(.8)


def test_negative_numeric_pad_and_budget_are_isolated():
    d=inputs();d['loads'][0,0,0]=-.1
    d['remaining_distance_m']=torch.full_like(d['loads'],.001);d['remaining_distance_m'][0,0,1]=-.1
    r=acquire(**d)
    assert r['geometry_eligible'][0,0,:3].tolist()==[False,False,True]
    assert r['reason_code'][0,0,0]==r['reason_codes']['invalid_pad']
    assert r['reason_code'][0,0,1]==r['reason_codes']['invalid_budget']


def test_sidewall_radial_projection_preserves_axial_coordinate():
    d=inputs();d['points'][0,0,0]=torch.tensor([.012,.11,.016],dtype=torch.float64)
    d['pad_outward'][0,0,0]=torch.tensor([-.6,0.,-.8],dtype=torch.float64)
    r=acquire(**d)
    assert r['geometry_eligible'][0,0,0]
    torch.testing.assert_close(r['closest_surface_points'][0,0,0],torch.tensor([.006,.11,.008],dtype=torch.float64))
    assert r['target_points'][0,:,0,0,1].eq(.11).all()
