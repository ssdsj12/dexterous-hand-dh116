"""CPU constraints for an opt-in local geometry feasibility repair."""
import importlib

import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_surface_mpc import SurfaceGripMPC


def planner(**kwargs):
    module=importlib.import_module('go2_pvcnn.control.dh116.tabletop_geometric_constrained_mpc')
    return module.GeometricConstrainedMPC(**kwargs)


def inputs(n=1,dtype=torch.float64):
    lo,hi=(torch.tensor(x,dtype=dtype).repeat(2,1) for x in active_limits())
    q=((lo+hi)/2)[None].repeat(n,1,1)
    p=torch.zeros(n,2,2,3,dtype=dtype);p[...,0]=.0082
    j=torch.zeros(n,2,2,3,6,dtype=dtype);j[:,:,0,0,0]=.1;j[:,:,1,0,1]=.1
    targets=p[:,None].repeat(1,4,1,1,1);targets[...,0]-=.004
    kwargs=dict(q=q,command=q.clone(),nominal=q[:,None].repeat(1,4,1,1),points=p,jacobian=j,
        target_points=targets,wrist_motion=torch.zeros_like(targets),weights=torch.ones(n,2,2,dtype=dtype),lower=lo,upper=hi,response=.5)
    pose=torch.zeros(n,7,dtype=dtype);pose[:,3]=1.
    outward=torch.zeros_like(p);outward[...,0]=-1.
    ctx=dict(geometry_mask=torch.ones(n,2,2,dtype=torch.bool),object_pose=pose,cad_outward=outward,
        cylinder_radius_m=.008,cylinder_length_m=.5,pressure_inward=torch.zeros_like(p),
        pressure_protected=torch.zeros(n,2,2,dtype=torch.bool))
    return kwargs,ctx


def assert_exact_geometry(kwargs,ctx,diag):
    points=diag['predicted_points'];radial=points[..., [0,2]].norm(dim=-1)
    gap=radial-ctx['cylinder_radius_m'];original=kwargs['points'][...,[0,2]].norm(dim=-1)-ctx['cylinder_radius_m']
    mask=(ctx['geometry_mask']&(kwargs['weights']>0))[:,None].expand_as(gap)
    assert bool((gap[mask]>=0).all())
    assert bool((gap<(original[:,None]-1e-10))[mask].all())
    inward=-points.clone();inward[...,1]=0.;inward/=radial[...,None]
    normal=ctx['cad_outward']/ctx['cad_outward'].norm(dim=-1,keepdim=True)
    assert bool(((inward*normal[:,None]).sum(-1)[mask]>=.8).all())
    assert bool((points[...,1].abs()[mask]<ctx['cylinder_length_m']/2).all())
    u=diag['trajectory'];step=torch.cat([u[:,:1]-kwargs['command'][:,None],u[:,1:]-u[:,:-1]],1)
    assert bool((step.abs()<=.005).all())
    assert bool((u>=kwargs['lower']).all()&(u<=kwargs['upper']).all())


def test_repair_can_cost_more_than_illegal_soft_candidate():
    kwargs,ctx=inputs();model=planner();model.set_geometry_context(ctx)
    soft,old=SurfaceGripMPC().plan(**kwargs)
    assert (old['predicted_points'][...,0]<.008).any()
    action,diag=model.plan(**kwargs)
    assert_exact_geometry(kwargs,ctx,diag)
    assert diag['feasibility_repair'].all()
    assert (diag['optimized_cost_per_hand']>old['optimized_cost_per_hand']).all()
    torch.testing.assert_close(diag['baseline_soft_cost_per_hand'],old['optimized_cost_per_hand'],rtol=0,atol=0)
    torch.testing.assert_close(diag['repair_cost_delta_per_hand'],diag['optimized_cost_per_hand']-old['optimized_cost_per_hand'])
    torch.testing.assert_close(action,diag['trajectory'][:,0])


def test_already_feasible_soft_is_retained_bitwise_without_repair():
    kwargs,ctx=inputs();kwargs['target_points'][...,0]=.0081
    expected,old=SurfaceGripMPC().plan(**kwargs)
    assert_exact_geometry(kwargs,ctx,old)
    model=planner();model.set_geometry_context(ctx);actual,new=model.plan(**kwargs)
    assert torch.equal(actual,expected)
    assert torch.equal(new['trajectory'],old['trajectory'])
    assert torch.equal(new['optimized_cost_per_hand'],old['optimized_cost_per_hand'])
    assert not new['feasibility_repair'].any()
    assert new['baseline_soft_feasible'].all()
    assert new['baseline_soft_feasibility_checked'].all()
    assert not new['repair_cost_delta_per_hand'].any()
    assert all(not hand['attempted'] for row in new['geometry_constraint_results'] for hand in row)


@pytest.mark.parametrize('empty',['none','mask','weights'])
def test_no_geometry_is_bitwise_original_surface_result(empty):
    kwargs,ctx=inputs();model=planner()
    if empty=='mask':ctx['geometry_mask'].zero_()
    if empty=='weights':kwargs['weights'].zero_()
    if empty!='none':model.geometry_context=ctx
    expected,old=SurfaceGripMPC().plan(**kwargs);actual,new=model.plan(**kwargs)
    assert torch.equal(expected,actual) and old.keys()==new.keys()
    for k,v in old.items():
        assert torch.equal(v,new[k]) if isinstance(v,torch.Tensor) else v==new[k]


def test_failed_hand_keeps_soft_and_other_environment_repairs():
    kwargs,ctx=inputs(2);kwargs['jacobian'][0,0].zero_();kwargs['wrist_motion'][0,:,0,:,0]=.0001
    model=planner();model.set_geometry_context(ctx)
    _,old=SurfaceGripMPC().plan(**kwargs);_,diag=model.plan(**kwargs)
    assert not diag['feasibility_repair'][0,0]
    assert diag['feasibility_repair'][0,1] and diag['feasibility_repair'][1].all()
    assert torch.equal(diag['trajectory'][0,:,0],old['trajectory'][0,:,0])
    assert diag['geometry_constraint_results'][0][0]['accepted'] is False


def test_overloaded_pressure_constraint_uses_relative_motion_without_wrists():
    kwargs,ctx=inputs();ctx['geometry_mask'][...,1]=False
    ctx['pressure_protected'][...,1]=True;ctx['pressure_inward'][...,1,0]=-1.
    kwargs['wrist_motion'][...,1,0]=-.0001
    model=planner();model.set_geometry_context(ctx);_,diag=model.plan(**kwargs)
    assert_exact_geometry(kwargs,ctx,diag)
    relative=diag['predicted_points']-kwargs['points'][:,None]-kwargs['wrist_motion']
    projected=(relative*ctx['pressure_inward'][:,None]).sum(-1)
    assert (projected[...,1]<=1e-10).all()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_dtype_roundtrip_and_inputs_unchanged(dtype):
    kwargs,ctx=inputs(dtype=dtype);copies={k:v.clone() for k,v in kwargs.items() if isinstance(v,torch.Tensor)}
    ctxcopy={k:v.clone() for k,v in ctx.items() if isinstance(v,torch.Tensor)}
    model=planner();model.geometry_context=ctx;action,diag=model.plan(**kwargs)
    assert action.dtype==dtype and action.device==kwargs['q'].device
    assert_exact_geometry(kwargs,ctx,diag)
    for k,v in copies.items():assert torch.equal(v,kwargs[k])
    for k,v in ctxcopy.items():assert torch.equal(v,ctx[k])


def test_context_active_mask_tracks_removed_objectives():
    kwargs,ctx=inputs();ctx['geometry_mask'][...,1]=False
    kwargs['weights'][0,0,0]=0
    model=planner();model.set_geometry_context(ctx);_,old=SurfaceGripMPC().plan(**kwargs);_,diag=model.plan(**kwargs)
    assert torch.equal(diag['trajectory'][0,:,0],old['trajectory'][0,:,0])
    assert not diag['feasibility_repair'][0,0]
    assert diag['feasibility_repair'][0,1]


def test_masked_bad_directions_do_not_poison_constrained_sibling():
    kwargs,ctx=inputs();ctx['geometry_mask'][...,1]=False
    ctx['cad_outward'][...,1,:]=float('nan')
    ctx['pressure_inward'].fill_(float('nan'))
    model=planner();model.set_geometry_context(ctx);_,diag=model.plan(**kwargs)
    assert diag['feasibility_repair'].all()
    assert_exact_geometry(kwargs,ctx,diag)


def test_nanometre_gap_with_tangential_wrist_transport_is_feasible():
    kwargs,ctx=inputs();kwargs['points'][...,0]=.008+3.4e-8
    kwargs['target_points']=kwargs['points'][:,None].repeat(1,4,1,1,1)
    kwargs['target_points'][...,0]=.008
    kwargs['wrist_motion'][...,2]=torch.arange(1,5,dtype=torch.float64)[None,:,None,None]*.0001
    model=planner();model.set_geometry_context(ctx);_,diag=model.plan(**kwargs)
    assert diag['feasibility_repair'].all()
    assert_exact_geometry(kwargs,ctx,diag)


def test_repair_enforces_measured_joint_lead_even_with_wide_caller_bounds():
    kwargs,ctx=inputs();kwargs['command']=kwargs['q']+.099
    kwargs['nominal']=kwargs['command'][:,None].repeat(1,4,1,1)
    kwargs['jacobian'].neg_()
    model=planner();model.set_geometry_context(ctx);_,diag=model.plan(**kwargs)
    assert diag['feasibility_repair'].all()
    assert (diag['trajectory']<=kwargs['q'][:,None]+.1).all()
    assert (diag['trajectory']>=kwargs['q'][:,None]-.1).all()
    assert_exact_geometry(kwargs,ctx,diag)


@pytest.mark.parametrize('field',['geometry_mask','object_pose','cad_outward','pressure_inward','pressure_protected'])
def test_context_schema_rejects_wrong_shape_and_dtype(field):
    kwargs,ctx=inputs();ctx[field]=ctx[field][:0]
    model=planner();model.set_geometry_context(ctx)
    with pytest.raises(ValueError):model.plan(**kwargs)
    kwargs,ctx=inputs();ctx[field]=ctx[field].float()
    model.set_geometry_context(ctx)
    with pytest.raises(ValueError):model.plan(**kwargs)


@pytest.mark.parametrize('field',['cylinder_radius_m','cylinder_length_m'])
def test_context_dimensions_require_explicit_finite_positive(field):
    kwargs,ctx=inputs();ctx[field]=float('nan');model=planner();model.geometry_context=ctx
    with pytest.raises(ValueError):model.plan(**kwargs)
