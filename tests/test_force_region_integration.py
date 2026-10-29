import copy
import math
import pytest
import torch
from test_dh116_policy_control import enabled


def constrained():
    from go2_pvcnn.control.dh116.tabletop_policy_force_region import enable
    t=enabled();enable(t)
    t.m['contact_valid'][:]=True
    t.m['contact_all_below_centerline'][:]=True
    t.m['contact_center_relative_m'][:]=0.
    t.m['contact_center_relative_m'][:,:,2]=-.002
    t.last_observation=t.observe().clone()
    return t


@pytest.mark.parametrize('load,ready',[(1.,False),(1.999,False),(2.,True),(12.,True),(12.001,False),(15.,False)])
def test_exact_measured_two_to_twelve_newtons_is_required(load,ready):
    t=constrained()
    t.f_requested.fill_(min(12.,max(2.,load)))
    assert bool(t.grip_force_ready(torch.full((2,2),load)).all())==ready


@pytest.mark.parametrize('below',[True,False])
def test_axis_region_is_real_close_admission_gate(below):
    t=constrained();t.contact_time[:]=.2
    t.m['contact_all_below_centerline'][:]=below
    t.consume(t.m,~t.done,1/480,vertical_force_n=torch.ones(2))
    assert bool((t.phase==3).all())==below


def test_last_channels_request_force_not_stiffness():
    a=constrained();b=copy.deepcopy(a)
    aa=torch.zeros((2,20),dtype=torch.float64);bb=aa.clone()
    aa[:,18:]=-1;bb[:,18:]=1
    a.control(aa,~a.done,1/480);b.control(bb,~b.done,1/480)
    torch.testing.assert_close(a.f_requested,torch.full_like(a.f_requested,2.))
    torch.testing.assert_close(b.f_requested,torch.full_like(b.f_requested,12.))
    torch.testing.assert_close(a.policy_stiffness,b.policy_stiffness)
    assert t_obs_size(a)==t_obs_size(enabled())+6


def t_obs_size(t):return t.observe().shape[-1]


def test_no_success_credit_if_grasp_region_is_wrong_during_lift():
    t=constrained();t.phase[:]=3;t.success_time[:]=.5
    t.m['bottom'][:]=.34;t.m['contact_all_below_centerline'][:]=False
    t.consume(t.m,~t.done,1/480,vertical_force_n=torch.ones(2))
    assert not t.success.any() and not t.success_time.any()


def test_contact_region_uses_local_rod_axis_for_tilted_object():
    from go2_pvcnn.control.dh116.tabletop_contact_geometry import aggregate_contact_geometry
    # Axis tilted 45 degrees upwards: world-positive-Z point lies below its local axial centerline.
    axis=torch.tensor([[0.,math.sqrt(.5),math.sqrt(.5)]],dtype=torch.float64)
    point=torch.tensor([0.,.02,.018],dtype=torch.float64).expand(1,2,1,3).clone()
    force=torch.ones_like(point)
    geom=aggregate_contact_geometry(point,force,torch.zeros((1,3)),object_axis_w=axis)
    assert geom['all_below_centerline'].all()
    point[:,:,:,2]=.022
    assert not aggregate_contact_geometry(point,force,torch.zeros((1,3)),object_axis_w=axis)['all_below_centerline'].any()


def test_contact_reward_half_above_double_below_without_weakening_admission():
    a=constrained();b=copy.deepcopy(a)
    a.m['contact_all_below_centerline'][:]=False
    for t in (a,b):
        t.m['vertical_hand_force_n'][:]=1.
        t.consume(t.m,~t.done,.01,vertical_force_n=torch.full((2,),2.))
    assert a.terms['contact'][0]==pytest.approx(.005)
    assert b.terms['contact'][0]==pytest.approx(.02)
    assert not a.grip_force_ready(a.m['hand_object_loads']).any()
