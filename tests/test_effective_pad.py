"""CPU tests; DH116_AGENT_B_DIR loads isolated uploads without changing /root/hand."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch


def load_module(name):
    root = Path(os.environ.get('DH116_AGENT_B_DIR', Path(__file__).parents[1] / 'src/go2_pvcnn/control/dh116'))
    qualified = 'go2_pvcnn.control.dh116.' + name
    path = root / (name + '.py')
    assert path.exists(), 'task B module is not implemented: ' + name
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules():
    pad = load_module('tabletop_effective_pad')
    runtime = load_module('tabletop_effective_pad_runtime')
    return pad, runtime


def world_fixture(n=3):
    links = ['base_link'] + ['finger' + f for f in ('11_link','12_link','13_link','14_link','21_link','22_link','23_link',
             '31_link','32_link','33_link','41_link','42_link','43_link','51_link','52_link','53_link')]
    hands = []
    for _ in range(2):
        poses = torch.zeros(n, len(links), 7, dtype=torch.float64)
        poses[..., 3] = 1.
        poses[..., :3] = torch.arange(n)[:, None, None] * 2.
        hands.append(NS(body_names=links, data=NS(body_link_pose_w=poses)))
    sensors = []
    for _ in range(34):
        data = [torch.empty(0, 1), torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, 1),
                torch.zeros(n, 1, dtype=torch.long), torch.zeros(n, 1, dtype=torch.long)]
        view = NS(data=data, filter_count=1)
        view.get_contact_data = lambda dt, view=view: tuple(view.data)
        sensors.append(NS(contact_physx_view=view))
    return NS(n=n, device='cpu', dt=1/480, hands=hands, w=NS(sensor_links=links, sensors=sensors))


def fixture_atlas(pad):
    # Explicit planar patch in BODY coordinates. Production atlas has CAD-derived planes.
    row = dict(center_local_m=[0., -.01, .04], tangent_u=[1.,0.,0.], tangent_v=[0.,0.,1.],
               half_widths_m=[.004, .004], planes_local=[[0.,-1.,0.,-.01]],
               surface_tolerance_m=.001, normal_cosine_min=.8,
               source='test planar CAD fixture', geometry_only=True)
    return dict(version='dh116_effective_pad_v1', sides={s: {b: dict(row) for b in pad.DISTAL_LINKS}
                for s in ('left','right')}, limitations=['synthetic test fixture'])


def sampler(modules, monkeypatch, n=3):
    pad, runtime = modules
    monkeypatch.setattr(runtime, 'build_pad_atlas', lambda world: fixture_atlas(pad))
    w = world_fixture(n)
    return w, runtime.PadContactSampler(w)


def contacts(w, link, envs, points, forces, normals=None, separation=None, hand=0):
    view = w.w.sensors[hand*17+w.w.sensor_links.index(link)].contact_physx_view
    counts = torch.bincount(torch.tensor(envs), minlength=w.n)
    starts = counts.cumsum(0)-counts
    p = torch.tensor(points, dtype=torch.float64) + torch.tensor(envs)[:,None]*2.
    nn = normals if normals is not None else [[0.,1.,0.]]*len(points)
    view.data = [torch.tensor(forces, dtype=torch.float64)[:,None], p, torch.tensor(nn,dtype=torch.float64),
                 torch.tensor(separation or [-.0002]*len(points),dtype=torch.float64)[:,None],
                 counts[:,None], starts[:,None]]
    return view.data


def test_backside_hinge_wrong_normal_and_palm_are_rejected(modules, monkeypatch):
    w, s = sampler(modules, monkeypatch)
    raw = contacts(w, 'finger23_link', [0,0,0,0],
                   [[0,-.01,.04],[0,.01,.04],[0,-.01,.012],[0,-.01,.04]], [2,3,4,5],
                   [[0,1,0],[0,-1,0],[0,1,0],[0,-1,0]])
    contacts(w,'base_link',[0],[[0,0,0]],[7])
    snapshots = [v.clone() for v in raw]
    out = s.sample()
    assert out['effective_pad_loads_n'][0,0,1] == 2
    assert out['rejected_contact_loads_n'][0,0] == 19
    torch.testing.assert_close(out['effective_pad_vectors_w'][0,0,1], torch.tensor([0.,2.,0.],dtype=torch.float64))
    for original, current in zip(snapshots,raw):
        torch.testing.assert_close(original,current)


def test_empty_contacts_zero_actual_but_real_fk_references(modules, monkeypatch):
    w,s = sampler(modules, monkeypatch)
    out = s.sample()
    for key in ('effective_pad_loads_n','effective_pad_vectors_w','effective_pad_points_w','effective_pad_depth_m','rejected_contact_loads_n'):
        assert torch.isfinite(out[key]).all() and not out[key].any()
    assert out['effective_pad_valid'].dtype == torch.bool
    assert not out['effective_pad_valid'].any()
    assert out['effective_pad_body_ids'].dtype == torch.long
    assert (out['effective_pad_body_ids'] == -1).all()
    expected=torch.tensor([0.,-.01,.04],dtype=torch.float64)+4.
    torch.testing.assert_close(out['pad_reference_points_w'][2,1,4],expected)
    assert s.atlas['sides']['left']['finger53_link']['geometry_only']
    json.dumps(s.atlas, allow_nan=False)


@pytest.mark.parametrize('residual',[-2.5026850551967217e-28,-2.9737719973809603e-27,
    -1.977040276282538e-12,-5.509183509389004e-08])
def test_native_near_zero_negative_residual_is_unloaded_not_invalid(modules,monkeypatch,residual):
    w,s=sampler(modules,monkeypatch)
    data=contacts(w,'finger23_link',[0,1],[[0,-.01,.04]]*2,[residual,2.])
    before=[v.clone() for v in data]
    out=s.sample()
    assert not out['effective_pad_bad_env'].any()
    assert not out['effective_pad_valid'][0].any()
    assert out['effective_pad_loads_n'][1,0,1]==2.
    torch.testing.assert_close(out['rejected_contact_loads_n'][0,0],torch.tensor(abs(residual),dtype=torch.float64))
    for actual,expected in zip(data,before):torch.testing.assert_close(actual,expected)


def test_signed_separating_native_force_is_recorded_without_becoming_grip(modules,monkeypatch):
    w,s=sampler(modules,monkeypatch)
    contacts(w,'finger23_link',[0],[[0,-.01,.04]],[-1.01e-6])
    out=s.sample()
    assert not out['effective_pad_bad_env'].any()
    assert not out['effective_pad_valid'].any()
    assert out['rejected_contact_loads_n'][0,0]==1.01e-6
    torch.testing.assert_close(s.last_signed_force_w[0,0],torch.tensor([0.,-1.01e-6,0.],dtype=torch.float64))


def test_gathers_offsets_aggregate_depth_and_all_five_digits(modules, monkeypatch):
    w,s=sampler(modules, monkeypatch)
    for link in modules[0].DISTAL_LINKS:
        contacts(w,link,[0,2,2],[[0,-.01,.04],[.001,-.01,.04],[-.001,-.01,.04]],
                 [1.,2.,6.], separation=[-.0001,-.0008,-.0003])
    out=s.sample()
    assert (out['effective_pad_loads_n'][2,0] == 8).all()
    assert not out['effective_pad_valid'][1].any()
    assert (out['effective_pad_depth_m'][2,0] == .0008).all()
    torch.testing.assert_close(out['effective_pad_points_w'][2,0,0],torch.tensor([3.9995,3.99,4.04],dtype=torch.float64))


def test_bad_env_does_not_erase_other_env_or_raw_state(modules, monkeypatch):
    w,s=sampler(modules, monkeypatch)
    contacts(w,'finger23_link',[0,1,2],[[0,-.01,.04]]*3,[1.,float('nan'),3.])
    w.hands[0].data.body_link_pose_w[1,0,0]=float('nan')
    out=s.sample()
    assert out['effective_pad_loads_n'][0,0,1] == 1
    assert out['effective_pad_loads_n'][2,0,1] == 3
    assert out['effective_pad_bad_env'].tolist() == [False,True,False]
    assert torch.isnan(w.hands[0].data.body_link_pose_w[1,0,0])
    for value in out.values():
        assert torch.isfinite(value).all()


@pytest.mark.parametrize('bad', ['negative_count','overrun','zero_normal','bad_quaternion'])
def test_malformed_native_data_is_local_and_finite(modules,monkeypatch,bad):
    w,s=sampler(modules,monkeypatch)
    data=contacts(w,'finger14_link',[0,1],[[0,-.01,.04]]*2,[1,2])
    if bad=='negative_count': data[4][1]=-1
    if bad=='overrun': data[5][1]=1000
    if bad=='zero_normal': data[2][1]=0
    if bad=='bad_quaternion': w.hands[0].data.body_link_pose_w[1,:,3:]=0
    out=s.sample()
    assert out['effective_pad_loads_n'][0,0,0] == 1
    assert not out['effective_pad_valid'][1].any()
    assert out['effective_pad_bad_env'][1]
    assert all(torch.isfinite(v).all() for v in out.values())


def test_vectors_preserve_native_scalar_times_normal(modules,monkeypatch):
    w,s=sampler(modules,monkeypatch)
    contacts(w,'finger23_link',[0],[[0,-.01,.04]],[4.],normals=[[.1,.99,0.]])
    out=s.sample()
    torch.testing.assert_close(out['effective_pad_vectors_w'][0,0,1],torch.tensor([.4,3.96,0.],dtype=torch.float64))


def test_nested_overlapping_native_intervals_flag_every_implicated_env(modules,monkeypatch):
    w,s=sampler(modules,monkeypatch)
    data=contacts(w,'finger23_link',[0]*10,[[0,-.01,.04]]*10,[1.]*10)
    data[4][:,0]=torch.tensor([10,1,1]);data[5][:,0]=torch.tensor([0,2,5])
    out=s.sample()
    assert out['effective_pad_bad_env'].tolist()==[True,True,True]
    assert not out['effective_pad_valid'].any()


def test_unsorted_native_offsets_and_more_than_per_prim_average_are_not_truncated(modules,monkeypatch):
    w,s=sampler(modules,monkeypatch)
    data=contacts(w,'finger23_link',[0]*40+[2],[[0,-.01,.04]]*41,[1.]*41)
    # Move the env2 block first in the buffer; contact starts are not env order.
    data[1][0]=torch.tensor([4.,3.99,4.04])
    data[1][1:]=torch.tensor([0.,-.01,.04])
    data[5][:,0]=torch.tensor([1,0,0])
    out=s.sample()
    assert out['effective_pad_loads_n'][0,0,1]==40
    assert out['effective_pad_loads_n'][2,0,1]==1
    assert not out['effective_pad_bad_env'].any()


def test_zero_force_does_not_manufacture_contact_and_next_sample_clears_old_contact(modules,monkeypatch):
    w,s=sampler(modules,monkeypatch)
    contacts(w,'finger23_link',[0],[[0,-.01,.04]],[1.])
    assert s.sample()['effective_pad_valid'][0,0,1]
    contacts(w,'finger23_link',[0],[[0,-.01,.04]],[0.])
    out=s.sample()
    assert not out['effective_pad_valid'].any()
    assert not out['effective_pad_points_w'].any()
    assert not out['effective_pad_depth_m'].any()


def test_fk_rotation_and_force_are_transformed_in_actual_link_frame(modules,monkeypatch):
    w,s=sampler(modules,monkeypatch)
    bi=w.hands[0].body_names.index('finger23_link')
    w.hands[0].data.body_link_pose_w[0,bi]=torch.tensor([1.,2.,3.,2**-.5,0.,0.,2**-.5])
    contacts(w,'finger23_link',[0],[[1.01,2.,3.04]],[2.],normals=[[-1.,0.,0.]])
    out=s.sample()
    assert out['effective_pad_valid'][0,0,1]
    torch.testing.assert_close(out['pad_reference_points_w'][0,0,1],torch.tensor([1.01,2.,3.04],dtype=torch.float64),atol=1e-7,rtol=0)


def test_finite_extreme_native_data_cannot_escape_as_infinite_output(modules,monkeypatch):
    w,s=sampler(modules,monkeypatch)
    contacts(w,'finger23_link',[0,0],[[0,-.01,.04]]*2,[1e308,1e308])
    out=s.sample()
    assert out['effective_pad_bad_env'][0]
    assert all(torch.isfinite(value).all() for value in out.values())


def test_missing_full_sensor_or_body_is_initialization_error(modules,monkeypatch):
    w,s=sampler(modules,monkeypatch)
    w.w.sensors.pop()
    with pytest.raises(ValueError): modules[1].PadContactSampler(w)


def test_real_source_cad_all_five_patches_and_distinct_thumb_frames(modules):
    # Source extraction only; pxr never launches Isaac.
    from pxr import Usd,UsdGeom
    root=Path(os.environ.get('DH116_AGENT_B_SOURCE_ROOT',Path(__file__).parents[1]))
    source=root/'assets/fixed_dual_panda_dh116_v1/dependencies'
    w=world_fixture()
    w.shapes=[]
    for hi,side in enumerate(('left','right')):
        stage=Usd.Stage.Open(str(next(source.glob('*DH116_'+side+'.usd'))))
        cache=UsdGeom.XformCache()
        for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
            if '/collisions/' not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Mesh): continue
            body=prim.GetParent()
            while body and body.GetName() not in w.hands[hi].body_names: body=body.GetParent()
            if not body: continue
            p=np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            rel=np.array(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
            w.shapes.append((hi,w.hands[hi].body_names.index(body.GetName()),torch.tensor((np.c_[p,np.ones(len(p))]@rel)[:,:3])))
    s=modules[1].PadContactSampler(w)
    json.dumps(s.atlas,allow_nan=False)
    centers=[]
    for hi,side in enumerate(('left','right')):
        for digit,link in enumerate(modules[0].DISTAL_LINKS):
            row=s.atlas['sides'][side][link]
            center=np.array(row['center_local_m']);normal=np.array(row['reference_outward_normal_local'])
            contacts(w,link,[0],[center.tolist()],[2.],normals=[(-normal).tolist()],hand=hi)
            centers.append(center)
    out=s.sample()
    assert out['effective_pad_valid'][0].all()
    assert np.linalg.norm(centers[0]-centers[5])>.03
    assert not np.allclose(centers[1],centers[2])
    assert not np.allclose(centers[1],centers[4])
