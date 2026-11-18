"""Uniform native-buffer CPU equivalence; no Isaac process is constructed.

DH116_PAD_BATCH_MODULE optionally selects an isolated candidate for remote tests.
"""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_effective_pad_runtime as runtime


def test_public_api_is_implemented():
    path = Path(os.environ.get(
        'DH116_PAD_BATCH_MODULE',
        Path(__file__).parents[1] / 'src/go2_pvcnn/control/dh116/tabletop_effective_pad_batch.py'))
    assert path.exists(), 'sample_native_batched is not implemented'


@pytest.fixture
def batch():
    path = Path(os.environ.get(
        'DH116_PAD_BATCH_MODULE',
        Path(__file__).parents[1] / 'src/go2_pvcnn/control/dh116/tabletop_effective_pad_batch.py'))
    assert path.exists(), 'sample_native_batched is not implemented'
    spec = importlib.util.spec_from_file_location('go2_pvcnn.control.dh116.tabletop_effective_pad_batch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_sampler(n=4, capacity=16, planes=1):
    links = ['base_link'] + ['finger' + f for f in (
        '11_link', '12_link', '13_link', '14_link', '21_link', '22_link', '23_link',
        '31_link', '32_link', '33_link', '41_link', '42_link', '43_link',
        '51_link', '52_link', '53_link')]
    hands = []
    for h in range(2):
        pose = torch.zeros(n, 17, 7, dtype=torch.float64)
        pose[..., 3] = 1.
        pose[..., :3] = torch.arange(n)[:, None, None] * 2. + h * .25
        hands.append(NS(body_names=links[:], data=NS(body_link_pose_w=pose)))
    s = runtime.PadContactSampler.__new__(runtime.PadContactSampler)
    s.n, s.device = n, 'cpu'
    s.world = NS(n=n, device='cpu', dt=1/480, hands=hands)
    s.rows, s.bindings = {}, []
    for h in range(2):
        for bi, link in enumerate(links):
            di = runtime.DISTAL_LINKS.index(link) if link in runtime.DISTAL_LINKS else None
            # NaN padding must neither own slots nor poison unrelated environments.
            buf = [torch.full((capacity, 1), float('nan'), dtype=torch.float64),
                   torch.full((capacity, 3), float('nan'), dtype=torch.float64),
                   torch.full((capacity, 3), float('nan'), dtype=torch.float64),
                   torch.full((capacity, 1), float('nan'), dtype=torch.float64),
                   torch.zeros(n, 1, dtype=torch.long), torch.zeros(n, 1, dtype=torch.long)]
            view = NS(data=buf, filter_count=1)
            view.get_contact_data = lambda dt, view=view: tuple(view.data)
            s.bindings.append((h, bi, di, NS(contact_physx_view=view)))
            if di is not None:
                # Ragged plane counts, cross the original 32-plane group boundary.
                pp = torch.tensor([[0., -1., 0., -.01]] * (planes + di), dtype=torch.float64)
                pp[1:, 3] -= torch.arange(1, len(pp), dtype=torch.float64) * 1e-6
                s.rows[h, di] = dict(
                    center_local_m=torch.tensor([0., -.01, .04], dtype=torch.float64),
                    tangent_u=torch.tensor([1., 0., 0.], dtype=torch.float64),
                    tangent_v=torch.tensor([0., 0., 1.], dtype=torch.float64),
                    half_widths_m=torch.tensor([.004, .004], dtype=torch.float64),
                    planes_local=pp, surface_tolerance_m=.001, normal_cosine_min=.8)
    return s


def data_for(s, hand=0, digit=1, body=None):
    return next(sensor.contact_physx_view.data for h, b, d, sensor in s.bindings
                if h == hand and (d == digit if body is None else b == body))


def put(s, env=0, hand=0, digit=1, slot=0, forces=(2.,), points=None,
        normals=None, separation=None, body=None):
    buf = data_for(s, hand, digit, body)
    k = len(forces)
    bi = next(b for h, b, d, _ in s.bindings
              if h == hand and (d == digit if body is None else b == body))
    pose = s.world.hands[hand].data.body_link_pose_w[env, bi]
    rotation = runtime._rotation(pose)
    local_p = torch.tensor(points or [[0., -.01, .04]] * k, dtype=torch.float64)
    local_n = torch.tensor(normals or [[0., 1., 0.]] * k, dtype=torch.float64)
    buf[0][slot:slot+k, 0] = torch.tensor(forces, dtype=torch.float64)
    buf[1][slot:slot+k] = local_p @ rotation.T + pose[:3]
    buf[2][slot:slot+k] = local_n @ rotation.T
    buf[3][slot:slot+k, 0] = torch.tensor(separation or [-.0002] * k, dtype=torch.float64)
    buf[4][env], buf[5][env] = k, slot
    return buf


def equivalent(batch, s):
    raw = [[v.clone() for v in sensor.contact_physx_view.data] for *_, sensor in s.bindings]
    poses = [h.data.body_link_pose_w.clone() for h in s.world.hands]
    expected = s.sample()
    signed = s.last_signed_force_w.clone()
    s.last_signed_force_w = torch.full_like(signed, 999.)
    actual = batch.sample_native_batched(s)
    assert actual is not None
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].shape == expected[key].shape, key
        assert actual[key].dtype == expected[key].dtype, key
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(s.last_signed_force_w, signed, rtol=1e-12, atol=1e-12)
    for (*_, sensor), snapshot in zip(s.bindings, raw):
        for current, before in zip(sensor.contact_physx_view.data, snapshot):
            torch.testing.assert_close(current, before, rtol=0, atol=0, equal_nan=True)
    for hand, before in zip(s.world.hands, poses):
        torch.testing.assert_close(hand.data.body_link_pose_w, before, rtol=0, atol=0, equal_nan=True)
    return actual


@pytest.mark.parametrize('capacity', [0, 16])
def test_empty_actuals_and_separate_fk_references(batch, capacity):
    out = equivalent(batch, make_sampler(capacity=capacity))
    assert not out['effective_pad_valid'].any()
    assert not out['effective_pad_points_w'].any()
    assert out['pad_reference_points_w'].any()


def test_all_34_sensors_both_hands_and_depth(batch):
    s = make_sampler()
    for h, b, d, _ in s.bindings:
        for e in range(s.n):
            put(s, e, h, d, e*3, (1., 2., 6.), body=b,
                points=[[0., -.01, .04], [.001, -.01, .04], [-.001, -.01, .04]],
                normals=[[.1, .99, 0.]]*3, separation=[-.0001, -.0008, -.0003])
    out = equivalent(batch, s)
    assert out['effective_pad_valid'].all()
    assert (out['effective_pad_loads_n'] == 9).all()
    assert (out['effective_pad_depth_m'] == .0008).all()
    assert (out['rejected_contact_loads_n'] == 12*9).all()


@pytest.mark.parametrize('negative', [-2.5e-28, -1.977e-12, -1.01e-6, -3.])
def test_negative_release_is_rejected_but_keeps_signed_force(batch, negative):
    s = make_sampler()
    put(s, forces=(negative,))
    put(s, env=2, slot=3, forces=(2.,))
    out = equivalent(batch, s)
    assert not out['effective_pad_bad_env'].any()
    assert out['rejected_contact_loads_n'][0, 0] == abs(negative)
    assert s.last_signed_force_w[0, 0, 1] == negative


@pytest.mark.parametrize('fault', ['count', 'start', 'overrun', 'count_nan', 'start_nan',
    'normal_nan', 'normal_zero', 'normal_inf', 'point_nan', 'force_nan', 'separation_nan',
    'pose_nan', 'pose_inf', 'quaternion_zero', 'quaternion_scale', 'quaternion_nan'])
def test_invalid_data_only_isolates_its_environment(batch, fault):
    s = make_sampler()
    for e in range(s.n):
        put(s, env=e, slot=e*2)
    b = data_for(s)
    if fault == 'count': b[4][1] = -1
    elif fault == 'start': b[5][1] = -1
    elif fault == 'overrun': b[4][1] = 100
    elif fault in ('count_nan', 'start_nan'):
        i = 4 if fault == 'count_nan' else 5
        b[i] = b[i].double(); b[i][1] = float('nan')
    elif fault.startswith('normal_'):
        b[2][2] = {'normal_nan': float('nan'), 'normal_zero': 0., 'normal_inf': float('inf')}[fault]
    elif fault == 'point_nan': b[1][2, 0] = float('nan')
    elif fault == 'force_nan': b[0][2] = float('nan')
    elif fault == 'separation_nan': b[3][2] = float('nan')
    else:
        # Invalid non-distal pose still quarantines only this env.
        pose = s.world.hands[1].data.body_link_pose_w
        if fault == 'pose_nan': pose[1, 0, 0] = float('nan')
        elif fault == 'pose_inf': pose[1, 0, 0] = float('inf')
        elif fault == 'quaternion_zero': pose[1, 0, 3:] = 0
        elif fault == 'quaternion_scale': pose[1, 0, 3:] *= 2
        elif fault == 'quaternion_nan': pose[1, 0, 3] = float('nan')
    out = equivalent(batch, s)
    assert out['effective_pad_bad_env'].tolist() == [False, True, False, False]
    assert out['effective_pad_loads_n'][2, 0, 1] == 2


@pytest.mark.parametrize('nested', [False, True])
def test_overlap_marks_both_owners_including_nested_intervals(batch, nested):
    s = make_sampler()
    b = put(s, forces=(1.,)*10)
    b[4][:, 0] = torch.tensor([10 if nested else 3, 1, 1, 0])
    b[5][:, 0] = torch.tensor([0, 2, 5 if nested else 9, 16])
    out = equivalent(batch, s)
    assert out['effective_pad_bad_env'].tolist() == [True, True, nested, False]


def test_unsorted_intervals_gaps_and_nan_padding(batch):
    s = make_sampler(capacity=64)
    put(s, env=0, slot=3, forces=(1.,)*40)
    put(s, env=2, slot=0, forces=(7.,))
    out = equivalent(batch, s)
    assert out['effective_pad_loads_n'][0, 0, 1] == 40
    assert out['effective_pad_loads_n'][2, 0, 1] == 7
    assert not out['effective_pad_bad_env'].any()


def test_rotated_frames_ragged_planes_and_geometry_gates(batch):
    s = make_sampler(capacity=32, planes=33)
    for h in range(2):
        s.world.hands[h].data.body_link_pose_w[..., 3:] = torch.tensor([2**-.5, 0., 0., 2**-.5])
        for d in range(5):
            put(s, hand=h, digit=d, forces=(2., 3., 4., 5., 6.),
                points=[[0., -.01, .04], [.0041, -.01, .04], [0., .01, .04],
                        [0., -.01, .012], [0., -.01, .04]],
                normals=[[0., 1., 0.]]*4 + [[0., -1., 0.]])
    out = equivalent(batch, s)
    assert (out['effective_pad_loads_n'][0] == 2).all()
    assert (out['rejected_contact_loads_n'][0] == 90).all()


def test_chunk_boundary_and_nonuniform_force_accumulation(batch, monkeypatch):
    sizes = []
    original = batch._patch_batched

    def observe(point, *args):
        sizes.append(point.shape[1])
        return original(point, *args)

    monkeypatch.setattr(batch, '_patch_batched', observe)
    s = make_sampler(n=3, capacity=8201, planes=33)
    for h in range(2):
        for d in range(5):
            put(s, env=0, hand=h, digit=d, forces=tuple((i % 7 + 1)*.125 for i in range(8195)))
            put(s, env=2, hand=h, digit=d, slot=8200, forces=(3.,))
    out = equivalent(batch, s)
    assert out['effective_pad_valid'][[0, 2]].all()
    assert sizes == [4096, 4096, 9]


@pytest.mark.parametrize('kind', ['ellipse', 'surface', 'normal', 'plane_tie', 'overflow'])
def test_thresholds_ties_and_overflow_match_without_relaxation(batch, kind):
    s = make_sampler(capacity=16, planes=33)
    if kind == 'ellipse':
        put(s, forces=(2., 3., 4.), points=[[.004, -.01, .04],
            [.004+1e-12, -.01, .04], [.004-1e-12, -.01, .04]])
    elif kind == 'surface':
        put(s, forces=(2., 3., 4.), points=[[0., -.011, .04],
            [0., -.011-1e-12, .04], [0., -.011+1e-12, .04]])
    elif kind == 'normal':
        put(s, forces=(2., 3., 4.), normals=[[.6, .8, 0.],
            [(.36+1.6e-12)**.5, .8-1e-12, 0.], [(.36-1.6e-12)**.5, .8+1e-12, 0.]])
    elif kind == 'plane_tie':
        pp = s.rows[0, 1]['planes_local']
        pp[:] = torch.tensor([0., -1., 0., -.02])
        pp[0] = torch.tensor([0., -1., 0., -.01])
        pp[32] = torch.tensor([0., 1., 0., .01])
        put(s)
    else:
        put(s, forces=(1e308, 1e308))
    out = equivalent(batch, s)
    if kind == 'overflow':
        assert out['effective_pad_bad_env'][0]


@pytest.mark.parametrize('seed', range(3))
def test_randomized_unsorted_intervals_across_34_sensors(batch, seed):
    s = make_sampler(n=7, capacity=91, planes=33)
    gen = torch.Generator().manual_seed(seed)
    for *_, sensor in s.bindings:
        b = sensor.contact_physx_view.data
        b[0][:] = torch.randn(b[0].shape, generator=gen, dtype=torch.float64)
        b[1][:] = torch.tensor([0., -.01, .04])
        b[2][:] = torch.tensor([0., 1., 0.])
        b[3][:] = -.0002
        b[4][:, 0] = torch.randint(0, 9, (7,), generator=gen)
        b[5][:, 0] = torch.randint(0, 91, (7,), generator=gen)
    equivalent(batch, s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA profiling is optional')
def test_cuda_batch_has_no_scalar_reads_or_explicit_stream_synchronizations(batch):
    s = make_sampler()
    put(s)
    s.device = s.world.device = 'cuda:0'
    for hand in s.world.hands:
        hand.data.body_link_pose_w = hand.data.body_link_pose_w.cuda()
    for *_, sensor in s.bindings:
        sensor.contact_physx_view.data = [v.cuda() for v in sensor.contact_physx_view.data]
    for row in s.rows.values():
        for key, value in row.items():
            if isinstance(value, torch.Tensor): row[key] = value.cuda()
    equivalent(batch, s)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        batch.sample_native_batched(s)
    events = {e.key: e.count for e in prof.key_averages()}
    assert events.get('aten::item', 0) == 0
    assert events.get('aten::_local_scalar_dense', 0) == 0
    assert events.get('cudaStreamSynchronize', 0) == 0


def test_new_sample_clears_previous_actuals_and_signed_force(batch):
    s = make_sampler()
    put(s)
    equivalent(batch, s)
    for *_, sensor in s.bindings:
        sensor.contact_physx_view.data[4].zero_()
    out = equivalent(batch, s)
    assert not out['effective_pad_valid'].any()
    assert not s.last_signed_force_w.any()


def test_same_physics_step_snapshot_avoids_duplicate_native_read_without_changing_measurements(batch):
    s=make_sampler();put(s,forces=(1.2,1.8))
    expected=runtime.PadContactSampler.sample(s)
    s.native_contact_data=[tuple(v.clone() for v in sensor.contact_physx_view.data)
        for *_,sensor in s.bindings]
    def duplicate_read(dt):raise AssertionError('duplicate native read')
    for *_,sensor in s.bindings:sensor.contact_physx_view.get_contact_data=duplicate_read
    actual=batch.sample_native_batched(s)
    for key in expected:torch.testing.assert_close(actual[key],expected[key],equal_nan=True)


def test_incomplete_step_snapshot_falls_back_without_reusing_old_force(batch):
    s=make_sampler();put(s)
    s.native_contact_data=[]
    sentinel=torch.tensor([123.]);s.last_signed_force_w=sentinel
    assert batch.sample_native_batched(s) is None
    assert s.last_signed_force_w is sentinel


def test_runtime_sampler_explicit_batched_flag_reuses_same_step_packet(batch):
    s=make_sampler();put(s,forces=(1.2,1.8))
    expected=runtime.PadContactSampler.sample(s)
    s.use_batched_native=True
    s.native_contact_data=[tuple(v.clone() for v in sensor.contact_physx_view.data)
        for *_,sensor in s.bindings]
    def duplicate_read(dt):raise AssertionError('batched runtime should reuse the native packet')
    for *_,sensor in s.bindings:sensor.contact_physx_view.get_contact_data=duplicate_read
    actual=runtime.PadContactSampler.sample(s)
    for key in expected:torch.testing.assert_close(actual[key],expected[key],equal_nan=True)


@pytest.mark.parametrize('unsupported', ['capacity', 'count_shape', 'point_shape', 'sensor_count'])
def test_unsupported_layout_returns_none_without_partial_state(batch, unsupported):
    s = make_sampler()
    b = data_for(s)
    if unsupported == 'capacity':
        for i in range(4): b[i] = b[i][:-1]
    elif unsupported == 'count_shape': b[4] = b[4][:-1]
    elif unsupported == 'point_shape': b[1] = b[1][:, :2]
    elif unsupported == 'sensor_count': s.bindings.pop()
    sentinel = torch.tensor([123.])
    s.last_signed_force_w = sentinel
    assert batch.sample_native_batched(s) is None
    assert s.last_signed_force_w is sentinel
