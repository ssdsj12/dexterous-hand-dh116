"""Frozen legacy numerical oracles; CPU geometry tests grant no native admission.

Oracle source SHA256s are recorded in runs/production-consolidation/
geometry-oracle-source-sha256.json. Expected values below were evaluated once
from the original functions, not rebuilt by the production implementation.
"""
from copy import deepcopy
import importlib
import importlib.util
import numpy as np
import pytest
import torch

def _math():
    name = 'dh116.physics.geometry'
    assert importlib.util.find_spec(name) is not None, 'production geometry_math extraction is missing'
    return importlib.import_module(name)

def _grid():
    (x, y, z) = np.indices((5, 5, 5))
    return dict(values=0.005 + x * 0.01 - 0.02 + y * z * 0.0007, origin=np.full(3, -0.02), spacing=np.array(0.01))

def _sampling_points():
    return np.array([[-0.02, -0.02, -0.02], [-0.013271, -0.00123, 0.00719], [0.01999999, 0.0117, -0.0036], [0.02, 0, 0], [-0.02000001, -0.011, 0.001], [0, 0, 0], [0.004321, -0.00291, -0.01371]])

def _near_case(fault):
    grid = _grid()
    nodes = np.random.default_rng(745).uniform(0.0051, 0.0151, (41, 3))
    nodes[0] = [-0.013271, -0.013271, -0.011719]
    if fault in ('zero', 'threshold'):
        (x, y, z) = nodes[0]
        grid['values'] -= x + (y + 0.02) / 0.01 * (z + 0.02) / 0.01 * 0.0007 - 1e-10
    if fault == 'lattice_face':
        nodes[0] = [-0.02 + 1e-07, -0.013271, -0.011719]
    if fault == 'threshold':
        grid['values'] -= 0.00015
    if fault == 'flat':
        grid['values'] = 1e-46 + np.indices((5, 5, 5))[0] * 1e-50
    if fault == 'outside':
        nodes[:] = 0.1
    nodes[:, 2] += 0.6
    return (grid, nodes)

def _poses(n):
    poses = torch.zeros((n, 2, 23, 7), dtype=torch.float64)
    poses[..., 6] = 1
    poses[..., 2] = 0.6
    return poses

def _capture_owner(grid=None):
    from dh116.physics.geometry import TensorGeometry as TensorGeometry
    shapes = [(side, 0, 'cooked:pad', np.zeros((1, 3))) for side in (0, 1)]
    pads = () if grid is None else [(side, 0, 'pad', grid) for side in (0, 1)]
    return TensorGeometry(shapes, pads, device='cpu')

def _materialized():
    names = tuple((tuple(('body' + str(i) for i in range(23))) for _ in (0, 1)))
    (geometry, grids, sources) = ({}, {}, {})
    for side in ('left', 'right'):
        geometry[side] = {kind: {'body' + str(i): np.array([[i * 0.001, 0, 0], [i * 0.001 + 0.0001, 0.0002, 0.0003]]) for i in range(count)} for (kind, count) in (('source', 17), ('cooked', 5))}
        grids[side] = {name: _grid() for name in ('body0', 'body1')}
        sources[side] = {'source_sha256': side, 'sdf_binding': {name: {'sha256': str(i) * 64, 'capture_sha256': 'a' * 64} for (i, name) in enumerate(grids[side])}}
    return (geometry, grids, sources, names)

def test_fp32_training_pad_sampling_uses_only_locked_lattice():
    owner = _capture_owner(_grid())
    nodes = torch.zeros((2, 41, 3), dtype=torch.float32)
    nodes[..., 2] = 0.6
    nodes[1, :, 0] = 0.1
    radii = torch.full((2,), 0.005)
    gaps, valid = owner.policy_pad_gaps(_poses(2).float(), nodes, radii)
    assert gaps.dtype == torch.float32
    torch.testing.assert_close(gaps[0], torch.full((2, 41), 0.0028), atol=1e-7, rtol=0)
    assert valid[0].all() and not valid[1].any()


def test_approach_distance_is_continuous_at_the_locked_lattice_face():
    from types import SimpleNamespace
    from dh116.physics.runtime import NativeTensorPhysicsRuntime

    owner = _capture_owner(_grid())
    nodes = torch.zeros((3, 41, 3), dtype=torch.float32)
    nodes[..., 2] = .6
    offsets = nodes.new_tensor((-.0001, 0., .0001, .001))
    inherited_x = .02 + offsets
    nodes[0, :4, 0] = inherited_x
    nodes[1, :, 0] = -.021
    # Let every node of each lane cross the same face to exercise the reward source.
    approach_nodes = nodes.new_zeros((4, 41, 3))
    approach_nodes[..., 0] = inherited_x[:, None]
    approach_nodes[..., 2] = .6
    runtime = SimpleNamespace(nodes=approach_nodes, state={'link_pose_xyzw': _poses(4).float()},
                              full_grasp=True, radii=nodes.new_full((4,), .005),
                              device='cpu', geometry=owner)
    distance = NativeTensorPhysicsRuntime.distance(runtime, stamp=None)
    torch.testing.assert_close(distance, (.0228 + offsets).double(), atol=1e-7, rtol=0)
    assert not runtime._pad_backface.any() and not runtime._soft_pad_depth.any()

    nodes[2, :, 1] = torch.nan
    gaps, valid = owner.policy_pad_gaps(_poses(3).float(), nodes, nodes.new_full((3,), .005))
    assert valid[0, 0, :4].tolist() == [True, False, False, False]
    torch.testing.assert_close(gaps[0, :, :4], (.0228 + offsets).expand(2, -1), atol=1e-7, rtol=0)
    assert not valid[1].any()
    torch.testing.assert_close(gaps[1], nodes.new_full((2, 41), .001), atol=1e-7, rtol=0)
    assert torch.isposinf(gaps[2]).all() and not valid[2].any()

    # A nearer exterior node still guides the policy when a farther node is in the box.
    approach_nodes[:, :, 0] = 0.
    approach_nodes[:, 10, 0] = nodes.new_tensor((-.0199, -.0201, -.0211, -.0221))
    distance = NativeTensorPhysicsRuntime.distance(runtime, stamp=None)
    torch.testing.assert_close(distance, nodes.new_tensor((0., .0001, .0011, .0021)).double(),
                               atol=1e-7, rtol=0)
    assert runtime._pad_backface.tolist() == [True, False, False, False]


def test_pose_transform_and_full_vertex_bounds_match_frozen_oracle():
    m = _math()
    pose = np.array([0.137, -0.293, 0.619, 1, 2, 3, 4], dtype=float)
    pose[3:] /= np.linalg.norm(pose[3:])
    vertices = np.array([[0.001, -0.002, 0.003], [-0.005, 0.004, -0.007], [0.009, 0.011, -0.013]])
    matrix = m.pose_matrix(pose)
    np.testing.assert_array_equal(matrix, FROZEN['matrix'])
    np.testing.assert_array_equal(m.transform(vertices, matrix), FROZEN['world'])
    assert matrix.dtype == np.float64
    bad = pose.copy()
    bad[3:] *= 1 + 2.1e-06
    with pytest.raises(ValueError, match='unit'):
        m.pose_matrix(bad)
    with pytest.raises(ValueError, match='finite'):
        m.pose_matrix([float('nan')] * 7)

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_ordered_trilinear_and_contact_match_frozen_original(dtype):
    m = _math()
    (grid, points) = (_grid(), _sampling_points())
    (phi, grad, inside) = m.sample_grid(grid, points, dtype=dtype)
    expected = FROZEN[np.dtype(dtype).name]
    np.testing.assert_array_equal(phi, np.array(expected['phi'], dtype=dtype))
    np.testing.assert_array_equal(grad, np.array(expected['gradient'], dtype=dtype))
    np.testing.assert_array_equal(inside, expected['valid'])
    actual = m.sample_contact(grid, np.eye(4), points, radius=0.0050000002, dtype=dtype)
    assert actual == expected['contact']
    assert actual['native_contact_observed'] is False

def test_eager_fp64_sampler_matches_frozen_order_and_open_upper_face():
    _math()
    from dh116.physics.geometry import _sample_grid_float64
    grid = _grid()
    (phi, gradient, valid) = _sample_grid_float64(torch.tensor(grid['values']), torch.tensor(grid['origin']), torch.tensor(grid['spacing']), torch.tensor(_sampling_points()))
    np.testing.assert_array_equal(phi.numpy(), FROZEN['float64']['phi'])
    np.testing.assert_array_equal(gradient.numpy(), FROZEN['float64']['gradient'])
    assert valid.tolist() == FROZEN['float64']['valid']
    assert valid[0] and (not valid[3]) and (not valid[4])

@pytest.mark.parametrize('invalid', ['values', 'origin', 'spacing', 'points'])
def test_invalid_locked_sdf_never_becomes_a_valid_measurement(invalid):
    m = _math()
    (grid, points) = (_grid(), _sampling_points())
    if invalid == 'values':
        grid['values'][0, 0, 0] = np.nan
    elif invalid == 'origin':
        grid['origin'][0] = np.inf
    elif invalid == 'spacing':
        grid['spacing'] = 0.0
    else:
        points[0, 0] = np.nan
    with np.errstate(divide='ignore'):
        with pytest.raises(ValueError, match='finite'):
            m.sample_grid(grid, points)

def test_finite_box_sat_crossing_and_near_touching_match_frozen_oracle():
    m = _math()
    vertices = np.array([[x, y, z] for x in (-0.01, 0.01) for y in (-0.001, 0.001) for z in (-0.001, 0.001)])
    for (gap, expected) in zip((-2e-10, 0.0, 5e-11, 2e-10), FROZEN['box']):
        actual = m.box_relation(vertices, np.array([0.02 + gap, 0, 0]), [0.02, 0.04, 0.04])
        assert actual == expected
    crossing = m.box_relation(vertices, [0, 0, 0], [0.001, 0.02, 0.02])
    assert crossing['intersects']
    with pytest.raises(ValueError, match='positive'):
        m.box_relation(vertices, [0, 0, 0], [0, 1, 1])

@pytest.mark.parametrize('fault', ['zero', 'threshold', 'lattice_face', 'flat', 'outside'])
def test_sdf_threshold_lattice_and_invalid_rows_match_frozen_cpu_fallback(fault):
    _math()
    (grid, nodes) = _near_case(fault)
    owner = _capture_owner(grid)
    thresholds = (-0.00015,) if fault == 'threshold' else ()
    result = owner.capture(_poses(1), torch.tensor(nodes)[None], frame=fault, radius=0.0050000002, sdf_thresholds=thresholds)
    expected = FROZEN['near'][fault]
    assert result.sdf_fallback[0].tolist() == [fault != 'outside'] * 2
    for side in (0, 1):
        wanted = [np.nan if x is None else x for x in expected['node_gaps_m']]
        np.testing.assert_array_equal(result.sdf_node_gaps_m[0, side].numpy(), wanted)
        assert result.sdf_node_valid[0, side].nonzero().flatten().tolist() == expected['in_lattice_nodes']
        assert result.sdf_negative_nodes[0, side].nonzero().flatten().tolist() == expected['negative_gap_nodes']
        assert result.sdf_contact_nodes[0, side].nonzero().flatten().tolist() == expected['contact_predicate_nodes']
        assert result.sdf_minimum_node_ids[0, side].item() == (-1 if expected['minimum_node_id'] is None else expected['minimum_node_id'])
    if fault in ('flat', 'outside'):
        assert not result.sdf_valid.any() and (not result.source_complete.any())
    assert result.native_qualification is False

def test_table_one_micrometre_band_keeps_original_strict_clearance_predicate():
    _math()
    owner = _capture_owner()
    poses = _poses(3)
    poses[:, 0, 0, 2] = torch.tensor([0.3021 - 1e-06, 0.3021, 0.3021 + 1e-06], dtype=torch.float64)
    poses[:, 1, 0, 0] = 1
    result = owner.capture(poses, torch.full((3, 41, 3), 10.0, dtype=torch.float64), frame='table')
    assert result.table_fallback[:, 0].all()
    np.testing.assert_array_equal(result.table_clearance_m[:, 0].numpy(), FROZEN['table'])
    assert result.table_failure[:, 0].tolist() == [gap < 0.0021 for gap in FROZEN['table']]

def test_negative_xz_bound_still_uses_full_polyline_certificate():
    m = _math()
    from dh116.physics.geometry import certify as certify
    from dh116.physics.geometry import validate
    from dh116.physics.geometry import TensorGeometry as TensorGeometry
    vertices = np.array([[x, y, z] for x in (-0.001, 0.001) for y in (-0.001, 0.001) for z in (-0.001, 0.001)])
    poses = _poses(1)
    poses[:, 0, 0, 1] = 0.1
    poses[:, 1, 0, 0] = 1
    nodes = np.c_[np.linspace(-0.02, 0.02, 41), np.zeros(41), np.full(41, 0.6)]
    world = m.transform(vertices, m.pose_matrix([0, 0.1, 0.6, 1, 0, 0, 0]))
    assert m.rod_support_lower(world, nodes) == FROZEN['coarse_support'] < 0
    packet = certify(world, nodes)
    assert validate(packet, world, nodes)
    owner = TensorGeometry([(s, 0, 'source:nonpad', vertices) for s in (0, 1)], device='cpu')
    result = owner.capture(poses, torch.tensor(nodes)[None], frame='certificate')
    assert result.support_fallback[0, 0]
    assert result.non_target_lower_m[0, 0].item() == packet['lower_m']
    assert packet['numeric_guard_m'] == 2e-06

def test_materialized_complete_inventory_is_owned_without_legacy_owner():
    from dh116.physics.geometry import TensorGeometry as TensorGeometry
    assert hasattr(TensorGeometry, 'from_materialized_geometry'), 'materialized source constructor is missing'
    (geometry, grids, sources, names) = _materialized()
    owner = TensorGeometry.from_materialized_geometry(geometry, grids, sources, names, required_pads=('body0', 'body1'), device='cpu')
    assert len(owner.columns) == 44 and len(owner.pad_columns) == 4
    assert list(owner.columns) == [(s, kind + ':' + name) for (s, side) in enumerate(('left', 'right')) for (kind, entries) in geometry[side].items() for name in entries]
    assert owner.source_bindings == {(s, name): sources[side]['sdf_binding'][name] for (s, side) in enumerate(('left', 'right')) for name in ('body0', 'body1')}
    assert owner.frame is owner.result is None
    assert owner._sdf is _math().sample_contact and owner._pose_matrix is _math().pose_matrix
    geometry['left']['source']['body0'][:] = 99
    grids['left']['body0']['values'][:] = 99
    sources['left']['sdf_binding']['body0']['sha256'] = 'changed'
    assert owner._shapes[0][3].max() < 1
    assert owner._pads[0][3]['values'].max() < 1
    assert owner.sources['left']['sdf_binding']['body0']['sha256'] != 'changed'

@pytest.mark.parametrize('fault', ['source_count', 'cooked_count', 'body_names', 'missing_grid', 'missing_binding', 'duplicate_pad'])
def test_materialized_inventory_refuses_incomplete_or_ambiguous_sources(fault):
    from dh116.physics.geometry import TensorGeometry as TensorGeometry
    assert hasattr(TensorGeometry, 'from_materialized_geometry'), 'materialized source constructor is missing'
    (geometry, grids, sources, names) = _materialized()
    pads = ('body0', 'body1')
    if fault == 'source_count':
        del geometry['left']['source']['body16']
    elif fault == 'cooked_count':
        del geometry['left']['cooked']['body4']
    elif fault == 'body_names':
        names = (names[0], names[0][:-1] + ('body0',))
    elif fault == 'missing_grid':
        del grids['left']['body0']
    elif fault == 'missing_binding':
        del sources['left']['sdf_binding']['body0']
    else:
        pads = ('body0', 'body0')
    with pytest.raises(ValueError):
        TensorGeometry.from_materialized_geometry(geometry, grids, sources, names, required_pads=pads, device='cpu')

@pytest.mark.parametrize('offset', [-2.1e-06, -1e-06, 1e-06, 2.1e-06])
def test_two_micrometre_routing_band_does_not_widen_actual_gap_predicate(offset):
    _math()
    grid = _grid()
    (x, _, _) = np.indices((5, 5, 5))
    grid['values'] = x * 0.01 - 0.02 + 0.005 - 0.006789 + offset
    nodes = np.c_[np.linspace(0.006789, 0.015789, 41), np.full(41, 0.013271), np.full(41, 0.611719)]
    owner = _capture_owner(grid)
    result = owner.capture(_poses(1), torch.tensor(nodes)[None], frame=('gap', offset))
    assert result.sdf_fallback[0].tolist() == [abs(offset) < 2e-06] * 2
    assert result.sdf_negative_nodes[0, :, 0].tolist() == [offset < 0] * 2
    if abs(offset) > 2e-06:
        np.testing.assert_allclose(result.sdf_minimum_gap_m[0].numpy(), [offset] * 2, atol=1e-16, rtol=0)

@pytest.mark.parametrize('face_offset', [-0.00011, -9e-05, 9e-05, 0.00011])
def test_lattice_margin_routes_only_original_cell_face_ambiguity(face_offset):
    _math()
    grid = _grid()
    (x, _, _) = np.indices((5, 5, 5))
    grid['values'] = 0.03 + x * 0.001
    nodes = np.c_[np.linspace(-0.003271, 0.013271, 41), np.full(41, 0.013271), np.full(41, 0.611719)]
    nodes[0, 0] = -0.01 + face_offset * 0.01
    result = _capture_owner(grid).capture(_poses(1), torch.tensor(nodes)[None], frame=('face', face_offset))
    assert result.sdf_fallback[0].tolist() == [abs(face_offset) < 0.0001] * 2
    assert result.sdf_node_valid.all()


def _tripod_box(low=(-.001, -.001, -.001), high=(.001, .001, .001)):
    return np.array([[x, y, z] for x in (low[0], high[0])
                     for y in (low[1], high[1]) for z in (low[2], high[2])])


def _tripod_guard(shapes, origins):
    from dh116.physics.geometry import TensorGeometry, TripodSupportGuard

    geometry = TensorGeometry(shapes, device='cpu')
    supports = [
        dict(name='support_0', center_m=[2., 0., .6], size_m=[.02, .02, .02]),
        dict(name='support_1', center_m=[-2., 0., .6], size_m=[.02, .02, .02]),
    ]
    return TripodSupportGuard(geometry, supports, origins)


def _tripod_inputs(origins):
    poses = _poses(len(origins))
    poses[:, 0, :, 0], poses[:, 1, :, 0] = -1., 1.
    poses[..., :3] += origins[:, None, None]
    nodes = torch.zeros(len(origins), 41, 3, dtype=torch.float64)
    nodes[..., 1] = torch.linspace(-.02, .02, 41, dtype=torch.float64)
    nodes[..., 2] = .6
    return poses, nodes + origins[:, None]


def test_tripod_raw_source_cooked_and_non_target_thresholds():
    origins = torch.zeros(4, 3, dtype=torch.float64)
    guard = _tripod_guard([
        (0, 0, 'source:finger33_link', _tripod_box()),
        (0, 1, 'cooked:finger33_link', _tripod_box()),
        (1, 0, 'source:finger43_link', _tripod_box()),
    ], origins)
    poses, nodes = _tripod_inputs(origins)
    poses[:, 0, 0, 0] = torch.tensor([.005849, .005851, .005851, .005851], dtype=torch.float64)
    poses[:, 0, 1, 0] = torch.tensor([.005851, .005849, .005851, .005851], dtype=torch.float64)
    poses[:, 1, 0, 0] = torch.tensor([.006251, .006251, .006249, .006251], dtype=torch.float64)
    failed = guard.capture(poses, nodes, frame=('cpu_fixture', 190))
    assert failed.dtype is torch.bool and failed.tolist() == [True, True, True, False]


def test_tripod_both_supports_and_raw_quaternion_survive_fallback():
    origins = torch.tensor([[0., 0., 0.], [5., 1., 0.], [-3., 2., 0.]], dtype=torch.float64)
    guard = _tripod_guard([
        (0, 0, 'source:finger33_link', _tripod_box()),
        (1, 0, 'cooked:finger43_link', _tripod_box()),
    ], origins)
    poses, nodes = _tripod_inputs(origins)
    poses[0, 0, 0, 0] = 2.
    poses[1, 1, 0, 0] = origins[1, 0] - 2.
    poses[2, 0, 0, 3:] = torch.tensor([0., 0., .7071074, .7071074], dtype=torch.float64)
    poses[2, 0, 0, 0] = origins[2, 0] + 2.011000002
    assert guard.capture(poses, nodes, frame=1).tolist() == [True, True, True]
    poses[..., 3:] /= poses[..., 3:].norm(dim=-1, keepdim=True)
    assert guard.capture(poses, nodes, frame=2).tolist() == [True, True, False]


def test_tripod_exact_fallback_certifies_the_full_bent_node_group():
    from scipy.spatial.transform import Rotation

    # The segment QPs alone cannot recover the closest support plane of the
    # triangle formed by the three nodes. The full group reaches x=0 before
    # rotation, so the analytic clearances are .004853 and .004840 m.
    origins = torch.zeros(2, 3, dtype=torch.float64)
    rotation = Rotation.from_euler('z', 1., degrees=True).as_matrix()
    vertices = _tripod_box((.004853, -.0001, -.001), (.006853, .0001, .001)) @ rotation.T
    guard = _tripod_guard([
        (0, 0, 'source:finger33_link', vertices),
        (1, 0, 'source:finger43_link', _tripod_box()),
    ], origins)
    poses, _ = _tripod_inputs(origins)
    poses[:, 0, 0, :2] = 0.
    poses[1, 0, 0, :3] -= torch.tensor(.000013 * rotation[:, 0])
    line = np.c_[np.zeros(41), np.linspace(0., .4, 41), np.zeros(41)]
    line[:3] = [[0., -.001, 0.], [-.001, 0., 0.], [0., .001, 0.]]
    nodes = torch.from_numpy(line @ rotation.T).expand(2, -1, -1).clone()
    nodes[..., 2] += .6
    assert guard.capture(poses, nodes, frame=1).tolist() == [False, True]


def test_tripod_clear_rows_use_device_geometry_and_invalidate_failed_capture(monkeypatch):
    origins = torch.zeros(1, 3, dtype=torch.float64)
    guard = _tripod_guard([
        (0, 0, 'source:finger33_link', _tripod_box()),
        (1, 0, 'source:finger43_link', _tripod_box()),
    ], origins)
    poses, nodes = _tripod_inputs(origins)

    def reject_host_copy(*args, **kwargs):
        pytest.fail('ordinary Tripod rows transferred to CPU fallback')

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'cpu', reject_host_copy)
        patch.setattr(torch.Tensor, 'numpy', reject_host_copy)
        result = guard.capture(poses, nodes, frame=('cpu_fixture', 1))
    assert result.tolist() == [False]
    assert torch.equal(guard.require_ready(frame=('cpu_fixture', 1)), result)
    with pytest.raises(ValueError, match='frame'):
        guard.require_ready(frame=('cpu_fixture', 2))
    with pytest.raises(ValueError, match='finite'):
        guard.capture(poses, nodes * torch.nan, frame=('cpu_fixture', 2))
    with pytest.raises(ValueError, match='frame'):
        guard.require_ready(frame=('cpu_fixture', 1))


FROZEN = {'matrix': [[-0.6666666666666669, 0.13333333333333336, 0.7333333333333335, 0.137], [0.6666666666666667, -0.33333333333333337, 0.6666666666666667, -0.293], [0.33333333333333337, 0.9333333333333336, 0.1333333333333333, 0.619], [0.0, 0.0, 0.0, 1.0]], 'world': [[0.13826666666666668, -0.28966666666666663, 0.6178666666666667], [0.13573333333333334, -0.30233333333333334, 0.6201333333333333], [0.12293333333333334, -0.29933333333333334, 0.6305333333333333]], 'float32': {'phi': [-0.014999999664723873, -0.004698506090790033, 0.02863915078341961, float('nan'), float('nan'), 0.007799999788403511, 0.010073472745716572], 'gradient': [[1.0, 0.0, 0.0], [1.0000001192092896, 0.19033001363277435, 0.1313900649547577], [1.0, 0.11479991674423218, 0.22189989686012268], [float('nan'), float('nan'), float('nan')], [float('nan'), float('nan'), float('nan')], [1.0, 0.14000004529953003, 0.14000004529953003], [1.0, 0.04402998462319374, 0.11962996423244476]], 'valid': [True, True, True, False, False, True, True], 'contact': {'minimum_gap_m': -0.019999999552965164, 'minimum_node_id': 0, 'gradient_norm_at_minimum': 1.0, 'in_lattice_nodes': [0, 1, 2, 5, 6], 'negative_gap_nodes': [0, 1], 'contact_predicate_nodes': [0, 1], 'node_gaps_m': [-0.019999999552965164, -0.009698506444692612, 0.023639149963855743, None, None, 0.0027999994345009327, 0.0050734723918139935], 'local_node_at_minimum': [-0.019999999552965164, -0.019999999552965164, -0.019999999552965164], 'cpu_dtype': 'float32', 'native_contact_observed': False}}, 'float64': {'phi': [-0.015, -0.0046985059, 0.02863915, float('nan'), float('nan'), 0.007800000000000001, 0.010073472699999999], 'gradient': [[0.9999999999999999, 0.0, 0.0], [0.9999999999999999, 0.19032999999999997, 0.1313899999999998], [1.0000000000000002, 0.11480000000000032, 0.22189999999999988], [float('nan'), float('nan'), float('nan')], [float('nan'), float('nan'), float('nan')], [0.9999999999999996, 0.13999999999999985, 0.13999999999999985], [0.9999999999999993, 0.04402999999999995, 0.1196299999999999]], 'valid': [True, True, True, False, False, True, True], 'contact': {'minimum_gap_m': -0.0200000002, 'minimum_node_id': 0, 'gradient_norm_at_minimum': 0.9999999999999999, 'in_lattice_nodes': [0, 1, 2, 5, 6], 'negative_gap_nodes': [0, 1], 'contact_predicate_nodes': [0, 1], 'node_gaps_m': [-0.0200000002, -0.0096985061, 0.0236391498, None, None, 0.002799999800000001, 0.005073472499999999], 'local_node_at_minimum': [-0.02, -0.02, -0.02], 'cpu_dtype': 'float64', 'native_contact_observed': False}}, 'box': [{'intersects': True, 'max_separating_axis_gap_m': -1.9999999920083944e-10, 'top_clearance_m': -0.021, 'bounds_world_m': [[-0.01, -0.001, -0.001], [0.01, 0.001, 0.001]]}, {'intersects': True, 'max_separating_axis_gap_m': 0.0, 'top_clearance_m': -0.021, 'bounds_world_m': [[-0.01, -0.001, -0.001], [0.01, 0.001, 0.001]]}, {'intersects': True, 'max_separating_axis_gap_m': 5.00000006675716e-11, 'top_clearance_m': -0.021, 'bounds_world_m': [[-0.01, -0.001, -0.001], [0.01, 0.001, 0.001]]}, {'intersects': False, 'max_separating_axis_gap_m': 1.9999999920083944e-10, 'top_clearance_m': -0.021, 'bounds_world_m': [[-0.01, -0.001, -0.001], [0.01, 0.001, 0.001]]}], 'near': {'zero': {'minimum_gap_m': -3.259629011154175e-09, 'minimum_node_id': 0, 'gradient_norm_at_minimum': 1.002785563468933, 'in_lattice_nodes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], 'negative_gap_nodes': [0], 'contact_predicate_nodes': [0], 'node_gaps_m': [-3.259629011154175e-09, 0.030726855620741844, 0.027867088094353676, 0.030682550743222237, 0.03138137608766556, 0.028573663905262947, 0.029565172269940376, 0.0288101639598608, 0.024525219574570656, 0.031001107767224312, 0.026088831946253777, 0.032569289207458496, 0.03362388536334038, 0.028464024886488914, 0.029559502378106117, 0.0285926703363657, 0.03341834247112274, 0.030559683218598366, 0.027277899906039238, 0.029900474473834038, 0.027850205078721046, 0.024211958050727844, 0.029880007728934288, 0.0293260607868433, 0.027829410508275032, 0.029318178072571754, 0.03252110257744789, 0.031077397987246513, 0.027450433000922203, 0.025908946990966797, 0.028800858184695244, 0.025781093165278435, 0.026995519176125526, 0.03518608212471008, 0.024357523769140244, 0.0331207700073719, 0.027745341882109642, 0.029553858563303947, 0.03420774266123772, 0.027047762647271156, 0.029308883473277092], 'local_node_at_minimum': [-0.013271000236272812, -0.013271000236272812, -0.011719048023223877], 'cpu_dtype': 'float32', 'native_contact_observed': False}, 'threshold': {'minimum_gap_m': -0.00015000347048044205, 'minimum_node_id': 0, 'gradient_norm_at_minimum': 1.002785563468933, 'in_lattice_nodes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], 'negative_gap_nodes': [0], 'contact_predicate_nodes': [0], 'node_gaps_m': [-0.00015000347048044205, 0.030576853081583977, 0.027717089280486107, 0.03053254820406437, 0.03123137168586254, 0.02842366136610508, 0.029415173456072807, 0.028660161420702934, 0.024375220760703087, 0.030851105228066444, 0.025938833132386208, 0.03241929039359093, 0.03347388282418251, 0.028314026072621346, 0.02940949983894825, 0.02844267152249813, 0.033268339931964874, 0.030409684404730797, 0.02712790109217167, 0.02975047565996647, 0.027700206264853477, 0.024061957374215126, 0.02973000891506672, 0.029176058247685432, 0.027679407969117165, 0.029168179258704185, 0.032371100038290024, 0.030927399173378944, 0.027300434187054634, 0.025758948177099228, 0.028650859370827675, 0.025631090626120567, 0.02684551663696766, 0.035036079585552216, 0.024207523092627525, 0.032970767468214035, 0.027595339342951775, 0.02940385602414608, 0.03405774012207985, 0.02689776010811329, 0.029158884659409523], 'local_node_at_minimum': [-0.013271000236272812, -0.013271000236272812, -0.011719048023223877], 'cpu_dtype': 'float32', 'native_contact_observed': False}, 'lattice_face': {'minimum_gap_m': -0.0196098443120718, 'minimum_node_id': 0, 'gradient_norm_at_minimum': 1.002785563468933, 'in_lattice_nodes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], 'negative_gap_nodes': [0], 'contact_predicate_nodes': [0], 'node_gaps_m': [-0.0196098443120718, 0.017845915630459785, 0.014986144378781319, 0.017801610752940178, 0.0185004323720932, 0.01569272391498089, 0.016684232279658318, 0.015929222106933594, 0.011644279584288597, 0.018120165914297104, 0.013207891955971718, 0.01968834549188614, 0.02074294164776802, 0.015583084896206856, 0.01667855866253376, 0.01571173407137394, 0.020537396892905235, 0.017678741365671158, 0.014396963641047478, 0.01701953634619713, 0.014969266951084137, 0.011331018060445786, 0.01699906773865223, 0.01644512079656124, 0.014948468655347824, 0.016437239944934845, 0.019640158861875534, 0.018196457996964455, 0.014569493010640144, 0.013028007000684738, 0.015919916331768036, 0.012900151312351227, 0.014114579185843468, 0.022305140271782875, 0.011476581916213036, 0.020239826291799545, 0.014864398166537285, 0.01667291671037674, 0.02132679708302021, 0.014166820794343948, 0.016427945345640182], 'local_node_at_minimum': [-0.019999900832772255, -0.013271000236272812, -0.011719048023223877], 'cpu_dtype': 'float32', 'native_contact_observed': False}, 'flat': {'minimum_gap_m': -0.005000000353902578, 'minimum_node_id': 0, 'gradient_norm_at_minimum': 0.0, 'in_lattice_nodes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], 'negative_gap_nodes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], 'contact_predicate_nodes': [], 'node_gaps_m': [-0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578, -0.005000000353902578], 'local_node_at_minimum': [-0.013271000236272812, -0.013271000236272812, -0.011719048023223877], 'cpu_dtype': 'float32', 'native_contact_observed': False}, 'outside': {'minimum_gap_m': None, 'minimum_node_id': None, 'gradient_norm_at_minimum': None, 'in_lattice_nodes': [], 'negative_gap_nodes': [], 'contact_predicate_nodes': [], 'node_gaps_m': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'local_node_at_minimum': None, 'cpu_dtype': 'float32', 'native_contact_observed': False}}, 'table': [0.0020990000000000175, 0.0020999999999999908, 0.002100999999999964], 'coarse_support': -0.005002}
