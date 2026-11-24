"""Pure CUDA measurement parity on the current local build; no SDK admission."""
from dataclasses import fields
from pathlib import Path
import numpy as np
import pytest
import torch
from dh116.physics.geometry import pose_matrix
from dh116.physics.geometry import sample_contact as sample_contact
from dh116.physics.geometry import transform
from dh116.physics.geometry import CudaGeometryProfile
from dh116.physics.geometry import CudaMeasurementGeometry
from dh116.physics.geometry import TensorGeometry as TensorGeometry

@pytest.fixture(autouse=True)
def forbid_runtime_graph_compilation(monkeypatch):
    calls = []

    def reject(*args, **kwargs):
        calls.append(True)
        raise AssertionError('measurement capture must not invoke torch.compile')
    monkeypatch.setattr(torch, 'compile', reject)
    yield
    assert calls == []

@pytest.fixture(scope='module')
def device():
    if not torch.cuda.is_available():
        pytest.skip('requires an explicitly selected CUDA tensor device')
    return torch.device('cuda', torch.cuda.current_device())

@pytest.fixture(scope='module')
def profile(device):
    result = CudaGeometryProfile.load()
    root = Path(__file__).resolve().parents[1]
    assert result.binary_path == root / 'build/cuda_geometry-current/libdh116_hull_support.so'
    assert torch.cuda.get_device_capability(device) == (8, 9)
    return result

def _pair(shapes, pads, device, profile):
    return (TensorGeometry(shapes, pads, device=device, backend='eager'), CudaMeasurementGeometry(shapes, pads, device=device, profile=profile, backend='eager'))

def _assert_same_frame(eager, cuda):
    for field in fields(eager):
        (expected, actual) = (getattr(eager, field.name), getattr(cuda, field.name))
        if isinstance(expected, torch.Tensor):
            if expected.dtype.is_floating_point:
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12, equal_nan=True, msg=lambda message: field.name + ': ' + message)
            else:
                assert torch.equal(actual, expected), field.name
        else:
            assert actual == expected, field.name

def _poses(count, device):
    result = torch.zeros((count, 2, 23, 7), dtype=torch.float64, device=device)
    result[..., 6] = 1
    result[..., 2] = 0.6
    return result

def _grid():
    (x, _, _) = np.indices((5, 5, 5))
    return dict(values=x * 0.01 - 0.02 + 0.005 - 0.006789, origin=np.full(3, -0.02), spacing=np.array(0.01))

def _pad_pair(grid, device, profile):
    shapes = [(side, 0, 'cooked:pad', np.zeros((1, 3))) for side in (0, 1)]
    pads = [(side, 0, 'pad', grid) for side in (0, 1)]
    return _pair(shapes, pads, device, profile)

def test_all_vertices_and_three_physical_geometry_guards_match_eager(device, profile):
    vertices = np.zeros((129, 3), dtype=np.float64)
    vertices[:8] = [[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (-0.01, 0.01)]
    vertices[-1] = [0.02, 0.03, -0.025]
    shapes = [(side, 0, 'source:hull', vertices) for side in (0, 1)]
    (eager, cuda) = _pair(shapes, (), device, profile)
    poses = _poses(4, device)
    poses[:, 1, :, 0] = 1
    poses[0, :, 0, 3:] = torch.tensor([2.0, 3.0, 4.0, 1.0], dtype=torch.float64, device=device) / np.sqrt(30.0)
    poses[1, 0, 0, 2] = 0.3 + 0.0021 + 0.025 - 1e-06
    poses[3, 1, 0, 0] = 0.03 - 1e-06
    nodes = torch.full((4, 41, 3), 10.0, dtype=torch.float64, device=device)
    nodes[2, :, 0] = torch.linspace(-0.05, 0.05, 41, dtype=torch.float64, device=device)
    nodes[2, :, 1] = 0
    nodes[2, :, 2] = 0.6
    expected = eager.capture(poses, nodes, frame='hull-guards')
    actual = cuda.capture(poses, nodes, frame='hull-guards')
    _assert_same_frame(expected, actual)
    torch.testing.assert_close(actual.minimum[1:], poses[1:, :, 0, :3] + torch.tensor(vertices.min(0), device=device), rtol=0, atol=0)
    torch.testing.assert_close(actual.maximum[1:], poses[1:, :, 0, :3] + torch.tensor(vertices.max(0), device=device), rtol=0, atol=0)
    for side in (0, 1):
        pose = poses[0, side, 0].cpu().numpy()
        world = transform(vertices, pose_matrix(pose[[0, 1, 2, 6, 3, 4, 5]]))
        np.testing.assert_allclose(actual.minimum[0, side].cpu().numpy(), world.min(0), rtol=0, atol=1e-15)
        np.testing.assert_allclose(actual.maximum[0, side].cpu().numpy(), world.max(0), rtol=0, atol=1e-15)
    assert actual.table_failure.tolist() == [[False, False], [True, False], [False, False], [False, False]]
    assert actual.non_target_failure.tolist() == [[False, False], [False, False], [True, False], [False, False]]
    assert actual.inter_hand_failure.tolist() == [False, False, False, True]
    assert actual.table_fallback[1, 0] and actual.support_fallback[2, 0] and actual.separation_fallback[3]
    assert actual.geometry_passed.tolist() == [True, False, False, False]
    assert actual.native_qualification is False

@pytest.mark.parametrize('boundary,offset', [('gap', -2.1e-06), ('gap', -1e-06), ('gap', 1e-06), ('gap', 2.1e-06), ('voxel', -0.00011), ('voxel', -9e-05), ('voxel', 9e-05), ('voxel', 0.00011)])
def test_sdf_decisions_and_sparse_cpu_routing_match_original_oracle(device, profile, boundary, offset):
    grid = _grid()
    nodes = np.c_[np.linspace(0.006789, 0.015789, 41), np.full(41, 0.013271), np.full(41, 0.611719)]
    if boundary == 'gap':
        grid['values'] += offset
        routed = abs(offset) < 2e-06
    else:
        (x, _, _) = np.indices((5, 5, 5))
        grid['values'] = 0.03 + x * 0.001
        nodes[:, 0] = np.linspace(-0.003271, 0.013271, 41)
        nodes[0, 0] = -0.01 + offset * 0.01
        routed = abs(offset) < 0.0001
    (eager, cuda) = _pad_pair(grid, device, profile)
    poses = _poses(1, device)
    material = torch.tensor(nodes, dtype=torch.float64, device=device)[None]
    expected = eager.capture(poses, material, frame=(boundary, offset))
    actual = cuda.capture(poses, material, frame=(boundary, offset))
    _assert_same_frame(expected, actual)
    assert actual.sdf_fallback.tolist() == [[routed, routed]]
    assert actual.sdf_node_valid.all()
    if boundary == 'gap':
        assert actual.sdf_negative_nodes[0, :, 0].tolist() == [offset < 0] * 2
        assert actual.sdf_contact_nodes[0, :, 0].tolist() == [offset < 0] * 2
    if routed:
        pose = np.eye(4)
        pose[2, 3] = 0.6
        oracle = sample_contact(grid, pose, nodes)
        for side in (0, 1):
            np.testing.assert_array_equal(actual.sdf_node_gaps_m[0, side].cpu().numpy(), oracle['node_gaps_m'])
            assert actual.sdf_minimum_node_ids[0, side].item() == oracle['minimum_node_id']
            assert actual.sdf_contact_nodes[0, side].nonzero().flatten().tolist() == oracle['contact_predicate_nodes']
    elif boundary == 'gap':
        torch.testing.assert_close(actual.sdf_minimum_gap_m, torch.full((1, 2), offset, dtype=torch.float64, device=device), rtol=0, atol=1e-16)

@pytest.mark.parametrize('invalid', ['pose', 'nodes', 'radius'])
def test_nan_inputs_reject_measurement_and_invalidate_prior_epoch(device, profile, invalid):
    (eager, cuda) = _pad_pair(_grid(), device, profile)
    poses = _poses(1, device)
    nodes = torch.tensor(np.c_[np.linspace(0.006789, 0.015789, 41), np.full(41, 0.013271), np.full(41, 0.611719)], dtype=torch.float64, device=device)[None]
    for owner in (eager, cuda):
        owner.capture(poses, nodes, frame='valid')
        (bad_pose, bad_nodes) = (poses.clone(), nodes.clone())
        radius = torch.tensor([0.005], dtype=torch.float64, device=device)
        if invalid == 'pose':
            bad_pose[0, 0, 0, 6] = torch.nan
        elif invalid == 'nodes':
            bad_nodes[0, 0, 0] = torch.nan
        else:
            radius[0] = torch.nan
        with pytest.raises(ValueError, match='complete finite'):
            owner.capture(bad_pose, bad_nodes, frame='invalid', radius=radius)
        with pytest.raises(ValueError, match='unavailable|changed'):
            owner.require_ready(frame='valid')
        assert owner.frame is owner.result is None

def test_compiled_runtime_backend_is_refused_before_any_graph_call():
    with pytest.raises(ValueError, match='compiled.*disabled'):
        TensorGeometry([(s, 0, 'source:hull', np.zeros((1, 3))) for s in (0, 1)], device='cpu', backend='compiled')
