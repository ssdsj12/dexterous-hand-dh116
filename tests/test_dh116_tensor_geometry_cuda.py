"""Operator-enabled pure Torch CUDA parity; never launches Isaac or an SDK.

Run only after reserving physical GPU 3:
CUDA_VISIBLE_DEVICES=3 DH116_TENSOR_GEOMETRY_CUDA=1 ... pytest <this file>
Without that explicit opt-in, even CUDA availability is not queried.
"""
from dataclasses import fields
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import required_sdf_valid, rod_support_lower
from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import certify, validate
from go2_pvcnn.control.dh116.tabletop_tensor_geometry import TensorGeometry
from scripts.audit_dh116_pinch_initial_pose import pose_matrix, transform
from scripts.prepare_dh116_pinch_closure_margin import sample_contact


@pytest.fixture(scope="module")
def cuda_device():
    if os.environ.get("DH116_TENSOR_GEOMETRY_CUDA") != "1":
        pytest.skip("operator must explicitly enable DH116_TENSOR_GEOMETRY_CUDA=1")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3", "operator must reserve physical GPU 3"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.device("cuda:0")
    print("pure_torch_geometry_cuda:", torch.cuda.get_device_properties(device))
    yield device
    torch.cuda.synchronize(device)


@pytest.fixture(scope="module")
def original_inventory(cuda_device):
    # This is the existing CPU USD/archived-source loader. It creates no Isaac
    # app, SDK object or simulation. Missing assets are a failure of this explicit
    # operator run, not a silently skipped piece of the claimed CUDA proof.
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import _canonical_pinch_history
    from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_pinch_distance import PolicyOwnedPinchGeometry
    from test_dh116_measured_closure_geometry import case
    with _canonical_pinch_history():
        owner = PolicyOwnedPinchGeometry(Path(__file__).resolve().parents[1], scope="cpu_fixture")
    _, _, _, _, hands, _ = case.__wrapped__(owner)
    names = tuple(tuple(hand.body_names) for hand in hands)
    shapes = [(s, names[s].index(name), kind+":"+name, vertices)
        for s, side in enumerate(("left", "right"))
        for kind, entries in owner._geometry[side].items() for name, vertices in entries.items()]
    assert len(shapes) == 44
    return owner, names, shapes


def identity_poses(n):
    poses = np.zeros((n, 2, 23, 7), dtype=np.float64)
    poses[..., 6] = 1
    poses[..., 2] = .6
    return poses


def original_world(vertices, pose):
    return transform(vertices, pose_matrix(pose[[0, 1, 2, 6, 3, 4, 5]]))


def original_pad(grid, pose, nodes, *, radius=.005, dtype=np.float32):
    return sample_contact(grid, pose_matrix(pose[[0, 1, 2, 6, 3, 4, 5]]), nodes,
                          radius=radius, dtype=dtype)


def cpu(value):
    return value.detach().cpu().numpy()


def assert_device_result(result, device):
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, torch.Tensor):
            assert value.device == device, field.name
    assert result.ready and not result.native_qualification
    assert not result.tripod_supported_guard_covered


def forbid_cpu_copy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary CUDA geometry copied a row to CPU/NumPy")
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)


def test_cuda_n128_all_original_44_hulls_safe_path_and_original_numpy_oracle(
        cuda_device, original_inventory, monkeypatch):
    original, names, shapes = original_inventory
    rng = np.random.default_rng(516)
    n = 128
    poses = identity_poses(n)
    poses[..., :3] = rng.uniform(-.005, .005, (n, 2, 23, 3))+[0, 0, 1.]
    poses[:, 0, :, 0] -= 2
    poses[:, 1, :, 0] += 2
    # Also exercise stable failing masks away from fallback boundaries.
    poses[0, :, :, 2] = .1
    poses[1, :, :, 0] = 1.5
    poses[..., 3:] = Rotation.random(n*2*23, random_state=rng).as_quat().reshape(n, 2, 23, 4)*(1+1e-7)
    nodes = rng.uniform(-.01, .01, (n, 41, 3))+[0, 0, .6]
    radii = rng.uniform(.004, .006, n)
    # Independent original NumPy/SciPy equations over every original vertex,
    # not a CPU execution of the tensor implementation under test.
    minimum = np.empty((n, 44, 3))
    maximum = np.empty_like(minimum)
    lower = np.empty((n, 44))
    separation = np.empty(n)
    non_target = np.array([key.split(":", 1)[1] not in original.REQUIRED_PADS for _, _, key, _ in shapes])
    for lane in range(n):
        whole = [[], []]
        for j, (side, link, _, vertices) in enumerate(shapes):
            world = original_world(vertices, poses[lane, side, link])
            minimum[lane, j], maximum[lane, j] = world.min(0), world.max(0)
            lower[lane, j] = rod_support_lower(world, nodes[lane])+.005-radii[lane]
            whole[side].append(world)
        left, right = [np.concatenate(v) for v in whole]
        separation[lane] = max((right.min(0)-left.max(0)).max(), (left.min(0)-right.max(0)).max())
    table = minimum[..., 2]-.3
    table_failure = table < CLEARANCE_M
    non_target_failure = (lower < 0) & non_target
    inter_hand_failure = separation < 0
    assert table_failure.any() and inter_hand_failure.any()
    assert not non_target_failure.any()
    geometry = TensorGeometry.from_policy_geometry(original, names, device=cuda_device)
    gpu_poses = torch.tensor(poses, device=cuda_device)
    gpu_nodes = torch.tensor(nodes, device=cuda_device)
    gpu_radii = torch.tensor(radii, device=cuda_device)
    with monkeypatch.context() as m:
        forbid_cpu_copy(m)
        actual = geometry.capture(gpu_poses, gpu_nodes, frame=("cuda-full44", 1), radius=gpu_radii)
    torch.cuda.synchronize(cuda_device)
    assert_device_result(actual, cuda_device)
    assert geometry.vertex_counts == tuple(len(v) for _, _, _, v in shapes)
    assert not bool(actual.fallback_rows.any())
    errors = []
    for actual_tensor, expected in ((actual.minimum, minimum), (actual.maximum, maximum),
            (actual.table_clearance_m, table), (actual.non_target_lower_m, lower),
            (actual.hand_separation_lower_m, separation)):
        result = cpu(actual_tensor)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-10)
        errors.append(float(np.abs(result-expected).max()))
    np.testing.assert_array_equal(cpu(actual.table_failure), table_failure)
    np.testing.assert_array_equal(cpu(actual.non_target_failure), non_target_failure)
    np.testing.assert_array_equal(cpu(actual.inter_hand_failure), inter_hand_failure)
    # All pads are deliberately away from this synthetic rod. Keep unavailable
    # SDF coverage explicit rather than treating these rows as a physical pass.
    assert not bool(actual.sdf_node_valid.any())
    assert not bool(actual.source_complete.any())
    assert not bool(actual.geometry_passed.any())
    print(f"N128 actual44: vertices={sum(geometry.vertex_counts)}, max_abs_bound_error={max(errors):.3g}, fallback_rows=0")


def test_cuda_sdf_float64_and_two_lane_sparse_original_fallback(cuda_device, monkeypatch):
    rng = np.random.default_rng(29)
    x, y, z = np.indices((9, 9, 9))
    grid = dict(values=.03+.001*x+.002*y+.003*z+.0001*x*y,
                origin=np.full(3, -.04), spacing=np.array(.01))
    vertices = np.array([[x, y, z] for x in (-.001, .001)
                         for y in (-.001, .001) for z in (-.001, .001)])
    shapes = [(s, link, key, vertices) for s in (0, 1)
              for link, key in ((0, "source:nonpad"), (3, "cooked:pad"))]
    geometry = TensorGeometry(shapes, [(s, 3, "pad", grid) for s in (0, 1)], device=cuda_device)
    poses = identity_poses(2)
    poses[:, 0, 0, 0], poses[:, 1, 0, 0] = -1, 1
    translation = np.array([.4, -.2, .6])
    rotations = Rotation.from_euler("xyz", [[.21, -.14, .09], [-.13, .18, -.23]])
    poses[:, :, 3, :3] = translation
    poses[:, :, 3, 3:] = rotations.as_quat()[:, None]
    local = rng.uniform(-.025, .025, (2, 41, 3))
    nodes = local @ rotations.as_matrix().transpose(0, 2, 1)+translation
    with monkeypatch.context() as m:
        forbid_cpu_copy(m)
        safe = geometry.capture(torch.tensor(poses, device=cuda_device), torch.tensor(nodes, device=cuda_device),
                                frame=("cuda-sdf", 1))
    assert_device_result(safe, cuda_device)
    assert not bool(safe.fallback_rows.any())
    assert bool(safe.source_complete.all())
    max_error = 0.
    for lane in range(2):
        for side in (0, 1):
            expected64 = original_pad(grid, poses[lane, side, 3], nodes[lane], dtype=np.float64)
            expected32 = original_pad(grid, poses[lane, side, 3], nodes[lane])
            values = cpu(safe.sdf_node_gaps_m)[lane, side]
            np.testing.assert_allclose(values, expected64["node_gaps_m"], rtol=0, atol=1e-10)
            max_error = max(max_error, float(np.abs(values-np.asarray(expected64["node_gaps_m"])).max()))
            assert bool(cpu(safe.sdf_valid)[lane, side]) == required_sdf_valid(expected32)
            assert int(cpu(safe.sdf_minimum_node_ids)[lane, side]) == expected64["minimum_node_id"]
            assert np.flatnonzero(cpu(safe.sdf_negative_nodes)[lane, side]).tolist() == expected32["negative_gap_nodes"]
            assert np.flatnonzero(cpu(safe.sdf_contact_nodes)[lane, side]).tolist() == expected32["contact_predicate_nodes"]
    # Only lane 1 changes. Its non-target negative coarse bound needs the exact
    # certificate, and its pad lattice face needs original FP32 sampling.
    local[1, 0] = [-.04, .00123, .00123]
    nodes[1] = local[1] @ rotations.as_matrix()[1].T+translation
    poses[1, 0, 0, :3] = translation+[0, .1, 0]
    copied_source_shapes = []
    tensor_cpu = torch.Tensor.cpu
    def tracked_cpu(tensor, *args, **kwargs):
        if tensor.is_cuda and tensor.ndim >= 3:
            copied_source_shapes.append(tuple(tensor.shape))
        return tensor_cpu(tensor, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "cpu", tracked_cpu)
        sparse = geometry.capture(torch.tensor(poses, device=cuda_device), torch.tensor(nodes, device=cuda_device),
                                  frame=("cuda-sdf", 2))
    assert_device_result(sparse, cuda_device)
    np.testing.assert_array_equal(cpu(sparse.fallback_rows), [False, True])
    assert copied_source_shapes and all(shape[0] == 1 for shape in copied_source_shapes)
    assert geometry.require_ready(frame=("cuda-sdf", 2)) is sparse
    world = original_world(vertices, poses[1, 0, 0])
    assert rod_support_lower(world, nodes[1]) < 0
    certificate = certify(world, nodes[1])
    assert validate(certificate, world, nodes[1])
    assert certificate["lower_m"] > 0
    assert cpu(sparse.non_target_lower_m)[1, 0] == certificate["lower_m"]
    for side in (0, 1):
        expected = original_pad(grid, poses[1, side, 3], nodes[1])
        np.testing.assert_allclose(cpu(sparse.sdf_node_gaps_m)[1, side],
            [np.nan if v is None else v for v in expected["node_gaps_m"]], rtol=0, atol=0, equal_nan=True)
        assert bool(cpu(sparse.sdf_valid)[1, side]) == required_sdf_valid(expected)
        assert int(cpu(sparse.sdf_minimum_node_ids)[1, side]) == expected["minimum_node_id"]
        assert np.flatnonzero(cpu(sparse.sdf_node_valid)[1, side]).tolist() == expected["in_lattice_nodes"]
        assert np.flatnonzero(cpu(sparse.sdf_negative_nodes)[1, side]).tolist() == expected["negative_gap_nodes"]
        assert np.flatnonzero(cpu(sparse.sdf_contact_nodes)[1, side]).tolist() == expected["contact_predicate_nodes"]
    print(f"N2 rotated SDF: max_abs_gap_error={max_error:.3g}; sparse source copies={copied_source_shapes}")


def test_cuda_large_world_origin_original_float32_sign_flip_is_sparse(cuda_device):
    x, _, _ = np.indices((9, 9, 9))
    grid = dict(values=x*.01-.04+.00499, origin=np.full(3, -.04), spacing=np.array(.01))
    vertices = np.array([[x, y, z] for x in (-.001, .001)
                         for y in (-.001, .001) for z in (-.001, .001)])
    geometry = TensorGeometry([(s, 0, "cooked:pad", vertices) for s in (0, 1)],
        [(s, 0, "pad", grid) for s in (0, 1)], device=cuda_device)
    poses = identity_poses(2)
    poses[1, :, :, 0] = 1000.
    local = np.random.default_rng(271).uniform(.005, .02, (2, 41, 3))
    local[:, 0] = [.00003, .0123, .0234]
    nodes = local+np.array([[0, 0, .6], [1000, 0, .6]])[:, None]
    expected32 = [original_pad(grid, poses[i, 0, 0], nodes[i]) for i in range(2)]
    expected64 = [original_pad(grid, poses[i, 0, 0], nodes[i], dtype=np.float64) for i in range(2)]
    assert expected32[0]["minimum_gap_m"] > 0
    assert expected32[1]["minimum_gap_m"] < 0 < expected64[1]["minimum_gap_m"]
    assert expected64[1]["minimum_gap_m"] > 2e-6
    actual = geometry.capture(torch.tensor(poses, device=cuda_device), torch.tensor(nodes, device=cuda_device),
                              frame=("cuda-world-origin", 1))
    assert_device_result(actual, cuda_device)
    np.testing.assert_array_equal(cpu(actual.fallback_rows), [False, True])
    np.testing.assert_array_equal(cpu(actual.sdf_fallback), [[False, False], [True, True]])
    for lane in range(2):
        for side in (0, 1):
            expected = expected64[lane] if lane == 0 else expected32[lane]
            assert cpu(actual.sdf_minimum_gap_m)[lane, side] == pytest.approx(expected["minimum_gap_m"], abs=1e-10)
            assert bool(cpu(actual.sdf_negative_nodes)[lane, side, 0]) == (expected32[lane]["minimum_gap_m"] < 0)
            assert bool(cpu(actual.sdf_valid)[lane, side]) == required_sdf_valid(expected32[lane])
    assert bool(actual.source_complete.all())
    print("N2 large-world-origin: original FP32 sign flip routed synchronously; fallback_rows=[False, True]")
