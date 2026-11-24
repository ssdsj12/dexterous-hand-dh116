"""Tensor numerical/source parity only; no Isaac or native qualification."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_pinch_initial_pose import pose_matrix, transform
from scripts.prepare_dh116_pinch_closure_margin import sample_contact
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import rod_support_lower, required_sdf_valid
from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import certify, validate
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M


def tensor_owner(shapes, pads=(), **kwargs):
    from go2_pvcnn.control.dh116.tabletop_tensor_geometry import TensorGeometry
    return TensorGeometry(shapes, pads, device="cpu", **kwargs)


def poses_for(n):
    p = torch.zeros((n, 2, 23, 7), dtype=torch.float64)
    p[..., 6] = 1
    p[..., 2] = .6
    return p


def world_at(vertices, pose):
    p = np.asarray(pose)
    return transform(vertices, pose_matrix(p[[0, 1, 2, 6, 3, 4, 5]]))


def test_full_44_shapes_all_vertices_float64_bounds_and_no_cpu_copy(monkeypatch):
    rng = np.random.default_rng(938)
    shapes = [(s, i, "source:link"+str(i), rng.normal(size=(257+i, 3))*.002)
              for s in (0, 1) for i in range(22)]
    owner = tensor_owner(shapes)
    poses = poses_for(3)
    poses[:, 0, :, 0], poses[:, 1, :, 0] = -1, 1
    poses[..., 3:] = torch.tensor(Rotation.random(3*2*23, random_state=rng).as_quat().reshape(3, 2, 23, 4))
    nodes = torch.tensor(rng.uniform(-.01, .01, (3, 41, 3))+[0, 0, .6])
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary tensor geometry copied to CPU/NumPy")
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "cpu", forbidden)
        m.setattr(torch.Tensor, "numpy", forbidden)
        result = owner.capture(poses, nodes, frame=("epoch", 1), radius=torch.tensor([.005, .006, .004]))
    assert result.table_clearance_m.shape == result.non_target_lower_m.shape == (3, 44)
    assert result.minimum.dtype == torch.float64
    assert len(owner.vertex_counts) == 44 and sum(owner.vertex_counts) == sum(len(s[3]) for s in shapes)
    assert not result.fallback_rows.any()
    for lane in range(3):
        whole = [[], []]
        for j, (side, link, key, vertices) in enumerate(shapes):
            world = world_at(vertices, poses[lane, side, link])
            whole[side].append(world)
            np.testing.assert_allclose(result.minimum[lane, j], world.min(0), atol=5e-13, rtol=0)
            np.testing.assert_allclose(result.maximum[lane, j], world.max(0), atol=5e-13, rtol=0)
            radius = [.005, .006, .004][lane]
            expected = rod_support_lower(world, nodes[lane].numpy())+.005-radius
            np.testing.assert_allclose(result.non_target_lower_m[lane, j], expected, atol=5e-10, rtol=0)
            assert result.table_failure[lane, j] == (world[:, 2].min()-.3 < CLEARANCE_M)
        left, right = [np.concatenate(v) for v in whole]
        sep = max((right.min(0)-left.max(0)).max(), (left.min(0)-right.max(0)).max())
        assert float(result.hand_separation_lower_m[lane]) == pytest.approx(sep, abs=5e-13)


@pytest.mark.parametrize("y_offset", [.1, 0.])
def test_negative_coarse_support_is_certified_synchronously_and_only_bad_rows_copy(monkeypatch, y_offset):
    vertices = np.array([[x, y, z] for x in (-.001, .001) for y in (-.001, .001) for z in (-.001, .001)])
    owner = tensor_owner([(0, 0, "source:nonpad", vertices), (1, 0, "source:nonpad", vertices)])
    poses = poses_for(3)
    poses[:, 0, :, 0], poses[:, 1, :, 0] = -1, 1
    poses[1, 0, 0, :3] = torch.tensor([0, y_offset, .6])
    nodes = torch.tensor(np.tile(np.c_[np.linspace(-.02, .02, 41), np.zeros(41), np.full(41, .6)], (3, 1, 1)))
    copied = []
    original = torch.Tensor.cpu
    def tracked(t, *args, **kwargs):
        if t.ndim >= 3:
            copied.append(tuple(t.shape))
        return original(t, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "cpu", tracked)
        result = owner.capture(poses, nodes, frame={"sdk_step": 2, "reset_ids": (0, 1, 0)})
    world = world_at(vertices, poses[1, 0, 0])
    assert rod_support_lower(world, nodes[1].numpy()) < 0
    certificate = certify(world, nodes[1].numpy())
    assert validate(certificate, world, nodes[1].numpy())
    assert result.non_target_lower_m[1, 0] == certificate["lower_m"]
    assert bool(result.non_target_failure[1, 0]) == (certificate["lower_m"] < 0)
    assert bool(result.geometry_passed[1]) == (certificate["lower_m"] >= 0)
    assert result.fallback_rows.tolist() == [False, True, False]
    assert copied and all(shape[0] == 1 for shape in copied)
    assert result.ready and owner.require_ready(frame={"sdk_step": 2, "reset_ids": (0, 1, 0)}) is result


def test_table_and_hand_separation_thresholds_use_original_math():
    vertices = np.array([[0, 0, 0], [.01, .01, .01]])
    owner = tensor_owner([(0, 0, "source:pad", vertices), (1, 0, "source:pad", vertices)])
    poses = poses_for(3)
    poses[:, 0, 0, 2] = torch.tensor([.3+CLEARANCE_M-1e-12, .3+CLEARANCE_M, .3+CLEARANCE_M+1e-12])
    poses[:, 1, 0, 2] = poses[:, 0, 0, 2]
    poses[:, 1, 0, 0] = torch.tensor([.01-1e-12, .01, .01+1e-12])
    nodes = torch.full((3, 41, 3), 10., dtype=torch.float64)
    result = owner.capture(poses, nodes, frame=3)
    for lane in range(3):
        left, right = [world_at(vertices, poses[lane, side, 0]) for side in (0, 1)]
        gap = left[:, 2].min()-.3
        sep = max((right.min(0)-left.max(0)).max(), (left.min(0)-right.max(0)).max())
        assert result.table_clearance_m[lane, 0] == gap
        assert result.table_failure[lane, 0] == (gap < CLEARANCE_M)
        assert result.hand_separation_lower_m[lane] == sep
        assert result.inter_hand_failure[lane] == (sep < 0)
    assert result.fallback_rows.all()


def affine_grid():
    x, y, z = np.indices((9, 9, 9))
    return dict(values=.03+.001*x+.002*y+.003*z+.0001*x*y,
                origin=np.full(3, -.04), spacing=np.array(.01))


def test_sdf_float64_explicit_trilinear_matches_original_math_and_fp32_masks(monkeypatch):
    rng = np.random.default_rng(29)
    grid = affine_grid()
    vertices = rng.uniform(-.001, .001, (17, 3))
    shapes = [(0, 0, "cooked:pad", vertices), (1, 0, "cooked:pad", vertices)]
    bindings = {(s, "pad"): {"sha256": str(s)*64, "capture_sha256": "a"*64} for s in (0, 1)}
    owner = tensor_owner(shapes, [(s, 0, "pad", grid) for s in (0, 1)], bindings=bindings)
    poses = poses_for(4)
    nodes = torch.tensor(rng.uniform(-.025, .025, (4, 41, 3))+[0, 0, .6])
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "cpu", lambda *a, **k: pytest.fail("safe SDF frame copied to CPU"))
        m.setattr(torch.Tensor, "numpy", lambda *a, **k: pytest.fail("safe SDF frame copied to NumPy"))
        result = owner.capture(poses, nodes, frame=("sdf", 4), radius=.005)
    assert not result.sdf_fallback.any()
    for lane in range(4):
        for side in (0, 1):
            matrix = pose_matrix(poses[lane, side, 0].numpy()[[0, 1, 2, 6, 3, 4, 5]])
            expected64 = sample_contact(grid, matrix, nodes[lane].numpy(), dtype=np.float64)
            expected32 = sample_contact(grid, matrix, nodes[lane].numpy())
            np.testing.assert_allclose(result.sdf_node_gaps_m[lane, side], expected64["node_gaps_m"], atol=1e-12, rtol=0)
            assert result.sdf_minimum_node_ids[lane, side] == expected64["minimum_node_id"]
            assert result.sdf_valid[lane, side] == required_sdf_valid(expected32)
            assert result.sdf_negative_nodes[lane, side].nonzero().flatten().tolist() == expected32["negative_gap_nodes"]
            assert result.sdf_contact_nodes[lane, side].nonzero().flatten().tolist() == expected32["contact_predicate_nodes"]
    assert torch.allclose(result.positive_pad_mean_m, result.sdf_minimum_gap_m.clamp_min(0))


@pytest.mark.parametrize("fault", ["gap_zero", "actual_threshold", "lattice_face", "zero_gradient", "minimum_tie"])
def test_sdf_boundary_falls_back_to_original_float32_predicates(fault):
    grid = affine_grid()
    nodes = torch.tensor(np.random.default_rng(8).uniform(-.02, .02, (1, 41, 3))+[0, 0, .6])
    thresholds = ()
    if fault in ("gap_zero", "actual_threshold"):
        nodes[0, 0] = torch.tensor([-.03123, -.03123, .56877])
        matrix = np.eye(4)
        matrix[2, 3] = .6
        initial = sample_contact(grid, matrix, nodes[0].numpy(), dtype=np.float64)["minimum_gap_m"]+.005
        grid["values"] -= initial-(.0050000001 if fault == "gap_zero" else .0048500001)
        if fault == "actual_threshold": thresholds = (-.00015,)
    if fault == "lattice_face": nodes[0, 0, :3] = torch.tensor([-.04, .00123, .60123])
    if fault == "zero_gradient": grid["values"] = 1e-46 + np.indices((9, 9, 9))[0]*1e-50
    if fault == "minimum_tie":
        nodes[0, :2] = torch.tensor([-.03123, -.03123, .56877])
    shapes = [(s, 0, "cooked:pad", np.zeros((1, 3))) for s in (0, 1)]
    owner = tensor_owner(shapes, [(s, 0, "pad", grid) for s in (0, 1)])
    poses = poses_for(1)
    if fault == "actual_threshold":
        assert not owner.capture(poses, nodes, frame="without_extra_threshold", radius=.0050000002).sdf_fallback.any()
    result = owner.capture(poses, nodes, frame=fault, radius=.0050000002, sdf_thresholds=thresholds)
    assert result.sdf_fallback.all()
    for side in (0, 1):
        expected = sample_contact(grid, pose_matrix(poses[0, side, 0].numpy()[[0, 1, 2, 6, 3, 4, 5]]), nodes[0].numpy(), radius=.0050000002)
        np.testing.assert_allclose(result.sdf_node_gaps_m[0, side], [np.nan if x is None else x for x in expected["node_gaps_m"]], atol=0, rtol=0, equal_nan=True)
        assert result.sdf_minimum_node_ids[0, side] == expected["minimum_node_id"]
        assert result.sdf_valid[0, side] == required_sdf_valid(expected)
        assert result.sdf_contact_nodes[0, side].nonzero().flatten().tolist() == expected["contact_predicate_nodes"]
        if fault == "zero_gradient":
            expected64 = sample_contact(grid, pose_matrix(poses[0, side, 0].numpy()[[0, 1, 2, 6, 3, 4, 5]]), nodes[0].numpy(), dtype=np.float64)
            assert required_sdf_valid(expected64) and not required_sdf_valid(expected)


def test_frame_bindings_ownership_and_failure_invalidation():
    grid = affine_grid()
    bindings = {(s, "pad"): {"sha256": str(s)*64} for s in (0, 1)}
    owner = tensor_owner([(s, 0, "source:pad", np.zeros((1, 3))) for s in (0, 1)],
                         [(s, 0, "pad", grid) for s in (0, 1)], bindings=bindings)
    poses, nodes = poses_for(1), torch.full((1, 41, 3), 12., dtype=torch.float64)
    frame = {"physics_step": 1, "reset_ids": (0,)}
    result = owner.capture(poses, nodes, frame=frame)
    poses.add_(1)
    frame["reset_ids"] = (1,)
    grid["values"][:] = 100
    bindings[0, "pad"]["sha256"] = "foreign"
    owner.require_ready(frame={"physics_step": 1, "reset_ids": (0,)})
    assert result.frame == {"physics_step": 1, "reset_ids": (0,)}
    assert result.poses[0, 0, 0, 0] == 0
    with pytest.raises(ValueError, match="frame"):
        owner.require_ready(frame=frame)
    with pytest.raises(ValueError, match="binding"):
        owner.validate_metadata(bindings=bindings)
    nodes[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        owner.capture(poses_for(1), nodes, frame=2)
    with pytest.raises(ValueError, match="frame"):
        owner.require_ready(frame={"physics_step": 1, "reset_ids": (0,)})


def test_float32_world_subtraction_can_flip_a_gap_outside_fixed_two_micrometre_band():
    # A shared batch may have large world origins. The original FP32 input cast
    # loses local x=30um at x=1000m, while the FP64 gap is +20um, away from zero.
    x, _, _ = np.indices((9, 9, 9))
    grid = dict(values=x*.01-.04+.00499, origin=np.full(3, -.04), spacing=np.array(.01))
    shapes = [(s, 0, "cooked:pad", np.zeros((1, 3))) for s in (0, 1)]
    owner = tensor_owner(shapes, [(s, 0, "pad", grid) for s in (0, 1)])
    poses = poses_for(1)
    poses[:, :, :, 0] = 1000.
    nodes = torch.tensor(np.random.default_rng(271).uniform(.005, .02, (1, 41, 3))+[1000., 0, .6])
    nodes[0, 0] = torch.tensor([1000.00003, .0123, .6234], dtype=torch.float64)
    matrix = pose_matrix(poses[0, 0, 0].numpy()[[0, 1, 2, 6, 3, 4, 5]])
    expected = sample_contact(grid, matrix, nodes[0].numpy())
    expected64 = sample_contact(grid, matrix, nodes[0].numpy(), dtype=np.float64)
    assert expected["minimum_gap_m"] < 0 < expected64["minimum_gap_m"]
    assert expected64["minimum_gap_m"] > 2e-6
    actual = owner.capture(poses, nodes, frame="large_world_origin")
    assert actual.sdf_fallback.all()
    assert actual.sdf_negative_nodes[0, 0, 0]
    assert actual.sdf_minimum_gap_m[0, 0] == expected["minimum_gap_m"]


def test_nonfinite_derived_bounds_never_become_a_clearance_pass():
    owner = tensor_owner([(s, 0, "source:nonpad", np.zeros((1, 3))) for s in (0, 1)])
    poses = poses_for(1)
    poses[..., :3] = 1e308
    nodes = torch.full((1, 41, 3), -1e308, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        owner.capture(poses, nodes, frame="overflow")
    with pytest.raises(ValueError, match="frame"):
        owner.require_ready(frame="overflow")


def test_original_44_hulls_and_locked_sdf_match_policy_owned_oracle():
    pytest.importorskip("pxr", reason="existing USD library path is required for original asset oracle")
    from go2_pvcnn.control.dh116.tabletop_tensor_geometry import TensorGeometry
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import _canonical_pinch_history
    from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_pinch_distance import PolicyOwnedPinchGeometry
    from test_dh116_measured_closure_geometry import case
    with _canonical_pinch_history():
        original = PolicyOwnedPinchGeometry(Path(__file__).resolve().parents[1], scope="cpu_fixture")
    original._radius = lambda exchange: {"value": .005}
    row, sim, bindings, read, hands, clock = case.__wrapped__(original)
    row["material_director"] = {"source_scope": "cpu_fixture"}
    expected = original.capture(row, sim=sim, bindings=bindings, read_clock=read)
    names = tuple(tuple(hand.body_names) for hand in hands)
    owner = TensorGeometry.from_policy_geometry(original, names, device="cpu")
    poses = torch.tensor(np.stack([h.transforms[0] for h in hands]))[None]
    nodes = torch.tensor(row["rod"]["node_positions"])
    actual = owner.capture(poses, nodes, frame=clock)
    assert len(owner.columns) == 44
    assert owner.source_bindings == {(s, n): original.sources[side]["sdf_binding"][n]
        for s, side in enumerate(("left", "right")) for n in original.REQUIRED_PADS}
    for (side, key), column in owner.columns.items():
        expected_side = expected["sides"][("left", "right")[side]]
        assert actual.table_clearance_m[0, column] == pytest.approx(expected_side["table_clearance_m"][key], abs=1e-12)
        if key in expected_side["non_target_surface_gap_lower_m"]:
            assert actual.non_target_lower_m[0, column] == pytest.approx(expected_side["non_target_surface_gap_lower_m"][key], abs=1e-12)
    for (side, name), column in owner.pad_columns.items():
        pad = expected["sides"][("left", "right")[side]]["required_pad_sdf"][name]
        assert actual.sdf_valid[0, column] == required_sdf_valid(pad)
        assert actual.sdf_minimum_gap_m[0, column] == pytest.approx(pad["minimum_gap_m"], abs=1e-7)
    assert actual.hand_separation_lower_m[0] == pytest.approx(expected["inter_hand_separation_lower_m"], abs=1e-12)
    assert bool(actual.geometry_passed[0]) == expected["passed"]
    assert not actual.native_qualification and not actual.tripod_supported_guard_covered
