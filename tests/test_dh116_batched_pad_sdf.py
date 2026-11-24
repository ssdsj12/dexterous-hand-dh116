"""Original pad-SDF arithmetic and cache parity; no native qualification."""
import os
import hashlib
import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from scripts.prepare_dh116_pinch_closure_margin import sample_contact
from scripts.audit_dh116_pinch_initial_pose import pose_matrix


TEST_DEVICE = os.environ.get("DH116_PAD_SDF_TEST_DEVICE", "cpu")


def poses_for(n):
    poses = np.zeros((n, 2, 23, 7), dtype=np.float64)
    poses[..., 6] = 1
    return poses


def assert_row(actual, expected, atol):
    assert actual.keys() == expected.keys()
    numeric = ("minimum_gap_m", "gradient_norm_at_minimum", "node_gaps_m", "local_node_at_minimum")
    for key in expected:
        if key in numeric and expected[key] is not None:
            if key == "node_gaps_m":
                assert [v is None for v in actual[key]] == [v is None for v in expected[key]]
                a = [v for v in actual[key] if v is not None]
                b = [v for v in expected[key] if v is not None]
                np.testing.assert_allclose(a, b, rtol=0, atol=atol)
            else:
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=atol)
        else:
            assert actual[key] == expected[key], key


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_transform_interpolation_and_minimum_match_original(dtype):
    from go2_pvcnn.control.dh116.tabletop_batched_pad_sdf import BatchedPadSDF
    rng = np.random.default_rng(783)
    n = 7
    poses = poses_for(n)
    poses[..., :3] = rng.normal(size=(n, 2, 23, 3)) * .1
    poses[..., 3:] = Rotation.random(n * 2 * 23, random_state=rng).as_quat().reshape(n, 2, 23, 4)
    poses[..., 3:] *= 1 + 1e-7
    nodes = rng.uniform(-.004, .004, (n, 41, 3))
    poses[:, 0, 4, :3] = 0
    poses[:, 1, 8, :3] = 0
    x, y, z = np.meshgrid(np.arange(6), np.arange(5), np.arange(7), indexing="ij")
    grid = dict(values=(x + 2*y - z + .3*x*y + .4*y*z + .1*x*y*z) * .0001,
                origin=np.array([-.003, -.002, -.004]), spacing=np.array(.0015))
    pads = [(0, 4, "finger14_link", grid), (1, 8, "finger23_link", grid)]
    batch = BatchedPadSDF(pads, device=TEST_DEVICE, dtype=dtype)
    batch.capture(torch.from_numpy(poses), torch.from_numpy(nodes), frame=("run", 17))
    npdtype = np.float32 if dtype == torch.float32 else np.float64
    for lane in range(n):
        batch.validate(lane, poses[lane], nodes[lane], frame=("run", 17))
        for side, link, name, source in pads:
            p = poses[lane, side, link]
            expected = sample_contact(source, pose_matrix(p[[0, 1, 2, 6, 3, 4, 5]]), nodes[lane], dtype=npdtype)
            assert_row(batch.row(lane, side, name), expected, 1e-7 if dtype == torch.float32 else 1e-10)


def test_lattice_boundary_negative_predicate_and_empty_lane_match_default():
    from go2_pvcnn.control.dh116.tabletop_batched_pad_sdf import BatchedPadSDF
    x = np.indices((4, 4, 4))[0] * .25
    grid = dict(values=x, origin=np.zeros(3), spacing=np.array(.25))
    poses = poses_for(2)
    nodes = np.full((2, 41, 3), 2.)
    nodes[0, :7] = [[0, 0, 0], [.75, .25, .25], [np.nextafter(np.float32(.75), np.float32(0)), .25, .25],
                        [-.00001, .25, .25], [.5, .5, .5], [0, .75, .25], [0, .25, .75]]
    batch = BatchedPadSDF([(0, 0, "pad", grid)], device=TEST_DEVICE)
    batch.capture(poses, nodes, frame=8)
    for lane in range(2):
        assert_row(batch.row(lane, 0, "pad"), sample_contact(grid, np.eye(4), nodes[lane]), 0)
    row = batch.row(0, 0, "pad")
    assert row["in_lattice_nodes"] == [0, 2, 4]
    assert row["contact_predicate_nodes"] == [0]
    assert batch.row(1, 0, "pad")["minimum_node_id"] is None


def test_zero_gradient_preserves_first_tie_and_owned_grid():
    from go2_pvcnn.control.dh116.tabletop_batched_pad_sdf import BatchedPadSDF
    grid = dict(values=np.full((3, 3, 3), .001), origin=np.zeros(3), spacing=np.array(.25))
    original = {k: np.array(v, copy=True) for k, v in grid.items()}
    batch = BatchedPadSDF([(1, 7, "pad", grid)], device=TEST_DEVICE)
    grid["values"][:] = 100
    poses = poses_for(1)
    nodes = np.full((1, 41, 3), .125)
    batch.capture(poses, nodes, frame={"step": 4})
    row = batch.row(0, 1, "pad")
    assert_row(row, sample_contact(original, np.eye(4), nodes[0]), 0)
    assert row["minimum_node_id"] == 0
    assert row["gradient_norm_at_minimum"] == 0
    assert row["contact_predicate_nodes"] == []
    assert row["negative_gap_nodes"] == list(range(41))
    row["in_lattice_nodes"].clear()
    assert len(batch.row(0, 1, "pad")["in_lattice_nodes"]) == 41


def test_cache_owns_inputs_and_rejects_stale_epoch_pose_nodes_and_failed_capture():
    from go2_pvcnn.control.dh116.tabletop_batched_pad_sdf import BatchedPadSDF
    grid = dict(values=np.ones((2, 2, 2)), origin=np.zeros(3), spacing=np.array(1.))
    batch = BatchedPadSDF([(0, 0, "pad", grid)], device=TEST_DEVICE)
    poses = poses_for(2)
    nodes = np.full((2, 41, 3), .25)
    frame = {"step": 4}
    batch.capture(poses, nodes, frame=frame)
    saved_poses, saved_nodes = poses.copy(), nodes.copy()
    poses[0, 0, 0, 0] += .01
    nodes[0, 0, 0] += .01
    frame["step"] = 5
    batch.validate(0, saved_poses[0], saved_nodes[0], frame={"step": 4})
    for p, q, f, message in [(poses[0], saved_nodes[0], {"step": 4}, "pose"),
                             (saved_poses[0], nodes[0], {"step": 4}, "nodes"),
                             (saved_poses[0], saved_nodes[0], frame, "frame")]:
        with pytest.raises(ValueError, match=message):
            batch.validate(0, p, q, frame=f)
    for lane in [-1, 2, True]:
        with pytest.raises(ValueError, match="lane"):
            batch.validate(lane, saved_poses[0], saved_nodes[0], frame={"step": 4})
    nodes[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        batch.capture(saved_poses, nodes, frame=6)
    with pytest.raises(ValueError, match="frame"):
        batch.validate(0, saved_poses[0], saved_nodes[0], frame={"step": 4})
    with pytest.raises(ValueError, match="frame"):
        batch.row(0, 0, "pad")


@pytest.mark.parametrize("change", ["values", "origin", "spacing", "quaternion", "node_count"])
def test_invalid_grid_or_actual_inputs_are_rejected(change):
    from go2_pvcnn.control.dh116.tabletop_batched_pad_sdf import BatchedPadSDF
    grid = dict(values=np.ones((2, 2, 2)), origin=np.zeros(3), spacing=np.array(.1))
    poses, nodes = poses_for(1), np.zeros((1, 41, 3))
    if change == "values": grid["values"][0, 0, 0] = np.nan
    if change == "origin": grid["origin"] = np.zeros(2)
    if change == "spacing": grid["spacing"] = np.array(0.)
    if change == "quaternion": poses[0, 1, 0, 3:] = 0
    if change == "node_count": nodes = nodes[:, :40]
    with pytest.raises(ValueError):
        batch = BatchedPadSDF([(0, 0, "pad", grid)], device=TEST_DEVICE)
        batch.capture(poses, nodes, frame=2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_original_locked_side_grids_match_original_sampler(dtype):
    from go2_pvcnn.control.dh116.tabletop_batched_pad_sdf import BatchedPadSDF
    from scripts.solve_dh116_tripod_sdf_constrained_direction import SDF_DIR, SDF_SHA
    summary_file = SDF_DIR / "result.json"
    if not summary_file.exists(): pytest.skip("original source-bound locked SDF evidence unavailable")
    assert hashlib.sha256(summary_file.read_bytes()).hexdigest() == SDF_SHA
    summary = json.loads(summary_file.read_text())
    rng = np.random.default_rng(819)
    grids = []
    for side, name in [(0, "left"), (1, "right")]:
        for link, pad in [(4, "finger14_link"), (8, "finger23_link")]:
            key = name + "." + pad
            path = SDF_DIR / (key + ".npz")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == summary["pads"][key]["sdf_file_sha256"]
            with np.load(path, allow_pickle=False) as d: grid = {k: d[k].copy() for k in d.files}
            grids.append((side, link, pad, grid))
    poses = poses_for(4)
    nodes = rng.uniform(-.012, .012, (4, 41, 3)) + [0, -.006, -.023]
    poses[..., 3:] = Rotation.from_euler("xyz", rng.uniform(-.2, .2, (4*2*23, 3))).as_quat().reshape(4, 2, 23, 4)
    batch = BatchedPadSDF(grids, device=TEST_DEVICE, dtype=dtype)
    batch.capture(poses, nodes, frame=("locked-grid", 9))
    npdtype = np.float32 if dtype == torch.float32 else np.float64
    for lane in range(4):
        for side, link, name, grid in grids:
            p = poses[lane, side, link]
            expected = sample_contact(grid, pose_matrix(p[[0, 1, 2, 6, 3, 4, 5]]), nodes[lane], dtype=npdtype)
            assert_row(batch.row(lane, side, name), expected, 1e-7 if dtype == torch.float32 else 1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_per_lane_radius_matches_original_cast_and_is_owned(dtype):
    from go2_pvcnn.control.dh116.tabletop_batched_pad_sdf import BatchedPadSDF
    grid = dict(values=np.full((2, 2, 2), .006), origin=np.zeros(3), spacing=np.array(.25))
    batch = BatchedPadSDF([(0, 0, "pad", grid)], device=TEST_DEVICE, dtype=dtype)
    radii = np.array([.0050000004, .0060000004])
    saved = radii.copy()
    nodes = np.full((2, 41, 3), .125)
    batch.capture(poses_for(2), nodes, frame=18, radius=radii)
    radii[:] = 1
    np.testing.assert_array_equal(batch.radii, saved)
    for lane in range(2):
        expected = sample_contact(grid, np.eye(4), nodes[lane], radius=saved[lane],
                                  dtype=np.float32 if dtype == torch.float32 else np.float64)
        assert_row(batch.row(lane, 0, "pad"), expected, 0)
