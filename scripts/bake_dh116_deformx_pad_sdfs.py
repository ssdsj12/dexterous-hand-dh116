#!/usr/bin/env python3
"""Opt-in standalone bake and registration of parity-verified DH116 pad hulls."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from scripts.audit_dh116_cooked_hull_device_parity import _pad_hull, physics_device_matches
from scripts.verify_dh116_deformx_rigid_bridge import write_json


_SPACING_M = 0.001
_MARGIN_M = 0.012
_ROD_RADIUS_M = 0.005


def checked_bytes(path, expected_sha):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError(f"SHA256 differs for {path.name}")
    return data


def checked_json(path, expected_sha):
    return json.loads(checked_bytes(path, expected_sha))


def validate_signed_grid(grid, vertices, *, rod_radius):
    values = grid.values
    if values.ndim != 3 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("SDF grid must be finite and three-dimensional")
    boundary_min = float(min(np.take(values, index, axis=axis).min()
                             for axis in range(3) for index in (0, -1)))
    if boundary_min <= rod_radius:
        raise ValueError("SDF boundary does not clear the selected rod radius")
    centre = vertices.mean(axis=0)
    outside = centre.copy()
    outside[0] = vertices[:, 0].max() + rod_radius + 2 * grid.spacing
    sampled, inside_grid = grid.sample(np.vstack((centre, outside, vertices)))
    if (not np.all(inside_grid) or not np.isfinite(sampled).all()
            or values.min() >= 0 or sampled[0] >= 0 or sampled[1] <= rod_radius):
        raise ValueError("SDF sign or support check failed")
    max_vertex_distance = float(np.abs(sampled[2:]).max())
    if max_vertex_distance > 2 * grid.spacing:
        raise ValueError("SDF at cooked hull vertices exceeds two voxels")
    return {
        "shape": list(values.shape), "dtype": str(values.dtype), "bytes": values.nbytes,
        "origin_link_m": grid.origin.tolist(), "spacing_m": grid.spacing,
        "lattice_hi_link_m": (grid.origin + grid.spacing * (np.asarray(values.shape) - 1)).tolist(),
        "inside_sample_link_m": centre.tolist(), "inside_signed_distance_m": float(sampled[0]),
        "outside_sample_link_m": outside.tolist(), "outside_signed_distance_m": float(sampled[1]),
        "distance_min_m": float(values.min()), "distance_max_m": float(values.max()),
        "min_six_lattice_faces_signed_distance_m": boundary_min,
        "max_abs_sdf_at_hull_vertices_m": max_vertex_distance,
        "configurable_truncation_distance_m": None,
    }


def input_packets(capture, lock):
    for name, digest in lock["outer_exit_sha256"].items():
        if checked_bytes(capture / name, digest).strip() != b"0":
            raise ValueError(f"required successful outer process: {name}")
    parity = checked_json(capture / "parity.json", lock["parity_sha256"])
    gpu = checked_json(capture / "gpu/result.json", lock["gpu_result_sha256"])
    if (not parity["parity_passed"] or not parity["backend_identity_passed"]
            or parity["reference_frames"] != ["mesh_local", "mesh_local"]
            or not physics_device_matches(gpu["physics_device"])
            or gpu["physics_device"]["requested_device"] != "cuda:0"):
        raise ValueError("GPU PhysX parity and coordinate basis are required")
    packets = {}
    for key, digest in lock["gpu_rows"].items():
        side, link = key.split(".")
        row = checked_json(capture / f"gpu/{side}_{link}.json", digest)
        if (not parity["pads"][key]["parity_passed"]
                or (row["side"], row["link"], row["clock"]) != (side, link, gpu["clock"])):
            raise ValueError(f"invalid parity-backed pad identity: {key}")
        hull, basis_delta = _pad_hull(row)
        if (len(hull["vertices"]) != parity["pads"][key]["gpu_vertices"]
                or len(hull["polygons"]) != parity["pads"][key]["gpu_polygons"]):
            raise ValueError(f"invalid parity-backed pad topology: {key}")
        packets[key] = (row, hull, basis_delta, digest)
    return packets


def bake_register(packets, lock, package_root, output):
    from scipy.spatial.transform import Rotation

    for name, digest in lock["backend_files"].items():
        checked_bytes(package_root / "ropesim" / name, digest)
    sys.path.insert(0, str(package_root))
    import torch
    import warp as wp
    import ropesim
    from ropesim import sdf
    from ropesim.isaac_bridge import ColliderSpec, bake_body, weld

    wp.config.kernel_cache_dir = str(output / "warp-cache")
    wp.init()
    device = wp.get_device("cuda:0")
    if not device.is_cuda or not torch.cuda.is_available() or torch.cuda.current_device() != 0:
        raise RuntimeError("expected CUDA:0 for the standalone bake and registry")
    # One continuous free rod, far from the recorded original hand locations.
    nodes = np.zeros((1, 21, 3), dtype=np.float32)
    nodes[0, :, 0] = np.linspace(0, 0.5, 21)
    nodes[0, :, 2] = 5.0
    batch = ropesim.RopeBatch.from_positions(nodes, 0.065, 800.0, 0.018, 1 / 5760, 8,
                                              pinned=np.zeros(21, dtype=bool), deterministic=True)
    batch.set_contact(_ROD_RADIUS_M)
    records = {}
    with wp.ScopedDevice(device):
        for key, (row, hull, basis_delta, digest) in packets.items():
            vertices, faces = weld(np.asarray(hull["vertices"], dtype=np.float32),
                                  np.asarray(hull["faces"], dtype=np.int64))
            if not sdf.is_closed(faces):
                raise ValueError(f"live hull is open after upstream weld: {key}")
            spec = ColliderSpec(asset=row["side"] + "_hand", bodies=[row["link"]], dynamic=True,
                                spacing=_SPACING_M, margin=_MARGIN_M, round=0., thickness=0.)
            grid = bake_body(vertices, faces, spec, key)
            metrics = validate_signed_grid(grid, vertices, rod_radius=_ROD_RADIUS_M)
            if grid.spacing != _SPACING_M:
                raise ValueError(f"unexpected grid spacing: {key}")
            position = np.asarray(row["sdk_body"]["position_m"], dtype=np.float32)
            quaternion = np.asarray(row["sdk_body"]["quaternion_xyzw"], dtype=np.float32)
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix().T
            matrix[3, :3] = position
            pose_delta = float(np.abs(matrix - np.asarray(row["sdk_body"]["link_to_world_row_major"])).max())
            if pose_delta > 5e-7:
                raise ValueError(f"SDK quaternion or translation differs from the link transform: {key}")
            idx = batch.add_mesh(*grid.as_args(), pos=position, quat=quaternion)
            if idx != len(records):
                raise RuntimeError("unexpected RopeBatch mesh slot")
            path = output / f"{key}.npz"
            np.savez_compressed(path, values=grid.values, origin=grid.origin, spacing=grid.spacing)
            record = {
                **metrics, "gpu_row_sha256": digest, "closed_welded_hull": True,
                "welded_vertices": len(vertices), "welded_faces": len(faces),
                "rod_radius_m": _ROD_RADIUS_M, "voxels_per_rod_radius": _ROD_RADIUS_M / _SPACING_M,
                "margin_m": _MARGIN_M, "thickness_m": 0., "round_m": 0.,
                "frame": "original_pad_coincident_local_basis_m",
                "native_coincident_basis_max_matrix_delta": basis_delta,
                "position_world_m": position.tolist(), "quaternion_world_xyzw": quaternion.tolist(),
                "sdk_pose_matrix_max_delta": pose_delta, "rope_mesh_index": idx,
                "sdf_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            write_json(output / f"{key}.json", record)
            records[key] = record
    batch.enable_contact_forces(True)
    shape = list(batch.mesh_wrenches().shape)
    if shape != [1, len(records), 6]:
        raise RuntimeError("registered mesh count disagrees with RopeBatch wrench buffer")
    torch.cuda.synchronize()
    return {
        "scope": "parity_verified_original_live_cooked_pad_sdf_and_explicit_registration",
        "wp_device": str(device), "torch_device": str(torch.cuda.current_device()),
        "gpu_name": torch.cuda.get_device_name(0), "rods_per_env": 1,
        "mesh_wrench_buffer_shape": shape, "simulation_steps": 0,
        "isaac_launched": False, "contact_verified": False, "pads": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--ropesim-package-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    packets = input_packets(args.capture_dir, lock)
    if not args.execute_gpu:
        print(json.dumps({"authenticated_native_pads": list(packets), "gpu_started": False}))
        return
    if args.ropesim_package_root is None or args.output is None:
        parser.error("--execute-gpu requires --ropesim-package-root and --output")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = bake_register(packets, lock, args.ropesim_package_root.resolve(), args.output)
        write_json(args.output / "result.json", result)
    except Exception:
        write_json(args.output / "failure.json", {"exception": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
