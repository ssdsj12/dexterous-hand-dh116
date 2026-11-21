#!/usr/bin/env python3
"""Bake independent original Power proximal SDF candidates from sealed Tensor View rays."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from scripts import audit_dh116_power_proximal_raycast as source
from scripts import bake_dh116_deformx_pad_sdfs as pad_bake
from scripts.bake_dh116_power_left_palm_sdf import BACKEND_LOCK, BACKEND_LOCK_SHA
from scripts.bake_dh116_power_right_palm_sdf import audit_grid, verify_weld
from scripts.audit_dh116_cooked_hull_device_parity import _pad_hull

CONTRACT = source.ROOT / "docs/contracts/dh116_teacher_power_proximal_tensor_pose_native_v1.json"


def measured_packets(case):
    receipt = json.loads(CONTRACT.read_bytes())["result"]
    if source.sha(case/"outer.json") != receipt["outer_sha256"]:
        raise ValueError("the exact accepted Power proximal Tensor View receipt is required")
    raw = source._receipt(case, "cuda:0")
    native = json.loads((raw/"result.json").read_bytes())
    calibration = native["reference_frame_calibration"]["live_simulation_cooked"]
    if calibration["confirmed_frame"] != "mesh_local":
        raise ValueError("original metre mesh-local cooking frame required")
    rays = json.loads((raw/"proximal_tensor_pose_rays.json").read_bytes())
    if rays["link"] != source.LINK or rays["sdk_clock"][0] != native["clock"]["physics_step"]:
        raise ValueError("Tensor View rays belong to a different link or SDK tick")
    packets, meshes = {}, {}
    for side in ("left", "right"):
        path = raw/f"{side}_{source.LINK}.json"
        row = json.loads(path.read_bytes())
        item = rays["comparisons"][side]
        witness = item["rays"]
        owned = [hit for hit in witness["rays"] if hit["native_hit_distance_m"] is not None]
        if (row["side"] != side or row["link"] != source.LINK or row["clock"] != witness["clock"]
                or witness["source_body"] != row["source"]["body_path"]
                or witness["owned_rays"] != len(owned) or len(owned) < 3
                or any(hit["native_body"] != row["source"]["body_path"]
                       or hit["native_collider"] not in (row["source"]["collision_owner"], row["source"]["mesh_path"])
                       or hit["original_fan_m"] is None for hit in owned)):
            raise ValueError(f"owned original same-clock proximal rays required for {side}")
        errors = [abs(hit["native_hit_distance_m"] - hit["original_fan_m"]) for hit in owned]
        normals = [float(np.dot(hit["native_normal_link"], -np.asarray(hit["direction_link"]))) for hit in owned]
        if (not np.isfinite(errors).all() or not np.isfinite(normals).all()
                or max(errors) > calibration["tolerance_m"] or min(normals) <= 0.):
            raise ValueError("proximal rays do not support the captured fan and its orientation")
        hull, basis_delta = _pad_hull(row)
        mesh = trimesh.Trimesh(vertices=hull["vertices"], faces=hull["faces"], process=False)
        if not mesh.is_volume or len(mesh.split(only_watertight=False)) != 1:
            raise ValueError("original proximal triangle fan must be a closed, outward single body")
        pose = item["tensor_pose"]
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(pose["quaternion_xyzw"]).as_matrix().T
        matrix[3, :3] = pose["position_m"]
        aligned = deepcopy(row)
        aligned["sdk_body"].update(pose, link_to_world_row_major=matrix.tolist())
        key = f"{side}.{source.LINK}"
        packets[key] = aligned, hull, basis_delta, source.sha(path)
        meshes[key] = mesh
    if len({pkt[-1] for pkt in packets.values()}) != 2:
        raise ValueError("the two proximal captures must be independent")
    return packets, meshes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--ropesim-package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spacing-m", choices=(.001, .0005), type=float, default=.001)
    args = parser.parse_args(argv)
    root = args.ropesim_package_root.resolve(strict=True)
    packets, meshes = measured_packets(args.capture.resolve(strict=True))
    lock = pad_bake.checked_json(BACKEND_LOCK, BACKEND_LOCK_SHA)
    for name, digest in lock["backend_files"].items():
        pad_bake.checked_bytes(root/"ropesim"/name, digest)
    if args.output is None:
        print(json.dumps(dict(verified_meshes=list(packets), gpu_started=False)))
        return 0

    sys.path.insert(0, str(root))
    from ropesim.isaac_bridge import weld
    from ropesim.sdf import SdfGrid

    topology = {name: verify_weld(mesh, weld) for name, mesh in meshes.items()}
    args.output.mkdir()
    result = pad_bake.bake_register(packets, lock, root, args.output, spacing_m=args.spacing_m)
    grids = result.pop("pads")
    for name, mesh in meshes.items():
        path = args.output/f"{name}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != grids[name]["sdf_file_sha256"]:
            raise ValueError("standalone proximal grid digest differs")
        with np.load(path, allow_pickle=False) as data:
            grid = SdfGrid(data["values"], data["origin"], float(data["spacing"]))
        grids[name]["reference_error"] = audit_grid(mesh, grid.sample, spacing_m=grid.spacing)
        grids[name]["reference_fixed_offsets_error"] = (grids[name]["reference_error"] if grid.spacing == .001
            else audit_grid(mesh, grid.sample, spacing_m=.001))
        grids[name]["welded_topology"] = topology[name]
    result.update(scope="isolated_finger52_link_gpu_sdf_candidates", proximal=grids,
                  raw_native_receipt_sha256=source.sha(args.capture/"outer.json"),
                  native_power_task_registration=False, native_power_contact_qualified=False,
                  grasp_verified=False, teacher_admission=False)
    pad_bake.write_json(args.output/"result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
