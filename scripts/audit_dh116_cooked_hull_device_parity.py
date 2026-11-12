#!/usr/bin/env python3
"""Compare matched CPU and GPU live-cooked PhysX hulls without starting Isaac."""

import argparse
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS, HAND_SCENE_KEYS
from scripts.verify_dh116_deformx_rigid_bridge import write_json


def physics_device_matches(record):
    requested = record["requested_device"]
    manager = record["manager_device"]
    ordinal = record["tensor_device_ordinal"]
    dynamics = record["gpu_dynamics_enabled"]
    broadphase = record["broadphase"]
    override = record["physx_device_override"]
    if requested == "cpu":
        return manager == "cpu" and ordinal == -1 and not dynamics and broadphase == "MBP" and override in (-1, 0)
    if requested.startswith("cuda:") and requested[5:].isdigit():
        return (manager == requested and ordinal == int(requested[5:])
                and dynamics and broadphase == "GPU" and override in (-1, 1))
    return False


def _polygon_topology(hull, index_map):
    result = []
    for polygon in hull["polygons"]:
        start = polygon["index_base"]
        stop = start + polygon["num_vertices"]
        ring = tuple(index_map[index] for index in hull["indices"][start:stop])
        first = ring.index(min(ring))
        result.append(ring[first:] + ring[:first])
    return sorted(result)


def compare_hull_topology(cpu, gpu, *, tolerance_m=5e-7):
    """Compare directed polygon cycles after bijective minimum-distance vertex matching."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    a, b = (np.asarray(hull["vertices"], dtype=float) for hull in (cpu, gpu))
    result = {"cpu_vertices": len(a), "gpu_vertices": len(b),
              "cpu_polygons": len(cpu["polygons"]), "gpu_polygons": len(gpu["polygons"]),
              "max_vertex_delta_m": None, "directed_polygon_topology_equal": False,
              "vertex_tolerance_m": tolerance_m, "parity_passed": False}
    if (a.ndim != 2 or a.shape != b.shape or a.shape[1] != 3 or len(a) < 4
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        return result
    cost = cdist(a, b)
    rows, columns = linear_sum_assignment(cost)
    delta = float(cost[rows, columns].max())
    gpu_to_cpu = dict(zip(columns.tolist(), rows.tolist()))
    polygons_match = _polygon_topology(cpu, range(len(a))) == _polygon_topology(gpu, gpu_to_cpu)
    result.update(max_vertex_delta_m=delta, directed_polygon_topology_equal=polygons_match,
                  parity_passed=delta <= tolerance_m and polygons_match)
    return result


def _pad_hull(record):
    owner = record["source"]["collision_owner"]
    query = record["queries"][owner]["live_simulation_cooked"]
    if (not record["sdk_body"]["sdk_rigid_body_found"] or query.get("error")
            or len(query["hulls"] or []) != 1):
        raise ValueError("expected an active SDK body and one live owner hull")
    matrices = {name: np.asarray(value) for name, value in record["usd_transforms_row_major"].items()}
    inverse = np.linalg.inv(matrices["body_path"])
    basis_delta = max(float(np.max(np.abs(matrices[name] @ inverse - np.eye(4))))
                      for name in ("collision_owner", "mesh_path"))
    if basis_delta > 5e-7:
        raise ValueError("pad local bases diverge from link-local")
    return query["hulls"][0], basis_delta


def compare_native_runs(cpu_run, gpu_run):
    runs = (cpu_run, gpu_run)
    cpu, gpu = (json.loads((run / "result.json").read_text()) for run in runs)
    reference = [row["reference_frame_calibration"].get("live_simulation_cooked", {}).get("confirmed_frame")
                 for row in (cpu, gpu)]
    devices = [row.get("physics_device", {}) for row in (cpu, gpu)]
    device_ok = bool(all(devices) and [d["requested_device"] for d in devices] == ["cpu", "cuda:0"]
                     and all(physics_device_matches(d) for d in devices))
    source_match = all(cpu["source_usds"][side]["sha256"] == gpu["source_usds"][side]["sha256"]
                       for side in HAND_SCENE_KEYS)
    clock_match = cpu["clock"] == gpu["clock"]
    rows = {}
    for side in HAND_SCENE_KEYS:
        for link in DISTAL_PAD_LINKS:
            key = f"{side}.{link}"
            try:
                a, b = (json.loads((run / f"{side}_{link}.json").read_text()) for run in runs)
                if any((row["side"], row["link"], row["clock"]) != (side, link, capture["clock"])
                       for row, capture in ((a, cpu), (b, gpu))):
                    raise ValueError("side, link or clock identity mismatch")
                identity = ("body_path", "collision_owner", "mesh_path", "geometry_sha256")
                if any(a["source"][name] != b["source"][name] for name in identity):
                    raise ValueError("native source identities differ")
                a_hull, a_basis = _pad_hull(a)
                b_hull, b_basis = _pad_hull(b)
                rows[key] = {**compare_hull_topology(a_hull, b_hull),
                             "cpu_local_basis_max_delta": a_basis, "gpu_local_basis_max_delta": b_basis}
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                rows[key] = {"parity_passed": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"scope": "original_dh116_live_physx_cpu_cuda0_cooking_parity",
            "physics_devices": devices, "backend_identity_passed": device_ok,
            "matching_source_sha": source_match, "matching_physics_clock": clock_match,
            "reference_frames": reference, "pads": rows,
            "parity_passed": (device_ok and source_match and clock_match
                              and reference == ["mesh_local", "mesh_local"]
                              and all(row["parity_passed"] for row in rows.values())),
            "sdf_baked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-run", type=Path, required=True)
    parser.add_argument("--gpu-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_native_runs(args.cpu_run, args.gpu_run)
    write_json(args.output, result)
    return int(not result["parity_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
