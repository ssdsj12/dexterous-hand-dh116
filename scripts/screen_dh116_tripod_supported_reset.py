#!/usr/bin/env python3
"""Screen both original Tripod Reset poses against the recorded supported rod tube."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference
from scripts.audit_dh116_pinch_initial_pose import load_geometry
from scripts.audit_dh116_wire_contact_witness import shapes_world
from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
from scripts.run_dh116_work_packages import file_hash

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT/"configs/dh116_tasks"
REQUEST = TASKS/"flexible_tripod_supported_seed_v1.json"
BASE_SCENE = TASKS/"flexible_power_supported_contact_v1.json"
ROUTES = {side: TASKS/f"flexible_tripod_{side}_parked_rod_approach_cpu_v1.json"
          for side in ("left", "right")}
SCOPE = "tripod_end_supported_contact_diagnostic"


def archived_tube(node_samples, *, radius_m=.005):
    """Represent all recorded material segments by conservative support planes."""
    samples = np.asarray(node_samples, dtype=float)
    if (samples.ndim != 3 or samples.shape[1:] != (41, 3) or not len(samples)
            or not np.isfinite(samples).all() or radius_m != .005):
        raise ValueError("finite original 41-node material and 5 mm rod radius required")
    azimuth = np.linspace(0., 2*np.pi, 72, endpoint=False)
    elevation = np.linspace(-np.pi/2, np.pi/2, 25)[:, None]
    normals = np.stack(np.broadcast_arrays(np.cos(elevation)*np.cos(azimuth),
        np.cos(elevation)*np.sin(azimuth), np.sin(elevation)), axis=-1).reshape(-1, 3)
    upper = np.stack([(samples[:, start:start+3].reshape(-1, 3)@normals.T).max(0)+radius_m
                      for start in range(0, 40, 2)])
    return normals, upper


def separation(vertices, normals, upper):
    lower = (np.asarray(vertices) @ normals.T).min(0)
    return float((lower[None, :] - upper).max(axis=-1).min())


def scene_request():
    request = json.loads(REQUEST.read_bytes())
    if (request["version"] != "dh116-tripod-supported-seed-v1" or request["scope"] != SCOPE
            or file_hash(BASE_SCENE) != request["shared_support_scene_sha256"]):
        raise ValueError("the independent Tripod request and original free supported scene are required")
    config = deepcopy(json.loads(BASE_SCENE.read_bytes()))
    config["scope"] = SCOPE
    config["offline_candidate_sha256"] = file_hash(REQUEST)
    for side, path in ROUTES.items():
        spec = request["hands"][side]
        row = json.loads(path.read_bytes())
        if file_hash(path) != spec["route_sha256"] or row["side"] != side:
            raise ValueError("own per-hand Tripod route required")
        first = row["waypoints"][0]
        delta = np.asarray(spec["world_offset_m"], dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("finite frozen world translation required")
        pose = np.asarray(first["pose_wxyz"], dtype=float)
        pose[:3] += delta
        config["hands"][side] = dict(pose_wxyz=pose.tolist(), q_rad=first["active_q_rad"])
    return request, config


def screen(receipt, request, config):
    samples = [load_parked_reference(receipt, request["archived_settling_outer_sha256"], tick)
               for tick in range(3, 241)]
    if any(list(row.support_boxes) != config["support_scene"]["supports"] for row in samples):
        raise ValueError("support boxes differ from the physical measurements")
    nodes = np.stack([config["rod_nodes_w_m"], *[row.node_positions for row in samples]])
    tube = archived_tube(nodes)
    supports = [(np.asarray(box["center_m"])-np.asarray(box["size_m"])/2,
                 np.asarray(box["center_m"])+np.asarray(box["size_m"])/2) for box in samples[0].support_boxes]
    rows = {}
    for side, state in config["hands"].items():
        shapes = shapes_world(load_geometry(ROOT, side=side), state["pose_wxyz"], state["q_rad"])
        gaps = {name: separation(v, *tube) for name, v in shapes.items()}
        shape = min(gaps, key=gaps.get)
        table = min(float(v[:, 2].min()-.3) for v in shapes.values())
        support = min(support_clearance_lower(v, low, high) for v in shapes.values() for low, high in supports)
        rows[side] = dict(rod_gap_lower_m=gaps[shape], table_clearance_m=table, support_gap_m=support,
                          limiting_rod_shape=shape, shape_count=len(shapes),
                          passed=gaps[shape] > .001 and table >= .0021 and support > 0.)
    return dict(scope="tripod_supported_reset_archived_rod_envelope_only", hands=rows,
                all_passed=all(row["passed"] for row in rows.values()), init_and_archived_sdk_samples=len(nodes),
                archived_samples_sha256=[row.sample_sha256 for row in samples], request_sha256=file_hash(REQUEST),
                native_reset_verified=False, grasp_qualified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived-settling-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    out = screen(args.archived_settling_receipt, *scene_request())
    with args.output.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
