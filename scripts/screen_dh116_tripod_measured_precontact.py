#!/usr/bin/env python3
"""Screen a hands-clear Tripod continuation from sealed native link and rod poses."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from scripts.screen_dh116_tripod_supported_reset import ROOT, archived_tube, separation
from scripts.verify_dh116_power_shared_rod import native, sources

CONTRACT = ROOT / "docs/contracts/dh116_teacher_tripod_waypoint_native_v1.json"
REQUEST = ROOT / "configs/dh116_tasks/flexible_tripod_measured_precontact_v1.json"
SIDES = ("left", "right")
PADS = ("finger14_link", "finger23_link", "finger33_link")
SDK = 432
SAMPLES = 17
ROD_RESERVE_M = .00025


def archive(receipt, *, contract=CONTRACT, sdk=SDK, task="tripod", expected=None):
    if task not in ("tripod", "pinch"):
        raise ValueError("owned supported distal grasp archive required")
    expected = json.loads(contract.read_bytes())["actual"] if expected is None else expected
    outer = sources.checked_json(receipt, expected["outer_sha256"])
    raw = receipt.parent / "native"
    marker = raw / "app_close_returned.json"
    close = sources.checked_json(marker, outer["artifact_sha256"][marker.name])
    if (outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or outer["forced_exit"] or outer["exceeded_budget"]
            or close["app_close_returned"] is not True):
        raise ValueError("sealed naturally closed supported distal run required")
    row_paths = [raw / f"{task}_control_{tick}.json" for tick in range(sdk - 8, sdk + 1)]
    rows = []
    for tick, path in enumerate(row_paths, sdk - 8):
        row = sources.checked_json(path, outer["artifact_sha256"][path.name])
        if (row["physics_step"] != tick or row["guard_failure"] is not None
                or row["sdk_hand_geometry"]["physics_step"] != tick
                or row["rod"]["physics_step"] != tick or row["rod"]["rod_tick"] != (tick - 2) * 12):
            raise ValueError("supported distal link, rod and action source clocks differ")
        rows.append(row)
    support_path = raw / f"{task}_support_scene.json"
    supports = sources.checked_json(support_path, outer["artifact_sha256"][support_path.name])
    if native.sha(row_paths[-1]) != expected["last_row_sha256"]:
        raise ValueError("terminal row is not the archived distal boundary")
    return rows, supports["supports"], [contract, receipt, marker, *row_paths, support_path]


def check_route(samples, *, pad_count=3):
    def row_passed(row):
        gaps = np.asarray(row["pad_gaps_m"], dtype=float)
        directions = np.asarray(row["opposition"], dtype=float)
        return (row["shape_count"] == 22
                and np.isfinite([row["rod_clearance_m"], row["support_clearance_m"], row["table_clearance_m"]]).all()
                and row["rod_clearance_m"] > ROD_RESERVE_M
                and row["support_clearance_m"] > 0.
                and row["table_clearance_m"] >= CLEARANCE_M
                and gaps.shape == (pad_count,) and np.isfinite(gaps).all() and (gaps > ROD_RESERVE_M).all()
                and directions.shape == (pad_count - 1,) and np.isfinite(directions).all())
    return bool(len(samples) == SAMPLES and all(row_passed(row) for row in samples)
                and min(samples[-1]["pad_gaps_m"]) < min(samples[0]["pad_gaps_m"])
                and max(samples[-1]["opposition"]) < -.2)


def as_matrix(position, quat):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
    matrix[:3, 3] = position
    return matrix


def measure_side(side, raw, nodes, tube, boxes, grids, goal, table_top, *,
                 pads=PADS, max_delta_m=.002, max_delta_rad=.025, contact_pads=()):
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
    from scripts.prepare_dh116_pinch_closure_margin import fk_poses, sample_grid

    hand = raw["action_readback"]["post"]["hands"][side]
    indices = hand["active_indices"]
    if [hand["joint_names"][i] for i in indices] != list(ACTIVE):
        raise ValueError("original Tripod active joint order required")
    active = hand["q"][0, indices]
    model = load_geometry(ROOT, side=side)
    actual = {key.split(".")[1]: as_matrix(*pose)
              for key, pose in raw["sdk_hand_geometry"]["poses"].items()
              if key.startswith(side + ".")}
    palm = actual["base_link"]
    start = np.r_[palm[:3, 3], Rotation.from_matrix(palm[:3, :3]).as_quat()[[3, 0, 1, 2]]]
    start_fk = fk_poses(model, start, active)
    corrections = {name: np.linalg.solve(start_fk[name], body) for name, body in actual.items()}
    shapes = [(group, name, shape["vertices"]) for group in ("source_colliders", "cooked_pads")
              for name, shape in model[group].items()]
    samples = []
    waypoints = goal if isinstance(goal, list) else [goal]
    chain = [(start, active)]
    for waypoint in waypoints:
        previous_pose, previous_q = chain[-1]
        target = np.asarray(waypoint["pose_wxyz"], dtype=float)
        goal_q = np.asarray(waypoint["active_q_rad"], dtype=float)
        if (target.shape != (7,) or not np.isfinite(target).all()
                or abs(np.linalg.norm(target[3:]) - 1.) > 1e-8
                or np.linalg.norm(target[:3] - previous_pose[:3]) > max_delta_m
                or goal_q.shape != (6,) or np.any(np.abs(goal_q - previous_q) > max_delta_rad)):
            raise ValueError("small finite measured precontact waypoint required")
        expand(goal_q)
        chain.append((target, goal_q))
    for a, b in zip(chain[:-1], chain[1:], strict=True):
        p0, q0 = a
        p1, q1 = b
        slerp = Slerp([0., 1.], Rotation.from_quat(np.stack((p0, p1))[:, [4, 5, 6, 3]]))
        for t in np.linspace(0., 1., SAMPLES):
            pose = np.r_[(1. - t) * p0[:3] + t * p1[:3], slerp(t).as_quat()[[3, 0, 1, 2]]]
            poses = fk_poses(model, pose, (1. - t) * q0 + t * q1)
            for name, correction in corrections.items():
                poses[name] = poses[name] @ correction
            world = {group + ":" + name: v @ poses[name][:3, :3].T + poses[name][:3, 3]
                     for group, name, v in shapes}
            distances = {label: separation(points, *tube) for label, points in world.items()}
            limiting_shape = min(distances, key=distances.get)
            rod = distances[limiting_shape]
            noncontact = min(gap for label, gap in distances.items()
                             if label.split(":", 1)[1] not in contact_pads)
            support = min(support_clearance_lower(points, low, high) for points in world.values() for low, high in boxes)
            table = min(float(points[:, 2].min() - table_top) for points in world.values())
            gaps, normals, node_ids = [], [], []
            for name in pads:
                world_pose = poses[name]
                local = (nodes - world_pose[:3, 3]) @ world_pose[:3, :3]
                phi, gradient, valid = sample_grid(grids[f"{side}.{name}"], local, dtype=np.float64)
                if not valid.any():
                    raise ValueError("Tripod pad has no original SDF lattice witness")
                i = int(np.nanargmin(phi))
                n = world_pose[:3, :3] @ gradient[i]
                norm = float(np.linalg.norm(n))
                if norm <= 0.:
                    raise ValueError("Tripod pad has no SDF normal witness")
                normals.append(n / norm)
                gaps.append(float(phi[i] - .005))
                node_ids.append(i)
            samples.append(dict(shape_count=len(world), rod_clearance_m=rod, support_clearance_m=support,
                                table_clearance_m=table, limiting_shape=limiting_shape, pad_gaps_m=gaps, node_ids=node_ids,
                                opposition=[float(normals[0] @ n) for n in normals[1:]]))
            if contact_pads:
                samples[-1]["noncontact_rod_clearance_m"] = noncontact
                samples[-1]["contact_rod_clearance_m"] = min(
                    gap for label, gap in distances.items() if label.split(":", 1)[1] in contact_pads)
    return samples


def screen(receipt):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene

    rows, supports, files = archive(receipt)
    request = json.loads(REQUEST.read_bytes())
    if request["version"] != "dh116-tripod-measured-precontact-v1" or set(request["goals"]) != set(SIDES):
        raise ValueError("separate left and right Tripod precontact targets required")
    rows = [decode_owned(row) for row in rows]
    samples = np.stack([row["rod"]["node_positions"][0] for row in rows])
    tube = archived_tube(samples)
    boxes = [(np.asarray(box["center_m"]) - np.asarray(box["size_m"]) / 2,
              np.asarray(box["center_m"]) + np.asarray(box["size_m"]) / 2) for box in supports]
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task

    table = load_flexible_grasp_task(ROOT / "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json", ROOT)["scene"]
    table_top = table["table_center_m"][2] + .5 * table["table_size_m"][2]
    grids = scene.load_sdf_arrays(scene.prepare_scene(ROOT))
    measurements = {side: measure_side(side, rows[-1], samples[-1], tube, boxes, grids, request["goals"][side], table_top)
                    for side in SIDES}
    return dict(version=request["version"], request_sha256=native.sha(REQUEST),
                outer_sha256=native.sha(receipt), last_row_sha256=native.sha(files[-2]),
                before_sdk=SDK, route_sample_count=SAMPLES, samples=measurements,
                all_passed=all(check_route(v) for v in measurements.values()),
                native_contact_verified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = screen(args.receipt)
    native.write_new(args.output, result)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
