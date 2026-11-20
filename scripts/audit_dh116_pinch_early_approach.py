#!/usr/bin/env python3
"""Screen simultaneous opposed Pinch geometry on a retained native journal."""
import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import DT
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_ANGULAR
from go2_pvcnn.control.dh116.tabletop_native_teacher_swept_projection import (
    ANGULAR_STEP_RAD, LINEAR_STEP_M, _finite,
)
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts.audit_dh116_pinch_wrist_sweep import SIDES, _bound_model, _frames
from scripts.audit_dh116_tripod_fixed_root_path import certify_path
from scripts.audit_dh116_pinch_recentered_tracking import _checked_bytes
from scripts.audit_dh116_teacher_window_wall import _windows
from scripts.prepare_dh116_pinch_closure_margin import MARGIN, REQUIRED, sample_grid


PADS = tuple("cooked:" + name for name in REQUIRED)
OPPOSITION_COS = -.25
# Search directions need only have norm <= 1 to yield a distance lower bound.
ANGLES = np.arange(64) * 2 * np.pi / 64
NORMALS = np.stack((np.cos(ANGLES), np.zeros_like(ANGLES), np.sin(ANGLES)), axis=1)
NORMALS /= np.linalg.norm(NORMALS, axis=1)[:, None] + 1e-12


def minimum_sdk_steps(delta):
    delta = _finite(delta, (8,))
    rate_steps = max(np.linalg.norm(delta[:3]) / LINEAR_STEP_M,
                     np.max(abs(delta[3:5])) / (ACTION_CONTRACT["joint_rate_rad_s"] * DT),
                     np.linalg.norm(delta[5:]) / ANGULAR_STEP_RAD)
    return math.ceil(rate_steps - 1e-12)


def bilateral_summary(rows):
    """Reduce four gaps within a row before comparing recorded SDK steps."""
    best = []
    opposed = []
    admissible = []
    for row in rows:
        both = [row[side] for side in SIDES]
        gaps = [gap for hand in both for gap in hand["gaps_m"]]
        if len(gaps) != 4:
            raise ValueError("two independent pad measurements per hand required")
        if any(gap is None or not np.isfinite(gap) for gap in gaps):
            continue
        frame_gap = max(gaps)
        best.append((frame_gap, row["sdk_step"]))
        if frame_gap > -MARGIN or any(hand["opposition_cos"] is None
                or not -1. <= hand["opposition_cos"] <= OPPOSITION_COS for hand in both):
            continue
        opposed.append(row["sdk_step"])
        if row.get("interhand_sweep_gap_m", 0.) > 0. and all(
                hand.get("table", {}).get("proved") is True
                and hand.get("non_target_rod", {}).get("proved") is True
                and len(hand.get("initial_gaps_m", ())) == 2
                and all(gap is not None and np.isfinite(gap) and gap >= 0.
                        for gap in hand["initial_gaps_m"])
                for hand in both):
            admissible.append(row["sdk_step"])
    gap, tick = min(best, default=(None, None))
    return dict(best_bilateral_gap_m=gap, best_bilateral_gap_sdk_step=tick,
                bilateral_opposed_target_ticks=opposed, bilateral_geometry_ticks=admissible,
                accepted_for_native=False)


def _contact(grid, transform, nodes, radius):
    local = (nodes.astype(np.float32) - transform[:3, 3].astype(np.float32)) @ transform[:3, :3].astype(np.float32)
    phi, gradient, inside = sample_grid(grid, local)
    if not inside.any():
        return None, None, None
    node = int(np.nanargmin(phi))
    normal = transform[:3, :3] @ gradient[node]
    norm = np.linalg.norm(normal)
    return float(phi[node] - np.float32(radius)), node, normal / norm if norm > 0 else None


def _non_target_certificate(model, poses, travel, nodes, radius):
    shapes = tuple(name for name in model.paths if name.split(":", 1)[1] not in REQUIRED)
    projected_rod = (nodes @ NORMALS.T).max(axis=0)

    def clearance(t):
        values = {}
        for name, transform in poses(t).items():
            if name in shapes:
                points = model.vertices[name] @ transform[:3, :3].T + transform[:3, 3]
                separation = (points @ NORMALS.T).min(axis=0) - projected_rod
                values[name] = float(separation.max() - radius)
        return values

    proof = certify_path(clearance, {name: travel[name] for name in shapes},
                         dict.fromkeys(shapes, 0.), max_depth=8, max_samples=129)
    return dict(proved=proof["proved"], status=proof["status"],
                sample_count=len(proof["samples"]),
                start_lower_m=min(proof["samples"][0.].values()),
                endpoint_lower_m=min(proof["samples"][1.].values()),
                violations=proof["violations"])


def _same_clock(row):
    readback, rod = row["action_readback"]["post"], row["rod"]
    identity, = readback["identity"]
    raw = row["closure_geometry"]["raw"]
    keys = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid", "physics_step",
            "rod_tick", "timestamp_s", "solver_time_s")
    if (raw["identity"] != identity
            or identity["clock_valid"] is not True or rod["clock_valid"] is not True
            or identity["physics_step"] != row["physics_step"]
            or any(identity[k] != rod[k] for k in keys)
            or not np.array_equal(raw["rod_nodes_m"], rod["node_positions"][0])):
        raise ValueError("owned link and rod snapshots must share the same native boundary")
    return {key: identity[key] for key in keys}


def _evaluate(model, row, side, grids, delta):
    raw = row["closure_geometry"]["raw"]["hands"][side]
    measured, xyz = _frames(model, raw["body_names"], raw["link_poses_xyzw"])
    post = row["action_readback"]["post"]["hands"][side]
    names = post["joint_names"]
    readback, = post["q"]
    q = readback[[names.index(name) for name in JOINTS]]
    wrist = readback[[names.index(name) for name in WRIST_ANGULAR]]
    if [names[i] for i in post["active_indices"]] != list(ACTIVE):
        raise ValueError("original active joint identity required")
    active = readback[post["active_indices"]].astype(float)
    target = active.copy()
    target[1:3] += delta[3:5]
    low, high = active_limits()
    if np.any(target < low) or np.any(target > high) or np.any(abs(wrist + delta[5:]) > np.pi / 2):
        raise ValueError("candidate leaves original joint or wrist angular limits")
    dq = expand(target.astype(np.float32)) - q
    rod = _finite(row["rod"]["node_positions"][0], (41, 3))
    radius = row["closure_geometry"]["rod_radius_m"]
    if not np.isfinite(radius) or radius <= 0.:
        raise ValueError("positive finite rod radius required")
    anchor, = post["native_anchor_pose_w"]
    rotation = Rotation.from_quat(np.roll(anchor[3:], -1)).as_matrix()
    frames = model._frames(rotation, xyz, wrist, q)
    corrections = {name: measured[name] @ np.linalg.inv(frames[path])
                   for name, path in model.paths.items()}

    def poses(t):
        state = model._frames(rotation, xyz + t * delta[:3], wrist + t * delta[5:], q + t * dq)
        return {name: corrections[name] @ state[path] for name, path in model.paths.items()}

    boundary = model.boundary(measured_link_frames=measured, anchor_rotation=rotation,
                              start_xyz=xyz, start_angles=wrist, start_q=q, final_q=q + dq)
    table = boundary.prove(xyz + delta[:3], wrist + delta[5:])
    travel = {key: float(np.linalg.norm(delta[:3]) + abs(delta[5:]).sum() * model.wrist_radii[key]
                         + model.joint_radii[key] @ abs(dq)) for key in model.paths}
    separate = _non_target_certificate(model, poses, travel, rod, radius)
    bounds = np.array([
        ((model.vertices[key] @ value[1, :3]).min() + value[1, 3] - travel[key],
         (model.vertices[key] @ value[1, :3]).max() + value[1, 3] + travel[key])
        for key, value in measured.items()
    ])
    initial = [_contact(grids[name], measured[name], rod, radius)[0] for name in PADS]
    final = poses(1.)
    result = [_contact(grids[name], final[name], rod, radius) for name in PADS]
    dot = (float(np.clip(result[0][2] @ result[1][2], -1., 1.))
           if all(normal is not None for _, _, normal in result) else None)
    return dict(gaps_m=[gap for gap, _, _ in result], minimum_node_ids=[node for _, node, _ in result],
                opposition_cos=dot, table=table, non_target_rod=separate, initial_gaps_m=initial,
                swept_y_bounds_m=[float(bounds[:, 0].min()), float(bounds[:, 1].max())])


def audit(project, journal, journal_sha256, candidate):
    raw = [json.loads(line) for line in _checked_bytes(journal, journal_sha256).splitlines()]
    wanted = candidate["sdk_steps"]
    budget = candidate["hypothetical_sdk_step_budget"]
    if (type(budget) is not int or not 1 <= budget <= 240
            or not wanted or any(type(tick) is not int for tick in wanted)
            or sorted(set(wanted)) != wanted):
        raise ValueError("unique ordered measured SDK steps and bounded diagnostic budget required")
    commands = {side: _finite(candidate["relative_targets"][side], (8,)) for side in SIDES}
    horizon = {side: minimum_sdk_steps(delta) for side, delta in commands.items()}
    if max(horizon.values()) > budget:
        raise ValueError("candidate exceeds the original translation/rotation/digit slew budget")
    models, sources = zip(*[_bound_model(project, side) for side in SIDES])
    packet = {row["physics_step"]: row for w in _windows(raw)
              for row in map(decode_owned, w["physical"]) if row["physics_step"] in wanted}
    if set(packet) != set(wanted):
        raise ValueError("all requested SDK steps must exist in the retained journal")
    grids = {side: {} for side in SIDES}
    grid_locks = {}
    rows = []
    for tick in wanted:
        row = packet[tick]
        result = dict(sdk_step=tick, identity=_same_clock(row))
        for model, side, source in zip(models, SIDES, sources):
            for pad, key in zip(REQUIRED, PADS):
                full = f"{side}.{pad}"
                capture = next(value for path, value in source.items() if path.endswith(f"/{side}_{pad}.json"))
                binding = row["rod"]["mesh_bindings"][full]
                hand = row["closure_geometry"]["raw"]["hands"][side]
                index = hand["body_names"].index(pad)
                if (binding["capture_sha256"] != capture or binding["side"] != side
                        or binding["body_name"] != pad or binding["body_index"] != index
                        or binding["prim_path"] != hand["link_paths"][0][index]
                        or binding["frame"] != "PhysX_link_origin" or binding["unit"] != "m"):
                    raise ValueError("locked cooked collision and SDF bindings must agree")
                lock = (binding["sdf_path"], binding["sdf_sha256"])
                if full not in grid_locks:
                    with np.load(io.BytesIO(_checked_bytes(Path(lock[0]), lock[1])), allow_pickle=False) as npz:
                        grids[side][key] = {name: npz[name].copy() for name in npz.files}
                    grid_locks[full] = lock
                if lock != grid_locks[full]:
                    raise ValueError("stable signed-distance source required across retained SDK rows")
            result[side] = _evaluate(model, row, side, grids[side], commands[side])
        left, right = [result[side]["swept_y_bounds_m"] for side in SIDES]
        result["interhand_sweep_gap_m"] = max(left[0], right[0]) - min(left[1], right[1])
        rows.append(result)
    return dict(schema="dh116-pinch-early-approach-replay-v1", journal_sha256=journal_sha256,
                hypothetical_sdk_step_budget=budget, lower_bound_sdk_steps=horizon,
                ordered_pad_names=list(REQUIRED), candidate=candidate, rows=rows,
                sdf_sha256={k: lock[1] for k, lock in grid_locks.items()},
                summary=bilateral_summary(rows), actual_isaac_steps=0, ppo_updates=0,
                teacher_admission=False, native_qualification=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "journal", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--journal-sha256", required=True)
    args = parser.parse_args(argv)
    result = audit(args.project, args.journal, args.journal_sha256, json.loads(args.candidate.read_text()))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
