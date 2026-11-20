#!/usr/bin/env python3
"""Evaluate bounded commands against both actual archived Pinch hand boundaries."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_ANGULAR
from go2_pvcnn.control.dh116.tabletop_native_teacher_swept_projection import WristSweep, preaction_target
from scripts.audit_dh116_pinch_initial_pose import load_geometry, pose_matrix
from scripts.audit_dh116_pinch_recentered_tracking import _checked_bytes
from scripts.audit_dh116_teacher_window_wall import _windows


SIDES = ("left", "right")


def _bound_model(project, side):
    geometry = load_geometry(project, side=side)
    layout = geometry["reference"]["reference_layout"]
    top = layout["table_center"][2] + layout["table_size"][2] / 2
    fk = geometry["fk"]
    source = geometry["source_colliders"]
    vertices = {"source:" + name: row["vertices"] for name, row in source.items()}
    vertices.update({"cooked:" + name: row["vertices"] for name, row in geometry["cooked_pads"].items()})
    paths = {key: source[key.split(":")[1]]["source"]["body_path"] for key in vertices}
    model = WristSweep(root=fk.root, edges=fk.edges, vertices=vertices,
                       body_paths=paths, table_top_m=top)
    return model, geometry["source_sha256"]


def _frames(model, names, poses):
    if len(names) != len(set(names)) or not {"base_link"} <= set(names):
        raise ValueError("complete unique native link names required")
    lookup = dict(zip(names, np.asarray(poses, dtype=float)[0]))
    paths = {name: path.rsplit("/", 1)[-1] for name, path in model.paths.items()}
    mapped = {key: pose_matrix(lookup[name][[0, 1, 2, 6, 3, 4, 5]]) for key, name in paths.items()}
    return mapped, np.asarray(lookup["base_link"][:3], dtype=float)


def _decision(model, frames, base_xyz, readback, supplied, nominal_xyz, nominal_angles,
              prior_xyz, prior_angles, seed, reserve):
    joint_names = readback["joint_names"]
    if [joint_names[i] for i in readback["active_indices"]] != list(ACTIVE):
        raise ValueError("unmodified canonical active joint readback required")
    measured = readback["q"][0]
    hand_q = measured[[joint_names.index(name) for name in JOINTS]]
    wrist_q = measured[[joint_names.index(name) for name in WRIST_ANGULAR]]
    anchor = readback["native_anchor_pose_w"][0]
    rotation = Rotation.from_quat(np.roll(anchor[3:], -1)).as_matrix()
    boundary = model.boundary(measured_link_frames=frames, anchor_rotation=rotation,
                           start_xyz=base_xyz, start_angles=wrist_q,
                           start_q=hand_q, final_q=expand(supplied.astype(np.float32)))
    original = boundary.prove(nominal_xyz, nominal_angles)
    candidate = preaction_target(nominal_xyz=nominal_xyz, nominal_angles=nominal_angles,
                                 last_xyz=prior_xyz, last_angles=prior_angles,
                                 measured_angles=wrist_q, seed_xyz=seed,
                                 certify=boundary.prove, terminal_reserve_m=reserve)
    if candidate["accepted"]:
        candidate["xyz"] = candidate["xyz"].tolist()
        candidate["angles"] = candidate["angles"].tolist()
    result = dict(original=original, proposed=candidate)
    if not candidate["accepted"]:
        result["best_rate_limited_endpoint_upper_m"] = boundary.endpoint_upper_bound(prior_xyz, prior_angles)
    return result


def audit(project, journal, digest, reserve):
    events = [json.loads(line) for line in _checked_bytes(journal, digest).splitlines()]
    models, sources = zip(*[_bound_model(project, side) for side in SIDES])
    reset, = [decode_owned(row["payload"]) for row in events if row["operation"] == "owner_event"
              and row["payload"]["kind"] == "reset"]
    seed = np.asarray(reset["reset"]["seed"]["reset_root_pose_w"], dtype=float)[:, :3]
    last_xyz, last_angles = seed, np.zeros((2, 3))
    initial = {row["payload"]["side"]: row["payload"]["evidence"] for row in events
               if row["operation"] == "initial_geometry"}
    if set(initial) != set(SIDES) or any(row["identity"]["sdk_step"] != 2 for row in initial.values()):
        raise ValueError("both original Reset link frame snapshots required")
    frames = [_frames(model, initial[side]["actual_body_names"], initial[side]["actual_link_poses_xyzw"])
              for side, model in zip(SIDES, models)]
    previous = None
    results = []
    for window in _windows(events):
        for packet in window["physical"]:
            row = decode_owned(packet)
            readback, projection = row["action_readback"], row["wrist_projection"]
            identity = readback["pre"]["identity"][0]
            if identity["physics_step"] + 1 != row["physics_step"]:
                raise ValueError("nonadjacent SDK target packet")
            if previous is not None:
                fixed = ("env_id", "episode_uid", "physics_step", "reset_id", "rod_tick", "run_uuid",
                         "rod_uid", "solver_time_s", "timestamp_s")
                if any(identity[key] != previous[key] for key in fixed):
                    raise ValueError("archived measured link and request boundaries differ")
            nominal_xyz = projection["projected_root_poses"][0, :, :3]
            nominal_angles = projection["projected_angles"][0]
            fingers = readback["request"]["requested_finger_targets_rad"][0]
            output = {}
            for idx, (side, model) in enumerate(zip(SIDES, models)):
                measured_links, base_xyz = frames[idx]
                supplied = fingers[6 * idx:6 * (idx + 1)]
                backend = readback["submission"]["expected_backend_targets"][side]["values"][0, :6]
                if not np.array_equal(supplied.astype(np.float32), backend):
                    raise ValueError("SDK finger target differs from the original request")
                output[side] = _decision(model, measured_links, base_xyz, readback["pre"]["hands"][side],
                    supplied, nominal_xyz[idx], nominal_angles[idx], last_xyz[idx], last_angles[idx], seed[idx], reserve)
                observed = row["closure_geometry"]["raw"]["hands"][side]
                frames[idx] = _frames(model, observed["body_names"], observed["link_poses_xyzw"])
            results.append(dict(sdk_step=row["physics_step"], **output))
            last_xyz, last_angles = nominal_xyz, nominal_angles
            previous = readback["post"]["identity"][0]
    return dict(schema="dh116-archived-wrist-sweep-v1", journal_sha256=digest,
                collision_sources=dict(zip(SIDES, sources)), shape_counts=[len(model.paths) for model in models],
                rows=results, terminal_reserve_m=reserve, actual_isaac_steps=0,
                runtime_projection_enabled=False, native_qualification=False, teacher_admission=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--journal", type=Path, required=True)
    p.add_argument("--journal-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--terminal-reserve-m", type=float, default=0.)
    args = p.parse_args(argv)
    evidence = audit(args.project, args.journal, args.journal_sha256, args.terminal_reserve_m)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(evidence, target, indent=2, allow_nan=False)
        target.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
