#!/usr/bin/env python3
"""Replay Pinch v6 observed link motion on the native v7 Reset geometry (CPU)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from scripts.audit_dh116_pinch_initial_pose import load_geometry


def transplant_link_motion(origin, measured, target):
    poses = [np.asarray(rows, dtype=float) for rows in (origin, measured, target)]
    if (any(rows.ndim != 2 or rows.shape[1] != 7 or rows.shape != poses[0].shape
            or not len(rows) or not np.isfinite(rows).all() for rows in poses)
            or any(not np.allclose(np.linalg.norm(rows[:, 3:], axis=1), 1., rtol=0., atol=1e-5)
                   for rows in poses)):
        raise ValueError("complete finite matched SDK link poses with xyzw quaternions required")
    old, live, new = poses
    motion = Rotation.from_quat(live[:, 3:]) * Rotation.from_quat(old[:, 3:]).inv()
    orientation = motion * Rotation.from_quat(new[:, 3:])
    return np.column_stack((new[:, :3] + (live[:, :3] - old[:, :3]), orientation.as_quat()))


def _table_minimum(geometry, names, poses):
    layout = geometry["reference"]["reference_layout"]
    top = layout["table_center"][2] + layout["table_size"][2] / 2
    rot = Rotation.from_quat(poses[:, 3:]).as_matrix()
    values = []
    for kind in ("source_colliders", "cooked_pads"):
        for link, shape in geometry[kind].items():
            i = names.index(link)
            z_m = shape["vertices"] @ rot[i, 2] + poses[i, 2]
            values.append((float(z_m.min() - top), kind, link))
    return min(values)


def audit_rows(entries, new_reset, *, project):
    reset_rows = [item["payload"]["reset"] for item in entries
                  if item["operation"] == "owner_event" and item["payload"].get("kind") == "reset"]
    events = [item["payload"] for item in entries if item["operation"] == "delivered_exchange"]
    if (len(reset_rows) != 1 or not events
            or [row["physics_step"] for row in events] != list(range(3, 3+len(events)))):
        raise ValueError("one original Reset and all contiguous delivered SDK substeps required")
    old_reset, = reset_rows
    if any(reset["after_clock"]["physics_step"] != 2 or reset["after_clock"]["rod_tick"] != 0
           for reset in (old_reset, new_reset)):
        raise ValueError("SDK2 native Reset pose readbacks required")
    if old_reset["seed"]["sha256"] == new_reset["seed"]["sha256"]:
        raise ValueError("distinct recentered Pinch Reset required")

    sides = ("left", "right")
    models = {side: load_geometry(project, side=side) for side in sides}
    results = []
    for entry in events:
        step = entry["physics_step"]
        raw = entry["closure_geometry"]["raw"]
        clearance = entry["post_step_clearance"]
        clock = raw["identity"]
        if (clock["physics_step"] != step or clearance["physics_step"] != step
                or any(clock[k] != old_reset["after_clock"][k] for k in ("env_id", "episode_uid", "reset_id", "run_uuid"))
                or entry["closure_geometry"]["source_complete"] is not True or clearance["floor_m"] != CLEARANCE_M):
            raise ValueError("same original rod owner, SDK clock and table gate required")
        row = {"sdk_step": step}
        for i, side in enumerate(sides):
            model = models[side]
            old, live, new = old_reset["hands"][side], raw["hands"][side], new_reset["hands"][side]
            names = old["body_names"]
            if (any(hand["body_names"] != names or hand["link_paths"] != old["link_paths"] for hand in (live, new))
                    or entry["closure_geometry"]["sources"][side]["source_sha256"] != model["source_sha256"]):
                raise ValueError("same ordered original side-specific link geometry required")
            actual = np.asarray(live["link_poses_xyzw"])[0]
            transplanted = transplant_link_motion(old["body_poses_xyzw"][0], actual, new["body_poses_xyzw"][0])
            recovered, _, _ = _table_minimum(model, names, actual)
            observed = clearance["clearance_m"][0][i]
            if abs(recovered - observed) > 1e-6:
                raise ValueError("source/cooked FK replay must reproduce recorded floor within 1 micrometre")
            minimum, kind, link = _table_minimum(model, names, transplanted)
            row[side] = dict(original_measured_m=observed, recovered_original_m=recovered,
                             counterfactual_m=minimum, collider_source=kind, link=link,
                             margin_m=minimum-CLEARANCE_M)
        results.append(row)
    minima = {side: min([row[side]["margin_m"] for row in results]) for side in sides}
    return dict(checked_sdk_steps=len(results), rows=results, minimum_counterfactual_margin_m=minima,
                observed_motion_transfer_passed=min(minima.values()) >= 0., new_physics_steps=0,
                native_tracking_verified=False, ppo_updates=0, teacher_admission=False)


def _checked_bytes(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("pinned input bytes differ")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--journal-sha256", required=True)
    parser.add_argument("--recentered-reset", type=Path, required=True)
    parser.add_argument("--recentered-reset-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = [decode_owned(json.loads(line)) for line in _checked_bytes(args.journal, args.journal_sha256).splitlines()]
    reset = decode_owned(json.loads(_checked_bytes(args.recentered_reset, args.recentered_reset_sha256)))
    result = audit_rows(entries, reset["readback"]["completed_reset_receipt"],
                        project=Path(__file__).resolve().parents[1])
    result.update(journal_sha256=args.journal_sha256, recentered_reset_sha256=args.recentered_reset_sha256)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["observed_motion_transfer_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
