#!/usr/bin/env python3
"""Prepare a left Pinch Reset proposal against three source-bound native replays."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from scripts.audit_dh116_pinch_initial_pose import checked_json, load_geometry
from scripts.prepare_dh116_flexible_style_poses import verify_candidate_path
from scripts.prepare_dh116_paired_pinch_reset import (
    LEFT, LEFT_SHA, independent_sweep_gap, left_clearance_candidate, make_joint_seed,
)
from scripts.prepare_dh116_pinch_v10_clearance_candidate import _minimum_replay, _read_motion
from scripts.refine_dh116_right_pinch_geometry import measure_strict, supported_geometry
from scripts.validate_dh116_right_pinch_runtime_clearance import TASK, minimum_clearance, qualifies


PREVIOUS_RESET_SHA = "df093cce59882cdd23f28799bddade29e2cc7681d37374b1b2d17242c90bcc80"
V11_RESET_SHA = "2aacc88ef2cafa0a8d1bc990396546bdcad5016f99aba80fc1c5187a7324de1c"
V11_PAIR_SHA = "48d007fcdffa16cdc0fd3a1c711a708067de0d5cd584c8e670afe83d04927cfb"
PINCH_ARCHIVES = {
    "v9": ("24cf6d9c3bdc574ec09cb80c26b4f7cf6275af49af25a411b94294c257e17593", 48, PREVIOUS_RESET_SHA),
    "v10": ("fa7e6300ac5a48804a00b4f9e0b30bbda010e88168bf2114cb0850d18bbdb878", 19, PREVIOUS_RESET_SHA),
    "v11": ("002f76be32d57ea269e3f9e737cabe0de0a61aedafbede97b2f5d966df23119f", 72, V11_RESET_SHA),
}
LEFT_RESERVE_M = .000175
WORLD_XY_ROTATION_RAD = (-.0012, -.0004)


def _left_seed(project):
    reference = checked_json(project / LEFT, LEFT_SHA)
    seed = left_clearance_candidate(reference, LEFT_RESERVE_M, root_x_m=.000085, thumb_seed_rad=.00275)
    pose = np.asarray(seed["base_pose_wxyz"], dtype=float)
    rotated = (Rotation.from_rotvec([*WORLD_XY_ROTATION_RAD, 0.])
               * Rotation.from_quat(np.roll(pose[3:], -1)))
    pose[3:] = np.roll(rotated.as_quat(), 1)
    seed["base_pose_wxyz"] = pose.tolist()
    seed["candidate_id"] = "pinch-v12-left-guard-replay"
    seed["local_change"]["world_rotation_vector_rad"] = [*WORLD_XY_ROTATION_RAD, 0.]
    return seed


def prepare_pair(previous_pair, ledgers, *, project):
    if (set(ledgers) != set(PINCH_ARCHIVES)
            or previous_pair["reset_seed"]["sha256"] != V11_RESET_SHA
            or previous_pair["passes_static_policy_clearance"] is not True
            or any(previous_pair[name] is not False for name in
                   ("native_qualified", "grasp_qualified", "teacher_admission"))):
        raise ValueError("original v11 CPU pair and all three original journals required")

    task = load_flexible_grasp_task(project / TASK, project)
    right_geometry, rod = supported_geometry(load_geometry(project, side="right"), task)
    if rod != previous_pair["rod_seed"]:
        raise ValueError("original supported free rod differs")
    left_geometry = load_geometry(project, side="left")
    layout = left_geometry["reference"]["reference_layout"]
    right_layout = right_geometry["reference"]["reference_layout"]
    if (layout["wire_points"] != rod["original_nodes_m"]
            or any(layout[key] != right_layout[key] for key in ("table_center", "table_size"))
            or any(left_geometry["old_candidate"]["rod"][key] != rod[key] for key in
                   ("radius_m", "pinned_node_ids", "gravity_world_m_s2"))):
        raise ValueError("left source must use the same original table and free rod")
    layout["wire_points"] = deepcopy(rod["proposed_nodes_m"])

    prepared = {}
    for side, geometry in (("left", left_geometry), ("right", right_geometry)):
        previous = previous_pair[side]
        original = previous["candidate"]
        source = original["source_sha256"] if side == "left" else previous["source_sha256"]
        if not qualifies(original, previous["path"]) or source != geometry["source_sha256"]:
            raise ValueError("same independent hand sources, original geometry and free rod required")
        motions = {
            run: _read_motion(ledgers[run], digest, expected_steps=count, reset_seed_sha=reset_sha,
                              geometry=geometry, side=side)
            for run, (digest, count, reset_sha) in PINCH_ARCHIVES.items()
        }
        current = _left_seed(project) if side == "left" else original
        baseline = {run: _minimum_replay(original, geometry, motion) for run, motion in motions.items()}
        replayed = (baseline if side == "right" else
                    {run: _minimum_replay(current, geometry, motion) for run, motion in motions.items()})
        minimum = min(min(row["minimum_actual_m"], row["minimum_target_m"]) for row in replayed.values())
        if not math.isfinite(minimum) or minimum < CLEARANCE_M:
            raise ValueError(f"{side} fails original physical clearance in a replayed motion")
        if side == "left":
            current = measure_strict(geometry, current)
            path = verify_candidate_path(geometry, current)
            if (not qualifies(current, path) or minimum_clearance(path) < CLEARANCE_M + LEFT_RESERVE_M):
                raise ValueError("left candidate fails strict endpoints, continuous geometry or declared reserve")
        else:
            path = previous["path"]
        prepared[side] = dict(candidate=current, path=path, baseline=baseline, replayed=replayed, minimum=minimum)

    left, right = prepared["left"], prepared["right"]
    pair = deepcopy(previous_pair)
    gap = independent_sweep_gap(left["path"]["continuous_hand_y_bounds_m"],
                                right["path"]["continuous_hand_y_bounds_m"])
    pair.update(left=dict(candidate=left["candidate"], path=left["path"]),
                reset_seed=make_joint_seed(left["candidate"], right["candidate"]),
                left_extra_clearance_m=LEFT_RESERVE_M,
                left_continuous_clearance_m=minimum_clearance(left["path"]),
                independent_hand_y_gap_m=gap)
    evidence = dict(
        journal_sha256={run: sha for run, (sha, _, _) in PINCH_ARCHIVES.items()},
        baseline={side: row["baseline"] for side, row in prepared.items()},
        replayed={side: row["replayed"] for side, row in prepared.items()},
        minimum_replayed_m={side: row["minimum"] for side, row in prepared.items()},
        contact_or_tracking_validated=False, native_qualification=False, teacher_admission=False,
    )
    return pair, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pair", *PINCH_ARCHIVES, "candidate-output", "report-output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    outputs = (args.candidate_output, args.report_output)
    if outputs[0].resolve() == outputs[1].resolve() or any(path.exists() for path in outputs):
        parser.error("distinct new candidate and report outputs required")
    pair, evidence = prepare_pair(
        checked_json(args.pair, V11_PAIR_SHA), {run: getattr(args, run) for run in PINCH_ARCHIVES},
        project=Path(__file__).resolve().parents[1])
    for path, payload in zip(outputs, (pair, evidence)):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
