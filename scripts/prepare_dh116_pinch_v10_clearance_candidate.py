#!/usr/bin/env python3
"""Prepare a right Pinch Reset candidate from two pinned motion replays."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from scripts.audit_dh116_pinch_dynamic_fk import full_named_poses, keep_reference_mimic_residual
from scripts.audit_dh116_pinch_initial_pose import checked_json, load_geometry, pose_matrix
from scripts.audit_dh116_pinch_recentered_tracking import _checked_bytes
from scripts.prepare_dh116_flexible_style_poses import verify_candidate_path
from scripts.prepare_dh116_paired_pinch_reset import independent_sweep_gap, make_joint_seed
from scripts.prepare_dh116_pinch_v8_clearance_candidate import measured_pose, named_joint_values
from scripts.refine_dh116_right_pinch_geometry import measure_strict, supported_geometry
from scripts.validate_dh116_right_pinch_runtime_clearance import TASK, minimum_clearance, qualifies

RUNS = {"v9": 48, "v10": 19}
RIGHT_CHANGE = dict(root_z_m=.000175, root_x_m=-.00005, world_pitch_rad=-.001, thumb_seed_rad=.0005)


def _read_motion(path, expected_sha, *, expected_steps, reset_seed_sha, geometry, side="right"):
    raw = _checked_bytes(path, expected_sha)
    rows = [decode_owned(json.loads(line)) for line in raw.splitlines()]
    if [row["sequence"] for row in rows] != list(range(len(rows))):
        raise ValueError("complete original journal sequence required")
    resets = [row["payload"]["reset"] for row in rows if row["operation"] == "owner_event"
              and row["payload"]["kind"] == "reset"]
    exchanges = [row["payload"] for row in rows if row["operation"] == "delivered_exchange"]
    if (len(resets) != 1 or len(exchanges) != expected_steps
            or [row["physics_step"] for row in exchanges] != list(range(3, 3 + expected_steps))):
        raise ValueError("one real Reset and every recorded physical tick required")
    reset, = resets
    clock = reset["after_clock"]
    if clock["physics_step"] != 2 or reset["seed"]["sha256"] != reset_seed_sha:
        raise ValueError("motion must start from its pinned Pinch Reset")
    hand = reset["hands"][side]
    q0 = named_joint_values(hand, "q", JOINTS)
    root_inv = np.linalg.inv(measured_pose(hand, "body_poses_xyzw", "base_link"))
    motion = []
    for exchange in exchanges:
        packet = exchange["closure_geometry"]
        source = packet["raw"]
        identity = source["identity"]
        if (packet["source_complete"] is not True
                or packet["sources"][side]["source_sha256"] != geometry["source_sha256"]
                or exchange["post_step_clearance"]["floor_m"] != CLEARANCE_M
                or identity["physics_step"] != exchange["physics_step"]
                or any(identity[key] != clock[key] for key in ("env_id", "episode_uid", "reset_id", "run_uuid", "rod_uid"))):
            raise ValueError("same ordered physical owner, geometry and clearance floor required")
        root = measured_pose(source["hands"][side], "link_poses_xyzw", "base_link")
        sdk = exchange["action_readback"]["post"]["hands"][side]
        actual = named_joint_values(sdk, "q", JOINTS)
        target = keep_reference_mimic_residual(q0, named_joint_values(sdk, "backend_targets", ACTIVE))
        motion.append((root_inv @ root, actual - q0, target - q0))
    return motion


def _minimum_replay(candidate, geometry, motion):
    candidate_root = pose_matrix(candidate["base_pose_wxyz"])
    q = expand(candidate["active_q_rad"])
    layout = geometry["reference"]["reference_layout"]
    table_top = layout["table_center"][2] + layout["table_size"][2] / 2
    shapes = [(name, shape["vertices"]) for source in ("source_colliders", "cooked_pads")
              for name, shape in geometry[source].items()]

    def clearance(root, q11):
        poses = full_named_poses(geometry["fk"], root, dict(zip(JOINTS, q11)))
        return min(float((vertices @ poses[name][2, :3] + poses[name][2, 3]).min() - table_top)
                   for name, vertices in shapes)

    actual, target = float("inf"), float("inf")
    for root_delta, actual_delta, target_delta in motion:
        root = candidate_root @ root_delta
        actual = min(actual, clearance(root, q + actual_delta))
        target = min(target, clearance(root, q + target_delta))
    return dict(minimum_actual_m=actual, minimum_target_m=target, samples=len(motion))


def prepare_candidate(parent_pair, ledgers, *, project):
    if (set(ledgers) != set(RUNS) or parent_pair["passes_static_policy_clearance"] is not True
            or parent_pair["native_qualified"] is not False):
        raise ValueError("two pinned Pinch ledgers and a verified CPU Reset candidate required")
    geometry, rod = supported_geometry(
        load_geometry(project, side="right"), load_flexible_grasp_task(project / TASK, project))
    if rod != parent_pair["rod_seed"]:
        raise ValueError("same grounded free rod required")
    motions = {name: _read_motion(*ledgers[name], expected_steps=steps,
                                 reset_seed_sha=parent_pair["reset_seed"]["sha256"], geometry=geometry)
               for name, steps in RUNS.items()}
    original = parent_pair["right"]["candidate"]
    candidate = deepcopy(original)
    candidate.update(candidate_id="pinch-v10-guard-replay-right",
                     parent_candidate_id=original["candidate_id"], local_change=RIGHT_CHANGE.copy())
    pose = np.asarray(candidate["base_pose_wxyz"], dtype=float)
    pose[[0, 2]] += [RIGHT_CHANGE["root_x_m"], RIGHT_CHANGE["root_z_m"]]
    rotated = (Rotation.from_rotvec([0., RIGHT_CHANGE["world_pitch_rad"], 0.])
               * Rotation.from_quat(np.roll(pose[3:], -1)))
    pose[3:] = np.roll(rotated.as_quat(), 1)
    candidate["base_pose_wxyz"] = pose.tolist()
    candidate["active_q_rad"][ACTIVE.index("finger12")] += RIGHT_CHANGE["thumb_seed_rad"]
    baseline = {name: _minimum_replay(original, geometry, motion) for name, motion in motions.items()}
    replay = {name: _minimum_replay(candidate, geometry, motion) for name, motion in motions.items()}
    lowest = min(min(data["minimum_actual_m"], data["minimum_target_m"]) for data in replay.values())
    if not np.isfinite(lowest) or lowest < CLEARANCE_M:
        raise ValueError("candidate failed original clearance in at least one motion replay")
    candidate = measure_strict(geometry, candidate)
    path = verify_candidate_path(geometry, candidate)
    if not qualifies(candidate, path):
        raise ValueError("candidate failed strict endpoints or continuous geometry")
    bilateral = independent_sweep_gap(parent_pair["left"]["path"]["continuous_hand_y_bounds_m"],
                                      path["continuous_hand_y_bounds_m"])
    new_pair = deepcopy(parent_pair)
    new_pair["right"] = dict(candidate=candidate, path=path, source_sha256=geometry["source_sha256"])
    new_pair["right_continuous_clearance_m"] = minimum_clearance(path)
    new_pair["independent_hand_y_gap_m"] = bilateral
    new_pair["reset_seed"] = make_joint_seed(parent_pair["left"]["candidate"], candidate)
    evidence = dict(ledgers={name: sha for name, (_, sha) in ledgers.items()}, baseline=baseline,
                    candidate_replay=replay, minimum_replay_clearance_m=lowest,
                    static_right_continuous_clearance_m=minimum_clearance(path),
                    max_required_endpoint_abs_gap_m=candidate["max_required_endpoint_abs_gap_m"],
                    bilateral_y_gap_m=bilateral, actual_isaac_steps=0, actual_ppo_updates=0,
                    sdk_targets_are_measurements=False, ideal_tracking_counterfactual=True,
                    native_qualification=False, teacher_admission=False)
    return new_pair, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pair", "v9", "v10", "candidate-output", "report-output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("pair", "v9", "v10"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    outputs = (args.candidate_output, args.report_output)
    if outputs[0].resolve() == outputs[1].resolve() or any(path.exists() for path in outputs):
        parser.error("distinct new candidate and report outputs required")
    root = Path(__file__).resolve().parents[1]
    pair, report = prepare_candidate(checked_json(args.pair, args.pair_sha256),
                                     {name: (getattr(args, name), getattr(args, name + "_sha256"))
                                      for name in RUNS}, project=root)
    for path, data in zip(outputs, (pair, report)):
        with path.open("x", encoding="utf-8") as out:
            json.dump(data, out, indent=2, allow_nan=False)
            out.write("\n")


if __name__ == "__main__":
    main()
