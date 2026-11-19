#!/usr/bin/env python3
"""Replay the right Pinch v8 terminal and test one bounded CPU Reset candidate."""
import argparse
from copy import deepcopy
from pathlib import Path
import json
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
from scripts.refine_dh116_right_pinch_geometry import measure_strict, supported_geometry
from scripts.validate_dh116_right_pinch_runtime_clearance import TASK, minimum_clearance, qualifies


def shifted_right(parent):
    candidate = deepcopy(parent)
    candidate.update(candidate_id="pinch-v8-right-clearance-150um",
                     parent_candidate_id=parent["candidate_id"],
                     local_change=dict(root_z_m=.00015, root_x_m=-.000075,
                                       world_pitch_rad=-.001, thumb_seed_rad=.001))
    pose = np.asarray(candidate["base_pose_wxyz"])
    pose[[0, 2]] += [-.000075, .00015]
    rotation = Rotation.from_rotvec([0., -.001, 0.]) * Rotation.from_quat(np.roll(pose[3:], -1))
    pose[3:] = np.roll(rotation.as_quat(), 1)
    candidate["base_pose_wxyz"] = pose.tolist()
    candidate["active_q_rad"][ACTIVE.index("finger12")] += .001
    expand(candidate["active_q_rad"])
    return candidate


def named_joint_values(hand, field, names):
    return np.asarray(hand[field], dtype=float)[0, [hand["joint_names"].index(n) for n in names]]


def measured_pose(hand, field, name):
    pose = np.asarray(hand[field], dtype=float)[0, hand["body_names"].index(name)]
    return pose_matrix(np.r_[pose[:3], pose[6], pose[3:6]])


def clearance_replay(rows, pair, candidate, geometry):
    if [row["sequence"] for row in rows] != list(range(len(rows))):
        raise ValueError("complete original continuous journal required")
    resets = [row["payload"]["reset"] for row in rows if row["operation"] == "owner_event"
              and row["payload"]["kind"] == "reset"]
    exchanges = [row["payload"] for row in rows if row["operation"] == "delivered_exchange"]
    if len(resets) != 1 or [x["physics_step"] for x in exchanges] != list(range(3, 8)):
        raise ValueError("exact Pinch v8 Reset and five recorded substeps required")
    reset = resets[0]
    if reset["after_clock"]["physics_step"] != 2 or reset["seed"]["sha256"] != pair["reset_seed"]["sha256"]:
        raise ValueError("journal Reset differs from the prior paired proposal")
    if exchanges[-1]["original_guard_failure"] != "post_step_source_clearance:right":
        raise ValueError("terminal must be the observed right source table guard")
    old = reset["hands"]["right"]
    q0 = named_joint_values(old, "q", JOINTS)
    base0 = measured_pose(old, "body_poses_xyzw", "base_link")
    candidate_base = pose_matrix(candidate["base_pose_wxyz"])
    candidate_q = expand(candidate["active_q_rad"])
    layout = geometry["reference"]["reference_layout"]
    table = layout["table_center"][2] + layout["table_size"][2]/2
    target_link = "finger13_link"
    vertices = geometry["source_colliders"][target_link]["vertices"]

    def clearance(v, pose):
        return float((v @ pose[2, :3] + pose[2, 3]).min() - table)

    def poses(base, full):
        return full_named_poses(geometry["fk"], base, dict(zip(JOINTS, full)))

    def all_clearance(base, full):
        link_poses = poses(base, full)
        return min(clearance(shape["vertices"], link_poses[name])
                   for kind in ("source_colliders", "cooked_pads")
                   for name, shape in geometry[kind].items())

    measured_delta = []
    for exchange in exchanges:
        packet = exchange["closure_geometry"]
        raw = packet["raw"]
        keys = ("run_uuid", "episode_uid", "env_id", "reset_id", "rod_uid")
        if (packet["source_complete"] is not True or
                packet["sources"]["right"]["source_sha256"] != geometry["source_sha256"] or
                any(raw["identity"][key] != reset["after_clock"][key] for key in keys) or
                exchange["post_step_clearance"]["floor_m"] != CLEARANCE_M):
            raise ValueError("source, episode identity or frozen table gate mismatch")
        hand = raw["hands"]["right"]
        readback = exchange["action_readback"]["post"]["hands"]["right"]
        actual = named_joint_values(readback, "q", JOINTS)
        active = named_joint_values(readback, "q", ACTIVE)
        issued = named_joint_values(readback, "backend_targets", ACTIVE)
        measured_base = measured_pose(hand, "link_poses_xyzw", "base_link")
        link = measured_pose(hand, "link_poses_xyzw", target_link)
        measured = clearance(vertices, link)
        if abs(measured - packet["sides"]["right"]["table_clearance_m"]["source:"+target_link]) > 1e-8:
            raise ValueError("source convex hull no longer reproduces native clearance")
        zero = clearance(vertices, poses(measured_base, q0)[target_link])
        active_q = keep_reference_mimic_residual(q0, active)
        active_only = clearance(vertices, poses(measured_base, active_q)[target_link])
        actual_model = clearance(vertices, poses(measured_base, actual)[target_link])
        target_q = keep_reference_mimic_residual(q0, issued)
        ideal_target = clearance(vertices, poses(measured_base, target_q)[target_link])
        candidate_frame = candidate_base @ np.linalg.inv(base0) @ measured_base
        measured_delta.append(dict(sdk_step=exchange["physics_step"], measured_m=measured,
            active_fk_delta_m=active_only-zero, passive_residual_delta_m=actual_model-active_only,
            sdk_target_fk_delta_m=ideal_target-zero, actual_fk_error_m=measured-actual_model,
            recovered_candidate_from_measured_joints_m=all_clearance(candidate_frame, candidate_q+actual-q0),
            recovered_candidate_from_issued_targets_m=all_clearance(candidate_frame, candidate_q+target_q-q0)))
    return measured_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("journal", "pair", "output", "candidate-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--journal-sha256", required=True)
    parser.add_argument("--pair-sha256", required=True)
    args = parser.parse_args()
    if args.output == args.candidate_output or args.output.exists() or args.candidate_output.exists():
        parser.error("two distinct new evidence outputs required")
    project = Path(__file__).resolve().parents[1]
    pair = checked_json(args.pair, args.pair_sha256)
    if pair["passes_static_policy_clearance"] is not True or pair["native_qualified"] is not False:
        raise ValueError("original statically prepared pair required")
    geometry, rod = supported_geometry(load_geometry(project, side="right"),
        load_flexible_grasp_task(project/TASK, project))
    if rod != pair["rod_seed"]:
        raise ValueError("same original grounded free rod required")
    candidate = measure_strict(geometry, shifted_right(pair["right"]["candidate"]))
    path = verify_candidate_path(geometry, candidate)
    rows = [decode_owned(json.loads(line)) for line in _checked_bytes(args.journal, args.journal_sha256).splitlines()]
    replay = clearance_replay(rows, pair, candidate, geometry)
    actual_min = min(row["recovered_candidate_from_measured_joints_m"] for row in replay)
    target_min = min(row["recovered_candidate_from_issued_targets_m"] for row in replay)
    bilateral = independent_sweep_gap(pair["left"]["path"]["continuous_hand_y_bounds_m"],
                                      path["continuous_hand_y_bounds_m"])
    passed = qualifies(candidate, path) and min(actual_min, target_min) >= CLEARANCE_M
    result = dict(parent_pair_sha256=args.pair_sha256, journal_sha256=args.journal_sha256,
        measured_steps=len(replay), minimum_measured_motion_replay_m=actual_min,
        minimum_sdk_target_replay_m=target_min, static_right_continuous_clearance_m=minimum_clearance(path),
        right_required_endpoint_gap_m=candidate["max_required_endpoint_abs_gap_m"],
        geometry_candidate_passed=passed, bilateral_y_gap_m=bilateral, rows=replay,
        sdk_targets_are_measurements=False, ideal_tracking_counterfactual=True,
        native_qualification=False, teacher_admission=False, actual_isaac_steps=0)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if passed:
        pair["right"] = dict(candidate=candidate, path=path, source_sha256=geometry["source_sha256"])
        pair["right_continuous_clearance_m"] = minimum_clearance(path)
        pair["independent_hand_y_gap_m"] = bilateral
        pair["reset_seed"] = make_joint_seed(pair["left"]["candidate"], candidate)
        with args.candidate_output.open("x") as stream:
            json.dump(pair, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
