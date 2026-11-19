#!/usr/bin/env python3
"""Bind one explicit same-rod Pinch reset proposal to independent CPU evidence."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state
from scripts.audit_dh116_pinch_initial_pose import checked_json, load_geometry
from scripts.prepare_dh116_flexible_style_poses import verify_candidate_path
from scripts.refine_dh116_right_pinch_geometry import measure_strict
from scripts.validate_dh116_right_pinch_runtime_clearance import minimum_clearance, qualifies


LEFT = "configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json"
LEFT_SHA = "a2eefb1a8f487fdba7e567c51aab005211dec3f658d2aa939c0ff7289de0dc44"
RIGHT_SHA = "9e47b1990493cf9ea62f7324e17c5fb596cd5dd337ca4ec8557ea7098136af1d"


def independent_sweep_gap(left, right):
    intervals = np.asarray([left, right], dtype=float)
    if (intervals.shape != (2, 2) or not np.isfinite(intervals).all()
            or np.any(intervals[:, 0] > intervals[:, 1])):
        raise ValueError("ordered finite full-hand swept Y bounds required")
    gap = float(max(left[0], right[0]) - min(left[1], right[1]))
    if gap <= 0:
        raise ValueError("hand envelopes overlap along the rod's Y axis")
    return gap


def make_joint_seed(left, right):
    if (left["side"], right["side"]) != ("left", "right"):
        raise ValueError("two independent side-specific seeds required")
    return prepare_reset_state(
        hand_roots={h["side"]: h["base_pose_wxyz"] for h in (left, right)},
        origin=[0.,0.,0.], root_xy=[[0.,0.],[0.,0.]], initial_offset_m=0.,
        joint_positions=[expand(h["active_q_rad"]).tolist() for h in (left,right)],
        joint_velocities=[[0.] * 11 for _ in range(2)],
        source_description="independent side-specific Pinch CPU proposal; unchanged mounted action contract")


def left_clearance_candidate(reference, extra_clearance_m, *, root_x_m=0., thumb_seed_rad=0.):
    if not math.isfinite(extra_clearance_m) or extra_clearance_m < 0.:
        raise ValueError("finite nonnegative left reset clearance margin required")
    if (not math.isfinite(root_x_m) or not math.isfinite(thumb_seed_rad)
            or abs(root_x_m) > .0001 or abs(thumb_seed_rad) > .003125
            or (not extra_clearance_m and (root_x_m or thumb_seed_rad))):
        raise ValueError("bounded recenter requires an independent left clearance margin")
    rise = .00011 + extra_clearance_m
    candidate = deepcopy(reference)
    candidate.update(candidate_id="left-pinch-policy-clearance-01", local_change={"root_z_m": rise})
    if extra_clearance_m:
        candidate.update(candidate_id="left-pinch-policy-clearance-margin",
                         parent_candidate_id=reference.get("candidate_id", Path(LEFT).stem))
    candidate["base_pose_wxyz"][2] += rise
    if root_x_m or thumb_seed_rad:
        candidate["local_change"].update(root_x_m=root_x_m, thumb_seed_rad=thumb_seed_rad)
        candidate["base_pose_wxyz"][0] += root_x_m
        candidate["active_q_rad"][1] += thumb_seed_rad
        expand(candidate["active_q_rad"])
    return candidate


def prepare(project, right_result, *, left_extra_clearance_m=0., left_root_x_m=0., left_thumb_seed_rad=0.):
    left = checked_json(project / LEFT, LEFT_SHA)
    right = checked_json(right_result, RIGHT_SHA)
    geometry = load_geometry(project, side="left")
    if (geometry["source_sha256"] != left["source_sha256"]
            or right["passes_static_policy_clearance"] is not True
            or not qualifies(right["candidate"], right["path"])):
        raise ValueError("original side-specific evidence and wrist floor required")
    layout = geometry["reference"]["reference_layout"]
    rod = right["rod_seed"]
    candidate_radius = geometry["old_candidate"]["rod"]["radius_m"]
    top = layout["table_center"][2] + layout["table_size"][2] / 2
    original = np.asarray(layout["wire_points"])
    expected = original.copy()
    expected[:, 2] = top + candidate_radius
    if (rod["pinned_node_ids"] != [] or rod["fix_root_orientation"] is not False
            or rod["gravity_world_m_s2"] != [0., 0., -9.81] or rod["radius_m"] != candidate_radius
            or not np.array_equal(rod["original_nodes_m"], original)
            or not np.array_equal(rod["proposed_nodes_m"], expected)
            or not np.array_equal(rod["runtime_float32_nodes_m"], expected.astype(np.float32))):
        raise ValueError("same free grounded rod, unmodified XY and locked table required")
    layout["wire_points"] = expected.tolist()
    candidate = left_clearance_candidate(left, left_extra_clearance_m,
                                         root_x_m=left_root_x_m, thumb_seed_rad=left_thumb_seed_rad)
    candidate = measure_strict(geometry, candidate)
    path = verify_candidate_path(geometry, candidate)
    left_clearance = minimum_clearance(path)
    left_passed = qualifies(candidate, path) and left_clearance >= CLEARANCE_M + left_extra_clearance_m
    if not left_passed and not left_extra_clearance_m:
        raise ValueError("left initial/continuous wrist floor or closure geometry failed")
    gap = independent_sweep_gap(path["continuous_hand_y_bounds_m"],
                                right["path"]["continuous_hand_y_bounds_m"])
    result = dict(version="dh116-paired-pinch-reset-preflight-cpu-v1",
        left_pose_sha256=LEFT_SHA, right_evidence_sha256=RIGHT_SHA,
        left=dict(candidate=candidate, path=path),
        right={k:right[k] for k in ("candidate","path","source_sha256")},
        reset_seed=make_joint_seed(candidate, right["candidate"]) if left_passed else None, rod_seed=rod,
        wrist_clearance_floor_m=CLEARANCE_M, independent_hand_y_gap_m=gap,
        native_qualified=False, teacher_admission=False, grasp_qualified=False,
        actual_sdk_steps=0, actual_ppo_updates=0)

    if left_extra_clearance_m:
        result.update(version="dh116-paired-pinch-reset-preflight-cpu-margin-v1",
                      left_extra_clearance_m=left_extra_clearance_m,
                      left_continuous_clearance_m=left_clearance,
                      right_continuous_clearance_m=minimum_clearance(right["path"]),
                      passes_static_policy_clearance=left_passed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right-result",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--left-extra-clearance-m", type=float, default=0.,
                        help="independently recheck a raised left Reset candidate")
    parser.add_argument("--left-root-x-m", type=float, default=0.)
    parser.add_argument("--left-thumb-seed-rad", type=float, default=0.)
    args = parser.parse_args()
    record = prepare(Path(__file__).resolve().parents[1], args.right_result,
                     left_extra_clearance_m=args.left_extra_clearance_m,
                     left_root_x_m=args.left_root_x_m, left_thumb_seed_rad=args.left_thumb_seed_rad)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
        stream.write("\n")
    seed = record["reset_seed"]
    print(json.dumps(dict(reset_seed_sha256=None if seed is None else seed["sha256"],
        independent_hand_y_gap_m=record["independent_hand_y_gap_m"], native_qualified=False)))
    return 3 if seed is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
