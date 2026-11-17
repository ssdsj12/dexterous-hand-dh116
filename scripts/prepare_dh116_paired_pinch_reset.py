#!/usr/bin/env python3
"""Bind one explicit same-rod Pinch reset proposal to independent CPU evidence."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import prepare_reset_state
from scripts.audit_dh116_pinch_initial_pose import checked_json, load_geometry
from scripts.prepare_dh116_flexible_style_poses import verify_candidate_path
from scripts.refine_dh116_right_pinch_geometry import measure_strict
from scripts.validate_dh116_right_pinch_runtime_clearance import qualifies


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


def prepare(project, right_result):
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
    candidate = deepcopy(left)
    candidate.update(candidate_id="left-pinch-policy-clearance-01",
                     local_change={"root_z_m":.00011})
    candidate["base_pose_wxyz"][2] += .00011
    candidate = measure_strict(geometry, candidate)
    path = verify_candidate_path(geometry, candidate)
    if not qualifies(candidate,path):
        raise ValueError("left initial/continuous wrist floor or closure geometry failed")
    gap = independent_sweep_gap(path["continuous_hand_y_bounds_m"],
                                right["path"]["continuous_hand_y_bounds_m"])
    return dict(version="dh116-paired-pinch-reset-preflight-cpu-v1",
        left_pose_sha256=LEFT_SHA, right_evidence_sha256=RIGHT_SHA,
        left=dict(candidate=candidate, path=path),
        right={k:right[k] for k in ("candidate","path","source_sha256")},
        reset_seed=make_joint_seed(candidate,right["candidate"]), rod_seed=rod,
        wrist_clearance_floor_m=CLEARANCE_M, independent_hand_y_gap_m=gap,
        native_qualified=False, teacher_admission=False, grasp_qualified=False,
        actual_sdk_steps=0, actual_ppo_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right-result",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    record = prepare(Path(__file__).resolve().parents[1],args.right_result)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(reset_seed_sha256=record["reset_seed"]["sha256"],
        independent_hand_y_gap_m=record["independent_hand_y_gap_m"], native_qualified=False)))


if __name__ == "__main__":
    main()
