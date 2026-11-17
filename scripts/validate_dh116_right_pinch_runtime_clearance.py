#!/usr/bin/env python3
"""Recheck one right Pinch variant against the actual wrist clearance floor."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from scripts.audit_dh116_pinch_initial_pose import checked_json, digest, load_geometry
from scripts.prepare_dh116_flexible_style_poses import verify_candidate_path
from scripts.refine_dh116_right_pinch_geometry import measure_strict, supported_geometry


PROPOSAL_SHA = "02b79aaa83f2c21f58a16e377cf67bd273511aae4b26714ab46fc30df1fa4e43"
TASK = "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json"


def runtime_candidate(parent):
    if parent["candidate_id"] != "right-targeted-092" or parent["side"] != "right":
        raise ValueError("exact independent right Pinch proposal required")
    candidate = deepcopy(parent)
    candidate.update(candidate_id="right-pinch-policy-clearance-01",
                     parent_candidate_id=parent["candidate_id"],
                     local_change=dict(root_z_m=.000065, thumb_seed_rad=.0001))
    candidate["base_pose_wxyz"][2] += .000065
    candidate["active_q_rad"][1] += .0001
    expand(candidate["active_q_rad"])
    return candidate


def minimum_clearance(path):
    return float(min(value for shapes in path["continuous_table_clearance_lower_bound_m"].values()
                     for value in shapes.values()))


def qualifies(candidate, path):
    return (path["candidate_id"] == candidate["candidate_id"]
            and candidate["strict_endpoint_passed"] is True
            and path["geometry_preparation_passed"] is True
            and path["reset_geometry_passed"] is True
            and minimum_clearance(path) >= CLEARANCE_M)


def evaluate(project, proposal_file):
    proposal = checked_json(proposal_file, PROPOSAL_SHA)
    if digest(project / TASK) != proposal["task_sha256"]:
        raise ValueError("original flexible task differs")
    for name, sha in proposal["source_sha256"].items():
        if digest(project / name) != sha:
            raise ValueError(f"prior evidence source changed: {name}")
    task = load_flexible_grasp_task(project / TASK, project)
    geometry = load_geometry(project, side="right")
    geometry, rod_seed = supported_geometry(geometry, task)
    if rod_seed != proposal["rod_seed"]:
        raise ValueError("proposed grounded free-rod seed differs")
    candidate = measure_strict(geometry, runtime_candidate(proposal["candidate"]))
    path = verify_candidate_path(geometry, candidate)
    return dict(version="dh116-right-pinch-policy-clearance-cpu-v1",
        proposal_sha256=PROPOSAL_SHA, source_sha256=geometry["source_sha256"],
        task_sha256=proposal["task_sha256"], candidate=candidate, path=path, rod_seed=rod_seed,
        policy_clearance_floor_m=CLEARANCE_M, continuous_clearance_m=minimum_clearance(path),
        passes_static_policy_clearance=qualifies(candidate,path),
        native_qualification=False, teacher_admission=False, new_physics_steps=0, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(Path(__file__).resolve().parents[1], args.proposal)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "passes_static_policy_clearance", "continuous_clearance_m", "native_qualification")}))
    return 0 if result["passes_static_policy_clearance"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
