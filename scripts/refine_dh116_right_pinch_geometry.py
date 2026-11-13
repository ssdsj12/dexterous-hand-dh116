#!/usr/bin/env python3
"""At most 120 local right Pinch variants; strict static checks, no native grant."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import signal
import time

import numpy as np

from scripts.audit_dh116_pinch_initial_pose import (
    checked_json,digest,finite_array,load_geometry,settling_gap,world_vertices,
)
from scripts.prepare_dh116_flexible_style_poses import measure_closure_grid,verify_candidate_path
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm,SOURCES
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,expand

VERSION="dh116-right-pinch-targeted-geometry-v1"
PARENT_ID="pinch-right-derived_right-r10-p10-b0.025-x0"
PRIOR=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/"
           "right-pinch-closure-cpu-20261113-oIQiFC/run01/result.json")
PRIOR_SHA="1a271578961192096d3d4aa5ec03785807c3940908bd9444fd0c285be86dff1f"
STRICT_ENDPOINT=.00005
MAXIMUM_CANDIDATES=120


def load_parent(path,project,geometry):
    report=checked_json(path,PRIOR_SHA)
    for name in ("scripts/prepare_dh116_flexible_style_poses.py",
                 "scripts/audit_dh116_pinch_initial_pose.py"):
        if digest(Path(project)/name)!=report["source_sha256"][name]:
            raise ValueError("prior numerical geometry helper changed")
    if geometry["source_sha256"]!=report["geometry_source_sha256"]:
        raise ValueError("prior actual right geometry sources changed")
    rows=[r for r in report["rows"] if r["candidate_id"]==PARENT_ID]
    paths=[r for r in report["path_checks"] if r["candidate_id"]==PARENT_ID]
    if (len(rows)!=1 or len(paths)!=1 or rows[0]["side"]!="right"
            or rows[0]["max_required_endpoint_abs_gap_m"]>=STRICT_ENDPOINT
            or paths[0]["geometry_preparation_passed"] is not False
            or paths[0]["continuous_nonrequired_rod_gap_lower_bound_m"]["source"]["finger13_link"]>=0):
        raise ValueError("exact prior strict endpoint candidate and failed finger13 path required")
    return rows[0],paths[0]


def candidates(parent):
    if parent["candidate_id"]!=PARENT_ID or parent["side"]!="right":
        raise ValueError("exact independent right parent required")
    # Bounded local shifts, no rotations or reflection; first entry is baseline.
    for i,(dx,dz,dthumb,dindex) in enumerate(itertools.product(
            (0.,.000025,.00005,.000075,.0001),(0.,-.000025,.000025),
            (0.,-.00025,.00025,.0005),(0.,.00025))):
        pose=finite_array(parent["base_pose_wxyz"],(7,)).copy()
        q=finite_array(parent["active_q_rad"],(6,)).copy()
        pose[0]+=dx;pose[2]+=dz;q[1]+=dthumb;q[2]+=dindex
        expand(q)
        yield dict(candidate_id=f"right-targeted-{i:03d}",parent_candidate_id=PARENT_ID,
            side="right",independent_style="pinch",active_joint_names=list(ACTIVE),
            base_pose_wxyz=pose.tolist(),active_q_rad=q.tolist(),
            independent_seed_sha256=parent["independent_seed_sha256"],
            local_change=dict(root_x_m=dx,root_z_m=dz,thumb_seed_rad=dthumb,index_seed_rad=dindex))


def supported_geometry(geometry,task):
    layout=geometry["reference"]["reference_layout"]
    cfg=geometry["old_candidate"]["rod"]
    top=layout["table_center"][2]+layout["table_size"][2]/2
    radius=cfg["radius_m"]
    if (geometry["side"]!="right" or cfg["pinned_node_ids"]!=[]
            or task["rod"]["pinned_node_ids"]!=[]
            or task["scene"]["table_top_m"]!=top
            or task["scene"]["table_center_m"]!=layout["table_center"]
            or task["scene"]["table_size_m"]!=layout["table_size"]):
        raise ValueError("same finite task table and free unpinned right rod required")
    nodes=finite_array(layout["wire_points"])
    if nodes.ndim!=2 or nodes.shape[1]!=3 or len(nodes)<2:
        raise ValueError("explicit source rod nodes required")
    original=nodes.copy();nodes=nodes.copy();nodes[:,2]=top+radius
    center,size=np.asarray(layout["table_center"]),np.asarray(layout["table_size"])
    if (np.any(nodes[:,:2]-radius<(center-size/2)[:2])
            or np.any(nodes[:,:2]+radius>(center+size/2)[:2])):
        raise ValueError("entire supported capsule must lie over actual finite table")
    g=dict(geometry,reference=deepcopy(geometry["reference"]))
    g["reference"]["reference_layout"]["wire_points"]=nodes.tolist()
    record=dict(source="explicit_supported_seed_proposal_not_installed_reset",
        original_nodes_m=original.tolist(),proposed_nodes_m=nodes.tolist(),
        runtime_float32_nodes_m=nodes.astype(np.float32).astype(float).tolist(),
        original_bottom_gap_m=float((original[:,2]-radius-top).min()),
        nominal_bottom_gap_m=float((nodes[:,2]-radius-top).min()),
        max_float32_roundtrip_m=float(abs(nodes.astype(np.float32).astype(float)-nodes).max()),
        radius_m=radius,pinned_node_ids=[],fix_root_orientation=False,
        gravity_world_m_s2=cfg["gravity_world_m_s2"],
        current_registered_reset_seed_matches=bool(np.array_equal(original,nodes)),
        task_native_reset_source_sha256=task["grasps"]["pinch"]["native_reset_source_sha256"],
        unknown_task_rod_fields=[k for k,v in task["rod"].items() if v is None],
        r0_coefficient_source=task["reward"]["coefficient_source"],
        native_table_support_verified=False,native_qualified=False)
    return g,record


def measure_strict(geometry,candidate):
    row=measure_closure_grid(geometry,candidate,(1,2))
    row["strict_endpoint_tolerance_m"]=STRICT_ENDPOINT
    row["strict_endpoint_passed"]=bool(row["coarse_geometry_passed"]
                                       and row["max_required_endpoint_abs_gap_m"]<STRICT_ENDPOINT)
    if row["strict_endpoint_passed"]:
        q=np.asarray(row["active_q_rad"])+row["active_delta_rad"]
        shapes=world_vertices(geometry,row["base_pose_wxyz"],q)
        closest=settling_gap(shapes["finger13_link"],geometry["reference"]["reference_layout"],
                            geometry["old_candidate"]["rod"]["radius_m"])
        row["finger13_endpoint_gap_m"]=closest["capsule_surface_gap_m"]
        row["finger13_source_endpoint_closest"]=closest
    return row


def check_paths(geometry,rows,*,maximum=3,checks=None,progress=None):
    if type(maximum) is not int or not 1<=maximum<=3:
        raise ValueError("one to three complete path checks allowed")
    checks=[] if checks is None else checks
    available=sorted((r for r in rows if r["strict_endpoint_passed"]),
        key=lambda r:(-r["finger13_endpoint_gap_m"],r["candidate_id"]))
    for row in available[:maximum]:
        check=verify_candidate_path(geometry,row)
        with _commit_row_without_alarm():
            checks.append(check)
            if progress:progress(dict(record_type="complete_path",**check))
        if check["geometry_preparation_passed"]:
            return checks,row["candidate_id"]
    return checks,None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,default=50.)
    args=parser.parse_args(argv)
    if not np.isfinite(args.wall_seconds) or not 1<=args.wall_seconds<=50:
        parser.error("bounded 1..50 second CPU deadline required")
    project=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",scope="CPU_static_geometry_only",
        rows=[],errors=[],path_checks=[],selected_candidate_id=None,
        native_ready=False,teacher_admitted=False,new_physics_steps=0,optimizer_updates=0,
        all_17_live_cooked_available=False,remaining_candidate_ids=[])
    def expire(signum,frame):raise BudgetExpired()
    old_handler=signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,args.wall_seconds)
    finished=set();planned=[];stage="setup"
    with (args.output/"progress.jsonl").open("x") as ledger:
        def progress(row):
            ledger.write(json.dumps(row,allow_nan=False)+"\n");ledger.flush()
        try:
            original=load_geometry(project,side="right")
            parent,prior_path=load_parent(PRIOR,project,original)
            task_path=project/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json"
            task=json.loads(task_path.read_text())
            geometry,rod_seed=supported_geometry(original,task)
            plan=list(candidates(parent))
            planned=[r["candidate_id"] for r in plan]
            result.update(parent_result_sha256=PRIOR_SHA,parent_candidate=parent,
                prior_path=prior_path,rod_seed=rod_seed,task_sha256=digest(task_path),
                geometry_source_sha256=original["source_sha256"],candidate_ids=planned)
            stage="strict_endpoint_screen"
            for candidate in plan:
                try:
                    row=measure_strict(geometry,candidate)
                except ValueError as exc:
                    error=dict(candidate_id=candidate["candidate_id"],error=str(exc))
                    with _commit_row_without_alarm():
                        result["errors"].append(error);finished.add(candidate["candidate_id"])
                        progress(dict(record_type="error",**error))
                else:
                    with _commit_row_without_alarm():
                        result["rows"].append(row);finished.add(candidate["candidate_id"])
                        progress(dict(record_type="strict_row",**row))
            stage="continuous_path_check"
            _,selected=check_paths(geometry,result["rows"],checks=result["path_checks"],progress=progress)
            result["selected_candidate_id"]=selected
            result["status"]="complete_with_errors" if result["errors"] else "complete"
        except BudgetExpired:
            result.update(status="partial_budget",unfinished_stage=stage)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,old_handler)
    result.update(elapsed_s=time.monotonic()-started,completed_candidates=len(result["rows"]),
        remaining_candidate_ids=[name for name in planned if name not in finished],
        strict_endpoint_passed=sum(r["strict_endpoint_passed"] for r in result["rows"]),
        geometry_preparation_passed=result["selected_candidate_id"] is not None,
        budget=dict(max_candidates=120,max_complete_paths=3,soft_wall_seconds=args.wall_seconds),
        source_sha256={name:digest(project/name) for name in (*SOURCES,
            "scripts/refine_dh116_right_pinch_geometry.py",
            "src/go2_pvcnn/control/dh116/tabletop_dual_hand_reset.py",
            "src/go2_pvcnn/control/dh116/tabletop_dual_hand_rod_runtime.py",
            "src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_runtime.py")},
        progress_sha256=digest(args.output/"progress.jsonl"))
    with (args.output/"result.json").open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({key:result[key] for key in ("status","elapsed_s","completed_candidates",
        "strict_endpoint_passed","selected_candidate_id","geometry_preparation_passed","native_ready")}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
