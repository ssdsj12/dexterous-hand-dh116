#!/usr/bin/env python3
"""Bounded Tripod-only proximal obstacle screen: 60 independent variants per side."""
import argparse
import itertools
import json
from pathlib import Path
import signal
import time

import numpy as np

from scripts.audit_dh116_pinch_initial_pose import checked_json,digest,finite_array,load_geometry,settling_gap,world_vertices
from scripts.prepare_dh116_flexible_style_poses import (
    ACTIVE,TRIPOD_SHA,MARGIN,REACH_TOLERANCE,measure_closure_grid,verify_candidate_path,rotate_seed_pose)
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm,SOURCES
from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M as RUNTIME_FLOOR

VERSION="dh116-tripod-proximal-screen-v1"
SIDES=("left","right")
BLOCKER={"left":"finger22_link","right":"finger13_link"}
PARENTS={"left":"tripod-left-r20-p30-b0-x0.002","right":"tripod-right-r0-p15-b0-x-0.002"}
PRIOR=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/"
           "tripod-power-preparation-20261113-7zhi9kgm/final-preparation.json")
PRIOR_SHA="e6335c291c48ff40d1f19eb114a1f074594b2b7981e749b0f86335929babe4a6"
MAXIMUM_CANDIDATES=120


def load_parents(path,project,geometries):
    prior=checked_json(path,PRIOR_SHA)
    if (digest(Path(project)/"scripts/prepare_dh116_flexible_style_poses.py")!=prior["preparation_source_sha256"]
            or digest(Path(project)/"scripts/audit_dh116_pinch_initial_pose.py")!=prior["geometry_audit_sha256"]
            or prior["tripod"]["seed_sha256"]!=TRIPOD_SHA):
        raise ValueError("immutable Tripod numerical sources differ")
    parents={}
    for side in SIDES:
        record=prior["tripod"]["sides"][side]
        row,path=record["selected"]["candidate"],record["selected"]["path"]
        if (record["source_sha256"]!=geometries[side]["source_sha256"]
                or row["candidate_id"]!=PARENTS[side] or row["side"]!=side
                or path["geometry_preparation_passed"] is not False
                or path["continuous_nonrequired_rod_gap_lower_bound_m"]["source"][BLOCKER[side]]>=0):
            raise ValueError("exact independent Tripod parent and negative proximal path required")
        parents[side]=row
    return parents,prior["tripod"]


def endpoint_blocker(geometry,row):
    q=np.asarray(row["active_q_rad"])+row["active_delta_rad"]
    link=BLOCKER[row["side"]]
    shape=world_vertices(geometry,row["base_pose_wxyz"],q)[link]
    return dict(link=link,**settling_gap(shape,geometry["reference"]["reference_layout"],
                                       geometry["old_candidate"]["rod"]["radius_m"]))


def candidates(parent,diagnostic):
    side=parent["side"]
    if (side not in SIDES or parent["candidate_id"]!=PARENTS[side]
            or parent["independent_style"]!="tripod" or parent["independent_seed_sha256"]!=TRIPOD_SHA
            or diagnostic["link"]!=BLOCKER[side]):
        raise ValueError("exact side-specific Tripod parent and proximal blocker required")
    direction=finite_array(diagnostic["geometric_axis_to_hull_direction"],(3,)).copy()
    direction[1]=0  # Original rod is a straight world-Y segment; do not translate along it.
    norm=np.linalg.norm(direction)
    if norm<1e-8:
        raise ValueError("nonzero measured XZ escape direction required")
    direction/=norm
    distances=(0.,.001,.002,.003,.004) if side=="left" else (0.,.00025,.0005,.00075,.001)
    rolls=(0.,-4.,4.,8.) if side=="left" else (0.,-2.,2.,4.)
    spreads=(0.,-.04,.04) if side=="left" else (0.,-.02,.02)
    pivot=[.43,.2 if side=="left" else -.2,.308]
    for i,(distance,roll,spread) in enumerate(itertools.product(distances,rolls,spreads)):
        pose=np.asarray(rotate_seed_pose(parent["base_pose_wxyz"],pivot,roll,0.))
        pose[:3]+=distance*direction
        q=finite_array(parent["active_q_rad"],(6,)).copy();q[0]+=spread
        expand(q)
        yield dict(candidate_id=f"tripod-proximal-{side}-{i:02d}",parent_candidate_id=PARENTS[side],
            independent_style="tripod",side=side,independent_seed_sha256=TRIPOD_SHA,
            active_joint_names=list(ACTIVE),active_q_rad=q.tolist(),base_pose_wxyz=pose.tolist(),
            local_change=dict(measured_escape_xz=direction.tolist(),escape_distance_m=distance,
                              additional_world_roll_deg=roll,thumb_spread_seed_rad=spread))


def measure(geometry,candidate):
    row=measure_closure_grid(geometry,candidate,(1,2,3))
    row["targeted_endpoint"]=endpoint_blocker(geometry,row)
    row["required_endpoint_tolerance_m"]=REACH_TOLERANCE
    row["endpoint_runtime_floor_margin_m"]=row["endpoint_all_shape_table_clearance_m"]-RUNTIME_FLOOR
    return row


def check_side_paths(geometry,rows,checks,progress):
    # An already negative target endpoint disproves a clear full path.
    possible=sorted((r for r in rows if r["coarse_geometry_passed"]
                     and r["targeted_endpoint"]["capsule_surface_gap_m"]>=0),
                    key=lambda r:(r["max_required_endpoint_abs_gap_m"],r["candidate_id"]))
    for row in possible[:2]:
        path=verify_candidate_path(geometry,row)
        path["side"]=row["side"]
        path["runtime_floor_margin_m"]=min(v for group in path["continuous_table_clearance_lower_bound_m"].values()
                                          for v in group.values())-RUNTIME_FLOOR
        with _commit_row_without_alarm():
            checks.append(path);progress(dict(record_type="complete_path",**path))
        if path["geometry_preparation_passed"]:
            return path
    return None


def pair_result(selected):
    complete=all(side in selected and selected[side]["geometry_preparation_passed"] is True for side in SIDES)
    separation=(selected["left"]["continuous_hand_y_bounds_m"][0]-
                selected["right"]["continuous_hand_y_bounds_m"][1]) if complete else None
    minimum=min((v for path in selected.values()
                 for group in path["continuous_table_clearance_lower_bound_m"].values() for v in group.values()),
                default=None)
    margin=None if minimum is None else minimum-RUNTIME_FLOOR
    return dict(bilateral_geometry_preparation_passed=bool(complete and separation>0),
        bilateral_y_separation_lower_bound_m=separation,static_floor_m=MARGIN,
        runtime_floor_m=RUNTIME_FLOOR,minimum_runtime_floor_margin_m=margin,
        runtime_floor_passed=bool(complete and margin>=0),native_ready=False,teacher_admitted=False)


def write_proposal(output,result):
    if result["status"]!="complete" or not result["pair"]["bilateral_geometry_preparation_passed"]:
        return False
    selected=result["selected_paths"]
    ids={p["candidate_id"] for p in selected.values()}
    proposal=dict(version=VERSION,scope="static_reset_proposal_not_registered",
        candidates=[r for r in result["rows"] if r["candidate_id"] in ids],
        selected_paths=selected,pair=result["pair"],rod_reference=result["rod_reference"],
        source_sha256=result["source_sha256"],prior_result_sha256=PRIOR_SHA,
        current_registered_reset_seed_matches=False,native_ready=False,teacher_admitted=False)
    with (Path(output)/"static-reset-proposal.json").open("x") as stream:
        json.dump(proposal,stream,indent=2,allow_nan=False);stream.write("\n")
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,default=50.)
    args=parser.parse_args(argv)
    if not np.isfinite(args.wall_seconds) or not 1<=args.wall_seconds<=50:
        parser.error("CPU deadline must be 1..50 seconds")
    project=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();finished=set();planned=[];stage="setup"
    result=dict(version=VERSION,status="partial_setup",rows=[],errors=[],path_checks=[],selected_paths={},
        native_ready=False,teacher_admitted=False,physics_steps=0,optimizer_updates=0,
        all_17_live_cooked_available=False,static_reset_proposal_created=False)
    def expire(signum,frame):raise BudgetExpired()
    old_handler=signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,args.wall_seconds)
    with (args.output/"progress.jsonl").open("x") as ledger:
        def progress(row):
            ledger.write(json.dumps(row,allow_nan=False)+"\n");ledger.flush()
        try:
            geometries={side:load_geometry(project,side=side) for side in SIDES}
            parents,prior=load_parents(PRIOR,project,geometries)
            diagnostics={side:endpoint_blocker(geometries[side],parents[side]) for side in SIDES}
            per_side={side:list(candidates(parents[side],diagnostics[side])) for side in SIDES}
            plan=[row for pair in zip(per_side["left"],per_side["right"]) for row in pair]
            planned=[r["candidate_id"] for r in plan]
            if len(planned)!=MAXIMUM_CANDIDATES or len(set(planned))!=MAXIMUM_CANDIDATES:
                raise ValueError("exact 120 total independent candidates required")
            layout=geometries["left"]["reference"]["reference_layout"]
            result.update(prior_result_sha256=PRIOR_SHA,parent_candidates=parents,
                parent_endpoint_diagnostics=diagnostics,
                parent_selected_paths={s:prior["sides"][s]["selected"]["path"] for s in SIDES},
                geometry_source_sha256={s:g["source_sha256"] for s,g in geometries.items()},
                candidate_ids=planned,rod_reference=dict(
                    original_nodes_w=layout["wire_points"],radius_m=.005,
                    vertical_geometry_envelope_axis_z_m=[.305,.308],settled_endpoint_axis_z_m=.305,
                    current_registered_axis_z_m=.308,registered_bottom_table_gap_m=.003,
                    table_center_m=layout["table_center"],table_size_m=layout["table_size"],
                    native_support_or_settling_verified=False,registered_reset_modified=False))
            stage="endpoint_screen"
            for candidate in plan:
                try:
                    row=measure(geometries[candidate["side"]],candidate)
                except ValueError as exc:
                    record=dict(candidate_id=candidate["candidate_id"],side=candidate["side"],error=str(exc))
                    with _commit_row_without_alarm():
                        result["errors"].append(record);finished.add(candidate["candidate_id"])
                        progress(dict(record_type="error",**record))
                else:
                    with _commit_row_without_alarm():
                        result["rows"].append(row);finished.add(candidate["candidate_id"])
                        progress(dict(record_type="endpoint",**row))
            stage="continuous_paths"
            for side in SIDES:
                selected=check_side_paths(geometries[side],[r for r in result["rows"] if r["side"]==side],
                                          result["path_checks"],progress)
                if selected:
                    with _commit_row_without_alarm():result["selected_paths"][side]=selected
            result["status"]="complete_with_errors" if result["errors"] else "complete"
        except BudgetExpired:
            result.update(status="partial_timeout",unfinished_stage=stage)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,old_handler)
    result.update(elapsed_s=time.monotonic()-started,completed_candidates=len(result["rows"]),
        remaining_candidate_ids=[i for i in planned if i not in finished],
        pair=pair_result(result["selected_paths"]),
        budget=dict(total_candidates=120,candidates_per_side=60,max_paths_per_side=2,
                    path_intervals=160,soft_wall_seconds=args.wall_seconds),
        source_sha256={name:digest(project/name) for name in (*SOURCES,
            "scripts/screen_dh116_tripod_proximal.py",
            "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py")},
        progress_sha256=digest(args.output/"progress.jsonl"))
    result["static_reset_proposal_created"]=write_proposal(args.output,result)
    with (args.output/"result.json").open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:result[k] for k in ("status","elapsed_s","completed_candidates","pair",
                                           "static_reset_proposal_created")}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
