#!/usr/bin/env python3
"""Finite right-only CPU closure screen, never native execution or qualification."""
import argparse
from contextlib import contextmanager
import itertools
import json
import signal
import time
from pathlib import Path

import numpy as np

from scripts.audit_dh116_pinch_initial_pose import (
    checked_json, digest, finite_array, load_geometry, pose_matrix,
)
from scripts.prepare_dh116_flexible_style_poses import (
    clearance_corrected_pose, measure_closure_grid, rotate_seed_pose, verify_candidate_path,
)
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, expand

VERSION="dh116-right-pinch-closure-screen-v1"
SEED_FILE="configs/dh116_training/right_pinch_pregrasp_v1.json"
SEED_SHA="13ee6dfb0f5bfe1a349667f138f155fe55c3c41e8b0abac4a22fc822a6851888"
ROLL=(0.,-10.,10.)
PITCH=(0.,-10.,10.,-20.,20.)
BIAS=(0.,.025,.05)
SHIFT=(0.,-.002,.002,-.004,.004)
SOURCES=(
    "scripts/python.sh",
    "scripts/screen_dh116_right_pinch_closure.py",
    "scripts/prepare_dh116_flexible_style_poses.py",
    "scripts/audit_dh116_pinch_initial_pose.py",
    "scripts/audit_dh116_flexible_power_pose.py",
    "scripts/bake_dh116_deformx_pad_sdfs.py",
    "scripts/verify_dh116_cooked_convex_hulls_native.py",
    "src/go2_pvcnn/control/dh116/hand_point_kinematics.py",
    "src/go2_pvcnn/control/dh116/linear_coupling.py",
    "src/go2_pvcnn/control/dh116/usd_contract.py",
    "src/go2_pvcnn/control/dh116/usd_surface.py",
    "src/go2_pvcnn/control/dh116/deformx_dh116_colliders.py",
)


class BudgetExpired(Exception):
    pass


def load_seeds(project,geometry):
    if geometry["side"]!="right":
        raise ValueError("independent right geometry required")
    report=checked_json(Path(project)/SEED_FILE,SEED_SHA)
    if (report.get("geometry_only") is not True or report.get("qualified") is not False
            or report["source_hashes"]["right"]!=geometry["usd"]["sha256"]
            or len(report["candidates"])!=1 or report["candidates"][0]["qualified"] is not False):
        raise ValueError("original unqualified right-only reference source required")
    derived=report["candidates"][0]
    seeds=[
        dict(seed_id="original_right",source_sha256=geometry["source_sha256"]["pose_reference"],
             base_pose_wxyz=list(geometry["pose"]),active_q_rad=list(geometry["active_q"]),
             source_kind="original_right_pose_reference"),
        dict(seed_id="derived_right",source_sha256=SEED_SHA,
             base_pose_wxyz=derived["wrist_pose_w"][1],active_q_rad=derived["active_angles"][6:12],
             source_kind="right_reference_geometry_config_not_success_archive"),
    ]
    for seed in seeds:
        pose_matrix(seed["base_pose_wxyz"])
        expand(finite_array(seed["active_q_rad"],(6,)))
    return seeds,dict(path=SEED_FILE,sha256=SEED_SHA,reference=report["reference"],
        original_recording_accessed=False,historical_success_verified=False,
        source_qualification_preserved=False)


def _specs(seeds):
    for roll,pitch,bias,shift in itertools.product(ROLL,PITCH,BIAS,SHIFT):
        for seed in seeds:
            name=f"pinch-right-{seed['seed_id']}-r{roll:g}-p{pitch:g}-b{bias:g}-x{shift:g}"
            yield name,seed,roll,pitch,bias,shift


def candidates(geometry,seeds):
    if geometry.get("side")!="right" or len(seeds)!=2:
        raise ValueError("exactly two independent right seeds required")
    for name,seed,roll,pitch,bias,shift in _specs(seeds):
        q=finite_array(seed["active_q_rad"],(6,)).copy()
        q[1:3]+=bias
        pose=rotate_seed_pose(seed["base_pose_wxyz"],[.43,-.2,.308],roll,pitch)
        error=None
        try:
            expand(q)
            construction=clearance_corrected_pose(geometry,pose,q)
            pose=np.asarray(construction["base_pose_wxyz"]).copy()
        except ValueError as exc:
            error=str(exc)
            construction=None
            pose=np.asarray(pose).copy()
        pose[0]+=shift
        yield dict(candidate_id=name,seed_id=seed["seed_id"],side="right",
            independent_style="pinch",independent_seed_sha256=seed["source_sha256"],
            base_pose_wxyz=pose.tolist(),active_joint_names=list(ACTIVE),active_q_rad=q.tolist(),
            construction=construction,grid=dict(roll_deg=roll,pitch_deg=pitch,
                closure_bias_rad=bias,world_x_shift_m=shift),generation_error=error)



@contextmanager
def _commit_row_without_alarm():
    # A pending deadline may interrupt distance evaluation, but must not split
    # a completed row's result/journal/identity bookkeeping.
    previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK,previous)

def screen_grid(geometry,seeds,*,max_candidates=450,progress=None):
    if type(max_candidates) is not int or not 1<=max_candidates<=450:
        raise ValueError("candidate budget must be an integer in [1,450]")
    ids=[spec[0] for spec in _specs(seeds)][:max_candidates]
    result=dict(version=VERSION,scope="CPU_geometry_screen_only",candidate_ids=ids,
        rows=[],errors=[],path_checks=[],attempted_candidates=0,completed_candidates=0,
        status="running",unfinished_stage=None,geometry_preparation_passed=False,
        all_17_live_cooked_available=False,native_ready=False,teacher_admitted=False,
        new_physics_steps=0,optimizer_updates=0)
    finished=set()
    stage="coarse_closure_grid"
    try:
        for candidate in itertools.islice(candidates(geometry,seeds),max_candidates):
            result["attempted_candidates"]+=1
            name=candidate["candidate_id"]
            try:
                if candidate["generation_error"] is not None:
                    raise ValueError(candidate["generation_error"])
                row=measure_closure_grid(geometry,candidate,(1,2))
            except ValueError as exc:
                error=dict(candidate_id=name,error_type=type(exc).__name__,message=str(exc))
                with _commit_row_without_alarm():
                    result["errors"].append(error)
                    finished.add(name)
                    if progress:progress(dict(record_type="error",**error))
            else:
                with _commit_row_without_alarm():
                    result["rows"].append(row)
                    result["completed_candidates"]+=1
                    finished.add(name)
                    if progress:progress(dict(record_type="coarse_row",**row))
        stage="continuous_path_check"
        qualified=sorted((r for r in result["rows"] if r["coarse_geometry_passed"]),
            key=lambda r:(r["max_required_endpoint_abs_gap_m"],r["candidate_id"]))
        if qualified:
            # At most one existing 161-sample gate; never relabel a coarse pass.
            check=verify_candidate_path(geometry,qualified[0])
            with _commit_row_without_alarm():
                result["path_checks"].append(check)
                result["geometry_preparation_passed"]=check["geometry_preparation_passed"]
                if progress:progress(dict(record_type="complete_path_check",**check))
        result["status"]="complete_with_errors" if result["errors"] else "complete"
    except BudgetExpired:
        result["status"]="partial_budget"
        result["unfinished_stage"]=stage
    result["remaining_candidate_ids"]=[name for name in ids if name not in finished]
    result["coarse_passed_candidates"]=sum(r["coarse_geometry_passed"] for r in result["rows"])
    result["best_by_seed"]={}
    for seed in seeds:
        available=[r for r in result["rows"] if r["seed_id"]==seed["seed_id"]]
        if available:
            result["best_by_seed"][seed["seed_id"]]=min(available,key=lambda r:(
                len(r["coarse_blockers"]),r["max_required_endpoint_abs_gap_m"],r["candidate_id"]))["candidate_id"]
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-candidates",type=int,default=450)
    parser.add_argument("--wall-seconds",type=float,default=45.)
    args=parser.parse_args(argv)
    if (not np.isfinite(args.wall_seconds) or not 1<=args.wall_seconds<=50
            or not 1<=args.max_candidates<=450):
        parser.error("bounded CPU budget requires 1..50 seconds and 1..450 candidates")
    project=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",scope="CPU_geometry_screen_only",
        native_ready=False,teacher_admitted=False,new_physics_steps=0,optimizer_updates=0)
    def expire(signum,frame):
        raise BudgetExpired()
    prior=signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,args.wall_seconds)
    with (args.output/"progress.jsonl").open("x") as ledger:
        def progress(row):
            ledger.write(json.dumps(row,allow_nan=False)+"\n");ledger.flush()
        try:
            geometry=load_geometry(project,side="right")
            seeds,seed_evidence=load_seeds(project,geometry)
            result=screen_grid(geometry,seeds,max_candidates=args.max_candidates,progress=progress)
            result.update(seed_evidence=seed_evidence,seeds=seeds,
                geometry_source_sha256=geometry["source_sha256"])
        except BudgetExpired:
            result["setup_timeout"]=True
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,prior)
    result.update(elapsed_screen_s=time.monotonic()-started,
        requested_budget=dict(max_candidates=args.max_candidates,soft_wall_seconds=args.wall_seconds,
            external_process_timeout_seconds=60,cpu_threads=2),
        source_sha256={name:digest(project/name) for name in SOURCES},
        progress_sha256=digest(args.output/"progress.jsonl"))
    with (args.output/"result.json").open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({key:result.get(key) for key in (
        "status","elapsed_screen_s","attempted_candidates","completed_candidates",
        "coarse_passed_candidates","geometry_preparation_passed","native_ready")}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
