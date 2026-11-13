#!/usr/bin/env python3
"""One independent Power joint/palm direction per side, at most 90 residuals."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_pinch_initial_pose import (
    digest,finite_array,load_geometry,world_vertices,convex_patch_closest)
from scripts.audit_dh116_power_closure_constraints import (
    POWER_SHA,PRIOR,PRIOR_SHA,load_selection,support_plane_bound)
from scripts.solve_dh116_tripod_joint_root_direction import (
    local_direction,EvaluationBudget,EvaluationLimit,envelope_support,endpoint_gates,pair_gates)
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm,SOURCES
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,active_limits,expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-power-joint-palm-direction-v1"
SIDES=("left","right")
SELECTED={"left":15,"right":1}
PARENT_IDS={s:f"power-constraint-{s}-{i}" for s,i in SELECTED.items()}
REQUIRED=("finger14_link","finger23_link","finger33_link","finger43_link","finger53_link")
SCALES=np.array([.015,.015,.015,.1,.1,.1,.1,.1,.1,.1])


def validate_parent(parent):
    side=parent["side"]
    if (side not in SIDES or parent["candidate_id"]!=PARENT_IDS[side]
            or parent["archive_start_index"]!=SELECTED[side]
            or parent["independent_style"]!="power" or parent["independent_seed_sha256"]!=POWER_SHA
            or parent["mode"]!="minimum_table_clearance_root"
            or parent["initial_all_shape_table_clearance_m"]<.0019
            or min(v for shapes in parent["initial_rod_settling_gaps_m"].values() for v in shapes.values())<0):
        raise ValueError("exact nonoverlapping original Power source seed required")
    expand(finite_array(parent["active_q_rad"],(6,)))
    pose=finite_array(parent["base_pose_wxyz"],(7,))
    if not np.isclose(np.linalg.norm(pose[3:]),1,atol=1e-10,rtol=0):
        raise ValueError("original Power WXYZ root required")


def load_power_parents(path,project,geometries):
    selected,prior=load_selection(path,project,geometries)
    parents={}
    for side in SIDES:
        rows=[r for r in selected if r["side"]==side and r["archive_start_index"]==SELECTED[side]]
        if len(rows)!=1:raise ValueError("unique independent Power parent required")
        validate_parent(rows[0]);parents[side]=rows[0]
    return parents,prior


def decode(parent,x):
    validate_parent(parent);x=finite_array(x,(10,))
    if np.any(abs(x)>1+1e-12):raise ValueError("bounded ten-variable Power direction required")
    pose=finite_array(parent["base_pose_wxyz"],(7,)).copy()
    shift=SCALES[:3]*x[:3];angle=float(SCALES[3]*x[3])
    pose[:3]+=shift
    rotation=Rotation.from_rotvec([0.,angle,0.])*Rotation.from_quat(pose[[4,5,6,3]])
    pose[3:]=np.roll(rotation.as_quat(),1)
    delta=SCALES[4:]*x[4:];q=np.asarray(parent["active_q_rad"])+delta
    expand(q)
    return dict(side=parent["side"],parent_candidate_id=parent["candidate_id"],
        archive_start_index=parent["archive_start_index"],
        parent_center="original_nonoverlapping_power_q_no_added_closure_delta",
        independent_style="power",independent_seed_sha256=POWER_SHA,
        base_pose_wxyz=pose.tolist(),active_joint_names=list(ACTIVE),active_q_rad=q.tolist(),
        normalized_joint_palm_delta=x.tolist(),root_translation_m=shift.tolist(),
        world_y_rotation_rad=angle,joint_seed_delta_rad=delta.tolist(),
        root_translation_norm_m=float(np.linalg.norm(shift)),
        reset_verified=False,native_ready=False)


def parameter_bounds(parent):
    validate_parent(parent);q=np.asarray(parent["active_q_rad"])
    low,high=active_limits();lower,upper=-np.ones(10),np.ones(10)
    lower[4:]=np.maximum(-1,(low-q)/SCALES[4:]);upper[4:]=np.minimum(1,(high-q)/SCALES[4:])
    return lower,upper


def power_endpoint_gates(row):
    if np.asarray(row["required_gap_intervals_m"]).shape!=(5,2) or len(row["geometric_thumb_partner_dot"])!=4:
        raise ValueError("all five Power pad intervals and four thumb partner directions required")
    return dict(endpoint_gates(row),scoped_constraints=
        "required_five_pad_intervals_directions_nonrequired_envelope_table_original_limits")


def measure(geometry,parent,x):
    candidate=decode(parent,x);q=candidate["active_q_rad"];pose=candidate["base_pose_wxyz"]
    layout=geometry["reference"]["reference_layout"];radius=geometry["old_candidate"]["rod"]["radius_m"]
    top=layout["table_center"][2]+layout["table_size"][2]/2
    source=world_vertices(geometry,pose,q);cooked=world_vertices(geometry,pose,q,cooked=True)
    sets={"source":source,"cooked":cooked}
    axis=np.array(layout["wire_points"])[[0,-1]].copy();axis[:,2]=top+radius
    required={}
    for name in REQUIRED:
        qp=convex_patch_closest(cooked[name],axis[0],[axis[-1]-axis[0]])
        support=support_plane_bound(cooked[name],axis,radius)
        lower=support["surface_gap_lower_bound_m"]-2e-6
        upper=qp["distance_m"]-radius+2e-6
        if lower>upper+1e-10:raise ValueError("required-pad support bound exceeds feasible distance")
        required[name]=dict(qp=qp,support=support,surface_gap_interval_m=[lower,upper],
                            estimated_surface_gap_m=qp["distance_m"]-radius)
    thumb=required[REQUIRED[0]]["qp"]["geometric_axis_to_hull_direction"]
    dots=[]
    for name in REQUIRED[1:]:
        partner=required[name]["qp"]["geometric_axis_to_hull_direction"]
        dots.append(None if thumb is None or partner is None else float(np.dot(thumb,partner)))
    table={kind:{n:float(v[:,2].min()-top) for n,v in shapes.items()} for kind,shapes in sets.items()}
    gaps={kind:{n:envelope_support(v,layout,radius) for n,v in shapes.items() if n not in REQUIRED}
          for kind,shapes in sets.items()}
    residual=[(required[n]["estimated_surface_gap_m"]-.000025)/.001 for n in REQUIRED]
    labels=[n+":estimated_gap_target_25um" for n in REQUIRED]
    residual.extend(1. if d is None else max(0,d+.2)/.2 for d in dots)
    labels.extend("thumb_"+n+"_direction" for n in REQUIRED[1:])
    for kind,shapes in table.items():
        for name,gap in shapes.items():
            residual.append(min(0,gap-CLEARANCE_M)/.001);labels.append(kind+":"+name+":table")
    for kind,shapes in gaps.items():
        for name,witness in shapes.items():
            residual.append(min(0,witness["surface_gap_lower_bound_m"])/.001)
            labels.append(kind+":"+name+":nonrequired_rod")
    vertices=list(source.values())+list(cooked.values())
    row=dict(candidate=candidate,required_pads=required,geometric_thumb_partner_dot=dots,
        required_gap_intervals_m=[required[n]["surface_gap_interval_m"] for n in REQUIRED],
        table_gap_by_shape_m=table,nonrequired_rod_support=gaps,
        minimum_table_gap_m=min(v for group in table.values() for v in group.values()),
        minimum_nonrequired_support_gap_m=min(v["surface_gap_lower_bound_m"] for group in gaps.values() for v in group.values()),
        hand_y_bounds_m=[float(min(v[:,1].min() for v in vertices)),float(max(v[:,1].max() for v in vertices))],
        residual=residual,residual_labels=labels)
    row.update(power_endpoint_gates(row))
    return row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,default=50.)
    args=parser.parse_args(argv)
    if not np.isfinite(args.wall_seconds) or not 1<=args.wall_seconds<=50:
        parser.error("CPU deadline must be 1..50 seconds")
    project=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",sides={},evaluation_records=[],
        side_evaluation_attempts={},native_ready=False,teacher_admitted=False,
        physics_steps=0,training_optimizer_updates=0,reset_proposal_created=False)
    def expire(signum,frame):raise BudgetExpired()
    handler=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,args.wall_seconds)
    with (args.output/"progress.jsonl").open("x") as ledger:
        try:
            geometries={s:load_geometry(project,side=s) for s in SIDES}
            parents,prior=load_power_parents(PRIOR,project,geometries)
            result.update(parent_result_sha256=PRIOR_SHA,parent_candidates=parents,
                geometry_source_sha256={s:g["source_sha256"] for s,g in geometries.items()},
                parameter_scales=SCALES.tolist(),parameter_names=[
                    "root_x","root_y","root_z","world_rotation_y",*ACTIVE])
            for side in SIDES:
                def query(x):
                    row=measure(geometries[side],parents[side],x)
                    record=dict(side=side,evaluation_index=len([r for r in result["evaluation_records"] if r["side"]==side]),
                                x=np.asarray(x).tolist(),measurement=row)
                    with _commit_row_without_alarm():
                        result["evaluation_records"].append(record)
                        ledger.write(json.dumps(record,allow_nan=False)+"\n");ledger.flush()
                    return row
                budget=EvaluationBudget(query)
                try:
                    lower,upper=parameter_bounds(parents[side])
                    fit=local_direction(budget,lower=lower,upper=upper)
                    with _commit_row_without_alarm():result["sides"][side]=fit
                finally:
                    result["side_evaluation_attempts"][side]=budget.count
            result["status"]="complete"
        except BudgetExpired:
            result["status"]="partial_timeout"
        except (ValueError,EvaluationLimit) as exc:
            result.update(status="diagnostic_error",error=str(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
    result.update(elapsed_s=time.monotonic()-started,
        pair=pair_gates({s:r["best"] for s,r in result["sides"].items()}),
        budget=dict(max_total_geometry_evaluations=90,max_per_side=45,
                    max_linear_solver_iterations_per_side=40,soft_wall_seconds=args.wall_seconds),
        source_sha256={n:digest(project/n) for n in (*SOURCES,
            "scripts/audit_dh116_power_closure_constraints.py","scripts/screen_dh116_tripod_proximal.py",
            "scripts/solve_dh116_tripod_joint_root_direction.py","scripts/solve_dh116_power_joint_palm_direction.py",
            "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py")},
        progress_sha256=digest(args.output/"progress.jsonl"))
    with (args.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result[k] for k in ("status","elapsed_s","side_evaluation_attempts","pair")}))
    return 0 if result["status"]!="diagnostic_error" else 1


if __name__=="__main__":
    raise SystemExit(main())
