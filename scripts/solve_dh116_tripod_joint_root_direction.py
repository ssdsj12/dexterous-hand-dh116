#!/usr/bin/env python3
"""One bounded joint/root Tripod direction per side; candidate diagnostics only."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_pinch_initial_pose import (
    digest,finite_array,load_geometry,world_vertices,convex_patch_closest)
from scripts.audit_dh116_power_closure_constraints import support_plane_bound
from scripts.screen_dh116_tripod_proximal import (
    PARENTS,TRIPOD_SHA,PRIOR,PRIOR_SHA,load_parents,SIDES)
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm,SOURCES
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,active_limits,expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-tripod-joint-root-direction-v1"
SCALES=np.array([.005,.005,.005,.1,.1,.1,.1,.1,.1,.1])
REQUIRED=("finger14_link","finger23_link","finger33_link")
STRICT=.00005
COARSE=.0005
FD_STEP=.01
MAX_EVALUATIONS_PER_SIDE=45


class EvaluationLimit(RuntimeError):
    pass


class EvaluationBudget:
    def __init__(self,query,*,maximum=MAX_EVALUATIONS_PER_SIDE):
        if type(maximum) is not int or not 1<=maximum<=MAX_EVALUATIONS_PER_SIDE:
            raise ValueError("one to forty-five geometry evaluations per side")
        self.query,self.maximum,self.count=query,maximum,0
    def __call__(self,x):
        if self.count>=self.maximum:raise EvaluationLimit("geometry evaluation cap")
        self.count+=1
        return self.query(x)


def decode(parent,x):
    x=finite_array(x,(10,))
    if (np.any(abs(x)>1+1e-12) or parent["candidate_id"]!=PARENTS[parent["side"]]
            or parent["independent_style"]!="tripod" or parent["independent_seed_sha256"]!=TRIPOD_SHA):
        raise ValueError("original side-specific parent and bounded ten-vector required")
    pose=finite_array(parent["base_pose_wxyz"],(7,)).copy()
    pose[:3]+=SCALES[:3]*x[:3]
    rotation=Rotation.from_rotvec(SCALES[3:6]*x[3:6])*Rotation.from_quat(pose[[4,5,6,3]])
    pose[3:]=np.roll(rotation.as_quat(),1)
    q=np.asarray(parent["active_q_rad"])+parent["active_delta_rad"]
    q=q.copy();q[:4]+=SCALES[6:]*x[6:]
    expand(q)
    return dict(side=parent["side"],parent_candidate_id=parent["candidate_id"],
        parent_center="immutable_previous_three_digit_closure_endpoint",
        independent_style="tripod",independent_seed_sha256=TRIPOD_SHA,
        base_pose_wxyz=pose.tolist(),active_joint_names=list(ACTIVE),active_q_rad=q.tolist(),
        normalized_joint_root_delta=x.tolist(),reset_verified=False,native_ready=False)


def parameter_bounds(parent):
    q=np.asarray(parent["active_q_rad"])+parent["active_delta_rad"]
    low,high=active_limits()
    lower,upper=-np.ones(10),np.ones(10)
    lower[6:]=np.maximum(-1,(low[:4]-q[:4])/SCALES[6:])
    upper[6:]=np.minimum(1,(high[:4]-q[:4])/SCALES[6:])
    if np.any(lower>0) or np.any(upper<0):
        raise ValueError("original parent endpoint outside original active limits")
    return lower,upper


def envelope_support(vertices,layout,radius):
    # With the reused directions n_y=0, these two diagonal corners have
    # exactly the same max projection as all four vertical rectangle corners.
    axis=np.array(layout["wire_points"])[[0,-1]].copy()
    axis[0,2]=layout["table_center"][2]+layout["table_size"][2]/2+radius
    witness=support_plane_bound(vertices,axis,radius)
    if witness["direction_world"][1]!=0:
        raise ValueError("XZ-only support directions required for the rectangle projection")
    return dict(witness,scope="vertical_geometric_envelope_support_lower_bound",
                surface_gap_lower_bound_m=witness["surface_gap_lower_bound_m"]-2e-6)


def endpoint_gates(row):
    intervals=np.asarray(row["required_gap_intervals_m"])
    opposed=all(x is not None and np.isfinite(x) and x<-.2 for x in row["geometric_thumb_partner_dot"])
    coarse=bool(np.all(intervals[:,0]>=-COARSE) and np.all(intervals[:,1]<=COARSE) and opposed)
    strict=bool(np.all(intervals[:,0]>=-STRICT) and np.all(intervals[:,1]<=STRICT) and opposed)
    static=bool(row["minimum_table_gap_m"]>=.0019)
    runtime=bool(row["minimum_table_gap_m"]>=CLEARANCE_M)
    nonrequired=bool(row["minimum_nonrequired_support_gap_m"]>=0)
    return dict(coarse_endpoint_passed=coarse,strict_endpoint_passed=strict,
        static_1p9mm_passed=static,runtime_2p1mm_passed=runtime,
        nonrequired_rod_envelope_clear=nonrequired,
        candidate_all_endpoint_constraints_passed=bool(strict and runtime and nonrequired),
        scoped_constraints="required_three_pad_intervals_directions_nonrequired_envelope_table_original_limits",
        self_collision_and_arm_reach_verified=False,continuous_path_verified=False,
        reset_verified=False,native_ready=False,teacher_admitted=False)


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
    residual.extend(1. if d is None else max(0,d+.2)/.2 for d in dots);labels.extend(["thumb_index_direction","thumb_middle_direction"])
    for kind,shapes in table.items():
        for name,gap in shapes.items():
            residual.append(min(0,gap-CLEARANCE_M)/.001);labels.append(kind+":"+name+":table")
    for kind,shapes in gaps.items():
        for name,witness in shapes.items():
            residual.append(min(0,witness["surface_gap_lower_bound_m"])/.001)
            labels.append(kind+":"+name+":nonrequired_rod")
    all_vertices=list(source.values())+list(cooked.values())
    row=dict(candidate=candidate,required_pads=required,geometric_thumb_partner_dot=dots,
        required_gap_intervals_m=[required[n]["surface_gap_interval_m"] for n in REQUIRED],
        table_gap_by_shape_m=table,nonrequired_rod_support=gaps,
        minimum_table_gap_m=min(v for shapes in table.values() for v in shapes.values()),
        minimum_nonrequired_support_gap_m=min(v["surface_gap_lower_bound_m"] for shapes in gaps.values() for v in shapes.values()),
        hand_y_bounds_m=[float(min(v[:,1].min() for v in all_vertices)),float(max(v[:,1].max() for v in all_vertices))],
        residual=residual,residual_labels=labels)
    row.update(endpoint_gates(row))
    return row


def local_direction(evaluate,*,lower,upper):
    lower=finite_array(lower,(10,));upper=finite_array(upper,(10,))
    if np.any(lower>-FD_STEP) or np.any(upper<FD_STEP):
        raise ValueError("central difference neighborhood crosses an original limit")
    zero=np.zeros(10);baseline=evaluate(zero);r=np.asarray(baseline["residual"])
    jacobians=[]
    for h in (FD_STEP,FD_STEP/2):
        columns=[]
        for i in range(10):
            delta=np.zeros(10);delta[i]=h
            plus=np.asarray(evaluate(delta)["residual"]);minus=np.asarray(evaluate(-delta)["residual"])
            columns.append((plus-minus)/(2*h))
        jacobians.append(np.stack(columns,axis=1))
    coarse,fine=jacobians
    relative=float(np.linalg.norm(coarse-fine)/max(np.linalg.norm(fine),1e-12))
    objective=float(.5*np.dot(r,r))
    result=dict(baseline=baseline,baseline_objective=objective,jacobian_coarse=coarse.tolist(),
        jacobian_fine=fine.tolist(),jacobian_relative_difference=relative,
        jacobian_stability_limit=.2,jacobian_supported=relative<=.2,
        model_supported=False,observed_descent=False,best_x=zero.tolist(),
        best_objective=objective,best=baseline,trials=[],linear_solver_calls=0)
    if relative>.2 or objective<1e-14:return result
    fit=lsq_linear(np.vstack((fine,.05*np.eye(10))),np.r_[-r,np.zeros(10)],
                   bounds=(lower,upper),max_iter=40,tol=1e-9)
    direction=fit.x;predicted=float(.5*np.linalg.norm(r+fine@direction)**2)
    result.update(direction=direction.tolist(),linear_solver_calls=1,
        linear_solver_iterations=int(fit.nit),linear_solver_success=bool(fit.success),
        predicted_objective=predicted,
        model_supported=bool(fit.success and predicted<objective-1e-10))
    if not result["model_supported"]:return result
    for amount in (1.,.5,.25,.125):
        x=amount*direction;row=evaluate(x)
        value=float(.5*np.dot(row["residual"],row["residual"]))
        result["trials"].append(dict(line_fraction=amount,x=x.tolist(),objective=value,measurement=row))
        if value<result["best_objective"]:
            result.update(best_x=x.tolist(),best_objective=value,best=row,observed_descent=True)
    return result


def pair_gates(rows):
    complete=all(s in rows and rows[s]["candidate_all_endpoint_constraints_passed"] for s in SIDES)
    separation=(rows["left"]["hand_y_bounds_m"][0]-rows["right"]["hand_y_bounds_m"][1]) if complete else None
    return dict(bilateral_endpoint_passed=bool(complete and separation>0),
        hand_y_separation_m=separation,continuous_path_verified=False,
        reset_proposal_created=False,native_ready=False,teacher_admitted=False)


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
            parents,prior=load_parents(PRIOR,project,geometries)
            result.update(parent_result_sha256=PRIOR_SHA,parent_candidates=parents,
                geometry_source_sha256={s:g["source_sha256"] for s,g in geometries.items()},
                parameter_scales=SCALES.tolist(),parameter_names=[
                    "root_x","root_y","root_z","world_rotation_x","world_rotation_y","world_rotation_z",
                    "finger11","finger12_thumb","finger21_index","finger31_middle"])
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
                    direction=local_direction(budget,lower=lower,upper=upper)
                    with _commit_row_without_alarm():result["sides"][side]=direction
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
            "scripts/screen_dh116_tripod_proximal.py","scripts/audit_dh116_power_closure_constraints.py",
            "scripts/solve_dh116_tripod_joint_root_direction.py",
            "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py")},
        progress_sha256=digest(args.output/"progress.jsonl"))
    with (args.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result[k] for k in ("status","elapsed_s","side_evaluation_attempts","pair")}))
    return 0 if result["status"]!="diagnostic_error" else 1


if __name__=="__main__":
    raise SystemExit(main())
