#!/usr/bin/env python3
"""One source-bound Tripod constrained direction per side, no native qualification."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_pinch_initial_pose import (
    digest,checked_json,finite_array,load_geometry,world_vertices,convex_patch_closest)
from scripts.solve_dh116_tripod_joint_root_direction import (
    SCALES,REQUIRED,envelope_support,EvaluationBudget,EvaluationLimit)
from scripts.screen_dh116_tripod_proximal import TRIPOD_SHA
from scripts.prepare_dh116_pinch_closure_margin import sample_contact
from scripts.screen_dh116_right_pinch_closure import BudgetExpired,_commit_row_without_alarm
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,active_limits,expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

VERSION="dh116-tripod-sdf-constrained-direction-v1"
RUNTIME_FLOOR=CLEARANCE_M
ROOT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
PARENT=ROOT/"tripod-joint-root-direction-20261113-3ar3y5k7/run01/result.json"
PARENT_SHA="726ca01dc554cb1a2a8a1ee03222c240118b536ae4cd7ad7348a19ca36fa39ba"
SDF_DIR=ROOT/"flex-f4-sdf-register-rK7gw0/bake"
SDF_SHA="db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab"


def load_parent(path=PARENT):
    r=checked_json(Path(path),PARENT_SHA)
    if r["status"]!="complete" or r["side_evaluation_attempts"]!={"left":45,"right":45}:
        raise ValueError("complete original Tripod direction ledger required")
    return r


def decode(parent,x):
    x=finite_array(x,(10,))
    if (parent["independent_style"]!="tripod" or parent["independent_seed_sha256"]!=TRIPOD_SHA
            or parent["side"] not in ("left","right") or np.any(abs(x)>1+1e-12)):
        raise ValueError("independent original Tripod source and bounded direction required")
    pose=finite_array(parent["base_pose_wxyz"],(7,)).copy()
    pose[:3]+=SCALES[:3]*x[:3]
    rot=Rotation.from_rotvec(SCALES[3:6]*x[3:6])*Rotation.from_quat(pose[[4,5,6,3]])
    pose[3:]=np.roll(rot.as_quat(),1)
    q=finite_array(parent["active_q_rad"],(6,)).copy();q[:4]+=SCALES[6:]*x[6:]
    expand(q)
    return dict(side=parent["side"],independent_style="tripod",independent_seed_sha256=TRIPOD_SHA,
        base_pose_wxyz=pose.tolist(),active_joint_names=list(ACTIVE),active_q_rad=q.tolist(),
        normalized_delta=x.tolist(),native_ready=False,reset_verified=False)


def bounds(parent):
    low,high=active_limits();q=np.array(parent["active_q_rad"])
    lo,hi=-np.ones(10),np.ones(10)
    lo[6:]=np.maximum(lo[6:],(low[:4]-q[:4])/SCALES[6:])
    hi[6:]=np.minimum(hi[6:],(high[:4]-q[:4])/SCALES[6:])
    return lo,hi


def grids_for(geometry,summary):
    result={};bindings={}
    for name in geometry["cooked_pads"]:
        key=geometry["side"]+"."+name;r=summary["pads"][key];p=SDF_DIR/(key+".npz")
        if (digest(p)!=r["sdf_file_sha256"] or r["gpu_row_sha256"]!=geometry["cooked_pads"][name]["capture_sha256"]
                or r["spacing_m"]!=.001 or r["rod_radius_m"]!=.005
                or r["frame"]!="original_pad_coincident_local_basis_m"):
            raise ValueError("original per-side 1mm locked SDF/cooked identity differs")
        with np.load(p,allow_pickle=False) as d:result[name]={k:d[k].copy() for k in d.files}
        bindings[name]=dict(path=str(p),sha256=r["sdf_file_sha256"],capture_sha256=r["gpu_row_sha256"])
    return result,bindings


def qualified_constraints(values):
    v=finite_array(values)
    return bool(len(v)>0 and np.all(v>=0))


def measure(geometry,parent,grids,x):
    candidate=decode(parent,x);q=candidate["active_q_rad"];pose=candidate["base_pose_wxyz"]
    layout=geometry["reference"]["reference_layout"];top=layout["table_center"][2]+layout["table_size"][2]/2
    source=world_vertices(geometry,pose,q);cooked=world_vertices(geometry,pose,q,cooked=True)
    delta=np.eye(4)
    from scripts.audit_dh116_pinch_initial_pose import pose_matrix
    delta=pose_matrix(pose)@np.linalg.inv(pose_matrix(geometry["pose"]))
    fk={k.rsplit("/",1)[-1]:delta@v for k,v in geometry["fk"].poses(q).items()}
    nodes=np.array(layout["wire_points"]);nodes[:,2]=top+.005
    constraints=[];labels=[];tables={};envelopes={}
    def add(label,value):labels.append(label);constraints.append(float(value))
    for kind,shapes in (("source",source),("cooked",cooked)):
        for name,v in shapes.items():
            key=kind+":"+name
            tables[key]=float(v[:,2].min()-top)
            envelopes[key]=envelope_support(v,layout,.005)
            add(key+":table_2p1mm",(tables[key]-RUNTIME_FLOOR)/.001)
            add(key+":initial_rod_lower",envelopes[key]["surface_gap_lower_bound_m"]/.001)
    pads={};normals=[]
    for name in REQUIRED:
        sdf=sample_contact(grids[name],fk[name],nodes)
        if sdf["minimum_gap_m"] is None or sdf["gradient_norm_at_minimum"]<=0:
            raise ValueError("required Tripod pad lacks supported node SDF direction")
        qp=convex_patch_closest(cooked[name],nodes[0],[nodes[-1]-nodes[0]])
        upper=qp["distance_m"]-.005+2e-6
        add(name+":strict_hull_upper_50um",(.00005-upper)/.001)
        add(name+":node_sdf_precontact_50um",(sdf["minimum_gap_m"]-.00005)/.001)
        add(name+":node_sdf_reach_0p5mm",(.0005-sdf["minimum_gap_m"])/.001)
        pads[name]=dict(node_sdf=sdf,continuous_hull_feasible_gap_m=qp["distance_m"]-.005,
            continuous_hull_upper_m=upper,geometric_direction=qp["geometric_axis_to_hull_direction"])
        normals.append(qp["geometric_axis_to_hull_direction"])
    dots=[]
    for i in (1,2):
        dot=None if normals[0] is None or normals[i] is None else float(np.dot(normals[0],normals[i]))
        dots.append(dot);add("thumb_partner"+str(i)+":opposed",-1. if dot is None else -.200001-dot)
    values=np.asarray(constraints)
    return dict(candidate=candidate,constraint_labels=labels,constraint_values=constraints,
        all_initial_constraints_passed=qualified_constraints(values),
        original_runtime_floor_m=RUNTIME_FLOOR,table_gap_m=tables,all_shape_envelope_lower=envelopes,
        required_pads=pads,geometric_thumb_partner_dot=dots,
        violated_constraints={k:float(v) for k,v in zip(labels,values) if v<0},
        sdf_precontact_scope="diagnostic_50_to_500um_node_gap_not_new_physics_threshold",
        native_ready=False,continuous_path_verified=False,reset_verified=False,
        hand_y_bounds_m=[min(v[:,1].min() for v in [*source.values(),*cooked.values()]),
                         max(v[:,1].max() for v in [*source.values(),*cooked.values()])])


def constrained_direction(evaluate,lower,upper):
    lo,hi=finite_array(lower,(10,)),finite_array(upper,(10,))
    if np.any(lo>-.01) or np.any(hi<.01):raise ValueError("FD neighborhood crosses original joint limit")
    base=evaluate(np.zeros(10));labels=base["constraint_labels"]
    def values(row):
        if row["constraint_labels"]!=labels:raise ValueError("constraint identity/order changed")
        return finite_array(row["constraint_values"],(len(labels),))
    c=values(base);jac=[]
    for h in (.01,.005):
        jac.append(np.stack([(values(evaluate(np.eye(10)[j]*h))-values(evaluate(-np.eye(10)[j]*h)))/(2*h)
                             for j in range(10)],axis=1))
    relative=float(np.linalg.norm(jac[1]-jac[0])/max(np.linalg.norm(jac[1]),1e-12))
    violation=lambda v:float(max(0.,-np.min(v)))
    result=dict(baseline=base,jacobians=[j.tolist() for j in jac],
        jacobian_relative_difference=relative,linear_solver_calls=0,
        baseline_common_violation=violation(c),best=base,best_x=np.zeros(10).tolist(),
        best_common_violation=violation(c),trials=[],global_infeasibility_proven=False)
    if relative>.2:
        result["rejection"]="two_scale_constraint_jacobian_unstable";return result
    # One LP minimizes a common normalized constraint violation; no soft reward
    # is allowed to relabel a physical violation as a passing candidate.
    fit=linprog(np.r_[np.zeros(10),1.],
        A_ub=-np.column_stack((jac[-1],np.ones(len(c)))),b_ub=c,
        bounds=[*zip(lo,hi),(0,None)],method="highs",
        options={"time_limit":1.,"maxiter":200,"threads":2})
    result.update(linear_solver_calls=1,linear_solver_status=int(fit.status),linear_solver_message=fit.message)
    if not fit.success:return result
    direction=fit.x[:10]
    result.update(direction=direction.tolist(),predicted_common_violation=float(fit.x[-1]),
        predicted_constraints=(c+jac[-1]@direction).tolist(),
        linearized_only_not_nonlinear_or_global_certificate=True)
    for amount in (1.,.5,.25,.125):
        row=evaluate(direction*amount);value=violation(values(row))
        result["trials"].append(dict(fraction=amount,common_violation=value,measurement=row))
        if value<result["best_common_violation"]:
            result.update(best=row,best_x=(direction*amount).tolist(),best_common_violation=value)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wall-seconds",type=float,default=50.)
    args=p.parse_args(argv)
    if not np.isfinite(args.wall_seconds) or not 1<=args.wall_seconds<=50:p.error("CPU soft limit 1..50s")
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial_setup",sides={},evaluation_records=[],attempts={},
        native_ready=False,teacher_admitted=False,reset_proposal_created=False,physics_steps=0,training_updates=0)
    def expire(*unused):raise BudgetExpired()
    handler=signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,args.wall_seconds)
    try:
        project=Path(__file__).resolve().parents[1];old=load_parent()
        summary=checked_json(SDF_DIR/"result.json",SDF_SHA)
        result.update(parent_result_path=str(PARENT),parent_result_sha256=PARENT_SHA,
            source_sha256={str(project/n):digest(project/n) for n in (
                "scripts/solve_dh116_tripod_sdf_constrained_direction.py",
                "scripts/solve_dh116_tripod_joint_root_direction.py",
                "scripts/prepare_dh116_pinch_closure_margin.py",
                "scripts/audit_dh116_power_closure_constraints.py",
                "scripts/audit_dh116_pinch_initial_pose.py",
                "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py")},
            sdf_summary_sha256=SDF_SHA,parameter_scales=SCALES.tolist(),geometry_sources={},sdf_bindings={})
        for side in ("left","right"):
            g=load_geometry(project,side=side)
            if g["source_sha256"]!=old["geometry_source_sha256"][side]:raise ValueError("Tripod geometry source differs")
            grids,bindings=grids_for(g,summary)
            result["geometry_sources"][side]=g["source_sha256"];result["sdf_bindings"][side]=bindings
            parent=old["sides"][side]["best"]["candidate"]
            def query(x):
                row=measure(g,parent,grids,x)
                with _commit_row_without_alarm():result["evaluation_records"].append(dict(side=side,x=np.asarray(x).tolist(),measurement=row))
                return row
            budget=EvaluationBudget(query)
            try:
                fit=constrained_direction(budget,*bounds(parent))
                with _commit_row_without_alarm():result["sides"][side]=fit
            finally:result["attempts"][side]=budget.count
        result["status"]="complete"
    except BudgetExpired:result["status"]="partial_timeout"
    except (ValueError,EvaluationLimit) as exc:result.update(status="diagnostic_error",error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
    result["elapsed_s"]=time.monotonic()-start
    with (args.output/"result.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result[k] for k in ("status","elapsed_s","attempts")}))
    return 1 if result["status"]=="diagnostic_error" else 0


if __name__=="__main__":raise SystemExit(main())
