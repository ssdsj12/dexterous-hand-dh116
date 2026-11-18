#!/usr/bin/env python3
"""CPU-only Tripod endpoint closure using full original geometry and locked SDF.

An optimizer proposal is never a native contact or continuous-path certificate.
Original actuator/physical limits stay immutable. Only a new independent reset
and endpoint are proposed; no simulator or native recipe is launched.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned

VERSION="dh116-tripod-full-geometry-closure-v3"
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-tracking-margin-20261114-run03/result.json")
PARENT_SHA="85c84a028f36a094719d2c5f27967233fef5784f5ee5cbc63456597059ff30a9"
REQUIRED=("finger14_link","finger23_link","finger33_link")
CONFIG=dict(version=VERSION,mode="cpu_only",native_execution=False,
    translation_scales_m=[.0004,.0004],reset_rotation_scales_rad=[.01]*3,joint_scales_rad=[.01]*4,
    table_clearance_m=.0021,native_required_gap_m=-.0003,
    required_planning_gap_m=-.00015,non_target_gap_m=0.,
    reserve_m=.00001,sdf_target_gap_m=-.000015,
    physics_hz=480,control_hz=60,motion_steps=200,
    maximum_evaluations=600,maximum_seconds=60.,teacher_admission=False)

class BudgetExceeded(RuntimeError):pass

def validate_config(c):
    if c!=CONFIG:raise ValueError("independent frozen CPU proposal and unchanged native guards required")

def decode(parent,x):
    x=np.asarray(x,dtype=float)
    if x.shape!=(9,) or not np.isfinite(x).all() or (abs(x)>1.+1e-12).any():
        raise ValueError("bounded x/z, constant reset rotation and four active-joint changes")
    pose=np.array(parent["end_pose_wxyz"],dtype=float);q=np.array(parent["q1_rad"],dtype=float)
    if pose.shape!=(7,) or q.shape!=(6,) or not np.isfinite(pose).all() or not np.isfinite(q).all():
        raise ValueError("finite original endpoint")
    pose[[0,2]]+=np.array(CONFIG["translation_scales_m"])*x[:2]
    rot=Rotation.from_rotvec(np.array(CONFIG["reset_rotation_scales_rad"])*x[2:5])*Rotation.from_quat(pose[[4,5,6,3]])
    pose[3:]=np.roll(rot.as_quat(),1)
    q[:4]+=np.array(CONFIG["joint_scales_rad"])*x[5:]
    lo,hi=active_limits()
    if (q<lo).any() or (q>hi).any():raise ValueError("original active joint limit")
    expand(q)
    return pose,q

def physical_constraints(table,rod,pads,c):
    validate_config(c)
    if set(pads)!=set(REQUIRED) or not all(v is not None and math.isfinite(v) for v in pads.values()):
        raise ValueError("three finite supported SDF pad predictions required")
    out={}
    for n,v in table.items():out["table:"+n]=float(v-c["table_clearance_m"]-c["reserve_m"])
    for n,v in rod.items():
        threshold=c["required_planning_gap_m"] if n.split(":")[-1] in REQUIRED else c["non_target_gap_m"]
        out["rod:"+n]=float(v-threshold-c["reserve_m"])
    for n,v in pads.items():out["sdf_target:"+n]=float(c["sdf_target_gap_m"]-v)
    if not all(math.isfinite(v) for v in out.values()):raise ValueError("finite geometric margins")
    return out

def candidate_admission(constraints):
    return dict(static_constraints_passed=bool(constraints) and all(math.isfinite(v) and v>=0 for v in constraints.values()),
        continuous_path_proved=False,tracking_qualified=False,native_ready=False,
        actual_contact_observed=False,teacher_admission=False)

def bounded_search(query,start,*,maximum_evaluations,maximum_seconds):
    if type(maximum_evaluations) is not int or not 1<=maximum_evaluations<=600 or not 0<maximum_seconds<=60:
        raise ValueError("bounded CPU search")
    started=time.monotonic();cache={};identity=None;best=None;count=0
    def evaluate(x):
        nonlocal identity,best,count
        key=np.asarray(x,dtype="<f8").tobytes()
        if key in cache:return cache[key]
        if count>=maximum_evaluations or time.monotonic()-started>=maximum_seconds:raise BudgetExceeded()
        r=query(x);c=r["constraints"]
        if not c or not all(math.isfinite(v) for v in c.values()):raise ValueError("finite complete constraints")
        if identity is None:identity=tuple(c)
        elif tuple(c)!=identity:raise ValueError("constraint identity changed")
        count+=1;r=deepcopy(r);r["admission"]=candidate_admission(c)
        violation=np.minimum(np.array(list(c.values()),dtype=float),0)
        # All constraints are metres; rank feasibility before small change size.
        r["violation_score"]=float(violation@violation)
        r["minimum_margin_m"]=float(min(c.values()));cache[key]=r
        rank=(max(0.,-r["minimum_margin_m"]),r["violation_score"],float(np.asarray(x)@np.asarray(x)))
        if best is None or rank<best[0]:best=(rank,r)
        return r
    numerical=[];termination="iteration_budget_exhausted";x=np.asarray(start,dtype=float);trust=.25
    try:
        row=evaluate(x)
        for iteration in range(40):
            if row["admission"]["static_constraints_passed"]:
                termination="nonlinear_constraints_passed";break
            c=np.array(list(row["constraints"].values()))/.001;jac=[]
            for j in range(len(x)):
                a=x.copy();b=x.copy();h=min(.005,trust/2);a[j]=max(-1.,x[j]-h);b[j]=min(1.,x[j]+h)
                ca=np.array(list(evaluate(a)["constraints"].values()))/.001
                cb=np.array(list(evaluate(b)["constraints"].values()))/.001
                jac.append((cb-ca)/(b[j]-a[j]))
            jac=np.stack(jac,axis=1)
            bounds=list(zip(np.maximum(-trust,-1-x),np.minimum(trust,1-x)))
            lp=linprog(np.r_[np.zeros(len(x)),1.],
                A_ub=-np.column_stack((jac,np.ones(len(c)))),b_ub=c-1e-4,
                bounds=[*bounds,(0,None)],method="highs",
                options={"time_limit":1.,"maxiter":100})
            item=dict(iteration=iteration,trust=trust,status=int(lp.status),
                nonlinear_before_m=float(c.min()*.001),trials=[])
            numerical.append(item)
            if not lp.success:
                termination="linear_solver_failed";break
            item["linear_slack_m"]=float(lp.x[-1]*.001);item["direction"]=lp.x[:-1].tolist()
            old_score=(max(0.,-row["minimum_margin_m"]),row["violation_score"]);improved=None
            for fraction in (1.,.5,.25,.125):
                proposed=np.clip(x+fraction*lp.x[:-1],-1.,1.);r=evaluate(proposed)
                item["trials"].append(dict(fraction=fraction,minimum_margin_m=r["minimum_margin_m"],
                    violation_score=r["violation_score"]))
                if (max(0.,-r["minimum_margin_m"]),r["violation_score"])<old_score:
                    improved=(proposed,r);break
            if improved is None:
                trust*=.5
                if trust<1e-4:
                    termination="no_nonlinear_progress";break
            else:
                x,row=improved
                trust=min(.25,trust*1.5)
    except BudgetExceeded:termination="budget_exhausted"
    if best is None:raise ValueError("budget exhausted before any evidence")
    return dict(termination=termination,evaluations=count,numerical=numerical,best=best[1],
        elapsed_s=time.monotonic()-started,optimizer_status_is_not_qualification=True)

def load_inputs():
    from scripts.audit_dh116_pinch_initial_pose import checked_json,load_geometry
    from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for,SDF_DIR,SDF_SHA
    from scripts.certify_dh116_tripod_tracking_envelope import actual_nodes
    p=checked_json(PARENT,PARENT_SHA)
    if not p["independent_proof_check"]["complete_disjoint_cover"]:raise ValueError("parent conditional proof required")
    g=load_geometry(ROOT,side="left");grids,bindings=grids_for(g,checked_json(SDF_DIR/"result.json",SDF_SHA))
    nodes,identity=actual_nodes()
    return p["plan"],g,grids,bindings,nodes,identity

def measure(parent,g,grids,nodes,x):
    from scripts.audit_dh116_pinch_initial_pose import pose_matrix
    from scripts.audit_dh116_tripod_fixed_root_path import shapes
    from scripts.prepare_dh116_pinch_closure_margin import sample_contact
    from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import refined_lower
    pose,q=decode(parent,x);mesh=shapes(g,pose,q)
    table={n:float(v[:,2].min()-.3) for n,v in mesh.items()}
    rod={n:float(refined_lower(v,nodes)) for n,v in mesh.items()
        if n.startswith("cooked:") or n.split(":")[-1] not in REQUIRED}
    delta=pose_matrix(pose)@np.linalg.inv(pose_matrix(g["pose"]))
    fk={n.rsplit("/",1)[-1]:delta@t for n,t in g["fk"].poses(q).items()}
    sdf={n:sample_contact(grids[n],fk[n],nodes) for n in REQUIRED}
    gaps={n:r["minimum_gap_m"] for n,r in sdf.items()}
    if any(r["gradient_norm_at_minimum"] is None or r["gradient_norm_at_minimum"]<=0 for r in sdf.values()):
        raise ValueError("all required SDF gradients supported")
    c=physical_constraints(table,rod,gaps,CONFIG)
    return dict(x=np.asarray(x).tolist(),pose_wxyz=pose,q_rad=q,table=table,rod=rod,
        sdf=sdf,constraints=c,actual_contact=False)

def run(output):
    from scripts.audit_dh116_tripod_native_tracking import sha
    from scripts.prepare_dh116_tripod_fixed_orientation import compile_candidate
    config=json.loads((ROOT/"configs/dh116_tasks/flexible_tripod_contact_closure_cpu_v3.json").read_text())
    validate_config(config)
    parent,g,grids,bindings,nodes,identity=load_inputs()
    with (output/"evaluations.jsonl").open("x") as f:
        def query(x):
            r=measure(parent,g,grids,nodes,x)
            f.write(json.dumps(encode_owned(r),allow_nan=False)+"\n");f.flush()
            return r
        result=bounded_search(query,np.zeros(9),
            maximum_evaluations=config["maximum_evaluations"],maximum_seconds=config["maximum_seconds"])
        os.fsync(f.fileno())
    b=result["best"];p=deepcopy(parent);p["end_pose_wxyz"]=b["pose_wxyz"];p["q1_rad"]=b["q_rad"]
    proposal=compile_candidate(p,raise_m=.0015,steps=200)
    # Full nonlinear source/SDF recalculation remains independent of optimizer exit flag.
    check=measure(parent,g,grids,nodes,b["x"])
    if check["constraints"]!=b["constraints"]:raise ValueError("endpoint source recomputation differs")
    return dict(version=VERSION,status="complete",config=config,search=result,
        independent_endpoint_rechecked=True,plan=proposal,parent_sha256=PARENT_SHA,
        sources=g["source_sha256"],sdf_bindings=bindings,node_identity=identity,
        source_node_positions=nodes,readiness=candidate_admission(check["constraints"]),
        source_sha256=sha(__file__),new_physics_steps=0,ppo_updates=0,
        limitation="SDF negative gap is contact prediction, not measured wrench; conditional tracking/path and new native required")

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    try:r=run(a.output)
    except Exception as e:r=dict(status="blocked",reason=type(e).__name__+": "+str(e),
        new_physics_steps=0,ppo_updates=0,teacher_admission=False)
    with (a.output/"result.json").open("x") as f:
        f.write(json.dumps(encode_owned(r),indent=2,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:r.get(k) for k in ("status","reason","readiness")}))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
