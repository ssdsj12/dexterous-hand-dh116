#!/usr/bin/env python3
"""Finite CPU right-hand direction from actual SDK234; never native execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np

from scripts import audit_dh116_pinch_segment_postexit_v2 as evidence
from scripts.audit_dh116_pinch_initial_pose import (
    load_geometry,pose_matrix,transform,path_displacement_bound,digest,checked_json)
from scripts.prepare_dh116_pinch_closure_margin import (
    fk_poses,relative_pose,sample_contact,sample_grid)
from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for,SDF_DIR,SDF_SHA
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,expand,active_limits
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned
from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import rod_support_lower
from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import certify,validate

VERSION="dh116-right-pinch-measured-direction-v1"
ROOT=Path(__file__).resolve().parents[1]
AUDIT=evidence.NATIVE.parents[3]/"postexit-v2-audit.json"
AUDIT_SHA="57cbc1b6b97eb594368e4702746241c3be3a3b0c562d99293ab2b470b30706cf"
PADS=("finger14_link","finger23_link")
TARGET=-.00015
SAMPLES=161
MAX_ENDPOINTS=12
WALL_SECONDS=90

def load_anchor(*,side="right"):
    if side!="right":raise ValueError("independent right side required")
    receipt=evidence.load_locked()
    if digest(AUDIT)!=AUDIT_SHA:raise ValueError("independent native reaudit bytes differ")
    a=json.loads(AUDIT.read_bytes())
    if (a.get("lifecycle_qualified") is not True or a.get("qualification_scope")!=evidence.SCOPE
        or a.get("original_scheduler_receipt_sha256")!=evidence.RECEIPT_SHA
        or a.get("physics_steps")!=234 or a.get("original_scheduler_qualification") is not False):
        raise ValueError("exact independently reaudited234-step execution required")
    last=None;count=0
    with (evidence.NATIVE/"physical.jsonl").open() as f:
        for line in f:
            r=json.loads(line)
            if r.get("operation")=="physical":
                last=decode_owned(r["payload"])["exchange"];count+=1
    if count!=232:raise ValueError("complete232 rows required")
    raw=last["closure_geometry"]["raw"];clock=raw["identity"]
    if clock["physics_step"]!=234 or clock["env_id"]!=0 or clock["rod_tick"]!=2784:
        raise ValueError("actual terminal boundary identity differs")
    joints=raw["joint_readback"]["hands"][side];body=raw["hands"][side]
    if (raw["joint_readback"]["identity"]!=[clock] or len(body["body_names"])!=23
        or len(joints["joint_names"])!=17 or len(set(joints["joint_names"]))!=17):
        raise ValueError("complete actual same-clock right joint/body arrays required")
    names=joints["joint_names"];q=np.array([joints["q"][0,names.index(n)] for n in ACTIVE],float)
    expand(q)
    poses=np.asarray(body["link_poses_xyzw"])[0]
    actual={n:pose_matrix(np.r_[v[:3],v[6],v[3:6]]) for n,v in zip(body["body_names"],poses)}
    if any("/rightHand/" not in p for p in body["link_paths"][0]):
        raise ValueError("right side physical link paths required")
    # Last policy action is held zero. Full actual action source is retained.
    from scripts.prepare_dh116_pinch_segment_contact import action_schedule
    issued=action_schedule()[clock["policy_step"]].numpy().copy()
    return dict(side=side,identity=clock,q=q,qd=joints["qd"].copy(),joint_names=list(names),
        actual=actual,nodes=np.asarray(raw["rod_nodes_m"]).copy(),raw=raw,
        issued_action24=issued,issued_action_source="SHA-verified frozen schedule independently checked against ledger",
        original_receipt_sha256=evidence.RECEIPT_SHA,independent_reaudit_sha256=AUDIT_SHA,
        ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],native_steps=0)

def direction(measure,maximum):
    maximum=np.asarray(maximum,dtype=float)
    if maximum.shape!=(2,) or not np.isfinite(maximum).all() or np.any(maximum<=0) or np.any(maximum>.1):
        raise ValueError("original160/480s at0.3rad/s budget required")
    def f(x):
        v=np.asarray(measure(x),dtype=float)
        if v.shape!=(2,) or not np.isfinite(v).all():raise ValueError("two finite actual supported SDF predictions required")
        return v
    g=f(np.zeros(2));jac=[]
    for h in (.001,.0005):
        cols=[]
        for j in range(2):
            d=np.zeros(2);d[j]=h
            cols.append((f(d)-f(-d))/(2*h))
        jac.append(np.column_stack(cols))
    error=float(np.linalg.norm(jac[0]-jac[1])/max(np.linalg.norm(jac[1]),1e-15))
    if error>.05 or np.linalg.cond(jac[1])>1e6:raise ValueError("nonlocal or singular measured direction")
    d=np.linalg.solve(jac[1],np.full(2,TARGET)-g)
    if np.any(d<0) or np.any(d>maximum):raise ValueError("direction exceeds original bounded finger closure")
    return dict(initial_gap_m=g.tolist(),jacobians=jac,delta_rad=d.tolist(),
        jacobian_relative_error=error,target_sdf_gap_m=TARGET,maximum_delta_rad=maximum.tolist())

def refine_direction(measure,current,maximum):
    current=np.asarray(current,dtype=float);maximum=np.asarray(maximum,dtype=float)
    if (current.shape!=(2,) or maximum.shape!=(2,) or not np.isfinite(current).all()
            or np.any(current<.001) or np.any(maximum-current<=.001)):
        raise ValueError("refinement needs original bounded finite-difference neighborhood")
    r=direction(lambda d:measure(current+d),maximum-current)
    r.update(parent_delta_rad=current.tolist(),delta_rad=(current+np.asarray(r["delta_rad"])).tolist())
    return r

def gates(table,other,minimum_depth,worst_end,normal_dot):
    values=[table,other,minimum_depth,worst_end,normal_dot]
    if not np.isfinite(values).all():raise ValueError("finite complete geometry gates required")
    flags=dict(table_clear=bool(table>=.0021),non_target_clear=bool(other>=0),
        required_depth_safe=bool(minimum_depth>=-.0003),both_end_margin=bool(worst_end<=-.0001),
        geometric_opposed=bool(normal_dot<-.2))
    return dict(**flags,static_path_passed=all(flags.values()),native_ready=False,
        native_contact_observed=False,teacher_admission=False)

def predicted_poses(actual,old,new,required):
    if not set(required)<=set(actual)&set(old)&set(new):
        raise ValueError("all required source/cooked links must have actual and FK transforms")
    return {name:relative_pose(actual[name],old[name],new[name]) for name in required}

def failed_path_choice(rows):
    if len(rows)!=3:raise ValueError("exact three saved nonlinear candidates required")
    def error(row):
        gaps=np.array([row["pads"][n]["minimum_gap_m"] for n in PADS],float)
        delta=np.array(row["delta_rad"],float)
        if (not np.isfinite(gaps).all() or delta.shape!=(2,) or not np.isfinite(delta).all()
                or np.any(delta<0) or np.any(delta>.1)):raise ValueError("saved candidate invalid")
        return float(np.linalg.norm(gaps-TARGET))
    chosen=min(rows,key=error)
    gaps=[chosen["pads"][n]["minimum_gap_m"] for n in PADS]
    return dict(delta_rad=list(chosen["delta_rad"]),residual_l2_m=error(chosen),
        contact_margin_passed=all(-.0003<=v<=-.0001 for v in gaps))

def run(result,emit,refine_from=None,path_from=None):
    a=load_anchor();g=load_geometry(ROOT,side="right")
    summary=checked_json(SDF_DIR/"result.json",SDF_SHA);grids,binding=grids_for(g,summary)
    old=fk_poses(g,g["pose"],a["q"]);q=a["q"]
    result.update(source={k:a[k] for k in ("identity","original_receipt_sha256","independent_reaudit_sha256","ledger_sha256")},
        geometry_sources=g["source_sha256"],sdf_bindings=binding,initial_q_rad=q.tolist(),
        source_body_transforms=a["actual"],source_nodes_m=a["nodes"],endpoint_evaluations=[],
        actual_baseline_action24=a["issued_action24"],actual_baseline_qd=a["qd"],raw_joint_names=a["joint_names"])
    def poses(d):
        n=q.copy();n[1:3]+=d;expand(n)
        fk=fk_poses(g,g["pose"],n)
        return predicted_poses(a["actual"],old,fk,set(g["source_colliders"])|set(g["cooked_pads"]))
    def pads(transforms):
        rows={n:sample_contact(grids[n],transforms[n],a["nodes"]) for n in PADS}
        normals=[]
        for name,row in rows.items():
            if row["minimum_gap_m"] is None or row["gradient_norm_at_minimum"]<=0:
                raise ValueError("missing locked SDF support")
            _,grad,_=sample_grid(grids[name],[row["local_node_at_minimum"]])
            normal=transforms[name][:3,:3]@grad[0];normal/=np.linalg.norm(normal);normals.append(normal)
        return rows,float(np.dot(*normals))
    def measure(d):
        if len(result["endpoint_evaluations"])>=MAX_ENDPOINTS:raise ValueError("frozen12 endpoint budget exhausted")
        t=poses(d);r,dot=pads(t)
        row=dict(delta_rad=d.tolist(),pads=r,predicted_normal_dot=dot,native_contact_observed=False)
        result["endpoint_evaluations"].append(row);emit(dict(phase="endpoint",**row))
        return np.array([r[n]["minimum_gap_m"] for n in PADS])
    low,high=active_limits()
    if np.any(q[1:3]-.001<low[1:3]) or np.any(q[1:3]+.001>high[1:3]):
        raise ValueError("finite difference would cross original active bounds")
    maximum=np.minimum(.1,high[1:3]-q[1:3])
    if path_from is not None:
        if refine_from is not None or digest(path_from)!="ca22f02d999a520b640ed6c261c8456c20f1a7b14348ae8a517d8a946f5d39cb":
            raise ValueError("exact saved refinement failure required, no further optimization")
        prior=decode_owned(json.loads(Path(path_from).read_bytes()))
        if (prior["status"]!="complete_blocked" or prior["reason"]!="no bounded nonlinear SDF margin candidate"
                or prior["source"]!=result["source"] or prior["initial_q_rad"]!=q.tolist()
                or prior["geometry_sources"]!=g["source_sha256"] or prior["sdf_bindings"]!=binding):
            raise ValueError("saved failure/source differs")
        choice=failed_path_choice(prior["endpoint_evaluations"][-3:])
        chosen=np.array(choice["delta_rad"])
        result["saved_failure_path_audit"]=dict(parent=str(path_from),sha256=digest(path_from),**choice,
            new_endpoint_evaluations=0,new_path_sample_budget=161)
    else:
        if refine_from is None:
            solved=direction(measure,maximum)
        else:
            if digest(refine_from)!="de54cdf504215ac568950a938408e58233c05f02408fb48460734fac5b5d06b6":
                raise ValueError("only exact original nonlinear failure may be refined once")
            prior=decode_owned(json.loads(Path(refine_from).read_bytes()))
            if (prior["status"]!="complete_blocked" or prior["reason"]!="no bounded nonlinear SDF margin candidate"
                    or prior["source"]!=result["source"] or prior["initial_q_rad"]!=q.tolist()
                    or prior["geometry_sources"]!=g["source_sha256"] or prior["sdf_bindings"]!=binding
                    or len(prior["endpoint_evaluations"])!=12):
                raise ValueError("same actual right source and fixed first diagnostic required")
            center=np.asarray(prior["direction"]["delta_rad"])
            solved=refine_direction(measure,center,maximum)
            result["refinement_parent"]=dict(path=str(refine_from),sha256=digest(refine_from),
                original_endpoint_evaluations=12,new_endpoint_budget=12,total_maximum_delta_rad=maximum.tolist())
        result["direction"]=solved
        delta=np.asarray(solved["delta_rad"]);chosen=None
        for factor in (1.,.75,.5):
            got=measure(delta*factor)
            if np.all((got>=-.0003)&(got<=-.0001)):
                chosen=delta*factor;break
        if chosen is None:
            result.update(status="complete_blocked",reason="no bounded nonlinear SDF margin candidate");return
    full=np.zeros(6);full[1:3]=chosen
    travel={**{"source:"+n:v for n,v in path_displacement_bound(g,q,full).items()},
            **{"cooked:"+n:v for n,v in path_displacement_bound(g,q,full,cooked=True).items()}}
    result.update(selected_delta_rad=full,velocity_rad_s=full*3,
        action24=np.r_[np.zeros(6),full*10,np.zeros(12)],path_samples=[])
    table=float("inf");other=float("inf");depth=float("inf")
    for i,f in enumerate(np.linspace(0,1,SAMPLES)):
        t=poses(chosen*f);r,dot=pads(t);row=dict(index=i,fraction=float(f),tables={},others={},support_certificates={},pads=r,predicted_normal_dot=dot)
        for kind,key in (("source","source_colliders"),("cooked","cooked_pads")):
            for n,shape in g[key].items():
                name=kind+":"+n;v=transform(shape["vertices"],t[n]);remainder=travel[name]/320
                row["tables"][name]=float(v[:,2].min()-.3-remainder)
                table=min(table,row["tables"][name])
                if n not in PADS:
                    bound=rod_support_lower(v,a["nodes"])
                    if bound<0:
                        c=certify(v,a["nodes"]);validate(c,v,a["nodes"])
                        bound=max(bound,c["lower_m"])
                        row["support_certificates"][name]=c
                    row["others"][name]=float(bound-remainder);other=min(other,row["others"][name])
        depth=min(depth,min(v["minimum_gap_m"] for v in r.values()))
        result["path_samples"].append(row);emit(dict(phase="path",**row))
    end=result["path_samples"][-1]
    result.update(minimum_table_lower_m=table,minimum_non_target_lower_m=other,
        minimum_required_sdf_gap_m=depth,**gates(table,other,depth,
            max(v["minimum_gap_m"] for v in end["pads"].values()),end["predicted_normal_dot"]))
    result.update(status="complete_diagnostic",reason=None if result["static_path_passed"] else "original geometry gate blocked")
    result["limitations"]=["Actual SDK frame anchor, hypothetical FK increments; not measured candidate tracking.",
        "Nodes frozen at SDK234; no prediction of rod motion/contact feedback.",
        "Actual physical joint/mimic limits not changed; no native recipe generated.",
        "Source17/cooked5 only; live cooking/self-collision, Reset, full episode, R0 remain unqualified.",
        "Continuous travel bound is kinematic, not a physical/model uncertainty bound."]

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--refine-from",type=Path);p.add_argument("--path-from",type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial",native_steps=0,ppo_updates=0,native_ready=False,
        teacher_admission=False,budget=dict(endpoint_evaluations=12,path_samples=161,wall_seconds=90))
    old=signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(TimeoutError("bounded CPU budget exhausted")))
    signal.setitimer(signal.ITIMER_REAL,WALL_SECONDS)
    try:
        with (a.output/"progress.jsonl").open("x") as f:
            def emit(row):f.write(json.dumps(encode_owned(row),allow_nan=False)+"\n");f.flush()
            try:run(result,emit,a.refine_from,a.path_from)
            finally:f.flush();os.fsync(f.fileno())
    except (ValueError,TimeoutError,KeyError,np.linalg.LinAlgError) as exc:
        result.update(status="complete_blocked",reason=f"{type(exc).__name__}: {exc}")
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    result["elapsed_s"]=time.monotonic()-start
    result["implementation_sha256"]=digest(__file__)
    with (a.output/"result.json").open("x") as f:
        json.dump(encode_owned(result),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status","reason","elapsed_s","static_path_passed","native_steps")}))
    return 0 if result["status"]=="complete_diagnostic" else 1
if __name__=="__main__":raise SystemExit(main())
