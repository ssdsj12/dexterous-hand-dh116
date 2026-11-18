#!/usr/bin/env python3
"""Bounded CPU proof of an explicitly conditional Tripod kinematic lag tube.

Uses immutable source17/cooked5, original joint coupling and safety thresholds.
A proof under bounded pure delays is not actuator identification or native
qualification. No native recipe, controller edits or physics execution.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scripts.prepare_dh116_tripod_fixed_orientation import compile_candidate,recorded_lags,lag_scenarios,screen
from scripts.audit_dh116_tripod_native_tracking import sha,RECEIPT,RECEIPT_SHA,LOCKS
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned

VERSION="dh116-tripod-conditional-tracking-envelope-v1"
ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/dh116_tasks/flexible_tripod_tracking_margin_cpu_v1.json"

def shifted_plan(parent,offset):
    offset=np.asarray(offset,dtype=float)
    if (offset.shape!=(3,) or not np.isfinite(offset).all() or offset[1]!=0
        or np.any(np.abs(offset)> .0004)):raise ValueError("bounded x/z-only independent pose proposal")
    p=deepcopy(parent);p["end_pose_wxyz"]=np.array(p["end_pose_wxyz"],dtype=float)
    p["end_pose_wxyz"][:3]+=offset
    return compile_candidate(p,raise_m=.0015,steps=200)

def configuration_at(plan,step,delays,translation_delay):
    ds=np.asarray(delays,dtype=float)
    if (not math.isfinite(step) or not 0<=step<=224 or ds.shape!=(6,)
        or not np.isfinite(ds).all() or (ds<0).any()
        or not math.isfinite(translation_delay) or translation_delay<0):
        raise ValueError("finite bounded continuous time and delays required")
    a=np.array(plan["seed_pose_wxyz"]);b=np.array(plan["end_pose_wxyz"])
    if not np.array_equal(a[3:],b[3:]):raise ValueError("exactly fixed wrist orientation required")
    q0=np.array(plan["q0_rad"]);q1=np.array(plan["q1_rad"])
    q=q0+np.clip((step-ds)/200,0,1)*(q1-q0)
    pos=a[:3]+np.clip((step-translation_delay)/200,0,1)*(b[:3]-a[:3])
    return np.r_[pos,a[3:]],q

def envelope_flags(proved):
    return dict(conditional_kinematic_envelope_proved=bool(proved),actual_tracking_qualified=False,
        native_ready=False,teacher_admission=False,contact_established=False)

def phase_radii(lower,upper):
    a=np.asarray(lower,dtype=float);b=np.asarray(upper,dtype=float);mid=(a+b)/2
    if a.ndim!=1 or len(a)<2 or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or (b<a).any():
        raise ValueError("finite ordered time/delay box")
    center=np.clip(mid[0]-mid[1:],0,200)
    lowest=np.clip(a[0]-b[1:],0,200);highest=np.clip(b[0]-a[1:],0,200)
    return np.maximum(center-lowest,highest-center)

def certify_box(query,bounds,lower,upper,*,max_samples=4096,max_seconds=240,radius_bound=None):
    """Each returned lower bound minus global point-travel radius proves a box.

    Caller supplies named, valid geometric lower bounds and uniform point
    displacement coefficients. The lower-bound algorithm itself need not be
    smooth: the actual distance is 1-Lipschitz under bounded point displacement.
    """
    lo=np.asarray(lower,dtype=float);hi=np.asarray(upper,dtype=float)
    if (lo.ndim!=1 or not len(lo) or hi.shape!=lo.shape or not np.isfinite(lo).all()
        or not np.isfinite(hi).all() or (hi<lo).any() or not bounds
        or type(max_samples) is not int or max_samples<1 or not 0<max_seconds<=600):
        raise ValueError("finite ordered proof domain and budget required")
    bounds={k:np.asarray(v,dtype=float) for k,v in bounds.items()}
    if any(v.shape!=lo.shape or not np.isfinite(v).all() or (v<0).any() for v in bounds.values()):
        raise ValueError("nonnegative global point travel coefficients")
    start=time.monotonic();queue=[(lo,hi)];done=[];samples=[];violations=[];unresolved=[]
    while queue:
        a,b=queue.pop()
        if len(samples)>=max_samples or time.monotonic()-start>=max_seconds:
            unresolved=[dict(lower=x.tolist(),upper=y.tolist()) for x,y in [(a,b),*queue]];break
        x=(a+b)/2;values=query(x)
        if set(values)!=set(bounds) or not all(math.isfinite(v) for v in values.values()):
            raise ValueError("complete finite named geometric margins required")
        samples.append(dict(point=x.tolist(),margin=values))
        bad={k:float(v) for k,v in values.items() if v<0}
        if bad:
            violations.append(dict(point=x.tolist(),margins=bad));break
        radius=(b-a)/2
        global_radius={k:float(bounds[k]@radius) for k in values}
        measured_radius=global_radius if radius_bound is None else radius_bound(a,b)
        if (set(measured_radius)!=set(bounds) or any(not math.isfinite(v) or v<0
                or v>global_radius[k]+1e-12 for k,v in measured_radius.items())):
            raise ValueError("finite conservative tightened radius within global bound required")
        lower_margin={k:float(values[k]-measured_radius[k]) for k in values}
        key=min(lower_margin,key=lower_margin.get)
        if lower_margin[key]>=0:
            done.append(dict(lower=a.tolist(),upper=b.tolist(),minimum_margin_m=lower_margin[key],
                             limiting_constraint=key,sample_index=len(samples)-1))
            continue
        contribution=bounds[key]*radius;j=int(contribution.argmax())
        middle=(a[j]+b[j])/2
        if contribution[j]<=0 or middle==a[j] or middle==b[j]:
            unresolved=[dict(lower=a.tolist(),upper=b.tolist())];break
        lhi=b.copy();lhi[j]=middle;rlo=a.copy();rlo[j]=middle
        queue.extend(((rlo,b),(a,lhi)))
    proved=not violations and not unresolved
    return dict(proved=proved,samples=samples,certified_boxes=done,
        violations=violations,unresolved_boxes=unresolved,
        elapsed_s=time.monotonic()-start,native_ready=False,
        proof_kind="conditional_original_geometry_under_componentwise_pure_delays")

def verify_box_certificate(proof,query,bounds,lower,upper,*,radius_bound=None):
    """Re-query certificate centers and verify disjoint full-domain coverage."""
    a=np.asarray(lower,dtype=float);b=np.asarray(upper,dtype=float)
    if not proof["proved"] or proof["violations"] or proof["unresolved_boxes"]:
        raise ValueError("complete proof required")
    boxes=proof["certified_boxes"]
    if not boxes:raise ValueError("nonempty cover")
    lower_rows=[];upper_rows=[];volume=[];minima=[]
    for box in boxes:
        lo=np.asarray(box["lower"],dtype=float);hi=np.asarray(box["upper"],dtype=float)
        if (lo.shape!=a.shape or hi.shape!=a.shape or not np.isfinite(lo).all()
            or not np.isfinite(hi).all() or (hi<=lo).any() or (lo<a).any() or (hi>b).any()):
            raise ValueError("box outside ordered proof domain")
        x=(lo+hi)/2;sample=proof["samples"][box["sample_index"]]
        if not np.array_equal(x,sample["point"]):raise ValueError("sample not box center")
        values=query(x)
        if set(values)!=set(bounds) or set(sample["margin"])!=set(bounds):
            raise ValueError("proof constraint identity differs")
        if any(not math.isfinite(v) or not math.isclose(v,sample["margin"][k],abs_tol=1e-12,rel_tol=0) for k,v in values.items()):
            raise ValueError("proof sample differs from source recomputation")
        global_radius={k:float(np.asarray(v)@((hi-lo)/2)) for k,v in bounds.items()}
        radii=global_radius if radius_bound is None else radius_bound(lo,hi)
        if set(radii)!=set(bounds) or any(not math.isfinite(v) or v<0 or v>global_radius[k]+1e-12 for k,v in radii.items()):
            raise ValueError("invalid recomputed radius")
        margins={k:values[k]-radii[k] for k in values};minimum=min(margins.values())
        if minimum<0 or not math.isclose(minimum,box["minimum_margin_m"],abs_tol=1e-12,rel_tol=0):
            raise ValueError("false certified margin")
        lower_rows.append(lo);upper_rows.append(hi);volume.append(float(np.prod(hi-lo)));minima.append(float(minimum))
    ls=np.stack(lower_rows);us=np.stack(upper_rows)
    for i in range(len(ls)-1):
        overlap=np.minimum(us[i],us[i+1:])-np.maximum(ls[i],ls[i+1:])
        if np.any(np.all(overlap>0,axis=1)):raise ValueError("overlapping certificate boxes")
    total=math.fsum(volume);expected=float(np.prod(b-a))
    if not math.isclose(total,expected,rel_tol=1e-13,abs_tol=1e-13):
        raise ValueError("hole in certificate domain")
    return dict(independently_requeried_boxes=len(boxes),complete_disjoint_cover=True,
        domain_volume=expected,minimum_margin_m=min(minima),native_qualified=False)

def endpoint_sdf(g,plan,nodes):
    from scripts.solve_dh116_tripod_sdf_constrained_direction import grids_for,SDF_DIR,SDF_SHA
    from scripts.audit_dh116_pinch_initial_pose import checked_json,pose_matrix
    from scripts.prepare_dh116_pinch_closure_margin import sample_contact
    grids,bindings=grids_for(g,checked_json(SDF_DIR/"result.json",SDF_SHA))
    pose=np.array(plan["end_pose_wxyz"]);q=plan["q1_rad"]
    delta=pose_matrix(pose)@np.linalg.inv(pose_matrix(g["pose"]))
    transforms={k.rsplit("/",1)[-1]:delta@v for k,v in g["fk"].poses(q).items()}
    return dict(sdf_binding=bindings,pads={name:sample_contact(grids[name],transforms[name],nodes)
        for name in ("finger14_link","finger23_link","finger33_link")},
        actual_contact=False,teacher_admission=False)

def actual_nodes():
    from scripts.replay_dh116_segment_contact_r0 import stream_events
    if sha(RECEIPT)!=RECEIPT_SHA:raise ValueError("consumed native source changed")
    native=Path(json.loads(RECEIPT.read_bytes())["output"]);last=None
    for e in stream_events(native/"physical.jsonl",LOCKS["physical.jsonl"]):
        if e.get("operation")=="physical":last=e["payload"]["exchange"]
    if last is None or last["physics_step"]!=199:raise ValueError("consumed terminal node snapshot required")
    return np.asarray(last["rod"]["node_positions"])[0].copy(),last["action_readback"]["post"]["identity"][0]

def setup_query(g,plan,lags,nodes,reserve):
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    from scripts.audit_dh116_tripod_fixed_root_path import shapes
    from scripts.audit_dh116_pinch_initial_pose import path_displacement_bound
    from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import refined_lower
    q0=np.array(plan["q0_rad"]);delta=np.abs(np.array(plan["q1_rad"])-q0)/200
    moving=np.flatnonzero(delta)
    upper=np.array([224,*[lags[ACTIVE[i]]["screening_upper_ticks"] for i in moving],
        max(lags[n]["screening_upper_ticks"] for n in ("wrist_x","wrist_y","wrist_z"))],float)
    translation=np.linalg.norm(np.array(plan["end_pose_wxyz"][:3])-plan["seed_pose_wxyz"][:3])/200
    travel={}
    for cooked,prefix in ((False,"source:"),(True,"cooked:")):
        basis=[]
        for i in moving:
            dq=np.zeros(6);dq[i]=delta[i]
            basis.append(path_displacement_bound(g,q0,dq,cooked=cooked))
        for name in basis[0]:
            joint=np.array([b[name] for b in basis])
            travel[prefix+name]=np.r_[joint.sum()+translation,joint,translation]
    required={"finger14_link","finger23_link","finger33_link"}
    bounds={}
    for name,d in travel.items():
        bounds["table:"+name]=d
        if name.startswith("cooked:") or name.split(":")[1] not in required:bounds["rod:"+name]=d
    def query(x):
        delays=np.zeros(6);delays[moving]=x[1:-1]
        pose,q=configuration_at(plan,float(x[0]),delays,float(x[-1]))
        all_shapes=shapes(g,pose,q);result={}
        for name,v in all_shapes.items():
            result["table:"+name]=float(v[:,2].min()-.3)-.0021-reserve
            if "rod:"+name in bounds:
                threshold=-.00005 if name.split(":")[1] in required else 0.
                result["rod:"+name]=float(refined_lower(v,nodes))-threshold-reserve
        return result
    return query,bounds,upper,moving.tolist()

def run(output,maximum):
    from scripts.prepare_dh116_tripod_owned_approach import load_pair
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    config=json.loads((ROOT/CONFIG).read_text())
    if (config["native_execution_enabled"] is not False or config["reserve_m"]!=.00001
        or config["physics_hz"]!=480 or config["control_hz"]!=60):
        raise ValueError("CPU-only unchanged clock and conservative planning reserve")
    plan=shifted_plan(load_pair()["original_24d_plan"],config["world_offset_m"])
    g=load_geometry(ROOT,side="left");lags=recorded_lags();nodes,identity=actual_nodes()
    query,bounds,upper,moving=setup_query(g,plan,lags,nodes,config["reserve_m"])
    proof=certify_box(query,bounds,np.zeros_like(upper),upper,
        max_samples=config["maximum_proof_samples"],max_seconds=maximum,
        radius_bound=lambda a,b:{k:float(v[1:]@phase_radii(a,b)) for k,v in bounds.items()})
    verified=(verify_box_certificate(proof,query,bounds,np.zeros_like(upper),upper,
        radius_bound=lambda a,b:{k:float(v[1:]@phase_radii(a,b)) for k,v in bounds.items()})
        if proof["proved"] else None)
    endpoint=endpoint_sdf(g,plan,nodes)
    # Existing sampled pure-delay screen retained for apples-to-apples comparison.
    with (output/"sampled.jsonl").open("x") as f:
        def emit(row):f.write(json.dumps(row,allow_nan=False)+"\n")
        sampled=screen(g,plan,lag_scenarios(plan,lags),emit,max_seconds=maximum)
        f.flush();os.fsync(f.fileno())
    return dict(version=VERSION,status="complete",plan=plan,config_sha256=sha(ROOT/CONFIG),
        recorded_lags=lags,moving_active_indices=moving,domain_upper=upper,
        original_geometry_sources=g["source_sha256"],source_receipt_sha256=RECEIPT_SHA,
        source_ledger_sha256=LOCKS["physical.jsonl"],node_identity=identity,node_positions=nodes,
        reserve_m=config["reserve_m"],point_travel_bounds=bounds,
        proof=proof,independent_proof_check=verified,endpoint_sdf=endpoint,
        sampled=sampled,readiness=envelope_flags(proof["proved"]),
        new_physics_steps=0,ppo_updates=0,
        limitations=["pure-delay ranges are conditional hypotheses from one consumed trajectory, not future actuator guarantees",
          "original passive coupling is exact in this geometric proof; actual residuals and new reset must be checked",
          "planning reserve10um is stricter geometry, not relaxed physical collision or actuator limits",
          "same rod snapshot is a reference; new-instance node displacement must remain covered",
          "centerline capsule proof is not actual SDF contact/force closure or Teacher qualification",
          "no new native recipe emitted; single-instance safety unchanged"])

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-seconds",type=float,default=240);a=p.parse_args()
    if not 1<=a.max_seconds<=600:p.error("CPU proof/screen budgets1..600s each")
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:r=run(a.output,a.max_seconds)
    except Exception as e:r=dict(version=VERSION,status="blocked",reason=type(e).__name__+": "+str(e),
        new_physics_steps=0,ppo_updates=0,teacher_admission=False)
    r.update(elapsed_s=time.monotonic()-start,script_sha256=sha(__file__))
    encoded=json.dumps(encode_owned(r),indent=2,allow_nan=False)+"\n"
    with (a.output/"result.json").open("x") as f:f.write(encoded);f.flush();os.fsync(f.fileno())
    print(json.dumps({k:r.get(k) for k in ("status","reason","elapsed_s","readiness")}))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
