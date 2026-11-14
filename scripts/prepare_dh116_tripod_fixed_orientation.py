#!/usr/bin/env python3
"""CPU-only fixed-wrist-orientation Tripod candidate with measured-lag screens.

Does not tune gains, change native physics or grant native execution readiness.
Sampled lag corners are a diagnostic, not a continuous reachable-set proof.
"""
import argparse
from itertools import product
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scripts.audit_dh116_tripod_sdk_path import compile_actions, runtime_reference, target_path_proof, budget_audit
from scripts.audit_dh116_tripod_native_tracking import RECEIPT,RECEIPT_SHA,LOCKS,sha
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE

VERSION="dh116-tripod-fixed-orientation-candidate-v1"
ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/dh116_tasks/flexible_tripod_fixed_orientation_cpu_v1.json"

def compile_candidate(parent,*,raise_m,steps):
    if not math.isfinite(raise_m) or not 0<raise_m<=.003:raise ValueError("bounded positive candidate raise")
    end=np.array(parent["end_pose_wxyz"],dtype=float);start=end.copy();start[2]+=raise_m
    return compile_actions(start,end,parent["q0_rad"],parent["q1_rad"],
        side="left",physics_steps=steps,mode="intrinsic_linear")

def delayed_state(plan,step,finger_delay,translation_delay):
    d=np.asarray(finger_delay,dtype=float)
    if (type(step) is not int or not 0<=step<=plan["physics_steps"]+24 or d.shape!=(6,)
        or not np.isfinite(d).all() or (d<0).any() or not math.isfinite(translation_delay)
        or translation_delay<0):raise ValueError("finite bounded nonnegative lag and fixed observation horizon")
    # Diagnostic pure delays only, not changed control commands.
    q0=np.array(plan["q0_rad"]);q1=np.array(plan["q1_rad"])
    f=np.clip((step-d)/plan["physics_steps"],0,1)
    pf=float(np.clip((step-translation_delay)/plan["physics_steps"],0,1))
    p0=np.array(plan["seed_pose_wxyz"]);p1=np.array(plan["end_pose_wxyz"])
    if not np.array_equal(p0[3:],p1[3:]):raise ValueError("fixed orientation required")
    return np.r_[p0[:3]+pf*(p1[:3]-p0[:3]),p0[3:]],q0+f*(q1-q0)

def readiness(*,nominal,sampled):
    return dict(nominal_reference_proven=bool(nominal),sampled_lag_screen_passed=bool(sampled),
        continuous_tracking_envelope_proven=False,native_ready=False,teacher_admission=False,
        blocked_by="tracking envelope and real contact qualification remain pending")

def recorded_lags():
    from scripts.replay_dh116_segment_contact_r0 import stream_events
    if sha(RECEIPT)!=RECEIPT_SHA:raise ValueError("consumed native receipt changed")
    r=json.loads(RECEIPT.read_bytes());native=Path(r["output"])
    values={n:[] for n in (*ACTIVE,"wrist_x","wrist_y","wrist_z","wrist_roll","wrist_pitch","wrist_yaw")}
    previous=None
    for e in stream_events(native/"physical.jsonl",LOCKS["physical.jsonl"]):
        if e.get("operation")!="physical":continue
        h=e["payload"]["exchange"]["action_readback"]["post"]["hands"]["left"]
        names=h["joint_names"];target=np.asarray(h["backend_targets"])[0];q=np.asarray(h["q"])[0]
        if previous is not None:
            for n in values:
                i=names.index(n);delta=target[i]-previous[i]
                if abs(delta)>1e-7:values[n].append(float((target[i]-q[i])/delta))
        previous=target
    result={}
    for name,v in values.items():
        if not v:result[name]=None;continue
        if not np.isfinite(v).all() or min(v)<-1:raise ValueError("observed tracking cannot be represented as nonnegative delay")
        result[name]=dict(count=len(v),minimum_ticks=min(v),maximum_ticks=max(v),
            screening_upper_ticks=max(0,int(math.ceil(max(v))))+1,
            measured_native_model_validation=False)
    return result

def lag_scenarios(plan,lags):
    moving=np.flatnonzero(np.asarray(plan["q1_rad"])!=np.asarray(plan["q0_rad"]))
    upper=np.zeros(6)
    for j in moving:
        if lags[ACTIVE[j]] is None:raise ValueError("missing moving-finger lag source")
        upper[j]=lags[ACTIVE[j]]["screening_upper_ticks"]
    translation=max(lags[n]["screening_upper_ticks"] for n in ("wrist_x","wrist_y","wrist_z"))
    result=[]
    for corner in product((0,1),repeat=len(moving)+1):
        d=np.zeros(6)
        d[moving]=upper[moving]*np.array(corner[:-1])
        result.append(dict(finger_ticks=d.tolist(),translation_ticks=translation*corner[-1]))
    return result

def screen(g,plan,scenarios,emit,*,max_seconds):
    from scripts.audit_dh116_tripod_fixed_root_path import shapes
    from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import refined_lower
    nodes=np.array(g["reference"]["reference_layout"]["wire_points"]);nodes[:,2]=.3049999177455902
    required={"finger14_link","finger23_link","finger33_link"}
    points=sorted(set(range(0,plan["physics_steps"]+1,8))|{plan["physics_steps"]+n for n in (8,16,24)})
    minima={};rejected=[];count=0;start=time.monotonic()
    for step in points:
        for i,s in enumerate(scenarios):
            if time.monotonic()-start>max_seconds:raise TimeoutError("bounded CPU lag screen exhausted")
            pose,q=delayed_state(plan,step,s["finger_ticks"],s["translation_ticks"])
            current=shapes(g,pose,q);metrics={}
            for name,v in current.items():
                metrics["table:"+name]=float(v[:,2].min()-.3)-.0021
                if name.startswith("cooked:") or name.split(":")[1] not in required:
                    threshold=-.00005 if name.split(":")[1] in required else 0.
                    metrics["rod:"+name]=float(refined_lower(v,nodes))-threshold
            margin,key=min((v,k) for k,v in metrics.items())
            record=dict(step=step,scenario=i,lag=s,minimum_margin_m=margin,limiting_constraint=key)
            emit(record);count+=1
            for k,v in metrics.items():
                if k not in minima or v<minima[k]["margin_m"]:
                    minima[k]=dict(margin_m=v,step=step,scenario=i)
            if margin<0:rejected.append(record)
    return dict(sampled_states=count,distinct_scenarios=len(scenarios),sampled_steps=points,
        all_sampled_constraints_passed=not rejected,failures=rejected,minima=minima,
        continuous_proof=False,contact_observed=False,
        source="same original source17/cooked5 and unchanged rod radius/2.1mm table guard",
        node_height_scope="read-only actual previous-run settled node Z; new-run quiet required")

def pack_proof(proof):
    return {**proof,"samples":[dict(fraction=float(t),measurements=v)
                            for t,v in sorted(proof["samples"].items())]}

def validate_config(config):
    required=dict(version=VERSION,mode="offline_candidate_only",motion_steps=200,
        native_execution_enabled=False,maximum_new_physics_steps=0,physics_hz=480,
        control_hz=60,table_clearance_m=.0021,required_pad_minimum_reference_gap_m=-.00005,
        continuous_tracking_proof_required=True,teacher_admission=False)
    if any(config.get(k)!=v or type(config.get(k)) is not type(v) for k,v in required.items()):
        raise ValueError("unchanged CPU scope, original guard and clock contract required")

def run(output,max_seconds):
    from scripts.prepare_dh116_tripod_owned_approach import load_pair
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    config=json.loads((ROOT/CONFIG).read_text())
    validate_config(config)
    old=load_pair()["original_24d_plan"]
    p=compile_candidate(old,raise_m=config["initial_world_z_raise_m"],steps=config["motion_steps"])
    lags=recorded_lags();scenarios=lag_scenarios(p,lags)
    g=load_geometry(ROOT,side="left")
    replay=runtime_reference(p,g)
    nominal=target_path_proof(p,replay,g)
    with (output/"lag_screen.jsonl").open("x") as f:
        def emit(row):f.write(json.dumps(row,allow_nan=False)+"\n")
        sampled=screen(g,p,scenarios,emit,max_seconds=max_seconds)
        f.flush();os.fsync(f.fileno())
    return dict(version=VERSION,status="complete",config_sha256=sha(ROOT/CONFIG),
        original_attempt_receipt_sha256=RECEIPT_SHA,original_ledger_sha256=LOCKS["physical.jsonl"],
        plan=p,geometry_sources=g["source_sha256"],recorded_lags=lags,scenarios=scenarios,
        nominal_target_path=pack_proof(nominal),sampled_lag_screen=sampled,
        readiness=readiness(nominal=nominal["proved"],sampled=sampled["all_sampled_constraints_passed"]),
        prospective_budget=budget_audit(setup=2,quiet=8,motion=200,hold=24,maximum=240),
        reset_change="left root final orientation and +1.5mm start height; original q0; proposal only",
        actual_native_steps=0,ppo_updates=0,teacher_admission=False,
        limitations=["sampled pure delays do not bound every native tracking transient",
          "same terminal source inference is not new experimental calibration",
          "original recipe unchanged and consumed; no native execution entry is emitted",
          "positive SDF/geometry gaps do not establish contact","right hand not changed or requalified"])

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-seconds",type=float,default=240);a=p.parse_args()
    if not 1<=a.max_seconds<=600:p.error("CPU screen budget1..600s")
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:r=run(a.output,a.max_seconds)
    except Exception as e:r=dict(version=VERSION,status="blocked",reason=type(e).__name__+": "+str(e),
        actual_native_steps=0,ppo_updates=0,teacher_admission=False)
    r.update(elapsed_s=time.monotonic()-start,script_sha256=sha(__file__))
    serialized=json.dumps(encode_owned(r),indent=2,allow_nan=False)+"\n"
    with (a.output/"result.json").open("x") as f:
        f.write(serialized);f.flush();os.fsync(f.fileno())
    print(json.dumps({k:r.get(k) for k in ("status","reason","elapsed_s","readiness")}))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
