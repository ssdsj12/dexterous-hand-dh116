#!/usr/bin/env python3
"""Reuse original continuous-path audit for a frozen empirical endpoint."""
import argparse
from importlib.util import spec_from_file_location,module_from_spec
import json
import os
from pathlib import Path
import time
import numpy as np
from scripts import audit_dh116_effective_pinch_target_path as path
from scripts import solve_dh116_pinch_tracking_targets as targets
from go2_pvcnn.control.dh116.tabletop_bilateral_owned_geometry import separation_lower
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned,decode_owned

VERSION="dh116-pinch-tracking-target-path-cpu-v1"
SOURCE=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-tracking-target-20261114/search-03/result.json")
SOURCE_SHA="fceb4da1085c658142c56f82d311f2488de18242158d7eb63bc69996aef847b2"

PROFILE=dict(version="dh116-pinch-same-amplitude-plateau-cpu-v1",profile="plateau",
    bow_amplitude_m=.0001,maximum_x_rate_m_s=.00114,segments=20,physics_hz=480,control_hz=60,
    prior_failure_sha256="e0879d44e640a99807f4675cf75442665646ab99220d41f74863516cabe6164e",
    native_execution=False,teacher_admission=False)
def plateau_bow(fraction,amplitude_m):
    if not np.isfinite([fraction,amplitude_m]).all() or not 0<=fraction<=1 or not 0<=amplitude_m<=.0001:
        raise ValueError("bounded original100um bow")
    t=np.linspace(0,1,21)
    y=np.minimum.reduce([np.full(21,amplitude_m),
        19*amplitude_m*t/5,19*amplitude_m*(1-t)/5,1.15*4*amplitude_m*t*(1-t)])
    return float(np.interp(fraction,t,y))

def isolated_auditor(file,sha,bow_function):
    """Separate globals: original plan's module is never mutated."""
    spec=spec_from_file_location("_dh116_tracking_path_audit",path.__file__)
    auditor=module_from_spec(spec)
    spec.loader.exec_module(auditor)
    auditor.PARENT=Path(file);auditor.PARENT_SHA=sha;auditor.bow=bow_function
    return auditor

def whole_hand(g,poses):
    return np.concatenate([targets.target.surface.base.transform(shape["vertices"],poses[n])
        for key in ("source_colliders","cooked_pads") for n,shape in g[key].items()])

def interhand_lower(sample_gap,travel,samples):
    if type(samples) is not int or samples<2 or not np.isfinite([sample_gap,travel]).all() or travel<0:
        raise ValueError("finite whole-hand travel and at least two samples")
    return float(sample_gap-travel/(2*(samples-1)))

def static_interhand(engine,result,audit,pair,bow_function):
    base=targets.target.surface.base
    end=result["all_observed_validation"]["endpoint"]
    g=base.load_geometry(targets.ROOT,side="left")
    left=whole_hand(g,base.fk_poses(g,pair["reset_seed"]["reset_root_pose_w"][0],
        pair["left"]["candidate"]["active_q_rad"]))
    count=path.CONFIG["path_samples"];right=audit["sides"]["right"]
    if right.get("path_samples")!=count:raise ValueError("complete original path certificate required")
    travel=max(right["travel_bounds_m"].values());minimum=float("inf")
    for fraction in np.linspace(0,1,count):
        pose,_=targets.target.original_path(end,pair["right"]["candidate"],float(fraction))
        pose["root_pose_wxyz"][0]+=bow_function(float(fraction),.0001)
        cloud=whole_hand(engine.g,base.fk_poses(engine.g,pose["root_pose_wxyz"],pose["q_rad"]))
        minimum=min(minimum,interhand_lower(separation_lower(left,cloud),travel,count))
    return dict(samples=count,continuous_lower_m=minimum,passed=minimum>=0.,
        method="same whole-hand source/cooked axis planes minus maximum displacement bound",
        native_ready=False)


def load_candidate(*,expected_sha=SOURCE_SHA):
    if expected_sha!=SOURCE_SHA or targets.digest(SOURCE)!=expected_sha:
        raise ValueError("exact candidate CPU evidence SHA required")
    r=decode_owned(json.loads(SOURCE.read_bytes()))
    if (r["config"]!=targets.REFINED_CONFIG or not r["qualification"]["empirical_endpoint_passed"]
        or r["native_ready"] or r["new_physics_steps"] or r["ppo_updates"]):
        raise ValueError("CPU-only complete candidate required")
    source=targets.load_observed_scenarios();g=targets.Geometry(source,physical_gate=True)
    checked=g.evaluate(np.array(r["search"]["best"]["x"]),source["scenarios"])
    if checked!=r["all_observed_validation"]:raise ValueError("full endpoint recomputation differs")
    return r,g

def parent_record(result,geometry):
    v=result["all_observed_validation"]
    if set(v["scenarios"])!={str(k) for k in range(211,235)}:
        raise ValueError("complete observed endpoint scenarios required")
    verdict=targets.admission(v["nominal"],list(v["scenarios"].values()))
    if not verdict["empirical_endpoint_passed"]:raise ValueError("empirical endpoint failed")
    end=dict(v["endpoint"],admission=dict(static_constraints_passed=True))
    return dict(version=VERSION,native_ready=False,teacher_admission=False,sides={
        "left":{"search":{"best":{"admission":{"static_constraints_passed":False}}}},
        "right":{"source_geometry_sha256":geometry.g["source_sha256"],
            "sdf_bindings":geometry.bindings,"search":{"best":end}}})

def interpret(result):
    return dict(static_path_passed=result["sides"]["right"].get("static_path_passed") is True,
        tracking_qualified=False,native_ready=False,teacher_admission=False)

def path_guards(audit):
    values=[*audit["table_m"].values(),*audit["non_target_rod_gap_m"].values(),
        *[p["minimum_gap_m"] for p in audit["pads"].values()]]
    if not values or not np.isfinite(values).all():raise ValueError("finite complete geometry bounds")
    return (min(audit["table_m"].values())>=.0021
        and min(audit["non_target_rod_gap_m"].values())>=0.
        and min(p["minimum_gap_m"] for p in audit["pads"].values())>=-.0003)

def observed_path(output,result,engine,original,bow_function):
    receipt=json.loads(targets.RECEIPT.read_bytes());ledger=Path(receipt["output"])/"physical.jsonl"
    if targets.digest(ledger)!=receipt["artifact_sha256"]["physical.jsonl"]:raise ValueError("path ledger SHA mismatch")
    end=result["all_observed_validation"]["endpoint"]
    rows=[];worst={"table_m":float("inf"),"other_m":float("inf"),"depth_m":float("inf"),"interhand_m":float("inf")}
    base=targets.target.surface.base
    left_geometry=base.load_geometry(targets.ROOT,side="left")
    with (output/"observed_path.jsonl").open("x") as f:
        for line in ledger.open():
            if not line.endswith("\n"):raise ValueError("partial native source")
            item=json.loads(line)
            if item.get("operation")!="physical":continue
            x=decode_owned(item["payload"])["exchange"];raw=x["closure_geometry"]["raw"];ident=raw["identity"]
            old=targets.plan.expected_boundary(original,x["physics_step"]-2)
            commanded,_=targets.target.original_path(end,original["right"]["candidate"],old["fraction"])
            commanded["root_pose_wxyz"][0]+=bow_function(old["fraction"],.0001)
            h=raw["hands"]["right"];v=h["link_poses_xyzw"][0,h["body_names"].index("base_link")]
            actual=np.r_[v[:3],v[6],v[3:6]]
            q=targets.tracking.active_values(raw["joint_readback"]["hands"]["right"])
            scenario=dict(identity=ident,joint_identity=raw["joint_readback"]["identity"][0],rod_identity=ident,
                root_translation_error_m=(actual[:3]-old["root_poses_w"][1,:3]).tolist(),
                world_rotation_error_xyzw=(targets.rotation(actual)*targets.rotation(old["root_poses_w"][1]).inv()).as_quat().tolist(),
                active_error_rad=(q-old["right_q_rad"]).tolist())
            p,q=targets.apply_scenario(commanded,scenario)
            a=targets.target.surface.geometry_record(engine.g,engine.grids,
                targets.target.surface.base.pose_matrix(p),q,np.asarray(raw["rod_nodes_m"]))
            if x["physics_step"]!=3+len(rows) or ident["physics_step"]!=x["physics_step"]:
                raise ValueError("continuous same-clock physical source required")
            left=raw["hands"]["left"]
            left_poses={n:base.pose_matrix(np.r_[v[:3],v[6],v[3:6]])
                for n,v in zip(left["body_names"],left["link_poses_xyzw"][0])}
            gap=separation_lower(whole_hand(left_geometry,left_poses),
                whole_hand(engine.g,base.fk_poses(engine.g,p,q)))
            worst["interhand_m"]=min(worst["interhand_m"],gap)
            row=dict(identity=ident,proposed_root_wxyz=p.tolist(),proposed_active_q_rad=q.tolist(),
                geometry=a,interhand_lower_m=gap,
                conditional_guard_passed=bool(path_guards(a) and gap>=0),cpu_counterfactual_only=True)
            for key,values in (("table_m",a["table_m"].values()),("other_m",a["non_target_rod_gap_m"].values()),
                ("depth_m",[z["minimum_gap_m"] for z in a["pads"].values()])):
                worst[key]=min(worst[key],min(values))
            rows.append(row);f.write(json.dumps(encode_owned(row),allow_nan=False)+"\n")
        f.flush();os.fsync(f.fileno())
    if len(rows)!=232:raise ValueError("complete 232-step observed path required")
    return dict(samples=len(rows),passed=all(r["conditional_guard_passed"] for r in rows),
        worst=worst,first_failure=next((r["identity"] for r in rows if not r["conditional_guard_passed"]),None),
        observed_source_sha256=targets.digest(ledger),no_future_dynamics_guarantee=True,
        interhand_observed_scenarios_only=True,native_ready=False)

def run(output,profile=None):
    if profile not in (None,PROFILE):raise ValueError("exact independent plateau profile required")
    result,g=load_candidate();parent=parent_record(result,g);file=output/"parent.json"
    with file.open("x") as f:json.dump(encode_owned(parent),f,indent=2,allow_nan=False);f.write("\n")
    if profile is not None:
        prior=SOURCE.parent.parent/"path-01"/"result.json"
        if targets.digest(prior)!=profile["prior_failure_sha256"]:raise ValueError("prior path failure SHA mismatch")
    original=targets.plan.load_pair(targets.ROOT)
    auditor=isolated_auditor(file,targets.digest(file),plateau_bow if profile else path.bow)
    audited=auditor.run(output,auditor.CONFIG)
    empirical=observed_path(output,result,g,original,auditor.bow)
    interhand=static_interhand(g,result,audited,original,auditor.bow)
    return dict(version=VERSION,status="complete",source_sha256=SOURCE_SHA,
        path_module_sha256=targets.digest(path.__file__),path_config=path.CONFIG,profile=profile,
        continuous_interhand=interhand,
        continuous_static_path=audited,observed_path=empirical,qualification=interpret(audited),
        new_physics_steps=0,ppo_updates=0,native_ready=False,teacher_admission=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--profile",type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:r=run(a.output,None if a.profile is None else json.loads(a.profile.read_bytes()))
    except Exception as e:r=dict(version=VERSION,status="blocked",reason=type(e).__name__+": "+str(e),
        new_physics_steps=0,ppo_updates=0,native_ready=False,teacher_admission=False)
    r.update(elapsed_s=time.monotonic()-start,script_sha256=targets.digest(__file__))
    with (a.output/"result.json").open("x") as f:json.dump(encode_owned(r),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:r.get(k) for k in ("status","reason","elapsed_s","qualification","observed_path")}))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
