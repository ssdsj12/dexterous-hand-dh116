#!/usr/bin/env python3
"""Own-side left closure path with the published right command held unchanged.

All checks are CPU counterfactuals. Static held-left native discrepancies do
not qualify future closed-loop contact dynamics.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,os,time
import numpy as np
from scripts import solve_dh116_left_pinch_tracking as left
from scripts import audit_dh116_pinch_tracking_target_path as prior
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned,decode_owned

VERSION="dh116-left-pinch-tracking-path-cpu-v1"
SOURCE=left.PRIOR.parent.parent/"search-03/result.json"
SOURCE_SHA="f8acb5e5d1f54e92d037554f9d995eaa8cc0b0a797b8fdb7a08c52e25903c2a7"
CONFIG=dict(version=VERSION,path_samples=1025,source_measurements=232,
    maximum_seconds_per_stage=60.,left_bow_m=0.,original_motion_unchanged=True,
    native_execution=False,teacher_admission=False)

APPROACH_CONFIG={**CONFIG,"version":"dh116-left-pinch-raised-approach-cpu-v2",
    "left_start_raise_m":.0015,"original_motion_unchanged":False,
    "original_finger_motion_unchanged":True,"right_motion_unchanged":True,
    "left_z_rate_m_s":-.0045,"prior_path_failure_sha256":"c9a4d332d4d5309a1391794d905cfc34d1e9c28284a235a469d7869bd198931c"}

def validate_approach(config):
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    if config!=APPROACH_CONFIG:raise ValueError("independent frozen raised approach")
    speed=config["left_start_raise_m"]/(160/480)
    if speed>ACTION_CONTRACT["translation_rate_m_s"]:raise ValueError("original wrist rate cap")
    return dict(left_z_rate_m_s=-speed,original_limits_preserved=True)

def make_auditor(file,sha,height):
    from types import SimpleNamespace
    auditor=prior.isolated_auditor(file,sha,lambda t,a:0.)
    auditor.CONFIG={**auditor.CONFIG,"bow_world_x_m":0.}
    if height:
        # Isolated dependency namespace; never change historical plan globals.
        original=auditor.targets
        packet=deepcopy(original.parent.load_pair(left.ROOT))
        packet["left"]["candidate"]["root_translation_delta_m"]=[0.,0.,-height]
        auditor.targets=SimpleNamespace(**vars(original))
        auditor.targets.parent=SimpleNamespace(load_pair=lambda root:deepcopy(packet))
    return auditor

def load_candidate():
    r=decode_owned(json.loads(left.source.read_locked(SOURCE,SOURCE_SHA)))
    if (r["config"]!=left.RESET_CONFIG or not r["qualification"]["empirical_endpoint_passed"]
        or r["new_physics_steps"] or r["ppo_updates"] or r["native_ready"]):
        raise ValueError("complete frozen own-side CPU candidate required")
    packet=left.load_packet()
    if packet["source_sha256"]!=r["source"]["source_sha256"]:raise ValueError("source identity changed")
    g=left.Geometry(packet,config=left.RESET_CONFIG)
    v=g.evaluate(np.asarray(r["search"]["best"]["x"]),packet["scenarios"])
    if v!=r["all_observed_validation"]:raise ValueError("endpoint replay mismatch")
    return r,g

def parent_record(r,g):
    end={**r["all_observed_validation"]["endpoint"],"admission":{"static_constraints_passed":True}}
    return dict(version=VERSION,native_ready=False,teacher_admission=False,sides={
        "left":dict(source_geometry_sha256=g.g["source_sha256"],sdf_bindings=g.bindings,search={"best":end}),
        "right":{"search":{"best":{"admission":{"static_constraints_passed":False}}}}})

def commanded_pair(r,pair,fraction,*,left_start_raise_m=0.):
    if not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError("bounded path fraction")
    endpoint=r["all_observed_validation"]["endpoint"]
    a,_=left.target.original_path(endpoint,r["source"]["original_candidate"],float(fraction))
    if left_start_raise_m not in (0.,.0015):raise ValueError("frozen approach height")
    a["root_pose_wxyz"][2]+=left_start_raise_m*(1-fraction)
    c=pair["right"]["candidate"];root=np.array(pair["reset_seed"]["reset_root_pose_w"][1],float)
    root[:3]+=[prior.plateau_bow(fraction,.0001),0.,-.0005*fraction]
    b=dict(root_pose_wxyz=root.tolist(),q_rad=(np.asarray(c["active_q_rad"])+fraction*np.asarray(c["active_delta_rad"])).tolist())
    return {"left":a,"right":b}

def combined_gap(sample_gap,left_travel,right_travel,samples):
    if not np.isfinite([left_travel,right_travel]).all() or min(left_travel,right_travel)<0:
        raise ValueError("both nonnegative physical travel bounds required")
    return prior.interhand_lower(sample_gap,left_travel+right_travel,samples)

def static_interhand(r,engine,pair,audit,*,left_start_raise_m=0.):
    started=time.monotonic();base=left.target.surface.base
    right=base.load_geometry(left.ROOT,side="right")
    right_proof=decode_owned(json.loads(left.plan.PROOF.read_bytes()))["continuous_static_path"]["sides"]["right"]
    la=audit["sides"]["left"]
    if la.get("path_samples")!=CONFIG["path_samples"]:
        return dict(passed=False,samples=0,reason="left continuous path not complete")
    travel_l=max(la["travel_bounds_m"].values());travel_r=max(right_proof["travel_bounds_m"].values())
    worst=float("inf");where=None;n=CONFIG["path_samples"]
    for i,f in enumerate(np.linspace(0,1,n)):
        if time.monotonic()-started>CONFIG["maximum_seconds_per_stage"]:raise TimeoutError("interhand CPU budget")
        poses=commanded_pair(r,pair,float(f),left_start_raise_m=left_start_raise_m)
        clouds=[prior.whole_hand(g,base.fk_poses(g,poses[s]["root_pose_wxyz"],poses[s]["q_rad"]))
            for s,g in (("left",engine.g),("right",right))]
        v=combined_gap(prior.separation_lower(*clouds),travel_l,travel_r,n)
        if v<worst:worst=v;where=i
    return dict(samples=n,passed=worst>=0.,continuous_lower_m=worst,minimum_index=where,
        left_travel_bound_m=travel_l,right_travel_bound_m=travel_r,
        both_hands_move=True,native_ready=False)

def actual_pose(h):
    v=h["link_poses_xyzw"][0,h["body_names"].index("base_link")]
    return np.r_[v[:3],v[6],v[3:6]]

def observed_path(output,r,g,pair,*,left_start_raise_m=0.):
    started=time.monotonic();base=left.target.surface.base
    receipt=json.loads(left.source.read_locked(left.source.RECEIPT,left.source.RECEIPT_SHA))
    raw=left.source.read_locked(Path(receipt["output"])/"physical.jsonl",receipt["artifact_sha256"]["physical.jsonl"])
    if not raw.endswith(b"\n"):raise ValueError("complete source JSONL")
    right=base.load_geometry(left.ROOT,side="right");anchor=r["source"]["executed_anchor"]
    worst=dict(table_m=float("inf"),other_m=float("inf"),depth_m=float("inf"),interhand_m=float("inf"))
    count=0;first_failure=None
    with (output/"observed_path.jsonl").open("x") as f:
        for line in raw.splitlines():
            e=json.loads(line)
            if e.get("operation")!="physical":continue
            if time.monotonic()-started>CONFIG["maximum_seconds_per_stage"]:raise TimeoutError("observed CPU budget")
            x=decode_owned(e["payload"])["exchange"];v=x["closure_geometry"]["raw"];ident=v["identity"]
            tick=x["physics_step"]
            if tick!=3+count or ident["physics_step"]!=tick:raise ValueError("complete actual source clock")
            nominal=left.plan.expected_boundary(pair,tick-2)
            commands=commanded_pair(r,pair,nominal["fraction"],left_start_raise_m=left_start_raise_m)
            actual=actual_pose(v["hands"]["left"]);q=left.observed.tracking.active_values(v["joint_readback"]["hands"]["left"])
            scenario=dict(identity=ident,joint_identity=v["joint_readback"]["identity"][0],rod_identity=ident,
                root_translation_error_m=(actual[:3]-anchor["root_pose_wxyz"][:3]).tolist(),
                world_rotation_error_xyzw=(left.observed.rotation(actual)*left.observed.rotation(anchor["root_pose_wxyz"]).inv()).as_quat().tolist(),
                active_error_rad=(q-anchor["q_rad"]).tolist())
            p,q=left.observed.apply_scenario(commands["left"],scenario)
            a=left.target.surface.geometry_record(g.g,g.grids,base.pose_matrix(p),q,np.asarray(v["rod_nodes_m"]))
            rh=v["hands"]["right"]
            poses={n:base.pose_matrix(np.r_[z[:3],z[6],z[3:6]]) for n,z in zip(rh["body_names"],rh["link_poses_xyzw"][0])}
            gap=prior.separation_lower(prior.whole_hand(g.g,base.fk_poses(g.g,p,q)),prior.whole_hand(right,poses))
            passed=bool(prior.path_guards(a) and gap>=0)
            if not passed and first_failure is None:first_failure=deepcopy(ident)
            for key,values in (("table_m",a["table_m"].values()),("other_m",a["non_target_rod_gap_m"].values()),
                ("depth_m",[z["minimum_gap_m"] for z in a["pads"].values()]),("interhand_m",[gap])):
                worst[key]=min(worst[key],min(values))
            row=dict(identity=ident,proposed_root_wxyz=p.tolist(),proposed_active_q_rad=q.tolist(),geometry=a,
                interhand_lower_m=gap,conditional_guard_passed=passed,cpu_counterfactual_only=True,
                source_left_was_held=True,closure_dynamics_qualified=False)
            f.write(json.dumps(encode_owned(row),allow_nan=False)+"\n");count+=1
        f.flush();os.fsync(f.fileno())
    if count!=232:raise ValueError("complete 232 actual source boundaries")
    return dict(samples=count,passed=first_failure is None,first_failure=first_failure,worst=worst,
        source_sha256=receipt["artifact_sha256"]["physical.jsonl"],no_future_dynamics_guarantee=True)

def qualification(r):
    a=r["continuous_static_path"]["sides"]["left"];b=r["continuous_interhand"];c=r["observed_path"]
    passed=(a.get("static_path_passed") is True and a.get("path_samples")==1025 and
        b.get("passed") is True and b.get("samples")==1025 and c.get("passed") is True and
        c.get("samples")==232 and c.get("first_failure") is None)
    return dict(conditional_path_passed=passed,native_ready=False,teacher_admission=False,unobserved_dynamics_qualified=False)

def run(output,profile=None):
    if profile not in (None,APPROACH_CONFIG):raise ValueError("frozen path profile")
    height=0. if profile is None else profile["left_start_raise_m"]
    if profile is not None:
        validate_approach(profile)
        failure=SOURCE.parent.parent/"path-01/result.json"
        left.source.read_locked(failure,profile["prior_path_failure_sha256"])
    r,g=load_candidate();pair=left.plan.load_pair(left.ROOT);parent=output/"parent.json"
    with parent.open("x") as f:json.dump(encode_owned(parent_record(r,g)),f,indent=2,allow_nan=False);f.write("\n")
    auditor=make_auditor(parent,left.source.sha(parent),height)
    static=auditor.run(output,auditor.CONFIG)
    empirical=observed_path(output,r,g,pair,left_start_raise_m=height)
    interhand=static_interhand(r,g,pair,static,left_start_raise_m=height)
    out=dict(version=VERSION,status="complete",config=CONFIG,profile=profile,source_sha256=SOURCE_SHA,
        continuous_static_path=static,observed_path=empirical,continuous_interhand=interhand,
        original_right_plan_sha256=left.source.sha(left.ROOT/left.plan.CONFIG),
        new_physics_steps=0,ppo_updates=0,native_ready=False,teacher_admission=False)
    out["qualification"]=qualification(out);return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--profile",type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:r=run(a.output,None if a.profile is None else json.loads(a.profile.read_bytes()))
    except Exception as e:r=dict(version=VERSION,status="blocked",reason=type(e).__name__+": "+str(e),native_ready=False,teacher_admission=False,new_physics_steps=0,ppo_updates=0)
    r.update(elapsed_s=time.monotonic()-started,script_sha256=left.source.sha(__file__))
    with (a.output/"result.json").open("x") as f:json.dump(encode_owned(r),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:r.get(k) for k in ("status","reason","elapsed_s","qualification","observed_path","continuous_interhand")}))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
