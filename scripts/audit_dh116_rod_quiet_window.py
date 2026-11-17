#!/usr/bin/env python3
"""Read-only measured quiet-window metrics, never a cross-grasp Reset certificate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

VERSION="dh116-actual-rod-quiet-window-v1"
PHYSICAL=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-bilateral-owned-20261114-xizgjb7a/native-runs/flex.f8-bilateral-owned-pinch-native/attempt-w0i1tg3o/native/physical.jsonl")
PHYSICAL_SHA="d53b446425640fbbbf59ba952c52ec84ff952f0eca8b539131f9197bc899158d"
IDENTITY=("run_uuid","env_id","episode_uid","reset_id","rod_uid")

def finite(value,shape):
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError("complete finite measured array required")
    return a

def summarize_window(records,n):
    if type(n) is not int or n<8 or n>48 or n%8 or len(records)<n:
        raise ValueError("whole8-substep quiet windows, at most48 samples")
    rows=records[:n];first=rows[0];identity={k:first[k] for k in IDENTITY}
    positions=[];velocities=[];joint={s:{} for s in ("left","right")};table=hand=0
    for index,row in enumerate(rows):
        if any(row[k]!=value for k,value in identity.items()):raise ValueError("quiet identity changed")
        if (row["physics_step"]!=first["physics_step"]+index
                or row["rod_tick"]!=first["rod_tick"]+12*index
                or row["policy_step"]!=index//8 or row["substep"]!=index%8
                or not math.isclose(row["timestamp_s"],first["timestamp_s"]+index/480,abs_tol=2e-7,rel_tol=0)):
            raise ValueError("quiet clock gap, order or policy ownership changed")
        if np.any(finite(row["action24"],(24,))) or row["guard_failure"] is not None:
            raise ValueError("nonzero action or termination is not a quiet interval")
        positions.append(finite(row["positions"],(41,3)));velocities.append(finite(row["velocities"],(41,3)))
        for field in ("table_loaded_substeps","hand_loaded_substeps"):
            s=row[field]
            if len(s)!=len(set(s)) or any(type(k) is not int or not 0<=k<12 for k in s):
                raise ValueError("unique measured solver substeps required")
        table+=len(row["table_loaded_substeps"]);hand+=len(row["hand_loaded_substeps"])
        for side in ("left","right"):
            h=row["hands"][side];names=h["joint_names"]
            if len(names)!=len(set(names)) or not names:raise ValueError("distinct joint identity required")
            q=finite(h["q"],(len(names),));qd=finite(h["qd"],(len(names),))
            if index and set(joint[side])!=set(names):raise ValueError("joint identity changed")
            for name,value,velocity in zip(names,q,qd):
                unit=h["units"][name]
                if unit not in ("rad","m"):raise ValueError("explicit joint SI units required")
                entry=joint[side].setdefault(name,dict(unit=unit+"/s",values=[],positions=[]))
                if entry["unit"]!=unit+"/s":raise ValueError("joint unit changed")
                entry["values"].append(float(velocity));entry["positions"].append(float(value))
    pos=np.array(positions);vel=np.array(velocities)
    for side in joint:
        for name,entry in joint[side].items():
            v=np.array(entry.pop("values"));q=np.array(entry.pop("positions"))
            entry.update(max_abs=float(abs(v).max()),last=float(v[-1]),
                         position_peak_to_peak=float(np.ptp(q)),position_unit=entry["unit"][:-2])
    return dict(version=VERSION,identity=identity,samples=n,physics_step_range=[rows[0]["physics_step"],rows[-1]["physics_step"]],
        timestamp_range_s=[rows[0]["timestamp_s"],rows[-1]["timestamp_s"]],
        measured_interval_s=n/480,sample_span_s=(n-1)/480,
        max_node_speed_m_s=float(np.linalg.norm(vel,axis=-1).max()),
        last_max_node_speed_m_s=float(np.linalg.norm(vel[-1],axis=-1).max()),
        max_node_displacement_from_first_sample_m=float(np.linalg.norm(pos-pos[0],axis=-1).max()),
        node_center_z_range_m=[float(pos[...,2].min()),float(pos[...,2].max())],
        hand_joint_speed=joint,table_loaded_solver_substeps=table,hand_loaded_solver_substeps=hand,
        expected_solver_substeps=n*12,stable_grasp=None,native_reset_qualified=False,
        teacher_admitted=False,cross_grasp_admission=False,source_scope="actual_prior_Pinch_only")

def extract(payload):
    x=payload["exchange"];rod=x["rod"];post=x["action_readback"]["post"]
    clock={k:rod[k] for k in (*IDENTITY,"physics_step","rod_tick","timestamp_s")}
    if rod["clock_valid"] is not True or x["physics_step"]!=clock["physics_step"]:
        raise ValueError("same current physical and rod clock required")
    if len(post["identity"])!=1 or any(post["identity"][0][k]!=v for k,v in clock.items()):
        raise ValueError("post action/rod identity mismatch")
    if rod["source"]!="RopeBatch.positions/velocities/contact_forces/mesh_wrenches":
        raise ValueError("actual solver source required")
    captured=rod["rod_mesh_contact_pairs"]
    if (captured["physics_step"]!=rod["physics_step"] or captured["rod_tick"]!=rod["rod_tick"]
            or rod["pair_capture_status"]!="validated_solver_samples"):
        raise ValueError("actual complete same-clock contact capture required")
    table=set();hand=set()
    for pair in captured["pairs"]:
        step=pair["substep"]
        if type(step) is not int or not 0<=step<12:raise ValueError("solver substep identity")
        normal=float(pair["normal_force_on_rod_n"])
        if not math.isfinite(normal) or normal<0:raise ValueError("finite nonnegative measured normal force")
        if normal>0:
            if pair["mesh_name"]=="table":table.add(step)
            elif pair["mesh_name"].startswith(("left.","right.")):hand.add(step)
            else:raise ValueError("unknown contact partner")
    hands={}
    for side in ("left","right"):
        h=post["hands"][side]
        if h["measured_source"]!="physx_backend_get_dof_positions/get_dof_velocities":
            raise ValueError("measured joint source required")
        names=h["joint_names"];size=len(names)
        hands[side]=dict(joint_names=names,q=finite(h["q"],(1,size))[0],
            qd=finite(h["qd"],(1,size))[0],units=h["units_by_name"])
    return dict(**clock,policy_step=x["policy_step"],substep=x["substep"],
        action24=finite(x["action_readback"]["request"]["action24"],(1,24))[0],
        positions=finite(rod["node_positions"],(1,41,3))[0],velocities=finite(rod["velocities"],(1,41,3))[0],
        hands=hands,table_loaded_substeps=sorted(table),hand_loaded_substeps=sorted(hand),guard_failure=x["guard_failure"])

def load_actual(path=PHYSICAL):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):digest.update(block)
    if digest.hexdigest()!=PHYSICAL_SHA:raise ValueError("immutable native ledger SHA mismatch")
    records=[];reset=None;physical_count=0
    with path.open() as f:
        for line in f:
            if not line.endswith("\n"):raise ValueError("incomplete native JSONL line")
            r=json.loads(line)
            if r.get("operation")=="reset":
                if reset is not None:raise ValueError("unexpected second reset")
                reset=decode_owned(r["payload"])
            if r.get("operation")=="physical":
                physical_count+=1
                if len(records)<48:records.append(extract(decode_owned(r["payload"])))
    if physical_count!=232 or reset is None:raise ValueError("full original232-sample native ledger required")
    return records,reset,physical_count

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path);a=p.parse_args(argv)
    a.output.mkdir(parents=True,exist_ok=False)
    rows,reset,count=load_actual()
    result=dict(version=VERSION,status="complete",source_path=str(PHYSICAL),source_sha256=PHYSICAL_SHA,
        original_physical_samples=count,new_sdk_steps=0,ppo_updates=0,
        snapshots=[summarize_window(rows,n) for n in (8,16,24,32,48)],
        native_reset_qualified=False,teacher_admitted=False,tripod_qualified=False,power_qualified=False,
        no_threshold_or_stage_change=True,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps([dict(samples=s["samples"],max_node_speed_m_s=s["max_node_speed_m_s"],
        node_displacement_m=s["max_node_displacement_from_first_sample_m"],
        table_substeps=s["table_loaded_solver_substeps"],hand_substeps=s["hand_loaded_solver_substeps"])
        for s in result["snapshots"]]))
    return 0

if __name__=="__main__":raise SystemExit(main())
