#!/usr/bin/env python3
"""Read-only original bilateral234 receipt; audit all2784 solver load windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scripts.replay_dh116_bilateral_contact_r0 import load_source
from scripts.replay_dh116_segment_contact_r0 import stream_events
from scripts.audit_dh116_bilateral_native_evidence import ACTUAL_RECEIPT_SHA,require_episode
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_pair_substep_load_audit import audit_capture,bind_clock,SIDES,PADS,VERSION
from go2_pvcnn.control.dh116 import tabletop_surface_force_sharing

def run(output):
    native,receipt=load_source();all_rows=[];identity=None;physical=0
    with (output/"substep_loads.jsonl").open("x") as f:
        for e in stream_events(native/"physical.jsonl",receipt["artifact_sha256"]["physical.jsonl"]):
            p=decode_owned(e.get("payload",{}))
            if e.get("operation")=="reset":identity=p["reset"]["after_clock"];continue
            if e.get("operation")!="physical":continue
            if identity is None:raise ValueError("real reset identity required before physical rows")
            x=p["exchange"];rod=x["rod"];cap=rod["rod_mesh_contact_pairs"]
            clock=x["action_readback"]["post"]["identity"][0]
            require_episode(identity,clock)
            bind_clock(cap,rod,clock,expected_physics_step=physical+3)
            r=audit_capture(cap,solver_substeps=12,complete_capture=True)
            for row in r["substeps"]:
                record=dict(hands=row["hands"],table_loaded=row["table_loaded"],solver_substep=row["substep"],
                    identity=clock,solver_tick=physical*12+row["substep"]+1,
                    nominal_solver_timestamp_s=(physical*12+row["substep"]+1)/5760,
                    nominal_physics_timestamp_s=clock["timestamp_s"]-(11-row["substep"])/5760,
                    timestamp_scope="derived_from_native_tick_and_declared_dt_not_individual_clock_readback",
                    source_receipt_sha256=ACTUAL_RECEIPT_SHA,raw_effective_contact_qualification=False)
                f.write(json.dumps(record,allow_nan=False)+"\n");all_rows.append(record)
            physical+=1
        f.flush();os.fsync(f.fileno())
    if physical!=232 or len(all_rows)!=2784:raise ValueError("complete original bilateral234 source required")
    summary={}
    for side in SIDES:
        hands=[r["hands"][side] for r in all_rows];a=np.array([h["normal_load_n"] for h in hands])
        pair=np.array([h["raw_surface_paired"] for h in hands])
        needed=(a[:,:2]>.05).all(axis=1)
        peaks=[]
        for j,pad in enumerate(PADS):
            i=int(a[:,j].argmax());r=all_rows[i]
            peaks.append(dict(pad=pad,value_n=float(a[i,j]),identity=r["identity"],
                solver_tick=r["solver_tick"],solver_substep=r["solver_substep"],
                nominal_solver_timestamp_s=r["nominal_solver_timestamp_s"],
                nominal_physics_timestamp_s=r["nominal_physics_timestamp_s"]))
        summary[side]=dict(normal_load_mean_n=a.mean(0).tolist(),normal_load_p95_n=np.percentile(a,95,axis=0).tolist(),
            normal_load_max_n=a.max(0).tolist(),load_peak_identities=peaks,
            raw_surface_paired_substeps=int(pair.sum()),pinch_required_loaded_substeps=int(needed.sum()),
            pinch_loaded_and_raw_paired_substeps=int((needed&pair).sum()),
            positive_attributed_upward_substeps=sum(h["attributed_force_z_n"]>0 for h in hands))
    both=[all(r["hands"][s]["raw_surface_paired"] and all(r["hands"][s]["above_original_load_threshold"][:2])
              for s in SIDES) for r in all_rows]
    longest=current=0
    for b in both:
        current=current+1 if b else 0;longest=max(longest,current)
    return dict(version=VERSION,status="complete",source_receipt_sha256=ACTUAL_RECEIPT_SHA,
        source_ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],
        original_surface_source_sha256=hashlib.sha256(Path(tabletop_surface_force_sharing.__file__).read_bytes()).hexdigest(),
        physical_rows=physical,solver_rows=len(all_rows),sides=summary,
        bilateral_loaded_raw_paired_substeps=sum(both),longest_bilateral_raw_run_substeps=longest,
        longest_bilateral_raw_run_s=longest/5760,table_loaded_substeps=sum(r["table_loaded"] for r in all_rows),
        positive_effective_contact_qualified=False,full_r0_reward_ready=False,
        teacher_admission=False,native_qualification=False,new_physics_steps=0,ppo_updates=0,
        limitations=["raw SDF contacts have no qualified fingertip surface mask or fixed material slip",
            "normal loads and normals are same-substep source measurements; original criterion only diagnoses raw contact",
            "no load or direction statistics are assembled across different solver substeps",
            "maximum per-digit values may be different times; peak identities retained independently"])

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    try:r=run(a.output)
    except Exception as e:r=dict(status="blocked",reason=type(e).__name__+": "+str(e),new_physics_steps=0,ppo_updates=0)
    r["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (a.output/"result.json").open("x") as f:
        f.write(json.dumps(r,indent=2,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:r.get(k) for k in ("status","reason","solver_rows","bilateral_loaded_raw_paired_substeps","teacher_admission")}))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
