#!/usr/bin/env python3
"""Audit original pad membership of pinned real bilateral contact evidence."""
import argparse
from collections import Counter
import hashlib,json,os
from pathlib import Path
from scripts.replay_dh116_bilateral_contact_r0 import load_source
from scripts.replay_dh116_segment_contact_r0 import stream_events
from scripts.audit_dh116_bilateral_native_evidence import ACTUAL_RECEIPT_SHA,require_episode
from scripts.audit_dh116_pinch_initial_pose import load_geometry
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_pair_substep_load_audit import audit_capture,bind_clock,SIDES,PADS
from go2_pvcnn.control.dh116.tabletop_deformable_pad_region import classify_pair,bind_geometry,VERSION
from go2_pvcnn.control.dh116.tabletop_effective_pad import _patch
from go2_pvcnn.control.dh116 import tabletop_effective_pad,tabletop_effective_pad_runtime,tabletop_deformable_physical_exchange

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def validate_binding(binding,g,name):
    source=g["source_colliders"][name]["source"];cooked=g["cooked_pads"][name]
    if (binding["source_geometry_sha256"]!=source["geometry_sha256"]
        or binding["source_usd_sha256"]!=g["usd"]["sha256"]
        or binding["capture_sha256"]!=cooked["capture_sha256"]
        or binding["sdf_sha256"]!=sha(binding["sdf_path"])):
        raise ValueError("immutable own-side source/cooked capture/SDF identity mismatch")
    return dict(source_geometry_sha256=source["geometry_sha256"],
        source_usd_sha256=g["usd"]["sha256"],capture_sha256=cooked["capture_sha256"],
        sdf_sha256=binding["sdf_sha256"],binding_sha256=canonical(binding))

def run(output):
    native,receipt=load_source();gs={s:load_geometry(ROOT,side=s) for s in SIDES}
    atlas={s:{n:_patch(gs[s]["source_colliders"][n]["vertices"],s,n) for n in PADS} for s in SIDES}
    statistics={s+"."+n:Counter() for s in SIDES for n in PADS}
    first={};binding_proofs={};physical=0;accepted=0;rejected=0;identity=None;solver_rows=0
    with (output/"pairs.jsonl").open("x") as f,(output/"substep_regions.jsonl").open("x") as sf:
        for e in stream_events(native/"physical.jsonl",receipt["artifact_sha256"]["physical.jsonl"]):
            p=decode_owned(e.get("payload",{}))
            if e.get("operation")=="reset":identity=p["reset"]["after_clock"];continue
            if e.get("operation")!="physical":continue
            if identity is None:raise ValueError("actual reset identity required")
            x=p["exchange"];rod=x["rod"];cap=rod["rod_mesh_contact_pairs"]
            clock=x["action_readback"]["post"]["identity"][0];require_episode(identity,clock)
            bind_clock(cap,rod,clock,expected_physics_step=physical+3)
            audit_capture(cap,solver_substeps=12,complete_capture=True)
            raw=x["closure_geometry"]["raw"];poses={}
            for s in SIDES:
                for n in PADS:
                    key=s+"."+n;b=rod["mesh_bindings"][key]
                    if key not in binding_proofs:binding_proofs[key]=validate_binding(b,gs[s],n)
                    elif canonical(b)!=binding_proofs[key]["binding_sha256"]:raise ValueError("binding changed during episode")
                    poses[key]=bind_geometry(raw,clock,key,b)
            kept=[];counts=[{s:dict(raw=0,region_accepted=0) for s in SIDES} for _ in range(12)]
            for pair in cap["pairs"]:
                key=pair["mesh_name"]
                if key=="table":kept.append(pair);continue
                s,n=key.split(".");result=classify_pair(pair,poses[key],atlas[s][n])
                a=result["accepted_by_original_patch"];c=statistics[key]
                c["total"]+=1;c["accepted"]+=a;c["rejected"]+=not a
                if not a:c["first:"+result["first_rejection"]]+=1
                for reason in result["overlap_rejections"]:c["overlap:"+reason]+=1
                accepted+=a;rejected+=not a
                counts[pair["substep"]][s]["raw"]+=1
                counts[pair["substep"]][s]["region_accepted"]+=a
                if a:kept.append(pair)
                record=dict(identity=clock,solver_substep=pair["substep"],node_index=pair["node_index"],
                    mesh_name=key,pair=pair,region=result,source_receipt_sha256=ACTUAL_RECEIPT_SHA,
                    atlas_sha256=canonical(atlas[s][n]),geometry_binding_sha256=binding_proofs[key]["binding_sha256"])
                f.write(json.dumps(record,allow_nan=False)+"\n")
                if key not in first:first[key]=record
            filtered=dict(cap,pairs=kept)
            region_loads=audit_capture(filtered,solver_substeps=12,complete_capture=True)
            for k,row in enumerate(region_loads["substeps"]):
                sf.write(json.dumps(dict(identity=clock,solver_substep=k,pair_counts=counts[k],
                    region_loads=row,
                    known_no_effective_contact={s:counts[k][s]["region_accepted"]==0 for s in SIDES},
                    positive_effective_contact_qualified=False,teacher_admission=False),allow_nan=False)+"\n")
                solver_rows+=1
            physical+=1
        for file in (f,sf):file.flush();os.fsync(file.fileno())
    if physical!=232 or solver_rows!=2784:raise ValueError("complete original bilateral prefix required")
    return dict(version=VERSION,status="complete",source_receipt_sha256=ACTUAL_RECEIPT_SHA,
        source_ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],physical_rows=physical,solver_rows=solver_rows,
        accepted_hand_pairs=accepted,rejected_hand_pairs=rejected,
        by_pad={k:dict(v) for k,v in statistics.items()},first_pair_by_pad=first,
        bindings=binding_proofs,atlas=atlas,
        source_sha256={m.__name__:sha(m.__file__) for m in
            (tabletop_effective_pad,tabletop_effective_pad_runtime,tabletop_deformable_physical_exchange)},
        output_sha256={n:sha(output/n) for n in ("pairs.jsonl","substep_regions.jsonl")},
        timing_scope="actual post-SDK pose held across12rod substeps; no new SDK step before closure capture",
        contact_depth_qualified=False,material_slip_qualified=False,positive_effective_contact_qualified=False,
        native_qualification=False,teacher_admission=False,new_physics_steps=0,ppo_updates=0,
        limitations=["Read-only replay of actual consumed bilateral234 evidence; not a new native run.",
            "Original inner ellipse, supporting planes and ON_HAND normal test are unchanged.",
            "Region acceptance is a necessary gate only; no positive contact qualification is inferred.",
            "Zero accepted pairs proves this gate fails; it does not prove why all prior Student policies failed."])

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    try:r=run(a.output)
    except Exception as e:r=dict(status="blocked",reason=type(e).__name__+": "+str(e),new_physics_steps=0,ppo_updates=0)
    r["script_sha256"]=sha(__file__)
    with (a.output/"result.json").open("x") as f:
        f.write(json.dumps(r,indent=2,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:r.get(k) for k in ("status","reason","physical_rows","solver_rows","accepted_hand_pairs","rejected_hand_pairs")}))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
