#!/usr/bin/env python3
"""Read-only time-aligned contact analysis of the frozen 234-step native run."""
import argparse
from collections import Counter
import json,os,signal,sys,time
from pathlib import Path
import numpy as np
from scripts import replay_dh116_segment_contact_r0 as ledger
from go2_pvcnn.control.dh116.tabletop_deformable_pair_source_basis import ContactTimeline,verify_contact_source
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned

SOURCE=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/source/third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh")
VERSION="dh116-frozen-pinch-contact-timeline-replay-v1"

def run(result,emit):
    receipt=ledger.locked.load_locked()
    if ledger.locked.sha(ledger.REAUDIT)!=ledger.REAUDIT_SHA:
        raise ValueError("frozen independent postexit audit required")
    reaudit=json.loads(ledger.REAUDIT.read_bytes())
    if (reaudit["lifecycle_qualified"] is not True or
            reaudit["original_scheduler_receipt_sha256"]!=ledger.locked.RECEIPT_SHA or
            reaudit["qualification_scope"]!=ledger.locked.SCOPE):
        raise ValueError("independent lifecycle evidence scope differs")
    source=verify_contact_source(SOURCE)
    p0=decode_owned(json.loads(ledger.old.read_pinned(ledger.locked.NATIVE/"director_reset.json",
        receipt["artifact_sha256"]["director_reset.json"]))["payload"])
    result.update(source_contact_header=source,original_receipt_sha256=ledger.locked.RECEIPT_SHA,
        postexit_audit_sha256=ledger.REAUDIT_SHA,
        source_ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],
        historical_native_run_only=True,original_scheduler_status_unchanged=True)
    counts=Counter();observer=None;pending=None;last_boundary=None;directions=[]
    for event in ledger.stream_events(ledger.locked.NATIVE/"physical.jsonl",result["source_ledger_sha256"]):
        op=event.get("operation");counts[str(op)]+=1
        if op is None:
            if event!={"version":"dh116-dual-physical-evidence-v1","physics_budget_end":234,"solver_substeps":12}:
                raise ValueError("original ledger header differs")
            continue
        payload=event["payload"]
        if op in ("setup","initial_geometry"):continue
        if op=="reset":
            if observer is not None:raise ValueError("one actual reset required")
            radius=ledger.old.check_radius_packet(p0,payload["reset"]["rod_reset"])
            observer=ContactTimeline(grasp="pinch",radius_sample=dict(value=radius["value"],
                source="actual_parameter_metadata:"+ledger.old.BACKEND_SHA),max_physics_rows=232)
        elif observer is None:raise ValueError("pre-action reset missing")
        elif op=="begin":
            if pending is not None:raise ValueError("previous control not settled")
            pending=np.asarray(payload["action"])
            if pending.shape!=(1,24) or not np.isfinite(pending).all():raise ValueError("original action24 required")
        elif op=="physical":
            x=payload["exchange"];action=x["action_readback"]
            if pending is None or not np.array_equal(np.asarray(action["request"]["action24"]),pending):
                raise ValueError("actual per-physics action differs from pre-action decision")
            ids=action["post"]["identity"]
            if len(ids)!=1 or ids[0]["policy_step"]!=x["policy_step"] or ids[0]["substep"]!=x["substep"]:
                raise ValueError("actual action clock differs from exchange")
            rows=observer.record(x["rod"],action_identity=ids[0])
            for row in rows:
                emit(row)
                directions.extend(row["comparisons"] or [])
            last_boundary=ids[0]
        elif op=="end":
            if pending is None or last_boundary["substep"]!=7:
                raise ValueError("incomplete control")
            pending=None
        else:raise ValueError("unexpected ledger operation")
    if pending is not None or counts!=Counter({"None":1,"setup":2,"initial_geometry":2,"reset":1,"begin":29,"physical":232,"end":29}):
        raise ValueError("incomplete source sequence")
    if last_boundary["physics_step"]!=234:raise ValueError("wrong source end tick")
    result.update(observer.summary())
    def distribution(values):
        a=np.asarray([v for v in values if v is not None],float)
        return None if len(a)==0 else dict(count=len(a),min=float(a.min()),max=float(a.max()),
            p50=float(np.quantile(a,.5)),p95=float(np.quantile(a,.95)))
    result["direction_statistics"]={}
    for side in ("left","right"):
        rows=[c for c in directions if c["side"]==side]
        result["direction_statistics"][side]=dict(
            comparisons=len(rows),same_node=sum(c["pair_ids"][0][1]==c["pair_ids"][1][1] for c in rows),
            sign_true=sum(c["geometric_opposition_sign_candidate"] is True for c in rows),
            sign_false=sum(c["geometric_opposition_sign_candidate"] is False for c in rows),
            sign_unknown=sum(c["geometric_opposition_sign_candidate"] is None for c in rows),
            normal_dot=distribution([c["normal_dot"] for c in rows]),
            force_cosine=distribution([c["force_cosine"] for c in rows]),
            min_inward_normal=distribution([None if c["inward_normal_cosines"] is None else min(c["inward_normal_cosines"]) for c in rows]),
            min_inward_force_n=distribution([None if c["inward_force_projections_n"] is None else min(c["inward_force_projections_n"]) for c in rows]),
            material_station_gap_m=distribution([c["material_station_gap_m"] for c in rows]),
            same_node_center_residual_m=distribution([c["same_node_reconstructed_center_residual_m"] for c in rows]))
    result.update(version=VERSION,timeline_version=observer.summary()["version"],status="complete",source_operation_counts=dict(counts),
        native_imports=[n for n in sys.modules if n.split(".")[0] in ("omni","isaacsim","isaaclab","ropesim","warp")])
    if result["native_imports"]:raise ValueError("native runtime imported in read-only replay")

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial",new_physics_steps=0,ppo_updates=0,teacher_admission=False,
                native_qualified=False,budget=dict(max_physics_rows=232,wall_s=120))
    previous=signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError("CPU120s exhausted")))
    signal.setitimer(signal.ITIMER_REAL,120)
    try:
        with (args.output/"timeline.jsonl").open("x") as f:
            def emit(row):f.write(json.dumps(encode_owned(row),allow_nan=False)+"\n");f.flush()
            try:run(result,emit)
            finally:f.flush();os.fsync(f.fileno())
    except Exception as exc:result.update(status="blocked",reason=type(exc).__name__+": "+str(exc))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    result.update(elapsed_s=time.monotonic()-start,script_sha256=ledger.locked.sha(__file__))
    with (args.output/"result.json").open("x") as f:
        json.dump(encode_owned(result),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status","reason","physics_rows","solver_substep_rows","comparison_count","elapsed_s")}))
    return int(result["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
