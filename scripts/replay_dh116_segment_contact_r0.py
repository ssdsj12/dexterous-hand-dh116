#!/usr/bin/env python3
"""Stream already completed native data through explicit segment/R0 readers.

This is disk-only diagnostic replay. Original failed scheduler receipt, 234
physical steps and short left contact are not changed into new execution or
Teacher qualification. No unavailable contact/stability/slip source is filled.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
from scripts import audit_dh116_pinch_segment_postexit_v2 as locked
from scripts import replay_dh116_native_parameter_r0 as old
from go2_pvcnn.control.dh116.tabletop_segment_r0_approach import SegmentMeasuredPinchApproach
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import PhysicalR0Diagnostic,encode_owned,decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task,R0_TERMS

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-segment-contact-measured-r0-replay-v1"
REAUDIT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-pinch-segment-contact-20261113-41heHL/postexit-v2-audit.json")
REAUDIT_SHA="57cbc1b6b97eb594368e4702746241c3be3a3b0c562d99293ab2b470b30706cf"

def stream_events(path,expected_sha):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or locked.sha(path)!=expected_sha:
        raise ValueError("owned SHA-pinned complete native ledger required")
    with path.open("rb") as f:
        for index,line in enumerate(f,1):
            if not line.endswith(b"\n") or not line.strip():
                raise ValueError("incomplete/empty JSONL row "+str(index))
            row=json.loads(line)
            if not isinstance(row,dict):raise ValueError("object JSONL row required")
            yield decode_owned(row)
    if path.is_symlink() or locked.sha(path)!=expected_sha:
        raise ValueError("native source changed during replay")

def prefix_scope(controls,rows,setup):
    if any(type(x) is not int for x in (controls,rows,setup)) or (controls,rows,setup)!=(29,232,2):
        raise ValueError("exact existing 29x8 measured +2 setup prefix required")
    return dict(total_recorded_sdk_steps=234,recorded_physics_rows=232,recorded_controls=29,
        actual_record_duration_s=234/480,full_episode_evaluated=False,
        new_physics_steps=0,ppo_updates=0,training_reward=None,native_qualification=False,teacher_admission=False)

def run(result,emit):
    receipt=locked.load_locked()
    if locked.sha(REAUDIT)!=REAUDIT_SHA:raise ValueError("exact independent postexit audit required")
    reaudit=json.loads(REAUDIT.read_bytes())
    if (reaudit["lifecycle_qualified"] is not True or reaudit["original_scheduler_qualification"] is not False
            or reaudit["original_scheduler_receipt_sha256"]!=locked.RECEIPT_SHA
            or reaudit["qualification_scope"]!=locked.SCOPE):
        raise ValueError("postexit audit scope differs")
    result.update(original_receipt_sha256=locked.RECEIPT_SHA,reaudit_sha256=REAUDIT_SHA,
        ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],source_scope=locked.SCOPE,
        original_scheduler_qualification=False,source_native_lifecycle_reaudited=True)
    reader=SegmentMeasuredPinchApproach(ROOT)
    p0=decode_owned(json.loads(old.read_pinned(locked.NATIVE/"director_reset.json",
        receipt["artifact_sha256"]["director_reset.json"]))["payload"])
    task=load_flexible_grasp_task(ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",ROOT)
    budget=json.loads((ROOT/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_bytes())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"]=2;budget["physics"]["solver_substeps_per_physics_step"]=12
    weights={name:float(i+1) for i,name in enumerate(R0_TERMS)}
    result.update(coefficient_scope="existing_CPU_fixture_only_not_training_coefficients",
        original_frozen_task_coefficients_unchanged=True)
    observer=None;counts=Counter();blocked=Counter();controls=[];approach=[]
    valid_fields=None;contacts_unknown=[];first=None;last=None
    for row in stream_events(locked.NATIVE/"physical.jsonl",result["ledger_sha256"]):
        op=row.get("operation");counts[str(op)]+=1
        if op is None:
            if row!={"version":"dh116-dual-physical-evidence-v1","physics_budget_end":234,"solver_substeps":12}:
                raise ValueError("exact measured ledger header required")
            continue
        payload=row["payload"]
        if op in ("setup","initial_geometry"):continue
        if op=="reset":
            if observer is not None:raise ValueError("duplicate actual reset")
            radius=old.check_radius_packet(p0,payload["reset"]["rod_reset"])
            reset_approach=reader.capture_reset(payload["reset"],p0)
            observer=PhysicalR0Diagnostic(task,"pinch",env_id=0,budget_config=budget,
                diagnostic_coefficients=weights,backend_artifact=old.BACKEND,backend_sha256=old.BACKEND_SHA,
                radius_sample=dict(value=radius["value"],source="native_host_metadata:RopeBatch.parameter_snapshot.contact_radius_m:"+old.BACKEND_SHA))
            adapted=deepcopy(task);adapted["backend"]["extension_sha256"]=old.BACKEND_SHA
            observer._reducer=old._ResetBoundApplicableR0(adapted,"pinch",env_id=0,budget_config=budget,
                diagnostic_coefficients=weights,reset_approach=reset_approach["reset_approach"])
            observer.record(payload)
            result["reset_approach"]=reset_approach
            emit(dict(operation="reset_approach",payload=reset_approach))
        elif observer is None:raise ValueError("actual reset missing before action")
        elif op=="begin":
            observer.begin(**payload)
            mask=np.asarray(payload["pre_action_state"]["common"]["valid"])
            indices=np.flatnonzero(mask).tolist()
            if valid_fields is not None and indices!=valid_fields:raise ValueError("common measured field set changed")
            valid_fields=indices
        elif op=="physical":
            measured=reader.capture(payload["exchange"])
            observer.record(payload,auxiliary=measured["auxiliary"])
            record=observer._rows[-1];r0=record["r0"]
            blocked.update(r0["blocked_terms"])
            if any(s in record["unavailable_sources"] for s in ("left","right")):
                contacts_unknown.append(dict(physics_step=r0["physics_step"],
                    missing={s:record["unavailable_sources"][s] for s in ("left","right") if s in record["unavailable_sources"]}))
            a=measured["auxiliary"]["samples"]["approach_distance_m"]["value"];approach.append(a)
            first=r0["physics_step"] if first is None else first;last=r0["physics_step"]
            emit(dict(operation="measured_r0",identity=measured["auxiliary"]["identity"],
                approach_distance_m=a,approach_source_sha256=measured["source_sha256"],
                terms=r0["terms"],blocked_terms=r0["blocked_terms"],
                unavailable_sources=record["unavailable_sources"],success_candidate=r0["success_candidate"],
                training_reward=r0["training_reward"]))
        elif op=="end":
            out=observer.end(payload);control=out["control_receipt"]
            if control is None or control["physical_samples"]!=8:raise ValueError("exact settled eight-step control required")
            controls.append(control);emit(dict(operation="control_receipt",payload=control))
        else:raise ValueError("unsupported native operation "+str(op))
    if counts!=Counter({"None":1,"setup":2,"initial_geometry":2,"reset":1,"begin":29,"physical":232,"end":29}):
        raise ValueError("recorded operation counts differ")
    if (first,last)!=(3,234):raise ValueError("actual prefix clocks differ")
    if observer._rows is not None:raise ValueError("unsettled tail cannot complete")
    result.update(prefix_scope(len(controls),len(approach),counts["setup"]))
    result.update(status="complete",controls=controls,source_operation_counts=dict(counts),
        first_physics_step=first,last_physics_step=last,common_valid_indices=valid_fields,
        common_valid_count=len(valid_fields),common_total=371,common_missing_policy="NaN_with_false_validity",
        approach_first_m=approach[0],approach_last_m=approach[-1],approach_valid_samples=len(approach),
        blocked_physical_term_counts=dict(blocked),unqualified_contact_rows=contacts_unknown,
        settled_control_count=len(controls),finite_diagnostic_reward_controls=sum(c["diagnostic_reward"] is not None for c in controls),
        original_R0_reset_native_order_qualified=False,
        missing_sources=["qualified_stability_predicate","positive_effective_opposed_contact",
            "fixed_material_anchor_and_tangential_slip","hand_table_contact_source",
            "training_reward_coefficients","native_R0_reset_owner_order"])
    result["native_imports"]=[n for n in sys.modules if n.split(".")[0] in ("omni","isaacsim","isaaclab","ropesim","warp")]
    if result["native_imports"]:raise ValueError("CPU replay imported native runtime")

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial",new_physics_steps=0,ppo_updates=0,
        native_qualification=False,teacher_admission=False,training_reward=None,wall_budget_s=600)
    old_handler=signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError("600s CPU replay exhausted")))
    signal.setitimer(signal.ITIMER_REAL,600)
    try:
        with (a.output/"progress.jsonl").open("x") as f:
            def emit(row):f.write(json.dumps(encode_owned(row),allow_nan=False)+"\n");f.flush()
            try:run(result,emit)
            finally:f.flush();os.fsync(f.fileno())
    except Exception as exc:
        result.update(status="blocked",reason=type(exc).__name__+": "+str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
    result.update(elapsed_s=time.monotonic()-start,source_sha256=locked.sha(__file__))
    with (a.output/"result.json").open("x") as f:
        json.dump(encode_owned(result),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status","reason","elapsed_s","recorded_physics_rows",
        "settled_control_count","approach_valid_samples","blocked_physical_term_counts","teacher_admission")}))
    return int(result["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
