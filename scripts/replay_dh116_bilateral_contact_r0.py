#!/usr/bin/env python3
"""Read-only bilateral native R0 replay; no physics, new reward or admission.

The immutable passed receipt supplies the actual 234-step, same-episode source.
The existing R0 reducer still owns units, clocks, terminal handling and NA.
Ordinal CPU diagnostic weights are not training coefficients.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
from scripts import audit_dh116_bilateral_native_evidence as pinned
from scripts import replay_dh116_segment_contact_r0 as previous
from scripts import replay_dh116_native_parameter_r0 as old
from go2_pvcnn.control.dh116.tabletop_bilateral_r0_approach import BilateralMeasuredPinchApproach
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import PhysicalR0Diagnostic,encode_owned,decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task,R0_TERMS

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-bilateral-contact-measured-r0-replay-v1"
CONFIG="configs/dh116_training/bilateral_r0_readonly_diagnostic_v1.json"

def require_source_scope(receipt):
    pinned.validate_receipt_scope(receipt)
    if (receipt["command_spec_sha256"]!=pinned.RECIPE_SHA or receipt["code_config_sha256"]!=pinned.CODE_SHA):
        raise ValueError("exact consumed bilateral recipe and source required")

def load_source():
    path=pinned.ACTUAL_RECEIPT
    if path.resolve(strict=True)!=path or pinned.sha(path)!=pinned.ACTUAL_RECEIPT_SHA:
        raise ValueError("pinned actual bilateral receipt changed")
    receipt=json.loads(path.read_bytes());require_source_scope(receipt)
    native=Path(receipt["output"])
    for name,expected in receipt["artifact_sha256"].items():
        target=native/name
        if target.is_symlink() or pinned.sha(target)!=expected:
            raise ValueError("immutable actual native artifact changed: "+name)
    post=path.parent/"bilateral-owned-post-exit-audit.json"
    if pinned.sha(post)!=receipt["lifetime_post_exit_audit_sha256"]:
        raise ValueError("independent postexit audit changed")
    evidence=json.loads(post.read_bytes())["evidence"]
    if evidence["lifecycle_qualified"] is not True or evidence["qualification_scope"]!=pinned.SCOPE:
        raise ValueError("bilateral native lifecycle audit differs")
    return native,receipt

def replay_scope(controls,rows,setup):
    result=previous.prefix_scope(controls,rows,setup)
    result.update(actual_source_native_execution_qualified=True,r0_native_reset_order_qualified=False)
    return result

def summary_gate(controls):
    if any(c["training_reward"] is not None for c in controls):
        raise ValueError("read-only diagnostic cannot produce a training reward")
    blocked=Counter()
    for c in controls:blocked.update(c["blocked_terms"])
    return dict(finite_diagnostic_reward_controls=sum(c["diagnostic_reward"] is not None for c in controls),
        training_reward_controls=0,full_reward_ready=False,blocked_control_term_counts=dict(blocked))

def config():
    value=json.loads((ROOT/CONFIG).read_bytes())
    expected=dict(version=VERSION,scope="readonly_native_evidence_diagnostic",enabled_by_default=False,
        source_receipt_sha256=pinned.ACTUAL_RECEIPT_SHA,geometry_contract="dh116-bilateral-owned-geometry-v1",
        expected_physics_rows=232,setup_physics_steps=2,control_steps=29,
        coefficient_source="existing_CPU_fixture_ordinal_weights_not_training_coefficients",
        training_reward_enabled=False,teacher_admission=False,new_physics_steps=0,ppo_updates=0)
    if value!=expected:raise ValueError("frozen diagnostic-only config differs")
    return value

def run(result,emit):
    cfg=config();native,receipt=load_source()
    result.update(source_receipt_sha256=pinned.ACTUAL_RECEIPT_SHA,
        source_recipe_sha256=receipt["command_spec_sha256"],source_code_config_sha256=receipt["code_config_sha256"],
        source_scope=pinned.SCOPE,ledger_sha256=receipt["artifact_sha256"]["physical.jsonl"],
        config_sha256=pinned.sha(ROOT/CONFIG),coefficient_scope=cfg["coefficient_source"])
    reader=BilateralMeasuredPinchApproach(ROOT)
    p0=decode_owned(json.loads((native/"director_reset.json").read_bytes())["payload"])
    task=load_flexible_grasp_task(ROOT/"configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",ROOT)
    budget=json.loads((ROOT/"configs/dh116_training/deformable_task_budget_examples_v2.json").read_bytes())["plans"]["frozen_eval"]
    budget["setup_physics_steps_per_env"]=2;budget["physics"]["solver_substeps_per_physics_step"]=12
    weights={name:float(i+1) for i,name in enumerate(R0_TERMS)}
    observer=None;counts=Counter();blocked=Counter();controls=[];approach=[];episode=None
    valid_fields=None;unknown_contact=[];first=last=None
    for event in previous.stream_events(native/"physical.jsonl",result["ledger_sha256"]):
        op=event.get("operation");counts[str(op)]+=1
        if op is None:
            if event!={"version":"dh116-dual-physical-evidence-v1","physics_budget_end":234,"solver_substeps":12}:
                raise ValueError("exact consumed measured ledger header required")
            continue
        p=event["payload"]
        if op in ("setup","initial_geometry"):continue
        if op=="reset":
            if observer is not None:raise ValueError("duplicate actual reset")
            episode=p["reset"]["after_clock"]
            radius=old.check_radius_packet(p0,p["reset"]["rod_reset"])
            reset_approach=reader.capture_reset(p["reset"],p0)
            observer=PhysicalR0Diagnostic(task,"pinch",env_id=0,budget_config=budget,
                diagnostic_coefficients=weights,backend_artifact=old.BACKEND,backend_sha256=old.BACKEND_SHA,
                radius_sample=dict(value=radius["value"],source="native_host_metadata:RopeBatch.parameter_snapshot.contact_radius_m:"+old.BACKEND_SHA))
            adapted=deepcopy(task);adapted["backend"]["extension_sha256"]=old.BACKEND_SHA
            observer._reducer=old._ResetBoundApplicableR0(adapted,"pinch",env_id=0,budget_config=budget,
                diagnostic_coefficients=weights,reset_approach=reset_approach["reset_approach"])
            observer.record(p);result["reset_approach"]=reset_approach
            emit(dict(operation="reset_approach",payload=reset_approach))
        elif observer is None:raise ValueError("same-episode reset required before action")
        elif op=="begin":
            pinned.require_episode(episode,p["pre_action_state"]["common"]["identity"])
            observer.begin(**p)
            indices=np.flatnonzero(np.asarray(p["pre_action_state"]["common"]["valid"])).tolist()
            if valid_fields is not None and indices!=valid_fields:raise ValueError("measured common field set changed")
            valid_fields=indices
        elif op=="physical":
            measured=reader.capture(p["exchange"])
            pinned.require_episode(episode,measured["auxiliary"]["identity"])
            observer.record(p,auxiliary=measured["auxiliary"])
            record=observer._rows[-1];r0=record["r0"];blocked.update(r0["blocked_terms"])
            if any(s in record["unavailable_sources"] for s in ("left","right")):
                unknown_contact.append(dict(identity=measured["auxiliary"]["identity"],
                    missing={s:record["unavailable_sources"][s] for s in ("left","right") if s in record["unavailable_sources"]}))
            distance=measured["auxiliary"]["samples"]["approach_distance_m"]["value"]
            approach.append(distance)
            first=r0["physics_step"] if first is None else first;last=r0["physics_step"]
            emit(dict(operation="measured_r0",identity=measured["auxiliary"]["identity"],
                approach_distance_m=distance,approach_source_sha256=measured["source_sha256"],
                terms=r0["terms"],blocked_terms=r0["blocked_terms"],unavailable_sources=record["unavailable_sources"],
                success_candidate=r0["success_candidate"],training_reward=r0["training_reward"]))
        elif op=="end":
            out=observer.end(p);control=out["control_receipt"]
            if control is None or control["physical_samples"]!=8:raise ValueError("complete actual eight-step control required")
            controls.append(control);emit(dict(operation="control_receipt",payload=control))
        else:raise ValueError("unsupported native operation "+str(op))
    expected=Counter({"None":1,"setup":2,"initial_geometry":2,"reset":1,"begin":29,"physical":232,"end":29})
    if counts!=expected or (first,last)!=(3,234) or observer._rows is not None:
        raise ValueError("complete exact same-episode prefix required")
    result.update(replay_scope(len(controls),len(approach),counts["setup"]))
    result.update(summary_gate(controls))
    result.update(status="complete",controls=controls,source_operation_counts=dict(counts),
        first_physics_step=first,last_physics_step=last,common_valid_indices=valid_fields,
        common_valid_count=len(valid_fields),common_total=371,common_missing_policy="NaN_with_false_validity",
        approach_first_m=approach[0],approach_last_m=approach[-1],
        approach_valid_samples=sum(value is not None for value in approach),
        blocked_physical_term_counts=dict(blocked),unqualified_contact_rows=unknown_contact,
        settled_control_count=len(controls),episode_identity=episode,
        original_R0_reset_native_order_qualified=False,original_frozen_task_coefficients_unchanged=True,
        missing_sources=["qualified_stability_predicate","positive_effective_opposed_contact",
            "fixed_material_anchor_and_tangential_slip","hand_table_contact_source",
            "training_reward_coefficients","native_R0_reset_owner_order"])
    result["native_imports"]=[n for n in sys.modules if n.split(".")[0] in ("omni","isaacsim","isaaclab","ropesim","warp")]
    if result["native_imports"]:raise ValueError("read-only replay imported native runtime")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(version=VERSION,status="partial",new_physics_steps=0,ppo_updates=0,
        native_qualification=False,teacher_admission=False,training_reward=None,wall_budget_s=600)
    previous_handler=signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError("600s CPU replay exhausted")))
    signal.setitimer(signal.ITIMER_REAL,600)
    try:
        with (args.output/"progress.jsonl").open("x") as f:
            def emit(row):
                f.write(json.dumps(encode_owned(row),allow_nan=False)+"\n");f.flush()
            try:run(result,emit)
            finally:f.flush();os.fsync(f.fileno())
    except Exception as exc:result.update(status="blocked",reason=type(exc).__name__+": "+str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_handler)
    result.update(elapsed_s=time.monotonic()-start,source_sha256=pinned.sha(__file__))
    with (args.output/"result.json").open("x") as f:
        json.dump(encode_owned(result),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result.get(k) for k in ("status","reason","elapsed_s","recorded_physics_rows",
        "settled_control_count","approach_valid_samples","blocked_physical_term_counts","teacher_admission")}))
    return int(result["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
