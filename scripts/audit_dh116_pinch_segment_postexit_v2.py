"""Independent disk-only repair of the segment experiment's post-exit routing.

The consumed scheduler receipt is read-only. No Isaac, solver construction, GPU
step, optimizer update, result edit or success-pool admission is performed.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from scripts import dh116_pinch_margin_contact_audit as margin_audit
from scripts import prepare_dh116_pinch_margin_contact as margin
from scripts import prepare_dh116_pinch_segment_contact as operation
from scripts import run_dh116_pinch_segment_contact_work_packages as wrapper
from scripts import verify_dh116_pinch_segment_contact as entry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from scripts.dh116_pinch_margin_contact_audit import original,native_artifact,lifetime_evidence,validate_native_physics_clock,native_physics_budget

SCOPE=operation.SCOPE
ROOT=Path(__file__).resolve().parents[1]
EXECUTED_ROOT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/integration")
RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-pinch-segment-contact-native/186855563b67e02e6998797fb57d416da79d35bfa70311b96815ff9f844f76df/attempt-oqk81uft/receipt.json")
RECEIPT_SHA="97f83e1e001ea28cfa08ebcf4017d717f7adcc11aad38402280e780f48311746"
OUTER_SHA="e209c69eea0f2c19b2ae75ede964cdcaa658c52c06b6cdaab6fdccaa913055dc"
NATIVE=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-pinch-segment-contact-20261113-41heHL/native-runs/flex.f7-pinch-segment-contact-native/attempt-oqk81uft/native")
RECIPE="configs/dh116_execution/deformx_pinch_segment_contact_v1.json"
RECIPE_SHA="ab5e73125f1bae212b8a4865a362dd1dad208a3535f58deab2e84dcf04802c80"
CODE_SHA="186855563b67e02e6998797fb57d416da79d35bfa70311b96815ff9f844f76df"
WORKER_SHA="61dd815e39e6ea293b1f7dd51333db1843386b565aa119b4cb4f57162d071ed1"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def load_locked():
    if RECEIPT.resolve(strict=True)!=RECEIPT or sha(RECEIPT)!=RECEIPT_SHA:
        raise ValueError("consumed receipt bytes differ")
    receipt=json.loads(RECEIPT.read_bytes())
    if (receipt["native_qualification"] is not False or receipt["outer_exit_code"]!=0
            or receipt["debugger_inferior_signal"] is not None or receipt["qualification_scope"]!=SCOPE
            or receipt["code_config_sha256"]!=CODE_SHA or receipt["command_spec_sha256"]!=RECIPE_SHA
            or Path(receipt["output"])!=NATIVE
            or "result exact guarded closure execution scope differs" not in receipt["reason"]):
        raise ValueError("only the actual consumed version-binding failure can be re-audited")
    for name,expected in receipt["artifact_sha256"].items():
        path=native_artifact(NATIVE,name)
        if sha(path)!=expected:raise ValueError("native artifact bytes differ: "+name)
    outer=RECEIPT.parent/"outer_exit.json"
    if sha(outer)!=OUTER_SHA:raise ValueError("outer exit bytes differ")
    exited=json.loads(outer.read_bytes())
    if (exited["outer_exit_code"]!=0 or exited["debugger_inferior_exit_code"]!=0
            or exited["debugger_inferior_signal"] is not None or exited["duration_seconds"]>180
            or sha(RECEIPT.parent/"native.log")!=exited["log_sha256"]):
        raise ValueError("natural bounded exit and immutable GDB log required")
    from scripts import run_dh116_work_packages as scheduler
    repo=scheduler.Repository(ROOT,evidence_dir=RECEIPT.parents[3])
    p=dict(command_spec=RECIPE,dependencies=[entry.PARENT_PACKAGE],
           dependency_scopes={entry.PARENT_PACKAGE:dict(scope=entry.PARENT_SCOPE,native=True)})
    frozen=repo.native_recipe(p)
    if (frozen["code_config_sha256"]!=CODE_SHA or frozen["command_spec_sha256"]!=RECIPE_SHA
            or entry.preflight(ROOT)["code_config_sha256"]!=WORKER_SHA):
        raise ValueError("executed source or recipe changed; cannot re-audit with a replacement runtime")
    # Preserve the exact recorded source identity across isolated audit worktrees.
    # Both files must independently match the frozen runtime configuration hash.
    reader=margin_audit.reader
    for root in (ROOT,EXECUTED_ROOT):
        if sha(root/reader.CONFIG)!=reader.CONFIG_SHA:
            raise ValueError("recorded parameter source config differs")
    return receipt

def reaudit():
    load_locked()
    # The previous validator imported the old operation *inside* its function.
    # Its module global cannot fix that. Bind an independently AST-checked body.
    with bindings((margin_audit,"validate_artifacts",validate_artifacts),
                  (margin_audit.reader,"ROOT",EXECUTED_ROOT)):
        result=wrapper.audit_outcome(NATIVE,RECEIPT.parent/"outer_exit.json","explicit")
    result.update(reaudit_version="dh116-pinch-segment-postexit-v2",
        evidence_kind="disk_only_reaudit_of_existing_native_execution",
        original_scheduler_receipt=str(RECEIPT),original_scheduler_receipt_sha256=RECEIPT_SHA,
        original_scheduler_qualification=False,original_recipe_sha256=RECIPE_SHA,
        original_code_config_sha256=CODE_SHA,new_native_steps=0,new_ppo_updates=0,
        teacher_admission=False,grasp_qualified=False,original_receipt_unchanged=sha(RECEIPT)==RECEIPT_SHA)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();result=reaudit()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({k:result.get(k) for k in ("lifecycle_qualified","reason","qualification_scope","new_native_steps")}))

def validate_artifacts(output,outer_exit,cleanup,*,profile_sha256):
    from scripts import prepare_dh116_pinch_segment_contact as operation
    output,outer_exit=Path(output).resolve(),Path(outer_exit).resolve()
    original.require(cleanup=="explicit","exact explicit cleanup required")
    original.require((operation.SETUP_STEPS,operation.MEASURED_STEPS,operation.TOTAL_STEPS)==(2,232,234),
                     "exact closure all-step budget required")
    outer=original._outer(output,outer_exit)
    result=original.read(native_artifact(output,"result.json"))
    original.exact(result,dict(version=operation.VERSION,case="explicit",physics_steps=234,
        measurement_steps=232,rod_steps=232,ppo_updates=0,play_simulations=False,failure=None,
        process_exit_verified=False,grasp_qualified=False,teacher_admission=False),
        "result exact guarded closure execution scope differs")
    acquisition=result["acquisition"]
    original.exact(acquisition,dict(version=operation.VERSION,physics_rows=232,ppo_updates=0,
        native_qualification=False,teacher_admission=False,grasp_qualified=False,
        hand_table_source_available=False,no_hand_table_contact_proven=False),
        "acquisition guarded closure scope differs")
    code=result.get("code_config_sha256")
    original.require(isinstance(code,str) and re.fullmatch("[0-9a-f]{64}",code)
        and acquisition.get("code_config_sha256")==code,"result/acquisition source identity differs")
    original.require(isinstance(profile_sha256,str) and re.fullmatch("[0-9a-f]{64}",profile_sha256)
        and acquisition.get("profile_sha256")==profile_sha256,
        "explicit expected frozen profile SHA and acquisition binding required")
    original.require(acquisition.get("action_schedule")==operation.action_schedule().tolist(),
                     "saved action schedule differs from frozen operation")
    lifetime=lifetime_evidence(output,result,expected_steps=234)
    journal=operation.inspect_interval(native_artifact(output,"physical.jsonl"),complete=True)
    original.require(journal==acquisition["journal"],"producer summary differs from independent journal replay")
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    decoded=[(r["operation"],decode_owned(r["payload"])) for r in original.rows(output/"physical.jsonl")[1:]]
    physical=[row for op,row in decoded if op=="physical"]
    clock_rows=[(i,row["exchange"]["physics_step"]) for i,row in enumerate(physical,1)]
    clock=validate_native_physics_clock(result,clock_rows,setup_steps=2,total_steps=234)
    original.require(acquisition["physics_clock"]==clock,"acquisition standard clock differs")
    ledger=original.rows(native_artifact(output,"sdk_clock.jsonl"))
    original.require(ledger==[margin.sdk_clock_sample(row,i) for i,row in enumerate(physical,1)]
        and all(type(row.get(key)) is int for row in ledger
                for key in ("phase_step","sdk_step","policy_step","substep")),
        "scheduler SDK/rod/post clock differs from durable measured rows")
    budget=native_physics_budget(initialization_physics_steps=2,warmup_physics_steps=0,
        measurement_physics_steps=232,other_physics_advancing_steps=0)
    original.require(result["physics_budget"]==acquisition["physics_budget"]==budget,
                     "standard all-step budget differs")
    for row in physical:
        original.require(row["exchange"]["wrist_projection"]["clearance_floor_m"]==.0021,
                         "original physical wrist safety floor differs")
    geometry=original._geometry(output,acquisition,decoded)
    return dict(lifecycle_qualified=True,qualification_scope=SCOPE,code_config_sha256=code,
        profile_sha256=profile_sha256,action_schedule=operation.action_schedule().tolist(),
        physics_steps=234,measurement_steps=232,rod_steps=232,ppo_updates=0,
        physics_clock=clock,physics_budget=budget,journal=journal,initial_geometry=geometry,
        **outer,**lifetime)

if __name__=="__main__":main()
