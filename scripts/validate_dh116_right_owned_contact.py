"""Independent scope binding; every original post-exit check retained verbatim."""
from pathlib import Path
import re
from scripts.dh116_pinch_margin_contact_audit import (
    original,native_artifact,lifetime_evidence,validate_native_physics_clock,native_physics_budget)
from scripts.prepare_dh116_right_pinch_native_interval import SCOPE

def validate_artifacts(output,outer_exit,cleanup,*,profile_sha256):
    from scripts import prepare_dh116_right_pinch_native_interval as operation
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
    original.require(ledger==[operation.sdk_clock_sample(row,i) for i,row in enumerate(physical,1)]
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
