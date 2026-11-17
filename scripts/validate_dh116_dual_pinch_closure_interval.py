"""Independent post-exit audit for exactly 2+8+8 guarded closure execution."""
from pathlib import Path
import re

from scripts import validate_dh116_dual_pinch_interval as original
from scripts.dh116_dual_pinch_lifetime_evidence import lifetime_evidence
from scripts.run_dh116_work_packages import (
    native_artifact,native_physics_budget,validate_native_physics_clock)

AUDIT_NAME="dual-pinch-closure-post-exit-audit.json"
SCOPE="exact_2_setup_plus_16_measured_guarded_closure_execution_only"
ERRORS=(OSError,ValueError,KeyError,TypeError,IndexError,OverflowError,RuntimeError,AttributeError,ImportError)


def validate_artifacts(output,outer_exit,cleanup,*,profile_sha256):
    from scripts import prepare_dh116_dual_pinch_closure_interval as operation
    output,outer_exit=Path(output).resolve(),Path(outer_exit).resolve()
    original.require(cleanup=="explicit","exact explicit cleanup required")
    original.require((operation.SETUP_STEPS,operation.MEASURED_STEPS,operation.TOTAL_STEPS)==(2,16,18),
                     "exact closure all-step budget required")
    outer=original._outer(output,outer_exit)
    result=original.read(native_artifact(output,"result.json"))
    original.exact(result,dict(version=operation.VERSION,case="explicit",physics_steps=18,
        measurement_steps=16,rod_steps=16,ppo_updates=0,play_simulations=False,failure=None,
        process_exit_verified=False,grasp_qualified=False,teacher_admission=False),
        "result exact guarded closure execution scope differs")
    acquisition=result["acquisition"]
    original.exact(acquisition,dict(version=operation.VERSION,physics_rows=16,ppo_updates=0,
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
    lifetime=lifetime_evidence(output,result,expected_steps=18)
    journal=operation.inspect_interval(native_artifact(output,"physical.jsonl"),complete=True)
    original.require(journal==acquisition["journal"],"producer summary differs from independent journal replay")
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    decoded=[(r["operation"],decode_owned(r["payload"])) for r in original.rows(output/"physical.jsonl")[1:]]
    physical=[row for op,row in decoded if op=="physical"]
    clock_rows=[(i,row["exchange"]["physics_step"]) for i,row in enumerate(physical,1)]
    clock=validate_native_physics_clock(result,clock_rows,setup_steps=2,total_steps=18)
    original.require(acquisition["physics_clock"]==clock,"acquisition standard clock differs")
    ledger=original.rows(native_artifact(output,"sdk_clock.jsonl"))
    original.require(ledger==[operation.sdk_clock_sample(row,i) for i,row in enumerate(physical,1)]
        and all(type(row.get(key)) is int for row in ledger
                for key in ("phase_step","sdk_step","policy_step","substep")),
        "scheduler SDK/rod/post clock differs from durable measured rows")
    budget=native_physics_budget(initialization_physics_steps=2,warmup_physics_steps=0,
        measurement_physics_steps=16,other_physics_advancing_steps=0)
    original.require(result["physics_budget"]==acquisition["physics_budget"]==budget,
                     "standard all-step budget differs")
    for row in physical:
        original.require(row["exchange"]["wrist_projection"]["clearance_floor_m"]==.0021,
                         "original physical wrist safety floor differs")
    geometry=original._geometry(output,acquisition,decoded)
    return dict(lifecycle_qualified=True,qualification_scope=SCOPE,code_config_sha256=code,
        profile_sha256=profile_sha256,action_schedule=operation.action_schedule().tolist(),
        physics_steps=18,measurement_steps=16,rod_steps=16,ppo_updates=0,
        physics_clock=clock,physics_budget=budget,journal=journal,initial_geometry=geometry,
        **outer,**lifetime)


def _partial(output):
    """Report what was observed even after a failed gate; never grant qualification."""
    observed=None
    try:
        result=original.read(output/"result.json")
        observed={key:result.get(key) for key in (
            "version","case","physics_steps","measurement_steps","rod_steps","ppo_updates","failure")}
    except ERRORS:
        pass
    journal=None;error=None
    try:
        from scripts.prepare_dh116_dual_pinch_closure_interval import inspect_interval
        journal=inspect_interval(output/"physical.jsonl",complete=False)
    except ERRORS as caught:
        error=type(caught).__name__+": "+str(caught)
    return dict(observed_result=observed,partial_journal=journal,partial_journal_error=error)


def audit_outcome(output,outer_exit,cleanup,*,profile_sha256=None):
    """Expected profile SHA is supplied by the independently frozen worker."""
    output=Path(output)
    try:
        result=validate_artifacts(output,outer_exit,cleanup,profile_sha256=profile_sha256)
    except ERRORS as error:
        result=dict(lifecycle_qualified=False,reason=type(error).__name__+": "+str(error),**_partial(output))
    result.update(case=cleanup,same_rod_contact_qualified=False,grasp_qualified=False,
        teacher_admission=False,hand_table_contacts=None,hand_table_source_available=False,
        no_hand_table_contact_proven=False)
    result.setdefault("simulation_liveness",{})
    for phase in original.PHASES:
        try:
            result["simulation_liveness"][phase]=original.read(output/(phase+".refs.json"))["resources"]["simulation"].get("alive")
        except ERRORS:
            result["simulation_liveness"][phase]=None
    result["available_artifact_sha256"]={path.name:original.sha(path) for path in sorted(output.glob("*"))
        if path.is_file() and not path.is_symlink() and path.name!=AUDIT_NAME}
    return result
