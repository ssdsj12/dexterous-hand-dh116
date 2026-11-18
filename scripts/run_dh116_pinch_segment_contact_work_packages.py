#!/usr/bin/env python3
"""Frozen 234-step candidate scheduler; no retry or Teacher qualification."""
from pathlib import Path
from scripts import run_dh116_physx_lifetime_work_packages as base
from scripts import run_dh116_native_parameter_task_work_packages as previous
from scripts import verify_dh116_pinch_segment_contact as entry
from scripts.dh116_physx_lifetime_debugger_stopped import render_gdb_commands
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
AUDIT_SOURCES=tuple(dict.fromkeys((*previous.AUDIT_SOURCES,*entry.FILES)))
base_case=base.cleanup_case

def cleanup_case(spec):
    case=base_case(spec);args=spec["argv"];flag="--approved-code-config-sha256"
    expected=dict(physics_steps=234,updates=0,environments=1,gpu_index=3,timeout_seconds=180)
    if (case!="explicit" or any(spec["budget"].get(k)!=v for k,v in expected.items())
        or spec["validator"]["equals"].get("version")!=entry.VERSION
        or args.count(flag)!=1 or args.index(flag)+1>=len(args)
        or not base.scheduler.SHA256.fullmatch(args[args.index(flag)+1])
        or spec["validator"]["equals"].get("code_config_sha256")!=args[args.index(flag)+1]
        or spec.get("dependency_receipt")!={"package":entry.PARENT_PACKAGE,"qualification_scope":entry.PARENT_SCOPE}):
        raise ValueError("exact 234-step left candidate recipe and real parent required")
    return case

def audit_outcome(output,outer_exit,cleanup):
    from scripts import validate_dh116_native_parameter_task as audit
    from scripts import prepare_dh116_pinch_segment_contact as operation
    from scripts import dh116_pinch_margin_contact_audit as new_audit
    with operation.scenario_scope(),bindings((operation,"PROFILE_SHA",entry.PROFILE_SHA)):
        result=audit.audit_outcome(output,outer_exit,cleanup,profile_sha256=entry.PROFILE_SHA)
    result.update(qualification_scope=operation.SCOPE,teacher_admission=False,grasp_qualified=False,
        contact_success_qualified=False)
    return result

def main():
    with bindings((base,"__file__",str(Path(__file__).resolve())),(base,"ENTRY",entry.ENTRY),
        (base,"WRAPPER",entry.WRAPPER),(base,"VERSION",entry.VERSION),(base,"WORKER_ENV",entry.WORKER_ENV),
        (base,"AUDIT_NAME","pinch-segment-contact-post-exit-audit.json"),(base,"AUDIT_SOURCES",AUDIT_SOURCES),
        (base,"cleanup_case",cleanup_case),(base,"render_gdb_commands",render_gdb_commands),
        (base,"audit_outcome",audit_outcome)):
        return base.main()
if __name__=="__main__":raise SystemExit(main())
