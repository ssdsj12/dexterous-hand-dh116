#!/usr/bin/env python3
"""Serial single-attempt bounded live Reset/R0 observer scheduler."""
from pathlib import Path
from scripts import run_dh116_effective_right_work_packages as previous
from scripts import verify_dh116_live_r0_reset as entry
from scripts import prepare_dh116_live_r0_native as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from scripts.dh116_physx_lifetime_debugger_stopped import render_gdb_commands

base=previous.base
AUDIT_SOURCES=tuple(dict.fromkeys((*previous.AUDIT_SOURCES,*entry.FILES)))

def cleanup_case(spec):
    with bindings((previous,"entry",entry)):
        return previous.cleanup_case(spec)

def audit_outcome(output,outer_exit,cleanup):
    result=previous.audit_outcome(output,outer_exit,cleanup)
    result["qualification_scope"]=operation.SCOPE
    if result.get("lifecycle_qualified"):
        from scripts.validate_dh116_live_r0_reset import audit
        result["live_r0_audit"]=audit(output)
    result.update(teacher_admission=False,grasp_qualified=False,full_R0_ready=False)
    return result

def main():
    with bindings((base,"__file__",str(Path(__file__).resolve())),(base,"ENTRY",entry.ENTRY),
        (base,"WRAPPER",entry.WRAPPER),(base,"VERSION",entry.VERSION),(base,"WORKER_ENV",entry.WORKER_ENV),
        (base,"AUDIT_NAME","live-r0-post-exit-audit.json"),(base,"AUDIT_SOURCES",AUDIT_SOURCES),
        (base,"cleanup_case",cleanup_case),(base,"render_gdb_commands",render_gdb_commands),
        (base,"audit_outcome",audit_outcome)):
        return base.main()
if __name__=="__main__":raise SystemExit(main())
