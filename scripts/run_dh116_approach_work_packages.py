#!/usr/bin/env python3
"""Same single-instance scheduler; measured R0 must match full post-exit replay."""
from pathlib import Path
import json
from scripts import run_dh116_sealed_work_packages as previous
from scripts import verify_dh116_approach_r0 as entry
from scripts.validate_dh116_approach_r0 import audit
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from scripts.dh116_physx_lifetime_debugger_stopped import render_gdb_commands
base=previous.base
AUDIT_SOURCES=tuple(dict.fromkeys((*previous.AUDIT_SOURCES,*entry.FILES,
    "scripts/replay_dh116_measurement_origin.py")))

def cleanup_case(spec):
    with bindings((previous,"entry",entry)):
        return previous.cleanup_case(spec)

def audit_outcome(output,outer_exit,cleanup):
    result=previous.audit_outcome(output,outer_exit,cleanup)
    result["qualification_scope"]=entry.SCOPE
    if result.get("lifecycle_qualified"):
        marker=json.loads((Path(output)/"approach_scope.json").read_bytes())
        if marker!=entry.scope_marker():raise ValueError("exact measured approach scope restoration required")
        measured=audit(output)
        if measured.get("reset_baseline_available") is not True or measured.get("valid_approach_samples")!=232:
            raise ValueError("complete actual reset and 232-sample approach coverage required")
        result.update(approach_scope=marker,measured_approach_audit=measured)
    result.update(teacher_admission=False,grasp_qualified=False,full_R0_ready=False)
    return result

def main():
    with bindings((base,"__file__",str(Path(__file__).resolve())),(base,"ENTRY",entry.ENTRY),
        (base,"WRAPPER",entry.WRAPPER),(base,"VERSION",entry.VERSION),(base,"WORKER_ENV",entry.WORKER_ENV),
        (base,"AUDIT_NAME","approach-r0-post-exit-audit.json"),(base,"AUDIT_SOURCES",AUDIT_SOURCES),
        (base,"cleanup_case",cleanup_case),(base,"render_gdb_commands",render_gdb_commands),
        (base,"audit_outcome",audit_outcome)):
        return base.main()
if __name__=="__main__":raise SystemExit(main())
