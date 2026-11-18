#!/usr/bin/env python3
"""One-attempt complete contact scope, same physical and natural exit guards."""
from pathlib import Path
import json
from scripts import run_dh116_fast_r0_work_packages as previous
from scripts import verify_dh116_origin_r0 as entry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from scripts.dh116_physx_lifetime_debugger_stopped import render_gdb_commands
base=previous.base
AUDIT_SOURCES=tuple(dict.fromkeys((*previous.AUDIT_SOURCES,*entry.FILES)))

def cleanup_case(spec):
    with bindings((previous,"entry",entry)):
        return previous.cleanup_case(spec)

def check_scope(output):
    marker=json.loads((Path(output)/"origin_scope.json").read_bytes())
    if marker!=entry.scope_marker():raise ValueError("exact complete-contact restoration marker required")
    return marker

def audit_outcome(output,outer_exit,cleanup):
    result=previous.audit_outcome(output,outer_exit,cleanup)
    result["qualification_scope"]=entry.SCOPE
    if result.get("lifecycle_qualified"):
        from scripts.validate_dh116_origin_r0 import audit
        result["origin_scope"]=check_scope(output)
        result["origin_r0_audit"]=audit(output)
    result.update(teacher_admission=False,grasp_qualified=False,full_R0_ready=False)
    return result

def main():
    with bindings((base,"__file__",str(Path(__file__).resolve())),(base,"ENTRY",entry.ENTRY),
        (base,"WRAPPER",entry.WRAPPER),(base,"VERSION",entry.VERSION),(base,"WORKER_ENV",entry.WORKER_ENV),
        (base,"AUDIT_NAME","origin-r0-post-exit-audit.json"),(base,"AUDIT_SOURCES",AUDIT_SOURCES),
        (base,"cleanup_case",cleanup_case),(base,"render_gdb_commands",render_gdb_commands),
        (base,"audit_outcome",audit_outcome)):
        return base.main()
if __name__=="__main__":raise SystemExit(main())
