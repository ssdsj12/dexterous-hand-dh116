#!/usr/bin/env python3
"""Sealed raw evidence is never qualified before full post-exit validation."""
from pathlib import Path
import json
from scripts import run_dh116_origin_work_packages as previous
from scripts import verify_dh116_sealed_r0 as entry
from scripts import validate_dh116_effective_right_contact as original_audit
from scripts.dh116_deferred_physical_audit import validate_artifacts
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from scripts.dh116_physx_lifetime_debugger_stopped import render_gdb_commands
from go2_pvcnn.control.dh116.tabletop_owned_clock_projection import clock_projection_scope
base=previous.base
AUDIT_SOURCES=tuple(dict.fromkeys((*previous.AUDIT_SOURCES,*entry.FILES)))

def cleanup_case(spec):
    with bindings((previous,"entry",entry)):
        return previous.cleanup_case(spec)

def audit_outcome(output,outer_exit,cleanup):
    with clock_projection_scope(),bindings((original_audit,"validate_artifacts",validate_artifacts)):
        result=previous.audit_outcome(output,outer_exit,cleanup)
    result["qualification_scope"]=entry.SCOPE
    if result.get("lifecycle_qualified"):
        marker=json.loads((Path(output)/"sealed_audit_scope.json").read_bytes())
        if marker!=entry.scope_marker():raise ValueError("exact sealed audit restoration marker required")
        result.update(sealed_scope=marker,full_post_exit_audit_passed=True)
    result.update(teacher_admission=False,grasp_qualified=False,full_R0_ready=False)
    return result

def main():
    with bindings((base,"__file__",str(Path(__file__).resolve())),(base,"ENTRY",entry.ENTRY),
        (base,"WRAPPER",entry.WRAPPER),(base,"VERSION",entry.VERSION),(base,"WORKER_ENV",entry.WORKER_ENV),
        (base,"AUDIT_NAME","sealed-r0-post-exit-audit.json"),(base,"AUDIT_SOURCES",AUDIT_SOURCES),
        (base,"cleanup_case",cleanup_case),(base,"render_gdb_commands",render_gdb_commands),
        (base,"audit_outcome",audit_outcome)):
        return base.main()
if __name__=="__main__":raise SystemExit(main())
