#!/usr/bin/env python3
"""Independent worker reentry and post-exit-only numerical observer qualification."""
from pathlib import Path
from scripts import run_dh116_projected_pinch_work_packages as previous
from scripts import verify_dh116_health_projected_pinch as entry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

base=previous.base
AUDIT_SOURCES=tuple(dict.fromkeys((*previous.AUDIT_SOURCES,*entry.FILES)))
AUDIT_NAME="health-projected-pinch-post-exit-audit.json"

def cleanup_case(spec):
    with bindings((previous,"entry",entry)):return previous.cleanup_case(spec)

def audit_outcome(output,outer_exit,cleanup):
    from scripts import prepare_dh116_health_projected_pinch as operation
    from scripts import validate_dh116_health_projected_pinch as audit
    marker=operation.scope_marker()
    with operation.scenario_scope(),bindings((previous,"entry",entry),
        (operation.previous,"scope_marker",lambda:marker)):
        parent=previous.audit_outcome(output,outer_exit,cleanup)
    return audit.audit_outcome(output,parent)

def main():
    with bindings((base,"__file__",str(Path(__file__).resolve())),(base,"ENTRY",entry.ENTRY),
        (base,"WRAPPER",entry.WRAPPER),(base,"VERSION",entry.VERSION),(base,"WORKER_ENV",entry.WORKER_ENV),
        (base,"AUDIT_NAME",AUDIT_NAME),(base,"AUDIT_SOURCES",AUDIT_SOURCES),
        (base,"cleanup_case",cleanup_case),(base,"render_gdb_commands",previous.previous.render_gdb_commands),
        (base,"audit_outcome",audit_outcome)):return base.main()

if __name__=="__main__":raise SystemExit(main())
