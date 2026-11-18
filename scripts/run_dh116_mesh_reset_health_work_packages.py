#!/usr/bin/env python3
"""Exact once-only bounded worker; full raw observation audit after natural exit."""
from pathlib import Path
from scripts import run_dh116_health_projected_pinch_work_packages as previous
from scripts import verify_dh116_mesh_reset_health_projected_pinch as entry
from scripts import prepare_dh116_mesh_reset_health_projected_pinch as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

base=previous.base
AUDIT_SOURCES=tuple(dict.fromkeys((*previous.AUDIT_SOURCES,*entry.FILES)))
AUDIT_NAME="mesh-reset-health-projected-pinch-post-exit-audit.json"

def cleanup_case(spec):
    with bindings((previous,"entry",entry)):
        return previous.cleanup_case(spec)

def audit_outcome(output,outer_exit,cleanup):
    with operation.scenario_scope(),bindings((previous,"entry",entry)):
        parent=previous.audit_outcome(output,outer_exit,cleanup)
        detail=None;prefix=None;prefix_error=None
        if parent.get("lifecycle_qualified") is True:
            detail=operation.inspect_observations(Path(output),complete=True,parent_lifecycle_qualified=True)
            for key in ("reset_mesh_qualified","substep_contact_frame_qualified",
                        "same_clock_health_getter_qualified"):
                if detail.get(key) is not True:
                    raise ValueError("complete native observation source not qualified: "+key)
        else:
            try:
                prefix=operation.inspect_observations(Path(output),complete=False,
                    parent_lifecycle_qualified=False)
            except Exception as error:
                prefix_error=type(error).__name__+": "+str(error)
    return dict(parent,qualification_scope=operation.SCOPE,mesh_observation_audit=detail,
        mesh_observation_failure_prefix=prefix,mesh_observation_prefix_error=prefix_error,
        reset_mesh_qualified=detail is not None,
        substep_contact_frame_qualified=detail is not None,
        same_clock_health_getter_qualified=detail is not None,
        teacher_admission=False,grasp_qualified=False,contact_success_qualified=False,
        hold_motion_qualified=False,live_r0_qualified=False)

def main():
    with bindings((base,"__file__",str(Path(__file__).resolve())),
        (base,"ENTRY",entry.ENTRY),(base,"WRAPPER",entry.WRAPPER),(base,"VERSION",entry.VERSION),
        (base,"WORKER_ENV",entry.WORKER_ENV),(base,"AUDIT_NAME",AUDIT_NAME),
        (base,"AUDIT_SOURCES",AUDIT_SOURCES),(base,"cleanup_case",cleanup_case),
        (base,"render_gdb_commands",previous.previous.previous.render_gdb_commands),
        (base,"audit_outcome",audit_outcome)):
        return base.main()

if __name__=="__main__":raise SystemExit(main())
