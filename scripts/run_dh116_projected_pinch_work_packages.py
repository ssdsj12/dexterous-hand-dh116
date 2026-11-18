#!/usr/bin/env python3
"""Fresh bounded recipe; preserve actual original protection and post-exit proof."""
from pathlib import Path
import json
from scripts import run_dh116_effective_right_work_packages as previous
from scripts import verify_dh116_projected_pinch as entry
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
base=previous.base
AUDIT_SOURCES=tuple(dict.fromkeys((*previous.AUDIT_SOURCES,*entry.FILES)))
def cleanup_case(spec):
    with bindings((previous,"entry",entry)):return previous.cleanup_case(spec)
def audit_outcome(output,outer_exit,cleanup):
    from scripts import prepare_dh116_projected_pinch_native as operation
    from scripts import validate_dh116_native_parameter_task as audit
    from scripts import validate_dh116_effective_right_contact as original
    from scripts import dh116_deferred_physical_audit as deferred
    from go2_pvcnn.control.dh116.tabletop_owned_clock_projection import clock_projection_scope
    with bindings((operation,"PROFILE_SHA",entry.PROFILE_SHA),
        (original,"validate_artifacts",deferred.validate_artifacts),(deferred,"SCOPE",operation.SCOPE)):
        with operation.scenario_scope(),clock_projection_scope():
            result=audit.audit_outcome(output,outer_exit,cleanup,profile_sha256=entry.PROFILE_SHA)
    if result.get("lifecycle_qualified"):
        for name in ("left_pinch_approach_scope.json","projected_pinch_scope.json"):
            if json.loads((Path(output)/name).read_bytes())!=operation.scope_marker():
                raise ValueError("new projected scope and restored owner required")
        result.update(full_post_exit_audit_passed=True)
    result.update(qualification_scope=operation.SCOPE,teacher_admission=False,
        grasp_qualified=False,contact_success_qualified=False,fixed_unprojected_path_qualified=False)
    return result
def main():
    with bindings((base,"__file__",str(Path(__file__).resolve())),(base,"ENTRY",entry.ENTRY),
        (base,"WRAPPER",entry.WRAPPER),(base,"VERSION",entry.VERSION),(base,"WORKER_ENV",entry.WORKER_ENV),
        (base,"AUDIT_NAME","projected-pinch-post-exit-audit.json"),(base,"AUDIT_SOURCES",AUDIT_SOURCES),
        (base,"cleanup_case",cleanup_case),(base,"render_gdb_commands",previous.render_gdb_commands),
        (base,"audit_outcome",audit_outcome)):return base.main()
if __name__=="__main__":raise SystemExit(main())
