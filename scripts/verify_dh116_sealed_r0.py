#!/usr/bin/env python3
"""Independent sealed owner; complete replay happens only after natural exit."""
from pathlib import Path
import json,os
from scripts import verify_dh116_origin_r0 as previous
from scripts import prepare_dh116_native_parameter_task as parameter
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from scripts.dh116_deferred_physical_audit import seal_pending,VERSION as SEAL_VERSION,SCOPE
from go2_pvcnn.control.dh116.tabletop_owned_clock_projection import clock_projection_scope,VERSION as CLOCK_VERSION
ENTRY="scripts/verify_dh116_sealed_r0.py"
WRAPPER="scripts/run_dh116_sealed_work_packages.py"
VERSION=previous.VERSION
WORKER_ENV="DH116_SEALED_R0_WORKER_V1"
PARENT_PACKAGE=previous.PARENT_PACKAGE
PARENT_SCOPE=previous.PARENT_SCOPE
PARENT_RECEIPT=previous.PARENT_RECEIPT
PARENT_SHA=previous.PARENT_SHA
PROFILE_SHA=previous.PROFILE_SHA
FILES=tuple(dict.fromkeys((*previous.FILES,ENTRY,WRAPPER,
    "scripts/dh116_deferred_physical_audit.py",
    "src/go2_pvcnn/control/dh116/tabletop_owned_clock_projection.py",
    "tests/test_owned_clock_projection.py","tests/test_deferred_physical_audit.py","tests/test_sealed_r0_native.py","tests/sealed_owner_cpu_fixture.py")))
old_preflight=previous.preflight
old_execute=previous.execute

def preflight(*args,**kwargs):
    result=old_preflight(*args,**kwargs)
    result.update(physical_audit_version=SEAL_VERSION,clock_projection_version=CLOCK_VERSION,
                  producer_audit_deferred=True)
    return result

def scope_marker():
    return dict(version=SEAL_VERSION,clock_projection_version=CLOCK_VERSION,
        bindings_restored=True,full_producer_audit_deferred=True,
        post_exit_complete_audit_required=True,native_qualification=False,teacher_admission=False,
        new_physics_steps=0,ppo_updates=0)

def execute(args,*rest,**kwargs):
    old=parameter.inspect_interval
    with clock_projection_scope(),bindings((parameter,"inspect_interval",seal_pending)):
        result=old_execute(args,*rest,**kwargs)
    if parameter.inspect_interval is not old:raise RuntimeError("audit owner binding not restored")
    with (Path(args.output)/"sealed_audit_scope.json").open("x") as file:
        json.dump(scope_marker(),file,allow_nan=False);file.write("\n");file.flush();os.fsync(file.fileno())
    return result

def main():
    with bindings((previous,"ENTRY",ENTRY),(previous,"WRAPPER",WRAPPER),
        (previous,"WORKER_ENV",WORKER_ENV),(previous,"FILES",FILES),
        (previous,"preflight",preflight),(previous,"execute",execute)):
        return previous.main()
if __name__=="__main__":main()
