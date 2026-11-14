#!/usr/bin/env python3
"""Single opt-in measured approach, preserving sealed owner lifecycle."""
from pathlib import Path
import json,os
from scripts import verify_dh116_sealed_r0 as previous
from scripts import verify_dh116_effective_right_pinch as file_owner
from scripts import prepare_dh116_live_r0_native as operation
from scripts.prepare_dh116_live_measured_approach import MeasuredApproachSession,VERSION as APPROACH_VERSION
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
ENTRY="scripts/verify_dh116_approach_r0.py"
WRAPPER="scripts/run_dh116_approach_work_packages.py"
VERSION=previous.VERSION
WORKER_ENV="DH116_APPROACH_R0_WORKER_V1"
PARENT_PACKAGE=previous.PARENT_PACKAGE
PARENT_SCOPE=previous.PARENT_SCOPE
PARENT_RECEIPT=previous.PARENT_RECEIPT
PARENT_SHA=previous.PARENT_SHA
PROFILE_SHA=previous.PROFILE_SHA
SCOPE="actual_reset_and_same_clock_measured_approach_R0_234_steps_only"
FILES=tuple(dict.fromkeys((*previous.FILES,ENTRY,WRAPPER,
    "scripts/prepare_dh116_live_measured_approach.py",
    "scripts/validate_dh116_approach_r0.py",
    "src/go2_pvcnn/control/dh116/tabletop_effective_right_r0_approach.py",
    "src/go2_pvcnn/control/dh116/tabletop_r0_measured_approach.py",
    "tests/test_live_measured_approach.py","tests/test_approach_r0_native.py")))
old_preflight=previous.preflight
old_execute=previous.execute

def preflight(*args,**kwargs):
    with bindings((file_owner,"FILES",FILES)):
        result=old_preflight(*args,**kwargs)
    result.update(measured_approach_version=APPROACH_VERSION)
    return result

def scope_marker():
    return dict(version=APPROACH_VERSION,bindings_restored=True,
        actual_reset_baseline_required=True,same_clock_source_required=True,
        full_post_exit_recomputation_required=True,
        native_qualification=False,teacher_admission=False,new_physics_steps=0,ppo_updates=0)

def execute(args,*rest,**kwargs):
    old=operation.LiveSession
    with bindings((operation,"LiveSession",MeasuredApproachSession)):
        result=old_execute(args,*rest,**kwargs)
    if operation.LiveSession is not old:raise RuntimeError("live session binding not restored")
    with (Path(args.output)/"approach_scope.json").open("x") as file:
        json.dump(scope_marker(),file,allow_nan=False);file.write("\n");file.flush();os.fsync(file.fileno())
    return result

def main():
    with bindings((previous,"ENTRY",ENTRY),(previous,"WRAPPER",WRAPPER),
        (previous,"WORKER_ENV",WORKER_ENV),(previous,"FILES",FILES),
        (previous,"preflight",preflight),(previous,"execute",execute)):
        return previous.main()
if __name__=="__main__":main()
