#!/usr/bin/env python3
"""New frozen live-observer recipe; reuse original native action/lifecycle."""
from pathlib import Path
import hashlib,json
from scripts import verify_dh116_effective_right_pinch as previous
from scripts import prepare_dh116_live_r0_native as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ENTRY="scripts/verify_dh116_live_r0_reset.py"
WRAPPER="scripts/run_dh116_live_r0_work_packages.py"
VERSION=previous.VERSION # retained original physics result contract
WORKER_ENV="DH116_LIVE_R0_RESET_WORKER_V1"
PARENT_PACKAGE=previous.PARENT_PACKAGE
PARENT_SCOPE=previous.PARENT_SCOPE
PARENT_RECEIPT=previous.PARENT_RECEIPT
PARENT_SHA=previous.PARENT_SHA
PROFILE_SHA=previous.PROFILE_SHA
ROOT=Path(__file__).resolve().parents[1]
FILES=tuple(dict.fromkeys((*previous.FILES,ENTRY,WRAPPER,
    "scripts/prepare_dh116_live_r0_native.py","scripts/validate_dh116_live_r0_reset.py",
    operation.CONFIG,"src/go2_pvcnn/control/dh116/tabletop_r0_live_observer.py",
    "src/go2_pvcnn/control/dh116/tabletop_r0_after_physical_reset.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_r0_replay.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_control_step.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_task.py",
    "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json",
    "configs/dh116_training/deformable_task_budget_examples_v2.json",
    "scripts/replay_dh116_native_parameter_r0.py",
    "tests/test_r0_after_physical_reset.py","tests/test_r0_live_observer.py",
    "tests/test_live_r0_native_session.py","tests/test_live_r0_native_worker.py")))
old_preflight=previous.preflight
old_execute=previous.execute

def preflight(*args,**kwargs):
    result=old_preflight(*args,**kwargs)
    result["live_observer_version"]=operation.VERSION
    return result

def execute(*args,**kwargs):
    with bindings((previous.operation,"run_existing_app",operation.run_existing_app)):
        return old_execute(*args,**kwargs)

def main():
    with bindings((previous,"ENTRY",ENTRY),(previous,"WRAPPER",WRAPPER),
        (previous,"WORKER_ENV",WORKER_ENV),(previous,"FILES",FILES),
        (previous,"preflight",preflight),(previous,"execute",execute)):
        return previous.main()
if __name__=="__main__":main()
