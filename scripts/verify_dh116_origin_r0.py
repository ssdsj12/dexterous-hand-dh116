#!/usr/bin/env python3
"""Independent stable R0 origin with every original atomic sample retained."""
from pathlib import Path
import json,os
from scripts import verify_dh116_fast_r0_reset as previous
from scripts import prepare_dh116_live_r0_native as operation
from scripts import verify_dh116_complete_pair_r0 as complete
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_dual_hand_rod_runtime as runtime
from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import read_complete_pair_capture,VERSION as CAPTURE_VERSION
from go2_pvcnn.control.dh116.tabletop_r0_measurement_origin import OriginLiveR0Observer,VERSION as SOURCE_VERSION
ENTRY="scripts/verify_dh116_origin_r0.py"
WRAPPER="scripts/run_dh116_origin_work_packages.py"
VERSION=previous.VERSION
WORKER_ENV="DH116_ORIGIN_R0_WORKER_V1"
PARENT_PACKAGE=previous.PARENT_PACKAGE
PARENT_SCOPE=previous.PARENT_SCOPE
PARENT_RECEIPT=previous.PARENT_RECEIPT
PARENT_SHA=previous.PARENT_SHA
PROFILE_SHA=previous.PROFILE_SHA
SCOPE="stable_measurement_origin_complete_atomic_R0_234_steps_only"
FILES=tuple(dict.fromkeys((*complete.FILES,ENTRY,WRAPPER,
    "scripts/validate_dh116_origin_r0.py",
    "src/go2_pvcnn/control/dh116/tabletop_r0_measurement_origin.py",
    "tests/test_r0_measurement_origin.py","tests/test_origin_r0_native.py")))
old_preflight=previous.preflight
old_execute=previous.execute

def preflight(*args,**kwargs):
    result=old_preflight(*args,**kwargs)
    result.update(complete_pair_capture_version=CAPTURE_VERSION,measurement_origin_version=SOURCE_VERSION)
    return result

def scope_marker():
    return dict(capture_version=CAPTURE_VERSION,source_version=SOURCE_VERSION,
        bindings_restored=True,raw_capacity_overflow_retained=True,
        per_frame_evidence_retained=True,stable_origins=True,
        positive_effective_contact_qualified=False,teacher_admission=False,
        new_physics_steps=0,ppo_updates=0)

def execute(args,*rest,**kwargs):
    reader=runtime.read_pair_capture;observer=operation.LiveR0Observer
    with bindings((runtime,"read_pair_capture",read_complete_pair_capture),
            (operation,"LiveR0Observer",OriginLiveR0Observer)):
        result=old_execute(args,*rest,**kwargs)
    if runtime.read_pair_capture is not reader or operation.LiveR0Observer is not observer:
        raise RuntimeError("complete contact binding not restored")
    with (Path(args.output)/"origin_scope.json").open("x") as f:
        json.dump(scope_marker(),f,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    return result

def main():
    with bindings((previous,"ENTRY",ENTRY),(previous,"WRAPPER",WRAPPER),
        (previous,"WORKER_ENV",WORKER_ENV),(previous,"FILES",FILES),
        (previous,"preflight",preflight),(previous,"execute",execute)):
        return previous.main()
if __name__=="__main__":main()
