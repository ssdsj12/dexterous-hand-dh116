#!/usr/bin/env python3
"""Independent complete atomic contact/R0 evidence scope, original actions intact."""
from pathlib import Path
import json,os
from scripts import verify_dh116_fast_r0_reset as previous
from scripts import prepare_dh116_live_r0_native as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_dual_hand_rod_runtime as runtime
from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import read_complete_pair_capture,VERSION as CAPTURE_VERSION
from go2_pvcnn.control.dh116.tabletop_live_effective_pad_source import CompletePairLiveR0Observer,VERSION as SOURCE_VERSION
ENTRY="scripts/verify_dh116_complete_pair_r0.py"
WRAPPER="scripts/run_dh116_complete_pair_work_packages.py"
VERSION=previous.VERSION
WORKER_ENV="DH116_COMPLETE_PAIR_R0_WORKER_V1"
PARENT_PACKAGE=previous.PARENT_PACKAGE
PARENT_SCOPE=previous.PARENT_SCOPE
PARENT_RECEIPT=previous.PARENT_RECEIPT
PARENT_SHA=previous.PARENT_SHA
PROFILE_SHA=previous.PROFILE_SHA
SCOPE="retained_atomic_contact_negative_pad_live_R0_234_steps_only"
FILES=tuple(dict.fromkeys((*previous.FILES,ENTRY,WRAPPER,
    "scripts/validate_dh116_complete_pair_r0.py",
    "scripts/audit_dh116_bilateral_effective_pad_regions.py",
    "src/go2_pvcnn/control/dh116/tabletop_complete_pair_capture.py",
    "src/go2_pvcnn/control/dh116/tabletop_live_effective_pad_source.py",
    "src/go2_pvcnn/control/dh116/tabletop_r0_effective_pad_source.py",
    "tests/test_complete_pair_capture.py","tests/test_live_effective_pad_source.py",
    "tests/test_live_effective_pad_epoch.py","tests/test_live_effective_pad_accounting.py",
    "tests/test_complete_pair_live_owner.py","tests/test_complete_pair_native_worker.py",
    "tests/test_complete_pair_native_audit.py")))
old_preflight=previous.preflight
old_execute=previous.execute

def preflight(*args,**kwargs):
    result=old_preflight(*args,**kwargs)
    result.update(complete_pair_capture_version=CAPTURE_VERSION,live_negative_source_version=SOURCE_VERSION)
    return result

def scope_marker():
    return dict(capture_version=CAPTURE_VERSION,source_version=SOURCE_VERSION,
        bindings_restored=True,raw_capacity_overflow_retained=True,
        positive_effective_contact_qualified=False,teacher_admission=False,
        new_physics_steps=0,ppo_updates=0)

def execute(args,*rest,**kwargs):
    reader=runtime.read_pair_capture;observer=operation.LiveR0Observer
    with bindings((runtime,"read_pair_capture",read_complete_pair_capture),
            (operation,"LiveR0Observer",CompletePairLiveR0Observer)):
        result=old_execute(args,*rest,**kwargs)
    if runtime.read_pair_capture is not reader or operation.LiveR0Observer is not observer:
        raise RuntimeError("complete contact binding not restored")
    with (Path(args.output)/"complete_pair_scope.json").open("x") as f:
        json.dump(scope_marker(),f,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    return result

def main():
    with bindings((previous,"ENTRY",ENTRY),(previous,"WRAPPER",WRAPPER),
        (previous,"WORKER_ENV",WORKER_ENV),(previous,"FILES",FILES),
        (previous,"preflight",preflight),(previous,"execute",execute)):
        return previous.main()
if __name__=="__main__":main()
