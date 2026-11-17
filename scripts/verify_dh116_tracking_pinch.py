#!/usr/bin/env python3
"""Independent tracking-aware candidate; original parent, limits and 234-step budget."""
from contextlib import contextmanager
import argparse
import json
import hashlib
from pathlib import Path
from scripts import verify_dh116_effective_right_pinch as previous
from scripts import prepare_dh116_tracking_pinch_native as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ENTRY="scripts/verify_dh116_tracking_pinch.py"
WRAPPER="scripts/run_dh116_tracking_pinch_work_packages.py"
VERSION=operation.VERSION
WORKER_ENV="DH116_TRACKING_AWARE_PINCH_WORKER_V1"
PARENT_PACKAGE=previous.PARENT_PACKAGE
PARENT_SCOPE=previous.PARENT_SCOPE
PARENT_RECEIPT=previous.PARENT_RECEIPT
PARENT_SHA=previous.PARENT_SHA
PROFILE_SHA=hashlib.sha256((operation.ROOT/operation.CONFIG).read_bytes()).hexdigest()
FILES=tuple(dict.fromkeys((*previous.FILES,ENTRY,WRAPPER,operation.CONFIG,
    "scripts/prepare_dh116_tracking_pinch.py",
    "scripts/prepare_dh116_tracking_pinch_native.py",
    "scripts/solve_dh116_pinch_tracking_targets.py",
    "scripts/audit_dh116_pinch_tracking_target_path.py",
    "src/go2_pvcnn/control/dh116/tabletop_tracking_pinch_geometry.py",
    "tests/test_tracking_pinch_plan.py","tests/test_tracking_pinch_native.py",
    "tests/test_tracking_pinch_owner.py","tests/test_tracking_pinch_worker.py",
    "scripts/dh116_deferred_physical_audit.py",
    "src/go2_pvcnn/control/dh116/tabletop_owned_clock_projection.py",
    "tests/test_deferred_physical_audit.py","tests/test_owned_clock_projection.py")))
sha=previous.sha
read=previous.read
load_parent=previous.load_parent

@contextmanager
def entry_scope():
    with bindings((previous,"ENTRY",ENTRY),(previous,"WRAPPER",WRAPPER),
        (previous,"VERSION",VERSION),(previous,"WORKER_ENV",WORKER_ENV),
        (previous,"PROFILE_SHA",PROFILE_SHA),(previous,"FILES",FILES),
        (previous,"operation",operation)):
        yield

def preflight(project,parent_path=PARENT_RECEIPT,parent_sha=PARENT_SHA):
    with entry_scope():return previous.preflight(project,parent_path,parent_sha)

def execute(args,project,contract):
    with entry_scope():return previous.execute(args,project,contract)

def require_worker(args,project):
    with entry_scope():return previous.require_worker(args,project)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cleanup",choices=("explicit",),required=True);p.add_argument("--output",type=Path)
    p.add_argument("--headless",action="store_true");p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,default=234);p.add_argument("--execute-native",action="store_true")
    p.add_argument("--approved-code-config-sha256");p.add_argument("--left-scheduler-receipt",type=Path)
    p.add_argument("--left-receipt-sha256")
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    if a.steps!=234:p.error("exact 2 setup + 48 quiet + 160 closure + 24 hold steps required")
    if a.execute_native:
        if a.output is None or not a.headless or a.device!="cuda:0" or a.left_scheduler_receipt is None:
            p.error("fresh output, headless, cuda:0 and actual parent receipt required")
        require_worker(a,project);previous.previous.prior.original.prior.assert_clean_import_boundary()
    c=preflight(project,a.left_scheduler_receipt or PARENT_RECEIPT,a.left_receipt_sha256 or PARENT_SHA)
    if not a.execute_native:
        print(json.dumps(c,indent=2));return
    if a.approved_code_config_sha256!=c["code_config_sha256"]:p.error("worker source fingerprint changed")
    a.output.mkdir(parents=True,exist_ok=False);execute(a,project,c)


if __name__=="__main__":main()
