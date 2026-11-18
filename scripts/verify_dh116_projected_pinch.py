#!/usr/bin/env python3
"""Original projection, independently identified once-only bounded diagnostic."""
from contextlib import contextmanager
from pathlib import Path
import argparse,json
from scripts import verify_dh116_left_pinch_approach as previous
from scripts import prepare_dh116_projected_pinch_native as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
ENTRY="scripts/verify_dh116_projected_pinch.py"
WRAPPER="scripts/run_dh116_projected_pinch_work_packages.py"
VERSION=operation.VERSION
WORKER_ENV="DH116_PROJECTED_PINCH_WORKER_V1"
PROFILE_SHA=previous.PROFILE_SHA
PARENT_PACKAGE,PARENT_SCOPE=previous.PARENT_PACKAGE,previous.PARENT_SCOPE
PARENT_RECEIPT,PARENT_SHA=previous.PARENT_RECEIPT,previous.PARENT_SHA
FILES=tuple(dict.fromkeys((*previous.FILES,ENTRY,WRAPPER,
    "scripts/prepare_dh116_projected_pinch_native.py",
    "src/go2_pvcnn/control/dh116/tabletop_wrist_projection_evidence.py",
    "src/go2_pvcnn/control/dh116/tabletop_projected_pinch_geometry.py",
    "tests/test_projected_pinch_geometry.py","tests/test_projected_pinch_native.py",
    "tests/test_projected_pinch_owner.py","tests/test_projected_pinch_worker.py")))
sha,read,load_parent=previous.sha,previous.read,previous.load_parent
@contextmanager
def entry_scope():
    with bindings((previous,"ENTRY",ENTRY),(previous,"WRAPPER",WRAPPER),(previous,"VERSION",VERSION),
        (previous,"WORKER_ENV",WORKER_ENV),(previous,"FILES",FILES),(previous,"operation",operation)):
        yield
def preflight(project,parent_path=PARENT_RECEIPT,parent_sha=PARENT_SHA):
    with entry_scope():return previous.preflight(project,parent_path,parent_sha)
def execute(args,project,contract):
    with entry_scope():return previous.execute(args,project,contract)
def require_worker(args,project):
    with entry_scope():return previous.require_worker(args,project)
assert_clean_import_boundary=previous.previous.previous.prior.original.prior.assert_clean_import_boundary

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
        require_worker(a,project);assert_clean_import_boundary()
    c=preflight(project,a.left_scheduler_receipt or PARENT_RECEIPT,a.left_receipt_sha256 or PARENT_SHA)
    if not a.execute_native:
        print(json.dumps(c,indent=2));return
    if a.approved_code_config_sha256!=c["code_config_sha256"]:p.error("worker source fingerprint changed")
    a.output.mkdir(parents=True,exist_ok=False);execute(a,project,c)



if __name__=="__main__":main()
