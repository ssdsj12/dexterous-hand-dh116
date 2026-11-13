#!/usr/bin/env python3
"""Independent right-owned approach, using the qualified parameter owner."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from scripts import verify_dh116_native_parameter_task as previous
from scripts import prepare_dh116_bilateral_owned_native_interval as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ENTRY="scripts/verify_dh116_bilateral_owned_pinch.py"
WRAPPER="scripts/run_dh116_bilateral_owned_work_packages.py"
VERSION=operation.VERSION
WORKER_ENV="DH116_BILATERAL_OWNED_WORKER_V1"
PARENT_PACKAGE="flex.f8-right-pinch-owned-contact-native"
PARENT_SCOPE="right_owned_pinch_2_setup_232_guarded_contact_diagnostic_only"
PARENT_RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-right-pinch-owned-contact-native/2fcb2478d3c04d8e52d47638a10ec0bbecc4f34247d308a058b662d45d38d873/attempt-z3xlac81/receipt.json")
PARENT_SHA="3e12bb3204f46df748e1de7b35b77306e57be016e6a7f626249f0790d466c83d"
LEFT_AUDIT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-pinch-segment-contact-20261113-41heHL/postexit-v2-audit.json")
LEFT_SHA="57cbc1b6b97eb594368e4702746241c3be3a3b0c562d99293ab2b470b30706cf"
PROFILE_SHA=hashlib.sha256((operation.ROOT/operation.CONFIG).read_bytes()).hexdigest()
from scripts import verify_dh116_right_pinch_owned_contact as margin_entry
FILES=tuple(dict.fromkeys((*margin_entry.FILES,ENTRY,WRAPPER,
    "scripts/audit_dh116_bilateral_owned_action_contract.py",
    "scripts/audit_dh116_pinch_segment_postexit_v2.py",
     "scripts/prepare_dh116_bilateral_owned_pinch.py",
    "scripts/prepare_dh116_bilateral_owned_native_interval.py",
    "scripts/validate_dh116_bilateral_owned_contact.py",
    "scripts/solve_dh116_right_pinch_surface_registration.py",
    "scripts/audit_dh116_right_pinch_measured_direction.py",
    "scripts/audit_dh116_right_approach_action_contract.py",
    "src/go2_pvcnn/control/dh116/tabletop_bilateral_owned_geometry.py",
    operation.CONFIG,
    "src/go2_pvcnn/control/dh116/tabletop_segment_closure_geometry.py",
    "src/go2_pvcnn/control/dh116/tabletop_polyline_support_certificate.py",
    "tests/test_dh116_pinch_segment_geometry.py",
    "tests/test_dh116_bilateral_owned_worker.py",
    "tests/test_dh116_bilateral_owned_owner.py")))
sha=previous.sha
read=previous.read

def load_parent(path=PARENT_RECEIPT,expected=PARENT_SHA):
    if Path(path)!=PARENT_RECEIPT or expected!=PARENT_SHA:
        raise ValueError("exact published parameter-task parent required")
    receipt=operation.read_locked(PARENT_RECEIPT,PARENT_SHA)
    if (receipt["native_qualification"] is not True or receipt["outer_exit_code"]!=0
        or receipt["qualification_scope"]!=PARENT_SCOPE):
        raise ValueError("qualified real parent scope required")
    inputs={str(PARENT_RECEIPT):PARENT_SHA}
    for name in ("physical.jsonl","result.json","sdk_clock.jsonl","director_reset.json"):
        p=Path(receipt["output"])/name;expected=receipt["artifact_sha256"][name]
        if p.resolve(strict=True)!=p or sha(p)!=expected:raise ValueError("parent bytes differ: "+name)
        inputs[str(p)]=expected
    left=operation.read_locked(LEFT_AUDIT,LEFT_SHA)
    from scripts import audit_dh116_pinch_segment_postexit_v2 as historical
    original=historical.load_locked()
    if (left["lifecycle_qualified"] is not True or left["qualification_scope"]!=historical.SCOPE
        or left["original_scheduler_receipt_sha256"]!=historical.RECEIPT_SHA
        or left["original_scheduler_qualification"] is not False
        or left["original_receipt_unchanged"] is not True or left["new_native_steps"]!=0
        or original["native_qualification"] is not False):
        raise ValueError("immutable independent LEFT re-audit required; never rewrite failed receipt")
    inputs[str(LEFT_AUDIT)]=LEFT_SHA
    inputs[str(historical.RECEIPT)]=historical.RECEIPT_SHA
    for name,expected in original["artifact_sha256"].items():
        inputs[str(historical.NATIVE/name)]=expected
    return receipt,inputs

def preflight(project,parent_path=PARENT_RECEIPT,parent_sha=PARENT_SHA):
    project=Path(project)
    parent,inputs=load_parent(parent_path,parent_sha)
    # Existing source/lifecycle checks remain evaluated in their ORIGINAL scope.
    contract=previous.preflight(project)
    pair=operation.load_pair(project)
    sources=deepcopy(contract["sources"])
    for name in FILES:sources["local_sources"][name]=sha(project/name)
    sources["external_sources"].update(inputs)
    sources["external_sources"].update(pair["sources"])
    code=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return dict(contract,version=VERSION,sources=sources,code_config_sha256=code,
        native_parent_receipt_sha256=PARENT_SHA,profile_sha256=PROFILE_SHA,
        physics_steps=234,parameter_endpoint_count=233,reset_seed_sha256=pair["reset_seed"]["sha256"],
        teacher_admission=False,grasp_qualified=False,native_execution_performed=False)

def execute(args,project,contract):
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as reader
    prior=previous.prior
    with operation.scenario_scope(project),bindings(
            (builder,"BACKEND",reader.BACKEND),(builder,"BACKEND_SHA",reader.BACKEND_SHA),
            (prior,"closure",operation),(prior,"PROFILE_SHA",PROFILE_SHA),
            (operation,"PROFILE_SHA",PROFILE_SHA)):
        return prior.execute(args,project,contract)

def require_worker(args,project):
    with bindings((previous.prior.original,"ENTRY",ENTRY),(previous.prior.original,"WRAPPER",WRAPPER),
            (previous.prior.original,"WORKER_ENV",WORKER_ENV),(previous.prior.original,"VERSION",VERSION)):
        return previous.prior.original.require_worker(args,project)

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
        require_worker(a,project);previous.prior.original.prior.assert_clean_import_boundary()
    c=preflight(project,a.left_scheduler_receipt or PARENT_RECEIPT,a.left_receipt_sha256 or PARENT_SHA)
    if not a.execute_native:
        print(json.dumps(c,indent=2));return
    if a.approved_code_config_sha256!=c["code_config_sha256"]:p.error("worker source fingerprint changed")
    a.output.mkdir(parents=True,exist_ok=False);execute(a,project,c)

if __name__=="__main__":main()
