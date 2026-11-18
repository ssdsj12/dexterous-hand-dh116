#!/usr/bin/env python3
"""Independent left Tripod approach, using the qualified parameter owner."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from scripts import verify_dh116_native_parameter_task as previous
from scripts import prepare_dh116_tripod_native_interval as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ENTRY="scripts/verify_dh116_tripod_owned_contact.py"
WRAPPER="scripts/run_dh116_tripod_owned_work_packages.py"
VERSION=operation.VERSION
WORKER_ENV="DH116_TRIPOD_OWNED_WORKER_V1"
PARENT_PACKAGE="flex.f8-bilateral-owned-pinch-native"
PARENT_SCOPE="bilateral_owned_pinch_2_setup_232_guarded_contact_diagnostic_only"
PARENT_RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-bilateral-owned-pinch-native/cc57fbb782979afea2148c066aef33020926ed8385ff229f064f4bb7e1441d2e/attempt-w0i1tg3o/receipt.json")
PARENT_SHA="8ff5b25afb5a46a1cb57899b5aedf42c1e0e766a5bf30d4628e229bd62850db8"
LEFT_AUDIT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-pinch-segment-contact-20261113-41heHL/postexit-v2-audit.json")
LEFT_SHA="57cbc1b6b97eb594368e4702746241c3be3a3b0c562d99293ab2b470b30706cf"
PROFILE_SHA=hashlib.sha256((operation.ROOT/operation.CONFIG).read_bytes()).hexdigest()
from scripts import verify_dh116_bilateral_owned_pinch as prior_entry
FILES=tuple(dict.fromkeys((*prior_entry.FILES,ENTRY,WRAPPER,operation.CONFIG,
'scripts/prepare_dh116_tripod_owned_approach.py','scripts/prepare_dh116_tripod_native_interval.py','scripts/validate_dh116_tripod_owned_contact.py','scripts/dh116_tripod_contact_audit.py','scripts/audit_dh116_tripod_quiet_replay.py','scripts/audit_dh116_rod_quiet_window.py','src/go2_pvcnn/control/dh116/tabletop_tripod_quiet_gate.py','src/go2_pvcnn/control/dh116/tabletop_tripod_owned_geometry.py','tests/test_tripod_live_plan.py','tests/test_tripod_quiet_gate.py','tests/test_tripod_projection_scope.py','tests/test_tripod_exchange_admission.py','tests/test_tripod_native_scope.py','tests/test_tripod_full_owner.py','tests/test_tripod_worker.py')))
FILES=tuple(dict.fromkeys((*FILES,'scripts/audit_dh116_tripod_sdk_path.py','scripts/audit_dh116_tripod_pose_path.py','scripts/audit_dh116_tripod_fixed_root_path.py','scripts/audit_dh116_power_closure_constraints.py','scripts/solve_dh116_tripod_joint_root_direction.py','scripts/screen_dh116_right_pinch_closure.py','src/go2_pvcnn/control/dh116/tabletop_free_wrist_grasp.py','src/go2_pvcnn/control/dh116/tabletop_command_kernels.py')))
sha=previous.sha
read=previous.read

def load_parent(path=PARENT_RECEIPT,expected=PARENT_SHA):
    if Path(path)!=PARENT_RECEIPT or expected!=PARENT_SHA:
        raise ValueError("exact published bilateral-owned parent required")
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
    if a.steps!=234:p.error("exact 2 setup + 8 quiet + 200 approach + 24 observation steps required")
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
