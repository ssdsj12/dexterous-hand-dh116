#!/usr/bin/env python3
"""Exact 18-step director endpoint acquisition derived from qualified closure."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from scripts import verify_dh116_dual_pinch_closure_interval as prior
from scripts import prepare_dh116_dual_pinch_director_interval as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ENTRY="scripts/verify_dh116_dual_pinch_director_interval.py"
WRAPPER="scripts/run_dh116_dual_pinch_director_work_packages.py"
VERSION=operation.VERSION
WORKER_ENV="DH116_DUAL_PINCH_DIRECTOR_WORKER_V1"
PARENT_PACKAGE="flex.f7-dual-pinch-guarded-closure-native"
PARENT_SCOPE="dual_pinch_same_free_rod_one_guarded_finger_action_interval"
PARENT_RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f7-dual-pinch-guarded-closure-native/"
    "74f8378ee4648f30c8ecb2ba8f0aa5f2caf9ef396a8e49b9c532cbf44ff7fad6/attempt-1ouifpxg/receipt.json")
PARENT_SHA="dd64cce38312cb3fe044e210c8560f44fae33b218e5b8e2ba97435a3969c9bc0"
PARENT_AUDIT_SHA="80abc7aadcb9f2b5ec03084df5c200497360bcee18e6e5d6ffa3a61e88f9d2c9"
PROFILE_SHA=prior.PROFILE_SHA
FILES=(ENTRY,WRAPPER,"scripts/prepare_dh116_dual_pinch_director_interval.py",
    "scripts/validate_dh116_dual_pinch_director_interval.py",
    "src/go2_pvcnn/control/dh116/tabletop_material_director_readback.py",
    "src/go2_pvcnn/control/dh116/tabletop_cosserat_material_marker_reference.py",
    "tests/test_dh116_material_director_readback.py",
    "tests/test_dh116_dual_pinch_director_interval.py",
    "tests/test_dh116_dual_pinch_director_postexit.py",
    "tests/test_dh116_dual_pinch_director_worker.py")
sha=prior.sha
read=prior.read


def load_parent(path=PARENT_RECEIPT,expected=PARENT_SHA):
    path=Path(path)
    if path!=PARENT_RECEIPT or expected!=PARENT_SHA:
        raise ValueError("exact published closure parent receipt required")
    value=prior._checked(path,expected)
    if (value["native_qualification"] is not True or value["outer_exit_code"]!=0
            or value["qualification_scope"]!=PARENT_SCOPE):
        raise ValueError("exact qualified closure parent scope required")
    audit_path=path.with_name("dual-pinch-closure-post-exit-audit.json")
    audit=prior._checked(audit_path,PARENT_AUDIT_SHA)["evidence"]
    if (audit["lifecycle_qualified"] is not True or audit["physics_steps"]!=18
            or audit["measurement_steps"]!=16 or audit["teacher_admission"] is not False
            or audit["profile_sha256"]!=PROFILE_SHA):
        raise ValueError("parent measured closure/profile scope differs")
    inputs={str(path):expected,str(audit_path):PARENT_AUDIT_SHA}
    for name in ("physical.jsonl","result.json","sdk_clock.jsonl"):
        artifact=Path(value["output"])/name
        if artifact.resolve(strict=True)!=artifact or sha(artifact)!=value["artifact_sha256"][name]:
            raise ValueError("parent actual artifact bytes differ: "+name)
        inputs[str(artifact)]=value["artifact_sha256"][name]
    return value,inputs


def preflight(project,parent_path=PARENT_RECEIPT,parent_sha=PARENT_SHA):
    from go2_pvcnn.control.dh116.tabletop_material_director_readback import source_basis
    parent,inputs=load_parent(parent_path,parent_sha)
    contract=prior.preflight(project)
    basis=source_basis()
    sources=deepcopy(contract["sources"])
    for name in FILES:sources["local_sources"][name]=sha(project/name)
    sources["external_sources"].update(inputs)
    sources["external_sources"].update(basis["additional_sources"])
    sources["external_sources"].update({v["path"]:v["sha256"] for v in basis["sources"].values()})
    code=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return dict(contract,version=VERSION,sources=sources,code_config_sha256=code,
        native_parent_receipt_sha256=PARENT_SHA,physics_steps=18,director_source_basis=basis,
        material_director_endpoint_count=17,material_slip_qualified=False)


def execute(args,project,contract):
    with bindings((prior,"closure",operation)):
        return prior.execute(args,project,contract)


def require_worker(args,project):
    with bindings((prior.original,"ENTRY",ENTRY),(prior.original,"WRAPPER",WRAPPER),
                  (prior.original,"WORKER_ENV",WORKER_ENV),(prior.original,"VERSION",VERSION)):
        return prior.original.require_worker(args,project)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup",choices=("explicit",),required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--steps",type=int,default=18)
    parser.add_argument("--execute-native",action="store_true")
    parser.add_argument("--approved-code-config-sha256")
    parser.add_argument("--left-scheduler-receipt",type=Path)
    parser.add_argument("--left-receipt-sha256")
    args=parser.parse_args();project=Path(__file__).resolve().parents[1]
    if args.steps!=18:parser.error("exact unchanged 2 setup + 8 zero + 8 finger-only steps required")
    if args.execute_native:
        if (args.output is None or not args.headless or args.device!="cuda:0"
                or args.left_scheduler_receipt is None or args.left_receipt_sha256 is None):
            parser.error("fresh output/headless/cuda:0 and exact closure parent required")
        require_worker(args,project)
        prior.original.prior.assert_clean_import_boundary()
    contract=preflight(project,args.left_scheduler_receipt or PARENT_RECEIPT,
                       args.left_receipt_sha256 or PARENT_SHA)
    if not args.execute_native:
        print(json.dumps(contract,indent=2));return
    if args.approved_code_config_sha256!=contract["code_config_sha256"]:
        parser.error("frozen director worker source/config SHA differs")
    args.output.mkdir(parents=True,exist_ok=False)
    execute(args,project,contract)


if __name__=="__main__":
    main()
