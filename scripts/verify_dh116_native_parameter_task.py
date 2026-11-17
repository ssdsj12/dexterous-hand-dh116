#!/usr/bin/env python3
"""Exact 18-step director endpoint acquisition derived from qualified closure."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from scripts import verify_dh116_dual_pinch_closure_interval as prior
from scripts import prepare_dh116_native_parameter_task as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ENTRY="scripts/verify_dh116_native_parameter_task.py"
WRAPPER="scripts/run_dh116_native_parameter_task_work_packages.py"
VERSION=operation.VERSION
WORKER_ENV="DH116_NATIVE_PARAMETER_TASK_WORKER_V1"
PARENT_PACKAGE="flex.f7-dual-pinch-parameter-readback-native"
PARENT_SCOPE="exact_2_setup_plus_16_parameter_provenance_endpoints_only"
PARENT_RECEIPT=Path('/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-dual-pinch-parameter-readback-native/7d7887f3739f97a593ea52fed72c5d052aabcf31c1f66fcfd246862986525f4c/attempt-02ajlaw4/receipt.json')
PARENT_SHA="a75a14a700b1286a266b83158e52384877a20b9b729b8fcbdfcc4d69c25198dc"
PARENT_AUDIT_SHA="1535ff14a1b1e19c7d065adcb5eda885f5e7b33309444e6cc764102ae114ec9e"
PROFILE_SHA=prior.PROFILE_SHA
FILES=(ENTRY,WRAPPER,"scripts/prepare_dh116_native_parameter_task.py",
    "scripts/validate_dh116_native_parameter_task.py",
    "src/go2_pvcnn/control/dh116/tabletop_native_parameter_endpoints.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformx_parameter_snapshot.py",
    "scripts/prepare_dh116_deformx_parameter_getters.py",
    "docs/contracts/deformx_parameter_getter_source_lock_v1.json",
    "tests/test_dh116_native_parameter_task.py","tests/test_dh116_native_parameter_task_worker.py")
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
    audit_path=path.with_name("dual-pinch-parameter-post-exit-audit.json")
    audit=prior._checked(audit_path,PARENT_AUDIT_SHA)["evidence"]
    if (audit["lifecycle_qualified"] is not True or audit["physics_steps"]!=18
            or audit["measurement_steps"]!=16 or audit["teacher_admission"] is not False
            or audit["profile_sha256"]!=PROFILE_SHA):
        raise ValueError("parent measured closure/profile scope differs")
    inputs={str(path):expected,str(audit_path):PARENT_AUDIT_SHA}
    for name in ("physical.jsonl","result.json","sdk_clock.jsonl","director_reset.json"):
        artifact=Path(value["output"])/name
        if artifact.resolve(strict=True)!=artifact or sha(artifact)!=value["artifact_sha256"][name]:
            raise ValueError("parent actual artifact bytes differ: "+name)
        inputs[str(artifact)]=value["artifact_sha256"][name]
    return value,inputs


def preflight(project,parent_path=PARENT_RECEIPT,parent_sha=PARENT_SHA):
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as reader
    parent,inputs=load_parent(parent_path,parent_sha)
    contract=prior.preflight(project)
    basis=reader.source_basis()
    sources=deepcopy(contract["sources"])
    for name in FILES:sources["local_sources"][name]=sha(project/name)
    sources["external_sources"].update(inputs)
    external=dict(basis["sources"])
    local_config_hash=external.pop(str(reader.ROOT/reader.CONFIG))
    if sha(project/reader.CONFIG)!=local_config_hash:
        raise ValueError("runtime and target checkout config bytes differ")
    sources["local_sources"][reader.CONFIG]=local_config_hash
    sources["external_sources"].update(external)
    code=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return dict(contract,version=VERSION,sources=sources,code_config_sha256=code,
        native_parent_receipt_sha256=PARENT_SHA,physics_steps=18,parameter_source_basis=basis,
        parameter_endpoint_count=17,material_slip_qualified=False)


def execute(args,project,contract):
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as reader
    with bindings((builder,"BACKEND",reader.BACKEND),(builder,"BACKEND_SHA",reader.BACKEND_SHA),
                  (prior,"closure",operation)):
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
