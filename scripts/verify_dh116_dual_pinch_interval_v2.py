#!/usr/bin/env python3
"""Dual Pinch successor for the original spawn_scene wire metadata requirement."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts import verify_dh116_dual_pinch_interval as prior
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ENTRY = "scripts/verify_dh116_dual_pinch_interval_v2.py"
WRAPPER = "scripts/run_dh116_dual_pinch_work_packages_v2.py"
VERSION = prior.VERSION
WORKER_ENV = "DH116_DUAL_PINCH_INTERVAL_WORKER_V2"
FAILED_RECEIPT = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f7-dual-pinch-zero-interval-native/50a33500aaad8a825a78a9c6891070149ba38560a23c6a05b74ce2bb72309358/"
    "attempt-d3ypguyp/receipt.json")
FAILED_SHA = "5825d7b12b8661ea540b30647bedb330691793d3595c2fa1e4bfa53b81a0ed02"
OUTPUTS = {
    "result.json": "fc8b87478924a4c05e1e1c1f2329623514cfcacd300026d24e90d0018eaee37a",
    "physical.jsonl": "f92ee09ba232321c70a5ced5022106b0333e22aa82097d685604cf0a09254a5a",
    "app_close_returned.json": "37b1ba15219bb2264687d2883ed6129cc5cd54fe64d0d7afbb1e5ca7d9308ae4",
}
FAILED_ARTIFACTS = {
    "outer_exit.json": "a2107bc4d29d2afe01fae5ac3aefc522da8b6cd7627f86b3012f792b5603d1d1",
    "native.log": "b64b30cfcfcb415706ee1110baab1af4ec8a84c880e8f2d4d06d78c216801bdf",
}
FILES = (ENTRY, WRAPPER, "tests/test_dh116_dual_pinch_worker_v2.py",
    "configs/dh116_execution/deformx_dual_pinch_zero_interval_v1.json",
    "src/go2_pvcnn/control/dh116/tabletop_dual_hand_scene_preparation_v2.py",
    "tests/test_dh116_external_wire_layout.py")
prior_preflight = prior.preflight


def preflight(project):
    if prior.prior.sha(FAILED_RECEIPT) != FAILED_SHA:
        raise ValueError("original dual Pinch failure receipt differs")
    receipt = prior.read(FAILED_RECEIPT)
    original = prior_preflight(project)
    sources = deepcopy(original["sources"])
    for name in FILES:
        sources["local_sources"][name] = prior.prior.sha(project/name)
    evidence = {FAILED_RECEIPT: FAILED_SHA}
    evidence.update({FAILED_RECEIPT.with_name(name): sha for name,sha in FAILED_ARTIFACTS.items()})
    output = Path(receipt["output"])
    evidence.update({output/name: sha for name,sha in OUTPUTS.items()})
    for path, sha in evidence.items():
        if prior.prior.sha(path) != sha:
            raise ValueError("original native failure artifact differs")
        sources["external_sources"][str(path)] = sha
    result = prior.read(output/"result.json")
    outer = prior.read(FAILED_RECEIPT.with_name("outer_exit.json"))
    if (receipt["native_qualification"] is not False or outer["outer_exit_code"] != 1
            or outer["debugger_inferior_exit_code"] != 1 or outer["debugger_inferior_signal"] is not None
            or result["physics_steps"] != 0 or "KeyError: 'wire_center'" not in result["failure"]):
        raise ValueError("exact original pre-physics wire metadata failure required")
    code = hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return dict(original,sources=sources,code_config_sha256=code,v1_failure_receipt_sha256=FAILED_SHA,
                correction_scope="legacy_external_wire_layout_metadata_only")


def run_existing_app(*args, **kwargs):
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation_v2 import external_wire_layout
    try:
        with external_wire_layout():
            return original_interval(*args, **kwargs)
    finally:
        args = kwargs = None


original_interval = prior.interval.run_existing_app


def execute(args, project, contract):
    with bindings((prior.interval,"run_existing_app",run_existing_app)):
        return prior.execute(args,project,contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup",choices=("explicit",),required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--steps",type=int,default=10)
    parser.add_argument("--execute-native",action="store_true")
    parser.add_argument("--approved-code-config-sha256")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.steps != 10:
        parser.error("exact 2 setup + 8 measured steps required")
    if args.execute_native:
        if args.output is None or not args.headless or args.device != "cuda:0":
            parser.error("fresh output/headless/cuda:0 required")
        with bindings((prior,"ENTRY",ENTRY),(prior,"WRAPPER",WRAPPER),(prior,"WORKER_ENV",WORKER_ENV)):
            prior.require_worker(args,project)
        prior.prior.assert_clean_import_boundary()
    contract = preflight(project)
    if not args.execute_native:
        print(json.dumps(contract,indent=2))
        return
    if args.approved_code_config_sha256 != contract["code_config_sha256"]:
        parser.error("frozen successor worker code/config SHA differs")
    args.output.mkdir(parents=True,exist_ok=False)
    execute(args,project,contract)


if __name__ == "__main__":
    main()
