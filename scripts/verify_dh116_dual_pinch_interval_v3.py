#!/usr/bin/env python3
"""Dual Pinch successor for the original launcher-owned scene preparation handoff."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts import verify_dh116_dual_pinch_interval as prior
from scripts import verify_dh116_dual_pinch_interval_v2 as previous
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ENTRY = "scripts/verify_dh116_dual_pinch_interval_v3.py"
WRAPPER = "scripts/run_dh116_dual_pinch_work_packages_v3.py"
VERSION = prior.VERSION
WORKER_ENV = "DH116_DUAL_PINCH_INTERVAL_WORKER_V3"
FAILED_RECEIPT = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f7-dual-pinch-zero-interval-v2-native/e658c044aca8513ab2bef1c9dc198a06bab1cbc5a487de4816e23cff25ee715e/"
    "attempt-uh5g5d5z/receipt.json")
FAILED_SHA = "4ca563a091115cf056b32c921e67081ff9ff582b6df3f0f66abc40d61986f555"
OUTPUTS = {
    "result.json": "56a8e164afb302d9a90a7cb50f7f5633e3e03846260918c5d22591eea468306f",
    "physical.jsonl": "8ee67bece621c9ad9be6d0fd856a78e160c474882c32a92f1e79de93e7dfc023",
    "app_close_returned.json": "37b1ba15219bb2264687d2883ed6129cc5cd54fe64d0d7afbb1e5ca7d9308ae4",
}
FAILED_ARTIFACTS = {
    "outer_exit.json": "9235b4ce58d78b610a2ffeab9a463b11550aad748bb66d1bebacda233d9814af",
    "native.log": "3c3ed4779e6d2eab1f7bd7a10898eb3ad922f80452721e60207a0de8cba2ab3d",
}
FILES = (ENTRY, WRAPPER, "tests/test_dh116_dual_pinch_worker_v3.py",
    "configs/dh116_execution/deformx_dual_pinch_zero_interval_v2.json",
    "src/go2_pvcnn/control/dh116/tabletop_dual_hand_scene_preparation_v2.py",
    "tests/test_dh116_external_wire_layout.py")
prior_preflight = previous.preflight


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
            or result["physics_steps"] != 2 or "TypeError: 'NoneType' object is not subscriptable" not in result["failure"]):
        raise ValueError("exact original post-reset preparation handoff failure required")
    code = hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return dict(original,sources=sources,code_config_sha256=code,v2_failure_receipt_sha256=FAILED_SHA,
                correction_scope="one_owned_prepared_scene_shared_by_builder_and_contact_reader")


def run_existing_app(*args, **kwargs):
    """Resolve one scene packet after App creation inside the existing lifecycle."""
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation_v2 import external_wire_layout
    try:
        if kwargs["prepared"] is None:
            kwargs["prepared"] = scene.prepare_scene(kwargs["project"])
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
