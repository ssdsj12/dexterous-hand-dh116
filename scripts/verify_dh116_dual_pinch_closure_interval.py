#!/usr/bin/env python3
"""One scoped native finger-only Pinch readback using the archived safe profile."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts import prepare_dh116_dual_pinch_closure_interval as closure
from scripts import verify_dh116_dual_pinch_interval as original
from scripts import verify_dh116_dual_pinch_interval_v4 as previous
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

ENTRY = "scripts/verify_dh116_dual_pinch_closure_interval.py"
WRAPPER = "scripts/run_dh116_dual_pinch_closure_work_packages.py"
VERSION = closure.VERSION
WORKER_ENV = "DH116_DUAL_PINCH_CLOSURE_WORKER_V1"
PARENT_PACKAGE = "flex.f7-dual-pinch-zero-interval-v4-native"
PARENT_SCOPE = "dual_pinch_same_free_rod_reset_zero_interval_v4"
PARENT_RECEIPT = Path(
    "/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f7-dual-pinch-zero-interval-v4-native/"
    "9e95d33e8b0565133c05d8fae5eb5c5eb68de269806ae561e4c5bd022bd9e202/"
    "attempt-fz50kogx/receipt.json"
)
PARENT_SHA = "aa1a6877c29b460e8f12589674d4ba09381faf8aea200400a7cd0e71d1a5a1ec"
PARENT_AUDIT_SHA = "c59e629c4dba249215a810a790a79719a56172d8312fc86c783d8ad1f99ac837"
PROFILE = Path(
    "/home/xk/.local/share/dh116-autonomous-3c80166/evidence/"
    "dual-pinch-readback-profile-20261113-dxcb62it/run02/profile.json"
)
PROFILE_SHA = "67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498"
RESULT_SHA = "75ec5cee02eca0676d15db8eb07d00273c2b50b40b7dc8d7d16d09d18e17f803"
LEDGER_SHA = "ead0dbf2c15b83864a1c440848ed9cf159b86c88a6ade0e4275208efebcf778c"
FILES = (
    ENTRY, WRAPPER, "scripts/prepare_dh116_dual_pinch_closure_interval.py",
    "scripts/dh116_dual_pinch_lifetime_evidence.py",
    "scripts/prepare_dh116_dual_pinch_readback_profile.py",
    "scripts/prepare_dh116_pinch_closure_margin.py",
    "scripts/solve_dh116_tripod_sdf_constrained_direction.py",
    "scripts/audit_dh116_power_closure_constraints.py",
    "scripts/screen_dh116_right_pinch_closure.py",
    "src/go2_pvcnn/control/dh116/tabletop_measured_closure_geometry.py",
    "tests/test_dh116_measured_closure_geometry.py",
    "tests/test_dh116_dual_pinch_readback_profile.py",
    "scripts/validate_dh116_dual_pinch_closure_interval.py",
    "configs/dh116_execution/deformx_dual_pinch_zero_interval_v4.json",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_clearance_guard.py",
    "tests/test_dh116_dual_pinch_closure_interval.py",
    "tests/test_dh116_dual_pinch_closure_postexit.py",
    "tests/test_dh116_dual_pinch_closure_worker.py",
)
read = original.read
sha = original.prior.sha


def _checked(path, expected):
    if path.resolve(strict=True) != path or sha(path) != expected:
        raise ValueError(f"frozen native or offline evidence differs: {path}")
    return read(path)


def load_profile(parent_path=PARENT_RECEIPT, parent_sha=PARENT_SHA):
    parent_path = Path(parent_path)
    if parent_path != PARENT_RECEIPT or parent_sha != PARENT_SHA:
        raise ValueError("exact published dual Pinch parent receipt required")
    parent = _checked(parent_path, parent_sha)
    if (parent["native_qualification"] is not True or parent["outer_exit_code"] != 0
            or parent["qualification_scope"] != PARENT_SCOPE):
        raise ValueError("exact native parent scope not qualified")
    audited = _checked(parent_path.with_name("dual-pinch-interval-post-exit-audit.json"),
                       PARENT_AUDIT_SHA)["evidence"]
    if (audited["lifecycle_qualified"] is not True or audited["physics_steps"] != 10
            or audited["teacher_admission"] is not False):
        raise ValueError("parent scope must be exact measured zero-action execution")
    profile = _checked(PROFILE, PROFILE_SHA)
    _checked(PROFILE.with_name("result.json"), RESULT_SHA)
    pair = original.builder.load_pair_proposal()
    ledger = Path(parent["output"]) / "physical.jsonl"
    if ledger.resolve(strict=True) != ledger or sha(ledger) != LEDGER_SHA:
        raise ValueError("native parent ledger bytes differ")
    phases = profile["phases"]
    if (profile["native_parent_receipt_path"] != str(parent_path)
            or profile["native_parent_receipt_sha256"] != parent_sha
            or profile["native_parent_ledger_path"] != str(ledger)
            or profile["native_parent_ledger_sha256"] != LEDGER_SHA
            or profile["paired_reset_path"] != str(original.builder.PAIR)
            or profile["paired_reset_sha256"] != original.builder.PAIR_SHA
            or profile["reset_seed_sha256"] != pair["reset_seed"]["sha256"]
            or profile["offline_result_sha256"] != RESULT_SHA
            or profile["setup_sdk_steps"] != closure.SETUP_STEPS
            or profile["measurement_sdk_steps"] != closure.MEASURED_STEPS
            or profile["total_sdk_steps"] != closure.TOTAL_STEPS
            or profile["control_intervals"] != 2
            or [p["name"] for p in phases] != ["repeat_original_zero_interval", "one_nonzero_readback_interval"]
            or [p["sdk_steps"] for p in phases] != [8, 8]
            or [p["action24"] for p in phases] != closure.action_schedule().tolist()
            or profile["no_commanded_lift"] is not True or profile["no_commanded_rotation"] is not True
            or any(profile[key] is not False for key in (
                "native_recipe_created", "native_qualification", "teacher_admission", "grasp_qualified"))):
        raise ValueError("frozen no-lift readback profile contract differs")
    return profile


def preflight(project, parent_path=PARENT_RECEIPT, parent_sha=PARENT_SHA):
    profile = load_profile(parent_path, parent_sha)
    contract = previous.preflight(project)
    sources = deepcopy(contract["sources"])
    for name in FILES:
        sources["local_sources"][name] = sha(project / name)
    sources["external_sources"].update({
        str(PROFILE): PROFILE_SHA, str(PROFILE.with_name("result.json")): RESULT_SHA,
        str(PARENT_RECEIPT): PARENT_SHA, profile["native_parent_ledger_path"]: LEDGER_SHA,
        str(PARENT_RECEIPT.with_name("dual-pinch-interval-post-exit-audit.json")): PARENT_AUDIT_SHA,
    })
    code = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(contract, version=VERSION, sources=sources, code_config_sha256=code,
                profile_sha256=PROFILE_SHA, native_parent_receipt_sha256=PARENT_SHA,
                physics_steps=closure.TOTAL_STEPS)


def diagnostic_reset_owners():
    from functools import partial
    scopes = {
        "controller": "fresh explicit fixed zero/finger-only two-interval diagnostic; no learned policy or MPC owner",
        "filter": "no task filter instantiated",
        "contact": "fresh diagnostic contact source bound after the single physical reset",
        "camera_history": "no camera or history owner instantiated",
    }
    def absent(name, context):
        return dict(owner=name, status="not_present", episode_uid=context["episode_uid"], source=scopes[name])
    return {name: partial(absent, name) for name in scopes}


def execute(args, project, contract):
    with bindings((original.interval, "run_existing_app", closure.run_existing_app),
                  (closure, "PROFILE_SHA", PROFILE_SHA),
                  (original, "diagnostic_reset_owners", diagnostic_reset_owners)):
        return original.execute(args, project, contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup", choices=("explicit",), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=closure.TOTAL_STEPS)
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--approved-code-config-sha256")
    parser.add_argument("--left-scheduler-receipt", type=Path)
    parser.add_argument("--left-receipt-sha256")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.steps != closure.TOTAL_STEPS:
        parser.error("exact 2 setup + 8 zero + 8 finger-only steps required")
    if args.execute_native:
        if (args.output is None or not args.headless or args.device != "cuda:0"
                or args.left_scheduler_receipt is None or args.left_receipt_sha256 is None):
            parser.error("fresh output/headless/cuda:0 and scheduler parent binding required")
        with bindings((original, "ENTRY", ENTRY), (original, "WRAPPER", WRAPPER),
                      (original, "WORKER_ENV", WORKER_ENV), (original, "VERSION", VERSION)):
            original.require_worker(args, project)
        original.prior.assert_clean_import_boundary()
    contract = preflight(project, args.left_scheduler_receipt or PARENT_RECEIPT,
                         args.left_receipt_sha256 or PARENT_SHA)
    if not args.execute_native:
        print(json.dumps(contract, indent=2))
        return
    if args.approved_code_config_sha256 != contract["code_config_sha256"]:
        parser.error("frozen worker code/config SHA differs")
    args.output.mkdir(parents=True, exist_ok=False)
    execute(args, project, contract)


if __name__ == "__main__":
    main()
