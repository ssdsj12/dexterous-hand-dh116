#!/usr/bin/env python3
"""Offline 371D source inventory; never starts Isaac or a Teacher model."""
import argparse
import hashlib
import json
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_teacher_preaction_readback import (
    CONTRACT_SHA256, COUPLING_SHA256, audit_legacy_pinch,
)

CONTRACT = "docs/contracts/stereo_privileged_state_contract.json"
COUPLING = "src/go2_pvcnn/control/dh116/linear_coupling.py"
ADAPTER = "src/go2_pvcnn/control/dh116/tabletop_teacher_preaction_readback.py"


def _read(path, expected):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return raw, dict(path=str(Path(path).resolve()), sha256=sha, bytes=len(raw))


def _reason(block):
    if block["id"] in ("B00", "B01", "B15"):
        return "left_raw_q_qd_exist_but_actual_SDK_joint_order_missing;right_hand_missing"
    if block["id"] in ("B04", "B27", "B31", "B32", "B33") or 45 <= int(block["id"][1:]) <= 56:
        return "aggregate_node_forces_and_mesh_wrenches_do_not_supply_per_contact_sources"
    if 58 <= int(block["id"][1:]):
        return "no_complete_previous_policy_interval_feedback_or_measured_wrist_endpoints"
    if block["id"] in ("B10", "B11", "B12", "B19", "B42", "B43", "B44", "B57"):
        return "post_step_rod_nodes_do_not_define_original_rigid_object_common_semantics"
    return "required_same_clock_controller_reset_or_bilateral_measured_source_not_recorded"


def audit_files(ledger, ledger_sha256, plan, plan_sha256, output, project):
    project = Path(project)
    raw, ledger_binding = _read(ledger, ledger_sha256)
    plan_raw, plan_binding = _read(plan, plan_sha256)
    contract_raw, contract_binding = _read(project / CONTRACT, CONTRACT_SHA256)
    _, coupling_binding = _read(project / COUPLING, COUPLING_SHA256)
    manifest = json.loads(plan_raw)
    sources = manifest.get("sources", {}).get("local_sources", {})
    if sources.get(CONTRACT) != CONTRACT_SHA256 or sources.get(COUPLING) != COUPLING_SHA256:
        raise ValueError("historical plan does not bind the exact observation/coupling contract")
    if manifest.get("state371_policy_binding") is not False:
        raise ValueError("only historical explicitly unbound common state is inventoried")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    result = audit_legacy_pinch(rows)
    contract = json.loads(contract_raw)
    result["blocks"] = [dict(id=b["id"], name=b["name"], slice=b["slice"],
                            normalization=b["normalization"], valid_channels=0,
                            missing_reason=_reason(b)) for b in contract["blocks"]]
    indices = [i for b in result["blocks"] for i in range(*b["slice"])]
    if indices != list(range(371)):
        raise ValueError("contract block indices do not cover the exact 371D order")
    result["inputs"] = dict(ledger=ledger_binding, plan=plan_binding,
                           state_contract=contract_binding, coupling=coupling_binding)
    result["implementation_sha256"] = {
        name: hashlib.sha256((project / name).read_bytes()).hexdigest()
        for name in (ADAPTER, "scripts/audit_dh116_teacher_preaction_sources.py")}
    result["scope"] = "read_only_source_inventory;not_native_execution_or_qualification"
    result["historical_native_execution_claim_in_plan"] = manifest.get("native_execution_performed")
    result["historical_producer_sha256_declared_in_plan"] = sources.get(
        "scripts/verify_dh116_deformx_pinch_contact.py")
    result["pre_action_clock_status"] = "derived_from_source_order;not_recorded_complete_shared_identity"
    result["next_capture_requirements"] = [
        "capture actual SDK joint_names and angular DOF types with owned q/qd before target write",
        "same live env/episode/reset/rod/policy/physics identity before and after each hand copy",
        "both hands plus validated controller/reset/reference and complete prior interval sources",
        "resolve flexible rod vs rigid object semantics without changing 371D normalization silently",
        "per-contact partner/point/normal/force provenance;aggregate force cannot substitute",
    ]
    output = Path(output)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ledger", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("ledger-sha256", "plan-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    result = audit_files(args.ledger, args.ledger_sha256, args.plan, args.plan_sha256,
                         args.output, Path(__file__).resolve().parents[1])
    print(json.dumps({k: result[k] for k in ("rows", "admitted_371d_channels",
                     "qualified_for_teacher_ppo", "new_physics_steps", "optimizer_updates")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
