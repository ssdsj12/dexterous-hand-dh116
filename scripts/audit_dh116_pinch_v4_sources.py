#!/usr/bin/env python3
"""Reopen the frozen v4 evidence with separate contact-source provenance."""

import argparse
import json
from pathlib import Path

from scripts import validate_dh116_pinch_v4 as audit
from scripts.audit_dh116_pinch_v3_evidence import ledger
from scripts.dh116_pinch_evidence_v2 import validate_table_contacts


VERSION = "dh116-pinch-v4-observed-contact-source-inventory-v1"
RECEIPT_SHA = "5731ca089fa1b06afa0d41d27ac70dcbb9853c86a6e48982e068764a48c83e44"
AUDIT_SHA = "ee88bbdb22dc8fa2164f4ad2ce086400e99652d721f44bb32b40c09ae3663519"


def table_source(log, *, filter_path, counts):
    if not counts or any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("nonempty explicit nonnegative contact counts required")
    warning = f"GPU contact filter for collider '{filter_path}' is not supported"
    unsupported = sum("[omni.physx.tensors.plugin]" in line and warning in line
                      for line in log.splitlines())
    return dict(filter_path=filter_path, unsupported_filter_warnings=unsupported,
        reported_contacts=sum(counts), raw_record_count=len(counts),
        observed_contacts=None, absence_verified=False,
        source_status="unsupported_static_gpu_filter" if unsupported else "unqualified_filter_capability")


def audit_sources(project, receipt_path):
    receipt_path = Path(receipt_path)
    if audit.full.sha(receipt_path) != RECEIPT_SHA:
        raise ValueError("frozen v4 receipt differs")
    receipt = audit.full.read(receipt_path)
    output = Path(receipt["output"])
    if receipt["native_qualification"] is not True or receipt["outer_exit_code"] != 0:
        raise ValueError("frozen completed native scope differs")
    for name, expected in receipt["artifact_sha256"].items():
        if audit.full.sha(output/name) != expected:
            raise ValueError(f"frozen v4 artifact differs: {name}")
    prior = receipt_path.with_name("pinch-v4-post-exit-audit.json")
    if audit.full.sha(prior) != AUDIT_SHA or receipt["lifetime_post_exit_audit_sha256"] != AUDIT_SHA:
        raise ValueError("frozen v4 post-exit audit differs")
    cfg = audit.full.read(project/audit.CONFIG)
    outer_path = receipt_path.with_name("outer_exit.json")
    result = audit.audit_outcome(output, outer_path, "explicit", cfg=cfg)
    archived = audit.full.read(prior)["evidence"]
    if result != archived or result["contact_observation_qualified"] is not True:
        raise ValueError("independent v4 post-exit replay differs")
    log = receipt_path.with_name("native.log")
    outer = audit.full.read(outer_path)
    if audit.full.sha(log) != outer["log_sha256"] or outer["log_sha256"] != receipt["log_sha256"]:
        raise ValueError("frozen v4 native log differs")
    rows = ledger(output/"physics.jsonl")
    count = [validate_table_contacts(sample, row) for row,sample in
             zip(rows, ledger(output/"table_contacts.jsonl"), strict=True)]
    plan = audit.full.read(output/"plan.json")
    source = table_source(log.read_text(errors="replace"), filter_path=plan["contact_table_filter"],
                          counts=count)
    if len(count) != 238 or not source["unsupported_filter_warnings"]:
        raise ValueError("original 238 rows or exact GPU filter warning missing")
    owners = {}
    for phase in ("resources_acquired", "before_stop", "after_stop", "before_app_close", "after_app_close"):
        path = output/(phase+".simulation-owners.json")
        owners[phase] = audit.full.read(path)
    return dict(version=VERSION, receipt_sha256=RECEIPT_SHA, post_exit_audit_sha256=AUDIT_SHA,
        actual_outer_exit=outer, actual_physics_steps=result["physics_steps"],
        initial_clearance=result["initial_clearance_audit"], force_precision=result["force_precision_audit"],
        actual_single_hand_rod_contact=result["contact_evidence"], physx_hand_table_source=source,
        simulation_clear_api=result["simulation_clear_receipt_audit"], simulation_owner_observations=owners,
        material_slip=None, effective_opposed_contact=None, grasp_qualified=False, teacher_admission=False,
        new_native_steps=0, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_sources(Path(__file__).resolve().parents[1], args.receipt)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ("actual_physics_steps","physx_hand_table_source","teacher_admission")}))


if __name__ == "__main__":
    main()
