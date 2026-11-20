#!/usr/bin/env python3
"""Certify a left Pinch pause before the first unilateral rod load."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts import prepare_dh116_pinch_prelift as prelift
from scripts import prepare_dh116_pinch_wrist_first as wrist


BOUNDARIES = (18, 26)


def validate_boundary(sdk):
    if sdk not in BOUNDARIES:
        raise ValueError("an early precontact boundary at SDK18 or SDK26 is required")


def left_hold_windows(actual, issued):
    actual = prelift._finite(actual, (12,))
    issued = prelift._finite(issued, (12,))
    target = prelift.retained_target(actual, issued, "left")
    right = np.zeros(12)
    commands = prelift.bounded_transition(prelift.global_channel_order(issued, right),
                                          prelift.global_channel_order(target, right))
    return commands[:2], target


def certify_left_hold(row, actual, issued, *, proof=wrist.right_geometry):
    # Include the original target in every independently lagging group.
    # The rollback targets and all intermediate requests lie inside this box.
    envelope = dict(left_hold=(np.zeros(4), np.ones(4)), wrist_residuals=(np.zeros(3), np.zeros(3)))
    proofs, _ = proof(row, actual, issued, issued, envelope, nodes=row["rod"]["node_positions"][0],
                      phase_names=("left_hold",), phase_seconds=25., side="left", require_pad_clearance=True)
    return proofs["left_hold"]


def screen(archive, sdk):
    validate_boundary(sdk)
    prelift.archived_endpoint(archive)
    ledger = archive / "native" / "physical.jsonl"
    with ledger.open() as stream:
        for line in stream:
            entry = json.loads(line)
            if entry.get("operation") != "physical":
                continue
            row = prelift.base.decode_owned(entry["payload"])["exchange"]
            pairs = row["rod"]["rod_mesh_contact_pairs"]["pairs"]
            if row["guard_failure"] is not None or any(pair["mesh_name"] != "table" for pair in pairs):
                raise ValueError("measured precontact prefix is required")
            if row["physics_step"] == sdk:
                break
        else:
            raise ValueError("source prefix ended before the selected SDK boundary")
    identity = prelift._same_clock(row)
    actual, issued = prelift.joint_and_wrist(row, "left")
    requests, target = left_hold_windows(actual, issued)
    result = certify_left_hold(row, actual, issued)
    return dict(schema="dh116-pinch-left-precontact-hold-v1", identity=identity,
                source_ledger_sha256=prelift.SOURCE_LEDGER_SHA, requested_windows=requests.tolist(),
                measured_left_state=actual.tolist(), previous_left_target=issued.tolist(),
                left_hold_target=target.tolist(), node_reserve_m=prelift.NODE_RESERVE_M, proof=result,
                native_qualification=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sdk", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = screen(args.archive, args.sdk)
    prelift.base.lifecycle.native.write_new(args.output, report)
    print(json.dumps({"sdk_step": args.sdk, "proved": report["proof"]["proved"],
                      "sha256": prelift.base.lifecycle.native.sha(args.output)}))
    return int(not report["proof"]["proved"])


if __name__ == "__main__":
    raise SystemExit(main())
