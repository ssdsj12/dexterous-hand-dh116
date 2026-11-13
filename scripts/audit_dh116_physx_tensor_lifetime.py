#!/usr/bin/env python3
"""Read frozen Pinch/Thumb evidence without importing or starting Isaac."""
import argparse
import json
from pathlib import Path

from go2_pvcnn.control.dh116 import tabletop_native_lifetime_audit as helper
from go2_pvcnn.control.dh116.tabletop_native_lifetime_audit import (
    VERSION, digest, first_fatal_capture, read_pinned_json, summarize_ownership,
)


def audit_run(receipt_path, receipt_sha, ownership_sha):
    receipt_path = Path(receipt_path)
    receipt = read_pinned_json(receipt_path, receipt_sha)
    log_path = receipt_path.parent / "native.log"
    if digest(log_path) != receipt["log_sha256"]:
        raise ValueError("native log SHA256 differs from pinned receipt")
    ownership_path = Path(receipt["output"]) / "ownership.jsonl"
    if digest(ownership_path) != ownership_sha:
        raise ValueError("ownership SHA256 differs")
    raw = ownership_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("ownership ledger is incomplete")
    rows = [json.loads(line) for line in raw.splitlines()]
    return dict(
        receipt_path=str(receipt_path), receipt_sha256=receipt_sha,
        log_sha256=receipt["log_sha256"], ownership_sha256=ownership_sha,
        tested_head=receipt["tested_head"], code_config_sha256=receipt["code_config_sha256"],
        outer_exit_code=receipt["outer_exit_code"], duration_seconds=receipt["duration_seconds"],
        debugger_status_verified=receipt["debugger_status_verified"],
        debugger_inferior_exit_code=receipt.get("debugger_inferior_exit_code"),
        debugger_inferior_signal=receipt.get("debugger_inferior_signal"),
        ownership=summarize_ownership(rows),
        fatal=first_fatal_capture(log_path.read_text(errors="replace")),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pinch", "thumb"):
        parser.add_argument("--" + name + "-receipt", type=Path, required=True)
        parser.add_argument("--" + name + "-receipt-sha", required=True)
        parser.add_argument("--" + name + "-ownership-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(version=VERSION, native_execution_performed=False, training_updates=0,
                  root_cause_proven=False, source_sha256=digest(__file__),
                  helper_sha256=digest(helper.__file__))
    for name in ("pinch", "thumb"):
        result[name] = audit_run(getattr(args, name + "_receipt"),
                                 getattr(args, name + "_receipt_sha"),
                                 getattr(args, name + "_ownership_sha"))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), sha256=digest(args.output),
                         native_execution_performed=False)))


if __name__ == "__main__":
    main()
