#!/usr/bin/env python3
"""Compare the fresh Pinch worker's update copy to a verified diagnostic receipt."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.run_dh116_work_packages import Repository, native_artifact


PACKAGE = "teacher.pinch-ppo-fresh-smoke32-v1"


def validate_owned_copy(lifecycle_bytes, owned_bytes, expected_lifecycle_sha):
    if hashlib.sha256(lifecycle_bytes).hexdigest() != expected_lifecycle_sha:
        raise ValueError("lifecycle bytes changed after diagnostic verification")
    lifecycle = json.loads(lifecycle_bytes)
    if not isinstance(lifecycle, dict) or not isinstance(lifecycle.get("acquisition"), dict):
        raise ValueError("complete R06 acquisition required")
    expected = dict(lifecycle["acquisition"], process_exit_verified=False,
                    native_qualification=False, teacher_admission=False)
    owned = json.loads(owned_bytes)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical(owned) != canonical(expected):
        raise ValueError("owned update does not match the unqualified R06 acquisition")
    return hashlib.sha256(owned_bytes).hexdigest()


def audit_owned_update(repository, expected_receipt_sha):
    verified = repository.verify_actual_diagnostic(PACKAGE, expected_receipt_sha)
    package = repository.registry()["packages"][PACKAGE]
    identity = repository.native_execution_identity(package)
    _, raw = repository._read_native_receipt_bytes(
        PACKAGE, attempt_fingerprint=identity["attempt_fingerprint"], expected=expected_receipt_sha)
    receipt = json.loads(raw)
    output = Path(receipt["output"])
    lifecycle = native_artifact(output, "lifecycle/result.json").read_bytes()
    owned = native_artifact(output, "owned_update.json").read_bytes()
    pinned = receipt["lifecycle_sha256"]["result.json"]
    return dict(verified, owned_update_sha256=validate_owned_copy(lifecycle, owned, pinned),
                lifecycle_result_sha256=pinned)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args(argv)
    repo = Repository(Path(__file__).resolve().parents[1], evidence_dir=args.evidence_dir)
    print(json.dumps(audit_owned_update(repo, args.receipt_sha256), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
