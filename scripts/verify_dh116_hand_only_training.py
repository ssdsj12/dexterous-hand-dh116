#!/usr/bin/env python3
"""Verify a hand-only teacher/student replay manifest."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["action_dim"] == 12
    assert manifest["previous_action_dim"] == 12
    assert manifest["arm_state_included"] is False
    assert manifest["entry_phase"] == "grasp"
    assert manifest["teacher"]["transitions"] == manifest["student"]["transitions"]
    for role in ("teacher", "student"):
        assert Path(manifest[role]["path"]).exists(), role
    print("PASS: DH116 hand-only grasp/lift manifest; no Panda state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
