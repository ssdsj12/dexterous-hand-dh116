#!/usr/bin/env python3
"""Verify a hand-only teacher/student replay manifest."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


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
    episodes = int(manifest["episodes"])
    steps = int(manifest["steps_per_episode"])
    if episodes <= 0 or steps < 2:
        raise AssertionError("invalid episode dimensions")
    for role in ("teacher", "student"):
        path = Path(manifest[role]["path"])
        assert path.exists(), role
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == episodes * steps, role
        assert all(row["phase"] in {"grasp", "lift"} for row in rows)
        for step in range(1, steps):
            for env in range(episodes):
                current = rows[step * episodes + env]
                previous = rows[(step - 1) * episodes + env]
                if not np.allclose(current["previous_action"], previous["action"], atol=1e-6, rtol=0):
                    raise AssertionError(f"{role} previous-action discontinuity at step={step}, env={env}")
    if manifest.get("rod_backend") == "torch_cuda":
        runtime = manifest.get("rod_runtime", {})
        assert runtime.get("state_device", "").startswith("cuda"), runtime
    assert manifest.get("physics_qualified") is False
    print("PASS: DH116 hand-only grasp/lift manifest; no Panda state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
