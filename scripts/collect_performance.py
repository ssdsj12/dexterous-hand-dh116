#!/usr/bin/env python3
"""Collect matched DH116 performance runs into a machine-readable report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def _percentile(values, p):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (x - lo)


def compare_runs(runs):
    walls = [float(run["wall_s"]) for run in runs]
    return {
        "runs": len(walls),
        "median_wall_s": statistics.median(walls),
        "p95_wall_s": _percentile(walls, .95),
        "median_sps": statistics.median(float(run.get("sps", 0.)) for run in runs),
        "median_rtf": statistics.median(float(run.get("rtf", 0.)) for run in runs),
    }


def _load(path: Path):
    metadata = json.loads((path / "comparison_manifest.json").read_text())
    summary = json.loads((path / "perf_summary.json").read_text())
    manifest = json.loads((path / "run_manifest.json").read_text())
    stages = summary.get("stages", {})
    wall = sum(float(stage["wall_s"]) for stage in stages.values())
    steps = sum(int(stage["steps"]) for stage in stages.values())
    dt = float(summary["physics_dt_s"])
    return {
        "path": str(path), "variant": metadata["variant"], "phase": metadata["phase"],
        "repetition": metadata["repetition"], "wall_s": wall, "steps": steps,
        "sim_time_s": steps * dt, "sps": steps / wall if wall > 0 else 0.,
        "rtf": steps * dt / wall if wall > 0 else 0.,
        "physics_dt_s": dt, "run_config": manifest.get("run_config", {}),
        "completed_steps": sum(int(stage.get("completed_steps", 0)) for stage in stages.values()),
    }


def collect(paths, output):
    runs = [_load(Path(path)) for path in paths]
    groups = {}
    for run in runs:
        groups.setdefault((run["variant"], run["phase"]), []).append(run)
    report = {"schema": "dh116-performance-v1", "groups": []}
    for (variant, phase), group in sorted(groups.items()):
        identities = {
            (run["physics_dt_s"], run["run_config"].get("wire_py_dt"),
             run["run_config"].get("contact_subsamples"), run["steps"])
            for run in group
        }
        comparable = len(group) >= 3 and len(identities) == 1 and all(
            run["completed_steps"] == run["steps"] for run in group)
        stats = compare_runs(group) if group else {}
        report["groups"].append({
            "variant": variant, "phase": phase, "comparable": comparable,
            "physics_identity": len(identities) == 1, "runs": group, "statistics": stats,
        })
    output = Path(output)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    collect(args.runs, args.output)
