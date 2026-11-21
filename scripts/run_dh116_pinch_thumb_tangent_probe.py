#!/usr/bin/env python3
"""Test a bounded Pinch thumb tangent adjustment on the sealed fixed-wrist prefix."""

from functools import partial
from pathlib import Path

from scripts import run_dh116_pinch_fixed_wrist_seating as seating


VERSION = "dh116-pinch-thumb-tangent-probe-v1"
INCREMENTS_RAD = (.00025, -.00005)
compile_probe = partial(seating.compile_seating, increments=INCREMENTS_RAD)


def prepare(args, *, screen_geometry=False):
    plan = seating.prepare(args, screen_geometry=screen_geometry, compiler=compile_probe)
    source = Path(__file__).resolve()
    plan["frozen_files"][str(source)] = seating.base.lifecycle.native.sha(source)
    plan.update(staged_version=VERSION, preserved_scope=
        "measured fixed-wrist prefix, left thumb tangent probe and index relief, no lift or policy update")
    return plan


def acquire(sim, owners, trace, output, *, plan):
    try:
        if plan["staged_version"] != VERSION:
            raise ValueError("separate Pinch thumb tangent probe recipe required")
        return seating.acquire(sim, owners, trace, output, plan=plan, compiler=compile_probe,
                               boundary_name="thumb_tangent_probe_gate")
    finally:
        sim = owners = trace = None


if __name__ == "__main__":
    raise SystemExit(seating.closure.main(
        extra_flags=("touched", "left-preload", "right-preload", "preloaded", "left-trim", "right-trim",
                     "seated-prefix"),
        prepare_operation=prepare,
        acquire_operation=acquire,
    ))
