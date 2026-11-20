#!/usr/bin/env python3
"""Measure a single left index preload increment on the fixed-wrist Pinch prefix."""

from pathlib import Path

from scripts import run_dh116_pinch_fixed_wrist_seating as seating


VERSION = "dh116-pinch-left-index-only-preload-v1"
INDEX_INCREMENT_RAD = .0001
CERTIFY = seating.closure.certify_closure


def compile_index(previous, states, nominal_targets=None):
    return seating.compile_seating(previous, states, nominal_targets,
                                   increments=(0., INDEX_INCREMENT_RAD))


def prepare(args, *, screen_geometry=False):
    plan = seating.prepare(args, screen_geometry=screen_geometry, compiler=compile_index, certificate=CERTIFY)
    plan["staged_version"] = VERSION
    plan["preserved_scope"] = "left finger21 positive preload only, both wrists fixed, no lift or policy update"
    own_path = Path(__file__).resolve()
    plan["frozen_files"][str(own_path)] = seating.base.lifecycle.native.sha(own_path)
    return plan


def acquire(sim, owners, trace, output, *, plan):
    try:
        return seating.acquire(sim, owners, trace, output, plan=plan, compiler=compile_index,
                               certificate=CERTIFY, boundary_name="left_index_only_gate")
    finally:
        sim = owners = trace = None


if __name__ == "__main__":
    raise SystemExit(seating.closure.main(
        extra_flags=("touched", "left-preload", "right-preload", "preloaded", "left-trim", "right-trim",
                     "seated-prefix"),
        prepare_operation=prepare, acquire_operation=acquire,
    ))
