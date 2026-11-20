#!/usr/bin/env python3
"""Measure a 10-window zero-action control after the sealed 45-window Pinch prefix."""

from pathlib import Path

from scripts import run_dh116_pinch_fixed_wrist_seating as seating


VERSION = "dh116-pinch-fixed-force-hold-control-v1"
CERTIFY = seating.closure.certify_closure


def compile_noop(previous, states, nominal_targets=None):
    return seating.compile_seating(previous, states, nominal_targets, increments=(0., 0.))


def prepare(args, *, screen_geometry=False):
    plan = seating.prepare(args, screen_geometry=screen_geometry, compiler=compile_noop, certificate=CERTIFY)
    plan["staged_version"] = VERSION
    plan["preserved_scope"] = "same fixed-wrist Pinch prefix with 10 zero-action control windows"
    source = Path(__file__).resolve()
    plan["frozen_files"][str(source)] = seating.base.lifecycle.native.sha(source)
    return plan


def acquire(sim, owners, trace, output, *, plan):
    try:
        return seating.acquire(sim, owners, trace, output, plan=plan, compiler=compile_noop,
                               certificate=CERTIFY, boundary_name="fixed_force_hold_control_gate")
    finally:
        sim = owners = trace = None


if __name__ == "__main__":
    raise SystemExit(seating.closure.main(
        extra_flags=("touched", "left-preload", "right-preload", "preloaded", "left-trim", "right-trim",
                     "seated-prefix"),
        prepare_operation=prepare, acquire_operation=acquire,
    ))
