#!/usr/bin/env python3
"""Run one source-bound early left-finger Pinch contact comparison."""
import argparse
import functools
import json
from pathlib import Path

import numpy as np

from scripts import run_dh116_pinch_late_wrist as late
from scripts import screen_dh116_pinch_early_finger as screen

base = late.base
START_SDK = 34
PREFLIGHT_SHA = "c2f5339da95a0dd3bcb9208306ce7d3e0f89478f4a17309c58083ab8ef39bb69"
SCREEN_SHA = "0bde01a475df8ab9a1bc9a30a225c59425cc1a6f5a932dee9bb538878b177f22"


def prepare(output, sealed, baseline, preflight):
    evidence = base._checked(preflight, PREFLIGHT_SHA)
    digest = base.lifecycle.native.sha
    if (digest(Path(screen.__file__)) != SCREEN_SHA or evidence["qualified_preflight"] is not True
            or evidence["frozen_original_ledger_sha256"] != screen.BASELINE_LEDGER_SHA
            or [row["sdk_step"] for row in evidence["rows"]] != list(range(START_SDK, 75))
            or not screen.qualified_preflight(evidence["rows"], terminal_row_count=41)):
        raise ValueError("verified source-bound early index screening required")
    plan = base.freeze(output, sealed)
    original, archive = screen.load_reference(baseline)
    changed = screen.compile_variant(original, thumb_rad=.0026, index_rad=.007)
    if (original.tolist() != plan["action_schedule"]
            or not np.array_equal(changed, np.asarray(evidence["action_schedule"]))):
        raise ValueError("reference and proven command schedules must agree")
    source = archive[START_SDK - 3]
    plan["frozen_files"].update({
        str(preflight): PREFLIGHT_SHA,
        str(baseline / "outer.json"): screen.BASELINE_OUTER_SHA,
        str(baseline / "native" / "physical.jsonl"): screen.BASELINE_LEDGER_SHA,
    })
    plan.update(staged_version="dh116-pinch-early-left-finger-v1", action_schedule=changed.tolist(),
                early_finger_prefix=late.prefix_measurements(source, expected_sdk=START_SDK),
                preserved_scope="original close lifetime and guards; only early left thumb/index command")
    return plan


def guard_prefix(row, reference):
    if row["physics_step"] != START_SDK or row["guard_failure"] is not None:
        return
    try:
        measured = late.prefix_measurements(row, expected_sdk=START_SDK)
        receipt = late.compare_prefix(measured, reference)
        row["early_finger_admission"] = receipt
        if not receipt["passed"]:
            row["guard_failure"] = "early_left_finger_prefix_mismatch"
    except Exception as error:
        row["early_finger_admission"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        row["guard_failure"] = "early_left_finger_source_invalid"


def acquire(sim, owners, trace, output, *, plan):
    class EarlyFingerObservedClose(base.ObservedClose):
        def _step(self, action, substep):
            row = None
            try:
                row = super()._step(action, substep)
                guard_prefix(row, plan["early_finger_prefix"])
                return row
            finally:
                self = action = row = None

    try:
        with base.bindings((base, "ObservedClose", EarlyFingerObservedClose)):
            return base.acquire(sim, owners, trace, output, plan=plan)
    finally:
        sim = owners = trace = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("output", "plan", "sealed-reset", "baseline", "preflight"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native = base.lifecycle.native
    native.validate_execution_context(base.ROOT)
    plan = prepare(args.output, args.sealed_reset, args.baseline, args.preflight)
    frozen_sha = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps({"frozen_sha256": frozen_sha, "budget": plan["budget"]}))
    else:
        if (args.approved_frozen_sha256 != frozen_sha
                or json.loads(args.plan.read_text()) != plan or args.output.exists()):
            raise ValueError("exact approved early-finger plan and fresh output required")
        base.lifecycle.execute(plan, acquire_operation=functools.partial(acquire, plan=plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
