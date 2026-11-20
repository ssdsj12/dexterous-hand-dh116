#!/usr/bin/env python3
"""Run the bounded left wrist relief comparison on the sealed early-finger prefix."""
import argparse
import functools
import json
from pathlib import Path

import numpy as np

from scripts import run_dh116_pinch_early_finger as early
from scripts import screen_dh116_pinch_wrist_relief as screen

base = early.base
PROOF_SHA = "deacfb73e810221e94b7f1ab6c1c72b78a7875a7ad321e7cf8201d4e5466aa21"
SCREEN_SHA = "f3e453fea84a84b6fe113ec7e8ff635446350b4c892ce2341b7f1079055a4474"


def prepare(args):
    proof = base._checked(args.wrist_preflight, PROOF_SHA)
    native = base.lifecycle.native
    if (native.sha(Path(screen.__file__)) != SCREEN_SHA or proof["qualified"] is not True
            or proof["source_ledger_sha256"] != screen.LEDGER_SHA):
        raise ValueError("verified offline wrist correction required")
    plan = early.prepare(args.output, args.sealed_reset, args.baseline, args.preflight)
    archived, rows = screen.load_archive(args.archive)
    if archived["action_schedule"] != plan["action_schedule"]:
        raise ValueError("exact completed early-finger contrast schedule must be retained")
    amended = screen.compile_schedule(plan["action_schedule"], proof["wrist_target_translation_delta_m"],
                                      proof["wrist_intrinsic_delta_rad"])
    if not np.array_equal(amended, np.asarray(proof["schedule"])):
        raise ValueError("recomputed action timing differs from verified wrist correction")
    plan["frozen_files"].update({
        str(args.wrist_preflight): PROOF_SHA,
        str(args.archive / "outer.json"): screen.OUTER_SHA,
        str(args.archive / "native" / "physical.jsonl"): screen.LEDGER_SHA,
    })
    plan.update(staged_version="dh116-pinch-wrist-relief-v1", action_schedule=amended.tolist(),
                wrist_relief_prefix=early.late.prefix_measurements(rows[screen.FIRST_SDK], expected_sdk=screen.FIRST_SDK),
                preserved_scope="source-bound two-window left wrist correction, no hold or lift")
    return plan


def guard_relief(row, reference):
    if row["physics_step"] != screen.FIRST_SDK or row["guard_failure"] is not None:
        return
    try:
        actual = early.late.prefix_measurements(row, expected_sdk=screen.FIRST_SDK)
        admission = early.late.compare_prefix(actual, reference)
        row["wrist_relief_admission"] = admission
        if not admission["passed"]:
            row["guard_failure"] = "wrist_relief_prefix_mismatch"
    except Exception as error:
        row["wrist_relief_admission"] = dict(passed=False, error=f"{type(error).__name__}: {error}")
        row["guard_failure"] = "wrist_relief_source_invalid"


def acquire(sim, owners, trace, output, *, plan):
    class WristReliefObservedClose(base.ObservedClose):
        def _step(self, action, substep):
            row = None
            try:
                row = super()._step(action, substep)
                guard_relief(row, plan["wrist_relief_prefix"])
                return row
            finally:
                self = action = row = None

    try:
        with base.bindings((base, "ObservedClose", WristReliefObservedClose)):
            return early.acquire(sim, owners, trace, output, plan=plan)
    finally:
        sim = owners = trace = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "plan", "sealed-reset", "baseline", "preflight", "archive", "wrist-preflight"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native = base.lifecycle.native
    native.validate_execution_context(base.ROOT)
    plan = prepare(args)
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
    else:
        if (args.approved_frozen_sha256 != digest or json.loads(args.plan.read_text()) != plan
                or args.output.exists()):
            raise ValueError("exact approved wrist relief plan and fresh output required")
        base.lifecycle.execute(plan, acquire_operation=functools.partial(acquire, plan=plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
