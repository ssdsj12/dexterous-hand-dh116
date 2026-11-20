#!/usr/bin/env python3
"""Retry only the near-rod approach timing using a longer, bounded wrist settle."""

import argparse
import functools
import json
from pathlib import Path
import time

import numpy as np

from scripts import run_dh116_tripod_near_corridor as near


staged = near.staged
VERSION = "dh116-right-tripod-near-settle-extension-v1"
EXTRA_WINDOWS = 3
GATE_SDK = staged.GATE_SDK + 8 * EXTRA_WINDOWS
SDK_STEPS = staged.SDK_STEPS + 8 * EXTRA_WINDOWS
OUTER_SHA = "c8323ee17528a4bf1998197cd79821e3e3d3d369e8a5e8abe8d89ece419bad5b"
METRICS_SHA = "af9c6abe1bc8674a8d3e667b3c5e00f713a96cf41b629848e85e581d272236dd"


def extend_reference(reference):
    cut, count = staged.APPROACH_START, staged.CONTROL_WINDOWS
    actions = staged.tracking.finite(reference["actions"], (count, 24))
    if (reference["physics_steps"] != staged.SDK_STEPS - 2 or reference["policy_steps"] != count
            or reference["native_ready"] is not False or np.any(actions[cut - 1])):
        raise ValueError("exact preflight path with a quiet predecessor required")
    expanded = np.vstack((actions[:cut], np.zeros((EXTRA_WINDOWS, 24)), actions[cut:]))
    updated = dict(reference, actions=expanded.tolist(), physics_steps=SDK_STEPS - 2,
                   policy_steps=count + EXTRA_WINDOWS)
    for key in ("intrinsic_knots_rad", "world_offset_knots_m"):
        knots = staged.tracking.finite(reference[key], (count + 1, 3))
        if not np.array_equal(knots[cut - 1], knots[cut]):
            raise ValueError("quiet predecessor must also have identical adjacent target knots")
        updated[key] = np.vstack((knots[:cut + 1], np.repeat(knots[cut:cut + 1], EXTRA_WINDOWS, axis=0),
                                  knots[cut + 1:])).tolist()
    return updated


def prepare(args):
    plan = near.prepare(args)
    outer = staged.tracking.checked_json(args.stop_receipt, OUTER_SHA)
    measured = staged.tracking.checked_json(args.stop_metrics, METRICS_SHA)
    wrist = measured["final_wrist_gate"]
    rows = wrist["rows"]
    pitch = np.abs(staged.tracking.finite([row["qd"] for row in rows], (8, 3))[:, 1])
    if (outer["exit_code"] != 1 or outer["forced_exit"] is not False
            or measured["ledger_sha256"] != outer["artifact_sha256"]["physical.jsonl"]
            or not measured["app_close_returned"] or measured["geometry_rows"] != staged.GATE_SDK - 2
            or measured["last_sdk"] != staged.GATE_SDK or wrist["sdk_step"] != staged.GATE_SDK
            or wrist["passed"] is not False or wrist["required_speed_rad_s"] != staged.SPEED_MAX_RAD_S
            or wrist["tracking_error_rad"] > staged.ANGLE_ERROR_MAX_RAD
            or pitch[0] <= staged.SPEED_MAX_RAD_S or np.any(np.diff(pitch) >= 0.) or pitch[-1] <= 0.):
        raise ValueError("same sealed speed-only wrist gate failure required")
    reference = extend_reference(plan["tracking_plan"]["right_reference"])
    budget = dict(staged.BUDGET, sdk_steps=SDK_STEPS, motion=(reference["policy_steps"] - 2) * 8)
    plan["tracking_plan"].update(budget=budget, right_reference=reference,
                                 actions24=reference["actions"], diagnostic_version=VERSION)
    plan["budget"].update(mode="right_tripod_near_settle_extension", sdk_steps=SDK_STEPS,
                          control_steps=reference["policy_steps"], solver_steps=(SDK_STEPS - 2) * 12)
    plan["frozen_files"].update({str(args.stop_receipt): OUTER_SHA, str(args.stop_metrics): METRICS_SHA,
                                 str(Path(__file__).resolve()): staged.native.sha(Path(__file__))})
    plan.update(version=VERSION, added_quiet_sdk_steps=EXTRA_WINDOWS * 8, wrist_gate_sdk=GATE_SDK,
                preserved_scope="unchanged no-contact Tripod geometry with 24 extra static wrist steps")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raised-study", "original", "proof", "near-proof", "stress", "stop-receipt",
                 "stop-metrics", "output", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native = staged.native
    native.validate_execution_context(staged.cal.launcher.ROOT)
    plan = prepare(args)
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
    else:
        if digest != args.approved_frozen_sha256 or json.loads(args.plan.read_bytes()) != plan or args.output.exists():
            raise ValueError("exact frozen settle extension with a fresh output required")
        deadline = time.monotonic() + near.WALL_SECONDS - staged.cal.launcher.CLOSE_RESERVE_SECONDS
        staged.reset.execute(
            plan, acquire_operation=functools.partial(near.acquire, plan=plan, args=args, deadline=deadline,
                                                       gate_sdk=GATE_SDK)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
