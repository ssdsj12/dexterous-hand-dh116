#!/usr/bin/env python3
"""Test late measured Pinch load recovery after the same thumb-tangent prefix."""

from functools import partial
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_pinch_force_hold_feedback import FixedWristPinchForceHold
from scripts import run_dh116_pinch_thumb_tangent_probe as tangent
from scripts.run_dh116_pinch_fixed_force_feedback import FeedbackAction

VERSION = "dh116-pinch-tangent-force-hold-v1"
FIRST_BOUNDARY = 418
PREFIX_CONTROLS = (FIRST_BOUNDARY-2)//8
OUTER_SHA = "02bcb1323379edcbd226f6c356ad3f4a7f807f2cc77dd1066fe56d1efbb14e4f"
LEDGER_SHA = "b332bb35f4840d4b824084366548d73cbff0ee48438c7a45c4abc129cee7cff2"
seating = tangent.seating
base, closure = seating.base, seating.closure


class DelayedTangentFeedback(FeedbackAction):
    """Only allow the original left-index force selector once the measured prefix ends."""

    def __init__(self):
        super().__init__()
        self._force_hold = FixedWristPinchForceHold(FIRST_BOUNDARY)

    def __call__(self, index, nominal_action, rows):
        if index < PREFIX_CONTROLS:
            return nominal_action, None
        return super().__call__(index, nominal_action, rows)


def prepare(args, *, screen_geometry=False):
    plan = tangent.prepare(args, screen_geometry=screen_geometry)
    archive_dir = args.tangent_archive.resolve(strict=True)
    receipt_path = archive_dir/"outer.json"
    ledger = archive_dir/"native/physical.jsonl"
    frozen = archive_dir/"native/frozen_plan.json"
    receipt = base._checked(receipt_path, OUTER_SHA)
    native = base.lifecycle.native
    if (receipt["exit_code"] != 0 or receipt["debugger_inferior_exit_code"] != 0
            or receipt["forced_exit"] or receipt["exceeded_budget"]
            or receipt["artifact_sha256"][ledger.name] != LEDGER_SHA or native.sha(ledger) != LEDGER_SHA):
        raise ValueError("complete naturally closed thumb tangent comparator required")
    archive = base._checked(frozen, receipt["artifact_sha256"][frozen.name])
    if archive["staged_version"] != tangent.VERSION or archive["action_schedule"] != plan["action_schedule"]:
        raise ValueError("independent exact thumb tangent native comparator required")
    checked = base.inspect_interval(ledger, np.asarray(archive["action_schedule"]), sdk_steps=seating.LAST_SDK,
                                    sealed_archive=archive)
    if checked["terminal_reason"] or checked["complete_control_windows"] != (seating.LAST_SDK-2)//8:
        raise ValueError("complete original physical and action comparator required")
    for source in (Path(__file__).resolve(), receipt_path, ledger, frozen):
        plan["frozen_files"][str(source)] = native.sha(source)
    plan.update(staged_version=VERSION, late_force_boundary_sdk=FIRST_BOUNDARY,
                preserved_scope="matched fixed-wrist thumb tangent prefix then measured left index load recovery")
    return plan


def acquire(sim, owners, trace, output, *, plan):
    try:
        if plan["staged_version"] != VERSION or plan["late_force_boundary_sdk"] != FIRST_BOUNDARY:
            raise ValueError("exact independent late force recovery recipe required")
        action_owner = partial(base.acquire, feedback_factory=DelayedTangentFeedback)
        with base.bindings((base, "acquire", action_owner)):
            return seating.acquire(sim, owners, trace, output, plan=plan, compiler=tangent.compile_probe,
                boundary_name="tangent_force_hold_gate")
    finally:
        sim = owners = trace = None


if __name__ == "__main__":
    raise SystemExit(closure.main(
        extra_flags=("touched","left-preload","right-preload","preloaded","left-trim","right-trim",
                     "seated-prefix","tangent-archive"),
        prepare_operation=prepare, acquire_operation=acquire,
    ))
