#!/usr/bin/env python3
"""Pace left Pinch index approach while the right wrist catches up."""
import argparse
import functools
import json
from pathlib import Path

import numpy as np

from scripts import run_dh116_pinch_precontact_pair as pair


base, prelift, late, wrist = pair.base, pair.prelift, pair.late, pair.wrist
INDEX_LEAD_RAD = .025
ADDED_WINDOWS = 6
LAST_SDK = pair.LAST_SDK + 8 * ADDED_WINDOWS
TRACKED = np.r_[1:3, 6:12]
OUTER_SHA = "3abac57052f958536b3028f910dbb2e13641eda6f9dcf4726c9526b72c1aeed1"
LEDGER_SHA = "6844a742d3e53d4ed6a9ab3b2d3786b5ec4e1b274bc03d656945d9c832947401"


def tracking_domain(actual, issued, target):
    actual, issued, target = [prelift._finite(x, (12,)) for x in (actual, issued, target)]
    before, after = (issued - actual)[TRACKED], (target - actual)[TRACKED]
    span = np.where(np.abs(before) >= np.abs(after), before, after)
    scale = np.where(span == 0., 1., span)
    ratios = np.stack((before / scale, after / scale))
    # A unit interval on a stationary dimension has a zero position derivative.
    lo = np.minimum(0., ratios.min(axis=0))
    hi = np.maximum(1., ratios.max(axis=0))
    reference = target.copy()
    reference[TRACKED] = actual[TRACKED] + span
    if np.any((issued - target)[[0, 3, 4, 5]]):
        raise ValueError("pacing keeps the other active commands fixed")
    return reference, np.stack((lo, hi))


def compile_index_pacing(previous, states):
    prefix = prelift._finite(previous, (7, 24))
    left, right = (states[side][1].copy() for side in ("left", "right"))
    finger_baseline = states["left"][0][2]
    left[2] = finger_baseline + INDEX_LEAD_RAD
    right[1:3] = states["right"][0][1:3]
    low, high = prelift.active_limits()
    if any(np.any(q[:6] < low) or np.any(q[:6] > high) for q in (left, right)):
        raise ValueError("pacing would exceed the original active finger limits")
    command = prelift.global_channel_order(*(states[s][1] for s in ("left", "right")))
    target = prelift.global_channel_order(left, right)
    remainder = (target - command) / prelift.WINDOW_SCALE
    actions = []
    for _ in range(ADDED_WINDOWS):
        remainder[np.abs(remainder) < 1e-12] = 0.
        action = np.clip(remainder, -1., 1.)
        if np.any(action[12:]):
            raise ValueError("pacing cannot change a wrist command")
        actions.append(action)
        remainder -= action
    if np.any(np.abs(remainder) > 1e-12):
        raise ValueError("pacing target exceeds the frozen window budget")
    return np.vstack((prefix, actions)), dict(left=left, right=right)


def certify_pacing(row, states, targets, *, prove=wrist.right_geometry):
    nodes = prelift._finite(row["rod"]["node_positions"][0], (41, 3))
    result = {}
    for side in ("left", "right"):
        actual, issued = states[side]
        reference, (lo, hi) = tracking_domain(actual, issued, targets[side])
        box = dict(index_pacing=(lo, hi), wrist_residuals=(np.zeros(3), np.zeros(3)))
        phases, _ = prove(row, actual, issued, reference, box, nodes=nodes,
                          phase_names=("index_pacing",), phase_seconds=25.,
                          side=side, independent_wrist=True, require_pad_clearance=side == "left")
        result[side] = phases["index_pacing"]
    return result


def archive_boundary(folder, *, outer_sha=OUTER_SHA, ledger_sha=LEDGER_SHA, sdk_steps=pair.LAST_SDK, windows=7):
    receipt = base._checked(folder / "outer.json", outer_sha)
    native = folder / "native"
    journal = native / "physical.jsonl"
    frozen = native / "frozen_plan.json"
    plan = base._checked(frozen, receipt["plan_sha256"])
    if (receipt["exit_code"] != 0 or receipt["forced_exit"] or receipt["output"] != str(native)
            or receipt["artifact_sha256"]["physical.jsonl"] != ledger_sha
            or base.lifecycle.native.sha(journal) != ledger_sha):
        raise ValueError("exact naturally closed precontact receipt required")
    result = base.inspect_interval(journal, np.asarray(plan["action_schedule"]), sdk_steps=sdk_steps)
    if (result["last_sdk_step"] != sdk_steps or result["terminal_reason"]
            or result["complete_control_windows"] != windows):
        raise ValueError("required precontact run failed its journal validation")
    last = None
    with journal.open() as stream:
        for line in stream:
            entry = json.loads(line)
            if entry.get("operation") == "physical":
                last = base.decode_owned(entry["payload"])["exchange"]
    if last is None or last["physics_step"] != sdk_steps:
        raise ValueError("settled precontact boundary required")
    pair.guard_hand_contact(last)
    late.prefix_measurements(last, expected_sdk=sdk_steps)
    return plan, last


def prepare(args, *, screen_geometry=False):
    previous = pair.prepare(args)
    archived, row = archive_boundary(args.precontact)
    if previous["action_schedule"] != archived["action_schedule"]:
        raise ValueError("live prefix plan differs from measured archived prefix")
    states = {s: prelift.joint_and_wrist(row, s) for s in ("left", "right")}
    actions, targets = compile_index_pacing(previous["action_schedule"], states)
    if screen_geometry:
        proofs = certify_pacing(row, states, targets)
        if not all(proof["proved"] for proof in proofs.values()):
            raise ValueError({s: proof.get("unresolved_shape") for s, proof in proofs.items()})
    for path in (args.precontact / "outer.json", args.precontact / "native" / "physical.jsonl",
                 args.precontact / "native" / "frozen_plan.json", Path(__file__).resolve()):
        previous["frozen_files"][str(path)] = base.lifecycle.native.sha(path)
    previous.update(staged_version="dh116-pinch-index-pacing-v1", action_schedule=actions.tolist(),
                    index_pacing=dict(prefix=late.prefix_measurements(row, expected_sdk=pair.LAST_SDK)),
                    preserved_scope="pace left index and unwind right finger lead; no contact or PPO")
    previous["budget"].update(control_steps=len(actions), sdk_steps=LAST_SDK,
                              solver_steps=(LAST_SDK - 2) * base.SUBSTEPS)
    return previous


def acquire(sim, owners, trace, output, *, plan, sdk_steps=LAST_SDK, after=None):
    def observe(row):
        if row["physics_step"] != pair.LAST_SDK:
            if after is not None:
                after(row)
            return
        receipt = row["index_pacing_gate"] = dict(passed=False)
        try:
            measured = late.prefix_measurements(row, expected_sdk=pair.LAST_SDK)
            receipt["prefix"] = late.compare_prefix(measured, plan["index_pacing"]["prefix"])
            states = {s: prelift.joint_and_wrist(row, s) for s in ("left", "right")}
            actions, targets = compile_index_pacing(plan["action_schedule"][:7], states)
            receipt["same_actions"] = bool(np.allclose(actions, plan["action_schedule"][:len(actions)], atol=1e-7, rtol=0))
            if receipt["prefix"]["passed"] and receipt["same_actions"]:
                receipt["proofs"] = certify_pacing(row, states, targets)
                receipt["passed"] = all(x["proved"] for x in receipt["proofs"].values())
            if not receipt["passed"]:
                row["guard_failure"] = "pinch_index_pacing_gate_rejected"
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            row["guard_failure"] = "pinch_index_pacing_source_invalid"

    try:
        return pair.acquire(sim, owners, trace, output, plan=plan, sdk_steps=sdk_steps, observe=observe)
    finally:
        sim = owners = trace = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "plan", "sealed-reset", "baseline", "preflight", "archive",
                 "wrist-preflight", "previous", "angular-proof", "left-proof", "right-proof", "precontact"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    native = base.lifecycle.native
    native.validate_execution_context(base.ROOT)
    plan = prepare(args, screen_geometry=args.approved_frozen_sha256 is None)
    digest = native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest, budget=plan["budget"])))
    else:
        if (args.approved_frozen_sha256 != digest or json.loads(args.plan.read_text()) != plan
                or args.output.exists()):
            raise ValueError("exact approved pacing plan and fresh output required")
        from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
        assert_clean_import_boundary()
        base.lifecycle.execute(plan, acquire_operation=functools.partial(acquire, plan=plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
