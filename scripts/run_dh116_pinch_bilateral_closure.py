#!/usr/bin/env python3
"""Measure bilateral Pinch contact after an independently verified wrist alignment."""
import argparse
import functools
import json
from pathlib import Path

import numpy as np

from scripts import run_dh116_pinch_wrist_alignment as alignment


base, prelift, late, wrist, pacing = alignment.base, alignment.prelift, alignment.late, alignment.wrist, alignment.pacing
CLOSING_WINDOWS = 4
SETTLE_WINDOWS = 2
LAST_SDK = alignment.LAST_SDK + 8 * (CLOSING_WINDOWS + SETTLE_WINDOWS)
WALL_S = 240
UNCHANGED = np.r_[0, 3:7, 9:24]
CONTACT_MESHES = frozenset(("table", "left.finger14_link", "left.finger23_link",
                            "right.finger14_link", "right.finger23_link"))
OUTER_SHA = "1c60be7bcae80ded0476133a6dcd0f0974534dc1bd0e2cf91477718a7e23de6e"
LEDGER_SHA = "9711361ae252af26ec92bef27c158fbd5f07bd9c7e55fd83419e02154fc3691c"
GOAL_SHA = {"left": "4f3b86a86d57ac487e4a3a2baec976caa41ab7ffc1ef74e8be22dfb74df0251f",
            "right": "56ec58d71207e10d0bd02d492ff6c7b7af7c9cbdb3dd3fe08892c13a41b17541"}


def compile_closure(previous, states, goals):
    prefix = prelift._finite(previous, (19, 24))
    targets, motions = {}, {}
    low, high = prelift.active_limits()
    for side in ("left", "right"):
        command = prelift._finite(states[side][1], (12,))
        target = command.copy()
        target[1:3] = prelift._finite(goals[side], (2,))
        if np.any(target[:6] < low) or np.any(target[:6] > high) or np.any(target[1:3] < command[1:3]):
            raise ValueError("original safe closing finger targets required")
        motions[side] = target - command
        targets[side] = target
    advance = prelift.global_channel_order(motions["left"], motions["right"]) / (prelift.WINDOW_SCALE * CLOSING_WINDOWS)
    if np.any(abs(advance) > 1.) or np.any(advance[UNCHANGED]):
        raise ValueError("selected four-finger closure exceeds input rate")
    return np.vstack((prefix, np.tile(advance, (CLOSING_WINDOWS, 1)), np.zeros((SETTLE_WINDOWS, 24)))), targets


def guard_selected_contact(row):
    if row["guard_failure"] is None and any(
            pair["mesh_name"] not in CONTACT_MESHES for pair in row["rod"]["rod_mesh_contact_pairs"]["pairs"]):
        row["guard_failure"] = "pinch_unselected_contact_pair"


def certify_closure(row, states, targets):
    nodes = prelift._finite(row["rod"]["node_positions"][0], (41, 3))
    result = {}
    for side in ("left", "right"):
        actual, issued = states[side]
        reference, bounds = pacing.tracking_domain(actual, issued, targets[side])
        envelope = dict(closure=tuple(bounds), wrist_residuals=(np.zeros(3), np.zeros(3)))
        phases, _ = wrist.right_geometry(row, actual, issued, reference, envelope, nodes=nodes,
            phase_names=("closure",), phase_seconds=25., side=side, independent_wrist=True)
        result[side] = phases["closure"]
    return result


def prepare(args, *, screen_geometry=False):
    previous = alignment.prepare(args)
    archive, row = pacing.archive_boundary(args.aligned, outer_sha=OUTER_SHA, ledger_sha=LEDGER_SHA,
                                            sdk_steps=alignment.LAST_SDK, windows=19)
    if previous["action_schedule"] != archive["action_schedule"]:
        raise ValueError("wrist alignment prefix differs from sealed native actions")
    goals = {}
    for side, name in (("left", args.left_closure), ("right", args.right_closure)):
        item = base._checked(name, GOAL_SHA[side])[0]
        if (item["side"] != side or item["sdk"] != alignment.LAST_SDK or item["depth_mm"] != .04
                or item["proved"] is not True):
            raise ValueError("exact earlier screened shallow closure targets required")
        goals[side] = item["target"][1:3]
        previous["frozen_files"][str(name)] = GOAL_SHA[side]
    states = {side: prelift.joint_and_wrist(row, side) for side in ("left", "right")}
    actions, targets = compile_closure(previous["action_schedule"], states, goals)
    if screen_geometry:
        proofs = certify_closure(row, states, targets)
        if not all(x["proved"] for x in proofs.values()):
            raise ValueError({side: x.get("unresolved_shape") for side, x in proofs.items()})
    for path in (args.aligned / "outer.json", args.aligned / "native" / "physical.jsonl",
                 args.aligned / "native" / "frozen_plan.json", Path(__file__).resolve()):
        previous["frozen_files"][str(path)] = base.lifecycle.native.sha(path)
    previous.update(staged_version="dh116-pinch-bilateral-closure-v1", action_schedule=actions.tolist(),
                    bilateral_closure=dict(prefix=late.prefix_measurements(row, expected_sdk=alignment.LAST_SDK),
                                           goals=goals),
                    preserved_scope="original safe bilateral closure and measured contact, no lift or PPO")
    previous["budget"].update(control_steps=len(actions), sdk_steps=LAST_SDK, soft_wall_seconds=WALL_S,
                              solver_steps=(LAST_SDK - 2) * base.SUBSTEPS)
    return previous


def acquire(sim, owners, trace, output, *, plan):
    def observe(row):
        step = row["physics_step"]
        if step > alignment.LAST_SDK:
            guard_selected_contact(row)
            return
        if step != alignment.LAST_SDK:
            return
        receipt = row["bilateral_closure_gate"] = dict(passed=False)
        try:
            measured = late.prefix_measurements(row, expected_sdk=alignment.LAST_SDK)
            cfg = plan["bilateral_closure"]
            receipt["prefix"] = late.compare_prefix(measured, cfg["prefix"])
            states = {side: prelift.joint_and_wrist(row, side) for side in ("left", "right")}
            actions, targets = compile_closure(plan["action_schedule"][:19], states, cfg["goals"])
            receipt["same_actions"] = bool(np.allclose(actions, plan["action_schedule"], atol=1e-7, rtol=0))
            if receipt["prefix"]["passed"] and receipt["same_actions"]:
                receipt["proofs"] = certify_closure(row, states, targets)
                receipt["passed"] = all(x["proved"] for x in receipt["proofs"].values())
            if not receipt["passed"]:
                row["guard_failure"] = "pinch_bilateral_closure_gate_rejected"
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            row["guard_failure"] = "pinch_bilateral_closure_source_invalid"

    try:
        return alignment.acquire(sim, owners, trace, output, plan=plan, sdk_steps=LAST_SDK,
                                 after=observe, contact_after=alignment.LAST_SDK)
    finally:
        sim = owners = trace = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "plan", "sealed-reset", "baseline", "preflight", "archive", "wrist-preflight",
                 "previous", "angular-proof", "left-proof", "right-proof", "precontact", "pacing", "aligned",
                 "left-closure", "right-closure"):
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
            raise ValueError("exact approved closure plan and fresh output required")
        from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
        assert_clean_import_boundary()
        base.lifecycle.execute(plan, acquire_operation=functools.partial(acquire, plan=plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
