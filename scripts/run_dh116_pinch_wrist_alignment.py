#!/usr/bin/env python3
"""Measure a certified wrist-only Pinch alignment after index pacing."""
import argparse
import functools
import json
from pathlib import Path

import numpy as np

from scripts import run_dh116_pinch_index_pacing as pacing


base, prelift, late, wrist = pacing.base, pacing.prelift, pacing.late, pacing.wrist
ALIGN_WINDOWS = 6
LAST_SDK = pacing.LAST_SDK + 8 * ALIGN_WINDOWS
OUTER_SHA = "32e260a1c920849e3c30c78d6305b1d4be41048eacd244c19b094f56edcf117f"
LEDGER_SHA = "880f582c251f18e5e1cbe10a40af9a357669a0db69d0f0a416d845de3f4dc8b1"
SOURCES = {
    "left": [0.0001868587695993483, 0., -0.00021101734563708308,
             0.0018235398316755892, 0.0003010580632835627, 0.],
    "right": [0., -.00040, -.00012, 0., 0., 0.],
}


def compile_alignment(previous, states, offsets):
    prefix = prelift._finite(previous, (13, 24))
    targets = {side: states[side][1].copy() for side in ("left", "right")}
    for side, target in targets.items():
        target[6:] += prelift._finite(offsets[side], (6,))
    cmd = prelift.global_channel_order(*(states[side][1] for side in ("left", "right")))
    goal = prelift.global_channel_order(*(targets[side] for side in ("left", "right")))
    remainder = (goal - cmd) / prelift.WINDOW_SCALE
    actions = []
    for _ in range(ALIGN_WINDOWS):
        remainder[np.abs(remainder) < 1e-12] = 0.
        action = np.zeros(24)
        for group in prelift.VECTORS:
            delta = remainder[group]
            action[group] = delta / max(1., np.linalg.norm(delta))
        actions.append(action)
        remainder -= action
    if np.any(abs(remainder) > 1e-12):
        raise ValueError("wrist target exceeds frozen alignment budget")
    return np.vstack((prefix, actions)), targets


def certify_alignment(row, states, targets):
    nodes = prelift._finite(row["rod"]["node_positions"][0], (41, 3))
    result = {}
    for side in ("left", "right"):
        actual, issued = states[side]
        reference, bounds = pacing.tracking_domain(actual, issued, targets[side])
        envelope = dict(alignment=tuple(bounds), wrist_residuals=(np.zeros(3), np.zeros(3)))
        phases, _ = wrist.right_geometry(row, actual, issued, reference, envelope, nodes=nodes,
              phase_names=("alignment",), phase_seconds=25., side=side, independent_wrist=True)
        result[side] = phases["alignment"]
    return result


def prepare(args, *, screen_geometry=False):
    previous = pacing.prepare(args)
    archive, row = pacing.archive_boundary(args.pacing, outer_sha=OUTER_SHA, ledger_sha=LEDGER_SHA,
                                           sdk_steps=pacing.LAST_SDK, windows=13)
    if previous["action_schedule"] != archive["action_schedule"]:
        raise ValueError("only the sealed index pacing prefix can be continued")
    states = {side: prelift.joint_and_wrist(row, side) for side in ("left", "right")}
    actions, targets = compile_alignment(previous["action_schedule"], states, SOURCES)
    if screen_geometry:
        proofs = certify_alignment(row, states, targets)
        if not all(p["proved"] for p in proofs.values()):
            raise ValueError({s: p.get("unresolved_shape") for s, p in proofs.items()})
    for path in (args.pacing / "outer.json", args.pacing / "native" / "physical.jsonl",
                 args.pacing / "native" / "frozen_plan.json", Path(__file__).resolve(),
                 Path(pacing.__file__).resolve()):
        previous["frozen_files"][str(path)] = base.lifecycle.native.sha(path)
    previous.update(staged_version="dh116-pinch-wrist-alignment-v1", action_schedule=actions.tolist(),
                    wrist_alignment=dict(prefix=late.prefix_measurements(row, expected_sdk=pacing.LAST_SDK),
                                         offsets=SOURCES),
                    preserved_scope="no-contact wrist alignment, unchanged finger targets, no PPO")
    previous["budget"].update(control_steps=len(actions), sdk_steps=LAST_SDK,
                              solver_steps=(LAST_SDK - 2) * base.SUBSTEPS)
    return previous


def acquire(sim, owners, trace, output, *, plan):
    def observe(row):
        if row["physics_step"] != pacing.LAST_SDK:
            return
        receipt = row["wrist_alignment_gate"] = dict(passed=False)
        try:
            measured = late.prefix_measurements(row, expected_sdk=pacing.LAST_SDK)
            cfg = plan["wrist_alignment"]
            receipt["prefix"] = late.compare_prefix(measured, cfg["prefix"])
            states = {side: prelift.joint_and_wrist(row, side) for side in ("left", "right")}
            actions, targets = compile_alignment(plan["action_schedule"][:13], states, cfg["offsets"])
            receipt["same_actions"] = bool(np.allclose(actions, plan["action_schedule"], atol=1e-7, rtol=0))
            if receipt["prefix"]["passed"] and receipt["same_actions"]:
                receipt["proofs"] = certify_alignment(row, states, targets)
                receipt["passed"] = all(x["proved"] for x in receipt["proofs"].values())
            if not receipt["passed"]:
                row["guard_failure"] = "pinch_wrist_alignment_gate_rejected"
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            row["guard_failure"] = "pinch_wrist_alignment_source_invalid"

    try:
        return pacing.acquire(sim, owners, trace, output, plan=plan, sdk_steps=LAST_SDK, after=observe)
    finally:
        sim = owners = trace = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "plan", "sealed-reset", "baseline", "preflight", "archive", "wrist-preflight",
                 "previous", "angular-proof", "left-proof", "right-proof", "precontact", "pacing"):
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
            raise ValueError("exact approved wrist alignment plan and fresh output required")
        from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
        assert_clean_import_boundary()
        base.lifecycle.execute(plan, acquire_operation=functools.partial(acquire, plan=plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
