#!/usr/bin/env python3
"""Independently preload right Pinch pads and the left thumb at fixed wrist targets."""
from pathlib import Path

import numpy as np

from scripts import run_dh116_pinch_bilateral_preload as preload


closure, base, prelift, pacing = preload.closure, preload.base, preload.prelift, preload.pacing
PREFIX_WINDOWS = (preload.LAST_SDK - 2) // 8
LAST_SDK = preload.LAST_SDK + 8 * (preload.RAMP_WINDOWS + preload.SETTLE_WINDOWS)
OUTER_SHA = "78463d4cef19304d99021ddfbe0e4057f74765bed07f78d290c1d15ed0fffe34"
LEDGER_SHA = "d238e15ba75f492bb3d867a033547c2b140cbb963507fa44a318358cd4fe1ec3"
SOURCES = {
    "left": (2, .105, "893bf39b7770ebe82b5e41ce8b5f54bb74ff83c9104156b480d2aaef36ddcc0b"),
    "right": (3, .116, "9f3828292042af7ed3f679db3a519baa59c63a8bc52ad5ab3c49e5d593f911b6"),
}
GATE_KEY = "pinch_asymmetric_preload"


def compile_asymmetric(previous, states, goals):
    left = prelift._finite(goals["left"], (2,))
    right = prelift._finite(goals["right"], (2,))
    if left[1] != states["left"][1][2]:
        raise ValueError("left index must retain its issued target")
    if left[0] <= states["left"][1][1] or np.any(right <= states["right"][1][1:3]):
        raise ValueError("asymmetric preload requires positive closing targets on the three selected channels")
    return closure.compile_closure(previous, states, goals, prefix_windows=PREFIX_WINDOWS,
                                   closing_windows=preload.RAMP_WINDOWS, settle_windows=preload.SETTLE_WINDOWS)


def prepare(args, *, screen_geometry=False):
    plan = preload.prepare(args)
    archive, row = pacing.archive_boundary(
        args.preloaded, outer_sha=OUTER_SHA, ledger_sha=LEDGER_SHA, sdk_steps=preload.LAST_SDK,
        windows=PREFIX_WINDOWS, contact_guard=closure.guard_selected_contact,
    )
    if plan["action_schedule"] != archive["action_schedule"] or not preload.boundary_has_opposition(
            row, sdk_step=preload.LAST_SDK):
        raise ValueError("sealed fixed-wrist preload boundary required")
    goals = {}
    for side, path in (("left", args.left_trim), ("right", args.right_trim)):
        index, depth, digest = SOURCES[side]
        item = base._checked(path, digest)[index]
        if (item["side"] != side or item["sdk"] != preload.LAST_SDK or item["depth_mm"] != depth
                or item["proved"] is not True):
            raise ValueError("exact side-specific geometry evidence is required")
        goals[side] = item["target"][1:3]
        plan["frozen_files"][str(path)] = digest
    states = {side: prelift.joint_and_wrist(row, side) for side in goals}
    actions, targets = compile_asymmetric(plan["action_schedule"], states, goals)
    if screen_geometry:
        proofs = closure.certify_closure(row, states, targets)
        if not all(proof["proved"] for proof in proofs.values()):
            raise ValueError({side: proof.get("unresolved_shape") for side, proof in proofs.items()})
    for path in (args.preloaded / "outer.json", args.preloaded / "native" / "physical.jsonl",
                 args.preloaded / "native" / "frozen_plan.json", Path(__file__).resolve()):
        plan["frozen_files"][str(path)] = base.lifecycle.native.sha(path)
    plan.update(
        staged_version="dh116-pinch-asymmetric-preload-v1", action_schedule=actions.tolist(),
        preserved_scope="bounded preload without moving wrists or left index, no lift or PPO",
    )
    plan[GATE_KEY] = dict(prefix=preload.late.prefix_measurements(row, expected_sdk=preload.LAST_SDK), goals=goals)
    plan["budget"].update(control_steps=len(actions), sdk_steps=LAST_SDK, soft_wall_seconds=360,
                          solver_steps=(LAST_SDK - 2) * base.SUBSTEPS)
    return plan


def acquire(sim, owners, trace, output, *, plan):
    def observe(row):
        if row["physics_step"] == preload.LAST_SDK:
            preload.measure_gate(row, plan, sdk_step=preload.LAST_SDK, prefix_windows=PREFIX_WINDOWS,
                                 config_key=GATE_KEY, gate_key="asymmetric_preload_gate", compiler=compile_asymmetric)

    try:
        return preload.acquire(sim, owners, trace, output, plan=plan, sdk_steps=LAST_SDK, after=observe)
    finally:
        sim = owners = trace = None


if __name__ == "__main__":
    raise SystemExit(closure.main(
        extra_flags=("touched", "left-preload", "right-preload", "preloaded", "left-trim", "right-trim"),
        prepare_operation=prepare, acquire_operation=acquire,
    ))
