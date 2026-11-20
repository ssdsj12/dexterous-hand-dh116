#!/usr/bin/env python3
"""Check left Pinch longitudinal seating with bounded bilateral pad preload."""
from pathlib import Path

import numpy as np

from scripts import run_dh116_pinch_asymmetric_preload as prior


preload, closure, prelift = prior.preload, prior.closure, prior.prelift
base, pacing = prior.base, prior.pacing
LEFT_SHA = "41474626a1a4321d9c6412ca9026776ec8a0c396c430de756d76ccd984c37699"
LEFT_INDEX = 4
MAX_SHIFT_M = .00016
GATE_KEY = "pinch_longitudinal_preload"
LAST_SDK = prior.LAST_SDK


def compile_longitudinal(previous, states, goals):
    prefix = prelift._finite(previous, (prior.PREFIX_WINDOWS, 24))
    commands = {side:prelift._finite(states[side][1], (12,)) for side in ("left", "right")}
    left, right = (commands[side].copy() for side in ("left", "right"))
    left[[1, 7]] = prelift._finite(goals["left"], (2,))
    right[1:3] = prelift._finite(goals["right"], (2,))
    dl, dr = left - commands["left"], right - commands["right"]
    if not 0. < dl[7] <= MAX_SHIFT_M + 1e-12:
        raise ValueError("bounded positive longitudinal left wrist translation required")
    if dl[1] <= 0 or np.any(dr[1:3] <= 0):
        raise ValueError("positive closing target for each selected digit required")
    lo, hi = prelift.active_limits()
    if any(np.any(q[:6] < lo) or np.any(q[:6] > hi) for q in (left, right)):
        raise ValueError("preload exceeds the original DH116 joint limits")
    ramp = prelift.global_channel_order(dl, dr) / (prelift.WINDOW_SCALE * preload.RAMP_WINDOWS)
    if np.any(np.abs(ramp) > 1.):
        raise ValueError("longitudinal action exceeds the original rate bounds")
    actions = np.vstack((prefix, np.tile(ramp, (preload.RAMP_WINDOWS, 1)),
                         np.zeros((preload.SETTLE_WINDOWS, 24))))
    return actions, {"left": left, "right": right}


def prepare(args, *, screen_geometry=False):
    plan = preload.prepare(args)
    archive, row = pacing.archive_boundary(
        args.preloaded, outer_sha=prior.OUTER_SHA, ledger_sha=prior.LEDGER_SHA,
        sdk_steps=preload.LAST_SDK, windows=prior.PREFIX_WINDOWS, contact_guard=closure.guard_selected_contact,
    )
    if plan["action_schedule"] != archive["action_schedule"] or not preload.boundary_has_opposition(
            row, sdk_step=preload.LAST_SDK):
        raise ValueError("sealed fixed-wrist preload boundary required")
    left_data = base._checked(args.left_longitudinal, LEFT_SHA)
    left = left_data["rows"][LEFT_INDEX]
    i, d, digest = prior.SOURCES["right"]
    right = base._checked(args.right_trim, digest)[i]
    if (left_data["sdk"] != preload.LAST_SDK or left["shift_m"] != [0., .00012, 0.]
            or left["depth_m"] != .00012 or left["thumb_target_proof"]["proved"] is not True
            or right["side"] != "right" or right["sdk"] != preload.LAST_SDK or right["depth_mm"] != d
            or right["proved"] is not True):
        raise ValueError("exact side-specific longitudinal preload evidence required")
    goals = {"left": [left["thumb_target"][1], left["thumb_target"][7]], "right": right["target"][1:3]}
    states = {side:prelift.joint_and_wrist(row, side) for side in goals}
    actions, targets = compile_longitudinal(plan["action_schedule"], states, goals)
    if screen_geometry:
        proofs = closure.certify_closure(row, states, targets)
        if not all(proof["proved"] for proof in proofs.values()):
            raise ValueError({side: proof.get("unresolved_shape") for side, proof in proofs.items()})
    plan["frozen_files"].update({str(args.left_longitudinal): LEFT_SHA, str(args.right_trim): digest})
    for path in (args.preloaded / "outer.json", args.preloaded / "native" / "physical.jsonl",
                 args.preloaded / "native" / "frozen_plan.json", Path(__file__).resolve()):
        plan["frozen_files"][str(path)] = base.lifecycle.native.sha(path)
    plan.update(staged_version="dh116-pinch-longitudinal-preload-v1", action_schedule=actions.tolist(),
                preserved_scope="left longitudinal wrist seating and pad preloads, no lift or PPO")
    plan[GATE_KEY] = dict(prefix=preload.late.prefix_measurements(row, expected_sdk=preload.LAST_SDK), goals=goals)
    plan["budget"].update(control_steps=len(actions), sdk_steps=LAST_SDK, soft_wall_seconds=360,
                          solver_steps=(LAST_SDK - 2) * base.SUBSTEPS)
    return plan


def acquire(sim, owners, trace, output, *, plan):
    def observe(row):
        if row["physics_step"] == preload.LAST_SDK:
            preload.measure_gate(row, plan, sdk_step=preload.LAST_SDK, prefix_windows=prior.PREFIX_WINDOWS,
                                 config_key=GATE_KEY, gate_key="longitudinal_preload_gate",
                                 compiler=compile_longitudinal)
    try:
        return preload.acquire(sim, owners, trace, output, plan=plan, sdk_steps=LAST_SDK, after=observe)
    finally:
        sim = owners = trace = None


if __name__ == "__main__":
    raise SystemExit(closure.main(
        extra_flags=("touched", "left-preload", "right-preload", "preloaded", "left-longitudinal", "right-trim"),
        prepare_operation=prepare, acquire_operation=acquire,
    ))
