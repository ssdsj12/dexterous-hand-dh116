#!/usr/bin/env python3
"""Test a bounded four-pad Pinch preload with both wrists held fixed."""
from pathlib import Path

import numpy as np

from scripts import run_dh116_pinch_bilateral_closure as closure


base, prelift, late, pacing = closure.base, closure.prelift, closure.late, closure.pacing
PREFIX_WINDOWS = (closure.LAST_SDK - 2) // 8
RAMP_WINDOWS = 2
SETTLE_WINDOWS = 8
LAST_SDK = closure.LAST_SDK + 8 * (RAMP_WINDOWS + SETTLE_WINDOWS)
WALL_S = 300
OUTER_SHA = "1b7ee7c43de04f74dd5d6c92460c9ceec731f62c1a859641f3f92f206441c4a5"
LEDGER_SHA = "ddf53e94dfa6cc95c34c107deee0b3f54161b1022a53524ff7a8cd2b57ff3382"
GOAL_SHA = {
    "left": "f16be41eb58684e07a75aaac10024a685ea6dbeab91bf5831744d352e216ee9b",
    "right": "66ee7c23ca592873a0150caeba56cc90a1ed82b5bda1fa15d8868277ff226ee6",
}


def compile_preload(previous, states, goals):
    for side in ("left", "right"):
        issued = prelift._finite(states[side][1], (12,))
        if np.any(prelift._finite(goals[side], (2,)) <= issued[1:3]):
            raise ValueError("positive preload on all selected fingers is required")
    return closure.compile_closure(
        previous, states, goals, prefix_windows=PREFIX_WINDOWS, closing_windows=RAMP_WINDOWS,
        settle_windows=SETTLE_WINDOWS,
    )


def boundary_has_opposition(row, *, sdk_step=closure.LAST_SDK):
    if row["physics_step"] != sdk_step or row["guard_failure"] is not None:
        return False
    records = row["stage_reaction"]["per_substep"]
    return [sample["substep"] for sample in records] == list(range(base.SUBSTEPS)) and all(
        sample[side]["signed_opposition"] is True for sample in records for side in ("left", "right")
    )


def prepare(args, *, screen_geometry=False):
    plan = closure.prepare(args)
    archive, row = pacing.archive_boundary(
        args.touched, outer_sha=OUTER_SHA, ledger_sha=LEDGER_SHA, sdk_steps=closure.LAST_SDK,
        windows=PREFIX_WINDOWS, contact_guard=closure.guard_selected_contact,
    )
    if plan["action_schedule"] != archive["action_schedule"] or not boundary_has_opposition(row):
        raise ValueError("sealed four-pad closure boundary is required for preload")
    goals = {}
    for side, name in (("left", args.left_preload), ("right", args.right_preload)):
        item = base._checked(name, GOAL_SHA[side])[1]
        if (item["side"] != side or item["sdk"] != closure.LAST_SDK or item["depth_mm"] != .085
                or item["proved"] is not True):
            raise ValueError("exact screened preload source is required for each hand")
        goals[side] = item["target"][1:3]
        plan["frozen_files"][str(name)] = GOAL_SHA[side]
    states = {side: prelift.joint_and_wrist(row, side) for side in goals}
    actions, targets = compile_preload(plan["action_schedule"], states, goals)
    if screen_geometry:
        proofs = closure.certify_closure(row, states, targets)
        if not all(proof["proved"] for proof in proofs.values()):
            raise ValueError({side: proof.get("unresolved_shape") for side, proof in proofs.items()})
    for path in (args.touched / "outer.json", args.touched / "native" / "physical.jsonl",
                 args.touched / "native" / "frozen_plan.json", Path(__file__).resolve()):
        plan["frozen_files"][str(path)] = base.lifecycle.native.sha(path)
    plan.update(
        staged_version="dh116-pinch-bilateral-preload-v1", action_schedule=actions.tolist(),
        preload=dict(prefix=late.prefix_measurements(row, expected_sdk=closure.LAST_SDK), goals=goals),
        preserved_scope="bounded bilateral finger preload and fixed wrist settling, no lift or PPO",
    )
    plan["budget"].update(
        control_steps=len(actions), sdk_steps=LAST_SDK, soft_wall_seconds=WALL_S,
        solver_steps=(LAST_SDK - 2) * base.SUBSTEPS,
    )
    return plan


def measure_gate(row, plan, *, sdk_step, prefix_windows, config_key, gate_key, compiler):
    receipt = row[gate_key] = dict(passed=False)
    try:
        config = plan[config_key]
        measured = late.prefix_measurements(row, expected_sdk=sdk_step)
        receipt["prefix"] = late.compare_prefix(measured, config["prefix"])
        receipt["signed_opposition"] = boundary_has_opposition(row, sdk_step=sdk_step)
        states = {side: prelift.joint_and_wrist(row, side) for side in config["goals"]}
        actions, targets = compiler(plan["action_schedule"][:prefix_windows], states, config["goals"])
        receipt["same_actions"] = bool(np.allclose(
            actions, plan["action_schedule"][:len(actions)], atol=1e-7, rtol=0,
        ))
        if receipt["prefix"]["passed"] and receipt["signed_opposition"] and receipt["same_actions"]:
            receipt["proofs"] = closure.certify_closure(row, states, targets)
            receipt["passed"] = all(proof["proved"] for proof in receipt["proofs"].values())
        if not receipt["passed"]:
            row["guard_failure"] = f"pinch_{gate_key}_rejected"
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        prefix = gate_key.removesuffix("_gate")
        row["guard_failure"] = f"pinch_{prefix}_source_invalid"


def acquire(sim, owners, trace, output, *, plan, sdk_steps=LAST_SDK, after=None):
    def observe(row):
        if row["physics_step"] == closure.LAST_SDK:
            measure_gate(row, plan, sdk_step=closure.LAST_SDK, prefix_windows=PREFIX_WINDOWS,
                         config_key="preload", gate_key="bilateral_preload_gate", compiler=compile_preload)
        elif after is not None:
            after(row)

    try:
        return closure.acquire(sim, owners, trace, output, plan=plan, sdk_steps=sdk_steps, after=observe)
    finally:
        sim = owners = trace = None


if __name__ == "__main__":
    raise SystemExit(closure.main(extra_flags=("touched", "left-preload", "right-preload"),
                                  prepare_operation=prepare, acquire_operation=acquire))
