#!/usr/bin/env python3
"""Compare bounded closed-loop left index force feedback on the same Pinch prefix."""

from functools import partial
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_pinch_force_hold_feedback import FixedWristPinchForceHold
from scripts import run_dh116_pinch_fixed_force_hold_control as baseline

seating = baseline.seating
base, prelift = seating.base, seating.prelift
VERSION = "dh116-pinch-fixed-wrist-force-feedback-v1"
CHANNEL = seating.SEATING_DOF[1]


class FeedbackAction:
    """Adapt only the retained left index channel and verify its swept interval."""

    def __init__(self, certify=baseline.CERTIFY):
        self._force_hold = FixedWristPinchForceHold(seating.PREFIX_SDK)
        self._certify = certify

    def __call__(self, index, nominal_action, rows):
        if index < seating.PREFIX_WINDOWS:
            return nominal_action, None
        command = np.asarray(nominal_action, dtype=float).copy()
        if command.shape != (24,) or np.any(command):
            raise ValueError("the bounded force experiment requires a fixed zero-action baseline")

        def certify(delta, row):
            states = {side: prelift.joint_and_wrist(row, side) for side in ("left", "right")}
            targets = {side: state[1].copy() for side, state in states.items()}
            targets["left"][CHANNEL] += delta
            lower, upper = prelift.active_limits()
            if any(np.any(target[:6] < lower) or np.any(target[:6] > upper) for target in targets.values()):
                return False
            result = self._certify(row, states, targets)
            return all(result[side]["proved"] is True for side in ("left", "right"))

        decision = self._force_hold.decide(rows, certify=certify)
        if decision["proposed_control_step"] != index:
            raise ValueError("force decision is bound to a different policy step")
        scale = float(prelift.WINDOW_SCALE[CHANNEL])
        command[CHANNEL] = decision["left_index_delta_rad"] / scale
        if abs(command[CHANNEL]) > 1.:
            raise ValueError("force preload exceeds the original action limit")
        return command, decision


def prepare(args, *, screen_geometry=False):
    plan = baseline.prepare(args, screen_geometry=screen_geometry)
    plan.update(staged_version=VERSION, preserved_scope="fixed wrists, measured left-index force preload only")
    source = Path(__file__).resolve()
    plan["frozen_files"][str(source)] = base.lifecycle.native.sha(source)
    return plan


def acquire(sim, owners, trace, output, *, plan):
    try:
        action_owner = partial(base.acquire, feedback_factory=FeedbackAction)
        with base.bindings((base, "acquire", action_owner)):
            return baseline.acquire(sim, owners, trace, output, plan=plan)
    finally:
        sim = owners = trace = None


if __name__ == "__main__":
    raise SystemExit(seating.closure.main(
        extra_flags=("touched","left-preload","right-preload","preloaded","left-trim","right-trim","seated-prefix"),
        prepare_operation=prepare, acquire_operation=acquire,
    ))
