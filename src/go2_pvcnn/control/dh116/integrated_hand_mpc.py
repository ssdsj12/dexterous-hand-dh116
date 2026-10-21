"""Pure control interfaces for separating Panda and DH116 action ownership."""
from __future__ import annotations

from typing import Any

import numpy as np

from .hand_only_env import HandOnlyState
from .mpc_feedback import to_numpy


class IntegratedHandMPC:
    """Adapt the existing 12D hand planner to the mounted two-hand scene."""

    def __init__(self, planner: Any) -> None:
        if not callable(getattr(planner, "plan", None)):
            raise TypeError("planner must expose plan(state, previous_action)")
        self.planner = planner

    def plan(
        self,
        hand_poses: Any,
        active_joints: Any,
        grasp_progress: float,
        previous_action: Any,
    ) -> np.ndarray:
        """Return bounded active joint targets shaped ``(2, 6)``.

        Inputs are observations of the mounted hands. Panda coordinates are
        intentionally absent from both the state and the returned action.
        """
        poses = np.asarray(hand_poses, dtype=np.float32)
        joints = np.asarray(active_joints, dtype=np.float32)
        previous = np.asarray(previous_action, dtype=np.float32)
        if (poses.shape != (2, 7) or joints.shape != (2, 6)
                or previous.shape != (12,)
                or not np.isfinite(poses).all() or not np.isfinite(joints).all()
                or not np.isfinite(previous).all()
                or not np.isfinite(grasp_progress)
                or not 0.0 <= grasp_progress <= 1.0):
            raise ValueError("invalid integrated hand MPC observations")
        state = HandOnlyState(
            hand_poses=poses,
            active_joints=joints,
            wire_features=np.asarray([grasp_progress, 0.0], dtype=np.float32),
            previous_action=previous,
        )
        action, _diagnostics = self.planner.plan(state, previous)
        result = to_numpy(action)
        if result.shape == (1, 12):
            result = result[0]
        if result.shape != (12,) or not np.isfinite(result).all():
            raise ValueError("hand MPC must return one finite 12D active-joint target")
        return result.reshape(2, 6).astype(np.float32, copy=True)


_PHASES = {
    "approach": ("arm_approach", "hand_approach", "panda", "approach_trajectory"),
    "close": ("arm_frozen", "hand_mpc", "frozen", "hand_mpc"),
    "hold": ("arm_frozen", "hand_hold", "frozen", "hand_mpc_hold"),
    "lift": ("arm_lift", "hand_hold", "panda", "hand_mpc_hold"),
}


def route_phase_targets(
    phase: str,
    *,
    arm_approach: Any,
    arm_lift: Any,
    arm_frozen: Any,
    hand_approach: Any,
    hand_mpc: Any,
    hand_hold: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Route targets while keeping arm and hand action ownership disjoint."""
    if phase not in _PHASES:
        raise ValueError(f"unsupported control phase: {phase!r}")
    arrays = {
        "arm_approach": (arm_approach, (2, 7)),
        "arm_lift": (arm_lift, (2, 7)),
        "arm_frozen": (arm_frozen, (2, 7)),
        "hand_approach": (hand_approach, (2, 6)),
        "hand_mpc": (hand_mpc, (2, 6)),
        "hand_hold": (hand_hold, (2, 6)),
    }
    converted = {}
    for name, (value, shape) in arrays.items():
        result = np.asarray(value, dtype=np.float32)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
        converted[name] = result
    arm_name, hand_name, arm_owner, hand_owner = _PHASES[phase]
    return (converted[arm_name].copy(), converted[hand_name].copy(),
            {"arm": arm_owner, "hand": hand_owner})


__all__ = ["IntegratedHandMPC", "route_phase_targets"]
