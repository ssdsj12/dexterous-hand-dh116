"""Pure control interfaces for separating Panda and DH116 action ownership."""
from __future__ import annotations

from typing import Any

import numpy as np

from .hand_only_env import HandOnlyState
from .mpc_feedback import to_numpy


class IntegratedHandMPC:
    """Adapt a hand-only MPC planner or student policy to the mounted scene."""

    def __init__(self, planner: Any) -> None:
        if not (callable(getattr(planner, "plan", None))
                or callable(getattr(planner, "plan_contact", None))
                or callable(getattr(planner, "predict", None))):
            raise TypeError("controller must expose plan_contact, plan or predict")
        self.planner = planner
        self.last_diagnostics = None

    def reset(self) -> None:
        """Reset this adapter and any planner-owned episode history."""
        self.last_diagnostics = None
        reset = getattr(self.planner, "reset", None)
        if callable(reset):
            reset()

    def plan(
        self,
        hand_poses: Any,
        active_joints: Any,
        grasp_progress: float,
        previous_action: Any,
        *,
        pad_positions: Any = None,
        pad_jacobians: Any = None,
        target_points: Any = None,
        actuator_load_offset: Any = None,
        clearances: Any = None,
        clearance_jacobians: Any = None,
        complete_observation: Any = None,
        contact_phase: str = 'close',
        held_coordinates: Any = None,
        reference_active_joints: Any = None,
        reference_tolerance_rad: Any = None,
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
        held=np.zeros(12,dtype=bool) if held_coordinates is None else np.asarray(held_coordinates)
        if held.shape!=(12,) or held.dtype.kind!='b':
            raise ValueError('held_coordinates requires 12 boolean hand-only coordinates')
        if callable(getattr(self.planner, "plan_contact", None)):
            if any(value is None for value in (pad_positions, pad_jacobians, target_points)):
                raise ValueError("contact MPC requires measured pad positions, hand Jacobians and targets")
            contact_input=dict(
                active_joints=joints, previous_action=previous,
                pad_positions=pad_positions, pad_jacobians=pad_jacobians,
                target_points=target_points)
            if actuator_load_offset is not None:contact_input['actuator_load_offset']=actuator_load_offset
            if held_coordinates is not None:contact_input['held_coordinates']=held
            if clearances is not None or clearance_jacobians is not None:
                contact_input.update(clearances=clearances,clearance_jacobians=clearance_jacobians)
            if reference_active_joints is not None or reference_tolerance_rad is not None:
                contact_input.update(reference_active_joints=reference_active_joints,
                                     reference_tolerance_rad=reference_tolerance_rad)
            if getattr(self.planner,'requires_complete_observation',False):
                contact_input.update(complete_observation=complete_observation,phase=contact_phase)
            action, self.last_diagnostics = self.planner.plan_contact(**contact_input)
            result = to_numpy(action).reshape(12)
        elif callable(getattr(self.planner, "predict", None)):
            # Student checkpoints were trained on active-joint state, previous
            # action, and grasp progress only (25 values); Panda state is never
            # part of the policy input.
            observation = np.concatenate((joints.reshape(-1), previous,
                                          np.asarray([grasp_progress], dtype=np.float32)))
            result = to_numpy(self.planner.predict(observation))
        else:
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
        # Learned policies share the measured preload handoff. This does not
        # freeze measured q or erase physical contact Jacobians from the state.
        result=np.where(held,previous,result)
        if self.last_diagnostics is not None:
            self.last_diagnostics['held_coordinates']=held.copy()
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
