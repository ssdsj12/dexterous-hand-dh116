"""State and action contract for DH116 hand-only control.

The fixed Panda arms are deliberately absent from this module.  Each hand has
six authoritative DH116 actuator coordinates; the five affine follower joints
are derived with :func:`linear_coupling.expand` by consumers that need the
full eleven-joint state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .linear_coupling import MIMIC, coupling_matrix


try:  # Torch is optional for the CPU/reference install.
    import torch
except ImportError:  # pragma: no cover - exercised in minimal CPU installs
    torch = None


Array = Any


def _is_torch(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _shape(value: Array) -> tuple[int, ...]:
    return tuple(int(v) for v in value.shape)


def _finite(value: Array) -> bool:
    if _is_torch(value):
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(np.asarray(value)).all())


def _as_array(value: Array, *, name: str) -> Array:
    if _is_torch(value):
        if not (value.is_floating_point() or value.is_complex() is False):
            raise ValueError(f"{name} must be a real floating-point array")
        out = value
    else:
        out = np.asarray(value, dtype=np.float32)
    if not _finite(out):
        raise ValueError(f"{name} contains NaN or infinite values")
    return out


@dataclass(frozen=True)
class HandOnlySpec:
    """Static dimensions and phase settings for the hand-only task.

    ``action_dim`` and ``previous_action_dim`` are intentionally fixed at 12:
    six active DH116 joints for each hand.  They are fields rather than module
    constants so a serialized task manifest can carry the contract explicitly,
    but non-12 values are rejected at construction.
    """

    action_dim: int = 12
    previous_action_dim: int = 12
    hands: int = 2
    active_joints_per_hand: int = 6
    pose_dim: int = 7  # position (3) + quaternion (4)
    wire_feature_dim: int = 2  # grasp progress, lift progress
    grasp_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.action_dim != 12 or self.previous_action_dim != 12:
            raise ValueError("hand-only DH116 contract requires 12D actions")
        if self.hands != 2 or self.active_joints_per_hand != 6:
            raise ValueError("hand-only DH116 contract requires two six-joint hands")
        if self.pose_dim <= 0 or self.wire_feature_dim != 2:
            raise ValueError("invalid hand-only feature dimensions")
        if not np.isfinite(self.grasp_threshold):
            raise ValueError("grasp_threshold must be finite")


@dataclass(frozen=True, init=False)
class HandOnlyState:
    """A single or batched hand-only state.

    Arrays may be NumPy arrays or torch tensors.  Unbatched values have shapes
    ``(2, 7)``, ``(2, 6)``, ``(2,)`` and ``(12,)`` respectively; batched values
    add a leading batch dimension.  ``hand_poses`` uses position/quaternion
    rows, while ``wire_features[..., 0]`` and ``wire_features[..., 1]`` are
    grasp and lift progress in ``[0, 1]`` (the planner clips values outside
    that range, but does not mutate the state).

    ``left_pose``/``right_pose`` and ``wire_grasp``/``wire_lift`` keyword
    aliases are accepted to make callers explicit without introducing Panda
    arm state into the contract.
    """

    hand_poses: Array
    active_joints: Array
    wire_features: Array
    previous_action: Array

    def __init__(
        self,
        hand_poses: Optional[Array] = None,
        active_joints: Optional[Array] = None,
        wire_features: Optional[Array] = None,
        previous_action: Optional[Array] = None,
        *,
        left_pose: Optional[Array] = None,
        right_pose: Optional[Array] = None,
        wire_grasp: Optional[Array] = None,
        wire_lift: Optional[Array] = None,
    ) -> None:
        if hand_poses is None:
            if left_pose is None or right_pose is None:
                raise ValueError("provide hand_poses or both left_pose and right_pose")
            left = _as_array(left_pose, name="left_pose")
            right = _as_array(right_pose, name="right_pose")
            if _shape(left) != _shape(right) or _shape(left)[-1:] != (7,):
                raise ValueError("left_pose and right_pose must have matching (...,7) shapes")
            hand_poses = torch.stack((left, right), dim=-2) if _is_torch(left) else np.stack((left, right), axis=-2)
        if active_joints is None:
            raise ValueError("active_joints is required")
        if previous_action is None:
            raise ValueError("previous_action is required")
        if wire_features is None:
            if wire_grasp is None or wire_lift is None:
                raise ValueError("provide wire_features or both wire_grasp and wire_lift")
            grasp = _as_array(wire_grasp, name="wire_grasp")
            lift = _as_array(wire_lift, name="wire_lift")
            if _shape(grasp) != _shape(lift):
                raise ValueError("wire_grasp and wire_lift must have matching shapes")
            wire_features = torch.stack((grasp, lift), dim=-1) if _is_torch(grasp) else np.stack((grasp, lift), axis=-1)

        poses = _as_array(hand_poses, name="hand_poses")
        joints = _as_array(active_joints, name="active_joints")
        wire = _as_array(wire_features, name="wire_features")
        previous = _as_array(previous_action, name="previous_action")
        if _shape(poses)[-2:] != (2, 7):
            raise ValueError("hand_poses must have shape (2,7) or (batch,2,7)")
        if _shape(joints)[-2:] != (2, 6):
            raise ValueError("active_joints must have shape (2,6) or (batch,2,6)")
        if _shape(wire)[-1:] != (2,):
            raise ValueError("wire_features must have shape (2,) or (batch,2)")
        if _shape(previous)[-1:] != (12,):
            raise ValueError("previous_action must have shape (12,) or (batch,12)")
        batch_shapes = [_shape(poses)[:-2], _shape(joints)[:-2], _shape(wire)[:-1], _shape(previous)[:-1]]
        # Every field is either unbatched or shares one leading batch size.
        sizes = [s[0] for s in batch_shapes if s]
        if sizes and any(size != sizes[0] for size in sizes):
            raise ValueError("state fields have inconsistent batch dimensions")
        backends = {_is_torch(v) for v in (poses, joints, wire, previous)}
        if len(backends) != 1:
            raise ValueError("state fields must all be NumPy arrays or all torch tensors")
        object.__setattr__(self, "hand_poses", poses)
        object.__setattr__(self, "active_joints", joints)
        object.__setattr__(self, "wire_features", wire)
        object.__setattr__(self, "previous_action", previous)

    @property
    def batch_size(self) -> int:
        """Number of parallel environments (one for an unbatched state)."""
        return int(self.previous_action.shape[0]) if self.previous_action.ndim == 2 else 1

    @property
    def left_pose(self) -> Array:
        return self.hand_poses[..., 0, :]

    @property
    def right_pose(self) -> Array:
        return self.hand_poses[..., 1, :]

    @property
    def grasp_progress(self) -> Array:
        return self.wire_features[..., 0]

    @property
    def lift_progress(self) -> Array:
        return self.wire_features[..., 1]

    @property
    def full_joints(self) -> Array:
        """Return all eleven DH116 joints with affine followers expanded.

        The returned shape is ``(..., 2, 11)`` and is derived solely from the
        authoritative six-joint ``active_joints`` state.  It is an observation
        helper; the policy action remains 12D.
        """
        if _is_torch(self.active_joints):
            matrix = torch.as_tensor(coupling_matrix(), device=self.active_joints.device, dtype=self.active_joints.dtype)
            offsets = torch.zeros(11, device=self.active_joints.device, dtype=self.active_joints.dtype)
            from .linear_coupling import JOINTS
            for follower, _, _, offset in MIMIC:
                offsets[JOINTS.index(follower)] = offset
            return torch.matmul(self.active_joints, matrix.T) + offsets
        matrix = coupling_matrix()
        offsets = np.zeros(11, dtype=self.active_joints.dtype)
        from .linear_coupling import JOINTS
        for follower, _, _, offset in MIMIC:
            offsets[JOINTS.index(follower)] = offset
        return np.matmul(self.active_joints, matrix.T) + offsets

    @property
    def expanded_joints(self) -> Array:
        """Alias for :attr:`full_joints` used by trajectory collectors."""
        return self.full_joints

    def batched(self) -> "HandOnlyState":
        """Return a state with an explicit leading batch dimension."""
        if self.previous_action.ndim == 2:
            return self
        if _is_torch(self.hand_poses):
            expand = lambda x: x.unsqueeze(0)
        else:
            expand = lambda x: np.expand_dims(x, 0)
        return HandOnlyState(expand(self.hand_poses), expand(self.active_joints), expand(self.wire_features), expand(self.previous_action))


class HandOnlyEnv:
    """Tiny deterministic state holder used by offline MPC collection.

    This is intentionally not a physics environment.  It only advances the
    hand-only action/previous-action contract and is useful for smoke tests.
    """

    def __init__(self, spec: Optional[HandOnlySpec] = None) -> None:
        self.spec = spec or HandOnlySpec()
        self.state: Optional[HandOnlyState] = None

    def reset(self, state: HandOnlyState) -> HandOnlyState:
        if not isinstance(state, HandOnlyState):
            raise TypeError("reset expects HandOnlyState")
        self.state = state
        return state

    def step(self, action: Array) -> HandOnlyState:
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        action = _as_array(action, name="action")
        if _shape(action) != _shape(self.state.previous_action):
            raise ValueError("action must match previous_action shape")
        self.state = HandOnlyState(
            self.state.hand_poses,
            self.state.active_joints,
            self.state.wire_features,
            action,
        )
        return self.state


__all__ = ["HandOnlyEnv", "HandOnlySpec", "HandOnlyState"]
