"""Deterministic batched joint-space MPC references for the DH116 hands.

The planner is intentionally a small reference generator rather than a physics
solver.  It closes the six active joints on each hand, then raises the target
as lift progress increases.  Passive joints are never part of the action and
Panda state is not accepted or consulted.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .hand_only_env import HandOnlySpec, HandOnlyState, _is_torch
from .linear_coupling import active_limits

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class ParallelJointMPC:
    """Generate bounded active-joint references for a batch of hand states.

    Parameters are deliberately deterministic.  ``device`` may be ``cpu`` or
    a CUDA device; requesting CUDA without a CUDA-enabled torch installation
    raises a clear ``RuntimeError`` instead of silently falling back to CPU.
    """

    def __init__(
        self,
        spec: Optional[HandOnlySpec] = None,
        *,
        device: str = "cpu",
        smoothing: float = 0.25,
        grasp_threshold: Optional[float] = None,
    ) -> None:
        self.spec = spec or HandOnlySpec()
        self.device = str(device)
        if self.device.startswith("cuda"):
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError("CUDA requested for ParallelJointMPC but CUDA is unavailable")
        elif self.device != "cpu":
            raise ValueError("device must be 'cpu' or a CUDA device such as 'cuda:0'")
        if not np.isfinite(smoothing) or not 0.0 <= smoothing <= 1.0:
            raise ValueError("smoothing must be finite and in [0,1]")
        self.smoothing = float(smoothing)
        self.grasp_threshold = float(self.spec.grasp_threshold if grasp_threshold is None else grasp_threshold)
        if not np.isfinite(self.grasp_threshold):
            raise ValueError("grasp_threshold must be finite")
        lower, upper = active_limits()
        self.lower = lower.astype(np.float32)
        self.upper = upper.astype(np.float32)

    @staticmethod
    def _state(value: Any) -> HandOnlyState:
        if isinstance(value, HandOnlyState):
            return value
        if isinstance(value, Mapping):
            allowed = {"hand_poses", "active_joints", "wire_features", "previous_action"}
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"unknown hand-only state fields: {sorted(unknown)}")
            try:
                return HandOnlyState(**value)
            except TypeError as exc:
                raise ValueError("state mapping must contain hand_poses, active_joints, wire_features, previous_action") from exc
        raise TypeError("batch_state must be HandOnlyState or a state mapping")

    def _numpy_plan(self, state: HandOnlyState, previous: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        batched = state.batched()
        wire = np.asarray(batched.wire_features, dtype=np.float32)
        grasp = np.clip(wire[:, 0], 0.0, 1.0)
        lift = np.clip(wire[:, 1], 0.0, 1.0)
        # Grasp closes from the lower limit toward a stable closure.  Lift
        # progress adds a small symmetric bias while preserving bounds.
        closure = self.lower + (self.upper - self.lower) * (0.25 + 0.65 * grasp[:, None])
        lift_bias = (lift[:, None] * 0.10 * (self.upper - self.lower))
        desired_hand = np.clip(closure + lift_bias, self.lower, self.upper)
        desired = np.concatenate((desired_hand, desired_hand), axis=1)
        targets = (1.0 - self.smoothing) * desired + self.smoothing * previous
        lower12 = np.tile(self.lower, 2)
        upper12 = np.tile(self.upper, 2)
        targets = np.clip(targets, lower12, upper12).astype(np.float32, copy=False)
        phase = np.where(grasp >= self.grasp_threshold, np.where(lift > 0.0, "lift", "grasp"), "grasp")
        diagnostics = {
            "phase": phase,
            "grasp_progress": grasp,
            "lift_progress": lift,
            "tracking_cost": np.sum((targets - previous) ** 2, axis=1).astype(np.float32),
        }
        return targets, diagnostics

    def _torch_plan(self, state: HandOnlyState, previous: Any) -> tuple[Any, dict[str, Any]]:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for tensor MPC planning")
        batched = state.batched()
        device = torch.device(self.device)
        wire = batched.wire_features.to(device=device, dtype=torch.float32)
        prev = previous.to(device=device, dtype=torch.float32)
        grasp = torch.clamp(wire[:, 0], 0.0, 1.0)
        lift = torch.clamp(wire[:, 1], 0.0, 1.0)
        lower = torch.as_tensor(self.lower, device=device)
        upper = torch.as_tensor(self.upper, device=device)
        closure = lower + (upper - lower) * (0.25 + 0.65 * grasp.unsqueeze(1))
        desired_hand = torch.clamp(closure + lift.unsqueeze(1) * 0.10 * (upper - lower), lower, upper)
        desired = torch.cat((desired_hand, desired_hand), dim=1)
        targets = (1.0 - self.smoothing) * desired + self.smoothing * prev
        targets = torch.clamp(targets, torch.tile(lower, (2,)), torch.tile(upper, (2,)))
        phase = torch.where(grasp >= self.grasp_threshold, torch.where(lift > 0.0, torch.ones_like(lift), torch.zeros_like(lift)), torch.zeros_like(lift))
        diagnostics = {
            "phase": phase,  # 0=grasp, 1=lift; numeric for device portability
            "grasp_progress": grasp,
            "lift_progress": lift,
            "tracking_cost": torch.sum((targets - prev) ** 2, dim=1),
        }
        return targets, diagnostics

    def plan(self, batch_state: Any, previous_action: Optional[Any] = None) -> tuple[Any, dict[str, Any]]:
        """Return ``(batch, 12)`` bounded targets and per-batch diagnostics.

        A single state is promoted to a one-element batch.  ``previous_action``
        overrides the state's copy and may be either ``(12,)`` or ``(batch,12)``.
        Return type follows the state (NumPy or torch); torch CUDA is selected
        explicitly through the planner's ``device`` argument.
        """
        state = self._state(batch_state)
        batched = state.batched()
        prev = batched.previous_action if previous_action is None else previous_action
        if _is_torch(prev):
            if prev.ndim == 1:
                prev = prev.unsqueeze(0)
            if tuple(prev.shape) != tuple(batched.previous_action.shape):
                raise ValueError("previous_action must have shape (12,) or (batch,12)")
            if not bool(torch.isfinite(prev).all().item()):
                raise ValueError("previous_action contains NaN or infinite values")
            return self._torch_plan(batched, prev)
        prev_np = np.asarray(prev, dtype=np.float32)
        if prev_np.ndim == 1:
            prev_np = prev_np[None, :]
        if prev_np.shape != tuple(batched.previous_action.shape):
            raise ValueError("previous_action must have shape (12,) or (batch,12)")
        if not np.isfinite(prev_np).all():
            raise ValueError("previous_action contains NaN or infinite values")
        if self.device.startswith("cuda"):
            # This branch is only reachable with a NumPy state and a CUDA
            # planner; make the backend conversion explicit.
            if torch is None:
                raise RuntimeError("CUDA requested for ParallelJointMPC but torch is unavailable")
            tensor_state = HandOnlyState(
                torch.as_tensor(batched.hand_poses, device=self.device),
                torch.as_tensor(batched.active_joints, device=self.device),
                torch.as_tensor(batched.wire_features, device=self.device),
                torch.as_tensor(batched.previous_action, device=self.device),
            )
            return self._torch_plan(tensor_state, torch.as_tensor(prev_np, device=self.device))
        return self._numpy_plan(batched, prev_np)


__all__ = ["ParallelJointMPC"]
