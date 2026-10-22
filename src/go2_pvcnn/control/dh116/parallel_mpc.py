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
        grasp_fraction: float = 0.65,
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
        if not np.isfinite(grasp_fraction) or not 0.0 <= grasp_fraction <= 0.75:
            raise ValueError("grasp_fraction must be finite and in [0, 0.75]")
        self.grasp_fraction = float(grasp_fraction)
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
        closure = self.lower + (self.upper - self.lower) * (0.25 + self.grasp_fraction * grasp[:, None])
        hook_mask = np.tile(np.asarray([1, 0, 1, 0, 0, 1], dtype=np.float32), 2)
        lift_bias = (lift[:, None] * 0.10 * np.tile(self.upper - self.lower, 2)) * hook_mask
        # Lift-stage extra curl is focused on thumb and middle-finger hook
        # joints.  The index/ring/little fingers hold the grasp instead of
        # continuing to squeeze the whole hand.
        desired = np.tile(closure, (1, 2)) + lift_bias
        desired = np.clip(desired, np.tile(self.lower, 2), np.tile(self.upper, 2))
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
        closure = lower + (upper - lower) * (0.25 + self.grasp_fraction * grasp.unsqueeze(1))
        hook_mask = torch.as_tensor(np.tile([1, 0, 1, 0, 0, 1], 2), device=device, dtype=torch.float32)
        lift_bias = lift.unsqueeze(1) * 0.10 * torch.tile(upper - lower, (2,)) * hook_mask
        desired = torch.cat((closure, closure), dim=1) + lift_bias
        desired = torch.clamp(desired, torch.tile(lower, (2,)), torch.tile(upper, (2,)))
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

    def plan_trajectory(
        self,
        batch_state: Any,
        *,
        grasp_steps: int,
        lift_steps: int,
        previous_action: Optional[Any] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Plan complete batched grasp/lift references before execution.

        The leading batch dimension represents independent scenes/episodes;
        the second trajectory dimension is time.  Grasp progress ramps from
        zero to one over ``grasp_steps`` and lift progress then ramps from zero
        to one over ``lift_steps``.  Each MPC step receives the preceding
        planned action as its reference, including the caller-supplied initial
        action.  Outputs have shapes ``(batch, grasp_steps + lift_steps, 12)``
        and per-step diagnostic shapes ``(batch, horizon)``.  Phase is encoded
        as 0=grasp and 1=lift so diagnostics remain portable to CUDA tensors.

        This is a batched reference-trajectory planner, not a physics-rollout
        optimizer: it deliberately builds on :meth:`plan` and keeps the whole
        trajectory on the selected device until returned.
        """
        if isinstance(grasp_steps, bool) or not isinstance(grasp_steps, int) or grasp_steps <= 0:
            raise ValueError("grasp_steps must be a positive integer")
        if isinstance(lift_steps, bool) or not isinstance(lift_steps, int) or lift_steps <= 0:
            raise ValueError("lift_steps must be a positive integer")

        state = self._state(batch_state).batched()
        use_torch = _is_torch(state.hand_poses) or self.device.startswith("cuda")
        if use_torch:
            if torch is None:  # pragma: no cover
                raise RuntimeError("torch is required for tensor trajectory planning")
            device = torch.device(self.device if self.device.startswith("cuda") else state.hand_poses.device)
            poses = torch.as_tensor(state.hand_poses, dtype=torch.float32, device=device)
            joints = torch.as_tensor(state.active_joints, dtype=torch.float32, device=device)
            previous = torch.as_tensor(
                state.previous_action if previous_action is None else previous_action,
                dtype=torch.float32, device=device,
            )
        else:
            poses = np.asarray(state.hand_poses, dtype=np.float32)
            joints = np.asarray(state.active_joints, dtype=np.float32)
            previous = np.asarray(
                state.previous_action if previous_action is None else previous_action,
                dtype=np.float32,
            )
        if previous.ndim == 1:
            previous = previous.unsqueeze(0) if use_torch else previous[None, :]
        expected = (state.batch_size, 12)
        if tuple(previous.shape) != expected:
            raise ValueError(f"previous_action must have shape (12,) or {expected}")
        finite = bool(torch.isfinite(previous).all().item()) if use_torch else bool(np.isfinite(previous).all())
        if not finite:
            raise ValueError("previous_action contains NaN or infinite values")

        actions = []
        diagnostic_rows: dict[str, list[Any]] = {"phase": [], "grasp_progress": [],
                                                  "lift_progress": [], "tracking_cost": []}
        horizon = grasp_steps + lift_steps
        for step in range(horizon):
            if step < grasp_steps:
                grasp = (step + 1) / grasp_steps
                lift = 0.0
                phase = 0
            else:
                grasp = 1.0
                lift = (step - grasp_steps + 1) / lift_steps
                phase = 1
            if use_torch:
                wire = torch.stack((torch.full((state.batch_size,), grasp, device=device),
                                    torch.full((state.batch_size,), lift, device=device)), dim=-1)
                step_state = HandOnlyState(poses, previous.reshape(-1, 2, 6), wire, previous)
            else:
                wire = np.broadcast_to(np.asarray([[grasp, lift]], dtype=np.float32),
                                       (state.batch_size, 2)).copy()
                step_state = HandOnlyState(poses, previous.reshape(-1, 2, 6), wire, previous)
            action, diagnostics = self.plan(step_state, previous)
            actions.append(action)
            diagnostic_rows["phase"].append(
                torch.full((state.batch_size,), phase, dtype=torch.int64, device=device)
                if use_torch else np.full((state.batch_size,), phase, dtype=np.int64)
            )
            for key in ("grasp_progress", "lift_progress", "tracking_cost"):
                diagnostic_rows[key].append(diagnostics[key])
            previous = action

        if use_torch:
            return torch.stack(actions, dim=1), {key: torch.stack(values, dim=1)
                                                  for key, values in diagnostic_rows.items()}
        return np.stack(actions, axis=1), {key: np.stack(values, axis=1)
                                            for key, values in diagnostic_rows.items()}


__all__ = ["ParallelJointMPC"]
