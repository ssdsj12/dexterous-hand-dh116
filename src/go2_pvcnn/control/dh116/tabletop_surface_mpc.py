"""Batched receding-horizon MPC for local DH116 pad geometry.

Only the twelve finger actuator targets are optimized. The student owns the
wrists and supplies their future world-frame pad displacements. This reduced
model has no object, friction, force, or contact-switching dynamics. Recompute
points/Jacobians/targets from real observations after applying only the first
command; prediction cost reduction alone is not evidence of physical grasping.
"""
from __future__ import annotations

import math
from numbers import Real

import torch

from .linear_coupling import active_limits


class SurfaceGripMPC:
    """Deterministic full-trajectory projected-gradient shooting, batched in N.

    There is no warm history or disturbance observer: every plan is anchored to
    the caller's measured state and last *applied* command. Environments and
    hands have independent objectives, step sizes, and nominal fallback choices.
    All tensors retain q's device and float32/float64 dtype.
    """

    max_command_step = .3 / 60.

    def __init__(self, horizon: int = 4, *, iterations: int = 48,
                 point_scale: float = .001, nominal_weight: float = .02,
                 smoothness_weight: float = .002):
        if (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1
                or isinstance(iterations, bool) or not isinstance(iterations, int)
                or iterations < 1):
            raise ValueError("horizon and iterations must be positive integers")
        for name, value, allow_zero in (("point_scale", point_scale, False),
                                        ("nominal_weight", nominal_weight, False),
                                        ("smoothness_weight", smoothness_weight, True)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0
                    or (not allow_zero and value == 0)):
                raise ValueError(f"invalid {name}")
        self.horizon = horizon
        self.iterations = iterations
        self.point_scale = float(point_scale)
        self.nominal_weight = float(nominal_weight)
        self.smoothness_weight = float(smoothness_weight)

    def reset(self, ids=None) -> None:
        """Stateless: resetting any subset of environments is a no-op."""

    @torch.no_grad()
    def plan(self, q: torch.Tensor, command: torch.Tensor, nominal: torch.Tensor,
             points: torch.Tensor, jacobian: torch.Tensor,
             target_points: torch.Tensor, wrist_motion: torch.Tensor,
             weights: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
             response: float = .5) -> tuple[torch.Tensor, dict]:
        """Return (first_command[N,2,6], diagnostics) without mutating inputs.

        Shapes: q/command [N,2,6], nominal [N,H,2,6], points [N,2,D,3],
        jacobian [N,2,D,3,6], target_points/wrist_motion [N,H,2,D,3],
        weights [N,2,D], lower/upper [2,6]. D is any positive pad count.
        Points, Jacobians and cumulative wrist displacements share a frame.
        Weights are finite, nonnegative effective-contact confidence values.

        Each step intersects the caller bounds with active_limits() and applies
        |command[t]-command[t-1]| <= .005 rad, beginning at command (not q).
        If these constraints are infeasible, or inputs/model arithmetic are
        invalid, raise ValueError so the native adapter can use its nominal path.

        With load_offset = command-q, the model is
        q_next = (1-response)*q_pred + response*(command_future-load_offset).
        Pad prediction is points + J*(q_pred-q) + wrist_motion. The objective
        averages squared point errors over H and normalized effective weights
        (in point_scale units), plus nominal tracking and successive-command
        smoothness (in .005 rad units), averaged over H and six joints per hand.
        Nominal tracking uses the supplied student trajectory, before clipping.
        The clipped nominal candidate is always retained unless improved. A
        hand with no effective contacts follows that candidate exactly, even
        when smoothness alone would prefer a different student trajectory.

        Diagnostics contain the whole command trajectory and H post-command
        predictions. Costs [N] sum the two independent per-hand costs [N,2].
        They describe local pad kinematics, not full object/friction dynamics.
        """
        if (not isinstance(q, torch.Tensor) or q.dtype not in
                (torch.float32, torch.float64) or q.ndim != 3
                or q.shape[1:] != (2, 6) or q.shape[0] < 1):
            raise ValueError("q must be a floating tensor with shape [N,2,6], N > 0")
        if (not isinstance(points, torch.Tensor) or points.ndim != 4
                or points.shape[:2] != q.shape[:2] or points.shape[2] < 1
                or points.shape[-1] != 3):
            raise ValueError("points must have shape [N,2,D,3], D > 0")
        n, d, h = q.shape[0], points.shape[2], self.horizon
        for name, value, shape in (
                ("q", q, (n, 2, 6)), ("command", command, (n, 2, 6)),
                ("nominal", nominal, (n, h, 2, 6)),
                ("points", points, (n, 2, d, 3)),
                ("jacobian", jacobian, (n, 2, d, 3, 6)),
                ("target_points", target_points, (n, h, 2, d, 3)),
                ("wrist_motion", wrist_motion, (n, h, 2, d, 3)),
                ("weights", weights, (n, 2, d)),
                ("lower", lower, (2, 6)), ("upper", upper, (2, 6))):
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.dtype != q.dtype or value.device != q.device
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f"{name} must be finite with shape {shape} and q's dtype/device")
        if bool((weights < 0).any()):
            raise ValueError("weights must be nonnegative")
        if (isinstance(response, bool) or not isinstance(response, Real)
                or not math.isfinite(response) or not 0 < response <= 1):
            raise ValueError("response must be finite in (0,1]")
        response = float(response)

        # Intersect in float64, then round *inward* to preserve authoritative
        # affine follower limits when float32 commands are later read as doubles.
        physical_lower, physical_upper = active_limits()
        lo64 = torch.maximum(lower.double(), torch.as_tensor(
            physical_lower, dtype=torch.float64, device=q.device))
        hi64 = torch.minimum(upper.double(), torch.as_tensor(
            physical_upper, dtype=torch.float64, device=q.device))
        lo, hi = lo64.to(q.dtype), hi64.to(q.dtype)
        lo = torch.where(lo.double() < lo64,
                         torch.nextafter(lo, torch.full_like(lo, float("inf"))), lo)
        hi = torch.where(hi.double() > hi64,
                         torch.nextafter(hi, torch.full_like(hi, -float("inf"))), hi)
        step = self.max_command_step
        if bool((lo > hi).any()):
            raise ValueError("lower/upper have no intersection with active_limits")
        if bool(((command + step < lo) | (command - step > hi)).any()):
            raise ValueError("command cannot reach joint limits within the rate limit")

        def project(trajectory):
            previous = command
            feasible = []
            for desired in trajectory.unbind(1):
                previous = desired.clamp(torch.maximum(lo, previous - step),
                                         torch.minimum(hi, previous + step))
                feasible.append(previous)
            return torch.stack(feasible, 1)

        # Scaling before summing also handles very large finite confidences.
        largest = weights.amax(-1, keepdim=True)
        relative = weights / torch.where(largest > 0, largest, torch.ones_like(largest))
        normalized = relative / relative.sum(-1, keepdim=True).clamp_min(1.)
        has_contacts = largest.squeeze(-1) > 0

        def evaluate(trajectory):
            # Delta coordinates are algebraically the loaded recurrence above,
            # and preserve an unchanged loaded equilibrium exactly in float32.
            delta = torch.zeros_like(q)
            deltas = []
            for future in trajectory.unbind(1):
                delta = (1 - response) * delta + response * (future - command)
                deltas.append(delta)
            deltas = torch.stack(deltas, 1)
            predicted = points[:, None] + torch.einsum(
                "nsdcj,nhsj->nhsdc", jacobian, deltas) + wrist_motion
            residual = predicted - target_points
            weighted_error = (residual / self.point_scale).square().sum(-1)
            geometry = (weighted_error * normalized[:, None]).sum(-1).mean(1)
            tracking = ((trajectory - nominal) / step).square().mean((1, 3))
            increments = torch.cat((trajectory[:, :1] - command[:, None],
                                    trajectory[:, 1:] - trajectory[:, :-1]), 1)
            smoothness = (increments / step).square().mean((1, 3))
            costs = (geometry + self.nominal_weight * tracking
                     + self.smoothness_weight * smoothness)
            return costs, q[:, None] + deltas, predicted, residual, increments

        nominal_trajectory = project(nominal)
        nominal_cost, _, _, _, _ = evaluate(nominal_trajectory)
        if not bool(torch.isfinite(nominal_cost).all()):
            raise ValueError("nonfinite model cost; check state/targets/jacobian scales")
        best = nominal_trajectory
        best_cost = nominal_cost

        # Sensitivity of post-command state[t] to command[k]. Absolute command
        # targets are the decision variables, not a repeated single-step action.
        indices = torch.arange(h, device=q.device)
        lag = indices[:, None] - indices[None, :]
        sensitivity = torch.where(lag >= 0, response * (1 - response) **
                                  lag.clamp_min(0), 0.).to(q.dtype)
        # ||sensitivity||_2 <= 1; weighted Jacobian Frobenius norm bounds its
        # spatial spectral norm. This yields a separate safe scale per N/hand.
        geometry_bound = (jacobian.square().sum((-1, -2)) * normalized).sum(-1)
        lipschitz = (2 * geometry_bound / (h * self.point_scale ** 2)
                     + (2 * self.nominal_weight + 8 * self.smoothness_weight)
                     / (h * 6 * step ** 2))
        if not bool(torch.isfinite(lipschitz).all()):
            raise ValueError("nonfinite jacobian/model curvature")
        learning_rate = lipschitz.reciprocal()[:, None, :, None]

        for _ in range(self.iterations):
            _, _, _, residual, increments = evaluate(best)
            weighted = residual * normalized[:, None, :, :, None] / self.point_scale ** 2
            state_gradient = (2 / h) * torch.einsum(
                "nhsdc,nsdcj->nhsj", weighted, jacobian)
            gradient = torch.einsum("tk,ntsj->nksj", sensitivity, state_gradient)
            gradient += (2 * self.nominal_weight / (h * 6 * step ** 2)) * (best - nominal)
            smooth_gradient = torch.cat((increments[:, :-1] - increments[:, 1:],
                                         increments[:, -1:]), 1)
            gradient += (2 * self.smoothness_weight / (h * 6 * step ** 2)) * smooth_gradient
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError("nonfinite model gradient")
            proposal = project(best - learning_rate * gradient)
            candidate_cost, _, _, _, _ = evaluate(proposal)
            if not bool(torch.isfinite(candidate_cost).all()):
                raise ValueError("nonfinite optimized model cost")
            improved = has_contacts & (candidate_cost < best_cost)
            best = torch.where(improved[:, None, :, None], proposal, best)
            best_cost = torch.where(improved, candidate_cost, best_cost)

        optimized_cost, predicted_joints, predicted_points, _, _ = evaluate(best)
        return best[:, 0].clone(), dict(
            trajectory=best, predicted_joints=predicted_joints,
            predicted_points=predicted_points, nominal_trajectory=nominal_trajectory,
            nominal_cost=nominal_cost.sum(-1), optimized_cost=optimized_cost.sum(-1),
            nominal_cost_per_hand=nominal_cost, optimized_cost_per_hand=optimized_cost,
            model="local_pad_kinematics_not_object_dynamics", response=response,
            load_offset=command - q, action_dim=12, horizon=h,
            iterations=self.iterations, max_command_step=step)
