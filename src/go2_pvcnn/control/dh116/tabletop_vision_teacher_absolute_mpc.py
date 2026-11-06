"""Opt-in training expert: absolute Teacher target tracking with bounded MPC residual.

The reference is reconstructed once from cold-reset targets, the canonical 60 Hz
velocity sequence and original affine joint limits. It is never re-anchored to
MPC-modified targets. This model still has no object/contact/force dynamics;
native qualification and the default local-pad expert remain unchanged.
"""
from __future__ import annotations

import math
from numbers import Real

import torch

from .linear_coupling import active_limits
from .tabletop_surface_contract import validate_actions
from .tabletop_surface_mpc import SurfaceGripMPC
from .tabletop_surface_mpc_runtime import NativeSurfaceMPC, action_from_joint_target
from .tabletop_vision_teacher_mpc import TeacherReferenceMPC, _json_value


class _ReferenceBoundedPlanner(SurfaceGripMPC):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def plan(self, **kwargs):
        return super().plan(**kwargs, reference_radius=self.radius)


class _AbsoluteNativeSurfaceMPC(NativeSurfaceMPC):
    def __init__(self, env, radius):
        super().__init__(env)
        self.planner = _ReferenceBoundedPlanner(radius)
        self.reference_trajectory = None
        self.alive = torch.ones(env.world.n, device=self.pending.device, dtype=torch.bool)

    def _nominal_trajectory(self, ids, command, actions, lower, upper):
        return self.reference_trajectory[ids].to(command)

    def _refine_ids(self, ids, nominal, result, failed):
        ids = ids[self.alive[ids]]
        if ids.numel():
            super()._refine_ids(ids, nominal, result, failed)


class AbsoluteTeacherReferenceMPC(TeacherReferenceMPC):
    """Explicit experimental profile; source tensor is [N,T,24], zero-padded.

    Call once per sequential context['tick'], starting at zero. ``alive`` marks
    rows still executing their source episode. Ended rows cannot be re-admitted
    after a native reset. The final source target is held beyond its horizon.
    Fallback pursues the absolute next Teacher target at the original rate; an
    unreachable trust band does not permit an instantaneous command jump.
    """

    def __init__(self, env, source_sha, reference_actions, *, trust_radius_rad=.001):
        if (isinstance(trust_radius_rad, bool) or not isinstance(trust_radius_rad, Real)
                or not math.isfinite(trust_radius_rad) or trust_radius_rad <= 0):
            raise ValueError("trust_radius_rad must be finite and positive")
        super().__init__(env, source_sha)
        n = env.world.n
        if (not isinstance(reference_actions, torch.Tensor) or reference_actions.ndim != 3
                or reference_actions.shape[0] != n or reference_actions.shape[1] < 1
                or reference_actions.shape[2] != 24):
            raise ValueError("reference sequence must have shape [N,T,24], T > 0")
        validate_actions(reference_actions.reshape(-1, 24))
        command = env.task.command
        if (command.shape != (n, 12) or not command.is_floating_point()
                or not bool(torch.isfinite(command).all())):
            raise ValueError("finite initial absolute Teacher command required")
        self._trust_radius_rad = float(trust_radius_rad)
        self.native = _AbsoluteNativeSurfaceMPC(env, self._trust_radius_rad)
        self.reference_actions = reference_actions.detach().to(command.device).clone()
        low, high = (command.new_tensor(v).repeat(2) for v in active_limits())
        low = low.maximum(env.task.low); high = high.minimum(env.task.high)
        if (not torch.isfinite(low).all() or not torch.isfinite(high).all()
                or bool((low > high).any()) or bool(((command < low) | (command > high)).any())):
            raise ValueError("initial absolute Teacher command must satisfy original joint limits")
        current = command.detach().clone()
        trajectory = []
        for action in self.reference_actions.unbind(1):
            # Same double-precision substep summation as canonical native control.
            for _ in range(8):
                current = (current + action[:, :12].to(current) * (.3 / 480)).maximum(low).minimum(high)
            trajectory.append(current.clone())
        self.absolute_commands = torch.stack(trajectory, 1).reshape(n, -1, 2, 6)
        self.next_tick = 0
        self.ended = torch.zeros(n, device=command.device, dtype=torch.bool)

    @property
    def trust_radius_rad(self):
        return self._trust_radius_rad

    @torch.no_grad()
    def refine(self, reference_actions, *, context=None, alive=None):
        if (not isinstance(context, dict) or not isinstance(context.get("tick"), int)
                or isinstance(context["tick"], bool) or context["tick"] != self.next_tick):
            raise ValueError("explicit sequential integer context tick required")
        tick = context["tick"]
        validate_actions(reference_actions)
        command = self.env.task.command
        n = command.shape[0]
        if reference_actions.shape != (n, 24) or reference_actions.device != command.device:
            raise ValueError("reference actions must match native batch/device")
        if alive is None:
            alive = ~self.ended
        if (not isinstance(alive, torch.Tensor) or alive.shape != (n,)
                or alive.dtype != torch.bool or alive.device != command.device):
            raise ValueError("alive must be a native-device boolean environment mask")
        if bool((alive & self.ended).any()):
            raise ValueError("ended source episodes cannot be restarted after native reset")
        expected = (self.reference_actions[:, tick] if tick < self.reference_actions.shape[1]
                    else torch.zeros_like(reference_actions))
        if not torch.equal(reference_actions[alive], expected[alive].to(reference_actions)):
            raise ValueError("reference actions do not match source sequence at tick")
        self._contexts(context)  # Validate identity before changing any planner state.
        indices = torch.arange(tick, tick + self.native.horizon, device=command.device)
        indices = indices.clamp_max(self.absolute_commands.shape[1] - 1)
        trajectory = self.absolute_commands[:, indices]
        finite = torch.isfinite(command).all(-1)
        safe_command = torch.where(finite[:, None], command, torch.zeros_like(command))
        fallback = action_from_joint_target(reference_actions, safe_command.reshape(n, 2, 6),
                                           trajectory[:, 0])
        fallback[:, :12] = torch.where(finite[:, None], fallback[:, :12], 0.)
        fallback[~alive] = 0.
        self.native.reference_trajectory = trajectory
        self.native.alive = alive & finite
        issued, diagnostics = super().refine(fallback, context=context)
        issued[~alive] = 0.
        self.native.reset((~alive).nonzero().flatten())
        # Bind prediction to the encoded 24D action that is actually issued,
        # including action dtype rounding, rather than an unencoded solve target.
        if self.native.last_prediction is not None:
            mask = self.native.pending
            delta = issued[:, :12].to(command).reshape(n, 2, 6) * .005
            predicted = self.env.task.m["q"].reshape(n, 2, 6) + .5 * delta
            self.native.last_prediction[mask] = predicted[mask]
        for index, row in enumerate(diagnostics["per_env"]):
            row["reference_actions"] = _json_value(reference_actions[index])
            row["absolute_tracking_actions"] = _json_value(fallback[index])
            row["issued_actions"] = _json_value(issued[index])
            row["action_delta"] = _json_value(issued[index] - reference_actions[index])
            row["changed"] = bool((issued[index] - reference_actions[index]).abs().max() > 1e-6)
            row["absolute_reference_trajectory_rad"] = _json_value(trajectory[index])
            row["absolute_target_error_rad"] = _json_value(
                command[index].reshape(2, 6) - trajectory[index, 0])
            if not bool(alive[index]):
                row.update(active=False, changed=False, reason="inactive_episode")
            elif not bool(finite[index]):
                row.update(active=False, reason="invalid_absolute_command")
        diagnostics.update(version="dh116-vision-teacher-absolute-reference-mpc-v1",
            profile="absolute_reference", trust_radius_rad=self.trust_radius_rad,
            reference_role="cold_reset_absolute_teacher_command_sequence",
            reference_reconstruction="canonical_480Hz_substep_rate_and_original_joint_limits",
            reference_guard_limit="measured_state_joint_lead_is_not_reconstructed",
            fallback="rate_limited_absolute_teacher_next_target",
            local_solver_change_statistics="relative_to_absolute_tracking_action",
            prediction_scope="encoded_requested_target_before_native_state_dependent_guard",
            force_prediction=False, experimental=True)
        self.next_tick += 1
        return issued, diagnostics

    @torch.no_grad()
    def observe_result(self, done):
        result = super().observe_result(done)
        self.ended |= done
        return result
