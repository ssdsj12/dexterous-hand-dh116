"""Privileged Teacher-reference optimization for training-label collection.

The canonical 24D Teacher action is the nominal reference at the actual state
being collected. The existing local pad MPC may change twelve finger channels;
all twelve Teacher wrist channels remain unchanged. This is neither full object
dynamics nor a Teacher policy evaluated at arbitrary Student-visited states.
No controller is installed, simulation stepped, or PPO storage/optimizer used.
"""
from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import torch

from .tabletop_surface_contract import validate_actions
from .tabletop_surface_mpc_runtime import NativeSurfaceMPC


def _json_value(value):
    """Return owned JSON values, leaving unknown numerical evidence as null."""
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class TeacherReferenceMPC:
    """Training-only label wrapper around the unchanged native local grip MPC.

    ``source_sha`` identifies the actual canonical 24D reference artifact. The
    collector remains responsible for verifying that artifact's qualification,
    action contract, reset family and reference tick. Source qualification does
    not qualify a newly optimized command for native corrective BC/DAgger.

    ``refine`` returns ``(issued24, diagnostics)``. Context accepts JSON scalars
    or one scalar per environment; the collector supplies physics_step, tick,
    episode and any other identity fields. Snapshots bind pre-action measured
    state and applied targets, independently of the collector's context.
    """

    def __init__(self, env, source_sha):
        if (not isinstance(source_sha, str) or len(source_sha) != 64
                or any(char not in "0123456789abcdef" for char in source_sha)):
            raise ValueError("actual Teacher reference source SHA256 required")
        self.env = env
        self.source_sha = source_sha
        self.native = NativeSurfaceMPC(env)

    def _contexts(self, context):
        n = self.env.world.n
        if context is None:
            return [{} for _ in range(n)]
        if not isinstance(context, dict) or any(not isinstance(key, str) for key in context):
            raise ValueError("context must be a mapping with string keys")
        rows = [{} for _ in range(n)]
        for key, raw in context.items():
            if isinstance(raw, torch.Tensor):
                raw = raw.detach().cpu().tolist()
            values = raw if isinstance(raw, (tuple, list)) else [raw] * n
            if len(values) != n:
                raise ValueError("context arrays must contain one value per environment")
            for row, value in zip(rows, values):
                if (value is not None and not isinstance(value, (str, bool, int, float))
                        or isinstance(value, float) and not math.isfinite(value)):
                    raise ValueError("context values must be finite JSON scalars")
                row[key] = value
        return rows

    def _snapshots(self):
        t, w = self.env.task, self.env.world
        fields = {"command_rad": t.command, "palms_m": w.palms,
                  "intrinsic_xyz_rad": w.wrist_angles, "roots_wxyz": w.roots,
                  "root_seed_wxyz": w.root_seed,
                  "reference_bottom_m": torch.as_tensor(t.surface_reference_bottom_m)}
        names = {"q": "q_rad", "bottom": "bottom_m",
                 "object_relative": "object_relative_m", "object_quat": "object_quat_wxyz",
                 "wrist_pose": "wrist_pose_wxyz",
                 "effective_pad_points_w": "effective_pad_points_w_m",
                 "effective_pad_loads_n": "effective_pad_loads_n",
                 "effective_pad_vectors_w": "effective_pad_vectors_w_n",
                 "effective_pad_valid": "effective_pad_valid",
                 "effective_pad_body_ids": "effective_pad_body_ids"}
        fields.update({name: t.m[key] for key, name in names.items() if key in t.m})
        # Active joint readback does not determine passive tracking deflection.
        # Bind the actual rigid-link and PhysX Jacobian inputs used by the local
        # model as well, before any command is issued by the collector.
        for side, hand, bids in zip(("left", "right"), w.hands, self.native.bodies):
            fields[side + "_pad_link_pose_wxyz"] = hand.data.body_link_pose_w[:, bids]
            fields[side + "_pad_link_com_m"] = hand.data.body_com_pos_w[:, bids]
            fields[side + "_pad_link_spatial_jacobian"] = (
                hand.root_physx_view.get_jacobians()[:, [bid - 1 for bid in bids]])
        fields = {key: value.detach().cpu().contiguous().clone()
                  for key, value in fields.items() if isinstance(value, torch.Tensor)}
        rows = []
        for index in range(w.n):
            snapshot = {}
            digest = hashlib.sha256()
            for key, value in sorted(fields.items()):
                item = value[index] if value.ndim and value.shape[0] == w.n else value
                snapshot[key] = _json_value(item)
                digest.update(json.dumps([key, str(item.dtype), list(item.shape)],
                                         separators=(",", ":")).encode("utf-8"))
                # Hash original bytes, including invalid values; JSON null never
                # turns a missing/invalid measurement into a finite observation.
                digest.update(item.numpy().tobytes())
            rows.append((snapshot, digest.hexdigest()))
        return rows

    def _inactive_reasons(self):
        """Read the existing activation conditions without changing admission."""
        t, w = self.env.task, self.env.world
        reasons = ["native_model_rejected"] * w.n
        from .tabletop_surface_exploration import allocation
        try:
            loads = t.m["effective_pad_loads_n"]
            vectors = t.m["effective_pad_vectors_w"]
            valid = t.m["effective_pad_valid"]
            bottom = t.m["bottom"]
            reference = torch.as_tensor(t.surface_reference_bottom_m,
                                        device=bottom.device, dtype=bottom.dtype)
            if reference.ndim == 0:
                reference = reference.expand(w.n)
            if (loads.shape != (w.n, 2, 5) or vectors.shape != (w.n, 2, 5, 3)
                    or valid.shape != loads.shape or valid.dtype != torch.bool
                    or bottom.shape != (w.n,) or reference.shape != (w.n,)):
                return reasons
            for index in range(w.n):
                values = (loads[index], vectors[index], bottom[index], reference[index])
                if (not all(bool(torch.isfinite(value).all()) for value in values)
                        or bool((loads[index] < 0).any())):
                    reasons[index] = "invalid_activation_measurements"
                    continue
                contact = valid[index:index + 1] & (loads[index:index + 1] > .05)
                subset = SimpleNamespace(m=dict(effective_pad_loads_n=loads[index:index + 1],
                    effective_pad_vectors_w=vectors[index:index + 1], effective_pad_valid=contact))
                if not bool(allocation(subset)["paired"].all()):
                    reasons[index] = "no_effective_opposition"
                elif bool(bottom[index] < reference[index] + .003):
                    reasons[index] = "below_activation_lift"
                elif int(self.native.bad_predictions[index]) >= 3:
                    reasons[index] = "prediction_limit_reached"
        except (ValueError, RuntimeError, IndexError, KeyError, TypeError, AttributeError):
            # The native adapter remains authoritative about invalid models.
            pass
        return reasons

    @torch.no_grad()
    def refine(self, reference_actions, *, context=None):
        validate_actions(reference_actions)
        if (reference_actions.shape[0] != self.env.world.n
                or reference_actions.device != self.native.pending.device):
            raise ValueError("Teacher reference must match native environment batch/device")
        contexts = self._contexts(context)
        snapshots = self._snapshots()
        reasons = self._inactive_reasons()
        issued = self.native.refine(reference_actions)
        valid = torch.isfinite(issued).all(-1) & (issued.abs() <= 1).all(-1)
        if not bool(valid.all()):
            issued = torch.where(valid[:, None], issued, reference_actions)
            self.native.reset((~valid).nonzero().flatten())
        rows = []
        for index, (snapshot, snapshot_sha) in enumerate(snapshots):
            active = bool(self.native.pending[index])
            reason = "optimized_local_pad_model" if active else reasons[index]
            if not bool(valid[index]):
                reason = "invalid_model_output"
            delta = issued[index] - reference_actions[index]
            rows.append(dict(env=index, context=contexts[index], active=active,
                reason=reason, changed=bool(delta.abs().max() > 1e-6),
                reference_actions=_json_value(reference_actions[index]),
                issued_actions=_json_value(issued[index]), action_delta=_json_value(delta),
                snapshot=snapshot, snapshot_sha256=snapshot_sha))
        diagnostics = dict(version="dh116-vision-teacher-reference-mpc-v1",
            source_sha256=self.source_sha, training_expert_privileged_only=True,
            reference_role="teacher_nominal_at_collected_native_state",
            optimized_action_dim=12, returned_action_dim=24,
            wrist_intent="unchanged_teacher_reference_translation_and_intrinsic_XYZ",
            model="local_pad_kinematics_not_object_dynamics", force_prediction=False,
            native_correction_qualified=False, student_state_dagger_label=False,
            per_env=rows, runtime_statistics=self._statistics())
        return issued, diagnostics

    def _statistics(self):
        stats = _json_value(self.native.stats)
        # The reused runtime calls every nominal action "student". Relabel only
        # the exported copy; its wrist channels came from the Teacher reference.
        stats["wrist_intent"] = "unchanged_teacher_reference_translation_and_intrinsic_XYZ"
        return stats

    @torch.no_grad()
    def observe_result(self, done):
        if (not isinstance(done, torch.Tensor) or done.dtype != torch.bool
                or done.shape != self.native.pending.shape
                or done.device != self.native.pending.device):
            raise ValueError("done must be a native-device boolean environment mask")
        self.native.observe_result(done)
        return self._statistics()
