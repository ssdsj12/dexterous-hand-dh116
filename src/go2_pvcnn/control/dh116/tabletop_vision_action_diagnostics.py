"""Pure CPU diagnostics for the 24D visual Student action contract.

This module only records and summarizes data supplied by the native runner. It
does not decode actions into physics commands and never changes a target or a
guard. The runner remains responsible for reading the actual SDK target after
all safety guards have run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json

import numpy as np


ACTION_SIZE = 24
FINGER_SLICE = slice(0, 12)
PALM_TRANSLATION_SLICE = slice(12, 18)
ROTATION_SLICE = slice(18, 24)
HAND_NAMES = ("left", "right")


def reset_relative_world_base(root_seed: Any, task_offset: Any) -> np.ndarray:
    """Return reset-relative roots with the scalar task offset applied to Z only."""
    roots = np.asarray(root_seed, dtype=np.float64)
    offset = np.asarray(task_offset, dtype=np.float64)
    if roots.ndim != 3 or roots.shape[-1] != 3 or not np.isfinite(roots).all():
        raise ValueError("root_seed must be finite with shape (N, 2, 3)")
    if offset.shape != (roots.shape[0],) or not np.isfinite(offset).all():
        raise ValueError("task_offset must be finite with shape (N,)")
    result = roots.copy()
    result[:, :, 2] += offset[:, None]
    return result


def relative_palm_pose_m(actual_pose: Any, reset_pose: Any) -> np.ndarray:
    """Return measured palm displacement from a fixed physical-reset pose."""
    actual = np.asarray(actual_pose, dtype=np.float64)
    reset = np.asarray(reset_pose, dtype=np.float64)
    if actual.shape != reset.shape or actual.ndim != 3 or actual.shape[-1] != 3:
        raise ValueError("actual_pose and reset_pose must have the same shape (N, 2, 3)")
    if not np.isfinite(actual).all() or not np.isfinite(reset).all():
        raise ValueError("actual_pose and reset_pose must be finite")
    return actual - reset


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def split_action24(actions: Any) -> dict[str, np.ndarray]:
    """Split an ``N x 24`` action without changing its values or order."""
    value = np.asarray(actions, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != ACTION_SIZE or not np.isfinite(value).all():
        raise ValueError("actions must be a finite N x 24 array")
    return {
        "finger12": value[:, FINGER_SLICE].copy(),
        "left_finger6": value[:, 0:6].copy(),
        "right_finger6": value[:, 6:12].copy(),
        "palm_translation6": value[:, PALM_TRANSLATION_SLICE].copy(),
        "left_palm_translation3": value[:, 12:15].copy(),
        "right_palm_translation3": value[:, 15:18].copy(),
        "rotation6": value[:, ROTATION_SLICE].copy(),
        "left_rotation3": value[:, 18:21].copy(),
        "right_rotation3": value[:, 21:24].copy(),
    }


def _stats(values: np.ndarray, *, saturation: float = 1.0) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("stats require a finite two-dimensional array")
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "saturation_rate": (np.abs(values) >= saturation - 1e-12).mean(axis=0).tolist(),
    }


def summarize_action24(actions: Any, *, saturation: float = 1.0) -> dict[str, Any]:
    """Return per-hand action statistics for one batch."""
    parts = split_action24(actions)
    return {
        "contract": "finger12|palm_translation6|rotation6",
        "action_size": ACTION_SIZE,
        "left": {
            "finger6": _stats(parts["left_finger6"], saturation=saturation),
            "palm_translation3": _stats(parts["left_palm_translation3"], saturation=saturation),
            "rotation3": _stats(parts["left_rotation3"], saturation=saturation),
        },
        "right": {
            "finger6": _stats(parts["right_finger6"], saturation=saturation),
            "palm_translation3": _stats(parts["right_palm_translation3"], saturation=saturation),
            "rotation3": _stats(parts["right_rotation3"], saturation=saturation),
        },
    }


def _projection_stats(requested: np.ndarray, projected: np.ndarray) -> dict[str, Any]:
    delta = projected - requested
    changed = np.abs(delta) > 1e-12
    return {
        "requested_mean": requested.mean(axis=0).tolist(),
        "projected_mean": projected.mean(axis=0).tolist(),
        "delta_mean": delta.mean(axis=0).tolist(),
        "delta_abs_mean": np.abs(delta).mean(axis=0).tolist(),
        "projection_rate": changed.mean(axis=0).tolist(),
    }


@dataclass
class VisionActionDiagnosticRecorder:
    """Own per-environment action/target evidence for later JSONL writing."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        env_id: Any,
        episode_id: Any,
        physics_step: Any,
        timestamp_s: Any,
        raw_action24: Any,
        requested_palm_target_m: Any,
        pre_projection_palm_target_m: Any,
        projected_palm_target_m: Any,
        final_sdk_palm_target_m: Any,
        actual_palm_pose_m: Any,
        actual_palm_velocity_m_s: Any,
        table_clearance_m: Any,
        safe_z_m: Any,
        contact_loads_n: Any = None,
        contact_valid: Any = None,
        projection_reasons: Any = None,
        palm_displacement_m: Any = None,
        control_frame_sdk_target_m: Any = None,
    ) -> dict[str, Any]:
        """Record one timestamped batch; values are copied and never aliased."""
        action = np.asarray(raw_action24, dtype=np.float64)
        if action.ndim != 2 or action.shape[1] != ACTION_SIZE or not np.isfinite(action).all():
            raise ValueError("raw_action24 must be a finite N x 24 array")
        n = action.shape[0]
        fields = {
            "requested_palm_target_m": (2, 3),
            "pre_projection_palm_target_m": (2, 3),
            "projected_palm_target_m": (2, 3),
            "final_sdk_palm_target_m": (2, 3),
            "actual_palm_pose_m": (2, 3),
            "actual_palm_velocity_m_s": (2, 3),
        }
        arrays: dict[str, np.ndarray] = {}
        for name, tail in fields.items():
            arrays[name] = _array(locals()[name], (n, *tail), name)
        clearance = _array(table_clearance_m, (n, 2), "table_clearance_m")
        safe_z = _array(safe_z_m, (n, 2), "safe_z_m")
        displacement = arrays["actual_palm_pose_m"] if palm_displacement_m is None else _array(
            palm_displacement_m, (n, 2, 3), "palm_displacement_m")
        control_sdk = arrays["final_sdk_palm_target_m"] if control_frame_sdk_target_m is None else _array(
            control_frame_sdk_target_m, (n, 2, 3), "control_frame_sdk_target_m")
        loads = None if contact_loads_n is None else _array(contact_loads_n, (n, 2), "contact_loads_n")
        valid = None if contact_valid is None else np.asarray(contact_valid, dtype=bool)
        if valid is not None and valid.shape != (n, 2):
            raise ValueError("contact_valid must have shape (N, 2)")
        ids = np.asarray(env_id)
        episodes = np.asarray(episode_id)
        steps = np.asarray(physics_step)
        times = np.asarray(timestamp_s, dtype=np.float64)
        if ids.shape != (n,) or episodes.shape != (n,) or steps.shape != (n,) or times.shape != (n,):
            raise ValueError("env_id, episode_id, physics_step and timestamp_s must have shape (N,)")
        if not np.isfinite(times).all():
            raise ValueError("timestamp_s must be finite")
        reasons = [None] * n if projection_reasons is None else list(projection_reasons)
        if len(reasons) != n:
            raise ValueError("projection_reasons must have length N")
        batch = []
        for i in range(n):
            row = {
                "env_id": int(ids[i]),
                "episode_id": int(episodes[i]),
                "physics_step": int(steps[i]),
                "timestamp_s": float(times[i]),
                "action24": action[i].tolist(),
                "action_split": {k: v[i].tolist() for k, v in split_action24(action).items()},
                "requested_palm_target_m": arrays["requested_palm_target_m"][i].tolist(),
                "pre_projection_palm_target_m": arrays["pre_projection_palm_target_m"][i].tolist(),
                "projected_palm_target_m": arrays["projected_palm_target_m"][i].tolist(),
                "final_sdk_palm_target_m": arrays["final_sdk_palm_target_m"][i].tolist(),
                "actual_palm_pose_m": arrays["actual_palm_pose_m"][i].tolist(),
                "actual_palm_velocity_m_s": arrays["actual_palm_velocity_m_s"][i].tolist(),
                "palm_displacement_m": displacement[i].tolist(),
                "control_frame_sdk_target_m": control_sdk[i].tolist(),
                "table_clearance_m": clearance[i].tolist(),
                "safe_z_m": safe_z[i].tolist(),
                "contact_loads_n": None if loads is None else loads[i].tolist(),
                "contact_valid": None if valid is None else valid[i].tolist(),
                "projection_reasons": reasons[i],
            }
            batch.append(row)
        self.rows.extend(batch)
        return {"rows": batch, "projection": _projection_stats(
            arrays["pre_projection_palm_target_m"], arrays["projected_palm_target_m"])}

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            return {"rows": 0, "by_hand": {}}
        result: dict[str, Any] = {"rows": len(self.rows), "by_hand": {}}
        for hand, start in zip(HAND_NAMES, (0, 1)):
            actions = np.asarray([row["action_split"][f"{hand}_palm_translation3"] for row in self.rows])
            pre = np.asarray([row["pre_projection_palm_target_m"][start] for row in self.rows])
            post = np.asarray([row["projected_palm_target_m"][start] for row in self.rows])
            sdk = np.asarray([row["control_frame_sdk_target_m"][start] for row in self.rows])
            loads = [row["contact_loads_n"][start] for row in self.rows
                     if row["contact_loads_n"] is not None]
            result["by_hand"][hand] = {
                "palm_action": _stats(actions),
                "projection": _projection_stats(pre, post),
                "sdk_delta_abs_mean": np.abs(sdk - post).mean(axis=0).tolist(),
                "max_table_clearance_m": max(row["table_clearance_m"][start] for row in self.rows),
                "min_safe_z_m": min(row["safe_z_m"][start] for row in self.rows),
                "contact_load_mean_n": (np.asarray(loads).mean(axis=0).tolist()
                                         if loads else None),
            }
        return result


@dataclass
class PhysicsActionDiagnosticWriter:
    """Stream one row for every 480 Hz substep without retaining the rollout."""

    stream: Any
    terminal_keys: set[tuple[int, int]] = field(default_factory=set)
    rows_written: int = 0
    terminal_rows: int = 0
    reset_epoch: int = 0
    last_timestamp_s: float | None = None
    last_physics_step: int | None = None

    def begin_reset(self) -> None:
        """Start a new diagnostic reset epoch without reusing episode UIDs."""
        self.reset_epoch += 1
        self.terminal_keys.clear()

    def record_batch(
        self,
        env_id: Any,
        episode_id: Any,
        policy_step: int,
        physics_step: Any,
        timestamp_s: Any,
        live: Any,
        terminal: Any,
        audit: dict[str, Any],
    ) -> None:
        """Write a physical-substep batch before any terminal reset/no-op step."""
        ids = np.asarray(env_id)
        episodes = np.asarray(episode_id)
        steps = np.asarray(physics_step)
        times = np.asarray(timestamp_s, dtype=np.float64)
        live = np.asarray(live, dtype=bool)
        terminal = np.asarray(terminal, dtype=bool)
        n = ids.shape[0]
        if any(value.shape != (n,) for value in (episodes, steps, times, live, terminal)):
            raise ValueError("physics diagnostic metadata must have shape (N,)")
        if not np.isfinite(times).all():
            raise ValueError("physics diagnostic timestamp must be finite")
        if self.last_timestamp_s is not None and float(times.min()) < self.last_timestamp_s - 1e-12:
            raise ValueError("physics diagnostic timestamp must be monotonic")
        if self.last_physics_step is not None and int(steps.min()) < self.last_physics_step:
            raise ValueError("physics diagnostic physics_step must be monotonic")

        def arr(name: str, shape: tuple[int, ...]) -> np.ndarray:
            value = np.asarray(audit[name], dtype=np.float64)
            if value.shape != (n, *shape) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {(n, *shape)}")
            return value

        action = np.asarray(audit["raw_action24"], dtype=np.float64)
        if action.shape != (n, ACTION_SIZE) or not np.isfinite(action).all():
            raise ValueError("raw_action24 must be finite with shape (N, 24)")
        fields = {
            "requested_palm_target_m": arr("requested_palm_target_m", (2, 3)),
            "pre_projection_palm_target_m": arr("pre_projection_palm_target_m", (2, 3)),
            "projected_palm_target_m": arr("projected_palm_target_m", (2, 3)),
            "final_sdk_palm_target_m": arr("final_sdk_palm_target_m", (2, 3)),
            "actual_palm_pose_m": arr("actual_palm_pose_m", (2, 3)),
            "palm_displacement_m": arr("palm_displacement_m", (2, 3)),
            "actual_palm_velocity_m_s": arr("actual_palm_velocity_m_s", (2, 3)),
        }
        clearance = arr("table_clearance_m", (2,))
        safe_z = arr("safe_z_m", (2,))
        loads = arr("contact_loads_n", (2,))
        valid = np.asarray(audit["contact_valid"], dtype=bool)
        if valid.shape != (n, 2):
            raise ValueError("contact_valid must have shape (N, 2)")
        reasons = list(audit.get("projection_reasons", [None] * n))
        if len(reasons) != n:
            raise ValueError("projection_reasons must have length N")
        for i in range(n):
            key = (int(ids[i]), int(episodes[i]))
            episode_uid = f"{self.reset_epoch}:{key[0]}:{key[1]}"
            is_terminal = bool(terminal[i])
            if is_terminal and (not live[i] or key in self.terminal_keys):
                raise ValueError("duplicate or inactive terminal sample")
            post_terminal_noop = key in self.terminal_keys and not is_terminal
            if is_terminal:
                self.terminal_keys.add(key)
                self.terminal_rows += 1
            row = {
                "env_id": key[0],
                "episode_id": key[1],
                "episode_uid": episode_uid,
                "policy_step": int(policy_step),
                "physics_step": int(steps[i]),
                "timestamp_s": float(times[i]),
                "live_before_substep": bool(live[i]),
                "terminal_sample": is_terminal,
                "failure_code": int(audit["failure_code"][i]) if "failure_code" in audit else None,
                "post_terminal_noop": post_terminal_noop,
                "action24": action[i].tolist(),
                "action_split": {k: v[i].tolist() for k, v in split_action24(action).items()},
                "projection_reasons": reasons[i],
                "contact_valid": valid[i].tolist(),
                "table_clearance_m": clearance[i].tolist(),
                "safe_z_m": safe_z[i].tolist(),
                "contact_loads_n": loads[i].tolist(),
            }
            row.update({name: value[i].tolist() for name, value in fields.items()})
            self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.rows_written += 1
        self.last_timestamp_s = float(times.max())
        self.last_physics_step = int(steps.max())
        self.stream.flush()
