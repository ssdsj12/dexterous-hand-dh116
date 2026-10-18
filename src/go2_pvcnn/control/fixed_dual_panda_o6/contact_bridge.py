"""Validated bridge between moving DeformX pads and Isaac body COM wrenches.

The bridge deliberately keeps link pose, physical COM twist, mesh COM, and the
fixed world origin distinct.  In particular, runtime torques are recovered from
the contact engine's time-integrated world-origin angular impulse; a single
pose cannot exactly shift a wrench accumulated while a pad moves.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np


CHANNEL_BODY_NAMES = {
    "left_thumb": "left_thumb_distal",
    "left_index": "left_index_distal",
    "right_thumb": "right_thumb_distal",
    "right_index": "right_index_distal",
}
_CHANNELS = tuple(CHANNEL_BODY_NAMES)
_MANIFEST_SHA256 = "0af0f15e32f4f8a1a201066951a5cc8dcb417ad121c8b0feff78712c62cd7e21"


def _scalar(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not bool")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _array(value: object, name: str, shape=(3,)) -> np.ndarray:
    source = np.asarray(value, dtype=object)
    if source.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, Real) for item in source.flat):
        raise TypeError(f"{name} must contain real values, not bool")
    result = np.array(source, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _rotation(value: object) -> np.ndarray:
    result = _array(value, "rotation_wb", (3, 3))
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(
        np.linalg.det(result), 1, atol=1e-7, rtol=0
    ):
        raise ValueError("rotation_wb must be a proper orthonormal rotation")
    return result


def normalized_wxyz_to_rotation(value: object) -> np.ndarray:
    """Normalize a measured wxyz quaternion in float64, then form its rotation."""
    quaternion = _array(value, "quaternion_wxyz", (4,)).astype(np.float64, copy=True)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        raise ValueError("quaternion norm must be finite and nonzero")
    w, x, y, z = quaternion / norm
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return _rotation(rotation)


def _mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or tuple(value) != _CHANNELS:
        actual = tuple(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{name} channels must be exactly {_CHANNELS}, got {actual}")
    return value


@dataclass(frozen=True)
class BodyState:
    body_name: str
    link_position_w: np.ndarray
    rotation_wb: np.ndarray
    physical_com_position_w: np.ndarray
    physical_com_velocity_w: np.ndarray
    omega_w: np.ndarray
    sample_time_s: float
    sample_token: str

    def __post_init__(self) -> None:
        if not isinstance(self.body_name, str) or not self.body_name:
            raise TypeError("body_name must be a nonempty string")
        if not isinstance(self.sample_token, str) or not self.sample_token:
            raise TypeError("sample_token must be a nonempty string")
        for name in ("link_position_w", "physical_com_position_w", "physical_com_velocity_w", "omega_w"):
            object.__setattr__(self, name, _array(getattr(self, name), name))
        object.__setattr__(self, "rotation_wb", _rotation(self.rotation_wb))
        object.__setattr__(self, "sample_time_s", _scalar(self.sample_time_s, "sample_time_s"))


def _validate_states(body_states: Mapping[str, BodyState]) -> dict[str, BodyState]:
    _mapping(body_states, "body state")
    result = {}
    tokens, times = set(), set()
    for channel in _CHANNELS:
        value = body_states[channel]
        if not isinstance(value, BodyState):
            raise TypeError(f"body state {channel} must be BodyState")
        if value.body_name != CHANNEL_BODY_NAMES[channel]:
            raise ValueError(f"body identity mismatch for {channel}: {value.body_name}")
        value = BodyState(value.body_name, value.link_position_w, value.rotation_wb,
                          value.physical_com_position_w, value.physical_com_velocity_w,
                          value.omega_w, value.sample_time_s, value.sample_token)
        result[channel] = value
        tokens.add(value.sample_token); times.add(value.sample_time_s)
    if len(tokens) != 1 or len(times) != 1:
        raise ValueError("body states must share one sample token and time")
    return result


def mesh_motions_from_bodies(body_states, mesh_com_offsets, previous_state, dt):
    """Build four MeshMotion values from actual body samples."""
    from co_sim.moving_contact_models import MeshMotion

    step = _scalar(dt, "dt", positive=True)
    current = _validate_states(body_states)
    _mapping(mesh_com_offsets, "mesh offset")
    previous = None if previous_state is None else _validate_states(previous_state)
    if previous is not None:
        current_time = next(iter(current.values())).sample_time_s
        previous_time = next(iter(previous.values())).sample_time_s
        if not np.isclose(current_time - previous_time, step, atol=1e-12, rtol=1e-9):
            raise ValueError("sample time interval must equal dt")
        if next(iter(current.values())).sample_token == next(iter(previous.values())).sample_token:
            raise ValueError("current sample token is stale")
    result = {}
    for channel, value in current.items():
        offset = _array(mesh_com_offsets[channel], f"{channel} mesh COM offset")
        position = value.link_position_w + value.rotation_wb @ offset
        velocity = value.physical_com_velocity_w + np.cross(
            value.omega_w, position - value.physical_com_position_w
        )
        if previous is None:
            acceleration = np.zeros(3); alpha = np.zeros(3)
        else:
            old = previous[channel]
            old_position = old.link_position_w + old.rotation_wb @ offset
            old_velocity = old.physical_com_velocity_w + np.cross(
                old.omega_w, old_position - old.physical_com_position_w
            )
            acceleration = (velocity - old_velocity) / step
            alpha = (value.omega_w - old.omega_w) / step
        result[channel] = MeshMotion(position, value.rotation_wb, velocity, value.omega_w,
                                     acceleration, alpha)
    return result


@dataclass(frozen=True)
class ReactionWrench:
    force_w: np.ndarray
    torque_com_w: np.ndarray
    position_w: np.ndarray

    def __post_init__(self):
        for name in ("force_w", "torque_com_w", "position_w"):
            object.__setattr__(self, name, _array(getattr(self, name), name))


@dataclass(frozen=True)
class ArticulationWrenchArrays:
    force_w: np.ndarray
    torque_com_w: np.ndarray
    position_w: np.ndarray

    def __post_init__(self):
        shape = np.asarray(self.force_w).shape
        if len(shape) != 3 or shape[0] != 1 or shape[2] != 3:
            raise ValueError("articulation wrench arrays must have shape (1, num_bodies, 3)")
        for name in ("force_w", "torque_com_w", "position_w"):
            object.__setattr__(self, name, _array(getattr(self, name), name, shape))


def compose_articulation_arrays(wrenches, body_indices, num_bodies):
    """Return full single-environment body arrays, zero away from contact pads."""
    _mapping(wrenches, "wrench")
    if not isinstance(body_indices, Mapping) or set(body_indices) != {"left", "right"}:
        raise ValueError("body indices must contain left and right")
    if not isinstance(num_bodies, Mapping) or set(num_bodies) != {"left", "right"}:
        raise ValueError("body counts must contain left and right")
    result = {}
    for side in ("left", "right"):
        count = num_bodies[side]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{side} body count must be positive")
        names = (f"{side}_thumb", f"{side}_index")
        indices = body_indices[side]
        if not isinstance(indices, Mapping) or set(indices) != set(names):
            raise ValueError(f"{side} body indices must identify both pads")
        selected = tuple(indices[name] for name in names)
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count
               for index in selected):
            raise ValueError(f"{side} body index is out of range")
        if len(set(selected)) != len(selected):
            raise ValueError(f"{side} body indices must be unique")
        arrays = [np.zeros((1, count, 3)) for _ in range(3)]
        for name, index in zip(names, selected, strict=True):
            wrench = wrenches[name]
            if not isinstance(wrench, ReactionWrench):
                raise TypeError(f"{name} wrench must be ReactionWrench")
            arrays[0][0, index] = wrench.force_w
            arrays[1][0, index] = wrench.torque_com_w
            arrays[2][0, index] = wrench.position_w
        result[side] = ArticulationWrenchArrays(*arrays)
    return result


def static_offset_torque(torque_mesh_w, mesh_position_w, body_position_w, force_w):
    """Shift one static wrench from mesh COM to body COM (not for integration)."""
    torque = _array(torque_mesh_w, "torque_mesh_w")
    mesh = _array(mesh_position_w, "mesh_position_w")
    body = _array(body_position_w, "body_position_w")
    force = _array(force_w, "force_w")
    return torque + np.cross(mesh - body, force)


def reaction_wrenches(impulses, diagnostics, body_com_positions, dt):
    step = _scalar(dt, "dt", positive=True)
    _mapping(impulses, "impulse"); _mapping(body_com_positions, "body COM position")
    if not isinstance(diagnostics, Mapping):
        raise TypeError("diagnostics must be a mapping")
    channels = diagnostics.get("channels")
    _mapping(channels, "diagnostic")
    integrated = _scalar(diagnostics.get("integrated_duration_s"), "integrated_duration_s", positive=True)
    if not np.isclose(integrated, step, atol=1e-15, rtol=1e-12):
        raise ValueError("diagnostic integrated duration must equal dt")
    if diagnostics.get("table_label") != "table":
        raise ValueError("diagnostic table label must identify the independent table load")
    evaluation_count = diagnostics.get("evaluation_count")
    if isinstance(evaluation_count, bool) or not isinstance(evaluation_count, int) or evaluation_count <= 0:
        raise ValueError("diagnostic evaluation_count must be a positive integer")
    evaluation_sum = 0
    from co_sim.moving_contact_models import ContactImpulse

    validated = {}
    for channel in _CHANNELS:
        impulse = impulses[channel]
        if not isinstance(impulse, ContactImpulse):
            raise TypeError(f"{channel} impulse must be ContactImpulse")
        # Reconstruct to guard against mutation of externally-owned values.
        impulse = ContactImpulse(impulse.linear_w, impulse.angular_com_w,
                                 impulse.peak_force_n, impulse.contact_duration_s)
        # Fine-step accumulation has the same floating-point tolerance as
        # integrated_duration_s above (e.g. 417 substeps at 240 Hz).
        if impulse.contact_duration_s > step and not np.isclose(
                impulse.contact_duration_s, step, atol=1e-15, rtol=1e-12):
            raise ValueError(f"{channel} contact duration exceeds dt")
        entry = channels[channel]
        if not isinstance(entry, Mapping):
            raise TypeError(f"{channel} diagnostics must be a mapping")
        evaluations = entry.get("evaluations")
        if isinstance(evaluations, bool) or not isinstance(evaluations, int) or evaluations <= 0:
            raise ValueError(f"{channel} evaluation count must be a positive integer")
        evaluation_sum += evaluations
        linear_balance = _array(entry.get("linear_balance_w"), "linear_balance_w")
        mesh_linear = _array(entry.get("mesh_linear_w"), "mesh_linear_w")
        rod_linear = _array(entry.get("rod_linear_w"), "rod_linear_w")
        if not np.allclose(mesh_linear, impulse.linear_w, atol=1e-12, rtol=1e-12):
            raise ValueError(f"{channel} mesh linear impulse does not match returned impulse")
        if not np.allclose(mesh_linear + rod_linear, linear_balance, atol=1e-12, rtol=1e-12):
            raise ValueError(f"{channel} independently computed linear balance mismatch")
        angular_balance = _array(entry.get("angular_origin_balance_w"), "angular_origin_balance_w")
        if not np.allclose(linear_balance, 0, atol=1e-10, rtol=0):
            raise ValueError(f"{channel} linear balance failed")
        if not np.allclose(angular_balance, 0, atol=1e-10, rtol=0):
            raise ValueError(f"{channel} angular origin balance failed")
        origin = _array(entry.get("mesh_angular_origin_w"), "mesh_angular_origin_w")
        rod_origin = _array(entry.get("rod_angular_origin_w"), "rod_angular_origin_w")
        if not np.allclose(origin + rod_origin, 0, atol=1e-10, rtol=0):
            raise ValueError(f"{channel} independent angular origin impulses do not balance")
        position = _array(body_com_positions[channel], f"{channel} body COM position")
        force = impulse.linear_w / step
        torque = (origin - np.cross(position, impulse.linear_w)) / step
        validated[channel] = ReactionWrench(force, torque, position)
    if evaluation_sum != evaluation_count:
        raise ValueError("aggregate evaluation count does not equal channel sum")
    return validated


class ReactionSink(Protocol):
    def submit(self, side: str, force: np.ndarray, torque: np.ndarray, position: np.ndarray) -> None: ...


@dataclass(frozen=True)
class PublishedBatch:
    token: str
    sample_time_s: float
    wrenches: Mapping[str, ReactionWrench]


class ContactReactionBridge:
    """Once-only, fail-closed publisher; validate everything before submission."""
    def __init__(self, sink: ReactionSink, *, max_force_n: float = float("inf")):
        if not hasattr(sink, "submit"):
            raise TypeError("sink must provide submit")
        self._sink = sink
        if max_force_n == float("inf"):
            self._max_force_n = max_force_n
        else:
            self._max_force_n = _scalar(max_force_n, "max_force_n", positive=True)
        self._last_token = None
        self._used_tokens = set()
        self._last_time = None
        self._unusable = False

    def publish(self, token, sample_time_s, impulses, diagnostics, body_com_positions, dt):
        if self._unusable:
            raise RuntimeError("contact reaction bridge is unusable after submit failure")
        if not isinstance(token, str) or not token:
            raise TypeError("batch token must be a nonempty string")
        time = _scalar(sample_time_s, "sample_time_s")
        if token in self._used_tokens or self._last_time is not None and time <= self._last_time:
            raise ValueError("duplicate or stale batch token/time")
        wrenches = reaction_wrenches(impulses, diagnostics, body_com_positions, dt)
        if (any(np.linalg.norm(value.force_w) > self._max_force_n for value in wrenches.values()) or
                any(impulses[channel].peak_force_n > self._max_force_n for channel in _CHANNELS)):
            raise ValueError("force gate exceeded before sink submission")
        payload = {}
        for side in ("left", "right"):
            names = (f"{side}_thumb", f"{side}_index")
            payload[side] = tuple(np.stack([getattr(wrenches[name], field) for name in names])
                                  for field in ("force_w", "torque_com_w", "position_w"))
        try:
            for side in ("left", "right"):
                self._sink.submit(side, *payload[side])
        except BaseException:
            self._unusable = True
            raise
        self._last_token = token; self._last_time = time
        self._used_tokens.add(token)
        return PublishedBatch(token, time, MappingProxyType(dict(wrenches)))


def _read_pinned(path: Path) -> tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read pinned file: {path}") from error
    return payload, hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _read_pinned(path)[1]


def validate_pad_manifest(repo_root: Path, manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Verify the exact collision manifest and all source/calibration/derived pins."""
    root = Path(repo_root).resolve(); path = Path(manifest_path).resolve()
    payload, digest = _read_pinned(path)
    if digest != _MANIFEST_SHA256:
        raise ValueError("pad manifest SHA-256 mismatch")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid pad manifest") from error
    calibration = root / manifest["calibration_path"]
    if _sha256(calibration) != manifest["calibration_sha256"]:
        raise ValueError("calibration SHA-256 mismatch")
    records = {}
    pads = manifest.get("pads")
    if not isinstance(pads, list) or len(pads) != 4:
        raise ValueError("manifest must contain four pads")
    for record in pads:
        channel = f"{record.get('side')}_{record.get('digit')}"
        if channel not in CHANNEL_BODY_NAMES or channel in records:
            raise ValueError("pad channel identity mismatch")
        if record.get("body_name") != CHANNEL_BODY_NAMES[channel]:
            raise ValueError(f"pad body identity mismatch for {channel}")
        source = root / record["source_path"]
        derived = path.parent / record["derived_path"]
        if _sha256(source) != record["source_sha256"]:
            raise ValueError(f"source SHA-256 mismatch for {channel}")
        if _sha256(derived) != record["derived_sha256"]:
            raise ValueError(f"derived SHA-256 mismatch for {channel}")
        records[channel] = dict(record)
    if tuple(records) != _CHANNELS:
        raise ValueError("pad records have missing, extra, or swapped channel identity")
    return records


__all__ = ["CHANNEL_BODY_NAMES", "BodyState", "ReactionWrench", "PublishedBatch",
           "ContactReactionBridge", "mesh_motions_from_bodies", "reaction_wrenches",
           "ArticulationWrenchArrays", "compose_articulation_arrays", "static_offset_torque",
           "normalized_wxyz_to_rotation", "validate_pad_manifest"]
