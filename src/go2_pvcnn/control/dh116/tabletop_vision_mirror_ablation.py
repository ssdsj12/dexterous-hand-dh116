"""Offline M0/M1/M2 mirror-equivalence diagnostics for the 3844D actor.

This module only transforms copied observations/actions and computes metrics. It
does not change the training contract, actor weights, reward, or deployment
preprocessing. M2 is an explicitly paired offline normalization: the mirrored
sample is mapped back into the same canonical frame before inference. It is not
silently enabled for an existing checkpoint.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from .tabletop_vision_quaternion import canonicalize_wrist_quaternion_history
from .tabletop_vision_student import FRAME_DIM, HISTORY, POINTS, POLICY_DIM


M0_RAW = "M0_raw"
M1_QUATERNION = "M1_quaternion"
M2_FULL_MIRROR = "M2_full_mirror"
MIRROR_MODES = (M0_RAW, M1_QUATERNION, M2_FULL_MIRROR)


def _require_observation(value: Any) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.ndim != 2
            or value.shape[1] != POLICY_DIM or not value.is_floating_point()):
        raise ValueError(f"observation must be floating [N,{POLICY_DIM}]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("observation must be finite")
    return value


def _require_action(value: Any) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.ndim != 2
            or value.shape[1] != 24 or not value.is_floating_point()):
        raise ValueError("action must be floating [N,24]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("action must be finite")
    return value


def mirror_action24(action: Any) -> torch.Tensor:
    """Mirror the existing 24D action contract without changing its order.

    The lateral reflection is ``diag(-1, 1, 1)`` in the world/palm action
    frame. Finger blocks are exchanged, palm X changes sign, and intrinsic XYZ
    rotations map ``(rx, ry, rz) -> (rx, -ry, -rz)`` before the hand exchange.
    """
    value = _require_action(action)
    result = value.clone()
    result[:, 0:6] = value[:, 6:12]
    result[:, 6:12] = value[:, 0:6]
    result[:, 12:15] = value[:, 15:18]
    result[:, 15:18] = value[:, 12:15]
    result[:, 12] *= -1.
    result[:, 15] *= -1.
    result[:, 18:21] = value[:, 21:24]
    result[:, 21:24] = value[:, 18:21]
    result[:, [19, 20, 22, 23]] *= -1.
    return result


def _mirror_frame(frame: torch.Tensor) -> torch.Tensor:
    frame = frame.clone()
    point_end = 2 * POINTS * 3
    mask_end = point_end + POINTS
    points = frame[:, :point_end].reshape(-1, 2, POINTS, 3)
    points = points[:, [1, 0]].clone()
    points[..., 0] *= -1.
    frame[:, :point_end] = points.reshape(-1, point_end)
    masks = frame[:, point_end:mask_end].reshape(-1, POINTS)
    frame[:, point_end:mask_end] = masks.clone()
    proprio = frame[:, mask_end:mask_end + 62].clone()
    joints = proprio[:, :12].reshape(-1, 2, 6)[:, [1, 0]]
    speeds = proprio[:, 12:24].reshape(-1, 2, 6)[:, [1, 0]]
    wrists = proprio[:, 24:38].reshape(-1, 2, 7)[:, [1, 0]].clone()
    wrists[..., 0] *= -1.
    # Under M=diag(-1,1,1), q=(w,x,y,z) maps to (w,x,-y,-z).
    wrists[..., 3 + 2] *= -1.
    wrists[..., 3 + 3] *= -1.
    previous = mirror_action24(proprio[:, 38:62])
    frame[:, mask_end:mask_end + 62] = torch.cat(
        (joints.reshape(-1, 12), speeds.reshape(-1, 12),
         wrists.reshape(-1, 14), previous), dim=-1)
    return frame


def mirror_observation(observation: Any) -> torch.Tensor:
    """Exchange hands and apply the lateral reflection to every four-frame field."""
    value = _require_observation(observation)
    frames = value.reshape(-1, HISTORY, FRAME_DIM)
    return torch.stack([_mirror_frame(frames[:, index]) for index in range(HISTORY)], dim=1).flatten(1)


def _observation_components(observation: torch.Tensor) -> dict[str, torch.Tensor]:
    frames = observation.reshape(-1, HISTORY, FRAME_DIM)
    point_end = 2 * POINTS * 3
    mask_end = point_end + POINTS
    proprio = frames[..., mask_end:mask_end + 62]
    return {
        "four_frame_history": frames,
        "local_point_clouds": frames[..., :point_end],
        "point_validity": frames[..., point_end:mask_end],
        "active_joints_12d": proprio[..., :12],
        "active_joint_speeds_12d": proprio[..., 12:24],
        "palm_xyz": proprio[..., 24:30],
        "wrist_quaternions": proprio[..., 27:38],
        "previous_action_24d": proprio[..., 38:62],
    }


def mirror_observation_roundtrip_metrics(observation: Any) -> dict[str, Any]:
    """Report component-wise error after applying the offline mirror twice."""
    value = _require_observation(observation)
    recovered = mirror_observation(mirror_observation(value))
    result = {}
    for name, original in _observation_components(value).items():
        actual = _observation_components(recovered)[name]
        delta = (actual - original).abs()
        result[name] = dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()))
    return result


def prepare_mirror_pair(observation: Any, mode: str):
    """Return ``(original, mirrored_pair, metadata)`` for one ablation mode."""
    if mode not in MIRROR_MODES:
        raise ValueError(f"unsupported mirror ablation mode: {mode}")
    value = _require_observation(observation)
    mirrored = mirror_observation(value)
    if mode == M0_RAW:
        return value.clone(), mirrored, dict(mode=mode, offline_only=False,
                                            preprocessing="raw")
    original = canonicalize_wrist_quaternion_history(value)
    mirrored = canonicalize_wrist_quaternion_history(mirrored)
    if mode == M1_QUATERNION:
        return original, mirrored, dict(mode=mode, offline_only=False,
                                        preprocessing="wrist_quaternion_hemisphere_v1")
    # M2 is paired normalization only. The mirrored sample is mapped back to
    # the same canonical local frame for an offline invariance check. This is
    # intentionally not a deployment preprocessing path.
    mirrored = mirror_observation(mirrored)
    return original, mirrored, dict(mode=mode, offline_only=True,
                                   preprocessing="quaternion_plus_full_mirror_pair")


def evaluate_mirror_outputs(actual: Any, expected: Any) -> dict[str, Any]:
    """Compute deterministic action L2, per-dimension, and direction metrics."""
    observed = _require_action(actual)
    target = _require_action(expected)
    if observed.shape != target.shape:
        raise ValueError("actual and expected action shapes must match")
    delta = observed - target
    abs_delta = delta.abs()
    direction = ((observed.abs() < 1e-8) & (target.abs() < 1e-8)) | (observed * target > 0.)
    return dict(
        sample_count=int(observed.shape[0]),
        action_dim=int(observed.shape[1]),
        l2_mean=float(torch.linalg.vector_norm(delta, dim=-1).mean()),
        l2_p95=float(torch.quantile(torch.linalg.vector_norm(delta, dim=-1), .95)),
        l2_max=float(torch.linalg.vector_norm(delta, dim=-1).max()),
        per_dim_abs_mean=abs_delta.mean(0).tolist(),
        per_dim_signed_mean=delta.mean(0).tolist(),
        direction_consistency_rate=float(direction.float().mean()),
    )


@torch.inference_mode()
def evaluate_actor_mirror_modes(actor: Callable[[torch.Tensor], torch.Tensor],
                                observation: Any) -> dict[str, Any]:
    """Run deterministic actor inference for M0/M1/M2 paired observations."""
    value = _require_observation(observation)
    report = {}
    for mode in MIRROR_MODES:
        original, paired, metadata = prepare_mirror_pair(value, mode)
        original_action = _require_action(actor(original))
        paired_action = _require_action(actor(paired))
        expected = original_action if mode == M2_FULL_MIRROR else mirror_action24(original_action)
        metrics = evaluate_mirror_outputs(paired_action, expected)
        report[mode] = dict(metadata=metadata, metrics=metrics,
                            output_semantics=("direct_canonical_output" if mode == M2_FULL_MIRROR
                                              else "mirrored_action_output"))
    return report
