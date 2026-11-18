"""Opt-in supervision for a coarse binocular optical-Z prediction.

The head consumes only shared RGB features. Its predictions are not stereo
measurements and do not replace matcher depth or privileged state.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .tabletop_stereo_encoder import StereoEncoder
from .tabletop_stereo_wire_segmentation import native_instance_targets


_TOKEN_STRIDE = 32


@dataclass(frozen=True)
class CoarseStereoDepth:
    depth_m: torch.Tensor
    valid: torch.Tensor
    y_px: torch.Tensor
    x_px: torch.Tensor


def _token_grid(rgb):
    height, width = rgb.shape[-2:]
    return (
        torch.arange(0, height, _TOKEN_STRIDE, device=rgb.device),
        torch.arange(0, width, _TOKEN_STRIDE, device=rgb.device),
    )


class StereoDepthAuxiliary(nn.Module):
    """Predict two coarse grids from the existing attended left/right RGB tokens."""

    def __init__(self, encoder: StereoEncoder, *, min_depth_m: float, max_depth_m: float):
        super().__init__()
        if not isinstance(encoder, StereoEncoder):
            raise TypeError("the existing StereoEncoder is required")
        if not (math.isfinite(min_depth_m) and 0 < min_depth_m < max_depth_m < math.inf):
            raise ValueError("finite positive optical-Z depth range required")
        self.encoder = encoder
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.project_depth = nn.ModuleList([nn.Linear(encoder.embed_dim, 1) for _ in range(2)])

    def forward(self, left_rgb, right_rgb, image_valid):
        tokens, token_valid = self.encoder(left_rgb, right_rgb, image_valid)
        y_px, x_px = _token_grid(left_rgb)
        shape = (left_rgb.shape[0], y_px.numel(), x_px.numel())
        if tokens.shape[1] != 2 * shape[1] * shape[2]:
            raise ValueError("RGB encoder tokens no longer match the stride-32 grid")
        logits = torch.stack(
            [head(view).reshape(shape) for head, view in zip(self.project_depth, tokens.chunk(2, dim=1))],
            dim=1,
        )
        depth = self.min_depth_m + (self.max_depth_m - self.min_depth_m) * logits.sigmoid()
        paired = token_valid.reshape(shape[0], 2, *shape[1:]).all(dim=1, keepdim=True)
        return CoarseStereoDepth(depth, paired.expand_as(depth), y_px, x_px)


def native_depth_targets(frame, reference, *, min_depth_m=0.0, max_depth_m=math.inf):
    """Sample existing native validation depth on the same grid, without interpolation."""
    for key in ("frame_id", "simulation_time", "calibration_sha256"):
        if frame.get(key) is None or frame[key] != reference.get(key):
            raise ValueError("native reference/frame identity mismatch: " + key)
    sha = frame["calibration_sha256"]
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or sha == "0" * 64
        or any(digit not in "0123456789abcdef" for digit in sha)
        or reference.get("source_scope") != "native"
        or reference.get("source_units") != "optical_axis_z_m"
        or reference.get("validation_only") is not True
        or reference.get("reference_used_as_actor_input") is not False
    ):
        raise ValueError("native optical-Z reference labels required")

    if not (0 <= min_depth_m < max_depth_m):
        raise ValueError("ordered optical-Z depth range required")
    rgb = frame["left_rgb"]
    y_px, x_px = _token_grid(rgb)
    expected_shape = (rgb.shape[0], *rgb.shape[-2:])

    def sample_view(prefix):
        depth = reference.get(prefix + "depth_m")
        valid = reference.get(prefix + "valid")
        if (
            not isinstance(depth, torch.Tensor)
            or not depth.is_floating_point()
            or depth.shape != expected_shape
            or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool
            or valid.shape != expected_shape
        ):
            raise ValueError("matching native depth and validity tensors required: " + prefix)
        z = depth.detach().to(device=rgb.device, dtype=rgb.dtype).index_select(1, y_px).index_select(2, x_px)
        support = valid.to(rgb.device).index_select(1, y_px).index_select(2, x_px)
        support = support & torch.isfinite(z) & (z > 0) & (z >= min_depth_m) & (z <= max_depth_m)
        return z.masked_fill(~support, float("nan")), support

    left_depth, left_valid = sample_view("")
    if reference.get("right_raw_instance_ids") is not None:
        _, _, audit = native_instance_targets(frame, reference)
        if audit["right_instance_source"] != "native_instance_id":
            raise ValueError("independently resolved right native source required")
        right_depth, right_valid = sample_view("right_")
    else:
        right_depth = torch.full_like(left_depth, float("nan"))
        right_valid = torch.zeros_like(left_valid)
    return CoarseStereoDepth(
        torch.stack((left_depth, right_depth), dim=1),
        torch.stack((left_valid, right_valid), dim=1),
        y_px,
        x_px,
    )


def depth_auxiliary_loss(prediction, targets):
    """Compute native-reference optical-Z loss only where label and RGB pair exist."""
    if (
        prediction.depth_m.shape != targets.depth_m.shape
        or prediction.valid.shape != prediction.depth_m.shape
        or targets.valid.shape != targets.depth_m.shape
        or prediction.valid.dtype != torch.bool
        or targets.valid.dtype != torch.bool
        or not torch.equal(prediction.y_px, targets.y_px)
        or not torch.equal(prediction.x_px, targets.x_px)
    ):
        raise ValueError("prediction and target grid must match")
    mask = prediction.valid & targets.valid
    if not mask.any():
        raise ValueError("no supervised binocular depth samples")
    observed = targets.depth_m[mask]
    if not torch.isfinite(observed).all():
        raise ValueError("supervised depth samples must be finite")
    return F.smooth_l1_loss(prediction.depth_m[mask], observed, beta=0.01)
