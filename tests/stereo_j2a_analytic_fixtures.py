"""Direct samples of continuous RGB radiance, never resized discrete images.

The frontoparallel plane obeys I_R(x,y)=I_L(x+d,y) and Z=fx*B/d.
Ground truth belongs to this CPU numerical fixture, not a native camera or
physical qualification. Missing observations have explicit unknown labels.
"""
from dataclasses import dataclass
import math

import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration


def bandlimited_rgb(x, y, *, texture='bandlimited'):
    """Evaluate bounded RGB radiance at arbitrary continuous pixel coordinates.

    All frequencies are below Nyquist; x and y remain float64 until the final
    fixture samples are converted to float32. No warp/interpolate/resize API is
    used to synthesize the second camera.
    """
    if texture == 'constant':
        gray = torch.full_like(x + y, .5)
    elif texture == 'repeated':
        # Exact four-pixel period gives multiple indistinguishable candidates
        # (2, 6, 10 px in the test search), including on textured patches.
        gray = .5 + .25 * torch.cos((2 * math.pi / 4) * x)
    elif texture == 'bandlimited':
        gray = (.5
            + .105 * torch.sin(2 * math.pi * (.035 * x + .017 * y) + .31)
            + .085 * torch.cos(2 * math.pi * (.061 * x - .029 * y) + 1.17)
            + .065 * torch.sin(2 * math.pi * (.089 * x + .043 * y) + 2.43)
            + .045 * torch.cos(2 * math.pi * (.133 * x - .037 * y) + .73))
    else:
        raise ValueError('unknown continuous texture kind')
    # Channels share the observable radiance signal. Fixed channel gains retain
    # a meaningful gray signal for the legacy SAD conversion and RGB contract.
    return torch.stack((.1 + .8 * gray, .05 + .9 * gray, gray), dim=0)


@dataclass(frozen=True)
class AnalyticStereoFixture:
    left_rgb: torch.Tensor
    right_rgb: torch.Tensor
    calibration: StereoCalibration
    left_valid: torch.Tensor
    right_valid: torch.Tensor
    reference_valid: torch.Tensor
    disparity_reference_px: torch.Tensor
    depth_reference_m: torch.Tensor
    boundary_left: torch.Tensor
    occlusion_left: torch.Tensor
    constant_left: torch.Tensor
    repeated_left: torch.Tensor
    width: int
    height: int
    metadata: dict


def analytic_stereo_fixture(disparity_px, *, texture='bandlimited', width=128,
                            height=40, right_missing_columns=None,
                            right_occluder_columns=None):
    """Sample a fixed plane and expose a fixed, matcher-independent denominator.

    ``reference_valid`` uses the same 2800 pixels for every tested fractional
    disparity (before optional missing observations). The six-pixel vertical
    margin and 18-pixel left margin exceed the nine-pixel block radius and the
    test search range; neither estimated validity nor estimated disparity enters
    reference construction. Missing and observed-occluder stripes are distinct:
    the latter remains image-valid but has no plane correspondence or plane GT.
    """
    if (not math.isfinite(disparity_px) or not 0 < disparity_px < 16
            or width < 64 or height < 24):
        raise ValueError('invalid analytic stereo fixture geometry')
    y, x = torch.meshgrid(torch.arange(height, dtype=torch.float64),
                          torch.arange(width, dtype=torch.float64), indexing='ij')
    left_rgb = bandlimited_rgb(x, y, texture=texture)[None].float()
    right_rgb = bandlimited_rgb(x + disparity_px, y, texture=texture)[None].float()
    left_valid = torch.ones((1, height, width), dtype=torch.bool)
    right_valid = torch.ones_like(left_valid)
    if right_missing_columns is not None:
        start, stop = right_missing_columns
        if not 0 <= start < stop <= width:
            raise ValueError('invalid right missing-observation stripe')
        right_valid[..., start:stop] = False
        # A missing observation is explicitly masked; the fill is not rendered
        # geometry and must never become a valid stereo correspondence.
        right_rgb[..., start:stop] = 0.
    if right_occluder_columns is not None:
        start, stop = right_occluder_columns
        if not 0 <= start < stop <= width:
            raise ValueError('invalid observed right-occluder stripe')
        right_rgb[..., start:stop] = .95
    radius = 4
    boundary = ((x < radius) | (x >= width - radius)
                | (y < radius) | (y >= height - radius))[None]
    xr = x - disparity_px
    occlusion = ((xr < 0) | (xr > width - 1))[None]
    for stripe in (right_missing_columns, right_occluder_columns):
        if stripe is None:
            continue
        start, stop = stripe
        # Both neighbors used by fractional correspondence must be observed.
        occlusion |= ((xr.floor() >= start) & (xr.floor() < stop))[None]
        occlusion |= ((xr.ceil() >= start) & (xr.ceil() < stop))[None]
    reference = ((x >= 18) & (x < width - 10) & (y >= 6) & (y < height - 6))[None]
    reference &= ~occlusion
    d = torch.full((1, height, width), float('nan'))
    z = torch.full_like(d, float('nan'))
    fx, baseline = 80., .08
    d[reference] = disparity_px
    z[reference] = fx * baseline / disparity_px
    k = torch.tensor([[fx, 0., width / 2], [0., fx, height / 2], [0., 0., 1.]], dtype=torch.float64)
    left_pose = torch.eye(4, dtype=torch.float64)
    right_pose = left_pose.clone()
    right_pose[0, 3] = baseline
    calibration = StereoCalibration(k, k, left_pose, right_pose, width, height, baseline)
    return AnalyticStereoFixture(left_rgb, right_rgb, calibration, left_valid, right_valid,
        reference, d, z, boundary, occlusion,
        torch.full_like(reference, texture == 'constant'),
        torch.full_like(reference, texture == 'repeated'), width, height,
        dict(rgb_generation='direct_continuous_bandlimited_sampling',
            texture_kind=texture, native_accuracy_qualified=False,
            disparity_px=float(disparity_px), depth_m=fx * baseline / disparity_px,
            fx_px=fx, baseline_m=baseline, reference_denominator=int(reference.sum()),
            unknown_right_stripe=right_missing_columns,
            observed_right_occluder_stripe=right_occluder_columns))
