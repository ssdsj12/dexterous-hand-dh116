"""RGB-only quarter/half/full ZNCC with conservative support and global audits.

Every finer level actually searches +/-2 pixels about its resized prior. A
local minimum, even a high-confidence interior one, cannot establish that the
coarse peak was correct. A streaming full-range audit therefore certifies each
finer result and supplies global uniqueness. This correctness-first candidate
makes no search-speedup claim. No disparity cost volume or reference depth is
used, and the returned disparity is in original-resolution horizontal pixels.
"""
import math

import torch
import torch.nn.functional as F

from .tabletop_stereo_calibration import StereoCalibration
from .tabletop_stereo_resolution import resize_calibration


def _raw_mask(image, mask):
    if mask is None:
        mask = torch.ones_like(image, dtype=torch.bool)
    if (not isinstance(mask, torch.Tensor) or mask.shape != image.shape
            or mask.dtype != torch.bool or mask.device != image.device):
        raise ValueError('raw validity requires bool [B,H,W] on image device')
    return mask & torch.isfinite(image)


def _downsample(image, raw_valid, size):
    """Area resize and ALL source support, directly from the original image."""
    safe = torch.where(raw_valid, image, 0.)
    if image.shape[-2:] == size:
        return safe, raw_valid
    # Area interpolation uses the same adaptive pooling bins. A single invalid
    # raw source pixel invalidates every output bin that includes that source.
    value = F.interpolate(safe[:, None], size=size, mode='area')[:, 0]
    invalid = F.adaptive_max_pool2d((~raw_valid).to(image.dtype)[:, None], size)[:, 0]
    return value, invalid == 0


def _resize_prior(disparity, valid, size):
    """Bilinear prior with exact horizontal units and fail-closed NaN support."""
    valid = valid & torch.isfinite(disparity)
    safe = torch.where(valid, disparity, 0.)
    value = F.interpolate(safe[:, None], size=size, mode='bilinear', align_corners=False)[:, 0]
    touched_invalid = F.interpolate((~valid).to(disparity.dtype)[:, None], size=size,
        mode='bilinear', align_corners=False)[:, 0]
    value = value * (size[1] / disparity.shape[-1])
    supported = (touched_invalid == 0) & torch.isfinite(value)
    return torch.where(supported, value, float('nan')), supported


def _invalid_result(image):
    invalid = torch.zeros_like(image, dtype=torch.bool)
    confidence = torch.zeros_like(image)
    disparity = torch.full_like(image, float('nan'))
    return disparity, confidence, invalid, dict(integer_disparity=disparity,
        integer_confidence=confidence, integer_good=invalid,
        refinement_valid=invalid, search_boundary=invalid, local_failed=~invalid)


@torch.no_grad()
def match_pyramid(a, b, sign, *, calibration, min_disparity, max_disparity,
                  block_size, subpixel, uniqueness_threshold, texture_threshold,
                  a_valid=None, b_valid=None, matcher=None):
    """Return disparity/confidence/validity/trace; input masks are RAW [B,H,W].

    ``matcher`` is a private dependency-injection hook for orchestration tests.
    The default backend is the bounded-plane ``match_cost(method='zncc')``.
    Requested .25/.5/1 scales use floor dimensions (minimum one pixel); every
    intrinsic and disparity conversion uses the resulting exact size ratios.
    """
    if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
            or a.ndim != 3 or a.shape != b.shape or min(a.shape) < 1
            or a.device != b.device or a.dtype != b.dtype or not a.is_floating_point()):
        raise ValueError('matching floating grayscale [B,H,W] images required')
    if (not isinstance(calibration, StereoCalibration)
            or (calibration.height, calibration.width) != a.shape[-2:]
            or not calibration.rectified):
        raise ValueError('matching rectified image calibration required')
    if (type(sign) is not int or sign not in (-1, 1)
            or type(min_disparity) is not int or type(max_disparity) is not int
            or not 1 <= min_disparity < max_disparity
            or type(block_size) is not int or block_size < 3 or block_size % 2 != 1
            or type(subpixel) is not bool
            or not math.isfinite(uniqueness_threshold) or not 0 <= uniqueness_threshold < 1
            or not math.isfinite(texture_threshold) or texture_threshold < 0):
        raise ValueError('invalid stereo pyramid configuration')
    am, bm = _raw_mask(a, a_valid), _raw_mask(b, b_valid)
    height, width = a.shape[-2:]
    levels = {}
    previous = None
    previous_width = None
    final = None
    for name, requested_scale in (('quarter', .25), ('half', .5), ('full', 1.)):
        w, h = max(1, int(width * requested_scale)), max(1, int(height * requested_scale))
        sx, sy = w / width, h / height
        low, high = math.floor(min_disparity * sx), min(w - 1, math.ceil(max_disparity * sx))
        resized_calibration = resize_calibration(calibration, w, h, allow_resource_override=True)
        # resize_calibration uses continuous u+.5/v+.5 centres: diag(sx,sy,1)K,
        # without an added principal-point shift. align_corners=False agrees.
        level = dict(requested_scale=requested_scale, resolution_wh=[w, h],
            scale_wh=[sx, sy], K_left=resized_calibration.K_left.tolist(),
            K_right=resized_calibration.K_right.tolist(), disparity_range_px=[low, high],
            disparity_scale_from_previous=None if previous_width is None else w / previous_width,
            search_radius_px=2, status='skipped')
        levels[name] = level
        if min(h, w) < block_size:
            level['skip_reason'] = 'image_smaller_than_block_size'
            continue
        if high < low:
            level['skip_reason'] = 'empty_disparity_range'
            continue
        if matcher is None:
            from .tabletop_stereo_matching_costs import match_cost
            matcher = match_cost
        ai, av = _downsample(a, am, (h, w))
        bi, bv = _downsample(b, bm, (h, w))
        kwargs = dict(method='zncc', min_disparity=low, max_disparity=high,
            block_size=block_size, subpixel=subpixel,
            uniqueness_threshold=uniqueness_threshold, texture_threshold=texture_threshold,
            a_valid=av, b_valid=bv)
        prior = None
        prior_valid = torch.zeros_like(ai, dtype=torch.bool)
        local = None
        if previous is not None:
            prior, prior_valid = _resize_prior(previous[0], previous[2], (h, w))
            # An all-NaN prior still executes the fail-closed local pass. This
            # keeps execution metadata independent of image batch composition.
            local = matcher(ai, bi, sign, prior=prior, search_radius=2, **kwargs)
        # Initial global pass or unconditional finer-level audit. The full-range
        # competitor, not local confidence, establishes uniqueness/rejection.
        gd, gc, gg, global_trace = matcher(ai, bi, sign, **kwargs)
        gg = gg & torch.isfinite(gd) & torch.isfinite(gc)
        local_disagreement = torch.zeros_like(gg)
        local_failed = torch.ones_like(gg)
        selected_local = torch.zeros_like(gg)
        if local is not None:
            ld, lc, lg, local_trace = local
            lg = lg & torch.isfinite(ld) & torch.isfinite(lc)
            local_failed = (~prior_valid | ~lg
                | local_trace.get('local_failed', torch.zeros_like(lg))
                | local_trace.get('search_boundary', torch.zeros_like(lg)))
            local_disagreement = lg & gg & ((ld - gd).abs() > 1e-5)
            selected_local = prior_valid & ~local_failed & ~local_disagreement & gg
            d = torch.where(selected_local, ld, gd)
        else:
            d = gd
        # Preserve globally measured confidence even when the certified local
        # winner is reused. A global rejection always rejects the local result.
        good = gg & torch.isfinite(d)
        d, c = torch.where(good, d, float('nan')), torch.where(good, gc, 0.)
        final_trace = dict(global_trace)
        final_trace['refinement_valid'] = global_trace.get('refinement_valid', torch.zeros_like(good)) & good
        level.update(status='processed', prior_valid=prior_valid,
            local_search_mask=prior_valid if local is not None else torch.zeros_like(good),
            local_search_executed=local is not None,
            local_failed=local_failed, local_disagreement=local_disagreement,
            local_selected_mask=selected_local,
            global_fallback_mask=~selected_local,
            global_audit_executed=previous is not None,
            global_audit_mask=torch.full_like(good, previous is not None),
            global_search_executed=True, refinement_valid=final_trace['refinement_valid'],
            search_boundary=global_trace.get('search_boundary', torch.zeros_like(good)))
        previous = (d, c, good)
        previous_width = w
        if name == 'full':
            final = (d, c, good, final_trace)
    # Never upsample a disparity result as an invisible final output step.
    # If the full level cannot run, return original-sized all-invalid tensors.
    if final is None:
        final = _invalid_result(a)
    d, c, good, trace = final
    trace['pyramid'] = dict(version='dh116-stereo-j2b-pyramid-v1', levels=levels,
        processed_level_count=sum(level['status'] == 'processed' for level in levels.values()),
        output_resolution_wh=[width, height], final_resize=False,
        scale_policy='floor_dimensions_exact_ratios',
        raw_support_policy='all_area_contributors_then_full_matching_patch',
        prior_support_policy='all_positive_bilinear_contributors_finite_and_valid',
        global_audit_policy='every_refined_level_full_range;global_rejection_overrides_local',
        effective_no_search_speedup=True, cost_volume_allocated=False,
        final_refinement_valid=trace['refinement_valid'])
    trace['metric'] = 'zncc'
    trace['resource_metadata'] = dict(trace.get('resource_metadata', {}),
        cost_volume_allocated=False, output_resolution_wh=[width, height],
        pyramid_global_audit=True, effective_no_search_speedup=True,
        retained_state='fixed_planes_per_level;no_disparity_axis')
    return d, c, good, trace
