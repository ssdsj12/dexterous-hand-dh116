"""RGB-only quarter/half/full ZNCC with explicit audit and accelerated search."""
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


_PLANE_COUNTERS = ('integer_plane_calls', 'fractional_plane_calls',
                   'integer_cost_evaluations', 'fractional_cost_evaluations',
                   'total_cost_evaluations')


def _search_totals(image, *traces):
    """Keep executed plane arithmetic separate from eligible center counts."""
    empty = torch.zeros(len(image), dtype=torch.int64, device=image.device)
    out = {key: empty.clone() for key in _PLANE_COUNTERS}
    out.update(integer_candidate_count=empty.clone(), fractional_candidate_count=empty.clone())
    for trace in traces:
        stats = trace.get('search_statistics', {}) if trace is not None else {}
        for key in _PLANE_COUNTERS:
            if key in stats:
                out[key] += stats[key]
        for kind in ('integer', 'fractional'):
            key = kind + '_candidates'
            if key in stats:
                out[kind + '_candidate_count'] += stats[key].flatten(1).sum(1)
    return out


def _choose_trace(local, global_result, use_local):
    """Choose diagnostics at the same centers as the returned disparity."""
    if global_result is None:
        return dict(local)
    selected = dict(global_result)
    for name, global_value in global_result.items():
        local_value = local.get(name)
        if isinstance(global_value, torch.Tensor) and global_value.shape == use_local.shape:
            selected[name] = torch.where(use_local, local_value, global_value)
        elif name == 'reasons':
            selected[name] = {key: torch.where(use_local, local_value[key], value)
                              for key, value in global_value.items()}
    return selected


@torch.no_grad()
def match_pyramid(a, b, sign, *, calibration, min_disparity, max_disparity,
                  block_size, subpixel, uniqueness_threshold, texture_threshold,
                  a_valid=None, b_valid=None, matcher=None, pyramid_mode='audit'):
    """Return disparity/confidence/validity/trace; input masks are RAW [B,H,W].

    ``matcher`` is a private dependency-injection hook for orchestration tests.
    The default backend is the bounded-plane ``match_cost(method='zncc')``.
    Requested .25/.5/1 scales use floor dimensions (minimum one pixel); every
    intrinsic and disparity conversion uses the resulting exact size ratios.
    Audit runs full-range validation at each finer level. Accelerated uses
    non-boundary coarse priors and accepts supported interior local matches
    with a cost margin. Every unresolved center is sent to full-range fallback.
    Local acceptance cannot establish uniqueness outside the searched window.
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
    if pyramid_mode not in ('audit', 'accelerated'):
        raise ValueError('pyramid_mode must be audit or accelerated')
    accelerated = pyramid_mode == 'accelerated'
    trust_margin = max(uniqueness_threshold, .05)
    am, bm = _raw_mask(a, a_valid), _raw_mask(b, b_valid)
    height, width = a.shape[-2:]
    levels = {}
    previous = None
    previous_width = None
    final = None
    totals = _search_totals(a)
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
            a_valid=av, b_valid=bv, count_search=True)
        prior = None
        prior_valid = torch.zeros_like(ai, dtype=torch.bool)
        local = None
        local_trace = None
        if previous is not None:
            prior, prior_valid = _resize_prior(previous[0], previous[2], (h, w))
            options = {'prune_search': True} if accelerated else {}
            local = matcher(ai, bi, sign, prior=prior, search_radius=2, **options, **kwargs)
        local_disagreement = torch.zeros_like(prior_valid)
        local_failed = torch.ones_like(prior_valid)
        selected_local = torch.zeros_like(prior_valid)
        if local is not None:
            ld, lc, lg, local_trace = local
            lg = lg & torch.isfinite(ld) & torch.isfinite(lc)
            local_failed = (~prior_valid | ~lg
                | local_trace.get('local_failed', torch.zeros_like(lg))
                | local_trace.get('search_boundary', torch.zeros_like(lg)))
            if accelerated:
                local_failed |= lc < trust_margin
                local_failed |= local_trace.get('reasons', {}).get('fractional_tie', torch.zeros_like(lg))
                if subpixel:
                    local_failed |= ~local_trace.get('refinement_valid', torch.zeros_like(lg))
                selected_local = ~local_failed

        global_mask = ~selected_local if accelerated else torch.ones_like(prior_valid)
        global_trace = None
        if not accelerated or bool(global_mask.any()):
            options = dict(search_mask=global_mask, prune_search=True) if accelerated else {}
            gd, gc, gg, global_trace = matcher(ai, bi, sign, **options, **kwargs)
            gg = gg & global_mask & torch.isfinite(gd) & torch.isfinite(gc)
        if accelerated and local is not None:
            if global_trace is None:
                d, c, good = ld, lc, selected_local
            else:
                d, c = torch.where(selected_local, ld, gd), torch.where(selected_local, lc, gc)
                good = selected_local | gg
            final_trace = _choose_trace(local_trace, global_trace, selected_local)
        elif local is not None:
            local_disagreement = lg & gg & ((ld - gd).abs() > 1e-5)
            selected_local = prior_valid & ~local_failed & ~local_disagreement & gg
            d = torch.where(selected_local, ld, gd)
            c, good, final_trace = gc, gg, dict(global_trace)
        else:
            d, c, good, final_trace = gd, gc, gg, dict(global_trace)
        good = good & torch.isfinite(d)
        d, c = torch.where(good, d, float('nan')), torch.where(good, c, 0.)
        final_trace['refinement_valid'] = final_trace.get('refinement_valid', torch.zeros_like(good)) & good
        counters = _search_totals(ai, local_trace, global_trace)
        for key, value in counters.items():
            totals[key] += value
        fallback_mask = ~selected_local
        level.update(status='processed', prior_valid=prior_valid,
            local_search_mask=prior_valid if local is not None else torch.zeros_like(good),
            local_search_executed=local is not None,
            local_failed=local_failed, local_disagreement=local_disagreement,
            local_selected_mask=selected_local,
            global_fallback_mask=fallback_mask,
            fallback_rate=(fallback_mask.flatten(1).float().mean(1) if previous is not None
                           else torch.zeros(len(ai), device=ai.device)),
            global_audit_executed=previous is not None and not accelerated,
            global_audit_mask=torch.full_like(good, previous is not None and not accelerated),
            global_search_executed=global_mask.flatten(1).any(1) if accelerated else True,
            search_statistics=counters, refinement_valid=final_trace['refinement_valid'],
            search_boundary=final_trace.get('search_boundary', torch.zeros_like(good)))
        credible_prior = good
        if accelerated:
            credible_prior = good & (c >= trust_margin) & ~level['search_boundary']
        previous = (d, c, credible_prior)
        previous_width = w
        if name == 'full':
            final = (d, c, good, final_trace)
    # Never upsample a disparity result as an invisible final output step.
    # If the full level cannot run, return original-sized all-invalid tensors.
    if final is None:
        final = _invalid_result(a)
    d, c, good, trace = final
    trace.pop('search_statistics', None)
    trace['pyramid'] = dict(version='dh116-stereo-pyramid-modes-v2', levels=levels,
        mode=pyramid_mode, global_certification=not accelerated,
        search_statistics=totals,
        search_statistics_scope='both_scans;eligible_centers_and_executed_full_plane_pixels',
        accelerated_trust_margin=trust_margin if accelerated else None,
        processed_level_count=sum(level['status'] == 'processed' for level in levels.values()),
        output_resolution_wh=[width, height], final_resize=False,
        scale_policy='floor_dimensions_exact_ratios',
        raw_support_policy='all_area_contributors_then_full_matching_patch',
        prior_support_policy='all_positive_bilinear_contributors_finite_and_valid',
        global_audit_policy=('untrusted_centers_only;no_global_check_on_accepted_local' if accelerated
                             else 'every_refined_level_full_range;global_rejection_overrides_local'),
        effective_no_search_speedup=not accelerated, cost_volume_allocated=False,
        final_refinement_valid=trace['refinement_valid'])
    trace['metric'] = 'zncc'
    trace['resource_metadata'] = dict(trace.get('resource_metadata', {}),
        cost_volume_allocated=False, output_resolution_wh=[width, height],
        pyramid_global_audit=not accelerated, effective_no_search_speedup=not accelerated,
        retained_state='fixed_planes_per_level;no_disparity_axis')
    return d, c, good, trace
