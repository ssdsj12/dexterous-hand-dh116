"""Opt-in, RGB-cost-only subpixel validation with constant disparity memory.

The streaming entry point takes five winner-relative cost planes, never a dense
image cost volume. The selected-sample adapter accepts only [P, D] diagnostic
costs. No native reference, depth, instance identity, or GT enters either API.
"""
import math
from collections.abc import Mapping

import torch


DEFAULT_RELIABILITY_POLICY = {
    'min_curvature': 1e-6,
    'fit_tolerance': .25,
    'absolute_fit_tolerance': .05,
    'mode_margin': .05,
    'fallback': 'integer',
}


def resolve_policy(policy=None):
    """Validate and copy a small serializable candidate policy."""
    if policy is not None and not isinstance(policy, Mapping):
        raise ValueError('subpixel reliability policy must be a mapping')
    result = dict(DEFAULT_RELIABILITY_POLICY)
    if policy is not None:
        if set(policy) - set(result):
            raise ValueError('unknown subpixel reliability policy field')
        result.update(policy)
    for key in ('min_curvature', 'fit_tolerance', 'absolute_fit_tolerance', 'mode_margin'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('subpixel reliability thresholds must be finite numbers')
        if value < 0 or (key == 'min_curvature' and value == 0):
            raise ValueError('invalid subpixel reliability threshold')
    if result['mode_margin'] >= 1 or result['fallback'] not in ('integer', 'invalid'):
        raise ValueError('invalid subpixel reliability policy')
    return result


def refine_winner_neighborhood(
    integer_disparity, costs_minus2, costs_minus1, costs_center, costs_plus1, costs_plus2,
    *, second_nonlocal_cost=None, available=None, image_ambiguity=None, policy=None,
):
    """Validate a quadratic or absolute-cost local model on five streamed planes.

    Every plane has the winner shape, e.g. [B,H,W]. ``available`` means all five
    disparities have valid image/search support. An optional nonlocal cost is
    the best supported cost at least two disparities from the winner; +inf
    means no competitor. Without it only the two supplied outer costs are
    checked, so distant modes are unobserved. ``image_ambiguity`` is an optional
    RGB-derived boundary proxy, never a true background or occlusion label.

    The central three points propose a parabola; +/-2 independently validate
    it. SAD may instead follow an L1 V. Its two outer slopes propose a vertex
    and nonnegative floor, with center cost and slope agreement as independent
    checks. Quadratic validation is preferred when both models fit. Positive
    curvature alone never validates a fractional offset.

    Model-fit/support/image-proxy rejection can retain the integer winner.
    Invalid integers, nonfinite costs, nonconvexity, flat/low curvature, and
    ambiguous modes remain invalid even with integer fallback. Rejection masks
    overlap; ``poor_quadratic_fit`` means neither validated local model passed.
    ``model_selected`` is 0=fallback/invalid, 1=quadratic, 2=absolute-cost.
    """
    settings = resolve_policy(policy)
    planes = (costs_minus2, costs_minus1, costs_center, costs_plus1, costs_plus2)
    if not all(isinstance(value, torch.Tensor) for value in (integer_disparity, *planes)):
        raise ValueError('integer disparity and five cost tensors required')
    if not costs_center.is_floating_point():
        raise ValueError('cost planes must have a floating dtype')
    shape, device, dtype = costs_center.shape, costs_center.device, costs_center.dtype
    if integer_disparity.shape != shape or integer_disparity.device != device:
        raise ValueError('integer disparity must match cost plane shape and device')
    if any(value.shape != shape or value.device != device or value.dtype != dtype for value in planes):
        raise ValueError('cost planes must share shape, dtype and device')
    for name, value in (('available', available), ('image_ambiguity', image_ambiguity)):
        if value is not None and (not isinstance(value, torch.Tensor) or value.shape != shape
                                  or value.device != device or value.dtype != torch.bool):
            raise ValueError(name + ' must be a boolean mask with the cost plane shape and device')
    if second_nonlocal_cost is not None and (
        not isinstance(second_nonlocal_cost, torch.Tensor) or second_nonlocal_cost.shape != shape
        or second_nonlocal_cost.device != device or not second_nonlocal_cost.is_floating_point()
    ):
        raise ValueError('nonlocal cost must match cost plane shape and device')

    cm2, cm1, c0, cp1, cp2 = planes
    finite = torch.isfinite(cm2) & torch.isfinite(cm1) & torch.isfinite(c0)
    finite = finite & torch.isfinite(cp1) & torch.isfinite(cp2)
    integer = integer_disparity.to(dtype=dtype)
    integer_ok = torch.isfinite(integer) & (integer > 0) & (integer == integer.round())
    supported = torch.ones_like(finite) if available is None else available
    image_ambiguous = torch.zeros_like(finite) if image_ambiguity is None else image_ambiguity
    floor = settings['min_curvature']
    # Scale roundoff only; this tolerance is not an empirical scene threshold.
    scale = torch.maximum(torch.maximum(cm2.abs(), cm1.abs()), c0.abs())
    scale = torch.maximum(scale, torch.maximum(cp1.abs(), cp2.abs()))
    roundoff = scale * (32 * torch.finfo(dtype).eps)

    # Preserve the historical central-three arithmetic order for its trace.
    curvature = cm1 + cp1 - 2 * c0
    left_curvature = cm2 - 2 * cm1 + c0
    right_curvature = c0 - 2 * cp1 + cp2
    nonconvex = finite & ((left_curvature < -roundoff) | (curvature < -roundoff)
                          | (right_curvature < -roundoff))
    low_curvature = finite & (curvature < floor)
    safe_curvature = torch.where(curvature > 0, curvature, torch.ones_like(curvature))
    raw_offset = .5 * (cm1 - cp1) / safe_curvature
    legacy_valid = (torch.isfinite(cm1) & torch.isfinite(c0) & torch.isfinite(cp1)
                    & (curvature > 1e-8))
    legacy_offset = torch.where(legacy_valid, raw_offset.clamp(-.5, .5), torch.zeros_like(c0))
    slope = .5 * (cp1 - cm1)
    predicted_minus2 = c0 + 2 * curvature - 2 * slope
    predicted_plus2 = c0 + 2 * curvature + 2 * slope
    residual = torch.maximum((cm2 - predicted_minus2).abs(), (cp2 - predicted_plus2).abs())
    residual = residual / curvature.abs().clamp_min(floor)
    residual = torch.where(finite, residual, torch.full_like(residual, float('inf')))
    offset_roundoff = 32 * torch.finfo(dtype).eps
    quadratic_fit_valid = (finite & (residual <= settings['fit_tolerance'])
                           & (raw_offset.abs() <= .5 + offset_roundoff))

    left_slope, right_slope = cm2 - cm1, cp2 - cp1
    absolute_slope = .5 * (left_slope + right_slope)
    safe_absolute_slope = torch.where(absolute_slope > 0, absolute_slope, torch.ones_like(c0))
    absolute_offset = .5 * (cm1 - cp1) / safe_absolute_slope
    absolute_floor = .5 * (cm1 + cp1) - absolute_slope
    absolute_center = absolute_floor + absolute_slope * absolute_offset.abs()
    absolute_scale = absolute_slope.abs().clamp_min(floor)
    absolute_residual = (c0 - absolute_center).abs() / absolute_scale
    absolute_slope_residual = (left_slope - right_slope).abs() / absolute_scale
    absolute_fit_valid = (finite & (absolute_slope >= floor)
                          & (left_slope > 0) & (right_slope > 0)
                          & (absolute_offset.abs() <= .5 + offset_roundoff) & (absolute_floor >= -roundoff)
                          & (absolute_residual <= settings['absolute_fit_tolerance'])
                          & (absolute_slope_residual <= settings['absolute_fit_tolerance']))

    # An adjacent tie may be a single half-pixel basin. Only a *strictly lower*
    # neighbor contradicts the caller's claimed integer winner.
    winner_not_minimum = finite & ((cm1 < c0 - roundoff) | (cp1 < c0 - roundoff))
    competing_cost = torch.minimum(cm2, cp2)
    if second_nonlocal_cost is not None:
        competing_cost = torch.minimum(competing_cost, second_nonlocal_cost.to(dtype=dtype))
    margin_scale = torch.maximum(competing_cost.abs(), c0.abs()).clamp_min(torch.finfo(dtype).tiny)
    mode_margin = (competing_cost - c0) / margin_scale
    multimodal = finite & (mode_margin <= settings['mode_margin'])
    nonfinite_competitor = (~torch.isfinite(competing_cost)) & (competing_cost != float('inf'))
    model_fit = quadratic_fit_valid | absolute_fit_valid
    reasons = {
        'invalid_integer': ~integer_ok,
        'search_boundary': ~supported,
        'nonfinite_cost': ~finite | nonfinite_competitor,
        'low_curvature': low_curvature,
        'nonconvex': nonconvex,
        'winner_not_minimum': winner_not_minimum,
        'multimodal': multimodal,
        'poor_quadratic_fit': finite & ~model_fit,
        'image_boundary_ambiguity': image_ambiguous,
    }
    base_valid = (integer_ok & finite & ~nonfinite_competitor & ~low_curvature & ~nonconvex
                  & ~winner_not_minimum & ~multimodal)
    refinement_valid = base_valid & supported & ~image_ambiguous & model_fit
    chosen_offset = torch.where(quadratic_fit_valid, raw_offset, absolute_offset).clamp(-.5, .5)
    refined_offset = torch.where(refinement_valid, chosen_offset, torch.zeros_like(c0))
    valid = base_valid if settings['fallback'] == 'integer' else refinement_valid
    disparity = torch.where(valid, integer + refined_offset, torch.full_like(c0, float('nan')))
    model_selected = torch.where(quadratic_fit_valid, torch.ones_like(integer, dtype=torch.int64),
                                 torch.full_like(integer, 2, dtype=torch.int64))
    model_selected = torch.where(refinement_valid, model_selected, torch.zeros_like(model_selected))
    return {
        'disparity_px': disparity, 'valid_mask': valid, 'valid': valid,
        'refinement_valid': refinement_valid, 'reasons': reasons,
        'legacy_offset': legacy_offset, 'legacy_refinement_valid': legacy_valid,
        'refined_offset': refined_offset, 'curvature': curvature,
        'fit_residual': residual, 'quadratic_fit_residual': residual,
        'quadratic_fit_valid': quadratic_fit_valid, 'absolute_fit_valid': absolute_fit_valid,
        'absolute_fit_residual': absolute_residual, 'absolute_slope_residual': absolute_slope_residual,
        'absolute_floor': absolute_floor, 'model_selected': model_selected,
    }


def reliable_subpixel_refinement(
    integer_disparity, costs, *, disparities, support=None, min_curvature=1e-6,
    fit_tolerance=.25, absolute_fit_tolerance=.05, mode_margin=.05, fallback='integer',
    image_ambiguity=None,
):
    """Selected [P,D] diagnostic adapter; dense image cost volumes are rejected.

    Disparities must be contiguous integer hypotheses. Unsupported costs never
    become a remote competitor; a missing winner or neighbor prevents refine.
    """
    if not isinstance(costs, torch.Tensor) or costs.ndim != 2 or not costs.is_floating_point():
        raise ValueError('selected cost samples must have shape [P,D] and floating dtype')
    if not isinstance(integer_disparity, torch.Tensor) or integer_disparity.shape != costs.shape[:1]:
        raise ValueError('selected integer disparities must have shape [P]')
    if integer_disparity.device != costs.device:
        raise ValueError('selected integer disparities must use the cost device')
    grid = torch.as_tensor(disparities, device=costs.device, dtype=costs.dtype)
    if (grid.ndim != 1 or grid.numel() != costs.shape[1] or grid.numel() < 1
            or not bool(torch.isfinite(grid).all()) or not bool((grid == grid.round()).all())
            or (grid.numel() > 1 and not bool(((grid[1:] - grid[:-1]) == 1).all()))):
        raise ValueError('disparities must be contiguous integer hypotheses matching D')
    if support is not None and (not isinstance(support, torch.Tensor) or support.shape != costs.shape
                                or support.device != costs.device or support.dtype != torch.bool):
        raise ValueError('selected support must be a boolean tensor with shape [P,D]')
    supported = torch.ones_like(costs, dtype=torch.bool) if support is None else support
    # Sanitize only the gather index; the original nonfinite/noninteger/negative
    # winner reaches the core unchanged and remains invalid.
    offset = integer_disparity.to(dtype=costs.dtype) - grid[0]
    safe_offset = torch.where(torch.isfinite(offset), offset, torch.zeros_like(offset))
    center_index = safe_offset.round().clamp(0, grid.numel() - 1).to(torch.int64)
    winner_present = (offset == safe_offset.round()) & (offset >= 0) & (offset < grid.numel())
    neighbors, all_available = [], winner_present
    for shift in (-2, -1, 0, 1, 2):
        index = center_index + shift
        in_range = (index >= 0) & (index < grid.numel())
        gather = index.clamp(0, grid.numel() - 1)[:, None]
        neighbor_support = supported.gather(1, gather).squeeze(1)
        cost = costs.gather(1, gather).squeeze(1)
        # Use the nearest finite placeholder for missing outer samples. Its
        # available mask, rather than an invented negative disparity, rejects
        # refinement and still allows an explicit integer fallback.
        if shift == 0:
            cost = torch.where(winner_present & neighbor_support, cost,
                               torch.full_like(cost, float('inf')))
        neighbors.append(cost)
        all_available = all_available & in_range & neighbor_support
    far = (grid[None] - integer_disparity[:, None]).abs() >= 2
    far_costs = torch.where(far & supported, costs, torch.full_like(costs, float('inf')))
    second_nonlocal_cost = far_costs.amin(dim=1)
    settings = dict(min_curvature=min_curvature, fit_tolerance=fit_tolerance,
                    absolute_fit_tolerance=absolute_fit_tolerance, mode_margin=mode_margin,
                    fallback=fallback)
    return refine_winner_neighborhood(
        integer_disparity, *neighbors, second_nonlocal_cost=second_nonlocal_cost,
        available=all_available, image_ambiguity=image_ambiguity, policy=settings,
    )


def photometric_refine_winner(
    a, b, integer_disparity, sign, block_size, a_valid, b_valid,
    *, min_curvature=1e-6, mode_margin=.05,
):
    """Independently verify a fractional match against actual grayscale images.

    Search thirteen fixed offsets -0.75:0.125:0.75 in two streaming passes.
    Only actually evaluated offsets in [-0.5,0.5] can be returned. The outer
    samples validate the basin and the best sample's two neighbors validate
    positive local curvature. A separated competing mode is at least 0.5px
    away. Returned costs are raw mean SAD in the input image intensity units.

    All pixels of each patch use its *center's* disparity. Warping a whole
    image with each neighboring pixel's own disparity before pooling would
    violate this contract. The two passes retain a constant number of [B,H,W]
    planes, and neither construct nor return a disparity cost volume.
    """
    resolve_policy({'min_curvature': min_curvature, 'mode_margin': mode_margin})
    if sign not in (-1, 1) or type(sign) is not int:
        raise ValueError('photometric search sign must be -1 or 1')
    if type(block_size) is not int or block_size < 3 or block_size % 2 != 1:
        raise ValueError('photometric block size must be odd and at least three')
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError('grayscale image tensors required')
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    if b.ndim == 4 and b.shape[1] == 1:
        b = b[:, 0]
    if (a.ndim != 3 or a.shape != b.shape or a.dtype != b.dtype or a.device != b.device
            or not a.is_floating_point()):
        raise ValueError('grayscale images must share [B,H,W] shape, floating dtype and device')
    for name, value in (('a_valid', a_valid), ('b_valid', b_valid)):
        if not isinstance(value, torch.Tensor) or value.shape != a.shape or value.device != a.device or value.dtype != torch.bool:
            raise ValueError(name + ' must be a boolean [B,H,W] image mask')
    if (not isinstance(integer_disparity, torch.Tensor) or integer_disparity.shape != a.shape
            or integer_disparity.device != a.device):
        raise ValueError('integer disparity must match grayscale images')
    batch, height, width = a.shape
    integer = integer_disparity.to(dtype=a.dtype)
    integer_ok = torch.isfinite(integer) & (integer > 0) & (integer == integer.round())
    safe_integer = torch.where(integer_ok, integer, torch.zeros_like(integer))
    yy = torch.arange(height, device=a.device).view(1, height, 1).expand(batch, height, width)
    xx = torch.arange(width, device=a.device).view(1, 1, width).expand(batch, height, width)
    a_flat, b_flat = a.reshape(batch, -1), b.reshape(batch, -1)
    av_flat = (a_valid & torch.isfinite(a)).reshape(batch, -1)
    bv_flat = (b_valid & torch.isfinite(b)).reshape(batch, -1)
    radius = block_size // 2

    def patch_cost(delta):
        target_x = xx.to(dtype=a.dtype) + sign * (safe_integer + delta)
        x0 = target_x.floor().to(torch.int64)
        x1 = target_x.ceil().to(torch.int64)
        alpha = target_x - x0.to(dtype=a.dtype)
        cost = torch.zeros_like(a)
        complete = integer_ok.clone()
        for dy in range(-radius, radius + 1):
            row = yy + dy
            row_valid = (row >= 0) & (row < height)
            row_base = row.clamp(0, height - 1) * width
            for dx in range(-radius, radius + 1):
                ax, bx0, bx1 = xx + dx, x0 + dx, x1 + dx
                inside = row_valid & (ax >= 0) & (ax < width) & (bx0 >= 0) & (bx1 < width)
                ia = (row_base + ax.clamp(0, width - 1)).reshape(batch, -1)
                ib0 = (row_base + bx0.clamp(0, width - 1)).reshape(batch, -1)
                ib1 = (row_base + bx1.clamp(0, width - 1)).reshape(batch, -1)
                va = a_flat.gather(1, ia).reshape_as(a)
                vb0 = b_flat.gather(1, ib0).reshape_as(a)
                vb1 = b_flat.gather(1, ib1).reshape_as(a)
                supported = (av_flat.gather(1, ia) & bv_flat.gather(1, ib0)
                             & bv_flat.gather(1, ib1)).reshape_as(a)
                complete = complete & inside & supported
                cost = cost + (va - (vb0 * (1 - alpha) + vb1 * alpha)).abs()
        cost = cost / (block_size * block_size)
        cost = torch.where(complete, cost, torch.full_like(cost, float('inf')))
        return cost, complete

    infinity = torch.full_like(a, float('inf'))
    best, integer_cost = infinity.clone(), infinity.clone()
    winner_index = torch.zeros_like(integer, dtype=torch.int64)
    best_supported = torch.zeros_like(integer_ok)
    for index in range(13):
        cost, supported = patch_cost((index - 6) * .125)
        # A flat curve should retain the central sample so its finite two
        # neighbors diagnose zero curvature instead of a spurious search edge.
        closer_to_integer = abs(index - 6) < (winner_index - 6).abs()
        replace = (cost < best) | ((cost == best) & supported & closer_to_integer)
        best = torch.where(replace, cost, best)
        winner_index = torch.where(replace, torch.full_like(winner_index, index), winner_index)
        best_supported = torch.where(replace, supported, best_supported)
        if index == 6:
            integer_cost = cost
    minus, plus, second_nonlocal = infinity.clone(), infinity.clone(), infinity.clone()
    minus_supported = torch.zeros_like(integer_ok)
    plus_supported = torch.zeros_like(integer_ok)
    for index in range(13):
        cost, supported = patch_cost((index - 6) * .125)
        minus_here, plus_here = winner_index == index + 1, winner_index == index - 1
        minus, plus = torch.where(minus_here, cost, minus), torch.where(plus_here, cost, plus)
        minus_supported = torch.where(minus_here, supported, minus_supported)
        plus_supported = torch.where(plus_here, supported, plus_supported)
        remote = (winner_index - index).abs() >= 4
        second_nonlocal = torch.minimum(second_nonlocal, torch.where(remote, cost, infinity))
    offset = (winner_index.to(dtype=a.dtype) - 6) * .125
    # Missing separated samples do not constitute evidence of uniqueness.
    complete = best_supported & minus_supported & plus_supported & torch.isfinite(second_nonlocal)
    interior = offset.abs() <= .5
    curvature = minus + plus - 2 * best
    finite = torch.isfinite(best) & torch.isfinite(minus) & torch.isfinite(plus)
    margin_scale = torch.maximum(second_nonlocal.abs(), best.abs()).clamp_min(torch.finfo(a.dtype).tiny)
    cost_margin = (second_nonlocal - best) / margin_scale
    multimodal = cost_margin <= mode_margin
    low_curvature = finite & (curvature < min_curvature)
    valid = integer_ok & complete & interior & finite & ~low_curvature & ~multimodal
    disparity = torch.where(valid, integer + offset, torch.full_like(integer, float('nan')))
    return {
        'disparity_px': disparity, 'valid_mask': valid, 'valid': valid, 'refinement_valid': valid,
        'photometric_offset': offset, 'best_cost': best, 'integer_cost': integer_cost,
        'minus_cost': minus, 'plus_cost': plus, 'second_nonlocal_cost': second_nonlocal,
        'curvature': curvature, 'cost_margin': cost_margin,
        'model_selected': torch.where(valid, torch.full_like(winner_index, 3), torch.zeros_like(winner_index)),
        'candidate_evaluations': torch.full_like(winner_index, 26),
        'reasons': {'invalid_integer': ~integer_ok, 'search_boundary': ~complete | ~interior,
                    'nonfinite_cost': ~finite, 'low_curvature': low_curvature, 'multimodal': multimodal},
    }


def repeated_texture_evidence(gray, valid, search_span):
    """Detect a supported single-frequency horizontal repetition from RGB alone.

    A cosine plus an arbitrary intensity offset satisfies
    X[n-1]+X[n+1] = a*X[n]+b, where a=2*cos(2*pi/period). Fit that recurrence
    over a fixed 31-triplet horizontal window using float64 moments. At least
    17 observed triplets, nonconstant variance, <=1e-5 normalized RMS model
    residual, two observed periods, and a period within the disparity search
    span are required. Missing pixels exclude *all* affected triplets before
    calculating moments; their fill values, including NaN, have no influence.

    This is positive evidence of this particular repeated texture class. A
    false mask does not establish uniqueness for arbitrary multi-frequency or
    nonstationary textures; integer nonlocal cost checks are still necessary.
    """
    if (isinstance(search_span, bool) or not isinstance(search_span, (int, float))
            or not math.isfinite(search_span) or search_span <= 0):
        raise ValueError('positive finite disparity search span required')
    if not isinstance(gray, torch.Tensor):
        raise ValueError('grayscale image tensor required')
    if gray.ndim == 4 and gray.shape[1] == 1:
        gray = gray[:, 0]
    if gray.ndim != 3 or not gray.is_floating_point():
        raise ValueError('grayscale image must have shape [B,H,W]')
    if (not isinstance(valid, torch.Tensor) or valid.shape != gray.shape
            or valid.device != gray.device or valid.dtype != torch.bool):
        raise ValueError('valid must be a boolean image mask with shape [B,H,W]')
    batch, height, width = gray.shape
    observed = valid & torch.isfinite(gray)
    triplet = observed & observed.roll(1, -1) & observed.roll(-1, -1)
    triplet = triplet & (torch.arange(width, device=gray.device).view(1, 1, width) > 0)
    triplet = triplet & (torch.arange(width, device=gray.device).view(1, 1, width) < width - 1)
    samples = gray.to(torch.float64)
    x = torch.where(triplet, samples, torch.zeros_like(samples))
    y = torch.where(triplet, samples.roll(1, -1) + samples.roll(-1, -1), torch.zeros_like(samples))

    def window_sum(value):
        pooled = torch.nn.functional.avg_pool1d(
            value.reshape(batch * height, 1, width), 31, stride=1, padding=15)
        return pooled.reshape_as(samples) * 31

    count = window_sum(triplet.to(torch.float64))
    divisor = count.clamp_min(1)
    mean_x, mean_y = window_sum(x) / divisor, window_sum(y) / divisor
    second_x, second_y = window_sum(x.square()) / divisor, window_sum(y.square()) / divisor
    cross = window_sum(x * y) / divisor
    variance_x = (second_x - mean_x.square()).clamp_min(0)
    variance_y = (second_y - mean_y.square()).clamp_min(0)
    covariance = cross - mean_x * mean_y
    # A roundoff bound for double-precision moment subtraction, independent of
    # native success thresholds, image truth, or the expected test disparity.
    numerical_variance_floor = 128 * torch.finfo(torch.float64).eps * second_x.clamp_min(1)
    nonconstant = variance_x > numerical_variance_floor
    a = covariance / variance_x.clamp_min(numerical_variance_floor)
    b = mean_y - a * mean_x
    residual_variance = (variance_y - a * covariance).clamp_min(0)
    normalized_residual = (residual_variance / variance_x.clamp_min(numerical_variance_floor)).sqrt()
    normalized_residual = torch.where(nonconstant, normalized_residual,
                                       torch.full_like(normalized_residual, float('inf')))
    # a=-2 is the Nyquist two-pixel alternation. A numerical excursion below
    # that endpoint is tolerated only at double-precision roundoff scale.
    coefficient_supported = nonconstant & (a >= -2 - 1e-10) & (a < 2)
    omega = (a.clamp(-2, 2) / 2).acos()
    period = 2 * math.pi / omega
    period = torch.where(coefficient_supported, period, torch.full_like(period, float('nan')))
    positions = torch.arange(width, device=gray.device, dtype=torch.float64).view(1, 1, width)
    maximum_positions = torch.where(triplet, positions, torch.full_like(samples, -float('inf')))
    minimum_positions = torch.where(triplet, positions, torch.full_like(samples, float('inf')))
    last = torch.nn.functional.max_pool1d(
        maximum_positions.reshape(batch * height, 1, width), 31, stride=1, padding=15).reshape_as(samples)
    first = -torch.nn.functional.max_pool1d(
        -minimum_positions.reshape(batch * height, 1, width), 31, stride=1, padding=15).reshape_as(samples)
    window_span = torch.where(count > 0, last - first + 1, torch.zeros_like(count))
    repeated = (observed & (count >= 17) & coefficient_supported
                & (normalized_residual <= 1e-5) & (period <= search_span)
                & (window_span >= 2 * period))
    return {
        'repeated_texture': repeated, 'mask': repeated,
        'support': count.round().to(torch.int64), 'period_px': period,
        'normalized_residual': normalized_residual,
        'coefficient_a': torch.where(nonconstant, a, torch.full_like(a, float('nan'))),
        'coefficient_b': torch.where(nonconstant, b, torch.full_like(b, float('nan'))),
        'window_span_px': window_span,
    }
