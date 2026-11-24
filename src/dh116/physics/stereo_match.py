"""Cost-volume matching, candidate refinement and wire-color matching primitives used by depth."""
import math as stere_match_candi__math
import math as stere_match_costs__math
import torch as stere_wire_match__torch
from collections.abc import Mapping
import torch as stere_match_costs__torch
METHODS = ('wire_census', 'wire_zncc', 'wire_silhouette')
import torch as stere_match_candi__torch
import torch.nn.functional as F
_CENTERS_PER_TILE = 128
DEFAULT_RELIABILITY_POLICY = {'min_curvature': 1e-06, 'fit_tolerance': 0.25, 'absolute_fit_tolerance': 0.05, 'mode_margin': 0.05, 'fallback': 'integer'}

def _validate(a, b, sign, method, low, high, block, subpixel, unique, texture, a_valid, b_valid, prior, radius):
    if not isinstance(a, stere_match_costs__torch.Tensor) or not isinstance(b, stere_match_costs__torch.Tensor) or a.ndim != 3 or (a.shape != b.shape) or (a.device != b.device) or (a.dtype != b.dtype) or (a.dtype not in (stere_match_costs__torch.float32, stere_match_costs__torch.float64)) or (min(a.shape) < 1):
        raise ValueError('gray images must share nonempty [B,H,W], float32/float64 dtype and device')
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError('matching sign must be -1 or 1')
    if method not in ('census', 'zncc', 'gradient'):
        raise ValueError('method must be census, zncc or gradient')
    if type(low) is not int or type(high) is not int or low < 0 or (high < low):
        raise ValueError('disparity bounds must be ordered nonnegative integers')
    if type(block) is not int or block < 3 or block % 2 != 1 or (min(a.shape[-2:]) < block):
        raise ValueError('block size must be odd, at least three and fit the image')
    if type(subpixel) is not bool:
        raise ValueError('subpixel must be bool')
    for (value, name) in ((unique, 'uniqueness'), (texture, 'texture')):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not stere_match_costs__math.isfinite(value)) or (value < 0):
            raise ValueError(name + ' threshold must be a finite nonnegative number')
    if unique >= 1:
        raise ValueError('uniqueness threshold must be below one')
    if type(radius) is not int or radius < 0:
        raise ValueError('search_radius must be a nonnegative integer')
    for (name, mask) in (('a_valid', a_valid), ('b_valid', b_valid)):
        if mask is not None and (not isinstance(mask, stere_match_costs__torch.Tensor) or mask.shape != a.shape or mask.device != a.device or (mask.dtype != stere_match_costs__torch.bool)):
            raise ValueError(name + ' must be a raw bool [B,H,W] mask on the image device')
    if prior is not None and (not isinstance(prior, stere_match_costs__torch.Tensor) or prior.shape != a.shape or prior.device != a.device or (not prior.is_floating_point())):
        raise ValueError('prior must be a floating [B,H,W] image on the image device')

def wire_colour_mask(rgb):
    """Use the same RGB predicate as the tracked wire extractor."""
    (red, green, blue) = rgb.unbind(1)
    return (red > 1.07 * green) & (green > 1.1 * blue) & (red > 0.12)

def resolve_policy(policy=None):
    """Validate and copy a small serializable candidate policy."""
    if policy is not None and (not isinstance(policy, Mapping)):
        raise ValueError('subpixel reliability policy must be a mapping')
    result = dict(DEFAULT_RELIABILITY_POLICY)
    if policy is not None:
        if set(policy) - set(result):
            raise ValueError('unknown subpixel reliability policy field')
        result.update(policy)
    for key in ('min_curvature', 'fit_tolerance', 'absolute_fit_tolerance', 'mode_margin'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not stere_match_candi__math.isfinite(value)):
            raise ValueError('subpixel reliability thresholds must be finite numbers')
        if value < 0 or (key == 'min_curvature' and value == 0):
            raise ValueError('invalid subpixel reliability threshold')
    if result['mode_margin'] >= 1 or result['fallback'] not in ('integer', 'invalid'):
        raise ValueError('invalid subpixel reliability policy')
    return result

def _pool(value, block):
    return F.avg_pool2d(value[:, None], block, 1, block // 2)[:, 0]

@stere_wire_match__torch.no_grad()
def match_wire(a, b, a_wire, b_wire, sign, *, method, low, high, block, texture, uniqueness, subpixel):
    """Match actual foreground centers, with a bounded [centers, disparities, patch] tile.

    Foreground membership comes only from the two RGB images and rectification
    masks. The downstream bidirectional and physical depth checks still apply.
    """
    (_, height, width) = a.shape
    high = min(high, width - 1)
    radius = block // 2
    offsets = stere_wire_match__torch.arange(-radius, radius + 1, device=a.device)
    (oy, ox) = stere_wire_match__torch.meshgrid(offsets, offsets, indexing='ij')
    (oy, ox) = (oy.flatten(), ox.flatten())
    center = len(ox) // 2
    disparities = stere_wire_match__torch.arange(low, high + 1, device=a.device, dtype=a.dtype)
    output = stere_wire_match__torch.full_like(a, float('nan'))
    confidence = stere_wire_match__torch.zeros_like(a)
    good = stere_wire_match__torch.zeros_like(a_wire)
    supported = stere_wire_match__torch.zeros_like(good)
    texture_ok = stere_wire_match__torch.zeros_like(good)
    unique_ok = stere_wire_match__torch.zeros_like(good)
    boundary = stere_wire_match__torch.zeros_like(good)
    quality_ok = stere_wire_match__torch.zeros_like(good)
    refined = stere_wire_match__torch.zeros_like(good)
    if not len(disparities):
        return (output, confidence, good, dict(search_boundary=boundary, optional_refinement_used=refined, reasons=dict(wire_no_joint_patch=~supported, wire_low_texture=~texture_ok, wire_ambiguous=~unique_ok, wire_cost_rejected=~quality_ok)))
    for pixels in a_wire.nonzero().split(_CENTERS_PER_TILE):
        if not len(pixels):
            continue
        (env, yy, xx) = pixels.unbind(1)
        rows = yy[:, None] + oy
        columns = xx[:, None] + ox
        rows_inside = (rows >= 0) & (rows < height)
        columns_inside = (columns >= 0) & (columns < width)
        sy = rows.clamp(0, height - 1)
        sx = columns.clamp(0, width - 1)
        source = a[env[:, None], sy, sx][:, None, :]
        source_mask = (a_wire[env[:, None], sy, sx] & rows_inside & columns_inside)[:, None, :]

        def costs(d):
            target_x = columns[:, None, :] + sign * d[..., None]
            inside = (target_x >= 0) & (target_x <= width - 1) & rows_inside[:, None, :]
            x0 = target_x.floor().long().clamp(0, width - 1)
            x1 = target_x.ceil().long().clamp(0, width - 1)
            (be, by) = (env[:, None, None], sy[:, None, :])
            target_mask = b_wire[be, by, x0] & b_wire[be, by, x1] & inside
            first = b[be, by, x0]
            target = first + (b[be, by, x1] - first) * (target_x - target_x.floor())
            joint = source_mask & target_mask
            count = joint.sum(-1)
            minimum = 0.6 * stere_wire_match__torch.maximum(source_mask.sum(-1), target_mask.sum(-1))
            support = target_mask[..., center] & (count >= 3) & (count >= minimum)
            if method == 'wire_silhouette':
                observed = inside.all(-1) & columns_inside.all(-1)[:, None]
                minority = stere_wire_match__torch.minimum(source_mask.sum(-1), (~source_mask).sum(-1))
                available = support & observed & (minority >= 2)
                mismatch = (source_mask ^ target_mask).float().mean(-1)
                return (mismatch.masked_fill(~available, float('inf')), support)
            denom = count.clamp_min(1)
            a_centered = source - (source * joint).sum(-1)[..., None] / denom[..., None]
            b_centered = target - (target * joint).sum(-1)[..., None] / denom[..., None]
            va = (a_centered.square() * joint).sum(-1) / denom
            vb = (b_centered.square() * joint).sum(-1) / denom
            has_texture = (va > max(texture, 0.0001) ** 2) & (vb > max(texture, 0.0001) ** 2)
            if method == 'wire_census':
                ra = source < source[..., center:center + 1]
                rb = target < target[..., center:center + 1]
                cost = ((ra ^ rb) & joint).sum(-1) / (count - 1).clamp_min(1)
            else:
                covariance = (a_centered * b_centered * joint).sum(-1) / denom
                cost = 1.0 - (covariance / (va * vb).sqrt().clamp_min(1e-12)).clamp(-1.0, 1.0)
            return (cost.masked_fill(~(support & has_texture), float('inf')), support)
        (planes, support) = costs(disparities[None, :])
        (best, winner) = planes.min(-1)
        winner_d = disparities[winner]
        alternative = planes.masked_fill((disparities - winner_d[:, None]).abs() <= 1, float('inf')).amin(-1)
        comparable = stere_wire_match__torch.isfinite(best) & stere_wire_match__torch.isfinite(alternative)
        margin = stere_wire_match__torch.where(comparable, (alternative - best) / alternative.abs().clamp_min(1e-08), 0.0)
        tied = (stere_wire_match__torch.isfinite(planes) & ((planes - best[:, None]).abs() <= 1e-06)).sum(-1) != 1
        clear = comparable & (margin > uniqueness) & ~tied
        max_cost = {'wire_silhouette': 0.0, 'wire_census': 0.25, 'wire_zncc': 0.2}[method]
        usable_cost = best <= max_cost
        edge = (winner_d == low) | (winner_d == high)
        accepted = clear & usable_cost & ~edge
        used = stere_wire_match__torch.zeros_like(accepted)
        if subpixel and method != 'wire_silhouette':
            phases = stere_wire_match__torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5], dtype=a.dtype, device=a.device)
            local = winner_d[:, None] + phases
            (fractional, _) = costs(local)
            (best_fractional, choice) = fractional.min(-1)
            used = accepted & (best_fractional < best - 1e-05)
            winner_d = stere_wire_match__torch.where(used, local.gather(1, choice[:, None])[:, 0], winner_d)
        idx = (env, yy, xx)
        supported[idx] = support.any(-1)
        texture_ok[idx] = stere_wire_match__torch.isfinite(best)
        unique_ok[idx] = clear
        quality_ok[idx] = usable_cost
        boundary[idx] = edge
        refined[idx] = used
        good[idx] = accepted
        output[idx] = stere_wire_match__torch.where(accepted, winner_d, float('nan'))
        confidence[idx] = stere_wire_match__torch.where(accepted, margin, 0.0)
    evidence_key = 'wire_no_shape_evidence' if method == 'wire_silhouette' else 'wire_low_texture'
    return (output, confidence, good, dict(search_boundary=boundary, optional_refinement_used=refined, metric=method, reasons={'wire_no_joint_patch': ~supported, evidence_key: ~texture_ok, 'wire_ambiguous': ~unique_ok, 'wire_cost_rejected': ~quality_ok}))

def refine_winner_neighborhood(integer_disparity, costs_minus2, costs_minus1, costs_center, costs_plus1, costs_plus2, *, second_nonlocal_cost=None, available=None, image_ambiguity=None, policy=None):
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
    if not all((isinstance(value, stere_match_candi__torch.Tensor) for value in (integer_disparity, *planes))):
        raise ValueError('integer disparity and five cost tensors required')
    if not costs_center.is_floating_point():
        raise ValueError('cost planes must have a floating dtype')
    (shape, device, dtype) = (costs_center.shape, costs_center.device, costs_center.dtype)
    if integer_disparity.shape != shape or integer_disparity.device != device:
        raise ValueError('integer disparity must match cost plane shape and device')
    if any((value.shape != shape or value.device != device or value.dtype != dtype for value in planes)):
        raise ValueError('cost planes must share shape, dtype and device')
    for (name, value) in (('available', available), ('image_ambiguity', image_ambiguity)):
        if value is not None and (not isinstance(value, stere_match_candi__torch.Tensor) or value.shape != shape or value.device != device or (value.dtype != stere_match_candi__torch.bool)):
            raise ValueError(name + ' must be a boolean mask with the cost plane shape and device')
    if second_nonlocal_cost is not None and (not isinstance(second_nonlocal_cost, stere_match_candi__torch.Tensor) or second_nonlocal_cost.shape != shape or second_nonlocal_cost.device != device or (not second_nonlocal_cost.is_floating_point())):
        raise ValueError('nonlocal cost must match cost plane shape and device')
    (cm2, cm1, c0, cp1, cp2) = planes
    finite = stere_match_candi__torch.isfinite(cm2) & stere_match_candi__torch.isfinite(cm1) & stere_match_candi__torch.isfinite(c0)
    finite = finite & stere_match_candi__torch.isfinite(cp1) & stere_match_candi__torch.isfinite(cp2)
    integer = integer_disparity.to(dtype=dtype)
    integer_ok = stere_match_candi__torch.isfinite(integer) & (integer > 0) & (integer == integer.round())
    supported = stere_match_candi__torch.ones_like(finite) if available is None else available
    image_ambiguous = stere_match_candi__torch.zeros_like(finite) if image_ambiguity is None else image_ambiguity
    floor = settings['min_curvature']
    scale = stere_match_candi__torch.maximum(stere_match_candi__torch.maximum(cm2.abs(), cm1.abs()), c0.abs())
    scale = stere_match_candi__torch.maximum(scale, stere_match_candi__torch.maximum(cp1.abs(), cp2.abs()))
    roundoff = scale * (32 * stere_match_candi__torch.finfo(dtype).eps)
    curvature = cm1 + cp1 - 2 * c0
    left_curvature = cm2 - 2 * cm1 + c0
    right_curvature = c0 - 2 * cp1 + cp2
    nonconvex = finite & ((left_curvature < -roundoff) | (curvature < -roundoff) | (right_curvature < -roundoff))
    low_curvature = finite & (curvature < floor)
    safe_curvature = stere_match_candi__torch.where(curvature > 0, curvature, stere_match_candi__torch.ones_like(curvature))
    raw_offset = 0.5 * (cm1 - cp1) / safe_curvature
    legacy_valid = stere_match_candi__torch.isfinite(cm1) & stere_match_candi__torch.isfinite(c0) & stere_match_candi__torch.isfinite(cp1) & (curvature > 1e-08)
    legacy_offset = stere_match_candi__torch.where(legacy_valid, raw_offset.clamp(-0.5, 0.5), stere_match_candi__torch.zeros_like(c0))
    slope = 0.5 * (cp1 - cm1)
    predicted_minus2 = c0 + 2 * curvature - 2 * slope
    predicted_plus2 = c0 + 2 * curvature + 2 * slope
    residual = stere_match_candi__torch.maximum((cm2 - predicted_minus2).abs(), (cp2 - predicted_plus2).abs())
    residual = residual / curvature.abs().clamp_min(floor)
    residual = stere_match_candi__torch.where(finite, residual, stere_match_candi__torch.full_like(residual, float('inf')))
    offset_roundoff = 32 * stere_match_candi__torch.finfo(dtype).eps
    quadratic_fit_valid = finite & (residual <= settings['fit_tolerance']) & (raw_offset.abs() <= 0.5 + offset_roundoff)
    (left_slope, right_slope) = (cm2 - cm1, cp2 - cp1)
    absolute_slope = 0.5 * (left_slope + right_slope)
    safe_absolute_slope = stere_match_candi__torch.where(absolute_slope > 0, absolute_slope, stere_match_candi__torch.ones_like(c0))
    absolute_offset = 0.5 * (cm1 - cp1) / safe_absolute_slope
    absolute_floor = 0.5 * (cm1 + cp1) - absolute_slope
    absolute_center = absolute_floor + absolute_slope * absolute_offset.abs()
    absolute_scale = absolute_slope.abs().clamp_min(floor)
    absolute_residual = (c0 - absolute_center).abs() / absolute_scale
    absolute_slope_residual = (left_slope - right_slope).abs() / absolute_scale
    absolute_fit_valid = finite & (absolute_slope >= floor) & (left_slope > 0) & (right_slope > 0) & (absolute_offset.abs() <= 0.5 + offset_roundoff) & (absolute_floor >= -roundoff) & (absolute_residual <= settings['absolute_fit_tolerance']) & (absolute_slope_residual <= settings['absolute_fit_tolerance'])
    winner_not_minimum = finite & ((cm1 < c0 - roundoff) | (cp1 < c0 - roundoff))
    competing_cost = stere_match_candi__torch.minimum(cm2, cp2)
    if second_nonlocal_cost is not None:
        competing_cost = stere_match_candi__torch.minimum(competing_cost, second_nonlocal_cost.to(dtype=dtype))
    margin_scale = stere_match_candi__torch.maximum(competing_cost.abs(), c0.abs()).clamp_min(stere_match_candi__torch.finfo(dtype).tiny)
    mode_margin = (competing_cost - c0) / margin_scale
    multimodal = finite & (mode_margin <= settings['mode_margin'])
    nonfinite_competitor = ~stere_match_candi__torch.isfinite(competing_cost) & (competing_cost != float('inf'))
    model_fit = quadratic_fit_valid | absolute_fit_valid
    reasons = {'invalid_integer': ~integer_ok, 'search_boundary': ~supported, 'nonfinite_cost': ~finite | nonfinite_competitor, 'low_curvature': low_curvature, 'nonconvex': nonconvex, 'winner_not_minimum': winner_not_minimum, 'multimodal': multimodal, 'poor_quadratic_fit': finite & ~model_fit, 'image_boundary_ambiguity': image_ambiguous}
    base_valid = integer_ok & finite & ~nonfinite_competitor & ~low_curvature & ~nonconvex & ~winner_not_minimum & ~multimodal
    refinement_valid = base_valid & supported & ~image_ambiguous & model_fit
    chosen_offset = stere_match_candi__torch.where(quadratic_fit_valid, raw_offset, absolute_offset).clamp(-0.5, 0.5)
    refined_offset = stere_match_candi__torch.where(refinement_valid, chosen_offset, stere_match_candi__torch.zeros_like(c0))
    valid = base_valid if settings['fallback'] == 'integer' else refinement_valid
    disparity = stere_match_candi__torch.where(valid, integer + refined_offset, stere_match_candi__torch.full_like(c0, float('nan')))
    model_selected = stere_match_candi__torch.where(quadratic_fit_valid, stere_match_candi__torch.ones_like(integer, dtype=stere_match_candi__torch.int64), stere_match_candi__torch.full_like(integer, 2, dtype=stere_match_candi__torch.int64))
    model_selected = stere_match_candi__torch.where(refinement_valid, model_selected, stere_match_candi__torch.zeros_like(model_selected))
    return {'disparity_px': disparity, 'valid_mask': valid, 'valid': valid, 'refinement_valid': refinement_valid, 'reasons': reasons, 'legacy_offset': legacy_offset, 'legacy_refinement_valid': legacy_valid, 'refined_offset': refined_offset, 'curvature': curvature, 'fit_residual': residual, 'quadratic_fit_residual': residual, 'quadratic_fit_valid': quadratic_fit_valid, 'absolute_fit_valid': absolute_fit_valid, 'absolute_fit_residual': absolute_residual, 'absolute_slope_residual': absolute_slope_residual, 'absolute_floor': absolute_floor, 'model_selected': model_selected}

def _erode(mask, block):
    return _pool(mask.to(stere_match_costs__torch.float32), block) >= 1.0

def reliable_subpixel_refinement(integer_disparity, costs, *, disparities, support=None, min_curvature=1e-06, fit_tolerance=0.25, absolute_fit_tolerance=0.05, mode_margin=0.05, fallback='integer', image_ambiguity=None):
    """Selected [P,D] diagnostic adapter; dense image cost volumes are rejected.

    Disparities must be contiguous integer hypotheses. Unsupported costs never
    become a remote competitor; a missing winner or neighbor prevents refine.
    """
    if not isinstance(costs, stere_match_candi__torch.Tensor) or costs.ndim != 2 or (not costs.is_floating_point()):
        raise ValueError('selected cost samples must have shape [P,D] and floating dtype')
    if not isinstance(integer_disparity, stere_match_candi__torch.Tensor) or integer_disparity.shape != costs.shape[:1]:
        raise ValueError('selected integer disparities must have shape [P]')
    if integer_disparity.device != costs.device:
        raise ValueError('selected integer disparities must use the cost device')
    grid = stere_match_candi__torch.as_tensor(disparities, device=costs.device, dtype=costs.dtype)
    if grid.ndim != 1 or grid.numel() != costs.shape[1] or grid.numel() < 1 or (not bool(stere_match_candi__torch.isfinite(grid).all())) or (not bool((grid == grid.round()).all())) or (grid.numel() > 1 and (not bool((grid[1:] - grid[:-1] == 1).all()))):
        raise ValueError('disparities must be contiguous integer hypotheses matching D')
    if support is not None and (not isinstance(support, stere_match_candi__torch.Tensor) or support.shape != costs.shape or support.device != costs.device or (support.dtype != stere_match_candi__torch.bool)):
        raise ValueError('selected support must be a boolean tensor with shape [P,D]')
    supported = stere_match_candi__torch.ones_like(costs, dtype=stere_match_candi__torch.bool) if support is None else support
    offset = integer_disparity.to(dtype=costs.dtype) - grid[0]
    safe_offset = stere_match_candi__torch.where(stere_match_candi__torch.isfinite(offset), offset, stere_match_candi__torch.zeros_like(offset))
    center_index = safe_offset.round().clamp(0, grid.numel() - 1).to(stere_match_candi__torch.int64)
    winner_present = (offset == safe_offset.round()) & (offset >= 0) & (offset < grid.numel())
    (neighbors, all_available) = ([], winner_present)
    for shift in (-2, -1, 0, 1, 2):
        index = center_index + shift
        in_range = (index >= 0) & (index < grid.numel())
        gather = index.clamp(0, grid.numel() - 1)[:, None]
        neighbor_support = supported.gather(1, gather).squeeze(1)
        cost = costs.gather(1, gather).squeeze(1)
        if shift == 0:
            cost = stere_match_candi__torch.where(winner_present & neighbor_support, cost, stere_match_candi__torch.full_like(cost, float('inf')))
        neighbors.append(cost)
        all_available = all_available & in_range & neighbor_support
    far = (grid[None] - integer_disparity[:, None]).abs() >= 2
    far_costs = stere_match_candi__torch.where(far & supported, costs, stere_match_candi__torch.full_like(costs, float('inf')))
    second_nonlocal_cost = far_costs.amin(dim=1)
    settings = dict(min_curvature=min_curvature, fit_tolerance=fit_tolerance, absolute_fit_tolerance=absolute_fit_tolerance, mode_margin=mode_margin, fallback=fallback)
    return refine_winner_neighborhood(integer_disparity, *neighbors, second_nonlocal_cost=second_nonlocal_cost, available=all_available, image_ambiguity=image_ambiguity, policy=settings)

def _std(value, block):
    mean = _pool(value, block)
    return (_pool(value.square(), block) - mean.square()).clamp_min(0).sqrt()

def photometric_refine_winner(a, b, integer_disparity, sign, block_size, a_valid, b_valid, *, min_curvature=1e-06, mode_margin=0.05):
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
    if not isinstance(a, stere_match_candi__torch.Tensor) or not isinstance(b, stere_match_candi__torch.Tensor):
        raise ValueError('grayscale image tensors required')
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    if b.ndim == 4 and b.shape[1] == 1:
        b = b[:, 0]
    if a.ndim != 3 or a.shape != b.shape or a.dtype != b.dtype or (a.device != b.device) or (not a.is_floating_point()):
        raise ValueError('grayscale images must share [B,H,W] shape, floating dtype and device')
    for (name, value) in (('a_valid', a_valid), ('b_valid', b_valid)):
        if not isinstance(value, stere_match_candi__torch.Tensor) or value.shape != a.shape or value.device != a.device or (value.dtype != stere_match_candi__torch.bool):
            raise ValueError(name + ' must be a boolean [B,H,W] image mask')
    if not isinstance(integer_disparity, stere_match_candi__torch.Tensor) or integer_disparity.shape != a.shape or integer_disparity.device != a.device:
        raise ValueError('integer disparity must match grayscale images')
    (batch, height, width) = a.shape
    integer = integer_disparity.to(dtype=a.dtype)
    integer_ok = stere_match_candi__torch.isfinite(integer) & (integer > 0) & (integer == integer.round())
    safe_integer = stere_match_candi__torch.where(integer_ok, integer, stere_match_candi__torch.zeros_like(integer))
    yy = stere_match_candi__torch.arange(height, device=a.device).view(1, height, 1).expand(batch, height, width)
    xx = stere_match_candi__torch.arange(width, device=a.device).view(1, 1, width).expand(batch, height, width)
    (a_flat, b_flat) = (a.reshape(batch, -1), b.reshape(batch, -1))
    av_flat = (a_valid & stere_match_candi__torch.isfinite(a)).reshape(batch, -1)
    bv_flat = (b_valid & stere_match_candi__torch.isfinite(b)).reshape(batch, -1)
    radius = block_size // 2

    def patch_cost(delta):
        target_x = xx.to(dtype=a.dtype) + sign * (safe_integer + delta)
        x0 = target_x.floor().to(stere_match_candi__torch.int64)
        x1 = target_x.ceil().to(stere_match_candi__torch.int64)
        alpha = target_x - x0.to(dtype=a.dtype)
        cost = stere_match_candi__torch.zeros_like(a)
        complete = integer_ok.clone()
        for dy in range(-radius, radius + 1):
            row = yy + dy
            row_valid = (row >= 0) & (row < height)
            row_base = row.clamp(0, height - 1) * width
            for dx in range(-radius, radius + 1):
                (ax, bx0, bx1) = (xx + dx, x0 + dx, x1 + dx)
                inside = row_valid & (ax >= 0) & (ax < width) & (bx0 >= 0) & (bx1 < width)
                ia = (row_base + ax.clamp(0, width - 1)).reshape(batch, -1)
                ib0 = (row_base + bx0.clamp(0, width - 1)).reshape(batch, -1)
                ib1 = (row_base + bx1.clamp(0, width - 1)).reshape(batch, -1)
                va = a_flat.gather(1, ia).reshape_as(a)
                vb0 = b_flat.gather(1, ib0).reshape_as(a)
                vb1 = b_flat.gather(1, ib1).reshape_as(a)
                supported = (av_flat.gather(1, ia) & bv_flat.gather(1, ib0) & bv_flat.gather(1, ib1)).reshape_as(a)
                complete = complete & inside & supported
                cost = cost + (va - (vb0 * (1 - alpha) + vb1 * alpha)).abs()
        cost = cost / (block_size * block_size)
        cost = stere_match_candi__torch.where(complete, cost, stere_match_candi__torch.full_like(cost, float('inf')))
        return (cost, complete)
    infinity = stere_match_candi__torch.full_like(a, float('inf'))
    (best, integer_cost) = (infinity.clone(), infinity.clone())
    winner_index = stere_match_candi__torch.zeros_like(integer, dtype=stere_match_candi__torch.int64)
    best_supported = stere_match_candi__torch.zeros_like(integer_ok)
    for index in range(13):
        (cost, supported) = patch_cost((index - 6) * 0.125)
        closer_to_integer = abs(index - 6) < (winner_index - 6).abs()
        replace = (cost < best) | (cost == best) & supported & closer_to_integer
        best = stere_match_candi__torch.where(replace, cost, best)
        winner_index = stere_match_candi__torch.where(replace, stere_match_candi__torch.full_like(winner_index, index), winner_index)
        best_supported = stere_match_candi__torch.where(replace, supported, best_supported)
        if index == 6:
            integer_cost = cost
    (minus, plus, second_nonlocal) = (infinity.clone(), infinity.clone(), infinity.clone())
    minus_supported = stere_match_candi__torch.zeros_like(integer_ok)
    plus_supported = stere_match_candi__torch.zeros_like(integer_ok)
    for index in range(13):
        (cost, supported) = patch_cost((index - 6) * 0.125)
        (minus_here, plus_here) = (winner_index == index + 1, winner_index == index - 1)
        (minus, plus) = (stere_match_candi__torch.where(minus_here, cost, minus), stere_match_candi__torch.where(plus_here, cost, plus))
        minus_supported = stere_match_candi__torch.where(minus_here, supported, minus_supported)
        plus_supported = stere_match_candi__torch.where(plus_here, supported, plus_supported)
        remote = (winner_index - index).abs() >= 4
        second_nonlocal = stere_match_candi__torch.minimum(second_nonlocal, stere_match_candi__torch.where(remote, cost, infinity))
    offset = (winner_index.to(dtype=a.dtype) - 6) * 0.125
    complete = best_supported & minus_supported & plus_supported & stere_match_candi__torch.isfinite(second_nonlocal)
    interior = offset.abs() <= 0.5
    curvature = minus + plus - 2 * best
    finite = stere_match_candi__torch.isfinite(best) & stere_match_candi__torch.isfinite(minus) & stere_match_candi__torch.isfinite(plus)
    margin_scale = stere_match_candi__torch.maximum(second_nonlocal.abs(), best.abs()).clamp_min(stere_match_candi__torch.finfo(a.dtype).tiny)
    cost_margin = (second_nonlocal - best) / margin_scale
    multimodal = cost_margin <= mode_margin
    low_curvature = finite & (curvature < min_curvature)
    valid = integer_ok & complete & interior & finite & ~low_curvature & ~multimodal
    disparity = stere_match_candi__torch.where(valid, integer + offset, stere_match_candi__torch.full_like(integer, float('nan')))
    return {'disparity_px': disparity, 'valid_mask': valid, 'valid': valid, 'refinement_valid': valid, 'photometric_offset': offset, 'best_cost': best, 'integer_cost': integer_cost, 'minus_cost': minus, 'plus_cost': plus, 'second_nonlocal_cost': second_nonlocal, 'curvature': curvature, 'cost_margin': cost_margin, 'model_selected': stere_match_candi__torch.where(valid, stere_match_candi__torch.full_like(winner_index, 3), stere_match_candi__torch.zeros_like(winner_index)), 'candidate_evaluations': stere_match_candi__torch.full_like(winner_index, 26), 'reasons': {'invalid_integer': ~integer_ok, 'search_boundary': ~complete | ~interior, 'nonfinite_cost': ~finite, 'low_curvature': low_curvature, 'multimodal': multimodal}}

def _features(value, valid, method):
    if method == 'zncc':
        return ((value,), valid)
    if method == 'gradient':
        gx = 0.5 * F.pad(value[:, :, 2:] - value[:, :, :-2], (1, 1))
        gy = 0.5 * F.pad(value[:, 2:] - value[:, :-2], (0, 0, 1, 1))
        vx = F.pad(valid[:, :, 2:] & valid[:, :, :-2], (1, 1), value=False)
        vy = F.pad(valid[:, 2:] & valid[:, :-2], (0, 0, 1, 1), value=False)
        return ((gx, gy), valid & vx & vy)
    code = stere_match_costs__torch.zeros_like(value, dtype=stere_match_costs__torch.int64)
    bit = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            neighbor = value.roll((-dy, -dx), dims=(-2, -1))
            code = code | (neighbor < value).to(stere_match_costs__torch.int64) << bit
            bit += 1
    return ((code,), _erode(valid, 5))

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
    if isinstance(search_span, bool) or not isinstance(search_span, (int, float)) or (not stere_match_candi__math.isfinite(search_span)) or (search_span <= 0):
        raise ValueError('positive finite disparity search span required')
    if not isinstance(gray, stere_match_candi__torch.Tensor):
        raise ValueError('grayscale image tensor required')
    if gray.ndim == 4 and gray.shape[1] == 1:
        gray = gray[:, 0]
    if gray.ndim != 3 or not gray.is_floating_point():
        raise ValueError('grayscale image must have shape [B,H,W]')
    if not isinstance(valid, stere_match_candi__torch.Tensor) or valid.shape != gray.shape or valid.device != gray.device or (valid.dtype != stere_match_candi__torch.bool):
        raise ValueError('valid must be a boolean image mask with shape [B,H,W]')
    (batch, height, width) = gray.shape
    observed = valid & stere_match_candi__torch.isfinite(gray)
    triplet = observed & observed.roll(1, -1) & observed.roll(-1, -1)
    triplet = triplet & (stere_match_candi__torch.arange(width, device=gray.device).view(1, 1, width) > 0)
    triplet = triplet & (stere_match_candi__torch.arange(width, device=gray.device).view(1, 1, width) < width - 1)
    samples = gray.to(stere_match_candi__torch.float64)
    x = stere_match_candi__torch.where(triplet, samples, stere_match_candi__torch.zeros_like(samples))
    y = stere_match_candi__torch.where(triplet, samples.roll(1, -1) + samples.roll(-1, -1), stere_match_candi__torch.zeros_like(samples))

    def window_sum(value):
        pooled = stere_match_candi__torch.nn.functional.avg_pool1d(value.reshape(batch * height, 1, width), 31, stride=1, padding=15)
        return pooled.reshape_as(samples) * 31
    count = window_sum(triplet.to(stere_match_candi__torch.float64))
    divisor = count.clamp_min(1)
    (mean_x, mean_y) = (window_sum(x) / divisor, window_sum(y) / divisor)
    (second_x, second_y) = (window_sum(x.square()) / divisor, window_sum(y.square()) / divisor)
    cross = window_sum(x * y) / divisor
    variance_x = (second_x - mean_x.square()).clamp_min(0)
    variance_y = (second_y - mean_y.square()).clamp_min(0)
    covariance = cross - mean_x * mean_y
    numerical_variance_floor = 128 * stere_match_candi__torch.finfo(stere_match_candi__torch.float64).eps * second_x.clamp_min(1)
    nonconstant = variance_x > numerical_variance_floor
    a = covariance / variance_x.clamp_min(numerical_variance_floor)
    b = mean_y - a * mean_x
    residual_variance = (variance_y - a * covariance).clamp_min(0)
    normalized_residual = (residual_variance / variance_x.clamp_min(numerical_variance_floor)).sqrt()
    normalized_residual = stere_match_candi__torch.where(nonconstant, normalized_residual, stere_match_candi__torch.full_like(normalized_residual, float('inf')))
    coefficient_supported = nonconstant & (a >= -2 - 1e-10) & (a < 2)
    omega = (a.clamp(-2, 2) / 2).acos()
    period = 2 * stere_match_candi__math.pi / omega
    period = stere_match_candi__torch.where(coefficient_supported, period, stere_match_candi__torch.full_like(period, float('nan')))
    positions = stere_match_candi__torch.arange(width, device=gray.device, dtype=stere_match_candi__torch.float64).view(1, 1, width)
    maximum_positions = stere_match_candi__torch.where(triplet, positions, stere_match_candi__torch.full_like(samples, -float('inf')))
    minimum_positions = stere_match_candi__torch.where(triplet, positions, stere_match_candi__torch.full_like(samples, float('inf')))
    last = stere_match_candi__torch.nn.functional.max_pool1d(maximum_positions.reshape(batch * height, 1, width), 31, stride=1, padding=15).reshape_as(samples)
    first = -stere_match_candi__torch.nn.functional.max_pool1d(-minimum_positions.reshape(batch * height, 1, width), 31, stride=1, padding=15).reshape_as(samples)
    window_span = stere_match_candi__torch.where(count > 0, last - first + 1, stere_match_candi__torch.zeros_like(count))
    repeated = observed & (count >= 17) & coefficient_supported & (normalized_residual <= 1e-05) & (period <= search_span) & (window_span >= 2 * period)
    return {'repeated_texture': repeated, 'mask': repeated, 'support': count.round().to(stere_match_candi__torch.int64), 'period_px': period, 'normalized_residual': normalized_residual, 'coefficient_a': stere_match_candi__torch.where(nonconstant, a, stere_match_candi__torch.full_like(a, float('nan'))), 'coefficient_b': stere_match_candi__torch.where(nonconstant, b, stere_match_candi__torch.full_like(b, float('nan'))), 'window_span_px': window_span}

def _hamming(left, right):
    code = left ^ right
    code = code - (code >> 1 & 1431655765)
    code = (code & 858993459) + (code >> 2 & 858993459)
    code = code + (code >> 4) & 252645135
    return code * 16843009 >> 24 & 255
stere_match_costs__repeated_texture_evidence = repeated_texture_evidence

def _relative_margin(other, best):
    finite = stere_match_costs__torch.isfinite(other) & stere_match_costs__torch.isfinite(best)
    scale = stere_match_costs__torch.maximum(other.abs(), best.abs()).clamp_min(stere_match_costs__torch.finfo(best.dtype).tiny)
    value = (other - best) / scale
    return stere_match_costs__torch.where(finite, value.clamp(0, 1), stere_match_costs__torch.zeros_like(best))

class _Metric:
    """Precomputed image features; no object state has a disparity dimension."""

    def __init__(self, a, b, av, bv, method, block, sign, texture):
        self.output_dtype = a.dtype
        if method == 'zncc':
            (a, b) = (a.to(stere_match_costs__torch.float64), b.to(stere_match_costs__torch.float64))
        (self.a, self.b, self.av, self.bv) = (a, b, av, bv)
        (self.method, self.block, self.sign, self.texture) = (method, block, sign, texture)
        (self.af, afv) = _features(a, av, method)
        (self.bf, self.bfv) = _features(b, bv, method)
        self.a_patch = _erode(afv, block)
        self.b_patch = _erode(self.bfv, block)
        (self.a_std, self.b_std) = (_std(a, block), _std(b, block))
        self.a_texture = self.a_std > max(texture, 1e-08)
        self.x = stere_match_costs__torch.arange(a.shape[-1], device=a.device).view(1, 1, -1)
        self.y = stere_match_costs__torch.arange(a.shape[-2], device=a.device).view(1, -1, 1)
        self.a_mean = _pool(a, block)
        self.a_variance = (_pool(a.square(), block) - self.a_mean.square()).clamp_min(0)
        if method == 'zncc':
            self.a_texture = self.a_texture & (self.a_variance > 1e-08)

    def _horizontal(self, value, column):
        return value.gather(2, column.clamp(0, value.shape[-1] - 1).expand_as(self.a))

    def integer_cost(self, disparity):
        column = self.x + self.sign * disparity
        shifted = tuple((self._horizontal(value, column) for value in self.bf))
        inside = (column >= 0) & (column < self.a.shape[-1])
        support = inside & self.a_patch & self._horizontal(self.b_patch, column) & self.a_texture & (self._horizontal(self.b_std, column) > max(self.texture, 1e-08))
        if self.method == 'census':
            cost = _pool(_hamming(self.af[0], shifted[0]).to(self.a.dtype) / 24.0, self.block)
        elif self.method == 'gradient':
            cost = 0.5 * _pool(sum(((a - b).abs() for (a, b) in zip(self.af, shifted))), self.block)
        else:
            mean_b = self._horizontal(_pool(self.b, self.block), column)
            variance_b = self._horizontal(self.b_std.square(), column)
            cross = _pool(self.a * shifted[0], self.block) - self.a_mean * mean_b
            denominator = (self.a_variance * variance_b).clamp_min(0).sqrt()
            cost = 1 - (cross / denominator.clamp_min(stere_match_costs__torch.finfo(self.a.dtype).tiny)).clamp(-1, 1)
            support = support & (variance_b > 1e-08) & (denominator > stere_match_costs__torch.finfo(self.a.dtype).eps ** 2)
        return stere_match_costs__torch.where(support & stere_match_costs__torch.isfinite(cost), cost, stere_match_costs__torch.full_like(cost, float('inf'))).to(self.output_dtype)

    def fractional_cost(self, winner, offset):
        """Direct metric at uniform center displacement, never pixelwise warp pooling."""
        phase_x = self.x.to(self.a.dtype) + self.sign * offset
        (low, high) = (phase_x.floor().long(), phase_x.ceil().long())
        fraction = phase_x - low
        (lo, hi) = (self._horizontal(self.b, low), self._horizontal(self.b, high))
        phase = lo + (hi - lo) * fraction
        phase_valid = (low >= 0) & (high < self.a.shape[-1]) & self._horizontal(self.bv, low) & self._horizontal(self.bv, high)
        (bf, bf_valid) = _features(phase, phase_valid, self.method)
        (width, height, batch) = (self.a.shape[-1], self.a.shape[-2], self.a.shape[0])
        safe_winner = stere_match_costs__torch.where(stere_match_costs__torch.isfinite(winner), winner, stere_match_costs__torch.zeros_like(winner)).long()
        target = self.x + self.sign * safe_winner
        complete = self.a_patch & self.a_texture & stere_match_costs__torch.isfinite(winner)
        (total, sum_b, square_b) = (stere_match_costs__torch.zeros_like(self.a), stere_match_costs__torch.zeros_like(self.a), stere_match_costs__torch.zeros_like(self.a))
        radius = self.block // 2

        def gather(value, row, column):
            index = row.clamp(0, height - 1) * width + column.clamp(0, width - 1)
            return value.reshape(batch, -1).gather(1, index.expand_as(self.a).reshape(batch, -1)).reshape_as(self.a)
        for dy in range(-radius, radius + 1):
            row = self.y + dy
            for dx in range(-radius, radius + 1):
                column = target + dx
                inside = (row >= 0) & (row < height) & (column >= 0) & (column < width)
                complete = complete & inside & gather(bf_valid, row, column)
                raw_b = gather(phase, row, column)
                sum_b = sum_b + raw_b
                square_b = square_b + raw_b.square()
                if self.method == 'census':
                    left = gather(self.af[0], row, self.x + dx)
                    right = gather(bf[0], row, column)
                    total = total + _hamming(left, right).to(self.a.dtype) / 24.0
                elif self.method == 'gradient':
                    for (left, right) in zip(self.af, bf):
                        total = total + 0.5 * (gather(left, row, self.x + dx) - gather(right, row, column)).abs()
                else:
                    total = total + gather(self.a, row, self.x + dx) * raw_b
        count = self.block ** 2
        mean_b = sum_b / count
        variance_b = (square_b / count - mean_b.square()).clamp_min(0)
        complete = complete & (variance_b.sqrt() > max(self.texture, 1e-08))
        if self.method == 'zncc':
            denominator = (self.a_variance * variance_b).sqrt()
            cost = 1 - ((total / count - self.a_mean * mean_b) / denominator.clamp_min(stere_match_costs__torch.finfo(self.a.dtype).tiny)).clamp(-1, 1)
            complete = complete & (variance_b > 1e-08) & (denominator > stere_match_costs__torch.finfo(self.a.dtype).eps ** 2)
        else:
            cost = total / count
        return stere_match_costs__torch.where(complete & stere_match_costs__torch.isfinite(cost), cost, stere_match_costs__torch.full_like(cost, float('inf'))).to(self.output_dtype)

@stere_match_costs__torch.no_grad()
def match_cost(a, b, sign, *, method, min_disparity, max_disparity, block_size, subpixel, uniqueness_threshold, texture_threshold, a_valid=None, b_valid=None, prior=None, search_radius=2, search_mask=None, prune_search=False, count_search=False):
    """Return (disparity, confidence, good, trace), all tensor planes [B,H,W].

    Invalid disparities remain NaN and invalid confidence is zero. A nonfinite
    local prior has no candidates; local_failed requests upper-layer fallback.
    Mask arguments are RAW masks, not pre-eroded matching-center masks.
    Global bounds include zero for pyramid levels; depth callers reject d<=0.
    search_mask limits candidate centers and does not erase neighboring pixels
    from their patches. Opt-in pruning limits the integer plane loop to the
    feasible prior range and skips planes without eligible centers. Pruned
    searches always return their work counters. Full scans report them only
    with count_search=True.
    """
    _validate(a, b, sign, method, min_disparity, max_disparity, block_size, subpixel, uniqueness_threshold, texture_threshold, a_valid, b_valid, prior, search_radius)
    if type(prune_search) is not bool or type(count_search) is not bool:
        raise ValueError('prune_search and count_search must be bool')
    count_search |= prune_search
    if search_mask is None:
        search_mask = stere_match_costs__torch.ones_like(a, dtype=stere_match_costs__torch.bool)
    if not isinstance(search_mask, stere_match_costs__torch.Tensor) or search_mask.shape != a.shape or search_mask.dtype != stere_match_costs__torch.bool or (search_mask.device != a.device):
        raise ValueError('search_mask requires bool [B,H,W] on the image device')
    if prune_search and len(a) > 1:

        def image(value, i):
            return None if value is None else value[i:i + 1]
        parts = [match_cost(a[i:i + 1], b[i:i + 1], sign, method=method, min_disparity=min_disparity, max_disparity=max_disparity, block_size=block_size, subpixel=subpixel, uniqueness_threshold=uniqueness_threshold, texture_threshold=texture_threshold, a_valid=image(a_valid, i), b_valid=image(b_valid, i), prior=image(prior, i), search_radius=search_radius, search_mask=image(search_mask, i), prune_search=True) for i in range(len(a))]

        def join(values):
            first = values[0]
            if isinstance(first, stere_match_costs__torch.Tensor):
                return stere_match_costs__torch.cat(values)
            if isinstance(first, dict):
                return {key: join([v[key] for v in values]) for key in first}
            return first
        return tuple((join(values) for values in zip(*parts)))
    av = stere_match_costs__torch.isfinite(a) & (stere_match_costs__torch.ones_like(a, dtype=stere_match_costs__torch.bool) if a_valid is None else a_valid)
    bv = stere_match_costs__torch.isfinite(b) & (stere_match_costs__torch.ones_like(b, dtype=stere_match_costs__torch.bool) if b_valid is None else b_valid)
    (safe_a, safe_b) = (stere_match_costs__torch.where(av, a, 0.0), stere_match_costs__torch.where(bv, b, 0.0))
    metric = _Metric(safe_a, safe_b, av, bv, method, block_size, sign, texture_threshold)
    stop = min(max_disparity, a.shape[-1] - 1)
    infinity = stere_match_costs__torch.full_like(a, float('inf'))
    (best, winner) = (infinity.clone(), stere_match_costs__torch.full_like(a, float('nan')))
    integer_candidates = stere_match_costs__torch.zeros_like(a, dtype=stere_match_costs__torch.int32) if count_search else None
    fractional_candidates = stere_match_costs__torch.zeros_like(integer_candidates) if count_search else None
    integer_plane_calls = 0
    fractional_plane_calls = 0

    def eligible(d):
        if prior is None:
            return search_mask
        return search_mask & stere_match_costs__torch.isfinite(prior) & ((prior - d).abs() <= search_radius)
    (low, high) = (min_disparity, stop)
    if prune_search:
        if prior is None:
            if not bool(search_mask.any()):
                high = low - 1
        else:
            viable_prior = prior[search_mask & stere_match_costs__torch.isfinite(prior)]
            if viable_prior.numel():
                low = max(low, stere_match_costs__math.ceil(float(viable_prior.min()) - search_radius))
                high = min(high, stere_match_costs__math.floor(float(viable_prior.max()) + search_radius))
            else:
                high = low - 1

    def integer_planes():
        nonlocal integer_plane_calls
        for d in range(low, high + 1):
            candidates = eligible(d)
            if prune_search and (not bool(candidates.any())):
                continue
            if count_search:
                integer_candidates.add_(candidates)
            integer_plane_calls += 1
            yield (d, stere_match_costs__torch.where(candidates, metric.integer_cost(d), infinity))
    for (d, cost) in integer_planes():
        win = cost < best
        (best, winner) = (stere_match_costs__torch.where(win, cost, best), stere_match_costs__torch.where(win, float(d), winner))
    far = infinity.clone()
    neighbors = [infinity.clone() for _ in range(5)]
    for (d, cost) in integer_planes():
        far = stere_match_costs__torch.minimum(far, stere_match_costs__torch.where((winner - d).abs() >= 2, cost, infinity))
        for (index, offset) in enumerate((-2, -1, 0, 1, 2)):
            neighbors[index] = stere_match_costs__torch.where(winner + offset == d, cost, neighbors[index])
    confidence = _relative_margin(far, best)
    ambiguous = ~stere_match_costs__torch.isfinite(far) | (confidence <= max(uniqueness_threshold, 1e-06))
    integer_good = stere_match_costs__torch.isfinite(best) & ~ambiguous
    periodic = stere_match_costs__repeated_texture_evidence(safe_a, av, max(1, max_disparity - min_disparity))
    period = periodic['period_px']
    alias = (winner - period >= min_disparity) | (winner + period <= stop)
    repeated = periodic['repeated_texture'] & alias
    integer_good = integer_good & ~repeated
    if prior is None:
        search_boundary = (winner == min_disparity) | (winner == stop)
    else:
        lower = (prior - search_radius).ceil().clamp_min(min_disparity)
        upper = (prior + search_radius).floor().clamp_max(stop)
        search_boundary = (winner == lower) | (winner == upper)
    search_boundary = search_boundary | ~stere_match_costs__torch.isfinite(winner)
    integer_confidence = stere_match_costs__torch.where(integer_good, confidence, 0.0)
    refinement_valid = stere_match_costs__torch.zeros_like(av)
    disparity = winner.clone()
    (fractional_best, fractional_minus, fractional_plus, fractional_far) = (infinity.clone() for _ in range(4))
    fractional_offset = stere_match_costs__torch.zeros_like(a)
    fractional_tie = stere_match_costs__torch.zeros_like(av)

    def fractional_planes():
        nonlocal fractional_plane_calls
        for index in range(13):
            offset = (index - 6) * 0.125
            candidate = winner + offset
            allowed = (candidate >= min_disparity) & (candidate <= stop) & eligible(candidate) & integer_good
            if prune_search and (not bool(allowed.any())):
                continue
            if count_search:
                fractional_candidates.add_(allowed)
            fractional_plane_calls += 1
            yield (index, stere_match_costs__torch.where(allowed, metric.fractional_cost(winner, offset), infinity))
    if subpixel:
        selected = stere_match_costs__torch.full_like(a, 6, dtype=stere_match_costs__torch.int64)
        tie_count = stere_match_costs__torch.zeros_like(selected)
        for (index, cost) in fractional_planes():
            win = cost < fractional_best
            tie = stere_match_costs__torch.isfinite(cost) & (cost == fractional_best)
            tie_count = stere_match_costs__torch.where(win, stere_match_costs__torch.ones_like(tie_count), tie_count + tie.to(tie_count.dtype))
            closer = abs(index - 6) < (selected - 6).abs()
            replace = win | tie & closer
            selected = stere_match_costs__torch.where(replace, index, selected)
            fractional_best = stere_match_costs__torch.minimum(fractional_best, cost)
        for (index, cost) in fractional_planes():
            fractional_minus = stere_match_costs__torch.where(selected == index + 1, cost, fractional_minus)
            fractional_plus = stere_match_costs__torch.where(selected == index - 1, cost, fractional_plus)
            fractional_far = stere_match_costs__torch.minimum(fractional_far, stere_match_costs__torch.where((selected - index).abs() >= 4, cost, infinity))
        fractional_offset = (selected.to(a.dtype) - 6) * 0.125
        curvature = fractional_minus + fractional_plus - 2 * fractional_best
        fractional_tie = tie_count > 1
        refinement_valid = integer_good & stere_match_costs__torch.isfinite(fractional_minus) & stere_match_costs__torch.isfinite(fractional_plus) & stere_match_costs__torch.isfinite(fractional_best) & (fractional_offset.abs() <= 0.5) & ~fractional_tie & (curvature > 1e-08) & (_relative_margin(fractional_far, fractional_best) > max(uniqueness_threshold, 1e-06))
        disparity = stere_match_costs__torch.where(refinement_valid, winner + fractional_offset, winner)
    good = integer_good
    disparity = stere_match_costs__torch.where(good, disparity, stere_match_costs__torch.full_like(a, float('nan')))
    trace = dict(integer_disparity=winner, integer_confidence=integer_confidence, integer_good=integer_good, refinement_valid=refinement_valid, search_boundary=search_boundary, local_failed=~integer_good | search_boundary if prior is not None else stere_match_costs__torch.zeros_like(av), best_cost=best, raw_cost=best, second_nonlocal_cost=far, minus2_cost=neighbors[0], minus_cost=neighbors[1], plus_cost=neighbors[3], plus2_cost=neighbors[4], fractional_best_cost=fractional_best, fractional_minus_cost=fractional_minus, fractional_plus_cost=fractional_plus, fractional_nonlocal_cost=fractional_far, fractional_offset=fractional_offset, reasons=dict(low_texture=~metric.a_texture, patch_support=~metric.a_patch, no_candidate=~stere_match_costs__torch.isfinite(best), separated_mode_ambiguity=ambiguous, repeated_texture=repeated, fractional_tie=fractional_tie, integer_fallback=integer_good & ~refinement_valid if subpixel else stere_match_costs__torch.zeros_like(av)), metric=method, subpixel_policy=dict(metric=method, sample_step_px=0.125, sample_count=13, direct_metric_validation=True, center_disparity_uniform_within_patch=True, fallback='integer', census_plateau_policy='integer_fallback', confidence_is_probability=False), resource_metadata=dict(cost_volume_allocated=False, disparity_planes_in_memory=1, cached_cost_state_planes=None, cost_state_bound_independent_of_disparity=True, descriptor_stencil_px=5 if method == 'census' else None, fractional_evaluations=26 if subpixel else 0, confidence_kind='relative_cost_margin', confidence_is_probability=False, implementation='experimental_pytorch_streaming', gpu_memory_measured=False))
    if count_search:
        integer_calls = stere_match_costs__torch.full((len(a),), integer_plane_calls, dtype=stere_match_costs__torch.int64, device=a.device)
        fractional_calls = stere_match_costs__torch.full_like(integer_calls, fractional_plane_calls)
        plane_pixels = a.shape[-2] * a.shape[-1]
        trace['search_statistics'] = dict(integer_candidates=integer_candidates, fractional_candidates=fractional_candidates, integer_plane_calls=integer_calls, fractional_plane_calls=fractional_calls, integer_cost_evaluations=integer_calls * plane_pixels, fractional_cost_evaluations=fractional_calls * plane_pixels, total_cost_evaluations=(integer_calls + fractional_calls) * plane_pixels)
    return (disparity, integer_confidence, good, trace)
