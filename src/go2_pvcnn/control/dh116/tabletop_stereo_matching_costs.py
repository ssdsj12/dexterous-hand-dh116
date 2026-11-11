"""Experimental RGB-only costs with bounded disparity memory.

These are transparent PyTorch reference candidates, not native stereo engines.
Census is fixed 5x5 (24 strict comparisons), aggregated by mean Hamming distance;
ZNCC costs 1-rho; gradient costs mean absolute central x/y differences. Positive
disparity samples b at x + sign*d. Raw masks and all metric stencils must be
supported. Nonfinite intensities are missing observations, never usable fills.

An integer scan and a second separated-mode scan each hold one disparity plane.
Subpixel search directly evaluates the SAME metric on thirteen linear-sampling
phases (-.75:.125:.75), twice, retaining a constant number of image planes. Each
patch uses its CENTER's disparity throughout. Only observed interior, uniquely
supported local minima are returned. Binary plateaus, insufficient support and
uncertain fractional evidence retain an explicitly diagnosed integer result.
Confidence is a separated integer relative cost margin, not a probability.
"""
import math

import torch
import torch.nn.functional as F

from .tabletop_stereo_matcher_candidates import repeated_texture_evidence


def _validate(a, b, sign, method, low, high, block, subpixel, unique, texture,
              a_valid, b_valid, prior, radius):
    if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
            or a.ndim != 3 or a.shape != b.shape or a.device != b.device
            or a.dtype != b.dtype or a.dtype not in (torch.float32, torch.float64)
            or min(a.shape) < 1):
        raise ValueError('gray images must share nonempty [B,H,W], float32/float64 dtype and device')
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError('matching sign must be -1 or 1')
    if method not in ('census', 'zncc', 'gradient'):
        raise ValueError('method must be census, zncc or gradient')
    if type(low) is not int or type(high) is not int or low < 0 or high < low:
        raise ValueError('disparity bounds must be ordered nonnegative integers')
    if type(block) is not int or block < 3 or block % 2 != 1 or min(a.shape[-2:]) < block:
        raise ValueError('block size must be odd, at least three and fit the image')
    if type(subpixel) is not bool:
        raise ValueError('subpixel must be bool')
    for value, name in ((unique, 'uniqueness'), (texture, 'texture')):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(name + ' threshold must be a finite nonnegative number')
    if unique >= 1:
        raise ValueError('uniqueness threshold must be below one')
    if type(radius) is not int or radius < 0:
        raise ValueError('search_radius must be a nonnegative integer')
    for name, mask in (('a_valid', a_valid), ('b_valid', b_valid)):
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.shape != a.shape
                                 or mask.device != a.device or mask.dtype != torch.bool):
            raise ValueError(name + ' must be a raw bool [B,H,W] mask on the image device')
    if prior is not None and (not isinstance(prior, torch.Tensor) or prior.shape != a.shape
                              or prior.device != a.device or not prior.is_floating_point()):
        raise ValueError('prior must be a floating [B,H,W] image on the image device')


def _pool(value, block):
    return F.avg_pool2d(value[:, None], block, 1, block // 2)[:, 0]


def _erode(mask, block):
    # count_include_pad=True deliberately invalidates the image border.
    return _pool(mask.to(torch.float32), block) >= 1.


def _std(value, block):
    mean = _pool(value, block)
    return (_pool(value.square(), block) - mean.square()).clamp_min(0).sqrt()


def _features(value, valid, method):
    if method == 'zncc':
        return (value,), valid
    if method == 'gradient':
        gx = .5 * F.pad(value[:, :, 2:] - value[:, :, :-2], (1, 1))
        gy = .5 * F.pad(value[:, 2:] - value[:, :-2], (0, 0, 1, 1))
        vx = F.pad(valid[:, :, 2:] & valid[:, :, :-2], (1, 1), value=False)
        vy = F.pad(valid[:, 2:] & valid[:, :-2], (0, 0, 1, 1), value=False)
        return (gx, gy), valid & vx & vy
    code = torch.zeros_like(value, dtype=torch.int64)
    bit = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            neighbor = value.roll((-dy, -dx), dims=(-2, -1))
            code = code | ((neighbor < value).to(torch.int64) << bit)
            bit += 1
    return (code,), _erode(valid, 5)


def _hamming(left, right):
    code = left ^ right
    # Parallel bit count on the fixed 24 useful bits; no descriptor axis.
    code = code - ((code >> 1) & 0x55555555)
    code = (code & 0x33333333) + ((code >> 2) & 0x33333333)
    code = (code + (code >> 4)) & 0x0F0F0F0F
    return (((code * 0x01010101) >> 24) & 0xFF)


def _relative_margin(other, best):
    finite = torch.isfinite(other) & torch.isfinite(best)
    scale = torch.maximum(other.abs(), best.abs()).clamp_min(torch.finfo(best.dtype).tiny)
    value = (other - best) / scale
    return torch.where(finite, value.clamp(0, 1), torch.zeros_like(best))


class _Metric:
    """Precomputed image features; no object state has a disparity dimension."""

    def __init__(self, a, b, av, bv, method, block, sign, texture):
        self.output_dtype = a.dtype
        if method == 'zncc':
            # Normalized moments subtract nearly equal terms. Float64 prevents
            # a mere affine brightness change from fabricating a cost margin.
            a, b = a.to(torch.float64), b.to(torch.float64)
        self.a, self.b, self.av, self.bv = a, b, av, bv
        self.method, self.block, self.sign, self.texture = method, block, sign, texture
        self.af, afv = _features(a, av, method)
        self.bf, self.bfv = _features(b, bv, method)
        self.a_patch = _erode(afv, block)
        self.b_patch = _erode(self.bfv, block)
        self.a_std, self.b_std = _std(a, block), _std(b, block)
        self.a_texture = self.a_std > max(texture, 1e-8)
        self.x = torch.arange(a.shape[-1], device=a.device).view(1, 1, -1)
        self.y = torch.arange(a.shape[-2], device=a.device).view(1, -1, 1)
        self.a_mean = _pool(a, block)
        self.a_variance = (_pool(a.square(), block) - self.a_mean.square()).clamp_min(0)
        if method == 'zncc':
            self.a_texture = self.a_texture & (self.a_variance > 1e-8)

    def _horizontal(self, value, column):
        return value.gather(2, column.clamp(0, value.shape[-1] - 1).expand_as(self.a))

    def integer_cost(self, disparity):
        column = self.x + self.sign * disparity
        shifted = tuple(self._horizontal(value, column) for value in self.bf)
        inside = (column >= 0) & (column < self.a.shape[-1])
        support = (inside & self.a_patch & self._horizontal(self.b_patch, column)
                   & self.a_texture & (self._horizontal(self.b_std, column) > max(self.texture, 1e-8)))
        if self.method == 'census':
            cost = _pool(_hamming(self.af[0], shifted[0]).to(self.a.dtype) / 24., self.block)
        elif self.method == 'gradient':
            cost = .5 * _pool(sum((a - b).abs() for a, b in zip(self.af, shifted)), self.block)
        else:
            mean_b = self._horizontal(_pool(self.b, self.block), column)
            variance_b = self._horizontal(self.b_std.square(), column)
            cross = _pool(self.a * shifted[0], self.block) - self.a_mean * mean_b
            denominator = (self.a_variance * variance_b).clamp_min(0).sqrt()
            cost = 1 - (cross / denominator.clamp_min(torch.finfo(self.a.dtype).tiny)).clamp(-1, 1)
            support = support & (variance_b > 1e-8) & (denominator > torch.finfo(self.a.dtype).eps ** 2)
        return torch.where(support & torch.isfinite(cost), cost, torch.full_like(cost, float('inf'))).to(self.output_dtype)

    def fractional_cost(self, winner, offset):
        """Direct metric at uniform center displacement, never pixelwise warp pooling."""
        phase_x = self.x.to(self.a.dtype) + self.sign * offset
        low, high = phase_x.floor().long(), phase_x.ceil().long()
        fraction = phase_x - low
        lo, hi = self._horizontal(self.b, low), self._horizontal(self.b, high)
        phase = lo + (hi - lo) * fraction
        phase_valid = ((low >= 0) & (high < self.a.shape[-1])
                       & self._horizontal(self.bv, low) & self._horizontal(self.bv, high))
        bf, bf_valid = _features(phase, phase_valid, self.method)
        width, height, batch = self.a.shape[-1], self.a.shape[-2], self.a.shape[0]
        safe_winner = torch.where(torch.isfinite(winner), winner, torch.zeros_like(winner)).long()
        target = self.x + self.sign * safe_winner
        complete = self.a_patch & self.a_texture & torch.isfinite(winner)
        total, sum_b, square_b = torch.zeros_like(self.a), torch.zeros_like(self.a), torch.zeros_like(self.a)
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
                    total = total + _hamming(left, right).to(self.a.dtype) / 24.
                elif self.method == 'gradient':
                    for left, right in zip(self.af, bf):
                        total = total + .5 * (gather(left, row, self.x + dx) - gather(right, row, column)).abs()
                else:
                    total = total + gather(self.a, row, self.x + dx) * raw_b
        count = self.block ** 2
        mean_b = sum_b / count
        variance_b = (square_b / count - mean_b.square()).clamp_min(0)
        complete = complete & (variance_b.sqrt() > max(self.texture, 1e-8))
        if self.method == 'zncc':
            denominator = (self.a_variance * variance_b).sqrt()
            cost = 1 - ((total / count - self.a_mean * mean_b)
                        / denominator.clamp_min(torch.finfo(self.a.dtype).tiny)).clamp(-1, 1)
            complete = complete & (variance_b > 1e-8) & (denominator > torch.finfo(self.a.dtype).eps ** 2)
        else:
            cost = total / count
        return torch.where(complete & torch.isfinite(cost), cost, torch.full_like(cost, float('inf'))).to(self.output_dtype)


@torch.no_grad()
def match_cost(a, b, sign, *, method, min_disparity, max_disparity, block_size,
               subpixel, uniqueness_threshold, texture_threshold, a_valid=None,
               b_valid=None, prior=None, search_radius=2):
    """Return (disparity, confidence, good, trace), all tensor planes [B,H,W].

    Invalid disparities remain NaN and invalid confidence is zero. A nonfinite
    local prior has no candidates; local_failed requests upper-layer fallback.
    Mask arguments are RAW masks, not pre-eroded matching-center masks.
    Global bounds include zero for pyramid levels; depth callers reject d<=0.
    """
    _validate(a, b, sign, method, min_disparity, max_disparity, block_size,
              subpixel, uniqueness_threshold, texture_threshold,
              a_valid, b_valid, prior, search_radius)
    av = torch.isfinite(a) & (torch.ones_like(a, dtype=torch.bool) if a_valid is None else a_valid)
    bv = torch.isfinite(b) & (torch.ones_like(b, dtype=torch.bool) if b_valid is None else b_valid)
    # All subsequent arithmetic sees finite placeholders; support excludes them.
    safe_a, safe_b = torch.where(av, a, 0.), torch.where(bv, b, 0.)
    metric = _Metric(safe_a, safe_b, av, bv, method, block_size, sign, texture_threshold)
    stop = min(max_disparity, a.shape[-1] - 1)
    infinity = torch.full_like(a, float('inf'))
    best, winner = infinity.clone(), torch.full_like(a, float('nan'))

    def eligible(d):
        if prior is None:
            return torch.ones_like(av)
        return torch.isfinite(prior) & ((prior - d).abs() <= search_radius)

    for d in range(min_disparity, stop + 1):
        cost = torch.where(eligible(d), metric.integer_cost(d), infinity)
        win = cost < best
        best, winner = torch.where(win, cost, best), torch.where(win, float(d), winner)
    far = infinity.clone()
    neighbors = [infinity.clone() for _ in range(5)]
    for d in range(min_disparity, stop + 1):
        cost = torch.where(eligible(d), metric.integer_cost(d), infinity)
        far = torch.minimum(far, torch.where((winner - d).abs() >= 2, cost, infinity))
        for index, offset in enumerate((-2, -1, 0, 1, 2)):
            neighbors[index] = torch.where(winner + offset == d, cost, neighbors[index])
    confidence = _relative_margin(far, best)
    ambiguous = ~torch.isfinite(far) | (confidence <= max(uniqueness_threshold, 1e-6))
    integer_good = torch.isfinite(best) & ~ambiguous
    periodic = repeated_texture_evidence(safe_a, av, max(1, max_disparity - min_disparity))
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
    search_boundary = search_boundary | ~torch.isfinite(winner)
    integer_confidence = torch.where(integer_good, confidence, 0.)
    refinement_valid = torch.zeros_like(av)
    disparity = winner.clone()
    fractional_best, fractional_minus, fractional_plus, fractional_far = (infinity.clone() for _ in range(4))
    fractional_offset = torch.zeros_like(a)
    fractional_tie = torch.zeros_like(av)
    if subpixel:
        selected = torch.full_like(a, 6, dtype=torch.int64)
        tie_count = torch.zeros_like(selected)
        for index in range(13):
            offset = (index - 6) * .125
            candidate = winner + offset
            allowed = ((candidate >= min_disparity) & (candidate <= stop) & eligible(candidate))
            cost = torch.where(allowed & integer_good, metric.fractional_cost(winner, offset), infinity)
            win = cost < fractional_best
            tie = torch.isfinite(cost) & (cost == fractional_best)
            tie_count = torch.where(win, torch.ones_like(tie_count), tie_count + tie.to(tie_count.dtype))
            closer = abs(index - 6) < (selected - 6).abs()
            replace = win | (tie & closer)
            selected = torch.where(replace, index, selected)
            fractional_best = torch.minimum(fractional_best, cost)
        for index in range(13):
            offset = (index - 6) * .125
            candidate = winner + offset
            allowed = ((candidate >= min_disparity) & (candidate <= stop) & eligible(candidate))
            cost = torch.where(allowed & integer_good, metric.fractional_cost(winner, offset), infinity)
            fractional_minus = torch.where(selected == index + 1, cost, fractional_minus)
            fractional_plus = torch.where(selected == index - 1, cost, fractional_plus)
            fractional_far = torch.minimum(fractional_far, torch.where((selected - index).abs() >= 4, cost, infinity))
        fractional_offset = (selected.to(a.dtype) - 6) * .125
        curvature = fractional_minus + fractional_plus - 2 * fractional_best
        fractional_tie = tie_count > 1
        refinement_valid = (integer_good & torch.isfinite(fractional_minus) & torch.isfinite(fractional_plus)
                            & torch.isfinite(fractional_best) & (fractional_offset.abs() <= .5)
                            & ~fractional_tie & (curvature > 1e-8)
                            & (_relative_margin(fractional_far, fractional_best) > max(uniqueness_threshold, 1e-6)))
        disparity = torch.where(refinement_valid, winner + fractional_offset, winner)
    good = integer_good
    disparity = torch.where(good, disparity, torch.full_like(a, float('nan')))
    trace = dict(
        integer_disparity=winner, integer_confidence=integer_confidence, integer_good=integer_good,
        refinement_valid=refinement_valid, search_boundary=search_boundary,
        local_failed=(~integer_good | search_boundary) if prior is not None else torch.zeros_like(av),
        best_cost=best, raw_cost=best, second_nonlocal_cost=far,
        minus2_cost=neighbors[0], minus_cost=neighbors[1], plus_cost=neighbors[3], plus2_cost=neighbors[4],
        fractional_best_cost=fractional_best, fractional_minus_cost=fractional_minus,
        fractional_plus_cost=fractional_plus, fractional_nonlocal_cost=fractional_far,
        fractional_offset=fractional_offset,
        reasons=dict(low_texture=~metric.a_texture, patch_support=~metric.a_patch,
                     no_candidate=~torch.isfinite(best), separated_mode_ambiguity=ambiguous,
                     repeated_texture=repeated, fractional_tie=fractional_tie,
                     integer_fallback=integer_good & ~refinement_valid if subpixel else torch.zeros_like(av)),
        metric=method,
        subpixel_policy=dict(metric=method, sample_step_px=.125, sample_count=13,
                             direct_metric_validation=True, center_disparity_uniform_within_patch=True,
                             fallback='integer', census_plateau_policy='integer_fallback',
                             confidence_is_probability=False),
        resource_metadata=dict(cost_volume_allocated=False, disparity_planes_in_memory=1,
                               cached_cost_state_planes=None, cost_state_bound_independent_of_disparity=True,
                               descriptor_stencil_px=5 if method == 'census' else None,
                               fractional_evaluations=26 if subpixel else 0,
                               confidence_kind='relative_cost_margin', confidence_is_probability=False,
                               implementation='experimental_pytorch_streaming',
                               gpu_memory_measured=False))
    return disparity, integer_confidence, good, trace
