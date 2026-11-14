"""Sparse rectified matching on mutually observed warm-colour pixels."""
import torch


METHODS = ('wire_census', 'wire_zncc', 'wire_silhouette')
_CENTERS_PER_TILE = 128


def wire_colour_mask(rgb):
    """Use the same RGB predicate as the tracked wire extractor."""
    red, green, blue = rgb.unbind(1)
    return (red > 1.07 * green) & (green > 1.1 * blue) & (red > .12)


@torch.no_grad()
def match_wire(a, b, a_wire, b_wire, sign, *, method, low, high, block,
               texture, uniqueness, subpixel):
    """Match actual foreground centers, with a bounded [centers, disparities, patch] tile.

    Foreground membership comes only from the two RGB images and rectification
    masks. The downstream bidirectional and physical depth checks still apply.
    """
    _, height, width = a.shape
    high = min(high, width - 1)
    radius = block // 2
    offsets = torch.arange(-radius, radius + 1, device=a.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
    oy, ox = oy.flatten(), ox.flatten()
    center = len(ox) // 2
    disparities = torch.arange(low, high + 1, device=a.device, dtype=a.dtype)
    output = torch.full_like(a, float('nan'))
    confidence = torch.zeros_like(a)
    good = torch.zeros_like(a_wire)
    supported = torch.zeros_like(good)
    texture_ok = torch.zeros_like(good)
    unique_ok = torch.zeros_like(good)
    boundary = torch.zeros_like(good)
    quality_ok = torch.zeros_like(good)
    refined = torch.zeros_like(good)
    if not len(disparities):
        return output, confidence, good, dict(search_boundary=boundary, optional_refinement_used=refined,
            reasons=dict(wire_no_joint_patch=~supported, wire_low_texture=~texture_ok,
                         wire_ambiguous=~unique_ok, wire_cost_rejected=~quality_ok))

    for pixels in a_wire.nonzero().split(_CENTERS_PER_TILE):
        if not len(pixels):
            continue
        env, yy, xx = pixels.unbind(1)
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
            be, by = env[:, None, None], sy[:, None, :]
            target_mask = b_wire[be, by, x0] & b_wire[be, by, x1] & inside
            first = b[be, by, x0]
            target = first + (b[be, by, x1] - first) * (target_x - target_x.floor())
            joint = source_mask & target_mask
            count = joint.sum(-1)
            minimum = .6 * torch.maximum(source_mask.sum(-1), target_mask.sum(-1))
            support = target_mask[..., center] & (count >= 3) & (count >= minimum)
            if method == 'wire_silhouette':
                # Keep only measured complete windows with both sides of a
                # visible RGB transition. This tests shape, not a depth prior.
                observed = inside.all(-1) & columns_inside.all(-1)[:, None]
                minority = torch.minimum(source_mask.sum(-1), (~source_mask).sum(-1))
                available = support & observed & (minority >= 2)
                mismatch = (source_mask ^ target_mask).float().mean(-1)
                return mismatch.masked_fill(~available, float('inf')), support
            denom = count.clamp_min(1)
            a_centered = source - (source * joint).sum(-1)[..., None] / denom[..., None]
            b_centered = target - (target * joint).sum(-1)[..., None] / denom[..., None]
            va = (a_centered.square() * joint).sum(-1) / denom
            vb = (b_centered.square() * joint).sum(-1) / denom
            has_texture = (va > max(texture, 1e-4) ** 2) & (vb > max(texture, 1e-4) ** 2)
            if method == 'wire_census':
                ra = source < source[..., center:center+1]
                rb = target < target[..., center:center+1]
                cost = ((ra ^ rb) & joint).sum(-1) / (count - 1).clamp_min(1)
            else:
                covariance = (a_centered * b_centered * joint).sum(-1) / denom
                cost = 1. - (covariance / (va * vb).sqrt().clamp_min(1e-12)).clamp(-1., 1.)
            return cost.masked_fill(~(support & has_texture), float('inf')), support

        planes, support = costs(disparities[None, :])
        best, winner = planes.min(-1)
        winner_d = disparities[winner]
        alternative = planes.masked_fill((disparities - winner_d[:, None]).abs() <= 1, float('inf')).amin(-1)
        # An isolated supported disparity has no measurable uniqueness. Keep
        # it invalid unless another separated hypothesis can be compared.
        comparable = torch.isfinite(best) & torch.isfinite(alternative)
        margin = torch.where(comparable, (alternative - best) / alternative.abs().clamp_min(1e-8), 0.)
        # Flat adjacent minima are ambiguous even when a distant mode is worse.
        tied = (torch.isfinite(planes) & ((planes - best[:, None]).abs() <= 1e-6)).sum(-1) != 1
        clear = comparable & (margin > uniqueness) & ~tied
        max_cost = {'wire_silhouette': 0., 'wire_census': .25, 'wire_zncc': .20}[method]
        usable_cost = best <= max_cost
        edge = (winner_d == low) | (winner_d == high)
        accepted = clear & usable_cost & ~edge
        used = torch.zeros_like(accepted)
        if subpixel and method != 'wire_silhouette':
            phases = torch.tensor([-.5, -.25, 0., .25, .5], dtype=a.dtype, device=a.device)
            local = winner_d[:, None] + phases
            fractional, _ = costs(local)
            best_fractional, choice = fractional.min(-1)
            used = accepted & (best_fractional < best - 1e-5)
            winner_d = torch.where(used, local.gather(1, choice[:, None])[:, 0], winner_d)

        idx = env, yy, xx
        supported[idx] = support.any(-1)
        texture_ok[idx] = torch.isfinite(best)
        unique_ok[idx] = clear
        quality_ok[idx] = usable_cost
        boundary[idx] = edge
        refined[idx] = used
        good[idx] = accepted
        output[idx] = torch.where(accepted, winner_d, float('nan'))
        confidence[idx] = torch.where(accepted, margin, 0.)

    evidence_key = 'wire_no_shape_evidence' if method == 'wire_silhouette' else 'wire_low_texture'
    return output, confidence, good, dict(search_boundary=boundary, optional_refinement_used=refined,
        metric=method, reasons={'wire_no_joint_patch': ~supported, evidence_key: ~texture_ok,
                                'wire_ambiguous': ~unique_ok, 'wire_cost_rejected': ~quality_ok})
