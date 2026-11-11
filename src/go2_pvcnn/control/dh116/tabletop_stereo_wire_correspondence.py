"""Diagnostic wire matching with full image patches and bounded RGB mask guidance.

The matcher has no validation-reference inputs. Reference visibility and
correspondence scoring are separate post-inference operations.
"""
import math

import torch

from .tabletop_stereo_depth import StereoDepth
from .tabletop_stereo_error_diagnostics import reference_visibility
from .tabletop_stereo_validation import validate_images
from .tabletop_stereo_wire_matching import wire_colour_mask


def segmentation_evidence(output):
    """Keep both RGB views and three logits distinct from physical visibility."""
    from .tabletop_stereo_wire_segmentation import segmentation_masks

    masks, _ = segmentation_masks(output, diagnostic_only=True)
    probabilities = output['logits'].sigmoid()
    result = {}
    for index, side in enumerate(('left', 'right')):
        record = {name + '_probability': probabilities[:, index, head]
                  for head, name in enumerate(('visible', 'occluded', 'unknown'))}
        visible = record['visible_probability'] >= .5
        record.update(visible_mask=visible, runtime_mask=masks[index],
                      unknown_rejection_mask=visible & ~masks[index])
        result[side] = record
    return result


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _curve(values):
    return [_finite(value) for value in values.tolist()]


class SoftWireCorrespondence(StereoDepth):
    """Opt-in sparse full-patch ZNCC; legacy processing identities stay intact."""

    def __init__(self, *, mask_weight=.03, **options):
        super().__init__(**options)
        if not math.isfinite(mask_weight) or not 0 <= mask_weight <= .05:
            raise ValueError('mask weight must be between 0 and .05')
        self.mask_weight = mask_weight

    def _search(self, a, b, pa, pb, centers, sign, point_traces=None):
        _, h, w = a.shape
        radius = self.block_size // 2
        o = torch.arange(-radius, radius + 1, device=a.device)
        dy, dx = (v.flatten() for v in torch.meshgrid(o, o, indexing='ij'))
        hi = min(self.max_disparity, w - 1)
        disparities = torch.arange(self.min_disparity, hi + 1, device=a.device, dtype=a.dtype)
        out = torch.full_like(a, float('nan'))
        quality = torch.zeros_like(a)
        patch = torch.zeros_like(centers)
        texture = patch.clone()
        unique = patch.clone()
        photo = patch.clone()
        edge = patch.clone()
        if disparities.numel() == 0:
            return out, quality, dict(patch=patch, texture=texture, unique=unique, photo=photo, edge=edge)

        for group in centers.nonzero().split(64):
            if not len(group):
                continue
            env, y, x = group.unbind(-1)
            sy = y[:, None] + dy
            sx = x[:, None] + dx
            source_inside = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
            sy, safe_sx = sy.clamp(0, h - 1), sx.clamp(0, w - 1)
            source = a[env[:, None], sy, safe_sx][:, None]
            centered = source - source.mean(-1, keepdim=True)
            va = centered.square().mean(-1)
            source_p = pa[env, y, x][:, None]

            def cost(candidate):
                xx = sx[:, None, :] + sign * candidate[..., None]
                lo, up = xx.floor(), xx.ceil()
                inside = source_inside.all(-1)[:, None] & (lo >= 0).all(-1) & (up < w).all(-1)
                lo_index = lo.long().clamp(0, w - 1)
                up_index = up.long().clamp(0, w - 1)
                be, by = env[:, None, None], sy[:, None, :]
                low = b[be, by, lo_index]
                target = low + (b[be, by, up_index] - low) * (xx - lo)
                target -= target.mean(-1, keepdim=True)
                vb = target.square().mean(-1)
                evidence = inside & (va > max(self.texture, 1e-4) ** 2) & (vb > max(self.texture, 1e-4) ** 2)
                raw = 1 - ((centered * target).mean(-1) / (va * vb).sqrt().clamp_min(1e-12)).clamp(-1, 1)
                # Target probabilities are a bounded additive cost, never a
                # stencil mask or permission to accept unsupported RGB.
                c = len(dx) // 2
                plo, pup = pb[be, by, lo_index][..., c], pb[be, by, up_index][..., c]
                target_p = plo + (pup - plo) * (xx - lo)[..., c]
                guided = raw + self.mask_weight * (source_p - target_p).abs()
                return guided.masked_fill(~evidence, float('inf')), raw, inside

            guided, raw, inside = cost(disparities[None])
            best, winner = guided.min(-1)
            integer = disparities[winner]
            d = integer
            rgb_cost = raw.gather(1, winner[:, None])[:, 0]
            final_rgb = rgb_cost
            alternative = guided.masked_fill((disparities - d[:, None]).abs() <= 1, float('inf')).amin(-1)
            finite = torch.isfinite(best) & torch.isfinite(alternative)
            margin = torch.where(finite, (alternative - best) / alternative.clamp_min(1e-8), 0.)
            tied = ((guided - best[:, None]).abs() <= 1e-6).sum(-1) > 1
            clear = finite & (margin > self.uniqueness) & ~tied
            rgb_ok = rgb_cost <= .20
            boundary = (d == self.min_disparity) | (d == hi)
            accepted = clear & rgb_ok & ~boundary
            fractional = None
            if self.subpixel and accepted.any():
                phase = torch.arange(-6, 7, device=a.device, dtype=a.dtype) / 8
                offsets = d[:, None] + phase
                scores, raw_fractional, _ = cost(offsets)
                improved, winner_fractional = scores.min(-1)
                selected = winner_fractional[:, None]
                improved_rgb = raw_fractional.gather(1, selected)[:, 0]
                choose = accepted & (improved < best - 1e-5) & (improved_rgb < rgb_cost)
                d = torch.where(choose, offsets.gather(1, selected)[:, 0], d)
                final_rgb = torch.where(choose, improved_rgb, rgb_cost)
                fractional = (offsets, scores, raw_fractional)
            idx = env, y, x
            patch[idx] = inside.any(-1)
            texture[idx] = torch.isfinite(best)
            unique[idx] = clear
            photo[idx] = rgb_ok
            edge[idx] = boundary & torch.isfinite(best)
            out[idx] = torch.where(accepted, d, float('nan'))
            quality[idx] = torch.where(accepted, margin, 0.)
            if point_traces:
                for i, key in enumerate(map(tuple, group.tolist())):
                    if key not in point_traces:
                        continue
                    raw_supported = raw[i].masked_fill(~torch.isfinite(guided[i]), float('nan'))
                    attempted = bool(accepted[i])
                    record = dict(pixel_eyx=key, integer_disparities_px=disparities.tolist(),
                        integer_total_cost=_curve(guided[i]), integer_rgb_cost=_curve(raw_supported),
                        integer_winner_px=float(integer[i]) if torch.isfinite(best[i]) else None,
                        integer_winner_rgb_cost=_finite(raw_supported[winner[i]]),
                        winner_rgb_cost=float(final_rgb[i]) if torch.isfinite(best[i]) else None,
                        subpixel_offset_px=float(d[i] - integer[i]) if attempted else None,
                        local_match_accepted=attempted, accepted_disparity_px=float(d[i]) if attempted else None,
                        margin=float(margin[i]), best_cost=_finite(best[i]), separated_cost=_finite(alternative[i]))
                    if fractional is not None and attempted:
                        phase, phase_cost, phase_rgb = (value[i] for value in fractional)
                        record['fractional_curve'] = dict(disparities_px=phase.tolist(),
                            total_cost=_curve(phase_cost),
                            rgb_cost=_curve(phase_rgb.masked_fill(~torch.isfinite(phase_cost), float('nan'))))
                    point_traces[key] = record
        return out, quality, dict(patch=patch, texture=texture, unique=unique, photo=photo, edge=edge)

    @torch.no_grad()
    def compute(self, left, right, calibration, *, probabilities, left_proposal=None, trace_pixels=()):
        validate_images(left, right, calibration)
        if not calibration.rectified:
            raise ValueError('rectified stereo required')
        if len(probabilities) != 2 or any(
                p.shape != left[:, 0].shape or p.device != left.device or not p.is_floating_point()
                or not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                for p in probabilities):
            raise ValueError('aligned finite RGB-derived probabilities required')
        lp, rp = probabilities
        if left_proposal is None:
            left_proposal = lp >= .5
        if left_proposal.shape != lp.shape or left_proposal.dtype != torch.bool or left_proposal.device != lp.device:
            raise ValueError('aligned boolean left wire proposal required')
        # Existing colour evidence remains eligible even when the learned
        # mask has low recall. The raw image patch is always available.
        candidates = wire_colour_mask(left) | left_proposal
        lg, rg = left.mean(1), right.mean(1)
        b, h, w = lg.shape
        left_points = {}
        for pixel in trace_pixels:
            if (len(pixel) != 3 or any(type(v) is not int for v in pixel)
                    or not all(0 <= v < bound for v, bound in zip(pixel, lg.shape))):
                raise ValueError('trace pixels must be integer env/row/column indices inside the image')
            left_points[tuple(pixel)] = None
        dl, lc, left_trace = self._search(lg, rg, lp, rp, candidates, -1, left_points)
        x = torch.arange(w, device=left.device, dtype=dl.dtype)[None, None].expand(b, h, w)
        xr = x - dl
        # Reverse consistency runs an independent full-range search at both
        # samples needed to interpolate the queried right position.
        reverse = torch.zeros_like(candidates)
        env, row, col = torch.isfinite(dl).nonzero(as_tuple=True)
        projected = xr[env, row, col]
        for position in (projected.floor(), projected.ceil()):
            inside = (position >= 0) & (position < w)
            reverse[env[inside], row[inside], position[inside].long()] = True
        right_points = {}
        right_queries = {}
        for key in left_points:
            target = _finite(xr[key])
            e, y, _ = key
            positions = () if target is None else sorted({math.floor(target), math.ceil(target)})
            right_queries[key] = [(e, y, x) for x in positions if 0 <= x < w]
            right_points.update(dict.fromkeys(right_queries[key]))
        dr, rc, _ = self._search(rg, lg, rp, lp, reverse, 1, right_points)
        reverse_valid = torch.isfinite(dr)
        back, available = self.sample_horizontal(dr, xr, reverse_valid)
        back_conf, conf_valid = self.sample_horizontal(rc, xr, reverse_valid)
        lr_ok = available & conf_valid & torch.isfinite(dl) & ((dl - back).abs() <= self.lr)
        z = float(calibration.K_left[0, 0]) * calibration.baseline_m / dl
        depth_ok = torch.isfinite(z) & (z >= self.near) & (z <= self.far)
        accepted = lr_ok & depth_ok
        reasons = dict(
            patch_border=~left_trace['patch'],
            low_texture_left=left_trace['patch'] & ~left_trace['texture'],
            candidate_wire_ambiguous=left_trace['texture'] & ~left_trace['unique'],
            candidate_wire_cost_rejected=left_trace['texture'] & ~left_trace['photo'],
            search_boundary=left_trace['edge'],
            left_matcher_rejected=~torch.isfinite(dl),
            right_sample_unavailable=torch.isfinite(dl) & ~available,
            left_right_inconsistent=available & ~lr_ok,
            depth_range=torch.isfinite(dl) & ~depth_ok,
        )
        diagnostics = dict(reasons=reasons, raw_disparity_px=dl,
                           right_search_center_count=int(reverse.sum()), physical_occlusion_inferred=False)
        if left_points:
            diagnostics['point_traces'] = {
                key: dict(left=record, right=[right_points[k] for k in right_queries[key]],
                          right_projected_x_px=_finite(xr[key]),
                          lr_error_px=_finite((dl[key]-back[key]).abs()) if available[key] else None,
                          reverse_sample_available=bool(available[key]), accepted=bool(accepted[key]))
                for key, record in left_points.items()}
        return dict(depth_m=z.masked_fill(~accepted, float('nan')),
                    disparity_px=dl.masked_fill(~accepted, float('nan')),
                    valid=accepted, confidence=torch.where(accepted, torch.minimum(lc, back_conf), 0.),
                    candidate_mask=candidates, lr_consistency=lr_ok,
                    diagnostics=diagnostics)


def silhouette_point_trace(left_rgb, right_rgb, masks, pixel, matcher, prediction):
    """Re-evaluate the integer silhouette costs at one actual audit position."""
    if matcher.method != 'wire_silhouette' or matcher.subpixel:
        raise ValueError('integer silhouette matcher required for this audit trace')
    left, right = left_rgb.mean(1), right_rgb.mean(1)
    height, width = left.shape[-2:]
    radius = matcher.block_size // 2
    offsets = torch.arange(-radius, radius+1, device=left.device)
    oy, ox = (values.flatten() for values in torch.meshgrid(offsets, offsets, indexing='ij'))
    high = min(matcher.max_disparity, width-1)
    disparities = torch.arange(matcher.min_disparity, high+1, device=left.device)

    def curve(source, target, source_mask, target_mask, center, sign):
        e, y, x = center
        if not source_mask[center] or not disparities.numel():
            return None
        yy, xx = y + oy, x + ox
        if not ((yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)).all():
            return dict(pixel_eyx=center, integer_disparities_px=disparities.tolist(),
                        integer_total_cost=[None]*len(disparities), integer_winner_px=None,
                        subpixel_offset_px=None, winner_rgb_cost=None, local_match_accepted=False,
                        accepted_disparity_px=None, metric='wire_silhouette')
        u = xx[None] + sign*disparities[:, None]
        in_bounds = (u >= 0) & (u < width)
        safe_u = u.clamp(0, width-1)
        a_mask = source_mask[e, yy, xx]
        b_mask = target_mask[e, yy, safe_u] & in_bounds
        joint = (a_mask & b_mask).sum(-1)
        count = int(a_mask.sum())
        support = (b_mask[:, len(ox)//2] & (joint >= 3)
                   & (joint >= .6*torch.maximum(a_mask.sum(), b_mask.sum(-1))))
        feasible = support & in_bounds.all(-1) & (min(count, len(ox)-count) >= 2)
        values = (a_mask ^ b_mask).float().mean(-1).masked_fill(~feasible, float('inf'))
        best, choice = values.min(0)
        integer = disparities[choice]
        alternative = values.masked_fill((disparities-integer).abs() <= 1, float('inf')).min()
        margin = ((alternative-best)/alternative.abs().clamp_min(1e-8)
                  if torch.isfinite(best) and torch.isfinite(alternative) else 0.)
        tied = (torch.isfinite(values) & ((values-best).abs() <= 1e-6)).sum() != 1
        accepted = bool(margin > matcher.uniqueness and not tied and best <= 0
                        and matcher.min_disparity < integer < high)
        a = source[e, yy, xx]
        b = target[e, yy, safe_u]
        a, b = a-a.mean(), b-b.mean(-1, keepdim=True)
        covariance_scale = (a.square().mean()*b.square().mean(-1)).sqrt()
        rgb = 1-(a*b).mean(-1).div(covariance_scale.clamp_min(1e-12)).clamp(-1, 1)
        rgb = rgb.masked_fill(~in_bounds.all(-1) | (covariance_scale <= 1e-12), float('nan'))
        return dict(pixel_eyx=center, integer_disparities_px=disparities.tolist(),
                    integer_total_cost=_curve(values), integer_rgb_cost=_curve(rgb),
                    integer_winner_px=float(integer) if torch.isfinite(best) else None,
                    subpixel_offset_px=0. if accepted else None,
                    winner_rgb_cost=_finite(rgb[choice]) if torch.isfinite(best) else None,
                    local_match_accepted=accepted, accepted_disparity_px=float(integer) if accepted else None,
                    metric='wire_silhouette', rgb_cost_used_for_ranking=False)

    forward = curve(left, right, masks[0], masks[1], pixel, -1)
    raw = _finite(prediction['diagnostics']['raw_disparity_px'][pixel])
    reproduced = None if forward is None else forward['accepted_disparity_px']
    if raw != reproduced:
        raise RuntimeError('integer silhouette trace disagrees with the actual forward match')
    e, y, x = pixel
    target_x = None if raw is None else x-int(raw)
    reverse = [] if target_x is None else [curve(right, left, masks[1], masks[0], (e, y, target_x), 1)]
    reverse_d = None if not reverse or reverse[0] is None else reverse[0]['accepted_disparity_px']
    # The enclosing StereoDepth also checks right texture and patch validity.
    available = not bool(prediction['diagnostics']['reasons']['right_sample_unavailable'][pixel])
    return dict(left=forward, right=reverse, right_projected_x_px=target_x,
                lr_error_px=abs(raw-reverse_d) if available and raw is not None and reverse_d is not None else None,
                reverse_sample_available=available, accepted=bool(prediction['valid'][pixel]))


def colour_anchor_fallback(baseline, soft, colour_masks):
    """Opt-in fallback on RGB silhouette support without re-ranking baseline depths."""
    left_colour, right_colour = colour_masks
    shape = baseline['valid'].shape
    if any(mask.shape != shape or mask.dtype != torch.bool
           for mask in (left_colour, right_colour, baseline['valid'], soft['valid'], soft['lr_consistency'])):
        raise ValueError('aligned RGB colour and stereo validity masks required')
    disparity = soft['disparity_px']
    columns = torch.arange(shape[-1], dtype=disparity.dtype, device=disparity.device)
    target_x = columns[None, None] - disparity
    _, right_support = StereoDepth.sample_horizontal(right_colour.float(), target_x, right_colour)
    baseline_valid = baseline['valid'] & left_colour
    fallback = ~baseline_valid & left_colour & soft['valid'] & soft['lr_consistency'] & right_support
    valid = baseline_valid | fallback
    result = {key: torch.where(baseline_valid, baseline[key], soft[key]).masked_fill(
        ~valid, 0. if key == 'confidence' else float('nan'))
        for key in ('disparity_px', 'depth_m', 'confidence')}
    reasons = dict(fallback_right_colour_unavailable=left_colour & soft['valid'] & ~right_support,
                   fallback_soft_stereo_rejected=left_colour & ~soft['valid'])
    result.update(valid=valid, lr_consistency=valid, candidate_mask=left_colour,
        diagnostics=dict(reasons=reasons, raw_disparity_px=result['disparity_px'],
                         baseline_depth_preserved=baseline_valid, soft_fallback_depth=fallback,
                         physical_occlusion_inferred=False))
    return result


def native_visibility(left_z, right_z, left_wire_valid, calibration, *,
                      right_wire=None, right_known=None):
    """Classify only verified right surfaces; missing labels remain unknown."""
    if (right_wire is None) != (right_known is None):
        raise ValueError('right native identity and known-pixel masks must be paired')
    empty = torch.zeros_like(left_wire_valid)
    if right_z is None or right_wire is None:
        return dict(visible=empty, occluded=empty.clone(), unknown=left_wire_valid.clone())
    visibility = reference_visibility(left_z, right_z, left_wire_valid,
                                      calibration.K_left, calibration.baseline_m)
    x = torch.arange(left_z.shape[-1], device=left_z.device, dtype=left_z.dtype)
    x = x[None, None].expand_as(left_z)
    xr = x - float(calibration.K_left[0, 0]) * calibration.baseline_m / left_z
    wire_fraction, known = StereoDepth.sample_horizontal(right_wire.float(), xr, right_known)
    visible = visibility['common_visible'] & known & (wire_fraction >= 1 - 1e-6)
    occluded = visibility['left_only_occluded'] & known
    return dict(visible=visible, occluded=occluded, unknown=left_wire_valid & ~(visible | occluded))


@torch.no_grad()
def score_native_correspondence(frame, reference, calibration, prediction, masks, *, raw_visible_masks=None):
    """Score same-frame native identity after matching; fixed reference support."""
    from .tabletop_stereo_wire_segmentation import native_instance_targets

    target, valid, admission = native_instance_targets(frame, reference)
    wire = target[:, 0, 0].bool()
    right_wire = target[:, 1, 0].bool() if admission['right_instance_source'] != 'missing' else None
    right_known = valid[:, 1, 0] if right_wire is not None else None
    left_z = reference['depth_m']
    right_z = reference.get('right_depth_m')
    right_valid = reference.get('right_valid')
    if right_z is not None and right_valid is not None:
        if right_valid.shape != right_z.shape or right_valid.dtype != torch.bool:
            raise ValueError('aligned right optical-Z validity required')
        right_z = right_z.masked_fill(~right_valid, float('nan'))
    reference_support = wire & reference['valid'] & torch.isfinite(left_z) & (left_z > 0)
    visibility = native_visibility(left_z, right_z, reference_support, calibration,
                                  right_wire=right_wire, right_known=right_known)
    baseline_px_m = float(calibration.K_left[0, 0]) * calibration.baseline_m
    disparity = prediction['disparity_px']
    disparity_error = disparity - baseline_px_m / left_z
    accepted = prediction['valid'] & torch.isfinite(disparity)
    predicted_right_wire = torch.zeros_like(wire)
    predicted_right_evidence = torch.zeros_like(wire)
    predicted_right_z_agrees = torch.zeros_like(wire)
    if right_wire is not None and right_z is not None:
        columns = torch.arange(left_z.shape[-1], dtype=left_z.dtype, device=left_z.device)
        x = columns[None, None].expand_as(left_z) - disparity
        fraction, available = StereoDepth.sample_horizontal(right_wire.float(), x, right_known)
        depth_known = torch.isfinite(right_z) & (right_z > 0)
        target_z, depth_available = StereoDepth.sample_horizontal(right_z, x, depth_known)
        predicted_right_wire = available & (fraction >= 1 - 1e-6)
        predicted_right_evidence = available & depth_available
        predicted_right_z_agrees = depth_available & ((target_z - left_z).abs() <= .001)
    paired = visibility['visible'] & accepted
    resolved = paired & predicted_right_evidence
    correct = resolved & predicted_right_wire & predicted_right_z_agrees & (disparity_error.abs() <= .5)
    def view_iou(env, view_masks):
        iou = []
        for view, mask in enumerate(view_masks):
            region = valid[env, view, 0]
            truth = target[env, view, 0].bool()
            union = int((region & (truth | mask[env])).sum())
            iou.append(int((region & truth & mask[env]).sum()) / union if union else None)
        return iou

    rows = []
    for env in range(len(wire)):
        iou = view_iou(env, masks)
        raw_iou = view_iou(env, raw_visible_masks) if raw_visible_masks is not None else (None, None)
        diff = disparity_error[env][reference_support[env] & accepted[env]].double()
        eligible = int(visibility['visible'][env].sum())
        predictions = int(paired[env].sum())
        known_predictions = int(resolved[env].sum())
        successes = int(correct[env].sum())
        view_counts = {}
        for view, side in enumerate(('left', 'right')):
            native = valid[env, view, 0] & target[env, view, 0].bool()
            count = int(native.sum())
            raw = None if raw_visible_masks is None else native & raw_visible_masks[view][env]
            visible_count = None if raw is None else int(raw.sum())
            rejected = None if raw is None else int((raw & ~masks[view][env]).sum())
            view_counts[side] = dict(native_wire_count=count,
                visible_iou=raw_iou[view], runtime_iou=iou[view],
                visible_coverage=visible_count/count if count and visible_count is not None else None,
                runtime_coverage=int((native & masks[view][env]).sum())/count if count else None,
                unknown_rejected_native_count=rejected if count else None,
                unknown_rejection_fraction=rejected/visible_count if visible_count else None)
        denominator = int(reference_support[env].sum())
        rows.append(dict(segmentation_iou_left=iou[0], segmentation_iou_right=iou[1],
                         visible_head_iou_left=raw_iou[0], visible_head_iou_right=raw_iou[1],
                         segmentation_view_metrics=view_counts,
                         stereo_coverage_denominator=denominator,
                         stereo_coverage= len(diff)/denominator if denominator else None,
                         visibility_counts={name: int(value[env].sum()) for name, value in visibility.items()},
                         disparity_reference_scope='left_optical_Z_at_fixed_native_wire_pixels',
                         disparity_comparison_count=len(diff),
                         disparity_mae_px=float(diff.abs().mean()) if len(diff) else None,
                         disparity_rmse_px=float(diff.square().mean().sqrt()) if len(diff) else None,
                         true_correspondence_threshold_px=.5,
                         true_correspondence_predictions=predictions,
                         true_correspondence_unresolved_predictions=predictions-known_predictions,
                         true_correspondence_correct=successes,
                         true_correspondence_accuracy=successes / known_predictions if known_predictions else None,
                         true_correspondence_coverage=predictions / eligible if eligible else None,
                         true_correspondence_completeness=successes / eligible if eligible else None,
                         right_instance_source=admission['right_instance_source']))
    return rows, visibility
