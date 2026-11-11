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


class SoftWireCorrespondence(StereoDepth):
    """Opt-in sparse full-patch ZNCC; legacy processing identities stay intact."""

    def __init__(self, *, mask_weight=.03, **options):
        super().__init__(**options)
        if not math.isfinite(mask_weight) or not 0 <= mask_weight <= .05:
            raise ValueError('mask weight must be between 0 and .05')
        self.mask_weight = mask_weight

    def _search(self, a, b, pa, pb, centers, sign):
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
            d = disparities[winner]
            rgb_cost = raw.gather(1, winner[:, None])[:, 0]
            alternative = guided.masked_fill((disparities - d[:, None]).abs() <= 1, float('inf')).amin(-1)
            finite = torch.isfinite(best) & torch.isfinite(alternative)
            margin = torch.where(finite, (alternative - best) / alternative.clamp_min(1e-8), 0.)
            tied = ((guided - best[:, None]).abs() <= 1e-6).sum(-1) > 1
            clear = finite & (margin > self.uniqueness) & ~tied
            rgb_ok = rgb_cost <= .20
            boundary = (d == self.min_disparity) | (d == hi)
            accepted = clear & rgb_ok & ~boundary
            if self.subpixel and accepted.any():
                phase = torch.arange(-6, 7, device=a.device, dtype=a.dtype) / 8
                offsets = d[:, None] + phase
                scores, raw_fractional, _ = cost(offsets)
                improved, winner_fractional = scores.min(-1)
                selected = winner_fractional[:, None]
                improved_rgb = raw_fractional.gather(1, selected)[:, 0]
                choose = accepted & (improved < best - 1e-5) & (improved_rgb < rgb_cost)
                d = torch.where(choose, offsets.gather(1, selected)[:, 0], d)
            idx = env, y, x
            patch[idx] = inside.any(-1)
            texture[idx] = torch.isfinite(best)
            unique[idx] = clear
            photo[idx] = rgb_ok
            edge[idx] = boundary & torch.isfinite(best)
            out[idx] = torch.where(accepted, d, float('nan'))
            quality[idx] = torch.where(accepted, margin, 0.)
        return out, quality, dict(patch=patch, texture=texture, unique=unique, photo=photo, edge=edge)

    @torch.no_grad()
    def compute(self, left, right, calibration, *, probabilities, left_proposal=None):
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
        dl, lc, left_trace = self._search(lg, rg, lp, rp, candidates, -1)
        b, h, w = lg.shape
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
        dr, rc, _ = self._search(rg, lg, rp, lp, reverse, 1)
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
        return dict(depth_m=z.masked_fill(~accepted, float('nan')),
                    disparity_px=dl.masked_fill(~accepted, float('nan')),
                    valid=accepted, confidence=torch.where(accepted, torch.minimum(lc, back_conf), 0.),
                    candidate_mask=candidates, lr_consistency=lr_ok,
                    diagnostics=dict(reasons=reasons, raw_disparity_px=dl,
                                     right_search_center_count=int(reverse.sum()),
                                     physical_occlusion_inferred=False))


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
    reference_support = wire & reference['valid'] & torch.isfinite(left_z) & (left_z > 0)
    visibility = native_visibility(left_z, reference.get('right_depth_m'), reference_support, calibration,
                                  right_wire=right_wire, right_known=right_known)
    baseline_px_m = float(calibration.K_left[0, 0]) * calibration.baseline_m
    disparity = prediction['disparity_px']
    disparity_error = disparity - baseline_px_m / left_z
    accepted = prediction['valid'] & torch.isfinite(disparity)
    predicted_right_wire = torch.zeros_like(wire)
    if right_wire is not None:
        columns = torch.arange(left_z.shape[-1], dtype=left_z.dtype, device=left_z.device)
        x = columns[None, None].expand_as(left_z) - disparity
        fraction, available = StereoDepth.sample_horizontal(right_wire.float(), x, right_known)
        predicted_right_wire = available & (fraction >= 1 - 1e-6)
    paired = visibility['visible'] & accepted
    correct = paired & predicted_right_wire & (disparity_error.abs() <= .5)
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
        successes = int(correct[env].sum())
        rows.append(dict(segmentation_iou_left=iou[0], segmentation_iou_right=iou[1],
                         visible_head_iou_left=raw_iou[0], visible_head_iou_right=raw_iou[1],
                         visibility_counts={name: int(value[env].sum()) for name, value in visibility.items()},
                         disparity_reference_scope='left_optical_Z_at_fixed_native_wire_pixels',
                         disparity_comparison_count=len(diff),
                         disparity_mae_px=float(diff.abs().mean()) if len(diff) else None,
                         disparity_rmse_px=float(diff.square().mean().sqrt()) if len(diff) else None,
                         true_correspondence_threshold_px=.5,
                         true_correspondence_predictions=predictions,
                         true_correspondence_correct=successes,
                         true_correspondence_accuracy=successes / predictions if predictions else None,
                         true_correspondence_coverage=predictions / eligible if eligible else None,
                         true_correspondence_completeness=successes / eligible if eligible else None,
                         right_instance_source=admission['right_instance_source']))
    return rows, visibility
