"""Per-pixel evidence for RGB wire candidates through the actual depth/tracker path."""
import torch
import torch.nn.functional as F

from .tabletop_stereo_error_diagnostics import distribution, region_metrics, reference_visibility
from .tabletop_stereo_wire_matching import wire_colour_mask
from .tabletop_stereo_wire_tracking import _paths_from_mask


_REJECTION_ORDER = (
    'patch_border', 'low_texture_left', 'candidate_wire_no_joint_patch',
    'candidate_wire_low_texture', 'candidate_wire_no_shape_evidence', 'candidate_wire_cost_rejected', 'candidate_wire_ambiguous',
    'repeated_texture', 'subpixel_rejected', 'left_matcher_rejected',
    'out_of_right_view', 'right_sample_unavailable', 'low_texture_right',
    'left_right_inconsistent', 'occlusion_composite', 'uniqueness', 'depth_range',
)


def wire_intersections(wire, stereo, sampled, reasons):
    """Return additive first rejects and independent overlapping conditions.

    First-reject priority is an attribution convention over observed flags. It
    is not a causal ordering inside every matching backend.
    """
    if any(mask.shape != wire.shape or mask.dtype != torch.bool for mask in (wire, stereo, sampled)):
        raise ValueError('aligned boolean wire/depth/point masks required')
    if bool((sampled & ~(wire & stereo)).any()):
        raise ValueError('a point must use a measured valid candidate pixel')
    keys = [name for name in _REJECTION_ORDER if reasons.get(name) is not None]
    keys += [name for name, mask in reasons.items() if name not in keys and mask is not None]
    first = torch.zeros_like(wire, dtype=torch.int16)
    first[~wire] = -1
    rejected_depth = wire & ~stereo
    for i, name in enumerate(keys, 1):
        mask = reasons[name]
        if mask.shape != wire.shape or mask.dtype != torch.bool:
            raise ValueError('unaligned rejection condition ' + name)
        first[rejected_depth & mask & (first == 0)] = i
    omitted = wire & stereo & ~sampled
    first[omitted] = len(keys) + 1
    if bool((rejected_depth & (first == 0)).any()):
        first[rejected_depth & (first == 0)] = len(keys) + 2
    labels = [*keys, 'not_sampled_as_3d_point', 'depth_rejection_reason_unavailable']
    lr_failed = torch.zeros_like(wire)
    for name in ('occlusion_composite', 'left_right_inconsistent', 'right_sample_unavailable'):
        if reasons.get(name) is not None:
            lr_failed |= reasons[name]
    range_failed = reasons.get('depth_range')
    maps = dict(wire_mask=wire, wire_left_depth_valid=wire & stereo,
                wire_left_right_consistent=wire & ~lr_failed if any(reasons.get(name) is not None for name in
                    ('occlusion_composite', 'left_right_inconsistent', 'right_sample_unavailable')) else None,
                wire_depth_range_valid=wire & ~range_failed if range_failed is not None else None,
                wire_3d_point_valid=sampled, first_reject_code=first)
    rows = []
    for index, mask in enumerate(wire):
        count = int(mask.sum())
        accepted = int(sampled[index].sum())
        codes = first[index]
        first_counts = {name: int((codes == i).sum()) for i, name in enumerate(labels, 1)}
        assert sum(first_counts.values()) + accepted == count
        overlaps = {name: int((mask & value[index]).sum()) if value is not None else None
                    for name, value in reasons.items()}
        overlaps['not_sampled_as_3d_point'] = int(omitted[index].sum())
        rows.append(dict(wire_candidate_count=count, valid_point_count=accepted,
                         valid_point_coverage=accepted / count if count else None,
                         intersection_counts={name: None if value is None else int(value[index].sum())
                                              for name, value in maps.items() if name != 'first_reject_code'},
                         first_reject_legend={i: name for i, name in enumerate(labels, 1)},
                         first_rejections=first_counts, overlapping_rejections=overlaps))
    return maps, rows


@torch.no_grad()
def audit_wire_frame(frame, prediction, extracted, calibration, reference, *, block_size, near, far, finger=None, wire_mask=None):
    """Use same-frame validation references only for scoring and occlusion audit."""
    wire = wire_colour_mask(frame['left_rgb']) if wire_mask is None else wire_mask
    depth = prediction['depth_m']
    accepted = prediction['valid'] & torch.isfinite(depth) & (depth > 0)
    sampled = torch.zeros_like(wire)
    for i, (pixel, valid) in enumerate(zip(extracted['point_pixels'], extracted['point_valid'])):
        yx = pixel[valid]
        sampled[i, yx[:, 0], yx[:, 1]] = True
    reasons = prediction['diagnostics']['reasons']
    maps, rows = wire_intersections(wire, accepted, sampled, reasons)
    skeleton = torch.zeros_like(wire)
    for i, mask in enumerate(wire.cpu().numpy()):
        for path in _paths_from_mask(mask)[0]:
            index = torch.as_tensor(path, device=wire.device)
            skeleton[i, index[:, 0], index[:, 1]] = True
    radius = block_size // 2
    band = F.max_pool2d(skeleton[:, None].float(), 3, 1, 1)[:, 0].bool() & wire
    maps.update(wire_candidate_centerline=skeleton, wire_candidate_centerline_band=band)
    left_gray = frame['left_rgb'].mean(1)
    def fraction(mask):
        return F.avg_pool2d(mask[:, None].float(), block_size, 1, radius)[:, 0]
    gray_std = (fraction(left_gray.square()) - fraction(left_gray).square()).clamp_min(0).sqrt()
    rgb_fraction = fraction(wire)

    native_wire = None
    if reference is not None and (reference.get('wire_mask_source_scope') == 'native_resolved_instance_id'
            and reference.get('wire_instance_resolution', {}).get('wire_inventory_complete') is True
            and not reference.get('wire_mask_blocked_reasons')):
        native_wire = reference.get('wire_reference_mask')
    finger_fraction = None if finger is None else fraction(finger)
    visibility = None
    if native_wire is not None:
        target = reference['depth_m']
        target_valid = reference['valid'] & torch.isfinite(target) & (target >= near) & (target <= far)
        visibility = reference_visibility(target, reference.get('right_depth_m'), target_valid & native_wire,
                                          calibration.K_left, calibration.baseline_m)
        maps['native_wire_reference_mask'] = native_wire
        maps['native_wire_right_occlusion_proxy'] = visibility['left_only_occluded']

    for i, row in enumerate(rows):
        identity = frame['identity'][i]
        row.update(identity=identity, tracker_valid_slots=int(extracted['point_valid'][i].sum()),
                   observed_surface_slots=int((extracted['point_provenance'][i] == 2).sum()),
                   missing_centerline_slots=int(((extracted['point_provenance'][i] == 1)
                                                & ~extracted['point_valid'][i]).sum()),
                   centerline_pixels=int(skeleton[i].sum()),
                   centerline_depth_valid=int((skeleton[i] & accepted[i]).sum()),
                   centerline_band_pixels=int(band[i].sum()),
                   centerline_band_depth_valid=int((band[i] & accepted[i]).sum()),
                   centerline_band_window_wire_rgb_fraction=distribution(rgb_fraction[i][band[i]]),
                   centerline_band_window_gray_std=distribution(gray_std[i][band[i]]),
                   centerline_band_window_left_finger_fraction=(None if finger_fraction is None
                                                                else distribution(finger_fraction[i][band[i]])))
        row['native_reference_metrics'] = None
        if native_wire is None:
            continue
        select = slice(i, i + 1)
        def score(valid, roi):
            return region_metrics(depth[select], target[select], valid[select], target_valid[select],
                                  roi[select], calibration.K_left)
        row['native_reference_metrics'] = dict(
            dense_depth=score(accepted, native_wire), sampled_points=score(sampled, native_wire),
            centerline_band_depth=score(accepted, native_wire & band),
            candidate_precision=int((native_wire[i] & wire[i]).sum()) / row['wire_candidate_count']
            if row['wire_candidate_count'] else None,
            right_visibility_proxy={name: None if mask is None else int(mask[i].sum())
                                    for name, mask in visibility.items()})
        if finger_fraction is not None:
            touches = finger_fraction > 0
            row['native_reference_metrics'].update(
                wire_left_window_touches_finger=score(accepted, native_wire & touches),
                wire_left_window_clear_of_finger=score(accepted, native_wire & ~touches),
                points_left_window_touches_finger=score(sampled, native_wire & touches))
    return maps, rows
