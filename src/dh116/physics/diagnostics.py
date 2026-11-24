"""Upper-layer reference/correspondence/visibility/retained-wire geometry diagnostics; never imported at top level by foundational vision/matching."""
import math as stere_error_diagn__math
import copy
import math as stere_wire_corre__math
import torch as stere_wire_diagn__torch
from dataclasses import dataclass
import torch as stere_error_diagn__torch
import importlib
import torch as stere_wire_corre__torch
import torch.nn.functional as stere_wire_diagn__F
from dataclasses import fields
import torch.nn.functional as stere_error_diagn__F
import hashlib
from dh116.physics.vision import StereoDepth as StereoDepth
from enum import IntEnum
from dh116.physics.vision import backproject
import inspect
import torch as stere_wire_geome__torch

def _aligned(*values):
    if not values or values[0].ndim != 3 or any((v.shape != values[0].shape for v in values)):
        raise ValueError('aligned [B,H,W] diagnostic tensors required')
import json

def distribution(values):
    v = values.detach().double().flatten()
    v = v[stere_error_diagn__torch.isfinite(v)]
    if not len(v):
        return dict(count=0, mean=None, p50=None, p90=None, p95=None, p99=None, max=None)
    return dict(count=len(v), mean=float(v.mean()), **{f'p{q}': float(v.quantile(q / 100)) for q in (50, 90, 95, 99)}, max=float(v.max()))
import math as stere_refer__math

def region_metrics(prediction, reference, predicted_valid, reference_valid, region, K, *, disparity=None):
    """Keep the reference denominator, prediction count and joint errors separate."""
    _aligned(prediction, reference, predicted_valid, reference_valid, region)
    if any((m.dtype != stere_error_diagn__torch.bool for m in (predicted_valid, reference_valid, region))):
        raise ValueError('explicit boolean masks required')
    eligible = region & reference_valid & stere_error_diagn__torch.isfinite(reference) & (reference > 0)
    accepted = region & predicted_valid & stere_error_diagn__torch.isfinite(prediction) & (prediction > 0)
    joint = eligible & accepted
    denominator = int(eligible.sum())
    count = int(joint.sum())
    signed = (prediction - reference)[joint].double()
    absolute = signed.abs()
    indices = joint.nonzero()
    k = stere_error_diagn__torch.as_tensor(K, device=prediction.device, dtype=stere_error_diagn__torch.float64)
    if k.shape == (3, 3):
        k = k[None].expand(prediction.shape[0], -1, -1)
    if k.shape != (prediction.shape[0], 3, 3) or not stere_error_diagn__torch.isfinite(k).all():
        raise ValueError('finite batched K required')
    pixel = stere_error_diagn__torch.stack((indices[:, 2].double() + 0.5, indices[:, 1].double() + 0.5, stere_error_diagn__torch.ones(count, device=prediction.device)), 1)
    rays = stere_error_diagn__torch.bmm(stere_error_diagn__torch.linalg.inv(k)[indices[:, 0]], pixel[:, :, None]).squeeze(-1)
    spatial = absolute * rays.norm(dim=-1)
    d = distribution(absolute)
    result = dict(reference_valid_count=denominator, predicted_valid_count=int(accepted.sum()), intersection_count=count, prediction_coverage=count / denominator if denominator else None, mae_m=d['mean'], rmse_m=float(signed.square().mean().sqrt()) if count else None, signed_bias_m=float(signed.mean()) if count else None, **{key + '_m': d[key] for key in ('p50', 'p90', 'p95', 'p99', 'max')}, correct_3d_points_5mm=int((spatial <= 0.005).sum()), correct_3d_point_completeness_5mm=float((spatial <= 0.005).sum()) / denominator if denominator else None, correct_z_completeness_5mm=float((absolute <= 0.005).sum()) / denominator if denominator else None, point_error_3d_m=distribution(spatial), point_error_3d_rmse_m=float(spatial.square().mean().sqrt()) if count else None)
    if disparity is not None:
        _aligned(prediction, disparity)
        values = disparity[joint].double()
        values = values[stere_error_diagn__torch.isfinite(values)]
        edges = [0.0, 8.0, 16.0, 32.0, 64.0, 96.0, 128.0, 192.0, 256.0, 512.0, 1024.0]
        result['disparity_histogram'] = dict(edges_px=edges, counts=stere_error_diagn__torch.histogram(values.cpu(), bins=stere_error_diagn__torch.tensor(edges, dtype=stere_error_diagn__torch.float64)).hist.long().tolist(), outside_histogram_count=int(((values < edges[0]) | (values > edges[-1])).sum()))
    return result
from pathlib import Path

def wire_bands(mask):
    if mask.ndim != 3 or mask.dtype != stere_error_diagn__torch.bool:
        raise ValueError('wire mask must be bool [B,H,W]')
    interior = stere_error_diagn__F.avg_pool2d(mask.float()[:, None], 3, 1, 1)[:, 0] >= 1.0
    return dict(wire=mask, wire_edge=mask & ~interior, wire_interior=interior)
import re

def reference_visibility(left_z, right_z, left_valid, K, baseline_m, *, tolerance_m=0.001):
    """Validation-only reprojection against actual right optical-Z.

    Missing right reference remains NA. Disagreement with a farther right
    surface is reported separately from right-side occlusion; neither becomes
    a claim about a hidden full wire. Threshold is a diagnostic parameter only.
    """
    if not stere_error_diagn__math.isfinite(tolerance_m) or tolerance_m < 0 or baseline_m <= 0:
        raise ValueError('invalid visibility tolerance/baseline')
    _aligned(left_z, left_valid)
    if right_z is None:
        return {k: None for k in ('common_visible', 'left_only_occluded', 'outside_right_fov', 'reference_disagreement', 'unknown_right')}
    _aligned(left_z, right_z)
    from dh116.physics.vision import StereoDepth as StereoDepth
    k = stere_error_diagn__torch.as_tensor(K, device=left_z.device, dtype=left_z.dtype)
    fx = k[0, 0] if k.ndim == 2 else k[:, 0, 0, None, None]
    good = left_valid & stere_error_diagn__torch.isfinite(left_z) & (left_z > 0)
    x = stere_error_diagn__torch.arange(left_z.shape[-1], device=left_z.device, dtype=left_z.dtype)[None, None, :].expand_as(left_z)
    xr = x - fx * baseline_m / stere_error_diagn__torch.where(good, left_z, 1.0)
    right_valid = stere_error_diagn__torch.isfinite(right_z) & (right_z > 0)
    (rz, available) = StereoDepth.sample_horizontal(stere_error_diagn__torch.where(right_valid, right_z, 0.0), xr, right_valid)
    inside = stere_error_diagn__torch.isfinite(xr) & (xr >= 0) & (xr <= left_z.shape[-1] - 1)
    delta = rz - left_z
    return dict(common_visible=good & available & (delta.abs() <= tolerance_m), left_only_occluded=good & available & (delta < -tolerance_m), outside_right_fov=good & ~inside, reference_disagreement=good & available & (delta > tolerance_m), unknown_right=good & inside & ~available)
import torch as stere_refer__torch

def rejection_summary(reasons, region):
    """Overlapping masks are not an additive exclusive causal decomposition."""
    if region.ndim != 3 or region.dtype != stere_error_diagn__torch.bool:
        raise ValueError('boolean [B,H,W] ROI required')
    remaining = region.clone()
    overlap = {}
    first = {}
    for (name, mask) in reasons.items():
        if mask is None:
            overlap[name] = None
            first[name] = None
            continue
        _aligned(region, mask)
        if mask.dtype != stere_error_diagn__torch.bool:
            raise ValueError('boolean reason masks required')
        overlap[name] = int((region & mask).sum())
        first[name] = int((remaining & mask).sum())
        remaining &= ~mask
    return dict(denominator=int(region.sum()), overlapping_counts=overlap, first_failure_counts=first, first_failure_order=list(reasons), not_rejected_count=int(remaining.sum()), scope='diagnostic_condition_counts;correlation_not_unique_cause')
AUDITED_CAMERA_VIEW_SHA256 = '58557bbcb900574a4b9858c13e9e479b96fd5dfe09d872b07d5c010cca64d99e'

@stere_error_diagn__torch.no_grad()
def sampled_cost_curves(left_rgb, right_rgb, pixels, disparities, block_size, *, wire_mask=None):
    """Bounded selected-pixel SAD/Census/gradient probes, not a new matcher.

    Full curves are saved only for explicit [env,row,column] samples. Census
    uses each patch centre as its threshold. Outside-window support is null.
    """
    if left_rgb.shape != right_rgb.shape or left_rgb.ndim != 4 or left_rgb.shape[1] != 3 or (pixels.ndim != 2) or (pixels.shape[1] != 3) or (block_size < 3) or (block_size % 2 != 1):
        raise ValueError('RGB pair, [P,3] pixels and odd block size required')
    a = left_rgb.mean(1)
    b = right_rgb.mean(1)
    (n, h, w) = a.shape
    r = block_size // 2
    output = []
    for (env, y, x) in pixels.long().tolist():
        if not (0 <= env < n and r <= y < h - r and (r <= x < w - r)):
            continue
        pa = a[env, y - r:y + r + 1, x - r:x + r + 1]
        sad = []
        census = []
        gradient = []
        for d in disparities:
            d = int(d)
            xx = x - d
            if not r <= xx < w - r:
                sad.append(None)
                census.append(None)
                gradient.append(None)
                continue
            pb = b[env, y - r:y + r + 1, xx - r:xx + r + 1]
            sad.append(float((pa - pb).abs().mean()))
            census.append(int(((pa < pa[r, r]) != (pb < pb[r, r])).sum()))
            gradient.append(float((pa[:, 1:] - pa[:, :-1] - (pb[:, 1:] - pb[:, :-1])).abs().mean()))
        finite = [(value, i) for (i, value) in enumerate(sad) if value is not None]
        finite.sort()
        non_adjacent = []
        best = finite[0][0] if finite else None
        for (value, i) in finite:
            if value > best + max(1e-08, 0.05 * abs(best)):
                break
            if all((abs(i - j) > 1 for j in non_adjacent)):
                non_adjacent.append(i)
        best_i = finite[0][1] if finite else None
        curvature = None
        offset = None
        if best_i is not None and 0 < best_i < len(sad) - 1 and (sad[best_i - 1] is not None) and (sad[best_i + 1] is not None):
            curvature = sad[best_i - 1] + sad[best_i + 1] - 2 * sad[best_i]
            if curvature > 1e-08:
                offset = 0.5 * (sad[best_i - 1] - sad[best_i + 1]) / curvature
        output.append(dict(env_index=env, row=y, column=x, disparities_px=list(disparities), sad_cost=sad, census_hamming=census, horizontal_gradient_cost=gradient, non_adjacent_minimum_count=len(non_adjacent), best_disparity_integer_px=disparities[best_i] if best_i is not None else None, fit_curvature=curvature, unclamped_subpixel_offset_px=offset, left_patch_gray_std=float(pa.std(unbiased=False)), left_patch_wire_fraction=float(wire_mask[env, y - r:y + r + 1, x - r:x + r + 1].float().mean()) if wire_mask is not None else None, scope='selected_pixel_offline_cost_probe;not_production_matcher_output'))
    return output
AUDITED_INSTANCE_ANNOTATOR_SHA256 = '756dbfb4f262e623606bcbff2ddb315be723cf371e7a72983493d7e9977ddd56'

@stere_error_diagn__torch.no_grad()
def replay_depth_conditions(left_rgb, right_rgb, calibration, config):
    """Replay existing private matcher functions; never change their decisions.

    Saved invalid disparity/confidence is NaN/zero, so its rejection cannot be
    recovered from the saved output alone. This is explicitly an offline replay
    on this device, checked against the saved native validity by the caller.
    """
    from dh116.physics.vision import StereoDepth as StereoDepth
    if not calibration.rectified:
        raise ValueError('rectified raw diagnostic inputs required')
    matcher = StereoDepth(**config)
    result = matcher.compute(left_rgb, right_rgb, calibration)
    diagnostic = result['diagnostics']
    r = diagnostic['reasons']
    reasons = dict(matcher_support_or_subpixel_rejection=r['left_matcher_rejected'], low_texture_left=r['low_texture_left'], right_support_or_texture_rejection=r['right_sample_unavailable'], left_right_inconsistent=r['left_right_inconsistent'], uniqueness_rejection=r['uniqueness'], out_of_depth_range=r['depth_range'], occlusion_composite=r['occlusion_composite'], patch_border_rejection=r['patch_border'], depth_edge_rejection=None, low_texture_right=r['low_texture_right'], search_boundary=r['search_boundary'], subpixel_rejected=r['subpixel_rejected'], true_occlusion=r['true_occlusion'])
    return dict(valid=result['valid'], reasons=reasons, raw_disparity_px=diagnostic['raw_disparity_px'], raw_confidence=diagnostic['raw_confidence'], matcher_diagnostics=diagnostic, scope='offline_replay_existing_baseline;not_saved_native_rejection_trace', uniqueness_scope='OpenCV_internal_rejection_not_separately_exposed' if matcher.method == 'sgbm' else diagnostic['uniqueness_scope'])

def validate_reference_timing(pair, reference, *, native=False):
    """Require the same actual render clocks, not only equal physics time."""
    left = pair.get('render_timing')
    right = reference.get('render_timing')
    if not native and left is None and (right is None):
        return
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError('independent render clocks unavailable')
    keys = ('frame_id', 'simulation_time', 'reference_time_numerator', 'reference_time_denominator', 'render_frame_number', 'swh_frame_number')
    if any((left.get(key) != right.get(key) for key in keys)):
        raise ValueError('independent render product clock mismatch')
    if native:
        from dh116.env.cameras import paired_render_timestamp
        for timing in (left, right):
            actual = paired_render_timestamp(dict(referenceTimeNumerator=timing.get('reference_time_numerator'), referenceTimeDenominator=timing.get('reference_time_denominator')), timing.get('isaac_read_times'), legacy_time=timing.get('legacy_simulation_time'))
            if any((actual[key] != timing.get(key) for key in keys)):
                raise ValueError('raw render clock does not reproduce timing metadata')

def resolve_wire_instance_mask(ids, info, wire_prim_paths, *, path_validator=None, wire_render_prim_paths=None, reference_valid=None):
    """Resolve actual instance-ID prim paths, never substring/class/colour labels.

    Unknown metadata is an explicit unavailable reference. Native callers supply
    an actual USD stage validator; CPU/offline callers can validate archived path
    readback. Each environment is matched only against its own real Wire root.
    """
    mapping = {}
    background = set()
    validated = set()
    renderer_namespaces = {}
    ignored = []
    unresolved = []

    def result(mask=None, reason=None, per_env=None, complete=False):
        return dict(mask=mask, id_to_path=mapping, per_env_wire_ids=per_env or [], stage_validated_paths=sorted(validated), blocked_reasons=[] if reason is None else [reason], background_ids=sorted(background), non_wire_renderer_instance_namespaces=renderer_namespaces, unresolved_ids=unresolved, unresolved_ids_outside_reference_validity=ignored, wire_inventory_complete=complete)
    failed = lambda reason: result(reason=reason)
    if not isinstance(ids, stere_refer__torch.Tensor) or ids.ndim != 3 or ids.dtype not in (stere_refer__torch.int32, stere_refer__torch.int64):
        return failed('instance_ids_must_be_signed_integer_BHW')
    if not isinstance(wire_prim_paths, (list, tuple)) or len(wire_prim_paths) != len(ids) or len(set(wire_prim_paths)) != len(wire_prim_paths):
        return failed('actual_per_environment_wire_roots_unavailable_or_duplicated')
    pattern = '/(?:[A-Za-z_][A-Za-z0-9_:]*)(?:/[A-Za-z_][A-Za-z0-9_:]*)*'

    def valid_path(value):
        return isinstance(value, str) and re.fullmatch(pattern, value) is not None
    if any((not valid_path(root) for root in wire_prim_paths)):
        return failed('actual_wire_root_is_not_absolute_USD_prim_path')
    if path_validator is not None and any((not path_validator(root) for root in wire_prim_paths)):
        return failed('actual_wire_root_missing_from_USD_stage_readback')
    validated.update(wire_prim_paths)
    inventory = wire_render_prim_paths
    if inventory is not None:
        if not isinstance(inventory, list) or len(inventory) != len(ids) or any((not isinstance(paths, list) or not paths for paths in inventory)):
            return failed('complete_per_environment_actual_Wire_render_geometry_inventory_required')
        for (root, paths) in zip(wire_prim_paths, inventory):
            if len(set(paths)) != len(paths) or any((not valid_path(path) or not (path == root or path.startswith(root + '/')) or (path_validator is not None and (not path_validator(path))) for path in paths)):
                return failed('Wire_render_geometry_inventory_not_actual_stage_descendants')
            validated.update(paths)
        if not isinstance(reference_valid, stere_refer__torch.Tensor) or reference_valid.shape != ids.shape or reference_valid.dtype != stere_refer__torch.bool or (reference_valid.device != ids.device):
            return failed('actual_optical_Z_reference_validity_required_for_instance_inventory')
    labels = info.get('idToLabels') if isinstance(info, dict) else None
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except ValueError:
            return failed('unsupported_idToLabels_JSON')
    if not isinstance(labels, dict):
        return failed('unsupported_instance_info_format_missing_idToLabels')
    for (key, value) in labels.items():
        if isinstance(key, bool) or not re.fullmatch('[0-9]+', str(key)):
            return failed('unsupported_instance_ID_key')
        number = int(key)
        if isinstance(value, str):
            if value == 'BACKGROUND' and number == 0:
                background.add(number)
                continue
            if value.lstrip().startswith('{'):
                try:
                    value = json.loads(value)
                except ValueError:
                    continue
        path = value.get('primPath') if isinstance(value, dict) else value
        if not valid_path(path):
            match = re.fullmatch('(.+)\\.proto_mesh_id[0-9]+', path) if isinstance(path, str) else None
            base = match[1] if match is not None else None
            if inventory is None or not valid_path(base) or any((base == root or base.startswith(root + '/') for root in wire_prim_paths)):
                continue
            renderer_namespaces[number] = base
            continue
        if path_validator is not None and (not path_validator(path)):
            continue
        mapping[number] = path
        validated.add(path)
    seen = set(stere_refer__torch.unique(ids).detach().cpu().tolist())
    unresolved = sorted((value for value in seen if value not in mapping and value not in background and (value not in renderer_namespaces)))
    for value in unresolved:
        if inventory is None or bool(((ids == value) & reference_valid).any()):
            return failed('visible_instance_ID_has_no_unambiguous_stage_validated_prim_path')
        ignored.append(value)
    per_env = []
    mask = stere_refer__torch.zeros_like(ids, dtype=stere_refer__torch.bool)
    for (env, root) in enumerate(wire_prim_paths):
        owned = [key for (key, path) in mapping.items() if path == root or path.startswith(root + '/')]
        if not owned:
            return failed('Wire_prim_not_resolved_to_instance_ID_for_environment_' + str(env))
        if inventory is not None and any((not any((path == mapping[key] or path.startswith(mapping[key] + '/') for key in owned)) for path in inventory[env])):
            return failed('actual_Wire_render_geometry_missing_from_instance_label_map_for_environment_' + str(env))
        per_env.append(sorted(owned))
        mask[env] = stere_refer__torch.isin(ids[env], stere_refer__torch.tensor(owned, device=ids.device, dtype=ids.dtype))
    return result(mask=mask, per_env=per_env, complete=inventory is not None)
stere_wire_corre__reference_visibility = reference_visibility
stere_wire_diagn__distribution = distribution
stere_wire_diagn__region_metrics = region_metrics
stere_wire_diagn__reference_visibility = reference_visibility
stere_wire_geome__region_metrics = region_metrics

class _NativeValidationDepthReader:

    def __init__(self, camera, wire_prim_paths=None):
        from dh116.env.cameras import _NativeRGBPairReader as _NativeRGBPairReader
        from dh116.env.cameras import _CameraAxesDeviceMixin
        from dh116.env.cameras import numeric_slot_order
        rgb = camera.reader
        if not isinstance(rgb, _NativeRGBPairReader):
            raise ValueError('validation reference requires an actual native RGB rig')
        base = importlib.import_module('isaacsim.sensors.camera').CameraView
        source = Path(inspect.getfile(base)).resolve()
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if source_sha != AUDITED_CAMERA_VIEW_SHA256:
            raise ValueError('unaudited CameraView source; optical-Z annotator mapping requires explicit source review')
        self.sdk_source_artifact = dict(path=str(source), sha256=source_sha)
        self.wire_prim_paths = list(wire_prim_paths) if wire_prim_paths is not None else None

        class DeviceSafeDepthView(_CameraAxesDeviceMixin, base):
            pass
        rep = importlib.import_module('omni.replicator.core')
        annotator = Path(inspect.getfile(importlib.import_module('omni.replicator.core.scripts.annotators_default'))).resolve()
        annotator_sha = hashlib.sha256(annotator.read_bytes()).hexdigest()
        if annotator_sha != AUDITED_INSTANCE_ANNOTATOR_SHA256:
            raise ValueError('unaudited renderer instance annotator source')
        self.sdk_instance_source_artifact = dict(path=str(annotator), sha256=annotator_sha)
        for name in ('sim', 'calibration', 'device', 'origins', 'n', 'count', 'Usd', 'UsdGeom', 'stage'):
            setattr(self, name, getattr(rgb, name))
        self.core = rgb.core
        c = self.calibration
        self.view = DeviceSafeDepthView(prim_paths_expr='/World/DH116StereoRGB/camera_.*', name='dh116_stereo_validation_depth', camera_resolution=(c.width, c.height), output_annotators=['depth', 'instance_id_segmentation_fast'], reset_xform_properties=False)
        self.paths = list(self.view.prim_paths)
        self.order = numeric_slot_order(self.paths, self.count).to(self.device)
        width = int(self.view.tiled_resolution[0])
        self.columns = width // c.width
        if width != self.columns * c.width:
            raise ValueError('invalid validation reference tile layout')
        self.reference = rep.AnnotatorRegistry.get_annotator('ReferenceTime', device='cpu')
        self.times = rep.AnnotatorRegistry.get_annotator('IsaacReadTimes', device='cpu')
        self.reference.attach([self.view.get_render_product_path()])
        self.times.attach([self.view.get_render_product_path()])

    def _wire_render_geometry(self):
        if self.wire_prim_paths is None:
            return None
        result = []
        for path in self.wire_prim_paths:
            root = self.stage.GetPrimAtPath(path)
            if not root.IsValid():
                return None
            result.append(sorted((str(prim.GetPath()) for prim in self.Usd.PrimRange(root) if prim.IsA(self.UsdGeom.Gprim) and self.UsdGeom.Imageable(prim).ComputeVisibility() != self.UsdGeom.Tokens.invisible)))
        return result

    def read(self):
        from dh116.env.cameras import _NativeRGBPairReader as _NativeRGBPairReader
        from dh116.env.cameras import paired_render_timestamp
        from dh116.env.cameras import detile
        reference = self.reference.get_data()
        if not isinstance(reference, dict) or 'referenceTimeNumerator' not in reference:
            raise ValueError('validation reference render frame unavailable')
        legacy = float(self.core.get_sim_time_at_swh_frame(int(reference['referenceTimeNumerator'])))
        before = paired_render_timestamp(reference, self.times.get_data(), legacy_time=legacy)
        c = self.calibration
        depth = detile(self.view.get_depth_tiled(device='cuda'), count=self.count, width=c.width, height=c.height, columns=self.columns, channels=1).to(self.device)[self.order].reshape(self.n, 2, c.height, c.width)
        segmentation = None
        segmentation_info = None
        segmentation_error = None
        try:
            value = self.view.get_data('instance_id_segmentation_fast', tiled=True)
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError('unsupported segmentation get_data return format')
            (data, segmentation_info) = value
            if isinstance(data, stere_refer__torch.Tensor):
                data = data.detach().cpu()
            elif hasattr(data, 'numpy'):
                data = stere_refer__torch.as_tensor(data.numpy())
            else:
                data = stere_refer__torch.as_tensor(data)
            segmentation = detile(data.long(), count=self.count, width=c.width, height=c.height, columns=self.columns, channels=1).to(self.device)[self.order].reshape(self.n, 2, c.height, c.width)
        except (ValueError, TypeError, RuntimeError) as error:
            segmentation_error = 'instance_reference_unavailable:' + str(error)
        after = paired_render_timestamp(self.reference.get_data(), self.times.get_data(), legacy_time=legacy)
        if (before['frame_id'], before['simulation_time'], before['render_frame_number']) != (after['frame_id'], after['simulation_time'], after['render_frame_number']):
            raise ValueError('validation reference frame/time changed during independent depth read')
        _NativeRGBPairReader._readback(self)
        inventory = self._wire_render_geometry()
        reference_valid = stere_refer__torch.isfinite(depth[:, 0]) & (depth[:, 0] > 0)
        resolution = resolve_wire_instance_mask(segmentation[:, 0], segmentation_info, self.wire_prim_paths, path_validator=lambda path: self.stage.GetPrimAtPath(path).IsValid(), wire_render_prim_paths=inventory, reference_valid=reference_valid) if segmentation is not None and inventory is not None else None
        right_resolution = resolve_wire_instance_mask(segmentation[:, 1], segmentation_info, self.wire_prim_paths, path_validator=lambda path: self.stage.GetPrimAtPath(path).IsValid(), wire_render_prim_paths=inventory, reference_valid=stere_refer__torch.isfinite(depth[:, 1]) & (depth[:, 1] > 0)) if segmentation is not None and inventory is not None else None
        return dict(depth_left_m=depth[:, 0].float(), depth_right_m=depth[:, 1].float(), frame_id=before['frame_id'], simulation_time=before['simulation_time'], calibration_sha256=c.sha256, render_timing=before, sdk_camera_view_source_artifact=self.sdk_source_artifact, sdk_instance_annotator_source_artifact=self.sdk_instance_source_artifact, raw_instance_ids=None if segmentation is None else segmentation[:, 0], raw_instance_info=segmentation_info, right_raw_instance_ids=None if segmentation is None else segmentation[:, 1], right_raw_instance_info=segmentation_info, right_wire_instance_resolution=right_resolution, wire_prim_paths=self.wire_prim_paths, wire_render_prim_paths=inventory, wire_instance_resolution=resolution, segmentation_blocked_reason=segmentation_error)
from dh116.physics.vision import validate_images as validate_images
from dh116.physics.stereo_match import wire_colour_mask as stere_wire_diagn__wire_colour_mask

class PointSource(IntEnum):
    UNKNOWN = 0
    CENTERLINE = 1
    OBSERVED_SURFACE = 2

class ValidationDepthReference:

    def __init__(self, reader):
        self.reader = reader

    @classmethod
    def from_camera(cls, camera, wire_prim_paths=None):
        return cls(_NativeValidationDepthReader(camera, wire_prim_paths))

    @stere_refer__torch.no_grad()
    def read(self, pair):
        value = self.reader.read()
        if not isinstance(value, dict):
            raise ValueError('independent reference frame unavailable')
        for name in ('frame_id', 'simulation_time', 'calibration_sha256'):
            if value.get(name) != pair.get(name):
                raise ValueError('reference RGB pair frame/time/calibration mismatch: ' + name)
        native = isinstance(self.reader, _NativeValidationDepthReader)
        validate_reference_timing(pair, value, native=native)
        (left, right) = (value.get('depth_left_m'), value.get('depth_right_m'))
        shape = (len(pair['left_rgb']), *pair['left_rgb'].shape[-2:])
        if not isinstance(left, stere_refer__torch.Tensor) or not isinstance(right, stere_refer__torch.Tensor) or left.shape != shape or (right.shape != shape):
            raise ValueError('reference optical depth shape mismatch')
        valid = stere_refer__torch.isfinite(left) & (left > 0)
        from dh116.env.cameras import brown_mask
        candidate = brown_mask(pair['left_rgb'].permute(0, 2, 3, 1), float_rgb_is_255=False)
        resolution = value.get('wire_instance_resolution') or {}
        right_resolution = value.get('right_wire_instance_resolution') or {}
        return dict(version='dh116-stereo-depth-reference-v1', source_scope='native' if native else 'cpu_fixture', validation_only=True, reference_used_as_actor_input=False, source_units='optical_axis_z_m', reference_source='rendered_depth_validation_only', frame_id=value['frame_id'], simulation_time=value['simulation_time'], calibration_sha256=value['calibration_sha256'], depth_m=left.clone(), valid=valid.clone(), right_depth_m=right.clone(), rgb_candidate_mask=candidate.clone(), candidate_mask_source='RGB_brown_candidate_not_semantic_ground_truth', wire_reference_mask=resolution.get('mask'), wire_mask_source_scope=('native_resolved_instance_id' if native else 'cpu_resolved_instance_id') if resolution.get('mask') is not None else 'unresolved', wire_mask_blocked_reasons=resolution.get('blocked_reasons', [value.get('segmentation_blocked_reason') or 'instance_reference_unavailable']), raw_instance_ids=value.get('raw_instance_ids'), raw_instance_info=copy.deepcopy(value.get('raw_instance_info')), right_raw_instance_ids=value.get('right_raw_instance_ids'), right_raw_instance_info=copy.deepcopy(value.get('right_raw_instance_info')), right_wire_instance_resolution=right_resolution, right_wire_reference_mask=right_resolution.get('mask'), right_wire_mask_blocked_reasons=right_resolution.get('blocked_reasons', ['right_instance_reference_unavailable']), right_valid=(stere_refer__torch.isfinite(right) & (right > 0)).clone(), wire_prim_paths=copy.deepcopy(value.get('wire_prim_paths')), wire_render_prim_paths=copy.deepcopy(value.get('wire_render_prim_paths')), wire_instance_resolution=resolution, actual_wire_path_source='RigidObject.root_physx_view.prim_paths' if native else 'cpu_fixture', render_timing=copy.deepcopy(value.get('render_timing')), sdk_instance_annotator_source_artifact=copy.deepcopy(value.get('sdk_instance_annotator_source_artifact')), sdk_camera_view_source_artifact=copy.deepcopy(value.get('sdk_camera_view_source_artifact')))
from dh116.physics.stereo_match import wire_colour_mask as stere_wire_corre__wire_colour_mask
from dh116.physics.wire import _paths_from_mask

@dataclass(frozen=True)
class WirePointGeometry:
    xyz: stere_wire_geome__torch.Tensor
    source_type: stere_wire_geome__torch.Tensor
    component_id: stere_wire_geome__torch.Tensor
    arc_length: stere_wire_geome__torch.Tensor
    visibility: stere_wire_geome__torch.Tensor
    confidence: stere_wire_geome__torch.Tensor
    valid: stere_wire_geome__torch.Tensor
    track_id: stere_wire_geome__torch.Tensor
    image_track_id: stere_wire_geome__torch.Tensor
    point_pixels: stere_wire_geome__torch.Tensor
    arc_length_px: stere_wire_geome__torch.Tensor
    depth_segment_id: stere_wire_geome__torch.Tensor

    @classmethod
    def from_extraction(cls, extraction):
        required = {f.name for f in fields(cls)} - {'source_type'}
        if not required.issubset(extraction) or 'point_provenance' not in extraction:
            raise ValueError('tracked wire extraction with measured source metadata required')
        return cls(**{key: extraction[key] for key in required}, source_type=extraction['point_provenance'])

    def __post_init__(self):
        if not isinstance(self.valid, stere_wire_geome__torch.Tensor) or self.valid.ndim != 2 or (not self.valid.numel()):
            raise ValueError('wire validity must be a nonempty [B,P] tensor')
        (shape, device) = (self.valid.shape, self.valid.device)
        dtypes = dict(xyz=None, source_type=stere_wire_geome__torch.uint8, component_id=stere_wire_geome__torch.long, arc_length=None, visibility=stere_wire_geome__torch.bool, confidence=None, valid=stere_wire_geome__torch.bool, track_id=stere_wire_geome__torch.long, image_track_id=stere_wire_geome__torch.long, point_pixels=stere_wire_geome__torch.long, arc_length_px=None, depth_segment_id=stere_wire_geome__torch.long)
        for (name, dtype) in dtypes.items():
            value = getattr(self, name)
            expected = (*shape, 3) if name == 'xyz' else (*shape, 2) if name == 'point_pixels' else shape
            if not isinstance(value, stere_wire_geome__torch.Tensor) or value.shape != expected or value.device != device or (not value.is_floating_point() if dtype is None else value.dtype != dtype):
                raise ValueError(f'{name} has incompatible wire geometry shape, dtype or device')
        known = self.source_type != PointSource.UNKNOWN
        center = self.source_type == PointSource.CENTERLINE
        surface = self.source_type == PointSource.OBSERVED_SURFACE
        if bool((self.source_type > PointSource.OBSERVED_SURFACE).any()) or not stere_wire_geome__torch.equal(known, self.visibility) or (not stere_wire_geome__torch.equal(known, self.component_id >= 0)) or (not stere_wire_geome__torch.equal(known, (self.point_pixels >= 0).all(-1))):
            raise ValueError('known source must bind one visible image component and pixel')
        if bool((self.valid & ~known).any()) or not stere_wire_geome__torch.equal(self.valid, stere_wire_geome__torch.isfinite(self.xyz).all(-1)) or (not bool(stere_wire_geome__torch.isnan(self.xyz[~self.valid]).all())):
            raise ValueError('unmeasured metric coordinates must be missing')
        if not bool(stere_wire_geome__torch.isfinite(self.confidence).all()) or bool(((self.confidence < 0) | (self.confidence > 1)).any()) or bool((self.confidence[~self.valid] != 0).any()):
            raise ValueError('wire confidence must be in [0,1] and zero without measured depth')
        if bool((surface & ~self.valid).any()) or bool(((self.track_id != -1) & ~center).any()) or bool(((self.image_track_id != -1) & ~center).any()) or bool(((self.depth_segment_id != -1) & ~(center & self.valid)).any()) or (not bool(stere_wire_geome__torch.isnan(self.arc_length[~(center & self.valid)]).all())) or (not bool(stere_wire_geome__torch.isnan(self.arc_length_px[~center]).all())):
            raise ValueError('observed surface and unknown slots cannot carry centerline arc or track identity')
        if not stere_wire_geome__torch.equal(center & self.valid, self.depth_segment_id >= 0) or not bool((stere_wire_geome__torch.isfinite(self.arc_length[center & self.valid]) & (self.arc_length[center & self.valid] >= 0)).all()) or (not bool((stere_wire_geome__torch.isfinite(self.arc_length_px[center]) & (self.arc_length_px[center] >= 0)).all())):
            raise ValueError('centerline arc lengths require their measured segment support')

    def as_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def sensor_fields(self):
        """Supply only the geometry tensors already admitted by SensorBatch."""
        return dict(points_world=self.xyz.float(), point_valid=self.valid, point_confidence=self.confidence.float(), point_provenance=self.source_type)

def archive_instance_reference(reference, path):
    """Archive unmodified ID/map readback separately from the derived mask.

    Unknown formats are archived too; a stored artifact does not grant source
    qualification. The native promotion gate independently resolves this data.
    """
    if reference.get('raw_instance_ids') is None:
        return None
    names = ('source_scope', 'frame_id', 'simulation_time', 'calibration_sha256', 'render_timing', 'raw_instance_ids', 'raw_instance_info', 'right_raw_instance_ids', 'right_raw_instance_info', 'right_wire_instance_resolution', 'right_wire_reference_mask', 'right_valid', 'wire_prim_paths', 'actual_wire_path_source', 'sdk_camera_view_source_artifact', 'sdk_instance_annotator_source_artifact', 'wire_render_prim_paths')
    raw = {key: copy.deepcopy(reference.get(key)) for key in names}
    raw.update(version='dh116-wire-instance-reference-v2', validation_only=True, reference_valid=reference['valid'].clone(), stage_validated_paths=copy.deepcopy(reference.get('wire_instance_resolution', {}).get('stage_validated_paths', [])))
    path = Path(path)
    with path.open('xb') as stream:
        stere_refer__torch.save(raw, stream)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def segmentation_evidence(output):
    """Keep both RGB views and three logits distinct from physical visibility."""
    from dh116.physics.wire import segmentation_masks
    (masks, _) = segmentation_masks(output, diagnostic_only=True, mask_policy='legacy_visible_and_unknown')
    probabilities = output['logits'].sigmoid()
    result = {}
    for (index, side) in enumerate(('left', 'right')):
        record = {name + '_probability': probabilities[:, index, head] for (head, name) in enumerate(('visible', 'occluded', 'unknown'))}
        visible = record['visible_probability'] >= 0.5
        record.update(visible_mask=visible, runtime_mask=masks[index], unknown_rejection_mask=visible & ~masks[index])
        result[side] = record
    return result
_REJECTION_ORDER = ('patch_border', 'low_texture_left', 'candidate_wire_no_joint_patch', 'candidate_wire_low_texture', 'candidate_wire_no_shape_evidence', 'candidate_wire_cost_rejected', 'candidate_wire_ambiguous', 'repeated_texture', 'subpixel_rejected', 'left_matcher_rejected', 'out_of_right_view', 'right_sample_unavailable', 'low_texture_right', 'left_right_inconsistent', 'occlusion_composite', 'uniqueness', 'depth_range')

def source_reference_metrics(geometry, depth, depth_valid, reference_depth, reference_valid, K):
    """Score each source against the same fixed native support and 5 mm rule.

    Caller establishes native ID provenance and the reference range before scoring.
    No 3D centerline ground truth is inferred from the native optical-Z image.
    """
    if depth.ndim != 3 or depth.shape[0] != geometry.valid.shape[0] or depth_valid.shape != depth.shape or (depth_valid.dtype != stere_wire_geome__torch.bool) or (depth_valid.device != depth.device) or (geometry.valid.device != depth.device):
        raise ValueError('aligned depth and wire geometry batch required')
    (env, slot) = geometry.valid.nonzero(as_tuple=True)
    (y, x) = geometry.point_pixels[env, slot].unbind(-1)
    if bool(((y < 0) | (x < 0) | (y >= depth.shape[1]) | (x >= depth.shape[2])).any()):
        raise ValueError('measured point pixel outside depth image')
    if not bool(depth_valid[env, y, x].all()):
        raise ValueError('measured geometry lacks corresponding valid stereo depth')
    flat_indices = (env * depth.shape[1] + y) * depth.shape[2] + x
    if flat_indices.unique().numel() != flat_indices.numel():
        raise ValueError('duplicated sampled depth pixel across sources')
    native = stere_wire_geome__torch.ones_like(reference_valid)
    scores = {}
    for source in PointSource:
        selected = stere_wire_geome__torch.zeros_like(depth_valid)
        keep = geometry.source_type[env, slot] == source
        selected[env[keep], y[keep], x[keep]] = True
        scores[source.name.lower()] = stere_wire_geome__region_metrics(depth, reference_depth, selected, reference_valid, native, K)
    return scores

def capture_validation_pair(camera, reference, identity, *, max_attempts=8):
    """Retry the WHOLE real RGB pair; reference read itself performs no render."""
    failures = []
    for _ in range(max_attempts):
        pair = camera.sample(identity=identity)
        try:
            ref = reference.read(pair)
            return (pair, ref, failures)
        except ValueError as error:
            failures.append(dict(frame_id=pair.get('frame_id'), simulation_time=pair.get('simulation_time'), error=str(error)))
    raise RuntimeError('same-frame independent reference unavailable; failed pairs=' + repr(failures))

def _finite(value):
    value = float(value)
    return value if stere_wire_corre__math.isfinite(value) else None

def wire_intersections(wire, stereo, sampled, reasons):
    """Return additive first rejects and independent overlapping conditions.

    First-reject priority is an attribution convention over observed flags. It
    is not a causal ordering inside every matching backend.
    """
    if any((mask.shape != wire.shape or mask.dtype != stere_wire_diagn__torch.bool for mask in (wire, stereo, sampled))):
        raise ValueError('aligned boolean wire/depth/point masks required')
    if bool((sampled & ~(wire & stereo)).any()):
        raise ValueError('a point must use a measured valid candidate pixel')
    keys = [name for name in _REJECTION_ORDER if reasons.get(name) is not None]
    keys += [name for (name, mask) in reasons.items() if name not in keys and mask is not None]
    first = stere_wire_diagn__torch.zeros_like(wire, dtype=stere_wire_diagn__torch.int16)
    first[~wire] = -1
    rejected_depth = wire & ~stereo
    for (i, name) in enumerate(keys, 1):
        mask = reasons[name]
        if mask.shape != wire.shape or mask.dtype != stere_wire_diagn__torch.bool:
            raise ValueError('unaligned rejection condition ' + name)
        first[rejected_depth & mask & (first == 0)] = i
    omitted = wire & stereo & ~sampled
    first[omitted] = len(keys) + 1
    if bool((rejected_depth & (first == 0)).any()):
        first[rejected_depth & (first == 0)] = len(keys) + 2
    labels = [*keys, 'not_sampled_as_3d_point', 'depth_rejection_reason_unavailable']
    lr_failed = stere_wire_diagn__torch.zeros_like(wire)
    for name in ('occlusion_composite', 'left_right_inconsistent', 'right_sample_unavailable'):
        if reasons.get(name) is not None:
            lr_failed |= reasons[name]
    range_failed = reasons.get('depth_range')
    maps = dict(wire_mask=wire, wire_left_depth_valid=wire & stereo, wire_left_right_consistent=wire & ~lr_failed if any((reasons.get(name) is not None for name in ('occlusion_composite', 'left_right_inconsistent', 'right_sample_unavailable'))) else None, wire_depth_range_valid=wire & ~range_failed if range_failed is not None else None, wire_3d_point_valid=sampled, first_reject_code=first)
    rows = []
    for (index, mask) in enumerate(wire):
        count = int(mask.sum())
        accepted = int(sampled[index].sum())
        codes = first[index]
        first_counts = {name: int((codes == i).sum()) for (i, name) in enumerate(labels, 1)}
        assert sum(first_counts.values()) + accepted == count
        overlaps = {name: int((mask & value[index]).sum()) if value is not None else None for (name, value) in reasons.items()}
        overlaps['not_sampled_as_3d_point'] = int(omitted[index].sum())
        rows.append(dict(wire_candidate_count=count, valid_point_count=accepted, valid_point_coverage=accepted / count if count else None, intersection_counts={name: None if value is None else int(value[index].sum()) for (name, value) in maps.items() if name != 'first_reject_code'}, first_reject_legend={i: name for (i, name) in enumerate(labels, 1)}, first_rejections=first_counts, overlapping_rejections=overlaps))
    return (maps, rows)

def source_temporal_metrics(current, *, previous=None, env):
    """Image/metric component retention in an externally verified same episode.

    For surface only, image retention means the associated RGB fragment has a
    tracked centerline component; it says nothing about a surface point's ID.
    """

    def component_tracks(geometry, source, field):
        selected = geometry.visibility[env] & (geometry.source_type[env] == source)
        components = geometry.component_id[env, selected].unique()
        centerline = geometry.source_type[env] == PointSource.CENTERLINE
        rows = centerline & stere_wire_geome__torch.isin(geometry.component_id[env], components)
        ids = getattr(geometry, field)[env, rows]
        return set(ids[ids >= 0].tolist())
    out = {}
    for source in PointSource:
        selected = current.source_type[env] == source
        visible = int((current.visibility[env] & selected).sum())
        measured = int((current.valid[env] & selected).sum())
        row = dict(visible_slots=visible, metric_valid_slots=measured, visible_slot_depth_coverage=measured / visible if visible else None, material_point_identity_verified=False)
        for (name, field) in (('image', 'image_track_id'), ('metric', 'track_id')):
            applicable = source == PointSource.CENTERLINE or (name == 'image' and source == PointSource.OBSERVED_SURFACE)
            ids = component_tracks(current, source, field) if applicable else None
            prior = component_tracks(previous, source, field) if applicable and previous is not None else None
            row[f'{name}_component_tracks'] = len(ids) if ids is not None else None
            row[f'previous_{name}_component_tracks'] = len(prior) if prior is not None else None
            row[f'shared_{name}_component_tracks'] = len(ids & prior) if prior is not None else None
        out[source.name.lower()] = row
    return out

def state_supervision_binding(identity, reference, reference_sha256):
    """Bind a native label-only reference to the RGB and pre-action sample."""
    from dh116.env.contracts import require_sha
    keys = ('frame_id', 'reset_id', 'physics_tick_start', 'physics_tick_end', 'simulation_timestamp', 'calibration_sha256')
    if identity['frame_id'] != reference.get('frame_id') or identity['simulation_timestamp'] != reference.get('simulation_time') or identity['calibration_sha256'] != reference.get('calibration_sha256') or (identity['physics_tick_start'] != identity['physics_tick_end']):
        raise ValueError('same-frame native state reference binding required')
    if reference.get('source_scope') != 'native' or reference.get('validation_only') is not True or reference.get('reference_used_as_actor_input') is not False or (reference.get('source_units') != 'optical_axis_z_m'):
        raise ValueError('independent native optical-Z reference is label-only')
    sha = require_sha(reference_sha256, 'native state reference SHA')
    if sha == '0' * 64:
        raise ValueError('nonzero native state reference SHA required')
    return dict(version='dh116-state-synchronization-v1', **{key: identity[key] for key in keys}, depth_input_source='rgb_stereo', reference_role='labels_and_audit_only', reference_sha256=sha, left_native_ids_available=isinstance(reference.get('raw_instance_ids'), stere_refer__torch.Tensor), right_native_ids_available=isinstance(reference.get('right_raw_instance_ids'), stere_refer__torch.Tensor))

def _curve(values):
    return [_finite(value) for value in values.tolist()]

def _pixel_bounds(mask):
    pixels = mask.nonzero()
    if not len(pixels):
        return None
    return [*pixels.amin(dim=0).tolist(), *pixels.amax(dim=0).tolist()]

def compare_reference(pair, reference):
    """Error arithmetic; visible-wire completeness uses an explicit ID mask."""
    for key in ('frame_id', 'simulation_time', 'calibration_sha256'):
        if pair.get(key) != reference.get(key):
            raise ValueError('reference alignment mismatch')
    (estimate, target) = (pair['depth_m'], reference['depth_m'])
    good_ref = reference['valid'] & stere_refer__torch.isfinite(target) & (target > 0)
    if 'evaluation_depth_range_m' in reference:
        (near, far) = reference['evaluation_depth_range_m']
        good_ref &= (target >= near) & (target <= far)
    valid = pair['valid'] & stere_refer__torch.isfinite(estimate) & good_ref
    if estimate.shape != target.shape or valid.shape != target.shape:
        raise ValueError('depth/reference shape mismatch')
    error = (estimate[valid] - target[valid]).float()
    count = int(valid.sum())
    denominator = int(good_ref.sum())
    candidate = reference.get('rgb_candidate_mask')
    candidate_den = good_ref & candidate if isinstance(candidate, stere_refer__torch.Tensor) else stere_refer__torch.zeros_like(valid)
    candidate_count = int(candidate_den.sum())
    candidate_valid = valid & candidate_den
    wire = reference.get('wire_reference_mask')
    wire_total = wire_reconstructed = None
    wire_error = None
    wire_count = None
    wire_reason = 'no_resolved_independent_instance_reference'
    if isinstance(wire, stere_refer__torch.Tensor):
        if wire.shape != valid.shape or wire.dtype != stere_refer__torch.bool:
            raise ValueError('wire instance reference mask shape/dtype mismatch')
        if reference.get('wire_mask_source_scope') not in ('native_resolved_instance_id', 'cpu_resolved_instance_id'):
            raise ValueError('wire reference must derive from explicit resolved instance IDs')
        if pair.get('K_left') is None:
            raise ValueError('calibration K required for 3D completeness')
        from dh116.physics.vision import backproject
        truth = backproject(target, pair['K_left'], good_ref)
        reconstructed = backproject(estimate, pair['K_left'], pair['valid'])
        distance = (truth - reconstructed).norm(dim=-1)
        wire_total = int((wire & good_ref).sum())
        wire_reconstructed = int((wire & valid & (distance <= 0.005)).sum())
        wire_error = (estimate[wire & valid] - target[wire & valid]).float()
        wire_count = len(wire_error)
        wire_reason = None if wire_total else 'no_reference_visible_wire_points'
    return dict(depth_mae_m=float(error.abs().mean()) if count else None, depth_rmse_m=float(error.square().mean().sqrt()) if count else None, depth_p95_m=float(error.abs().quantile(0.95)) if count else None, depth_validity_coverage=count / denominator if denominator else 0.0, depth_valid_count=count, depth_reference_valid_count=denominator, rgb_candidate_point_cloud_completeness=float(candidate_valid.sum()) / candidate_count if candidate_count else None, rgb_candidate_reference_points=candidate_count, rgb_candidate_reconstructed_points=int(candidate_valid.sum()), candidate_mask_source=reference.get('candidate_mask_source'), wire_point_cloud_completeness=wire_reconstructed / wire_total if wire_total else None, wire_reference_visible_points=wire_total, wire_reconstructed_points=wire_reconstructed, wire_depth_mae_m=float(wire_error.abs().mean()) if wire_count else None, wire_depth_rmse_m=float(wire_error.square().mean().sqrt()) if wire_count else None, wire_depth_p95_m=float(wire_error.abs().quantile(0.95)) if wire_count else None, wire_depth_count=wire_count, wire_depth_validity_coverage=wire_count / wire_total if wire_total else None, wire_completeness_scope='dense_reference_visible_wire_pixels;not_hidden_entire_object', wire_completeness_reason=wire_reason)

class SoftWireCorrespondence(StereoDepth):
    """Opt-in sparse full-patch ZNCC; legacy processing identities stay intact."""

    def __init__(self, *, mask_weight=0.03, **options):
        super().__init__(**options)
        if not stere_wire_corre__math.isfinite(mask_weight) or not 0 <= mask_weight <= 0.05:
            raise ValueError('mask weight must be between 0 and .05')
        self.mask_weight = mask_weight

    def _search(self, a, b, pa, pb, centers, sign, point_traces=None):
        (_, h, w) = a.shape
        radius = self.block_size // 2
        o = stere_wire_corre__torch.arange(-radius, radius + 1, device=a.device)
        (dy, dx) = (v.flatten() for v in stere_wire_corre__torch.meshgrid(o, o, indexing='ij'))
        hi = min(self.max_disparity, w - 1)
        disparities = stere_wire_corre__torch.arange(self.min_disparity, hi + 1, device=a.device, dtype=a.dtype)
        out = stere_wire_corre__torch.full_like(a, float('nan'))
        quality = stere_wire_corre__torch.zeros_like(a)
        patch = stere_wire_corre__torch.zeros_like(centers)
        texture = patch.clone()
        unique = patch.clone()
        photo = patch.clone()
        edge = patch.clone()
        if disparities.numel() == 0:
            return (out, quality, dict(patch=patch, texture=texture, unique=unique, photo=photo, edge=edge))
        for group in centers.nonzero().split(64):
            if not len(group):
                continue
            (env, y, x) = group.unbind(-1)
            sy = y[:, None] + dy
            sx = x[:, None] + dx
            source_inside = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
            (sy, safe_sx) = (sy.clamp(0, h - 1), sx.clamp(0, w - 1))
            source = a[env[:, None], sy, safe_sx][:, None]
            centered = source - source.mean(-1, keepdim=True)
            va = centered.square().mean(-1)
            source_p = pa[env, y, x][:, None]

            def cost(candidate):
                xx = sx[:, None, :] + sign * candidate[..., None]
                (lo, up) = (xx.floor(), xx.ceil())
                inside = source_inside.all(-1)[:, None] & (lo >= 0).all(-1) & (up < w).all(-1)
                lo_index = lo.long().clamp(0, w - 1)
                up_index = up.long().clamp(0, w - 1)
                (be, by) = (env[:, None, None], sy[:, None, :])
                low = b[be, by, lo_index]
                target = low + (b[be, by, up_index] - low) * (xx - lo)
                target -= target.mean(-1, keepdim=True)
                vb = target.square().mean(-1)
                evidence = inside & (va > max(self.texture, 0.0001) ** 2) & (vb > max(self.texture, 0.0001) ** 2)
                raw = 1 - ((centered * target).mean(-1) / (va * vb).sqrt().clamp_min(1e-12)).clamp(-1, 1)
                c = len(dx) // 2
                (plo, pup) = (pb[be, by, lo_index][..., c], pb[be, by, up_index][..., c])
                target_p = plo + (pup - plo) * (xx - lo)[..., c]
                guided = raw + self.mask_weight * (source_p - target_p).abs()
                return (guided.masked_fill(~evidence, float('inf')), raw, inside)
            (guided, raw, inside) = cost(disparities[None])
            (best, winner) = guided.min(-1)
            integer = disparities[winner]
            d = integer
            rgb_cost = raw.gather(1, winner[:, None])[:, 0]
            final_rgb = rgb_cost
            alternative = guided.masked_fill((disparities - d[:, None]).abs() <= 1, float('inf')).amin(-1)
            finite = stere_wire_corre__torch.isfinite(best) & stere_wire_corre__torch.isfinite(alternative)
            margin = stere_wire_corre__torch.where(finite, (alternative - best) / alternative.clamp_min(1e-08), 0.0)
            tied = ((guided - best[:, None]).abs() <= 1e-06).sum(-1) > 1
            clear = finite & (margin > self.uniqueness) & ~tied
            rgb_ok = rgb_cost <= 0.2
            boundary = (d == self.min_disparity) | (d == hi)
            accepted = clear & rgb_ok & ~boundary
            fractional = None
            if self.subpixel and accepted.any():
                phase = stere_wire_corre__torch.arange(-6, 7, device=a.device, dtype=a.dtype) / 8
                offsets = d[:, None] + phase
                (scores, raw_fractional, _) = cost(offsets)
                (improved, winner_fractional) = scores.min(-1)
                selected = winner_fractional[:, None]
                improved_rgb = raw_fractional.gather(1, selected)[:, 0]
                choose = accepted & (improved < best - 1e-05) & (improved_rgb < rgb_cost)
                d = stere_wire_corre__torch.where(choose, offsets.gather(1, selected)[:, 0], d)
                final_rgb = stere_wire_corre__torch.where(choose, improved_rgb, rgb_cost)
                fractional = (offsets, scores, raw_fractional)
            idx = (env, y, x)
            patch[idx] = inside.any(-1)
            texture[idx] = stere_wire_corre__torch.isfinite(best)
            unique[idx] = clear
            photo[idx] = rgb_ok
            edge[idx] = boundary & stere_wire_corre__torch.isfinite(best)
            out[idx] = stere_wire_corre__torch.where(accepted, d, float('nan'))
            quality[idx] = stere_wire_corre__torch.where(accepted, margin, 0.0)
            if point_traces:
                for (i, key) in enumerate(map(tuple, group.tolist())):
                    if key not in point_traces:
                        continue
                    raw_supported = raw[i].masked_fill(~stere_wire_corre__torch.isfinite(guided[i]), float('nan'))
                    attempted = bool(accepted[i])
                    record = dict(pixel_eyx=key, integer_disparities_px=disparities.tolist(), integer_total_cost=_curve(guided[i]), integer_rgb_cost=_curve(raw_supported), integer_winner_px=float(integer[i]) if stere_wire_corre__torch.isfinite(best[i]) else None, integer_winner_rgb_cost=_finite(raw_supported[winner[i]]), winner_rgb_cost=float(final_rgb[i]) if stere_wire_corre__torch.isfinite(best[i]) else None, subpixel_offset_px=float(d[i] - integer[i]) if attempted else None, local_match_accepted=attempted, accepted_disparity_px=float(d[i]) if attempted else None, margin=float(margin[i]), best_cost=_finite(best[i]), separated_cost=_finite(alternative[i]))
                    if fractional is not None and attempted:
                        (phase, phase_cost, phase_rgb) = (value[i] for value in fractional)
                        record['fractional_curve'] = dict(disparities_px=phase.tolist(), total_cost=_curve(phase_cost), rgb_cost=_curve(phase_rgb.masked_fill(~stere_wire_corre__torch.isfinite(phase_cost), float('nan'))))
                    point_traces[key] = record
        return (out, quality, dict(patch=patch, texture=texture, unique=unique, photo=photo, edge=edge))

    @stere_wire_corre__torch.no_grad()
    def compute(self, left, right, calibration, *, probabilities, left_proposal=None, trace_pixels=()):
        validate_images(left, right, calibration)
        if not calibration.rectified:
            raise ValueError('rectified stereo required')
        if len(probabilities) != 2 or any((p.shape != left[:, 0].shape or p.device != left.device or (not p.is_floating_point()) or (not stere_wire_corre__torch.isfinite(p).all()) or (p < 0).any() or (p > 1).any() for p in probabilities)):
            raise ValueError('aligned finite RGB-derived probabilities required')
        (lp, rp) = probabilities
        if left_proposal is None:
            left_proposal = lp >= 0.5
        if left_proposal.shape != lp.shape or left_proposal.dtype != stere_wire_corre__torch.bool or left_proposal.device != lp.device:
            raise ValueError('aligned boolean left wire proposal required')
        candidates = stere_wire_corre__wire_colour_mask(left) | left_proposal
        (lg, rg) = (left.mean(1), right.mean(1))
        (b, h, w) = lg.shape
        left_points = {}
        for pixel in trace_pixels:
            if len(pixel) != 3 or any((type(v) is not int for v in pixel)) or (not all((0 <= v < bound for (v, bound) in zip(pixel, lg.shape)))):
                raise ValueError('trace pixels must be integer env/row/column indices inside the image')
            left_points[tuple(pixel)] = None
        (dl, lc, left_trace) = self._search(lg, rg, lp, rp, candidates, -1, left_points)
        x = stere_wire_corre__torch.arange(w, device=left.device, dtype=dl.dtype)[None, None].expand(b, h, w)
        xr = x - dl
        reverse = stere_wire_corre__torch.zeros_like(candidates)
        (env, row, col) = stere_wire_corre__torch.isfinite(dl).nonzero(as_tuple=True)
        projected = xr[env, row, col]
        for position in (projected.floor(), projected.ceil()):
            inside = (position >= 0) & (position < w)
            reverse[env[inside], row[inside], position[inside].long()] = True
        right_points = {}
        right_queries = {}
        for key in left_points:
            target = _finite(xr[key])
            (e, y, _) = key
            positions = () if target is None else sorted({stere_wire_corre__math.floor(target), stere_wire_corre__math.ceil(target)})
            right_queries[key] = [(e, y, x) for x in positions if 0 <= x < w]
            right_points.update(dict.fromkeys(right_queries[key]))
        (dr, rc, _) = self._search(rg, lg, rp, lp, reverse, 1, right_points)
        reverse_valid = stere_wire_corre__torch.isfinite(dr)
        (back, available) = self.sample_horizontal(dr, xr, reverse_valid)
        (back_conf, conf_valid) = self.sample_horizontal(rc, xr, reverse_valid)
        lr_ok = available & conf_valid & stere_wire_corre__torch.isfinite(dl) & ((dl - back).abs() <= self.lr)
        z = float(calibration.K_left[0, 0]) * calibration.baseline_m / dl
        depth_ok = stere_wire_corre__torch.isfinite(z) & (z >= self.near) & (z <= self.far)
        accepted = lr_ok & depth_ok
        reasons = dict(patch_border=~left_trace['patch'], low_texture_left=left_trace['patch'] & ~left_trace['texture'], candidate_wire_ambiguous=left_trace['texture'] & ~left_trace['unique'], candidate_wire_cost_rejected=left_trace['texture'] & ~left_trace['photo'], search_boundary=left_trace['edge'], left_matcher_rejected=~stere_wire_corre__torch.isfinite(dl), right_sample_unavailable=stere_wire_corre__torch.isfinite(dl) & ~available, left_right_inconsistent=available & ~lr_ok, depth_range=stere_wire_corre__torch.isfinite(dl) & ~depth_ok)
        diagnostics = dict(reasons=reasons, raw_disparity_px=dl, right_search_center_count=int(reverse.sum()), physical_occlusion_inferred=False)
        if left_points:
            diagnostics['point_traces'] = {key: dict(left=record, right=[right_points[k] for k in right_queries[key]], right_projected_x_px=_finite(xr[key]), lr_error_px=_finite((dl[key] - back[key]).abs()) if available[key] else None, reverse_sample_available=bool(available[key]), accepted=bool(accepted[key])) for (key, record) in left_points.items()}
        return dict(depth_m=z.masked_fill(~accepted, float('nan')), disparity_px=dl.masked_fill(~accepted, float('nan')), valid=accepted, confidence=stere_wire_corre__torch.where(accepted, stere_wire_corre__torch.minimum(lc, back_conf), 0.0), candidate_mask=candidates, lr_consistency=lr_ok, diagnostics=diagnostics)

def _span_fraction(support, reference, axis):
    if support is None or reference is None:
        return None
    reference_span = reference[axis + 2] - reference[axis]
    return (support[axis + 2] - support[axis]) / reference_span if reference_span else None

def silhouette_point_trace(left_rgb, right_rgb, masks, pixel, matcher, prediction):
    """Re-evaluate the integer silhouette costs at one actual audit position."""
    if matcher.method != 'wire_silhouette' or matcher.subpixel:
        raise ValueError('integer silhouette matcher required for this audit trace')
    (left, right) = (left_rgb.mean(1), right_rgb.mean(1))
    (height, width) = left.shape[-2:]
    radius = matcher.block_size // 2
    offsets = stere_wire_corre__torch.arange(-radius, radius + 1, device=left.device)
    (oy, ox) = (values.flatten() for values in stere_wire_corre__torch.meshgrid(offsets, offsets, indexing='ij'))
    high = min(matcher.max_disparity, width - 1)
    disparities = stere_wire_corre__torch.arange(matcher.min_disparity, high + 1, device=left.device)

    def curve(source, target, source_mask, target_mask, center, sign):
        (e, y, x) = center
        if not source_mask[center] or not disparities.numel():
            return None
        (yy, xx) = (y + oy, x + ox)
        if not ((yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)).all():
            return dict(pixel_eyx=center, integer_disparities_px=disparities.tolist(), integer_total_cost=[None] * len(disparities), integer_winner_px=None, subpixel_offset_px=None, winner_rgb_cost=None, local_match_accepted=False, accepted_disparity_px=None, metric='wire_silhouette')
        u = xx[None] + sign * disparities[:, None]
        in_bounds = (u >= 0) & (u < width)
        safe_u = u.clamp(0, width - 1)
        a_mask = source_mask[e, yy, xx]
        b_mask = target_mask[e, yy, safe_u] & in_bounds
        joint = (a_mask & b_mask).sum(-1)
        count = int(a_mask.sum())
        support = b_mask[:, len(ox) // 2] & (joint >= 3) & (joint >= 0.6 * stere_wire_corre__torch.maximum(a_mask.sum(), b_mask.sum(-1)))
        feasible = support & in_bounds.all(-1) & (min(count, len(ox) - count) >= 2)
        values = (a_mask ^ b_mask).float().mean(-1).masked_fill(~feasible, float('inf'))
        (best, choice) = values.min(0)
        integer = disparities[choice]
        alternative = values.masked_fill((disparities - integer).abs() <= 1, float('inf')).min()
        margin = (alternative - best) / alternative.abs().clamp_min(1e-08) if stere_wire_corre__torch.isfinite(best) and stere_wire_corre__torch.isfinite(alternative) else 0.0
        tied = (stere_wire_corre__torch.isfinite(values) & ((values - best).abs() <= 1e-06)).sum() != 1
        accepted = bool(margin > matcher.uniqueness and (not tied) and (best <= 0) and (matcher.min_disparity < integer < high))
        a = source[e, yy, xx]
        b = target[e, yy, safe_u]
        (a, b) = (a - a.mean(), b - b.mean(-1, keepdim=True))
        covariance_scale = (a.square().mean() * b.square().mean(-1)).sqrt()
        rgb = 1 - (a * b).mean(-1).div(covariance_scale.clamp_min(1e-12)).clamp(-1, 1)
        rgb = rgb.masked_fill(~in_bounds.all(-1) | (covariance_scale <= 1e-12), float('nan'))
        return dict(pixel_eyx=center, integer_disparities_px=disparities.tolist(), integer_total_cost=_curve(values), integer_rgb_cost=_curve(rgb), integer_winner_px=float(integer) if stere_wire_corre__torch.isfinite(best) else None, subpixel_offset_px=0.0 if accepted else None, winner_rgb_cost=_finite(rgb[choice]) if stere_wire_corre__torch.isfinite(best) else None, local_match_accepted=accepted, accepted_disparity_px=float(integer) if accepted else None, metric='wire_silhouette', rgb_cost_used_for_ranking=False)
    forward = curve(left, right, masks[0], masks[1], pixel, -1)
    raw = _finite(prediction['diagnostics']['raw_disparity_px'][pixel])
    reproduced = None if forward is None else forward['accepted_disparity_px']
    if raw != reproduced:
        raise RuntimeError('integer silhouette trace disagrees with the actual forward match')
    (e, y, x) = pixel
    target_x = None if raw is None else x - int(raw)
    reverse = [] if target_x is None else [curve(right, left, masks[1], masks[0], (e, y, target_x), 1)]
    reverse_d = None if not reverse or reverse[0] is None else reverse[0]['accepted_disparity_px']
    available = not bool(prediction['diagnostics']['reasons']['right_sample_unavailable'][pixel])
    return dict(left=forward, right=reverse, right_projected_x_px=target_x, lr_error_px=abs(raw - reverse_d) if available and raw is not None and (reverse_d is not None) else None, reverse_sample_available=available, accepted=bool(prediction['valid'][pixel]))

def _reconstruction_evidence(dense, sparse, accepted, sampled, fixed, slots):
    """Use resolved wire pixels for 5 mm scoring and report unscored support separately."""
    reference_count = dense['reference_valid_count']
    dense_scored = dense['intersection_count']
    sparse_scored = sparse['intersection_count']
    sparse_pixels = int(sampled.sum())
    reference_bounds = _pixel_bounds(fixed)
    dense_bounds = _pixel_bounds(accepted & fixed)
    sparse_bounds = _pixel_bounds(sampled & fixed)
    return dict(version='wire_reconstruction_evidence_v1', error_contract='same_pixel_euclidean_3d_at_most_5mm', reference_visible_wire_pixels=reference_count, dense=dict(accepted_matcher_pixels=int(accepted.sum()), scored_roi_pixels=dense_scored, unscored_roi_pixels=dense['predicted_valid_count'] - dense_scored, correct_pixels_5mm=dense['correct_3d_points_5mm'], precision_5mm=dense['correct_3d_points_5mm'] / dense_scored if dense_scored else None, reference_recall_5mm=dense['correct_3d_point_completeness_5mm']), sparse=dict(slot_capacity=slots, accepted_unique_pixels=sparse_pixels, scored_unique_pixels=sparse_scored, unscored_unique_pixels=sparse_pixels - sparse_scored, correct_pixels_5mm=sparse['correct_3d_points_5mm'], point_precision_5mm=sparse['correct_3d_points_5mm'] / sparse_scored if sparse_scored else None, sampled_correct_pixel_contribution=sparse['correct_3d_point_completeness_5mm'], contribution_upper_bound=min(slots, reference_count) / reference_count if reference_count else None), image_spatial_support=dict(scope='scored_left_image_pixels; bounding_spans_do_not_measure_holes_or_hidden_surface', reference_bounds_yxyx=reference_bounds, dense_scored_bounds_yxyx=dense_bounds, sparse_scored_bounds_yxyx=sparse_bounds, dense_scored_row_span_fraction=_span_fraction(dense_bounds, reference_bounds, 0), dense_scored_column_span_fraction=_span_fraction(dense_bounds, reference_bounds, 1), sparse_scored_row_span_fraction=_span_fraction(sparse_bounds, reference_bounds, 0), sparse_scored_column_span_fraction=_span_fraction(sparse_bounds, reference_bounds, 1)))

def colour_anchor_fallback(baseline, soft, colour_masks):
    """Opt-in fallback on RGB silhouette support without re-ranking baseline depths."""
    (left_colour, right_colour) = colour_masks
    shape = baseline['valid'].shape
    if any((mask.shape != shape or mask.dtype != stere_wire_corre__torch.bool for mask in (left_colour, right_colour, baseline['valid'], soft['valid'], soft['lr_consistency']))):
        raise ValueError('aligned RGB colour and stereo validity masks required')
    disparity = soft['disparity_px']
    columns = stere_wire_corre__torch.arange(shape[-1], dtype=disparity.dtype, device=disparity.device)
    target_x = columns[None, None] - disparity
    (_, right_support) = StereoDepth.sample_horizontal(right_colour.float(), target_x, right_colour)
    baseline_valid = baseline['valid'] & left_colour
    fallback = ~baseline_valid & left_colour & soft['valid'] & soft['lr_consistency'] & right_support
    valid = baseline_valid | fallback
    result = {key: stere_wire_corre__torch.where(baseline_valid, baseline[key], soft[key]).masked_fill(~valid, 0.0 if key == 'confidence' else float('nan')) for key in ('disparity_px', 'depth_m', 'confidence')}
    reasons = dict(fallback_right_colour_unavailable=left_colour & soft['valid'] & ~right_support, fallback_soft_stereo_rejected=left_colour & ~soft['valid'])
    result.update(valid=valid, lr_consistency=valid, candidate_mask=left_colour, diagnostics=dict(reasons=reasons, raw_disparity_px=result['disparity_px'], baseline_depth_preserved=baseline_valid, soft_fallback_depth=fallback, physical_occlusion_inferred=False))
    return result

@stere_wire_diagn__torch.no_grad()
def audit_wire_frame(frame, prediction, extracted, calibration, reference, *, block_size, near, far, finger=None, wire_mask=None):
    """Use same-frame validation references only for scoring and occlusion audit."""
    wire = stere_wire_diagn__wire_colour_mask(frame['left_rgb']) if wire_mask is None else wire_mask
    depth = prediction['depth_m']
    accepted = prediction['valid'] & stere_wire_diagn__torch.isfinite(depth) & (depth > 0)
    sampled = stere_wire_diagn__torch.zeros_like(wire)
    for (i, (pixel, valid)) in enumerate(zip(extracted['point_pixels'], extracted['point_valid'])):
        yx = pixel[valid]
        sampled[i, yx[:, 0], yx[:, 1]] = True
    reasons = prediction['diagnostics']['reasons']
    (maps, rows) = wire_intersections(wire, accepted, sampled, reasons)
    skeleton = stere_wire_diagn__torch.zeros_like(wire)
    for (i, mask) in enumerate(wire.cpu().numpy()):
        for path in _paths_from_mask(mask)[0]:
            index = stere_wire_diagn__torch.as_tensor(path, device=wire.device)
            skeleton[i, index[:, 0], index[:, 1]] = True
    radius = block_size // 2
    band = stere_wire_diagn__F.max_pool2d(skeleton[:, None].float(), 3, 1, 1)[:, 0].bool() & wire
    maps.update(wire_candidate_centerline=skeleton, wire_candidate_centerline_band=band)
    left_gray = frame['left_rgb'].mean(1)

    def fraction(mask):
        return stere_wire_diagn__F.avg_pool2d(mask[:, None].float(), block_size, 1, radius)[:, 0]
    gray_std = (fraction(left_gray.square()) - fraction(left_gray).square()).clamp_min(0).sqrt()
    rgb_fraction = fraction(wire)
    native_wire = None
    if reference is not None and (reference.get('wire_mask_source_scope') == 'native_resolved_instance_id' and reference.get('wire_instance_resolution', {}).get('wire_inventory_complete') is True and (not reference.get('wire_mask_blocked_reasons'))):
        native_wire = reference.get('wire_reference_mask')
    finger_fraction = None if finger is None else fraction(finger)
    visibility = None
    if native_wire is not None:
        target = reference['depth_m']
        target_valid = reference['valid'] & stere_wire_diagn__torch.isfinite(target) & (target >= near) & (target <= far)
        visibility = stere_wire_diagn__reference_visibility(target, reference.get('right_depth_m'), target_valid & native_wire, calibration.K_left, calibration.baseline_m)
        maps['native_wire_reference_mask'] = native_wire
        maps['native_wire_right_occlusion_proxy'] = visibility['left_only_occluded']
    for (i, row) in enumerate(rows):
        identity = frame['identity'][i]
        row.update(identity=identity, tracker_valid_slots=int(extracted['point_valid'][i].sum()), observed_surface_slots=int((extracted['point_provenance'][i] == 2).sum()), missing_centerline_slots=int(((extracted['point_provenance'][i] == 1) & ~extracted['point_valid'][i]).sum()), centerline_pixels=int(skeleton[i].sum()), centerline_depth_valid=int((skeleton[i] & accepted[i]).sum()), centerline_band_pixels=int(band[i].sum()), centerline_band_depth_valid=int((band[i] & accepted[i]).sum()), centerline_band_window_wire_rgb_fraction=stere_wire_diagn__distribution(rgb_fraction[i][band[i]]), centerline_band_window_gray_std=stere_wire_diagn__distribution(gray_std[i][band[i]]), centerline_band_window_left_finger_fraction=None if finger_fraction is None else stere_wire_diagn__distribution(finger_fraction[i][band[i]]))
        row['native_reference_metrics'] = None
        if native_wire is None:
            continue
        select = slice(i, i + 1)

        def score(valid, roi):
            return stere_wire_diagn__region_metrics(depth[select], target[select], valid[select], target_valid[select], roi[select], calibration.K_left)
        dense = score(accepted, native_wire)
        sparse = score(sampled, native_wire)
        row['native_reference_metrics'] = dict(dense_depth=dense, sampled_points=sparse, reconstruction_evidence=_reconstruction_evidence(dense, sparse, accepted[i], sampled[i], target_valid[i] & native_wire[i], extracted['point_valid'].shape[1]), centerline_band_depth=score(accepted, native_wire & band), candidate_precision=int((native_wire[i] & wire[i]).sum()) / row['wire_candidate_count'] if row['wire_candidate_count'] else None, right_visibility_proxy={name: None if mask is None else int(mask[i].sum()) for (name, mask) in visibility.items()})
        if finger_fraction is not None:
            touches = finger_fraction > 0
            row['native_reference_metrics'].update(wire_left_window_touches_finger=score(accepted, native_wire & touches), wire_left_window_clear_of_finger=score(accepted, native_wire & ~touches), points_left_window_touches_finger=score(sampled, native_wire & touches))
    return (maps, rows)

def retain_wire_sample_slots(baseline, candidate, *, surface_start):
    """Reserve all original slots and propose new surfaces in unused capacity.

    The inputs are two tracker outputs on the same colour-mask topology. A
    proposed surface needs a distinct measured pixel on a baseline component.
    The candidate pool remains separate even when no output slot is available.
    This sensor-only function does not use reference labels or admit a policy.
    """
    _stere_wire_corre_import__PointSource = PointSource
    _stere_wire_corre_import__WirePointGeometry = WirePointGeometry
    if not isinstance(baseline, _stere_wire_corre_import__WirePointGeometry) or not isinstance(candidate, _stere_wire_corre_import__WirePointGeometry):
        raise TypeError('typed baseline and candidate wire geometry required')
    if baseline.valid.shape != candidate.valid.shape or baseline.valid.device != candidate.valid.device:
        raise ValueError('baseline and candidate geometry must be aligned')
    slots = baseline.valid.shape[1]
    if type(surface_start) is not int or not 0 <= surface_start <= slots or bool((baseline.source_type[:, :surface_start] == _stere_wire_corre_import__PointSource.OBSERVED_SURFACE).any()) or bool((baseline.source_type[:, surface_start:] == _stere_wire_corre_import__PointSource.CENTERLINE).any()):
        raise ValueError('invalid baseline centerline/surface slot partition')
    pool = stere_wire_corre__torch.zeros_like(candidate.valid)
    proposed_from = stere_wire_corre__torch.full_like(baseline.component_id, -1)
    result = {name: value.clone() for (name, value) in baseline.as_dict().items()}
    candidate_fields = candidate.as_dict()
    for env in range(len(pool)):
        visible = baseline.visibility[env]
        reserved = {tuple(p) for p in baseline.point_pixels[env, visible].tolist()}
        components = set(baseline.component_id[env, visible].tolist())
        surface = candidate.valid[env] & (candidate.source_type[env] == _stere_wire_corre_import__PointSource.OBSERVED_SURFACE)
        new_slots = []
        for (slot, pixel, component) in zip(surface.nonzero().flatten().tolist(), candidate.point_pixels[env, surface].tolist(), candidate.component_id[env, surface].tolist()):
            pixel = tuple(pixel)
            if component in components and pixel not in reserved:
                reserved.add(pixel)
                new_slots.append(slot)
        pool[env, new_slots] = True
        free = (~visible[surface_start:]).nonzero().flatten() + surface_start
        count = min(len(free), len(new_slots))
        if count:
            source = stere_wire_corre__torch.tensor(new_slots[:count], device=free.device)
            destination = free[:count]
            proposed_from[env, destination] = source
            for (name, value) in result.items():
                value[env, destination] = candidate_fields[name][env, source]
    return (_stere_wire_corre_import__WirePointGeometry(**result), pool, proposed_from)

def native_visibility(left_z, right_z, left_wire_valid, calibration, *, right_wire=None, right_known=None):
    """Classify only verified right surfaces; missing labels remain unknown."""
    if (right_wire is None) != (right_known is None):
        raise ValueError('right native identity and known-pixel masks must be paired')
    empty = stere_wire_corre__torch.zeros_like(left_wire_valid)
    if right_z is None or right_wire is None:
        return dict(visible=empty, occluded=empty.clone(), unknown=left_wire_valid.clone())
    visibility = stere_wire_corre__reference_visibility(left_z, right_z, left_wire_valid, calibration.K_left, calibration.baseline_m)
    x = stere_wire_corre__torch.arange(left_z.shape[-1], device=left_z.device, dtype=left_z.dtype)
    x = x[None, None].expand_as(left_z)
    xr = x - float(calibration.K_left[0, 0]) * calibration.baseline_m / left_z
    (wire_fraction, known) = StereoDepth.sample_horizontal(right_wire.float(), xr, right_known)
    visible = visibility['common_visible'] & known & (wire_fraction >= 1 - 1e-06)
    occluded = visibility['left_only_occluded'] & known
    return dict(visible=visible, occluded=occluded, unknown=left_wire_valid & ~(visible | occluded))

@stere_wire_corre__torch.no_grad()
def score_native_correspondence(frame, reference, calibration, prediction, masks, *, raw_visible_masks=None):
    """Score same-frame native identity after matching; fixed reference support."""
    from dh116.physics.wire import native_instance_targets
    (target, valid, admission) = native_instance_targets(frame, reference)
    wire = target[:, 0, 0].bool()
    right_wire = target[:, 1, 0].bool() if admission['right_instance_source'] != 'missing' else None
    right_known = valid[:, 1, 0] if right_wire is not None else None
    left_z = reference['depth_m']
    right_z = reference.get('right_depth_m')
    right_valid = reference.get('right_valid')
    if right_z is not None and right_valid is not None:
        if right_valid.shape != right_z.shape or right_valid.dtype != stere_wire_corre__torch.bool:
            raise ValueError('aligned right optical-Z validity required')
        right_z = right_z.masked_fill(~right_valid, float('nan'))
    reference_support = wire & reference['valid'] & stere_wire_corre__torch.isfinite(left_z) & (left_z > 0)
    visibility = native_visibility(left_z, right_z, reference_support, calibration, right_wire=right_wire, right_known=right_known)
    baseline_px_m = float(calibration.K_left[0, 0]) * calibration.baseline_m
    disparity = prediction['disparity_px']
    disparity_error = disparity - baseline_px_m / left_z
    accepted = prediction['valid'] & stere_wire_corre__torch.isfinite(disparity)
    predicted_right_wire = stere_wire_corre__torch.zeros_like(wire)
    predicted_right_evidence = stere_wire_corre__torch.zeros_like(wire)
    predicted_right_z_agrees = stere_wire_corre__torch.zeros_like(wire)
    if right_wire is not None and right_z is not None:
        columns = stere_wire_corre__torch.arange(left_z.shape[-1], dtype=left_z.dtype, device=left_z.device)
        x = columns[None, None].expand_as(left_z) - disparity
        (fraction, available) = StereoDepth.sample_horizontal(right_wire.float(), x, right_known)
        depth_known = stere_wire_corre__torch.isfinite(right_z) & (right_z > 0)
        (target_z, depth_available) = StereoDepth.sample_horizontal(right_z, x, depth_known)
        predicted_right_wire = available & (fraction >= 1 - 1e-06)
        predicted_right_evidence = available & depth_available
        predicted_right_z_agrees = depth_available & ((target_z - left_z).abs() <= 0.001)
    paired = visibility['visible'] & accepted
    resolved = paired & predicted_right_evidence
    correct = resolved & predicted_right_wire & predicted_right_z_agrees & (disparity_error.abs() <= 0.5)

    def view_iou(env, view_masks):
        iou = []
        for (view, mask) in enumerate(view_masks):
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
        for (view, side) in enumerate(('left', 'right')):
            native = valid[env, view, 0] & target[env, view, 0].bool()
            count = int(native.sum())
            raw = None if raw_visible_masks is None else native & raw_visible_masks[view][env]
            visible_count = None if raw is None else int(raw.sum())
            rejected = None if raw is None else int((raw & ~masks[view][env]).sum())
            view_counts[side] = dict(native_wire_count=count, visible_iou=raw_iou[view], runtime_iou=iou[view], visible_coverage=visible_count / count if count and visible_count is not None else None, runtime_coverage=int((native & masks[view][env]).sum()) / count if count else None, unknown_rejected_native_count=rejected if count else None, unknown_rejection_fraction=rejected / visible_count if visible_count else None)
        denominator = int(reference_support[env].sum())
        rows.append(dict(segmentation_iou_left=iou[0], segmentation_iou_right=iou[1], visible_head_iou_left=raw_iou[0], visible_head_iou_right=raw_iou[1], segmentation_view_metrics=view_counts, stereo_coverage_denominator=denominator, stereo_coverage=len(diff) / denominator if denominator else None, visibility_counts={name: int(value[env].sum()) for (name, value) in visibility.items()}, disparity_reference_scope='left_optical_Z_at_fixed_native_wire_pixels', disparity_comparison_count=len(diff), disparity_mae_px=float(diff.abs().mean()) if len(diff) else None, disparity_rmse_px=float(diff.square().mean().sqrt()) if len(diff) else None, true_correspondence_threshold_px=0.5, true_correspondence_predictions=predictions, true_correspondence_unresolved_predictions=predictions - known_predictions, true_correspondence_correct=successes, true_correspondence_accuracy=successes / known_predictions if known_predictions else None, true_correspondence_coverage=predictions / eligible if eligible else None, true_correspondence_completeness=successes / eligible if eligible else None, right_instance_source=admission['right_instance_source']))
    return (rows, visibility)
