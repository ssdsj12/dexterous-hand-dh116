"""RGB wire extraction/tracking and segmentation/source encoder used by Student vision."""
import numpy as stere_perce__np
import hashlib
from collections import deque
import torch as stere_perce__torch
import io
import math
from dh116.physics.vision import backproject as stere_perce__backproject
from pathlib import Path
import numpy as stere_wire_track__np
from dh116.physics.vision import transform_points as stere_perce__transform_points
import torch as stere_wire_segme__torch
import torch as stere_wire_track__torch

def _thin(mask):
    """Zhang–Suen thinning; delete-only topology operations on observed pixels."""
    image = stere_perce__np.pad(mask.astype(bool), 1)
    for _ in range(128):
        changed = False
        for second in (False, True):
            p = [image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:], image[2:, 2:], image[2:, 1:-1], image[2:, :-2], image[1:-1, :-2], image[:-2, :-2]]
            count = sum((x.astype(stere_perce__np.uint8) for x in p))
            transitions = sum(((~p[i] & p[(i + 1) % 8]).astype(stere_perce__np.uint8) for i in range(8)))
            center = image[1:-1, 1:-1]
            if second:
                (clear_a, clear_b) = (~(p[0] & p[2] & p[6]), ~(p[0] & p[4] & p[6]))
            else:
                (clear_a, clear_b) = (~(p[0] & p[2] & p[4]), ~(p[2] & p[4] & p[6]))
            remove = center & (count >= 2) & (count <= 6) & (transitions == 1) & clear_a & clear_b
            if remove.any():
                changed = True
                center[remove] = False
        if not changed:
            break
    return image[1:-1, 1:-1]
from torch import nn
from dh116.physics.vision import backproject as stere_wire_track__backproject

def _visible_path(skeleton):
    pixels = set(map(tuple, stere_perce__np.argwhere(skeleton)))
    if len(pixels) < 2:
        return None
    neighbours = {p: [q for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (q := (p[0] + dy, p[1] + dx)) in pixels] for p in pixels}
    ends = [p for p in pixels if len(neighbours[p]) == 1]
    if len(ends) != 2 or any((len(v) > 2 or not v for v in neighbours.values())):
        return None
    path = [min(ends)]
    used = {path[0]}
    while len(path) < len(pixels):
        nxt = [q for q in neighbours[path[-1]] if q not in used]
        if len(nxt) != 1:
            return None
        path.append(nxt[0])
        used.add(nxt[0])
    return stere_perce__np.asarray(path, dtype=stere_perce__np.int64)
import torch.nn.functional as F
from dh116.physics.vision import transform_points as stere_wire_track__transform_points

class WirePointExtractor:
    """A missing/branched/occluded cable produces invalid tokens, never truth."""

    def __init__(self, *, point_count=128, identity_motion_limit_m=0.005):
        if point_count != 128 or identity_motion_limit_m <= 0:
            raise ValueError('this contract uses 128 points and a positive consistency radius')
        self.point_count = point_count
        self.identity_motion_limit_m = float(identity_motion_limit_m)
        self._previous = None

    @stere_perce__torch.no_grad()
    def extract(self, left_rgb, depth_m, depth_valid, K_left, T_world_left, *, reset_mask=None):
        if left_rgb.ndim != 4 or left_rgb.shape[1] != 3 or depth_m.shape != left_rgb.shape[:1] + left_rgb.shape[2:] or (depth_valid.shape != depth_m.shape) or (depth_valid.dtype != stere_perce__torch.bool) or (not stere_perce__torch.isfinite(left_rgb).all()) or bool(((left_rgb < 0) | (left_rgb > 1)).any()):
            raise ValueError('finite RGB and matching stereo depth validity required')
        n = len(left_rgb)
        if self._previous is None:
            self._previous = [None] * n
        if len(self._previous) != n:
            raise ValueError('point extractor batch size changed; construct a new instance')
        if reset_mask is not None:
            if reset_mask.shape != (n,) or reset_mask.dtype != stere_perce__torch.bool:
                raise ValueError('boolean reset mask [B] required')
            for i in reset_mask.nonzero().flatten().tolist():
                self._previous[i] = None
        rgb = left_rgb
        candidate = (rgb[:, 0] > 1.07 * rgb[:, 1]) & (rgb[:, 1] > 1.1 * rgb[:, 2]) & (rgb[:, 0] > 0.12)
        candidate &= depth_valid & stere_perce__torch.isfinite(depth_m) & (depth_m > 0)
        camera = stere_perce__backproject(depth_m, K_left, depth_valid)
        world = stere_perce__transform_points(camera, T_world_left)
        points = world.new_full((n, 128, 3), float('nan'))
        valid = stere_perce__torch.zeros(n, 128, device=world.device, dtype=stere_perce__torch.bool)
        correspondence = stere_perce__torch.zeros(n, device=world.device, dtype=stere_perce__torch.bool)
        (reasons, lengths) = ([], world.new_full((n,), float('nan')))
        for i in range(n):
            path = _visible_path(_thin(candidate[i].cpu().numpy()))
            if path is None:
                self._previous[i] = None
                reasons.append('ambiguous_or_branched_visible_skeleton')
                continue
            pixel = stere_perce__torch.as_tensor(path, device=world.device)
            xyz = world[i, pixel[:, 0], pixel[:, 1]]
            axis = int((xyz.amax(0) - xyz.amin(0)).argmax())
            if xyz[0, axis] > xyz[-1, axis]:
                xyz = xyz.flip(0)
            arc = stere_perce__torch.cat((xyz.new_zeros(1), (xyz[1:] - xyz[:-1]).norm(dim=-1).cumsum(0)))
            length = float(arc[-1])
            old = self._previous[i]
            if old is not None:
                (endpoints, prior_length) = old
                movement = (xyz[[0, -1]].cpu() - endpoints).norm(dim=-1).max()
                correspondence[i] = bool(movement <= self.identity_motion_limit_m and abs(length - prior_length) <= self.identity_motion_limit_m)
            self._previous[i] = (xyz[[0, -1]].cpu().clone(), length)
            if len(xyz) > 128:
                indices = stere_perce__torch.searchsorted(arc, stere_perce__torch.linspace(0, length, 128, device=arc.device))
                xyz = xyz[indices.clamp_max(len(xyz) - 1)]
            points[i, :len(xyz)] = xyz
            valid[i, :len(xyz)] = True
            lengths[i] = length
            reasons.append(None)
        return dict(points_world=points, point_valid=valid, temporal_identity_valid=correspondence, visible_arc_length_m=lengths, rejection_reason=reasons, source='RGB_colour_visible_skeleton_and_stereo_only', correspondence_scope='visible_endpoint_consistency_not_native_identity_qualification', centerline_qualification=None)
HEADS = ('visible_wire', 'occluded_wire', 'unknown_instance')
from dh116.physics.stereo_match import wire_colour_mask
LegacyWirePointExtractor = WirePointExtractor
VIEWS = ('left', 'right')

def normalized_visible_paths(skeleton):
    """Return simple 8-connected paths, removing only redundant corner diagonals.

    A diagonal has a redundant axial route exactly when either of its shared
    axial intermediates is present. True diagonal chains retain their edges.
    Every connected component must be a simple path with at least two pixels.
    """
    pixels = set(map(tuple, stere_wire_track__np.argwhere(skeleton)))
    if not pixels:
        return ([], None)
    neighbours = {}
    for (y, x) in sorted(pixels):
        edges = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if not (dy or dx) or q not in pixels:
                    continue
                if dy and dx and ((y, x + dx) in pixels or (y + dy, x) in pixels):
                    continue
                edges.append(q)
        neighbours[y, x] = edges
    remaining = set(pixels)
    paths = []
    while remaining:
        (component, stack) = (set(), [min(remaining)])
        while stack:
            p = stack.pop()
            if p in component:
                continue
            component.add(p)
            stack.extend((q for q in neighbours[p] if q not in component))
        remaining.difference_update(component)
        ends = sorted((p for p in component if len(neighbours[p]) == 1))
        if len(ends) != 2 or any((len(neighbours[p]) > 2 or not neighbours[p] for p in component)):
            return ([], 'ambiguous_or_branched_visible_skeleton')
        (path, used) = ([ends[0]], {ends[0]})
        while len(path) < len(component):
            nexts = [q for q in neighbours[path[-1]] if q not in used]
            if len(nexts) != 1:
                return ([], 'ambiguous_or_branched_visible_skeleton')
            path.append(nexts[0])
            used.add(nexts[0])
        paths.append(stere_wire_track__np.asarray(path, dtype=stere_wire_track__np.int64))
    return (paths, None)
_legacy_visible_path = _visible_path
MASK_POLICY_VERSION = 'dh116-wire-mask-policy-v1'

def _arc_pixels(path):
    return stere_wire_track__np.concatenate(([0.0], stere_wire_track__np.linalg.norm(stere_wire_track__np.diff(path, axis=0), axis=1).cumsum()))
MASK_POLICIES = ('visible_proposal', 'legacy_visible_and_unknown', 'qualified_reliability_filter')

def _paths_from_mask(mask):
    """Return independently valid visible paths and rejected raw components."""
    _stere_wire_track_import___thin = _thin
    (paths, reason) = normalized_visible_paths(mask)
    if not reason:
        return (paths, reason, [])
    remaining = set(map(tuple, stere_wire_track__np.argwhere(mask)))
    paths = []
    rejected = []
    while remaining:
        (stack, component) = ([min(remaining)], set())
        while stack:
            p = stack.pop()
            if p in component:
                continue
            component.add(p)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (p[0] + dy, p[1] + dx)
                    if (dy or dx) and q in remaining and (q not in component):
                        stack.append(q)
        remaining.difference_update(component)
        pixel = stere_wire_track__np.array(sorted(component))
        offset = pixel.min(0)
        local = pixel - offset
        patch = stere_wire_track__np.zeros(tuple(local.max(0) + 1), dtype=bool)
        patch[local[:, 0], local[:, 1]] = True
        (local_paths, reason) = normalized_visible_paths(patch)
        thick = bool((patch[:-1, :-1] & patch[1:, :-1] & patch[:-1, 1:] & patch[1:, 1:]).any())
        if reason and thick:
            (local_paths, reason) = normalized_visible_paths(_stere_wire_track_import___thin(patch))
        if reason or len(local_paths) != 1:
            rejected.append(dict(reason='ambiguous_or_branched_visible_skeleton', pixel_count=len(component), min_yx=offset.tolist(), max_yx=pixel.max(0).tolist()))
            continue
        paths.extend((path + offset for path in local_paths))
    reason = None if paths else 'ambiguous_or_branched_visible_skeleton'
    return (paths, reason, rejected)

class StereoWireSegmentation(nn.Module):
    """Shared full-resolution image weights for left/right RGB, three logits each."""

    def __init__(self, width=16):
        super().__init__()
        self.width = width
        self.features = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.SiLU())
        self.head = nn.Conv2d(width, len(HEADS), 1)

    def forward(self, left, right):
        if left.ndim != 4 or left.shape != right.shape or left.shape[1] != 3 or (left.dtype != stere_wire_segme__torch.float32) or (right.dtype != stere_wire_segme__torch.float32) or (left.device != right.device) or (not stere_wire_segme__torch.isfinite(left).all()) or (not stere_wire_segme__torch.isfinite(right).all()) or (left < 0).any() or (left > 1).any() or (right < 0).any() or (right > 1).any():
            raise ValueError('synchronized finite float32 left/right RGB [B,3,H,W] required')
        output = self.head(self.features(stere_wire_segme__torch.cat((left, right), 0)))
        return dict(logits=stere_wire_segme__torch.stack(output.chunk(2), 1), head_names=HEADS, probability_calibrated=False, source='left_right_RGB_only')

def _uniform_indices(arc, count):
    """Unique observed pixels nearest uniform arc targets; never interpolate."""
    if count >= len(arc):
        return stere_wire_track__np.arange(len(arc))
    if count == 1:
        return stere_wire_track__np.array([len(arc) // 2])
    targets = stere_wire_track__np.linspace(0.0, arc[-1], count)
    right = stere_wire_track__np.searchsorted(arc, targets).clip(0, len(arc) - 1)
    left = stere_wire_track__np.maximum(0, right - 1)
    nearest = stere_wire_track__np.where(abs(arc[left] - targets) <= abs(arc[right] - targets), left, right)
    return stere_wire_track__np.unique(nearest)

def native_instance_targets(frame, reference):
    """Admit SHA-verified capture references after load_capture, never RGB labels.

    Native mask is re-derived from actual instance IDs and prim paths. Unknown
    IDs are not background negatives. Right supervision requires its own IDs.
    Occlusion supervision remains disabled without separate hidden-surface truth.
    """
    for key in ('frame_id', 'simulation_time', 'calibration_sha256'):
        if frame.get(key) != reference.get(key) or frame.get(key) is None:
            raise ValueError('native reference/frame identity mismatch: ' + key)
    sha = frame['calibration_sha256']
    if len(sha) != 64 or sha == '0' * 64 or reference.get('wire_mask_source_scope') != 'native_resolved_instance_id' or (reference.get('wire_instance_resolution', {}).get('wire_inventory_complete') is not True) or reference.get('wire_mask_blocked_reasons') or (reference.get('validation_only') is not True):
        raise ValueError('qualified native instance IDs required, not colour pseudo labels')
    shape = (len(frame['left_rgb']), 2, len(HEADS), *frame['left_rgb'].shape[-2:])
    target = stere_wire_segme__torch.zeros(shape, dtype=stere_wire_segme__torch.float32)
    valid = stere_wire_segme__torch.zeros(shape, dtype=stere_wire_segme__torch.bool)
    audit = dict(label_source='native_instance_id', occluded_wire_source='unavailable', generalization_qualified=False, right_instance_source='missing')
    for (side, prefix) in enumerate(('', 'right_')):
        (ids, info) = (reference.get(prefix + 'raw_instance_ids'), reference.get(prefix + 'raw_instance_info'))
        labels = info.get('idToLabels') if isinstance(info, dict) else None
        if ids is None and side == 1:
            continue
        if not isinstance(ids, stere_wire_segme__torch.Tensor) or ids.shape != target[:, side, 0].shape or ids.is_floating_point() or (not isinstance(labels, dict)) or (not labels):
            raise ValueError('native instance tensor and idToLabels required')
        if side == 1:
            resolution = reference.get('right_wire_instance_resolution', {})
            if resolution.get('wire_inventory_complete') is not True or resolution.get('blocked_reasons'):
                raise ValueError('right native instance source lacks independently resolved inventory')
        ids = ids.cpu()
        for (env, identity) in enumerate(frame['identity']):
            env_prefix = f"/World/envs/env_{identity['env_id']}/"
            (known, wire) = (stere_wire_segme__torch.zeros_like(ids[env], dtype=stere_wire_segme__torch.bool), stere_wire_segme__torch.zeros_like(ids[env], dtype=stere_wire_segme__torch.bool))
            for (raw_id, label) in labels.items():
                path = label if isinstance(label, str) else label.get('primPath', '')
                if not isinstance(path, str) or not (path.startswith('/') or path == 'BACKGROUND'):
                    continue
                hit = ids[env] == int(raw_id)
                known |= hit
                if path.startswith(env_prefix + 'Wire/') or path == env_prefix + 'Wire':
                    wire |= hit
            target[env, side, 0] = wire
            valid[env, side, 0] = known
            target[env, side, 2] = ~known
            valid[env, side, 2] = True
        if side == 0:
            provided = reference.get('wire_reference_mask')
            if provided is None or not stere_wire_segme__torch.equal(provided.cpu().bool(), target[:, side, 0].bool()):
                raise ValueError('native wire IDs disagree with reference mask')
        else:
            provided = reference.get('right_wire_reference_mask')
            if provided is None or not stere_wire_segme__torch.equal(provided.cpu().bool(), target[:, side, 0].bool()):
                raise ValueError('right native wire IDs disagree with reference mask')
            audit['right_instance_source'] = 'native_instance_id'
    return (target, valid, audit)

def _budgets(paths, capacity):
    if sum(map(len, paths)) <= capacity:
        return [len(path) for path in paths]
    sizes = stere_wire_track__np.array([len(path) for path in paths])
    lengths = stere_wire_track__np.array([_arc_pixels(path)[-1] for path in paths])
    allocation = stere_wire_track__np.zeros(len(paths), dtype=int)
    priority = stere_wire_track__np.argsort(-lengths, kind='stable')
    for index in priority:
        if allocation.sum() == capacity:
            break
        allocation[index] = min(2, capacity - allocation.sum())
    while allocation.sum() < capacity:
        score = stere_wire_track__np.where(allocation < sizes, lengths / stere_wire_track__np.maximum(allocation, 1), -1.0)
        allocation[int(score.argmax())] += 1
    return allocation.tolist()

def segmentation_loss(output, target, valid, *, mode='weighted_bce'):
    if mode not in ('weighted_bce', 'balanced_bce_dice'):
        raise ValueError('unsupported segmentation loss mode')
    logits = output['logits']
    if target.shape != logits.shape or valid.shape != target.shape or valid.dtype != stere_wire_segme__torch.bool:
        raise ValueError('per-view/head supervision masks required')
    if not valid.any() or not stere_wire_segme__torch.isfinite(target[valid]).all():
        raise ValueError('no valid finite supervised targets')
    safe = stere_wire_segme__torch.where(valid, target, 0.0)
    losses = []
    for view in range(2):
        for head in range(3):
            select = valid[:, view, head]
            if not select.any():
                continue
            y = safe[:, view, head][select]
            x = logits[:, view, head][select]
            positive_weight = ((1 - y).sum() / y.sum().clamp_min(1)).clamp(1, 100)
            if mode == 'weighted_bce':
                loss = F.binary_cross_entropy_with_logits(x, y, pos_weight=positive_weight)
            else:
                (positive, negative) = (y == 1, y == 0)
                terms = []
                if positive.any():
                    terms.append(F.softplus(-x[positive]).mean())
                if negative.any():
                    terms.append(F.softplus(x[negative]).mean())
                loss = stere_wire_segme__torch.stack(terms).mean()
                if positive.any():
                    probability = x.sigmoid()
                    loss = loss + 1.0 - (2 * (probability * y).sum() + 1) / (probability.sum() + y.sum() + 1)
            losses.append(loss)
    return (stere_wire_segme__torch.stack(losses).mean(), dict(supervised_pixels=valid.sum((0, 3, 4)).tolist(), probability_calibrated=False, loss_mode=mode, label_units='binary_native_instance_membership'))

def _unit(vector):
    norm = stere_wire_track__torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return stere_wire_track__torch.where(norm > 1e-10, vector / norm.clamp_min(1e-10), float('nan'))

class WireSourceEncoder(nn.Module):
    """Different learned branches for image centerline and observed surface.

    Neither source qualifies Cosserat material nodes. Empty/invalid slots have
    zero output and cannot silently become origin points.
    """

    def __init__(self, width=32):
        super().__init__()
        self.centerline = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.observed_surface = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, xyz, valid, provenance):
        if xyz.shape[:-1] != valid.shape or provenance.shape != valid.shape or valid.dtype != stere_wire_segme__torch.bool:
            raise ValueError('aligned XYZ, validity and measured point provenance required')
        if ((provenance < 0) | (provenance > 2)).any():
            raise ValueError('point provenance must be unknown/centerline/observed_surface')
        good = valid & stere_wire_segme__torch.isfinite(xyz).all(-1)
        (center, surface) = (good & (provenance == 1), good & (provenance == 2))
        value = stere_wire_segme__torch.where(good[..., None], xyz, 0.0)
        tokens = stere_wire_segme__torch.where(center[..., None], self.centerline(value), 0.0)
        tokens = tokens + stere_wire_segme__torch.where(surface[..., None], self.observed_surface(value), 0.0)
        return dict(tokens=tokens, centerline_valid=center, observed_surface_valid=surface, material_node_valid=stere_wire_segme__torch.zeros_like(valid), source_contract='image_centerline_or_measured_surface_v1')

def _descriptor(xyz, good):
    if len(xyz) < 2 or not bool(good.all()):
        return None
    arc = stere_wire_track__torch.cat((xyz.new_zeros(1), stere_wire_track__torch.diff(xyz, dim=0).norm(dim=-1).cumsum(0)))
    if not bool(stere_wire_track__torch.isfinite(arc[-1])) or float(arc[-1]) <= 1e-08:
        return None
    index = _uniform_indices(arc.cpu().numpy(), 17)
    curve = xyz[index].detach().cpu()
    delta = _unit(stere_wire_track__torch.diff(curve, dim=0))
    curvature = stere_wire_track__torch.diff(delta, dim=0).norm(dim=-1)
    return dict(endpoints=curve[[0, -1]], tangent=stere_wire_track__torch.stack((delta[0], delta[-1])), curve=curve, curvature=curvature, length=float(arc[-1]))

def _validate_head_capabilities(capabilities):
    from dh116.env.contracts import require_sha
    if capabilities is None:
        return {}
    allowed_heads = {view + '.' + head for view in VIEWS for head in HEADS}
    if not isinstance(capabilities, dict) or not capabilities.keys() <= allowed_heads:
        raise ValueError('per-view segmentation head capabilities required')
    for (key, record) in capabilities.items():
        if not isinstance(record, dict):
            raise ValueError('segmentation head capability record required: ' + key)
        for field in ('supervised', 'heldout_evaluated', 'runtime_veto_allowed'):
            if field in record and type(record[field]) is not bool:
                raise ValueError(f'{key}.{field} must be boolean')
        if record.get('heldout_evaluated') or 'heldout_evidence_sha256' in record:
            sha = require_sha(record.get('heldout_evidence_sha256'), key + ' held-out evidence')
            if sha == '0' * 64:
                raise ValueError(key + ' held-out evidence requires nonzero SHA256')
        if record.get('runtime_veto_allowed') and (not key.endswith('.unknown_instance') or record.get('semantics') != 'renderer_id_resolution' or record.get('supervised') is not True or (record.get('heldout_evaluated') is not True)):
            raise ValueError('unknown reliability veto requires explicit semantics, labels and held-out evidence')
    return capabilities

def _reverse_descriptor(descriptor):
    return dict(endpoints=descriptor['endpoints'].flip(0), tangent=-descriptor['tangent'].flip(0), curve=descriptor['curve'].flip(0), curvature=descriptor['curvature'].flip(0), length=descriptor['length'])

def segmentation_masks(output, *, qualified_views=(False, False), diagnostic_only=False, visible_threshold=0.5, unknown_threshold=0.5, mask_policy='visible_proposal', head_capabilities=None):
    """Build RGB-only proposals using evidence-qualified, versioned head decisions."""
    if mask_policy not in MASK_POLICIES:
        raise ValueError('unsupported segmentation mask_policy')
    if type(diagnostic_only) is not bool:
        raise ValueError('diagnostic_only must be boolean')
    if not isinstance(qualified_views, (list, tuple)) or len(qualified_views) != len(VIEWS) or any((type(value) is not bool for value in qualified_views)):
        raise ValueError('boolean qualification for both views required')
    if not 0 < visible_threshold < 1 or not 0 < unknown_threshold < 1:
        raise ValueError('finite probabilities required')
    capabilities = _validate_head_capabilities(head_capabilities)
    visible_qualified = []
    for (view, qualified) in zip(VIEWS, qualified_views):
        visible = capabilities.get(view + '.visible_wire', {})
        visible_qualified.append(qualified and visible.get('supervised') is True and (visible.get('heldout_evaluated') is True))
    if mask_policy == 'legacy_visible_and_unknown' and (not diagnostic_only):
        raise ValueError('legacy segmentation mask is restricted to diagnostic replay')
    if not diagnostic_only and (not all(visible_qualified)):
        raise ValueError('both visible segmentation heads must be qualified before deployment')
    unknown_veto = [mask_policy == 'legacy_visible_and_unknown'] * len(VIEWS)
    if mask_policy == 'qualified_reliability_filter':
        unknown_veto = [qualified and capabilities.get(view + '.unknown_instance', {}).get('runtime_veto_allowed') is True for (view, qualified) in zip(VIEWS, visible_qualified)]
        if not any(unknown_veto):
            raise ValueError('qualified reliability filter requires a qualified view and unknown head')
    probabilities = output['logits'].sigmoid()
    masks = tuple(((probabilities[:, index, 0] >= visible_threshold) & (probabilities[:, index, 2] < unknown_threshold if veto else True) for (index, veto) in enumerate(unknown_veto)))
    return (masks, dict(deployment_qualified=not diagnostic_only and all(visible_qualified), diagnostic_only=diagnostic_only, occlusion_head_used=False, probability_calibrated=False, mask_policy=mask_policy, mask_policy_version=MASK_POLICY_VERSION, unknown_veto_views=unknown_veto, visible_qualified_views=visible_qualified))

def _ribbon_pixels(paths, foreground, measured, radius):
    """Measured pixels reachable within a bounded walk on their RGB component."""
    (height, width) = measured.shape
    selected = []
    for (component, path) in enumerate(paths):
        seen = {tuple(pixel) for pixel in path}
        frontier = deque(((tuple(pixel), index, 0) for (index, pixel) in enumerate(path)))
        while frontier:
            ((y, x), index, distance) = frontier.popleft()
            if distance == radius:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    (yy, xx) = (int(y + dy), int(x + dx))
                    if 0 <= yy < height and 0 <= xx < width and foreground[yy, xx] and ((yy, xx) not in seen):
                        seen.add((yy, xx))
                        frontier.append(((yy, xx), index, distance + 1))
                        if measured[yy, xx]:
                            selected.append((component, index, yy, xx))
    return sorted(selected)

class SegmentedWireReconstruction(nn.Module):
    """Isolated learned RGB-mask -> stereo -> observed geometry/encoding path."""

    def __init__(self, segmenter, depth, tracker, width=32):
        super().__init__()
        (self.segmenter, self.depth, self.tracker) = (segmenter, depth, tracker)
        self.source_encoder = WireSourceEncoder(width)

    def forward(self, left, right, calibration, *, qualified_views=(False, False), diagnostic_only=False, timestamp=None, reset_mask=None, mask_policy='visible_proposal', head_capabilities=None):
        prediction = self.segmenter(left, right)
        (masks, admission) = segmentation_masks(prediction, qualified_views=qualified_views, diagnostic_only=diagnostic_only, mask_policy=mask_policy, head_capabilities=head_capabilities)
        depth = self.depth.compute(left, right, calibration, wire_masks=masks)
        points = self.tracker.extract(left, depth['depth_m'], depth['valid'], calibration.K_left, calibration.T_world_left, wire_mask=masks[0], point_confidence=depth['confidence'], timestamp=timestamp, reset_mask=reset_mask)
        encoding = self.source_encoder(points['points_world'], points['point_valid'], points['point_provenance'])
        return dict(segmentation=prediction, masks=masks, depth=depth, points=points, source_encoding=encoding, admission=admission)

class TrackedWirePointExtractor:
    """Sensor-only visible fragments; fixed-capacity tokens with explicit masks.

    timestamp is optional monotonic seconds (scalar or [B]). Without it, calls
    use a 30 Hz logical clock. Explicit resets, nonfinite/nonincreasing times,
    and gaps above max_timestamp_gap_s clear the environment's entire history.
    point_confidence, if supplied, is calibrated/uncalibrated matcher confidence
    [B,H,W] in [0,1]; it is copied only at valid observed depth pixels.
    """

    def __init__(self, *, point_count=128, identity_motion_limit_m=0.005, max_timestamp_gap_s=0.5, association_margin=0.15, point_sampling='centerline', ribbon_radius_px=1, observed_surface_capacity=None):
        if not isinstance(point_count, int) or not 2 <= point_count <= 128 or (not math.isfinite(identity_motion_limit_m)) or (identity_motion_limit_m <= 0) or (not math.isfinite(max_timestamp_gap_s)) or (max_timestamp_gap_s <= 0) or (not math.isfinite(association_margin)) or (association_margin <= 0):
            raise ValueError('2..128 samples and positive finite tracking limits required')
        if point_sampling not in ('centerline', 'observed_ribbon'):
            raise ValueError('wire point sampling must be centerline or observed_ribbon')
        if type(ribbon_radius_px) is not int or not 1 <= ribbon_radius_px <= 3 or (point_sampling == 'centerline' and ribbon_radius_px != 1):
            raise ValueError('ribbon radius must be 1..3 pixels in observed_ribbon mode')
        if observed_surface_capacity is None:
            observed_surface_capacity = point_count // 4 if point_sampling == 'observed_ribbon' else 0
        if type(observed_surface_capacity) is not int or observed_surface_capacity < 0 or observed_surface_capacity > point_count - 2 or (point_sampling == 'centerline' and observed_surface_capacity):
            raise ValueError('measured surface capacity must retain at least two centerline slots')
        self.observed_surface_capacity = observed_surface_capacity
        self.point_count = point_count
        self.point_sampling = point_sampling
        self.ribbon_radius_px = ribbon_radius_px
        self.identity_motion_limit_m = float(identity_motion_limit_m)
        self.max_timestamp_gap_s = float(max_timestamp_gap_s)
        self.association_margin = float(association_margin)
        self._history = None
        self._times = None
        self._next_id = 0
        self._image_history = None
        self._next_image_id = 0

    def _associate(self, descriptors, old, now):
        """Mutual unambiguous endpoint/tangent/curvature/arc/history matching."""
        matches = [(False, -1, None) for _ in descriptors]
        if not old:
            for (index, descriptor) in enumerate(descriptors):
                if descriptor is not None:
                    matches[index] = (False, self._next_id, False)
                    self._next_id += 1
            return matches
        candidates = []
        radius = self.identity_motion_limit_m
        for (current, descriptor) in enumerate(descriptors):
            if descriptor is None:
                continue
            for (prior, track) in enumerate(old):
                dt = now - track['time']
                prediction = track['endpoints'] + track['velocity'] * dt
                for reverse in (False, True):
                    oriented = _reverse_descriptor(descriptor) if reverse else descriptor
                    displacement = (oriented['endpoints'] - prediction).norm(dim=-1)
                    length_delta = abs(oriented['length'] - track['length'])
                    if float(displacement.max()) > radius or length_delta > max(radius, 0.2 * track['length']):
                        continue
                    tangent_cost = (1 - (oriented['tangent'] * track['tangent']).sum(-1)).mean()
                    if not oriented['curvature'].numel() and (not track['curvature'].numel()):
                        curvature_cost = stere_wire_track__torch.tensor(0.0)
                    elif oriented['curvature'].shape == track['curvature'].shape:
                        curvature_cost = (oriented['curvature'] - track['curvature']).square().mean().sqrt()
                    else:
                        curvature_cost = stere_wire_track__torch.tensor(1.0)
                    shape_cost = 0.0
                    if oriented['curve'].shape == track['curve'].shape:
                        translation = (track['velocity'] * dt).mean(0)
                        shape_cost = float((oriented['curve'] - track['curve'] - translation).norm(dim=-1).mean()) / radius
                    cost = float(displacement.mean()) / radius + 0.15 * float(tangent_cost) + 0.1 * float(curvature_cost) + 0.2 * length_delta / radius + 0.1 * shape_cost
                    if math.isfinite(cost):
                        candidates.append((cost, current, prior, reverse))
        for current in range(len(descriptors)):
            options = sorted((item for item in candidates if item[1] == current))
            if not options:
                continue
            best = options[0]
            if len(options) > 1 and options[1][0] - best[0] < self.association_margin:
                continue
            competitors = sorted((item for item in candidates if item[2] == best[2]))
            if competitors[0] != best:
                continue
            if len(competitors) > 1 and competitors[1][0] - best[0] < self.association_margin:
                continue
            matches[current] = (True, old[best[2]]['id'], best[3])
        return matches

    @stere_wire_track__torch.no_grad()
    def extract(self, left_rgb, depth_m, depth_valid, K_left, T_world_left, *, reset_mask=None, timestamp=None, point_confidence=None, wire_mask=None):
        if left_rgb.ndim != 4 or left_rgb.shape[1] != 3 or (not left_rgb.is_floating_point()) or (depth_m.shape != left_rgb.shape[:1] + left_rgb.shape[2:]) or (depth_valid.shape != depth_m.shape) or (depth_valid.dtype != stere_wire_track__torch.bool) or (not stere_wire_track__torch.isfinite(left_rgb).all()) or bool(((left_rgb < 0) | (left_rgb > 1)).any()):
            raise ValueError('finite RGB and matching stereo depth validity required')
        (n, device, dtype) = (len(left_rgb), depth_m.device, depth_m.dtype)
        if reset_mask is None:
            reset_mask = stere_wire_track__torch.zeros(n, dtype=stere_wire_track__torch.bool)
        if reset_mask.shape != (n,) or reset_mask.dtype != stere_wire_track__torch.bool:
            raise ValueError('boolean reset mask [B] required')
        if point_confidence is not None:
            point_confidence = stere_wire_track__torch.as_tensor(point_confidence, device=device, dtype=dtype)
            if point_confidence.shape != depth_m.shape or not stere_wire_track__torch.isfinite(point_confidence).all() or bool(((point_confidence < 0) | (point_confidence > 1)).any()):
                raise ValueError('point_confidence requires finite [B,H,W] in [0,1]')
        if self._history is None:
            (self._history, self._times) = ([[] for _ in range(n)], [None] * n)
            self._image_history = [[] for _ in range(n)]
        if len(self._history) != n:
            raise ValueError('point extractor batch size changed; construct a new instance')
        if timestamp is None:
            times = [0.0 if old is None else old + 1 / 30 for old in self._times]
        else:
            time_tensor = stere_wire_track__torch.as_tensor(timestamp, dtype=stere_wire_track__torch.float64)
            if time_tensor.ndim == 0:
                time_tensor = time_tensor.expand(n)
            if time_tensor.shape != (n,):
                raise ValueError('timestamp must be scalar or [B] seconds')
            times = time_tensor.tolist()
        candidate = wire_colour_mask(left_rgb) if wire_mask is None else wire_mask
        if candidate.shape != depth_m.shape or candidate.dtype != stere_wire_track__torch.bool or candidate.device != device:
            raise ValueError('RGB-derived wire_mask must be boolean aligned [B,H,W]')
        (image_paths, reasons, rejected_components) = ([], [], [])
        for mask in candidate.cpu().numpy():
            (paths, reason, rejected) = _paths_from_mask(mask)
            image_paths.append(paths)
            rejected_components.append(rejected)
            reasons.append(reason if reason else None if paths else 'missing_visible_wire')
        world = stere_wire_track__transform_points(stere_wire_track__backproject(depth_m, K_left, depth_valid), T_world_left)
        shape = (n, self.point_count)
        xyz = world.new_full((*shape, 3), float('nan'))
        valid = stere_wire_track__torch.zeros(shape, device=device, dtype=stere_wire_track__torch.bool)
        visibility = stere_wire_track__torch.zeros_like(valid)
        component_id = stere_wire_track__torch.full(shape, -1, device=device, dtype=stere_wire_track__torch.long)
        track_id = stere_wire_track__torch.full_like(component_id, -1)
        image_track_id = stere_wire_track__torch.full_like(component_id, -1)
        depth_segment_id = stere_wire_track__torch.full_like(component_id, -1)
        provenance = stere_wire_track__torch.zeros(shape, device=device, dtype=stere_wire_track__torch.uint8)
        pixels = stere_wire_track__torch.full((*shape, 2), -1, device=device, dtype=stere_wire_track__torch.long)
        arc_length = world.new_full(shape, float('nan'))
        arc_length_px = world.new_full(shape, float('nan'))
        tangent = stere_wire_track__torch.full_like(xyz, float('nan'))
        confidence = world.new_zeros(shape)
        topology_valid = stere_wire_track__torch.tensor([bool(p) and r is None for (p, r) in zip(image_paths, reasons)], device=device)
        temporal_valid = stere_wire_track__torch.zeros(n, device=device, dtype=stere_wire_track__torch.bool)
        anomaly = stere_wire_track__torch.zeros_like(temporal_valid)
        metric_complete = stere_wire_track__torch.zeros_like(temporal_valid)
        truncated = stere_wire_track__torch.zeros_like(temporal_valid)
        lengths = world.new_full((n,), float('nan'))
        component_count = stere_wire_track__torch.tensor([len(p) for p in image_paths], device=device, dtype=stere_wire_track__torch.long)
        rejected_count = stere_wire_track__torch.tensor([len(rows) for rows in rejected_components], device=device, dtype=stere_wire_track__torch.long)
        sampling_arc_space = [[] for _ in range(n)]
        surface_capacity = self.observed_surface_capacity
        centerline_capacity = self.point_count - surface_capacity
        for (env, paths) in enumerate(image_paths):
            (now, previous_time) = (times[env], self._times[env])
            bad_time = not math.isfinite(now) or (previous_time is not None and (now <= previous_time or now - previous_time > self.max_timestamp_gap_s))
            anomaly[env] = bad_time
            if bool(reset_mask[env]) or bad_time:
                self._history[env] = []
                self._image_history[env] = []
            if not math.isfinite(now):
                now = 0.0
            self._times[env] = now
            if not bool(topology_valid[env]):
                self._history[env] = []
                self._image_history[env] = []
                continue
            curves = []
            for path in paths:
                pixel = stere_wire_track__torch.as_tensor(path, device=device)
                curve = world[env, pixel[:, 0], pixel[:, 1]]
                good = stere_wire_track__torch.isfinite(curve).all(-1)
                curves.append((pixel, curve, good))
            image_descriptors = []
            for path in paths:
                p2 = stere_wire_track__torch.as_tensor(path, dtype=stere_wire_track__torch.float32)
                p3 = stere_wire_track__torch.cat((p2, stere_wire_track__torch.zeros(len(p2), 1)), -1) * (self.identity_motion_limit_m / 4.0)
                image_descriptors.append(_descriptor(p3, stere_wire_track__torch.ones(len(p3), dtype=stere_wire_track__torch.bool)))
            metric_counter = self._next_id
            self._next_id = self._next_image_id
            image_matches = self._associate(image_descriptors, self._image_history[env], now)
            (self._next_image_id, self._next_id) = (self._next_id, metric_counter)
            image_history = []
            for (descriptor, (_, identity, reverse)) in zip(image_descriptors, image_matches):
                if identity < 0 or descriptor is None:
                    continue
                oriented = _reverse_descriptor(descriptor) if reverse else descriptor
                prior = next((t for t in self._image_history[env] if t['id'] == identity), None)
                velocity = stere_wire_track__torch.zeros(2, 3) if prior is None else (oriented['endpoints'] - prior['endpoints']) / (now - prior['time'])
                image_history.append(dict(**oriented, id=identity, time=now, velocity=velocity))
            self._image_history[env] = image_history
            descriptors = [_descriptor(curve, good) for (_, curve, good) in curves]
            old = self._history[env]
            matches = self._associate(descriptors, old, now)
            budgets = _budgets(paths, centerline_capacity)
            truncated[env] = any((budget < len(path) for (budget, path) in zip(budgets, paths)))
            (history, cursor, next_segment) = ([], 0, 0)
            all_matched = bool(paths)
            lengths[env] = 0.0
            metric_complete[env] = True
            for (component, ((pixel, curve, good), descriptor, match, budget)) in enumerate(zip(curves, descriptors, matches, budgets)):
                (matched, identity, reverse) = match
                all_matched &= matched
                if reverse:
                    (pixel, curve, good) = (pixel.flip(0), curve.flip(0), good.flip(0))
                    descriptor = _reverse_descriptor(descriptor)
                if identity >= 0:
                    prior = next((track for track in old if track['id'] == identity), None)
                    velocity = stere_wire_track__torch.zeros(2, 3)
                    if prior is not None:
                        velocity = (descriptor['endpoints'] - prior['endpoints']) / (now - prior['time'])
                    history.append(dict(**descriptor, id=identity, time=now, velocity=velocity))
                metric_complete[env] &= bool(good.all())
                pixel_arc = _arc_pixels(pixel.cpu().numpy())
                metric_arc = curve.new_full((len(curve),), float('nan'))
                segments = stere_wire_track__torch.full((len(curve),), -1, device=device, dtype=stere_wire_track__torch.long)
                tangents = stere_wire_track__torch.full_like(curve, float('nan'))
                depth_flags = good.cpu().tolist()
                start = 0
                while start < len(curve):
                    if not depth_flags[start]:
                        start += 1
                        continue
                    end = start + 1
                    while end < len(curve) and depth_flags[end]:
                        end += 1
                    part = curve[start:end]
                    local_arc = stere_wire_track__torch.cat((part.new_zeros(1), stere_wire_track__torch.diff(part, dim=0).norm(dim=-1).cumsum(0)))
                    metric_arc[start:end] = local_arc
                    lengths[env] += local_arc[-1]
                    segments[start:end] = next_segment
                    next_segment += 1
                    if len(part) > 1:
                        deltas = stere_wire_track__torch.cat((part[1:2] - part[:1], part[2:] - part[:-2], part[-1:] - part[-2:-1]))
                        tangents[start:end] = _unit(deltas)
                    start = end
                complete = bool(good.all())
                sampling_arc_space[env].append('metric_3d' if complete else 'image_2d_partial_depth')
                sample_arc = metric_arc.cpu().numpy() if complete else pixel_arc
                indices = stere_wire_track__torch.as_tensor(_uniform_indices(sample_arc, budget), device=device) if budget else stere_wire_track__torch.empty(0, device=device, dtype=stere_wire_track__torch.long)
                count = len(indices)
                dest = slice(cursor, cursor + count)
                xyz[env, dest] = curve[indices]
                valid[env, dest] = good[indices]
                visibility[env, dest] = True
                provenance[env, dest] = 1
                component_id[env, dest] = component
                track_id[env, dest] = identity
                image_track_id[env, dest] = image_matches[component][1]
                pixels[env, dest] = pixel[indices]
                arc_length[env, dest] = metric_arc[indices]
                arc_length_px[env, dest] = stere_wire_track__torch.as_tensor(pixel_arc, device=device, dtype=dtype)[indices]
                depth_segment_id[env, dest] = segments[indices]
                tangent[env, dest] = tangents[indices]
                if point_confidence is None:
                    confidence[env, dest] = good[indices].to(dtype)
                else:
                    chosen = pixel[indices]
                    confidence[env, dest] = point_confidence[env, chosen[:, 0], chosen[:, 1]] * good[indices]
                cursor += count
            if surface_capacity:
                measured = (candidate[env] & stere_wire_track__torch.isfinite(world[env]).all(-1)).cpu().numpy()
                surface = _ribbon_pixels(paths, candidate[env].cpu().numpy(), measured, self.ribbon_radius_px)
                if surface:
                    indices = stere_wire_track__np.linspace(0, len(surface) - 1, min(surface_capacity, len(surface))).astype(int)
                    rows = stere_wire_track__torch.as_tensor([surface[i] for i in indices], device=device)
                    (y, x) = (rows[:, 2], rows[:, 3])
                    dest = slice(centerline_capacity, centerline_capacity + len(rows))
                    xyz[env, dest] = world[env, y, x]
                    valid[env, dest] = visibility[env, dest] = True
                    pixels[env, dest] = rows[:, 2:]
                    component_id[env, dest] = rows[:, 0]
                    provenance[env, dest] = 2
                    confidence[env, dest] = 1.0 if point_confidence is None else point_confidence[env, y, x]
                    truncated[env] |= len(surface) > surface_capacity
            self._history[env] = history
            temporal_valid[env] = all_matched
        return dict(points_world=xyz, point_valid=valid, xyz=xyz, valid=valid, component_id=component_id, depth_segment_id=depth_segment_id, arc_length=arc_length, arc_length_m=arc_length, arc_length_px=arc_length_px, tangent=tangent, confidence=confidence, visibility=visibility, track_id=track_id, image_track_id=image_track_id, point_pixels=pixels, point_provenance=provenance, topology_valid=topology_valid, temporal_identity_valid=temporal_valid, timestamp_anomaly=anomaly, visible_component_count=component_count, sampling_truncated=truncated, all_visible_components_simple=topology_valid & (rejected_count == 0), rejected_component_count=rejected_count, rejected_components=rejected_components, sampling_arc_space=sampling_arc_space, visible_arc_length_m=lengths, metric_arc_complete=metric_complete, full_cable_observed=stere_wire_track__torch.zeros_like(temporal_valid), rejection_reason=reasons, source='RGB_colour_visible_components_and_stereo_only' if wire_mask is None else 'explicit_RGB_segmentation_and_stereo_only', correspondence_scope='conservative_visible_component_sensor_tracks_not_material_point_identity', centerline_qualification=None)
stere_perce__TrackedWirePointExtractor = TrackedWirePointExtractor

def configured_segmentation_mask_policy(options):
    if not isinstance(options, dict) or options.get('diagnostic_only') is not True:
        raise ValueError('segmentation currently supports explicit diagnostic_only, not qualified deployment')
    policy = options.get('mask_policy')
    if policy not in MASK_POLICIES:
        raise ValueError('explicit supported segmentation mask_policy required')
    return policy

def configured_segmentation_depth(depth, config, device):
    """Shared opt-in diagnostic for native camera and independent RGB inference."""
    options = config.get('wire_segmentation')
    if options is None:
        return depth
    from dh116.rl.student_data import _validate_state
    from dh116.env.contracts import require_sha
    from dh116.physics.stereo_match import METHODS
    policy = configured_segmentation_mask_policy(options)
    if depth.method not in METHODS:
        raise ValueError('learned segmentation requires existing sparse wire matcher')
    expected = require_sha(options.get('checkpoint_sha256'), 'segmentation checkpoint SHA')
    data = Path(options['checkpoint_path']).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != expected or sha == '0' * 64:
        raise ValueError('segmentation checkpoint SHA mismatch')
    payload = stere_wire_segme__torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    contract = payload.get('contract', {}) if isinstance(payload, dict) else {}
    if not isinstance(contract, dict) or contract.get('version') != 'dh116-wire-segmentation-diagnostic-v1' or contract.get('source') != 'native_instance_id':
        raise ValueError('native-instance trained segmentation contract required')
    capabilities = _validate_head_capabilities(contract.get('head_capabilities'))
    qualified_views = contract.get('qualified_views', (False, False))
    with stere_wire_segme__torch.random.fork_rng(devices=[]):
        model = StereoWireSegmentation(payload['width'])
    _validate_state(payload['model'], model)
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval()

    class LearnedMaskDepth:

        def __getattr__(self, key):
            return getattr(depth, key)

        @stere_wire_segme__torch.no_grad()
        def compute(self, left, right, calibration, *, left_valid=None, right_valid=None):
            (masks, audit) = segmentation_masks(model(left, right), diagnostic_only=True, mask_policy=policy, qualified_views=qualified_views, head_capabilities=capabilities)
            result = depth.compute(left, right, calibration, wire_masks=masks, left_valid=left_valid, right_valid=right_valid)
            result.update(wire_mask=masks[0], wire_mask_right=masks[1], segmentation_identity=dict(checkpoint_sha256=sha, **audit))
            return result
    return LearnedMaskDepth()
