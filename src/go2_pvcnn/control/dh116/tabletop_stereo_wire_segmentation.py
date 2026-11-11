"""Opt-in RGB wire segmentation with independently qualified supervision heads.

Native instance IDs are training targets only. Occluded wire is not inferred
from an instance mask of visible surfaces. No default/old Actor is changed.
"""
import torch
from torch import nn
import torch.nn.functional as F


HEADS = ('visible_wire', 'occluded_wire', 'unknown_instance')


class StereoWireSegmentation(nn.Module):
    """Shared full-resolution image weights for left/right RGB, three logits each."""
    def __init__(self, width=16):
        super().__init__()
        self.width = width
        self.features = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.SiLU())
        self.head = nn.Conv2d(width, len(HEADS), 1)

    def forward(self, left, right):
        if (left.ndim != 4 or left.shape != right.shape or left.shape[1] != 3
                or left.dtype != torch.float32 or right.dtype != torch.float32
                or left.device != right.device or not torch.isfinite(left).all()
                or not torch.isfinite(right).all() or (left < 0).any() or (left > 1).any()
                or (right < 0).any() or (right > 1).any()):
            raise ValueError('synchronized finite float32 left/right RGB [B,3,H,W] required')
        output = self.head(self.features(torch.cat((left, right), 0)))
        return dict(logits=torch.stack(output.chunk(2), 1), head_names=HEADS,
                    probability_calibrated=False, source='left_right_RGB_only')


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
    if (len(sha) != 64 or sha == '0'*64 or reference.get('wire_mask_source_scope') != 'native_resolved_instance_id'
            or reference.get('wire_instance_resolution', {}).get('wire_inventory_complete') is not True
            or reference.get('wire_mask_blocked_reasons') or reference.get('validation_only') is not True):
        raise ValueError('qualified native instance IDs required, not colour pseudo labels')
    shape = (len(frame['left_rgb']), 2, len(HEADS), *frame['left_rgb'].shape[-2:])
    target = torch.zeros(shape, dtype=torch.float32)
    valid = torch.zeros(shape, dtype=torch.bool)
    audit = dict(label_source='native_instance_id', occluded_wire_source='unavailable',
                 generalization_qualified=False, right_instance_source='missing')
    for side, prefix in enumerate(('', 'right_')):
        ids, info = reference.get(prefix+'raw_instance_ids'), reference.get(prefix+'raw_instance_info')
        labels = info.get('idToLabels') if isinstance(info, dict) else None
        if ids is None and side == 1:
            continue
        if (not isinstance(ids, torch.Tensor) or ids.shape != target[:, side, 0].shape
                or ids.is_floating_point() or not isinstance(labels, dict) or not labels):
            raise ValueError('native instance tensor and idToLabels required')
        if side==1:
            resolution=reference.get('right_wire_instance_resolution',{})
            if resolution.get('wire_inventory_complete') is not True or resolution.get('blocked_reasons'):
                raise ValueError('right native instance source lacks independently resolved inventory')
        ids = ids.cpu()
        for env, identity in enumerate(frame['identity']):
            env_prefix = f"/World/envs/env_{identity['env_id']}/"
            known, wire = torch.zeros_like(ids[env], dtype=torch.bool), torch.zeros_like(ids[env], dtype=torch.bool)
            for raw_id, label in labels.items():
                path = label if isinstance(label, str) else label.get('primPath', '')
                # Resolved background is allowed; unknown integer 0 is not
                # implicitly a negative unless the renderer actually maps it.
                if not isinstance(path,str) or not (path.startswith('/') or path=='BACKGROUND'):
                    continue
                hit = ids[env] == int(raw_id)
                known |= hit
                if path.startswith(env_prefix+'Wire/') or path == env_prefix+'Wire':
                    wire |= hit
            target[env, side, 0] = wire
            valid[env, side, 0] = known
            target[env, side, 2] = ~known
            valid[env, side, 2] = True
        if side == 0:
            provided = reference.get('wire_reference_mask')
            if provided is None or not torch.equal(provided.cpu().bool(), target[:, side, 0].bool()):
                raise ValueError('native wire IDs disagree with reference mask')
        else:
            provided=reference.get('right_wire_reference_mask')
            if provided is None or not torch.equal(provided.cpu().bool(),target[:,side,0].bool()):
                raise ValueError('right native wire IDs disagree with reference mask')
            audit['right_instance_source'] = 'native_instance_id'
    return target, valid, audit


def segmentation_loss(output, target, valid, *, mode='weighted_bce'):
    if mode not in ('weighted_bce','balanced_bce_dice'):raise ValueError('unsupported segmentation loss mode')
    logits = output['logits']
    if target.shape != logits.shape or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('per-view/head supervision masks required')
    if not valid.any() or not torch.isfinite(target[valid]).all():
        raise ValueError('no valid finite supervised targets')
    safe = torch.where(valid, target, 0.)
    # Balance observed positive/negative labels per head; missing heads retain
    # exactly zero gradient, including the unavailable occlusion head.
    losses = []
    for view in range(2):
        for head in range(3):
            select = valid[:, view, head]
            if not select.any():
                continue
            y = safe[:, view, head][select]
            x = logits[:, view, head][select]
            positive_weight = ((1-y).sum()/y.sum().clamp_min(1)).clamp(1, 100)
            if mode=='weighted_bce':
                loss=F.binary_cross_entropy_with_logits(x, y, pos_weight=positive_weight)
            else:
                positive, negative=y==1,y==0
                terms=[]
                if positive.any():terms.append(F.softplus(-x[positive]).mean())
                if negative.any():terms.append(F.softplus(x[negative]).mean())
                loss=torch.stack(terms).mean()
                if positive.any():
                    probability=x.sigmoid()
                    loss=loss+1.-(2*(probability*y).sum()+1)/(probability.sum()+y.sum()+1)
            losses.append(loss)
    return torch.stack(losses).mean(), dict(supervised_pixels=valid.sum((0, 3, 4)).tolist(),
        probability_calibrated=False, loss_mode=mode, label_units='binary_native_instance_membership')


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
        if xyz.shape[:-1] != valid.shape or provenance.shape != valid.shape or valid.dtype != torch.bool:
            raise ValueError('aligned XYZ, validity and measured point provenance required')
        if ((provenance < 0) | (provenance > 2)).any():
            raise ValueError('point provenance must be unknown/centerline/observed_surface')
        good = valid & torch.isfinite(xyz).all(-1)
        center, surface = good & (provenance == 1), good & (provenance == 2)
        value = torch.where(good[..., None], xyz, 0.)
        tokens = torch.where(center[..., None], self.centerline(value), 0.)
        tokens = tokens + torch.where(surface[..., None], self.observed_surface(value), 0.)
        return dict(tokens=tokens, centerline_valid=center, observed_surface_valid=surface,
                    material_node_valid=torch.zeros_like(valid), source_contract='image_centerline_or_measured_surface_v1')


def segmentation_masks(output, *, qualified_views=(False, False), diagnostic_only=False,
                       visible_threshold=.5, unknown_threshold=.5):
    """Never silently deploy a left-only fitted model as a qualified stereo head."""
    if len(qualified_views) != 2 or (not all(qualified_views) and not diagnostic_only):
        raise ValueError('both segmentation views must be qualified before deployment')
    if not 0 < visible_threshold < 1 or not 0 < unknown_threshold < 1:
        raise ValueError('finite probabilities required')
    probabilities = output['logits'].sigmoid()
    visible = (probabilities[:, :, 0] >= visible_threshold) & (probabilities[:, :, 2] < unknown_threshold)
    # The unsupervised occlusion logit has no authority to fill missing depth.
    return (visible[:, 0], visible[:, 1]), dict(deployment_qualified=all(qualified_views),
        diagnostic_only=diagnostic_only, occlusion_head_used=False, probability_calibrated=False)


class SegmentedWireReconstruction(nn.Module):
    """Isolated learned RGB-mask -> stereo -> observed geometry/encoding path."""
    def __init__(self, segmenter, depth, tracker, width=32):
        super().__init__()
        self.segmenter, self.depth, self.tracker = segmenter, depth, tracker
        self.source_encoder = WireSourceEncoder(width)

    def forward(self, left, right, calibration, *, qualified_views=(False, False),
                diagnostic_only=False, timestamp=None, reset_mask=None):
        prediction = self.segmenter(left, right)
        masks, admission = segmentation_masks(prediction, qualified_views=qualified_views,
                                              diagnostic_only=diagnostic_only)
        depth = self.depth.compute(left, right, calibration, wire_masks=masks)
        points = self.tracker.extract(left, depth['depth_m'], depth['valid'],
            calibration.K_left, calibration.T_world_left, wire_mask=masks[0],
            point_confidence=depth['confidence'], timestamp=timestamp, reset_mask=reset_mask)
        encoding = self.source_encoder(points['points_world'], points['point_valid'], points['point_provenance'])
        return dict(segmentation=prediction, masks=masks, depth=depth, points=points,
                    source_encoding=encoding, admission=admission)


def configured_segmentation_depth(depth, config, device):
    """Used by native camera and independent RGB inference, explicit opt-in only."""
    options = config.get('wire_segmentation')
    if options is None:
        return depth
    from .tabletop_stereo_contract import file_sha256
    from .tabletop_stereo_wire_matching import METHODS
    if options.get('diagnostic_only') is not True:
        raise ValueError('segmentation currently supports explicit diagnostic_only, not qualified deployment')
    path = options['checkpoint_path']
    sha = file_sha256(path)
    if sha != options['checkpoint_sha256'] or sha == '0'*64:
        raise ValueError('segmentation checkpoint SHA mismatch')
    if depth.method not in METHODS:
        raise ValueError('learned segmentation requires existing sparse wire matcher')
    payload = torch.load(path, map_location=device, weights_only=False)
    contract = payload.get('contract', {})
    if (contract.get('version') != 'dh116-wire-segmentation-diagnostic-v1'
            or contract.get('source') != 'native_instance_id'):
        raise ValueError('native-instance trained segmentation contract required')
    model = StereoWireSegmentation(payload['width']).to(device).eval()
    model.load_state_dict(payload['model'], strict=True)

    class LearnedMaskDepth:
        def __getattr__(self, key):
            return getattr(depth, key)

        @torch.no_grad()
        def compute(self, left, right, calibration, **kwargs):
            masks, audit = segmentation_masks(model(left, right), diagnostic_only=True)
            result = depth.compute(left, right, calibration, wire_masks=masks, **kwargs)
            result.update(wire_mask=masks[0], wire_mask_right=masks[1],
                          segmentation_identity=dict(checkpoint_sha256=sha, **audit))
            return result
    return LearnedMaskDepth()
