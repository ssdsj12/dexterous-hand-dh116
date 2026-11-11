"""Typed measured wire geometry for existing RGB/stereo extraction and replay.

The sidecar is separate from the versioned Actor tensor contract. In particular,
no 2D fragment ID or observed surface point is a Cosserat material node.
"""
from dataclasses import dataclass, fields
from enum import IntEnum

import torch

from .tabletop_stereo_error_diagnostics import region_metrics


class PointSource(IntEnum):
    UNKNOWN = 0
    CENTERLINE = 1
    OBSERVED_SURFACE = 2


@dataclass(frozen=True)
class WirePointGeometry:
    xyz: torch.Tensor
    source_type: torch.Tensor
    component_id: torch.Tensor
    arc_length: torch.Tensor  # metres along each contiguous measured segment
    visibility: torch.Tensor  # visible in the left image, independent of depth
    confidence: torch.Tensor
    valid: torch.Tensor  # measured metric coordinates available
    track_id: torch.Tensor  # metric centerline component, not material point
    image_track_id: torch.Tensor  # centerline image component, not 3D
    point_pixels: torch.Tensor  # left image indices y, x
    arc_length_px: torch.Tensor  # along the 2D visible centerline
    depth_segment_id: torch.Tensor

    @classmethod
    def from_extraction(cls, extraction):
        required = {f.name for f in fields(cls)} - {'source_type'}
        if not required.issubset(extraction) or 'point_provenance' not in extraction:
            raise ValueError('tracked wire extraction with measured source metadata required')
        return cls(**{key: extraction[key] for key in required},
                   source_type=extraction['point_provenance'])

    def __post_init__(self):
        if not isinstance(self.valid, torch.Tensor) or self.valid.ndim != 2 or not self.valid.numel():
            raise ValueError('wire validity must be a nonempty [B,P] tensor')
        shape, device = self.valid.shape, self.valid.device
        dtypes = dict(xyz=None, source_type=torch.uint8, component_id=torch.long,
                      arc_length=None, visibility=torch.bool, confidence=None, valid=torch.bool,
                      track_id=torch.long, image_track_id=torch.long, point_pixels=torch.long,
                      arc_length_px=None, depth_segment_id=torch.long)
        for name, dtype in dtypes.items():
            value = getattr(self, name)
            expected = (*shape, 3) if name == 'xyz' else (*shape, 2) if name == 'point_pixels' else shape
            if (not isinstance(value, torch.Tensor) or value.shape != expected or value.device != device
                    or (not value.is_floating_point() if dtype is None else value.dtype != dtype)):
                raise ValueError(f'{name} has incompatible wire geometry shape, dtype or device')
        known = self.source_type != PointSource.UNKNOWN
        center = self.source_type == PointSource.CENTERLINE
        surface = self.source_type == PointSource.OBSERVED_SURFACE
        if (bool((self.source_type > PointSource.OBSERVED_SURFACE).any())
                or not torch.equal(known, self.visibility)
                or not torch.equal(known, self.component_id >= 0)
                or not torch.equal(known, (self.point_pixels >= 0).all(-1))):
            raise ValueError('known source must bind one visible image component and pixel')
        if (bool((self.valid & ~known).any())
                or not torch.equal(self.valid, torch.isfinite(self.xyz).all(-1))
                or not bool(torch.isnan(self.xyz[~self.valid]).all())):
            raise ValueError('unmeasured metric coordinates must be missing')
        if (not bool(torch.isfinite(self.confidence).all())
                or bool(((self.confidence < 0) | (self.confidence > 1)).any())
                or bool((self.confidence[~self.valid] != 0).any())):
            raise ValueError('wire confidence must be in [0,1] and zero without measured depth')
        if (bool((surface & ~self.valid).any()) or bool(((self.track_id != -1) & ~center).any())
                or bool(((self.image_track_id != -1) & ~center).any())
                or bool(((self.depth_segment_id != -1) & ~(center & self.valid)).any())
                or not bool(torch.isnan(self.arc_length[~(center & self.valid)]).all())
                or not bool(torch.isnan(self.arc_length_px[~center]).all())):
            raise ValueError('observed surface and unknown slots cannot carry centerline arc or track identity')
        if (not torch.equal(center & self.valid, self.depth_segment_id >= 0)
                or not bool((torch.isfinite(self.arc_length[center & self.valid])
                             & (self.arc_length[center & self.valid] >= 0)).all())
                or not bool((torch.isfinite(self.arc_length_px[center])
                             & (self.arc_length_px[center] >= 0)).all())):
            raise ValueError('centerline arc lengths require their measured segment support')

    def as_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def sensor_fields(self):
        """Supply only the geometry tensors already admitted by SensorBatch."""
        return dict(points_world=self.xyz.float(), point_valid=self.valid,
                    point_confidence=self.confidence.float(), point_provenance=self.source_type)


def source_reference_metrics(geometry, depth, depth_valid, reference_depth, reference_valid, K):
    """Score each source against the same fixed native support and 5 mm rule.

    Caller establishes native ID provenance and the reference range before scoring.
    No 3D centerline ground truth is inferred from the native optical-Z image.
    """
    if (depth.ndim != 3 or depth.shape[0] != geometry.valid.shape[0] or depth_valid.shape != depth.shape
            or depth_valid.dtype != torch.bool or depth_valid.device != depth.device
            or geometry.valid.device != depth.device):
        raise ValueError('aligned depth and wire geometry batch required')
    env, slot = geometry.valid.nonzero(as_tuple=True)
    y, x = geometry.point_pixels[env, slot].unbind(-1)
    if bool(((y < 0) | (x < 0) | (y >= depth.shape[1]) | (x >= depth.shape[2])).any()):
        raise ValueError('measured point pixel outside depth image')
    if not bool(depth_valid[env, y, x].all()):
        raise ValueError('measured geometry lacks corresponding valid stereo depth')
    flat_indices = (env * depth.shape[1] + y) * depth.shape[2] + x
    if flat_indices.unique().numel() != flat_indices.numel():
        raise ValueError('duplicated sampled depth pixel across sources')
    native = torch.ones_like(reference_valid)
    scores = {}
    for source in PointSource:
        selected = torch.zeros_like(depth_valid)
        keep = geometry.source_type[env, slot] == source
        selected[env[keep], y[keep], x[keep]] = True
        scores[source.name.lower()] = region_metrics(depth, reference_depth, selected, reference_valid, native, K)
    return scores


def source_temporal_metrics(current, *, previous=None, env):
    """Image/metric component retention in an externally verified same episode.

    For surface only, image retention means the associated RGB fragment has a
    tracked centerline component; it says nothing about a surface point's ID.
    """
    def component_tracks(geometry, source, field):
        selected = geometry.visibility[env] & (geometry.source_type[env] == source)
        components = geometry.component_id[env, selected].unique()
        centerline = geometry.source_type[env] == PointSource.CENTERLINE
        rows = centerline & torch.isin(geometry.component_id[env], components)
        ids = getattr(geometry, field)[env, rows]
        return set(ids[ids >= 0].tolist())

    out = {}
    for source in PointSource:
        selected = current.source_type[env] == source
        visible = int((current.visibility[env] & selected).sum())
        measured = int((current.valid[env] & selected).sum())
        row = dict(visible_slots=visible, metric_valid_slots=measured,
                   visible_slot_depth_coverage=measured / visible if visible else None,
                   material_point_identity_verified=False)
        for name, field in (('image', 'image_track_id'), ('metric', 'track_id')):
            applicable = source == PointSource.CENTERLINE or (
                name == 'image' and source == PointSource.OBSERVED_SURFACE)
            ids = component_tracks(current, source, field) if applicable else None
            prior = component_tracks(previous, source, field) if applicable and previous is not None else None
            row[f'{name}_component_tracks'] = len(ids) if ids is not None else None
            row[f'previous_{name}_component_tracks'] = len(prior) if prior is not None else None
            row[f'shared_{name}_component_tracks'] = len(ids & prior) if prior is not None else None
        out[source.name.lower()] = row
    return out
