import math

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_observation import SensorBatch
from go2_pvcnn.control.dh116.tabletop_stereo_wire_geometry import (
    PointSource,
    WirePointGeometry,
    source_reference_metrics,
    source_temporal_metrics,
)
from go2_pvcnn.control.dh116.tabletop_stereo_wire_tracking import TrackedWirePointExtractor


def extract(extractor=None, *, timestamp=0.):
    if extractor is None:
        extractor = TrackedWirePointExtractor(point_sampling='observed_ribbon')
    rgb = torch.zeros(1, 3, 30, 44)
    rgb[:, :, 8:11, 4:20] = torch.tensor([.75, .48, .24])[None, :, None, None]
    rgb[:, :, 8:11, 24:40] = torch.tensor([.75, .48, .24])[None, :, None, None]
    depth = torch.full_like(rgb[:, 0], float('nan'))
    depth[:, 8:11, 4:20] = 1.
    depth[:, 8:11, 24:40] = 1.
    depth[:, 8:11, 10] = float('nan')  # measured depth gap, not a color gap
    matrix = torch.tensor([[80., 0., 22.], [0., 80., 15.], [0., 0., 1.]])
    result = extractor.extract(rgb, depth, torch.isfinite(depth), matrix, torch.eye(4),
                               timestamp=timestamp)
    return result, rgb, depth, matrix


def test_geometry_preserves_image_fragments_without_filling_depth():
    raw, _, _, _ = extract()
    geometry = WirePointGeometry.from_extraction(raw)
    center = geometry.source_type == PointSource.CENTERLINE
    surface = geometry.source_type == PointSource.OBSERVED_SURFACE
    unknown = geometry.source_type == PointSource.UNKNOWN
    assert center.any() and surface.any() and unknown.any()
    assert (geometry.visibility & center & ~geometry.valid).any()
    assert torch.isnan(geometry.xyz[center & ~geometry.valid]).all()
    assert geometry.component_id[center].unique().tolist() == [0, 1]
    assert (geometry.track_id[surface] == -1).all() and torch.isnan(geometry.arc_length[surface]).all()
    assert geometry.valid[surface].all() and (~geometry.visibility[unknown]).all()
    assert torch.equal(geometry.sensor_fields()['point_provenance'], raw['point_provenance'])


def test_typed_observation_accepts_sidecar_without_expanding_actor_schema():
    raw, rgb, depth, _ = extract()
    geometry = WirePointGeometry.from_extraction(raw)
    fields = geometry.sensor_fields()
    proprio = torch.zeros(1, 62)
    proprio[:, (27, 34)] = 1.
    observation = SensorBatch(
        left_rgb=rgb, right_rgb=rgb, depth_m=depth, depth_valid=depth.isfinite(),
        depth_confidence=depth.isfinite().float(), proprio=proprio, dt=torch.ones(1),
        image_age=torch.zeros(1), image_valid=torch.ones(1, 2, dtype=torch.bool), **fields)
    assert torch.equal(observation.point_provenance, geometry.source_type)
    assert 'track_id' not in observation.as_dict()


def test_native_reference_label_sidecar_keeps_each_source_and_clones_storage():
    from go2_pvcnn.control.dh116.tabletop_stereo_runtime import wire_reference_labels

    raw, _, _, _ = extract()
    labels = wire_reference_labels(raw, tracked=True)
    sidecar = labels['state_wire_geometry']
    assert torch.equal(sidecar['source_type'], raw['point_provenance'])
    assert torch.equal(sidecar['track_id'], labels['state_wire_track_ids'])
    assert torch.equal(sidecar['point_pixels'], labels['state_wire_point_pixels'])
    assert torch.isnan(sidecar['xyz'][sidecar['visibility'] & ~sidecar['valid']]).all()
    raw['point_provenance'].zero_()
    assert bool((sidecar['source_type'] == PointSource.OBSERVED_SURFACE).any())
    assert 'state_wire_geometry' not in wire_reference_labels(raw, tracked=False)


def test_source_scores_keep_fixed_native_denominator_and_do_not_double_count():
    raw, _, depth, matrix = extract()
    geometry = WirePointGeometry.from_extraction(raw)
    reference = torch.ones_like(depth)
    support = torch.zeros_like(depth, dtype=torch.bool)
    support[:, 8:11, 4:20] = True
    support[:, 8:11, 24:40] = True
    scores = source_reference_metrics(geometry, depth, depth.isfinite(), reference, support, matrix)
    assert set(scores) == {'centerline', 'observed_surface', 'unknown'}
    assert all(row['reference_valid_count'] == 96 for row in scores.values())
    assert scores['centerline']['correct_3d_points_5mm'] > 0
    assert scores['observed_surface']['correct_3d_points_5mm'] > 0
    assert scores['unknown']['intersection_count'] == 0
    for key in ('centerline', 'observed_surface'):
        assert math.isclose(scores[key]['mae_m'], 0.)
        assert scores[key]['intersection_count'] == int(
            (geometry.valid & (geometry.source_type == PointSource[key.upper()])).sum())


def test_surface_point_identity_is_never_inferred_from_component_association():
    tracker = TrackedWirePointExtractor(point_sampling='observed_ribbon')
    first = WirePointGeometry.from_extraction(extract(tracker)[0])
    second = WirePointGeometry.from_extraction(extract(tracker, timestamp=.04)[0])
    rows = source_temporal_metrics(second, previous=first, env=0)
    assert rows['centerline']['shared_image_component_tracks'] == 2
    assert rows['centerline']['shared_metric_component_tracks'] == 1  # other component has a depth hole
    assert rows['observed_surface']['shared_image_component_tracks'] == 2
    assert rows['observed_surface']['shared_metric_component_tracks'] is None
    assert rows['observed_surface']['material_point_identity_verified'] is False
    assert source_temporal_metrics(second, env=0)['centerline']['shared_image_component_tracks'] is None


@pytest.mark.parametrize('name,value', [('track_id', 23), ('arc_length', .1)])
def test_geometry_rejects_centerline_quantities_forged_for_surface(name, value):
    raw, _, _, _ = extract()
    bad = raw['point_provenance'] == PointSource.OBSERVED_SURFACE
    raw[name] = raw[name].clone()
    raw[name][bad] = value
    with pytest.raises(ValueError, match='observed surface'):
        WirePointGeometry.from_extraction(raw)
