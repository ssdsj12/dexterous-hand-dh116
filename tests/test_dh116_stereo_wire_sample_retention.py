import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_wire_correspondence import retain_wire_sample_slots
from go2_pvcnn.control.dh116.tabletop_stereo_wire_geometry import WirePointGeometry


def geometry(sources, pixels, valid, components):
    source = torch.tensor([sources], dtype=torch.uint8)
    valid = torch.tensor([valid])
    component = torch.tensor([components])
    missing = torch.full(source.shape, float('nan'))
    absent = torch.full(source.shape, -1, dtype=torch.long)
    center = source == 1
    xyz = torch.arange(source.numel() * 3, dtype=torch.float32).reshape(1, -1, 3)
    return WirePointGeometry(
        xyz=xyz.masked_fill(~valid[..., None], float('nan')), source_type=source,
        component_id=component, arc_length=missing.masked_fill(center & valid, .02),
        visibility=source != 0, confidence=valid.float() * .7, valid=valid,
        track_id=absent.masked_fill(center, 8), image_track_id=absent.masked_fill(center, 9),
        point_pixels=torch.tensor([pixels]), arc_length_px=missing.masked_fill(center, 2.),
        depth_segment_id=absent.masked_fill(center & valid, 0),
    )


def candidate_geometry():
    # A newly valid centerline pixel and an already-sampled surface duplicate
    # are excluded. Surface (2, 5) is on the same observed component, while
    # (4, 1) has no corresponding baseline component.
    return geometry([1, 1, 1, 2, 2, 2], [(2, 2), (2, 3), (2, 4), (2, 1), (2, 5), (4, 1)],
                    [True] * 6, [0, 0, 0, 0, 0, 1])


def test_reserve_original_measurements_and_2d_only_slots_before_new_surface():
    baseline = geometry([1, 1, 0, 2, 0, 0], [(2, 2), (2, 3), (-1, -1), (2, 1), (-1, -1), (-1, -1)],
                        [False, True, False, True, False, False], [0, 0, -1, 0, -1, -1])
    result, pool, proposed_from = retain_wire_sample_slots(baseline, candidate_geometry(), surface_start=3)
    assert pool.tolist() == [[False, False, False, False, True, False]]
    assert proposed_from.tolist() == [[-1, -1, -1, -1, 4, -1]]
    assert result.source_type.tolist() == [[1, 1, 0, 2, 2, 0]]
    assert torch.isnan(result.xyz[0, 0]).all()
    assert result.point_pixels[0, 4].tolist() == [2, 5]
    for name, value in baseline.as_dict().items():
        torch.testing.assert_close(result.as_dict()[name][baseline.visibility], value[baseline.visibility],
                                   rtol=0, atol=0, equal_nan=True)


def test_saturated_pool_keeps_new_candidate_separate_without_eviction():
    baseline = geometry([1, 1, 0, 2, 2, 2], [(2, 2), (2, 3), (-1, -1), (2, 1), (3, 1), (3, 2)],
                        [False, True, False, True, True, True], [0, 0, -1, 0, 0, 0])
    result, pool, proposed_from = retain_wire_sample_slots(baseline, candidate_geometry(), surface_start=3)
    assert pool.sum() == 1
    assert (proposed_from == -1).all()
    for name, value in baseline.as_dict().items():
        torch.testing.assert_close(result.as_dict()[name], value, rtol=0, atol=0, equal_nan=True)


def test_capacity_boundary_cannot_reclassify_baseline_surface_slot():
    baseline = geometry([1, 1, 0, 2, 0, 0], [(2, 2), (2, 3), (-1, -1), (2, 1), (-1, -1), (-1, -1)],
                        [False, True, False, True, False, False], [0, 0, -1, 0, -1, -1])
    with pytest.raises(ValueError, match='partition'):
        retain_wire_sample_slots(baseline, candidate_geometry(), surface_start=4)



def test_candidate_reference_label_records_unobserved_and_never_promotes():
    from scripts.audit_dh116_wire_correspondence import retention_labels

    candidate = candidate_geometry()
    pool = candidate.source_type == 2
    proposed = torch.tensor([[-1, -1, -1, 3, 4, 5]])
    measured = torch.zeros((1, 6, 8), dtype=torch.bool)
    measured[0, 2, 1] = measured[0, 2, 5] = True
    accurate = torch.zeros_like(measured)
    accurate[0, 2, 1] = True
    fallback = measured.clone()
    result, rows = retention_labels(candidate, pool, proposed, measured, accurate, fallback)
    assert result['candidate_reference_label_5mm'].tolist() == [[-1, -1, -1, 1, 0, -1]]
    assert rows[0]['candidate_pool_without_reference'] == 1
    assert rows[0]['proposed_slots'] == 3 and rows[0]['proposed_slots_within_5mm'] == 1
    assert rows[0]['promoted_slots'] == 0


def _audit_reconstruction(depth, accepted, reference_depth, reference_valid, wire, points):
    from types import SimpleNamespace

    from go2_pvcnn.control.dh116.tabletop_stereo_wire_diagnostics import audit_wire_frame

    frame = dict(left_rgb=torch.zeros(1, 3, *wire.shape[-2:]), identity=[dict(env_id=0)])
    prediction = dict(depth_m=depth, valid=accepted, diagnostics=dict(reasons={}))
    reference = dict(depth_m=reference_depth, valid=reference_valid, wire_reference_mask=wire,
                     wire_mask_source_scope='native_resolved_instance_id',
                     wire_instance_resolution=dict(wire_inventory_complete=True))
    calibration = SimpleNamespace(K_left=torch.tensor([[100., 0., 4.], [0., 100., 2.5], [0., 0., 1.]]),
                                  baseline_m=.08)
    return audit_wire_frame(frame, prediction, points, calibration, reference,
                            block_size=3, near=.2, far=1.5, wire_mask=wire)


def test_offline_reconstruction_distinguishes_dense_sparse_and_fixed_denominators():
    depth = torch.ones(1, 5, 8)
    depth[0, 2, 2] += .01
    wire = torch.zeros_like(depth, dtype=torch.bool)
    wire[0, 2, 1:7] = True
    accepted = wire.clone()
    accepted[0, 2, 6] = False
    reference_valid = wire.clone()
    reference_valid[0, 2, 5] = False
    points = dict(point_pixels=torch.tensor([[(2, 1), (2, 2), (2, 5), (-1, -1)]]),
                  point_valid=torch.tensor([[True, True, True, False]]),
                  point_provenance=torch.tensor([[2, 2, 2, 0]]))

    _, (row,) = _audit_reconstruction(depth, accepted, torch.ones_like(depth), reference_valid, wire, points)
    native = row['native_reference_metrics']
    metrics = native['reconstruction_evidence']
    assert metrics['dense']['precision_5mm'] == pytest.approx(3/4)
    assert metrics['dense']['reference_recall_5mm'] == pytest.approx(3/5)
    assert metrics['dense']['unscored_roi_pixels'] == 1
    assert metrics['sparse']['point_precision_5mm'] == pytest.approx(1/2)
    assert metrics['sparse']['scored_unique_pixels'] == 2
    assert metrics['sparse']['unscored_unique_pixels'] == 1
    assert metrics['sparse']['sampled_correct_pixel_contribution'] == pytest.approx(1/5)
    assert metrics['sparse']['contribution_upper_bound'] == pytest.approx(4/5)
    assert native['sampled_points']['correct_3d_point_completeness_5mm'] == pytest.approx(1/5)
    assert metrics['image_spatial_support']['sparse_scored_column_span_fraction'] == pytest.approx(1/5)
    assert metrics['image_spatial_support']['sparse_scored_row_span_fraction'] is None


def test_reference_mutation_changes_score_without_changing_retained_slots_or_sensor_maps():
    baseline = geometry([1, 1, 0, 2, 0, 0], [(2, 2), (2, 3), (-1, -1), (2, 1), (-1, -1), (-1, -1)],
                        [False, True, False, True, False, False], [0, 0, -1, 0, -1, -1])
    candidate = candidate_geometry()
    depth = torch.ones(1, 5, 8)
    wire = torch.zeros_like(depth, dtype=torch.bool)
    wire[0, 2, 1:6] = True
    first_ref = torch.ones_like(depth)
    changed_ref = first_ref.clone()
    changed_ref[0, 2, 5] += .1

    def run(reference):
        retained, pool, slots = retain_wire_sample_slots(baseline, candidate, surface_start=3)
        points = dict(point_pixels=retained.point_pixels, **retained.sensor_fields())
        maps, (row,) = _audit_reconstruction(depth, wire, reference, wire, wire, points)
        return retained, pool, slots, maps, row['native_reference_metrics']['reconstruction_evidence']

    kept, pool, slots, maps, metrics = run(first_ref)
    changed, changed_pool, changed_slots, changed_maps, changed_metrics = run(changed_ref)
    for name, values in kept.as_dict().items():
        torch.testing.assert_close(values, changed.as_dict()[name], rtol=0, atol=0, equal_nan=True)
    assert torch.equal(pool, changed_pool)
    assert torch.equal(slots, changed_slots)
    assert slots[0, 4] == 4
    for key in ('wire_mask', 'wire_left_depth_valid', 'wire_3d_point_valid', 'first_reject_code'):
        assert torch.equal(maps[key], changed_maps[key])
    assert metrics['sparse']['point_precision_5mm'] == 1
    assert changed_metrics['sparse']['point_precision_5mm'] == pytest.approx(2/3)
