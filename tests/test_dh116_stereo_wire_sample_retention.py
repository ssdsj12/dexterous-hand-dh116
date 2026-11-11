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
