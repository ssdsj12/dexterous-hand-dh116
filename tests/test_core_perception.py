from dh116.physics.diagnostics import SoftWireCorrespondence
from dh116.physics.diagnostics import colour_anchor_fallback
from dh116.physics.diagnostics import native_visibility
from dh116.physics.diagnostics import segmentation_evidence
from dh116.physics.diagnostics import score_native_correspondence
import pytest
import torch

def test_segmentation_real_rgb_heads_and_masked_native_supervision():
    from dh116.physics.wire import StereoWireSegmentation
    from dh116.physics.wire import segmentation_loss
    torch.manual_seed(116)
    model = StereoWireSegmentation(width=8)
    rgb = torch.rand(2, 3, 24, 32)
    output = model(rgb, rgb.flip(-1))
    assert output['logits'].shape == (2, 2, 3, 24, 32)
    target = torch.zeros_like(output['logits'])
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[:, 0, 0] = True
    target[:, 0, 0, 10:12] = 1
    (loss, audit) = segmentation_loss(output, target, valid)
    loss.backward()
    assert model.head.weight.grad[0].abs().sum() > 0
    assert model.head.weight.grad[1:].count_nonzero() == 0
    assert audit['supervised_pixels'][1] == [0, 0, 0]
    assert not output['probability_calibrated']

def test_native_labels_require_ids_and_same_frame_not_colour_mask():
    from dh116.physics.wire import native_instance_targets
    ids = torch.tensor([[[1, 2, 0], [2, 1, 999]]])
    identity = dict(frame_id=3, simulation_time=0.05, calibration_sha256='a' * 64)
    frame = dict(**identity, identity=[dict(env_id=7)], left_rgb=torch.zeros(1, 3, 2, 3))
    reference = dict(**identity, raw_instance_ids=ids, raw_instance_info={'idToLabels': {'1': '/World/envs/env_7/Wire/rod', '2': '/World/envs/env_7/rightHand/finger'}}, wire_mask_source_scope='native_resolved_instance_id', validation_only=True, wire_instance_resolution={'wire_inventory_complete': True}, wire_mask_blocked_reasons=[], wire_reference_mask=ids == 1)
    (target, valid, audit) = native_instance_targets(frame, reference)
    assert target.shape == (1, 2, 3, 2, 3)
    assert not valid[:, 1].any() and (not valid[:, :, 1].any())
    assert target[0, 0, 2, 1, 2] == 1
    assert not valid[0, 0, 0, 1, 2]
    assert audit['right_instance_source'] == 'missing'
    with pytest.raises(ValueError, match='identity'):
        native_instance_targets(frame, dict(reference, frame_id=4))
    with pytest.raises(ValueError, match='native'):
        native_instance_targets(frame, dict(reference, wire_mask_source_scope='RGB_colour'))

def test_surface_point_encoder_never_assigns_material_identity():
    from dh116.physics.wire import WireSourceEncoder
    model = WireSourceEncoder(8)
    xyz = torch.ones(1, 4, 3)
    valid = torch.tensor([[True, True, False, True]])
    provenance = torch.tensor([[1, 2, 0, 2]])
    output = model(xyz, valid, provenance)
    assert output['centerline_valid'].tolist() == [[True, False, False, False]]
    assert output['observed_surface_valid'].tolist() == [[False, True, False, True]]
    assert output['material_node_valid'].count_nonzero() == 0
    assert not torch.equal(output['tokens'][0, 0], output['tokens'][0, 1])

def test_metrics_include_p99_and_fixed_roi_denominator():
    from dh116.physics.diagnostics import region_metrics as region_metrics
    target = torch.ones(1, 2, 3)
    pred = target + 0.001
    valid = torch.ones_like(target, dtype=torch.bool)
    accepted = valid.clone()
    accepted[:, 1] = False
    result = region_metrics(pred, target, accepted, valid, valid, torch.eye(3))
    assert result['p99_m'] == pytest.approx(0.001, abs=1e-06)
    assert result['point_error_3d_m']['p99'] is not None
    assert result['point_error_3d_rmse_m'] > result['rmse_m']
    assert result['prediction_coverage'] == 0.5

def test_tracking_2d_fragment_identity_survives_depth_holes_without_material_tracks():
    from dh116.physics.wire import TrackedWirePointExtractor as TrackedWirePointExtractor
    rgb = torch.zeros(1, 3, 20, 30)
    mask = torch.zeros(1, 20, 30, dtype=torch.bool)
    mask[:, 10, 4:24] = True
    depth = torch.full((1, 20, 30), float('nan'))
    tracker = TrackedWirePointExtractor()
    first = tracker.extract(rgb, depth, torch.isfinite(depth), torch.eye(3), torch.eye(4), wire_mask=mask, timestamp=1.0)
    second = tracker.extract(rgb, depth, torch.isfinite(depth), torch.eye(3), torch.eye(4), wire_mask=mask, timestamp=1.1)
    assert (second['image_track_id'][second['visibility']] >= 0).all()
    assert torch.equal(first['image_track_id'], second['image_track_id'])
    assert second['track_id'].max() == -1 and (not second['point_valid'].any())
    reset = tracker.extract(rgb, depth, torch.isfinite(depth), torch.eye(3), torch.eye(4), wire_mask=mask, timestamp=1.2, reset_mask=torch.tensor([True]))
    assert not torch.equal(second['image_track_id'], reset['image_track_id'])

def test_depth_explicit_rgb_segmentation_masks_can_match_nonwarm_wire():
    from dh116.physics.vision import StereoDepth as StereoDepth
    from dh116.physics.stereo_match import wire_colour_mask
    (left, right, calibration, disparity) = wire_scene()
    masks = (wire_colour_mask(left), wire_colour_mask(right))
    (left, right) = (left.flip(1), right.flip(1))
    model = StereoDepth(method='wire_census', min_disparity=5, max_disparity=9, block_size=5, subpixel=False, uniqueness_threshold=0.05)
    old = model.compute(left, right, calibration)
    new = model.compute(left, right, calibration, wire_masks=masks)
    assert not old['valid'].any()
    assert new['valid'][0, 15, 23] and new['disparity_px'][0, 15, 23] == disparity
    assert not new['valid'][0, 15, 26]
    with pytest.raises(ValueError, match='both'):
        model.compute(left, right, calibration, wire_masks=(masks[0], None))

def test_unqualified_segmentation_cannot_be_used_silently():
    from dh116.physics.wire import StereoWireSegmentation
    from dh116.physics.wire import segmentation_masks
    rgb = torch.rand(1, 3, 20, 24)
    output = StereoWireSegmentation(8)(rgb, rgb)
    with pytest.raises(ValueError, match='qualified'):
        segmentation_masks(output, qualified_views=(True, False))
    (masks, audit) = segmentation_masks(output, qualified_views=(True, False), diagnostic_only=True)
    assert len(masks) == 2
    assert audit['deployment_qualified'] is False
    assert audit['occlusion_head_used'] is False

def test_runtime_segmentation_checkpoint_identity_and_opt_in(tmp_path):
    from dh116.physics.vision import StereoDepth as StereoDepth
    from dh116.env.contracts import stere_contr__file_sha256 as file_sha256
    from dh116.physics.wire import StereoWireSegmentation
    from dh116.physics.wire import configured_segmentation_depth
    model = StereoWireSegmentation(8)
    path = tmp_path / 'seg.pt'
    torch.save(dict(model=model.state_dict(), width=8, contract=dict(version='dh116-wire-segmentation-diagnostic-v1', qualified_views=[False, False], source='native_instance_id')), path)
    config = dict(wire_segmentation=dict(checkpoint_path=str(path), checkpoint_sha256=file_sha256(path), diagnostic_only=True, mask_policy='visible_proposal'))
    depth = StereoDepth(method='wire_census', min_disparity=5, max_disparity=9, block_size=5, subpixel=False)
    wrapper = configured_segmentation_depth(depth, config, 'cpu')
    (left, right, calibration, _) = wire_scene()
    result = wrapper.compute(left, right, calibration)
    assert result['wire_mask'].shape == left[:, 0].shape
    assert result['segmentation_identity']['checkpoint_sha256'] == file_sha256(path)
    assert not result['segmentation_identity']['deployment_qualified']
    assert result['segmentation_identity']['mask_policy'] == 'visible_proposal'
    with pytest.raises(TypeError):
        wrapper.compute(left, right, calibration, reference_depth=torch.ones_like(left[:, 0]))
    old = dict(wire_segmentation={k: v for (k, v) in config['wire_segmentation'].items() if k != 'mask_policy'})
    with pytest.raises(ValueError, match='explicit.*mask_policy'):
        configured_segmentation_depth(depth, old, 'cpu')
    config['wire_segmentation']['diagnostic_only'] = False
    with pytest.raises(ValueError, match='diagnostic'):
        configured_segmentation_depth(depth, config, 'cpu')

def test_surface_capacity_uses_unique_measured_pixels_and_retains_missing_centerline():
    from dh116.physics.wire import TrackedWirePointExtractor as TrackedWirePointExtractor
    rgb = torch.zeros(1, 3, 24, 120)
    mask = torch.zeros(1, 24, 120, dtype=torch.bool)
    mask[:, 10:15, 6:114] = True
    depth = torch.full((1, 24, 120), float('nan'))
    depth[:, 10, 6:114] = 1.0
    tracker = TrackedWirePointExtractor(point_sampling='observed_ribbon', ribbon_radius_px=3, observed_surface_capacity=96)
    result = tracker.extract(rgb, depth, torch.isfinite(depth), torch.eye(3), torch.eye(4), wire_mask=mask)
    assert result['point_valid'].sum() == 96
    surface = result['point_provenance'] == 2
    center = result['point_provenance'] == 1
    assert center.sum() == 32 and (not result['point_valid'][center].any())
    assert torch.isnan(result['xyz'][center]).all()
    assert torch.unique(result['point_pixels'][surface], dim=0).shape[0] == 96
    assert (result['point_pixels'][surface][:, 0] == 10).all()

def test_balanced_segmentation_loss_keeps_rare_positive_gradient():
    from dh116.physics.wire import segmentation_loss
    logits = torch.zeros(1, 2, 3, 2, 200, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 0, 0, 0] = 1
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[0, 0, 0] = True
    (loss, audit) = segmentation_loss(dict(logits=logits), target, valid, mode='balanced_bce_dice')
    loss.backward()
    positive = logits.grad[0, 0, 0, 0, 0].abs()
    negative = logits.grad[0, 0, 0].abs().sum() - positive
    assert positive > 0.9 * negative
    assert audit['loss_mode'] == 'balanced_bce_dice'
    assert logits.grad[:, 1:].count_nonzero() == 0

def test_visible_proposal_preserves_visible_pixels_when_unknown_is_high():
    from dh116.physics.wire import segmentation_masks
    logits = torch.full((1, 2, 3, 1, 2), -10.0)
    logits[:, :, 0, 0, 0] = 10.0
    (original, audit) = segmentation_masks(dict(logits=logits), diagnostic_only=True)
    logits[:, :, 2] = 10.0
    (modified, _) = segmentation_masks(dict(logits=logits), diagnostic_only=True)
    assert all((torch.equal(before, after) for (before, after) in zip(original, modified)))
    assert modified[0].tolist() == [[[True, False]]]
    assert audit['mask_policy'] == 'visible_proposal'
    assert audit['unknown_veto_views'] == [False, False]

def test_legacy_unknown_veto_is_an_explicit_diagnostic_replay():
    from dh116.physics.wire import segmentation_masks
    logits = torch.full((1, 2, 3, 1, 2), 10.0)
    output = dict(logits=logits)
    (legacy, audit) = segmentation_masks(output, diagnostic_only=True, mask_policy='legacy_visible_and_unknown')
    assert not any((mask.any() for mask in legacy))
    assert audit['unknown_veto_views'] == [True, True]
    assert not audit['deployment_qualified']
    with pytest.raises(ValueError, match='diagnostic'):
        segmentation_masks(output, qualified_views=(True, True), mask_policy='legacy_visible_and_unknown')

def test_reliability_veto_needs_evidence_for_the_individual_view_and_head():
    from dh116.physics.wire import segmentation_masks
    output = dict(logits=torch.full((1, 2, 3, 1, 2), 10.0))
    kwargs = dict(qualified_views=(True, True), diagnostic_only=True, mask_policy='qualified_reliability_filter')
    qualified = dict(supervised=True, heldout_evaluated=True, heldout_evidence_sha256='a' * 64)
    unknown = dict(qualified, semantics='renderer_id_resolution', runtime_veto_allowed=True)
    capabilities = {'left.visible_wire': qualified, 'right.visible_wire': qualified, 'left.unknown_instance': unknown, 'right.unknown_instance': dict(unknown, heldout_evaluated=False, runtime_veto_allowed=False)}
    (masks, audit) = segmentation_masks(output, head_capabilities=capabilities, **kwargs)
    assert not masks[0].any() and masks[1].all()
    assert audit['unknown_veto_views'] == [True, False]
    assert not audit['deployment_qualified']
    with pytest.raises(ValueError, match='reliability'):
        segmentation_masks(output, **kwargs)
    invalid = dict(capabilities, **{'left.unknown_instance': dict(unknown, supervised='false')})
    with pytest.raises(ValueError, match='supervised'):
        segmentation_masks(output, head_capabilities=invalid, **kwargs)

def test_bilateral_deployment_requires_visible_head_evidence():
    from dh116.physics.wire import segmentation_masks
    output = dict(logits=torch.full((1, 2, 3, 1, 2), 10.0))
    with pytest.raises(ValueError, match='qualified'):
        segmentation_masks(output, qualified_views=(True, True))
    qualified = dict(supervised=True, heldout_evaluated=True, heldout_evidence_sha256='a' * 64)
    (masks, audit) = segmentation_masks(output, qualified_views=(True, True), head_capabilities={'left.visible_wire': qualified, 'right.visible_wire': qualified})
    assert all((mask.all() for mask in masks))
    assert audit['deployment_qualified']
    invalid = dict(qualified, heldout_evidence_sha256='0' * 64)
    with pytest.raises(ValueError, match='nonzero'):
        segmentation_masks(output, qualified_views=(True, True), head_capabilities={'left.visible_wire': invalid, 'right.visible_wire': qualified})
import pytest
import torch
from dh116.physics.diagnostics import retain_wire_sample_slots
from dh116.physics.diagnostics import WirePointGeometry

def geometry(sources, pixels, valid, components):
    source = torch.tensor([sources], dtype=torch.uint8)
    valid = torch.tensor([valid])
    component = torch.tensor([components])
    missing = torch.full(source.shape, float('nan'))
    absent = torch.full(source.shape, -1, dtype=torch.long)
    center = source == 1
    xyz = torch.arange(source.numel() * 3, dtype=torch.float32).reshape(1, -1, 3)
    return WirePointGeometry(xyz=xyz.masked_fill(~valid[..., None], float('nan')), source_type=source, component_id=component, arc_length=missing.masked_fill(center & valid, 0.02), visibility=source != 0, confidence=valid.float() * 0.7, valid=valid, track_id=absent.masked_fill(center, 8), image_track_id=absent.masked_fill(center, 9), point_pixels=torch.tensor([pixels]), arc_length_px=missing.masked_fill(center, 2.0), depth_segment_id=absent.masked_fill(center & valid, 0))

def candidate_geometry():
    return geometry([1, 1, 1, 2, 2, 2], [(2, 2), (2, 3), (2, 4), (2, 1), (2, 5), (4, 1)], [True] * 6, [0, 0, 0, 0, 0, 1])

def test_reserve_original_measurements_and_2d_only_slots_before_new_surface():
    baseline = geometry([1, 1, 0, 2, 0, 0], [(2, 2), (2, 3), (-1, -1), (2, 1), (-1, -1), (-1, -1)], [False, True, False, True, False, False], [0, 0, -1, 0, -1, -1])
    (result, pool, proposed_from) = retain_wire_sample_slots(baseline, candidate_geometry(), surface_start=3)
    assert pool.tolist() == [[False, False, False, False, True, False]]
    assert proposed_from.tolist() == [[-1, -1, -1, -1, 4, -1]]
    assert result.source_type.tolist() == [[1, 1, 0, 2, 2, 0]]
    assert torch.isnan(result.xyz[0, 0]).all()
    assert result.point_pixels[0, 4].tolist() == [2, 5]
    for (name, value) in baseline.as_dict().items():
        torch.testing.assert_close(result.as_dict()[name][baseline.visibility], value[baseline.visibility], rtol=0, atol=0, equal_nan=True)

def test_saturated_pool_keeps_new_candidate_separate_without_eviction():
    baseline = geometry([1, 1, 0, 2, 2, 2], [(2, 2), (2, 3), (-1, -1), (2, 1), (3, 1), (3, 2)], [False, True, False, True, True, True], [0, 0, -1, 0, 0, 0])
    (result, pool, proposed_from) = retain_wire_sample_slots(baseline, candidate_geometry(), surface_start=3)
    assert pool.sum() == 1
    assert (proposed_from == -1).all()
    for (name, value) in baseline.as_dict().items():
        torch.testing.assert_close(result.as_dict()[name], value, rtol=0, atol=0, equal_nan=True)

def test_capacity_boundary_cannot_reclassify_baseline_surface_slot():
    baseline = geometry([1, 1, 0, 2, 0, 0], [(2, 2), (2, 3), (-1, -1), (2, 1), (-1, -1), (-1, -1)], [False, True, False, True, False, False], [0, 0, -1, 0, -1, -1])
    with pytest.raises(ValueError, match='partition'):
        retain_wire_sample_slots(baseline, candidate_geometry(), surface_start=4)

def _audit_reconstruction(depth, accepted, reference_depth, reference_valid, wire, points):
    from types import SimpleNamespace
    from dh116.physics.diagnostics import audit_wire_frame
    frame = dict(left_rgb=torch.zeros(1, 3, *wire.shape[-2:]), identity=[dict(env_id=0)])
    prediction = dict(depth_m=depth, valid=accepted, diagnostics=dict(reasons={}))
    reference = dict(depth_m=reference_depth, valid=reference_valid, wire_reference_mask=wire, wire_mask_source_scope='native_resolved_instance_id', wire_instance_resolution=dict(wire_inventory_complete=True))
    calibration = SimpleNamespace(K_left=torch.tensor([[100.0, 0.0, 4.0], [0.0, 100.0, 2.5], [0.0, 0.0, 1.0]]), baseline_m=0.08)
    return audit_wire_frame(frame, prediction, points, calibration, reference, block_size=3, near=0.2, far=1.5, wire_mask=wire)

def test_offline_reconstruction_distinguishes_dense_sparse_and_fixed_denominators():
    depth = torch.ones(1, 5, 8)
    depth[0, 2, 2] += 0.01
    wire = torch.zeros_like(depth, dtype=torch.bool)
    wire[0, 2, 1:7] = True
    accepted = wire.clone()
    accepted[0, 2, 6] = False
    reference_valid = wire.clone()
    reference_valid[0, 2, 5] = False
    points = dict(point_pixels=torch.tensor([[(2, 1), (2, 2), (2, 5), (-1, -1)]]), point_valid=torch.tensor([[True, True, True, False]]), point_provenance=torch.tensor([[2, 2, 2, 0]]))
    (_, (row,)) = _audit_reconstruction(depth, accepted, torch.ones_like(depth), reference_valid, wire, points)
    native = row['native_reference_metrics']
    metrics = native['reconstruction_evidence']
    assert metrics['dense']['precision_5mm'] == pytest.approx(3 / 4)
    assert metrics['dense']['reference_recall_5mm'] == pytest.approx(3 / 5)
    assert metrics['dense']['unscored_roi_pixels'] == 1
    assert metrics['sparse']['point_precision_5mm'] == pytest.approx(1 / 2)
    assert metrics['sparse']['scored_unique_pixels'] == 2
    assert metrics['sparse']['unscored_unique_pixels'] == 1
    assert metrics['sparse']['sampled_correct_pixel_contribution'] == pytest.approx(1 / 5)
    assert metrics['sparse']['contribution_upper_bound'] == pytest.approx(4 / 5)
    assert native['sampled_points']['correct_3d_point_completeness_5mm'] == pytest.approx(1 / 5)
    assert metrics['image_spatial_support']['sparse_scored_column_span_fraction'] == pytest.approx(1 / 5)
    assert metrics['image_spatial_support']['sparse_scored_row_span_fraction'] is None

def test_reference_mutation_changes_score_without_changing_retained_slots_or_sensor_maps():
    baseline = geometry([1, 1, 0, 2, 0, 0], [(2, 2), (2, 3), (-1, -1), (2, 1), (-1, -1), (-1, -1)], [False, True, False, True, False, False], [0, 0, -1, 0, -1, -1])
    candidate = candidate_geometry()
    depth = torch.ones(1, 5, 8)
    wire = torch.zeros_like(depth, dtype=torch.bool)
    wire[0, 2, 1:6] = True
    first_ref = torch.ones_like(depth)
    changed_ref = first_ref.clone()
    changed_ref[0, 2, 5] += 0.1

    def run(reference):
        (retained, pool, slots) = retain_wire_sample_slots(baseline, candidate, surface_start=3)
        points = dict(point_pixels=retained.point_pixels, **retained.sensor_fields())
        (maps, (row,)) = _audit_reconstruction(depth, wire, reference, wire, wire, points)
        return (retained, pool, slots, maps, row['native_reference_metrics']['reconstruction_evidence'])
    (kept, pool, slots, maps, metrics) = run(first_ref)
    (changed, changed_pool, changed_slots, changed_maps, changed_metrics) = run(changed_ref)
    for (name, values) in kept.as_dict().items():
        torch.testing.assert_close(values, changed.as_dict()[name], rtol=0, atol=0, equal_nan=True)
    assert torch.equal(pool, changed_pool)
    assert torch.equal(slots, changed_slots)
    assert slots[0, 4] == 4
    for key in ('wire_mask', 'wire_left_depth_valid', 'wire_3d_point_valid', 'first_reject_code'):
        assert torch.equal(maps[key], changed_maps[key])
    assert metrics['sparse']['point_precision_5mm'] == 1
    assert changed_metrics['sparse']['point_precision_5mm'] == pytest.approx(2 / 3)
import pytest
import torch
from dh116.physics.vision import StereoCalibration as StereoCalibration
from dh116.physics.vision import StereoDepth as StereoDepth

@pytest.fixture(scope='module', autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(min(old, 4))
    yield
    torch.set_num_threads(old)

def wire_scene():
    (h, w, disparity) = (31, 52, 7)
    rng = torch.Generator().manual_seed(116)
    left = torch.rand((1, 1, h, w), generator=rng).expand(-1, 3, -1, -1).clone() * 0.4
    right = torch.rand((1, 1, h, w), generator=rng).expand(-1, 3, -1, -1).clone() * 0.4
    scale = torch.rand((22,), generator=rng) * 0.4 + 0.5
    warm = torch.tensor([0.95, 0.6, 0.3])[:, None] * scale
    left[0, :, 15, 18:40] = warm
    right[0, :, 15, 18 - disparity:40 - disparity] = warm * 0.89
    right[0, :, 15, 26 - disparity] = 0.12
    K = [[80.0, 0.0, w / 2], [0.0, 80.0, h / 2], [0.0, 0.0, 1.0]]
    T = torch.eye(4, dtype=torch.float64)
    right_pose = T.clone()
    right_pose[0, 3] = 0.08
    calibration = StereoCalibration(K, K, T, right_pose, w, h, 0.08)
    return (left, right, calibration, disparity)

@pytest.mark.parametrize('method', ['wire_census', 'wire_zncc'])
def test_local_wire_matching_ignores_independent_background_and_leaves_occlusion_missing(method):
    (left, right, calibration, disparity) = wire_scene()
    result = StereoDepth(method=method, min_disparity=5, max_disparity=9, block_size=5, subpixel=False, uniqueness_threshold=0.05).compute(left, right, calibration)
    accepted = result['valid'][0]
    assert accepted[15, 23] and accepted[15, 33]
    assert result['disparity_px'][0, 15, 23] == disparity
    assert result['disparity_px'][0, 15, 33] == disparity
    assert not accepted[15, 26]
    assert not accepted[:15].any() and (not accepted[16:].any())
    assert torch.isnan(result['depth_m'][0][~accepted]).all()

def test_first_reject_attribution_preserves_overlap_and_accepted_mask():
    from dh116.physics.diagnostics import wire_intersections
    wire = torch.ones(1, 1, 4, dtype=torch.bool)
    stereo = torch.tensor([[[False, False, True, True]]])
    sampled = torch.tensor([[[False, False, False, True]]])
    reasons = dict(low_texture_left=torch.tensor([[[True, False, True, False]]]), left_right_inconsistent=torch.tensor([[[True, True, False, False]]]), depth_range=torch.tensor([[[True, False, False, False]]]))
    (maps, counts) = wire_intersections(wire, stereo, sampled, reasons)
    assert maps['first_reject_code'].tolist() == [[[1, 2, 4, 0]]]
    assert counts[0]['overlapping_rejections']['left_right_inconsistent'] == 2
    assert counts[0]['first_rejections']['low_texture_left'] == 1
    assert counts[0]['first_rejections']['not_sampled_as_3d_point'] == 1
    assert counts[0]['valid_point_count'] == 1

def test_flat_wire_silhouette_requires_unique_observed_shape_in_both_views():
    (left, right, calibration, disparity) = wire_scene()
    flat = torch.tensor([0.75, 0.48, 0.24])[:, None]
    left[0, :, 15, 18:40] = flat
    right[0, :, 15, 18 - disparity:40 - disparity] = flat * 0.89
    right[0, :, 15, 26 - disparity] = 0.12
    result = StereoDepth(method='wire_silhouette', min_disparity=4, max_disparity=10, block_size=5, subpixel=False, lr_tolerance_px=0.25).compute(left, right, calibration)
    assert result['disparity_px'][0, 15, 18] == disparity
    assert not result['valid'][0, 15, 26]
    assert not result['valid'][0, 15, 33]

def test_observed_ribbon_keeps_missing_centerline_and_actual_measurement_pixels():
    from dh116.physics.wire import TrackedWirePointExtractor as TrackedWirePointExtractor
    (left, _, calibration, _) = wire_scene()
    left[0, :, 14:17, 18:40] = torch.tensor([0.75, 0.48, 0.24])[:, None, None]
    depth = torch.full_like(left[:, 0], float('nan'))
    depth[0, 14, 20:23] = 0.8
    depth[0, 17, 21] = 0.6
    result = TrackedWirePointExtractor(point_sampling='observed_ribbon').extract(left, depth, torch.isfinite(depth), calibration.K_left, calibration.T_world_left)
    surface = result['point_provenance'][0] == 2
    center = result['point_provenance'][0] == 1
    assert center.any() and (not result['point_valid'][0, center].any())
    assert torch.isnan(result['points_world'][0, center]).all()
    assert result['point_valid'][0, surface].all()
    assert set(map(tuple, result['point_pixels'][0, surface].tolist())) == {(14, 20), (14, 21), (14, 22)}
    assert torch.isnan(result['arc_length_m'][0, surface]).all()
    assert (result['track_id'][0, surface] == -1).all()
    assert result['points_world'].shape == (1, 128, 3)

def test_ribbon_expansion_cannot_cross_an_unobserved_colour_gap():
    import numpy as np
    from dh116.physics.wire import _ribbon_pixels
    path = np.array([[2, 2], [2, 3]])
    foreground = np.zeros((8, 8), dtype=bool)
    foreground[2, 2:4] = True
    foreground[3:5, 2] = True
    foreground[2, 5] = True
    measured = np.zeros_like(foreground)
    measured[4, 2] = measured[2, 5] = True
    assert _ribbon_pixels([path], foreground, measured, radius=1) == []
    assert _ribbon_pixels([path], foreground, measured, radius=3) == [(0, 0, 4, 2)]

def test_integer_fallback_rejected_by_right_view_is_not_a_subpixel_rejection(monkeypatch):
    import dh116.physics.vision as tabletop_stereo_depth
    from dh116.physics.diagnostics import wire_intersections
    (left, right, calibration, disparity) = wire_scene()
    index = (0, 15, 23)

    def matcher(a, b, a_wire, b_wire, sign, **kwargs):
        candidate = torch.full_like(a, float('nan'))
        confidence = torch.zeros_like(a)
        good = torch.zeros_like(a_wire)
        used = torch.zeros_like(a_wire)
        if sign == -1:
            candidate[index] = disparity
            confidence[index] = 0.9
            good[index] = True
        return (candidate, confidence, good, dict(search_boundary=used, refinement_valid=used, optional_refinement_used=used))
    monkeypatch.setattr(tabletop_stereo_depth, 'match_wire', matcher)
    output = StereoDepth(method='wire_zncc', min_disparity=5, max_disparity=9, subpixel=True).compute(left, right, calibration)
    diagnostics = output['diagnostics']
    assert not output['valid'][index]
    assert diagnostics['reasons']['right_sample_unavailable'][index]
    assert not diagnostics['reasons']['subpixel_rejected'][index]
    assert not diagnostics['optional_refinement_used'][index]
    roi = torch.zeros_like(output['valid'])
    roi[index] = True
    (_, rows) = wire_intersections(roi, output['valid'], torch.zeros_like(roi), diagnostics['reasons'])
    assert rows[0]['first_rejections']['right_sample_unavailable'] == 1

def rgb_pair():
    (height, width, disparity) = (19, 46, 6)
    left = torch.rand((1, 3, height, width), generator=torch.Generator().manual_seed(116))
    right = torch.zeros_like(left)
    right[..., :-disparity] = left[..., disparity:]
    K = [[80.0, 0.0, width / 2], [0.0, 80.0, height / 2], [0.0, 0.0, 1.0]]
    pose = torch.eye(4, dtype=torch.float64)
    right_pose = pose.clone()
    right_pose[0, 3] = 0.06
    return (left, right, StereoCalibration(K, K, pose, right_pose, width, height, 0.06))

def test_empty_right_prediction_does_not_destroy_the_observed_rgb_match():
    (left, right, calibration) = rgb_pair()
    left_p = torch.zeros_like(left[:, 0])
    right_p = torch.zeros_like(left_p)
    left_p[0, 8, 20] = 0.8
    matcher = SoftWireCorrespondence(min_disparity=3, max_disparity=9, block_size=5)
    first = matcher.compute(left, right, calibration, probabilities=(left_p, right_p))
    assert first['valid'][0, 8, 20]
    assert first['disparity_px'][0, 8, 20] == pytest.approx(6.0, abs=0.2)
    assert torch.isnan(first['depth_m'][~first['valid']]).all()
    second = matcher.compute(left, torch.zeros_like(right), calibration, probabilities=(left_p, right_p))
    assert not second['valid'][0, 8, 20]

def test_explicit_proposal_does_not_erase_rgb_wire_when_head_abstains():
    (left, right, calibration) = rgb_pair()
    wire_rgb = torch.tensor([0.85, 0.55, 0.32])
    left[0, :, 8, 20] = wire_rgb
    right[0, :, 8, 14] = wire_rgb
    p = torch.ones_like(left[:, 0]) * 0.6
    prediction = torch.zeros_like(p, dtype=torch.bool)
    matcher = SoftWireCorrespondence(min_disparity=3, max_disparity=9)
    result = matcher.compute(left, right, calibration, probabilities=(p, p), left_proposal=prediction)
    assert result['candidate_mask'][0, 8, 20]
    assert result['valid'][0, 8, 20]
    assert not result['candidate_mask'].all()

def test_mask_probabilities_alone_do_not_match_textureless_rgb():
    (left, right, calibration) = rgb_pair()
    p = torch.zeros_like(left[:, 0])
    p[0, 8, 20] = 0.8
    matcher = SoftWireCorrespondence(min_disparity=3, max_disparity=9)
    result = matcher.compute(left * 0, right * 0, calibration, probabilities=(p, p.roll(-6, -1)))
    assert not result['valid'].any()

def test_visibility_requires_right_id_and_depth_and_keeps_unknown_separate():
    (left, _, calibration) = rgb_pair()
    z = torch.full_like(left[:, 0], 0.8)
    wire = torch.zeros_like(z, dtype=torch.bool)
    wire[0, 8, 20:23] = True
    right_z = z.clone()
    right_z[0, 8, 15] = 0.75
    right_z[0, 8, 16] = float('nan')
    right_wire = wire.roll(-6, -1)
    right_wire[0, 8, 15] = False
    without_id = native_visibility(z, right_z, wire, calibration)
    assert without_id['unknown'].sum() == 3
    assert not without_id['visible'].any()
    assert not without_id['occluded'].any()
    both = native_visibility(z, right_z, wire, calibration, right_wire=right_wire, right_known=torch.ones_like(right_wire))
    assert both['visible'][0, 8, 20]
    assert both['occluded'][0, 8, 21]
    assert both['unknown'][0, 8, 22]

def test_reference_score_keeps_unmatched_visible_pixels_in_coverage():
    (left, _, calibration) = rgb_pair()
    depth = torch.full_like(left[:, 0], 0.8)
    native = torch.zeros_like(depth, dtype=torch.bool)
    native[0, 8, 20:22] = True
    right_native = native.roll(-6, -1)
    labels = {'idToLabels': {'0': 'BACKGROUND', '1': '/World/envs/env_0/Wire/rod'}}
    identity = dict(frame_id=21, simulation_time=0.1, calibration_sha256=calibration.sha256)
    frame = dict(**identity, left_rgb=left, identity=[{'env_id': 0}])
    ref = dict(**identity, depth_m=depth, right_depth_m=depth, valid=torch.ones_like(native), wire_mask_source_scope='native_resolved_instance_id', validation_only=True, wire_instance_resolution={'wire_inventory_complete': True}, raw_instance_info=labels, raw_instance_ids=native.long(), wire_reference_mask=native, right_wire_instance_resolution={'wire_inventory_complete': True}, right_raw_instance_info={'idToLabels': {'0': 'BACKGROUND', '9': '/World/envs/env_0/Wire/rod'}}, right_raw_instance_ids=right_native.long() * 9, right_wire_reference_mask=right_native)
    detected = native.clone()
    detected[0, 8, 21] = False
    prediction = dict(disparity_px=torch.full_like(depth, 6.0).masked_fill(~detected, float('nan')), valid=detected)
    (result, _) = score_native_correspondence(frame, ref, calibration, prediction, (detected, right_native))
    assert result[0]['true_correspondence_accuracy'] == 1.0
    assert result[0]['true_correspondence_coverage'] == 0.5
    assert result[0]['true_correspondence_completeness'] == 0.5
    assert result[0]['segmentation_iou_left'] == 0.5
    assert result[0]['segmentation_iou_right'] == 1.0
    (separated, _) = score_native_correspondence(frame, ref, calibration, prediction, (detected, right_native), raw_visible_masks=(native, right_native))
    assert separated[0]['segmentation_view_metrics']['left'] == dict(native_wire_count=2, visible_iou=1.0, runtime_iou=0.5, visible_coverage=1.0, runtime_coverage=0.5, unknown_rejected_native_count=1, unknown_rejection_fraction=0.5)
    assert separated[0]['stereo_coverage'] == 0.5
    uncertain = dict(ref, right_valid=torch.ones_like(native))
    uncertain['right_valid'][0, 8, 15] = False
    shifted = dict(prediction, disparity_px=prediction['disparity_px'] - 0.5)
    (checked, _) = score_native_correspondence(frame, uncertain, calibration, shifted, (detected, right_native))
    assert checked[0]['true_correspondence_unresolved_predictions'] == 1
    assert checked[0]['true_correspondence_accuracy'] is None
    assert checked[0]['true_correspondence_completeness'] == 0.0
    ref.pop('right_raw_instance_ids')
    (result, _) = score_native_correspondence(frame, ref, calibration, prediction, (detected, right_native))
    assert result[0]['true_correspondence_accuracy'] is None
    assert result[0]['segmentation_iou_right'] is None
    assert result[0]['visibility_counts'] == dict(visible=0, occluded=0, unknown=2)

def test_probability_heads_and_runtime_unknown_rejection_are_separate():
    logits = torch.zeros(1, 2, 3, 1, 2)
    logits[:, :, 0] = 2
    logits[:, 0, 1] = 4
    logits[:, 0, 2, 0, 0] = -2
    logits[:, 0, 2, 0, 1] = 2
    evidence = segmentation_evidence(dict(logits=logits))
    assert evidence['left']['visible_mask'].tolist() == [[[True, True]]]
    assert evidence['left']['runtime_mask'].tolist() == [[[True, False]]]
    assert evidence['left']['unknown_rejection_mask'].tolist() == [[[False, True]]]
    assert evidence['left']['occluded_probability'].max() > 0.9
    assert evidence['right']['occluded_probability'].max() == 0.5

def test_fixed_trace_records_actual_both_directions_without_changing_depth():
    (left, right, calibration) = rgb_pair()
    p = torch.zeros_like(left[:, 0])
    p[0, 8, 20] = 0.9
    matcher = SoftWireCorrespondence(min_disparity=3, max_disparity=9, subpixel=True)
    clean = matcher.compute(left, right, calibration, probabilities=(p, p.roll(-6, -1)))
    traced = matcher.compute(left, right, calibration, probabilities=(p, p.roll(-6, -1)), trace_pixels=[(0, 8, 20), (0, 0, 0)])
    torch.testing.assert_close(traced['depth_m'], clean['depth_m'], equal_nan=True)
    rows = traced['diagnostics']['point_traces']
    row = rows[0, 8, 20]
    assert row['left']['integer_winner_px'] == 6
    assert row['left']['subpixel_offset_px'] == 0
    assert row['left']['winner_rgb_cost'] == pytest.approx(0, abs=1e-06)
    assert len(row['left']['integer_total_cost']) == 7
    assert row['right'][0]['integer_winner_px'] == 6
    assert row['lr_error_px'] == 0
    assert rows[0, 0, 0]['left'] is None

def test_colour_anchor_fallback_keeps_baseline_and_requires_both_rgb_views():
    roi = torch.zeros((1, 3, 16), dtype=torch.bool)
    roi[0, 1, 7:10] = True
    right_colour = roi.roll(-3, -1)
    right_colour[0, 1, 6] = False
    baseline = roi.clone()
    baseline[0, 1, 8:10] = False

    def prediction(valid, d, z):
        disparity = torch.full(valid.shape, d).masked_fill(~valid, float('nan'))
        return dict(valid=valid, disparity_px=disparity, depth_m=torch.full(valid.shape, z).masked_fill(~valid, float('nan')), confidence=valid.float(), lr_consistency=valid, diagnostics=dict(raw_disparity_px=disparity, reasons={}))
    fused = colour_anchor_fallback(prediction(baseline, 4.0, 0.6), prediction(roi, 3.0, 0.8), (roi, right_colour))
    assert fused['depth_m'][0, 1, 7] == pytest.approx(0.6)
    assert fused['depth_m'][0, 1, 8] == pytest.approx(0.8)
    assert not fused['valid'][0, 1, 9]
    assert torch.equal(fused['candidate_mask'], roi)
