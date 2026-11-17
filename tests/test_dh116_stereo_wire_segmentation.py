import pytest
import torch


def test_segmentation_real_rgb_heads_and_masked_native_supervision():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import StereoWireSegmentation, segmentation_loss
    torch.manual_seed(116)
    model = StereoWireSegmentation(width=8)
    rgb = torch.rand(2, 3, 24, 32)
    output = model(rgb, rgb.flip(-1))
    assert output['logits'].shape == (2, 2, 3, 24, 32)
    target = torch.zeros_like(output['logits'])
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[:, 0, 0] = True
    target[:, 0, 0, 10:12] = 1
    loss, audit = segmentation_loss(output, target, valid)
    loss.backward()
    assert model.head.weight.grad[0].abs().sum() > 0
    assert model.head.weight.grad[1:].count_nonzero() == 0
    assert audit['supervised_pixels'][1] == [0, 0, 0]
    assert not output['probability_calibrated']


def test_native_labels_require_ids_and_same_frame_not_colour_mask():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import native_instance_targets
    ids = torch.tensor([[[1, 2, 0], [2, 1, 999]]])
    identity = dict(frame_id=3, simulation_time=.05, calibration_sha256='a'*64)
    frame = dict(**identity, identity=[dict(env_id=7)], left_rgb=torch.zeros(1, 3, 2, 3))
    reference = dict(**identity, raw_instance_ids=ids,
        raw_instance_info={'idToLabels': {'1':'/World/envs/env_7/Wire/rod', '2':'/World/envs/env_7/rightHand/finger'}},
        wire_mask_source_scope='native_resolved_instance_id', validation_only=True,
        wire_instance_resolution={'wire_inventory_complete':True}, wire_mask_blocked_reasons=[],
        wire_reference_mask=ids == 1)
    target, valid, audit = native_instance_targets(frame, reference)
    assert target.shape == (1, 2, 3, 2, 3)
    assert not valid[:, 1].any() and not valid[:, :, 1].any()
    assert target[0, 0, 2, 1, 2] == 1  # unknown instance, not assumed background
    assert not valid[0, 0, 0, 1, 2]
    assert audit['right_instance_source'] == 'missing'
    with pytest.raises(ValueError, match='identity'):
        native_instance_targets(frame, dict(reference, frame_id=4))
    with pytest.raises(ValueError, match='native'):
        native_instance_targets(frame, dict(reference, wire_mask_source_scope='RGB_colour'))


def test_surface_point_encoder_never_assigns_material_identity():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import WireSourceEncoder
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
    from go2_pvcnn.control.dh116.tabletop_stereo_error_diagnostics import region_metrics
    target = torch.ones(1, 2, 3)
    pred = target + .001
    valid = torch.ones_like(target, dtype=torch.bool)
    accepted = valid.clone(); accepted[:, 1] = False
    result = region_metrics(pred, target, accepted, valid, valid, torch.eye(3))
    assert result['p99_m'] == pytest.approx(.001, abs=1e-6)
    assert result['point_error_3d_m']['p99'] is not None
    assert result['point_error_3d_rmse_m'] > result['rmse_m']
    assert result['prediction_coverage'] == .5


def test_tracking_2d_fragment_identity_survives_depth_holes_without_material_tracks():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tracking import TrackedWirePointExtractor
    rgb = torch.zeros(1, 3, 20, 30)
    mask = torch.zeros(1, 20, 30, dtype=torch.bool); mask[:, 10, 4:24] = True
    depth = torch.full((1, 20, 30), float('nan'))
    tracker = TrackedWirePointExtractor()
    first = tracker.extract(rgb, depth, torch.isfinite(depth), torch.eye(3), torch.eye(4), wire_mask=mask, timestamp=1.)
    second = tracker.extract(rgb, depth, torch.isfinite(depth), torch.eye(3), torch.eye(4), wire_mask=mask, timestamp=1.1)
    assert (second['image_track_id'][second['visibility']] >= 0).all()
    assert torch.equal(first['image_track_id'], second['image_track_id'])
    assert second['track_id'].max() == -1 and not second['point_valid'].any()
    reset = tracker.extract(rgb, depth, torch.isfinite(depth), torch.eye(3), torch.eye(4), wire_mask=mask, timestamp=1.2, reset_mask=torch.tensor([True]))
    assert not torch.equal(second['image_track_id'], reset['image_track_id'])


def test_depth_explicit_rgb_segmentation_masks_can_match_nonwarm_wire():
    from test_dh116_stereo_wire_depth import wire_scene
    from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_matching import wire_colour_mask
    left, right, calibration, disparity = wire_scene()
    masks = (wire_colour_mask(left), wire_colour_mask(right))
    left, right = left.flip(1), right.flip(1)  # blue instead of warm RGB
    model = StereoDepth(method='wire_census', min_disparity=5, max_disparity=9,
                        block_size=5, subpixel=False, uniqueness_threshold=.05)
    old = model.compute(left, right, calibration)
    new = model.compute(left, right, calibration, wire_masks=masks)
    assert not old['valid'].any()
    assert new['valid'][0, 15, 23] and new['disparity_px'][0, 15, 23] == disparity
    assert not new['valid'][0, 15, 26]  # right occluder remains missing
    with pytest.raises(ValueError, match='both'):
        model.compute(left, right, calibration, wire_masks=(masks[0], None))


def test_unqualified_segmentation_cannot_be_used_silently():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import StereoWireSegmentation, segmentation_masks
    rgb = torch.rand(1, 3, 20, 24)
    output = StereoWireSegmentation(8)(rgb, rgb)
    with pytest.raises(ValueError, match='qualified'):
        segmentation_masks(output, qualified_views=(True, False))
    masks, audit = segmentation_masks(output, qualified_views=(True, False), diagnostic_only=True)
    assert len(masks) == 2
    assert audit['deployment_qualified'] is False
    assert audit['occlusion_head_used'] is False


def test_source_branch_is_in_actual_actor_with_legacy_default_preserved():
    from test_dh116_stereo_runtime_palms import fixture
    from go2_pvcnn.control.dh116.tabletop_stereo_state_estimator import StereoStateEstimator
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import normalize_network_config, make_stereo_actor
    h = fixture().history(1)
    assert not h.point_provenance.any()  # legacy observations stay unknown
    h.point_provenance[:] = 1
    model = StereoStateEstimator(embed_dim=32, point_coordinate_mode='palms', point_source_encoding=True).eval()
    a = model(h)['latent']
    h.point_provenance[:] = 2
    b = model(h)['latent']
    assert not torch.allclose(a, b)
    b.square().sum().backward()
    assert model.palm_point_encoder.source_encoder.observed_surface[0].weight.grad.abs().sum() > 0
    conf = normalize_network_config(dict(embed_dim=32, point_coordinate_mode='palms', point_source_encoding=True))
    assert make_stereo_actor(conf).estimator.point_source_encoding is True
    from go2_pvcnn.control.dh116.tabletop_stereo_checkpoint import _source_sha
    assert 'tabletop_stereo_wire_segmentation.py' in _source_sha()


def test_runtime_segmentation_checkpoint_identity_and_opt_in(tmp_path):
    from test_dh116_stereo_wire_depth import wire_scene
    from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import StereoWireSegmentation, configured_segmentation_depth
    model = StereoWireSegmentation(8)
    path = tmp_path/'seg.pt'
    torch.save(dict(model=model.state_dict(), width=8, contract=dict(version='dh116-wire-segmentation-diagnostic-v1',
        qualified_views=[False,False], source='native_instance_id')), path)
    config = dict(wire_segmentation=dict(checkpoint_path=str(path), checkpoint_sha256=file_sha256(path), diagnostic_only=True, mask_policy='visible_proposal'))
    depth = StereoDepth(method='wire_census', min_disparity=5, max_disparity=9, block_size=5, subpixel=False)
    wrapper = configured_segmentation_depth(depth, config, 'cpu')
    left, right, calibration, _ = wire_scene()
    result = wrapper.compute(left, right, calibration)
    assert result['wire_mask'].shape == left[:, 0].shape
    assert result['segmentation_identity']['checkpoint_sha256'] == file_sha256(path)
    assert not result['segmentation_identity']['deployment_qualified']
    assert result['segmentation_identity']['mask_policy'] == 'visible_proposal'
    with pytest.raises(TypeError):
        wrapper.compute(left, right, calibration, reference_depth=torch.ones_like(left[:, 0]))
    old = dict(wire_segmentation={k: v for k, v in config['wire_segmentation'].items() if k != 'mask_policy'})
    with pytest.raises(ValueError, match='explicit.*mask_policy'):
        configured_segmentation_depth(depth, old, 'cpu')
    config['wire_segmentation']['diagnostic_only'] = False
    with pytest.raises(ValueError, match='diagnostic'):
        configured_segmentation_depth(depth, config, 'cpu')


def test_surface_capacity_uses_unique_measured_pixels_and_retains_missing_centerline():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tracking import TrackedWirePointExtractor
    rgb = torch.zeros(1, 3, 24, 120)
    mask = torch.zeros(1, 24, 120, dtype=torch.bool); mask[:, 10:15, 6:114] = True
    depth = torch.full((1, 24, 120), float('nan'))
    depth[:, 10, 6:114] = 1.
    tracker = TrackedWirePointExtractor(point_sampling='observed_ribbon', ribbon_radius_px=3,
                                       observed_surface_capacity=96)
    result = tracker.extract(rgb, depth, torch.isfinite(depth), torch.eye(3), torch.eye(4), wire_mask=mask)
    assert result['point_valid'].sum() == 96
    surface = result['point_provenance'] == 2
    center = result['point_provenance'] == 1
    assert center.sum() == 32 and not result['point_valid'][center].any()
    assert torch.isnan(result['xyz'][center]).all()
    assert torch.unique(result['point_pixels'][surface], dim=0).shape[0] == 96
    assert (result['point_pixels'][surface][:, 0] == 10).all()


def test_balanced_segmentation_loss_keeps_rare_positive_gradient():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import segmentation_loss
    logits = torch.zeros(1, 2, 3, 2, 200, requires_grad=True)
    target = torch.zeros_like(logits); target[0, 0, 0, 0, 0] = 1
    valid = torch.zeros_like(target, dtype=torch.bool); valid[0, 0, 0] = True
    loss, audit = segmentation_loss(dict(logits=logits), target, valid, mode='balanced_bce_dice')
    loss.backward()
    positive = logits.grad[0, 0, 0, 0, 0].abs()
    negative = logits.grad[0, 0, 0].abs().sum()-positive
    assert positive > .9*negative
    assert audit['loss_mode'] == 'balanced_bce_dice'
    assert logits.grad[:, 1:].count_nonzero() == 0


def test_native_reference_retains_right_instance_ids_and_separate_resolution(tmp_path):
    from test_dh116_stereo_resolution_benchmark import DepthReader, packet
    from go2_pvcnn.control.dh116.tabletop_stereo_reference import ValidationDepthReference, archive_instance_reference
    class Reader(DepthReader):
        def read(self):
            value = super().read()
            ids = torch.ones_like(value['depth_right_m'], dtype=torch.long)*7
            value.update(raw_instance_ids=ids*0, raw_instance_info={'idToLabels':{'0':'BACKGROUND'}},
                right_raw_instance_ids=ids, right_raw_instance_info={'idToLabels':{'7':'/World/envs/env_0/Wire/rod'}},
                right_wire_instance_resolution=dict(mask=ids==7, wire_inventory_complete=True, blocked_reasons=[]))
            return value
    ref = ValidationDepthReference(Reader()).read(packet())
    assert (ref['right_raw_instance_ids']==7).all()
    assert ref['right_wire_reference_mask'].all()
    assert ref['right_wire_instance_resolution']['wire_inventory_complete']
    binding = archive_instance_reference(ref, tmp_path/'instance_pair.pt')
    saved = torch.load(binding['path'], weights_only=False)
    assert torch.equal(saved['right_raw_instance_ids'], ref['right_raw_instance_ids'])
    assert saved['right_raw_instance_info'] == ref['right_raw_instance_info']


def test_visible_proposal_preserves_visible_pixels_when_unknown_is_high():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import segmentation_masks
    logits = torch.full((1, 2, 3, 1, 2), -10.)
    logits[:, :, 0, 0, 0] = 10.
    original, audit = segmentation_masks(dict(logits=logits), diagnostic_only=True)
    logits[:, :, 2] = 10.
    modified, _ = segmentation_masks(dict(logits=logits), diagnostic_only=True)
    assert all(torch.equal(before, after) for before, after in zip(original, modified))
    assert modified[0].tolist() == [[[True, False]]]
    assert audit['mask_policy'] == 'visible_proposal'
    assert audit['unknown_veto_views'] == [False, False]


def test_legacy_unknown_veto_is_an_explicit_diagnostic_replay():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import segmentation_masks
    logits = torch.full((1, 2, 3, 1, 2), 10.)
    output = dict(logits=logits)
    legacy, audit = segmentation_masks(output, diagnostic_only=True, mask_policy='legacy_visible_and_unknown')
    assert not any(mask.any() for mask in legacy)
    assert audit['unknown_veto_views'] == [True, True]
    assert not audit['deployment_qualified']
    with pytest.raises(ValueError, match='diagnostic'):
        segmentation_masks(output, qualified_views=(True, True), mask_policy='legacy_visible_and_unknown')


def test_reliability_veto_needs_evidence_for_the_individual_view_and_head():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import segmentation_masks
    output = dict(logits=torch.full((1, 2, 3, 1, 2), 10.))
    kwargs = dict(qualified_views=(True, True), diagnostic_only=True, mask_policy='qualified_reliability_filter')
    qualified = dict(supervised=True, heldout_evaluated=True, heldout_evidence_sha256='a'*64)
    unknown = dict(qualified, semantics='renderer_id_resolution', runtime_veto_allowed=True)
    capabilities = {
        'left.visible_wire': qualified, 'right.visible_wire': qualified,
        'left.unknown_instance': unknown,
        'right.unknown_instance': dict(unknown, heldout_evaluated=False, runtime_veto_allowed=False),
    }
    masks, audit = segmentation_masks(output, head_capabilities=capabilities, **kwargs)
    assert not masks[0].any() and masks[1].all()
    assert audit['unknown_veto_views'] == [True, False]
    assert not audit['deployment_qualified']
    with pytest.raises(ValueError, match='reliability'):
        segmentation_masks(output, **kwargs)
    invalid = dict(capabilities, **{'left.unknown_instance': dict(unknown, supervised='false')})
    with pytest.raises(ValueError, match='supervised'):
        segmentation_masks(output, head_capabilities=invalid, **kwargs)


def test_bilateral_deployment_requires_visible_head_evidence():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_segmentation import segmentation_masks
    output = dict(logits=torch.full((1, 2, 3, 1, 2), 10.))
    with pytest.raises(ValueError, match='qualified'):
        segmentation_masks(output, qualified_views=(True, True))
    qualified = dict(supervised=True, heldout_evaluated=True, heldout_evidence_sha256='a'*64)
    masks, audit = segmentation_masks(output, qualified_views=(True, True),
        head_capabilities={'left.visible_wire': qualified, 'right.visible_wire': qualified})
    assert all(mask.all() for mask in masks)
    assert audit['deployment_qualified']
    invalid = dict(qualified, heldout_evidence_sha256='0'*64)
    with pytest.raises(ValueError, match='nonzero'):
        segmentation_masks(output, qualified_views=(True, True),
            head_capabilities={'left.visible_wire': invalid, 'right.visible_wire': qualified})
