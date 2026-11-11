import json
from pathlib import Path
import runpy

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_wire_correspondence import (
    SoftWireCorrespondence,
    native_visibility,
    score_native_correspondence,
)


def rgb_pair():
    height, width, disparity = 19, 46, 6
    left = torch.rand((1, 3, height, width), generator=torch.Generator().manual_seed(116))
    right = torch.zeros_like(left)
    right[..., :-disparity] = left[..., disparity:]
    K = [[80., 0., width / 2], [0., 80., height / 2], [0., 0., 1.]]
    pose = torch.eye(4, dtype=torch.float64)
    right_pose = pose.clone()
    right_pose[0, 3] = .06
    return left, right, StereoCalibration(K, K, pose, right_pose, width, height, .06)


def test_empty_right_prediction_does_not_destroy_the_observed_rgb_match():
    left, right, calibration = rgb_pair()
    left_p = torch.zeros_like(left[:, 0])
    right_p = torch.zeros_like(left_p)
    left_p[0, 8, 20] = .8
    matcher = SoftWireCorrespondence(min_disparity=3, max_disparity=9, block_size=5)
    first = matcher.compute(left, right, calibration, probabilities=(left_p, right_p))
    assert first['valid'][0, 8, 20]
    assert first['disparity_px'][0, 8, 20] == pytest.approx(6., abs=.2)
    assert torch.isnan(first['depth_m'][~first['valid']]).all()
    second = matcher.compute(left, torch.zeros_like(right), calibration, probabilities=(left_p, right_p))
    assert not second['valid'][0, 8, 20]


def test_explicit_proposal_does_not_erase_rgb_wire_when_head_abstains():
    left, right, calibration = rgb_pair()
    wire_rgb = torch.tensor([.85, .55, .32])
    left[0, :, 8, 20] = wire_rgb
    right[0, :, 8, 14] = wire_rgb
    p = torch.ones_like(left[:, 0]) * .6
    prediction = torch.zeros_like(p, dtype=torch.bool)
    matcher = SoftWireCorrespondence(min_disparity=3, max_disparity=9)
    result = matcher.compute(left, right, calibration, probabilities=(p, p), left_proposal=prediction)
    assert result['candidate_mask'][0, 8, 20]
    assert result['valid'][0, 8, 20]
    assert not result['candidate_mask'].all()


def test_mask_probabilities_alone_do_not_match_textureless_rgb():
    left, right, calibration = rgb_pair()
    p = torch.zeros_like(left[:, 0])
    p[0, 8, 20] = .8
    matcher = SoftWireCorrespondence(min_disparity=3, max_disparity=9)
    result = matcher.compute(left * 0, right * 0, calibration, probabilities=(p, p.roll(-6, -1)))
    assert not result['valid'].any()


def test_visibility_requires_right_id_and_depth_and_keeps_unknown_separate():
    left, _, calibration = rgb_pair()
    z = torch.full_like(left[:, 0], .8)
    wire = torch.zeros_like(z, dtype=torch.bool)
    wire[0, 8, 20:23] = True
    right_z = z.clone()
    right_z[0, 8, 15] = .75
    right_z[0, 8, 16] = float('nan')
    right_wire = wire.roll(-6, -1)
    right_wire[0, 8, 15] = False
    without_id = native_visibility(z, right_z, wire, calibration)
    assert without_id['unknown'].sum() == 3
    assert not without_id['visible'].any()
    assert not without_id['occluded'].any()
    both = native_visibility(z, right_z, wire, calibration, right_wire=right_wire,
                             right_known=torch.ones_like(right_wire))
    assert both['visible'][0, 8, 20]
    assert both['occluded'][0, 8, 21]
    assert both['unknown'][0, 8, 22]


def test_reference_score_keeps_unmatched_visible_pixels_in_coverage():
    left, _, calibration = rgb_pair()
    depth = torch.full_like(left[:, 0], .8)
    native = torch.zeros_like(depth, dtype=torch.bool)
    native[0, 8, 20:22] = True
    right_native = native.roll(-6, -1)
    labels = {'idToLabels': {'0': 'BACKGROUND', '1': '/World/envs/env_0/Wire/rod'}}
    identity = dict(frame_id=21, simulation_time=.1, calibration_sha256=calibration.sha256)
    frame = dict(**identity, left_rgb=left, identity=[{'env_id': 0}])
    ref = dict(**identity, depth_m=depth, right_depth_m=depth, valid=torch.ones_like(native),
        wire_mask_source_scope='native_resolved_instance_id', validation_only=True,
        wire_instance_resolution={'wire_inventory_complete': True}, raw_instance_info=labels,
        raw_instance_ids=native.long(), wire_reference_mask=native,
        right_wire_instance_resolution={'wire_inventory_complete': True}, right_raw_instance_info=labels,
        right_raw_instance_ids=right_native.long(), right_wire_reference_mask=right_native)
    detected = native.clone()
    detected[0, 8, 21] = False
    prediction = dict(disparity_px=torch.full_like(depth, 6.).masked_fill(~detected, float('nan')),
                      valid=detected)
    result, _ = score_native_correspondence(frame, ref, calibration, prediction, (detected, right_native))
    assert result[0]['true_correspondence_accuracy'] == 1.
    assert result[0]['true_correspondence_coverage'] == .5
    assert result[0]['true_correspondence_completeness'] == .5
    assert result[0]['segmentation_iou_left'] == .5
    ref.pop('right_raw_instance_ids')
    result, _ = score_native_correspondence(frame, ref, calibration, prediction, (detected, right_native))
    assert result[0]['true_correspondence_accuracy'] is None
    assert result[0]['segmentation_iou_right'] is None
    assert result[0]['visibility_counts'] == dict(visible=0, occluded=0, unknown=2)


def test_fixed_tail_fixture_refuses_changed_cohort_or_missing_frame():
    repo = Path(__file__).parents[1]
    fixture = repo / 'configs/dh116_training/stereo_wire_correspondence_roll20_tail_v1.json'
    saved = json.loads(fixture.read_text(encoding='utf-8'))
    functions = runpy.run_path(str(repo / 'scripts/audit_dh116_wire_correspondence.py'))
    calibration = type('CalibrationIdentity', (), {'sha256': saved['calibration_sha256']})()
    load = functions['load_tail_fixture']
    assert len(load(fixture, saved['capture_sha256'], calibration, [21, 22, 23, 24])) == 12
    with pytest.raises(ValueError, match='immutable cohort'):
        load(fixture, 'bad capture SHA', calibration, [21, 22, 23, 24])
    with pytest.raises(ValueError, match='immutable cohort'):
        load(fixture, saved['capture_sha256'], calibration, [21, 22, 23])
