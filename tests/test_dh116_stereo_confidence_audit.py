import json

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_confidence_audit import audit_confidence, load_replay_rows
from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256


def sample(index, initial, family, *, confidence=.7, label=1, right=True, calibration='c'):
    return dict(sample_id=f'sample-{index}', identity=dict(
        physical_initial_condition_sha256=initial*64, source_episode_group=family,
        calibration_sha256=calibration*64, grasp_style='pinch_thumb_index'),
        checkpoint_sha256='e'*64, depth_source='rgb_soft_fallback', confidence=confidence, reference_label_5mm=label,
        right_evidence=right)


def test_replayed_initials_and_transitive_families_do_not_leak():
    rows = [sample(0, 'a', 'first'), sample(1, 'a', 'second', calibration='d'),
            sample(2, 'b', 'second'), sample(3, 'b', 'third')]
    result = audit_confidence(rows)
    assert result['independent_groups'] == 1
    assert result['group_by_sample_id'] == {row['sample_id']: 0 for row in rows}
    assert all(x['heldout'] is None for x in result['strata'])


def test_missing_right_and_na_labels_are_not_negative_examples():
    rows = [sample(i, initial, initial, label=None if i == 2 else 1, right=False)
            for i, initial in enumerate('abc')]
    result = audit_confidence(rows)
    row, = result['strata']
    assert row['candidate_samples'] == 3 and row['unresolved_reference_samples'] == 1
    assert row['raw_left_reference_diagnostic']['samples'] == 2
    assert row['raw_left_reference_diagnostic']['observed_accuracy'] == 1.
    assert row['right_verified_reference_samples'] == 0
    assert row['heldout_status'] == 'blocked_missing_grouped_right_reference'
    assert not result['deployable']


def test_histogram_fit_uses_train_only_and_reports_unseen_bin_na():
    rows = [sample(i, initial, initial, label=i % 2) for i, initial in enumerate('abcdef')]
    first = audit_confidence(rows)
    test_ids = set(first['partitions']['test'])
    modified = [dict(row, reference_label_5mm=1-row['reference_label_5mm'], confidence=.05)
                if row['sample_id'] in test_ids else row for row in rows]
    second = audit_confidence(modified)
    assert first['partitions'] == second['partitions']
    a, b = first['strata'][0]['heldout'], second['strata'][0]['heldout']
    assert a is not None and a['calibrator_bins'] == b['calibrator_bins']
    assert a['test']['coverage'] == 1. and b['test']['coverage'] == 0.
    assert b['test']['scores'] is None


def test_group_weighting_does_not_reward_duplicate_frames():
    rows = [sample(i, 'a', 'a', label=1) for i in range(10)]
    rows.append(sample(10, 'b', 'b', label=0))
    row, = audit_confidence(rows)['strata']
    assert row['raw_left_reference_diagnostic']['observed_accuracy'] == pytest.approx(.5)


@pytest.mark.parametrize('field,value', [('confidence', float('nan')), ('confidence', 1.2),
    ('reference_label_5mm', False), ('reference_label_5mm', -1), ('right_evidence', 'unknown')])
def test_invalid_evidence_rejected(field, value):
    row = sample(0, 'a', 'a')
    row[field] = value
    with pytest.raises(ValueError, match='invalid candidate evidence'):
        audit_confidence([row])


def test_duplicate_candidates_are_rejected():
    row = sample(0, 'a', 'a')
    with pytest.raises(ValueError, match='duplicate'):
        audit_confidence([row, row])


def test_replay_manifest_checks_candidate_archive_hash_before_loading(tmp_path):
    summary = tmp_path/'summary.json'
    frame = tmp_path/'frame_0000.pt'
    torch.save({}, frame)
    value = dict(scope='offline_replay_of_existing_native_RGB_only',
        native_capture_started=False, capture_sha256='b'*64, calibration_sha256='c'*64,
        proof_sha256='d'*64, checkpoint_sha256='e'*64, reference_used_by_matcher=False,
        reference_sha256=['a'*64], baseline_retention=dict(reference_used_for_selection=False,
            candidate_5mm_label_is_offline_only=True, deployable=False),
        frames=[dict(artifact_file=frame.name, artifact_sha256='f'*64)])
    summary.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='artifact SHA'):
        load_replay_rows(summary, file_sha256(summary))

def test_replay_loader_verifies_same_frame_right_evidence_and_counts(tmp_path):
    summary = tmp_path/'summary.json'
    frame = tmp_path/'frame_0000.pt'
    identity = dict(frame_id=2, calibration_sha256='c'*64, env_id=0,
                    grasp_style='pinch_thumb_index')
    method = dict(visibility={'visible': torch.tensor([[[True, False]]])},
        baseline_slot_retention=dict(candidate_pool_mask=torch.tensor([[True, True]]),
            candidate_depth_source=torch.tensor([[1, 2]], dtype=torch.uint8),
            candidate_reference_label_5mm=torch.tensor([[1, -1]], dtype=torch.int8),
            candidate_geometry=dict(confidence=torch.tensor([[.8, .4]]),
                source_type=torch.tensor([[2, 2]], dtype=torch.uint8), valid=torch.tensor([[True, True]]),
                point_pixels=torch.tensor([[[0, 0], [0, 1]]], dtype=torch.long))))
    archive = dict(identity=[identity], reference_sha256='a'*64, methods={'colour_anchor_retained':method},
                   right_native_instance_ids=torch.ones((1,1,2), dtype=torch.long))
    evidence = dict(identity=identity, baseline_slot_retention=dict(promoted_slots=0,
        candidate_pool_size=2, candidate_pool_within_5mm=1, candidate_pool_above_5mm=0,
        candidate_pool_without_reference=1), correspondence=dict(right_instance_source='native_instance_id'),
        correct_point_transition_from_legacy_colour=dict(baseline_correct=91, retained_correct=91))
    value = dict(scope='offline_replay_of_existing_native_RGB_only',
        native_capture_started=False, capture_sha256='b'*64, calibration_sha256='c'*64,
        proof_sha256='d'*64, checkpoint_sha256='e'*64, reference_used_by_matcher=False,
        reference_sha256=['a'*64], baseline_retention=dict(reference_used_for_selection=False,
            candidate_5mm_label_is_offline_only=True, deployable=False))
    def write():
        torch.save(archive, frame)
        value['frames'] = [dict(frame_id=2, identity=[identity], artifact_file=frame.name,
            artifact_sha256=file_sha256(frame), methods={'colour_anchor_retained':{'environments':[evidence]}})]
        summary.write_text(json.dumps(value))
        return file_sha256(summary)
    rows, binding = load_replay_rows(summary, write())
    assert [(row['right_evidence'], row['reference_label_5mm']) for row in rows] == [(True,1), (False,None)]
    assert binding['retained_baselines'][0]['retained_correct'] == 91
    archive.pop('right_native_instance_ids')
    with pytest.raises(ValueError, match='right identity evidence'):
        load_replay_rows(summary, write())
