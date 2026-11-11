"""CPU fixtures exercise artifact boundaries; they are not native captures."""
import copy
import json

import pytest
import torch

from test_dh116_stereo_state_training_modes import center_rows, estimator
from go2_pvcnn.control.dh116 import tabletop_stereo_state_supervision as state
from go2_pvcnn.control.dh116.tabletop_stereo_contract import digest, file_sha256


def _bound_row(group, family, run):
    row = center_rows()[0]
    rgb = row['history']['left_rgb']
    processing = dict(version='dh116-stereo-processing-v1', calibration_sha256='c'*64,
        image_resolution_wh=[rgb.shape[-1], rgb.shape[-2]],
        depth_estimator={'method':'wire_silhouette'},
        geometry={'missing':'NaN_with_false_validity;no_simulator_or_rendered_depth_fallback'},
        source_sha256={'tabletop_stereo_depth.py':'a'*64}, native_qualified=False)
    processing['processing_sha256'] = digest(processing)
    identity = dict(run_uuid=run, env_id=0, episode_uid=run+':0:1', reset_id=1,
        policy_tick=0, frame_id=3, physics_tick_start=0, physics_tick_end=0,
        simulation_timestamp=0., calibration_sha256='c'*64, physical_initial_condition_sha256=group*64,
        source_episode_group=family, observation_preprocessor=processing,
        observation_preprocessor_sha256=processing['processing_sha256'],
        state_reference_artifact={'path':'fixture.pt', 'sha256':'e'*64})
    valid = row['history']['depth_valid'][0,-1]
    row.update(identity=identity, privileged_identity=copy.deepcopy(identity),
        teacher_mean=torch.zeros(24), issued_actions=torch.zeros(24), source_teacher_checkpoint_sha256='d'*64)
    row['extra_labels']['stereo_disparity_px'] = torch.ones_like(valid, dtype=torch.float32)
    row['label_validity']['stereo_disparity_px'] = valid.clone()
    return row


def _dataset(tmp_path, monkeypatch, groups):
    sources, rows = [], {}
    for i, (initial, family, run) in enumerate(groups):
        candidate = tmp_path/f'cohort-{i}.pt'
        raw = tmp_path/f'cohort-{i}.jsonl'
        candidate.write_bytes(f'candidate {i}'.encode())
        raw.write_bytes(f'ledger {i}'.encode())
        sources.append(dict(candidate=candidate.name, candidate_sha256=file_sha256(candidate),
                            raw=raw.name, raw_sha256=file_sha256(raw)))
        rows[candidate] = [_bound_row(initial, family, run)]
    manifest = tmp_path/'dataset.json'
    manifest.write_text(json.dumps(dict(version='dh116-stereo-state-dataset-v1', sources=sources)))
    calls = []

    def read(path, *, raw, sha256, raw_sha256):
        assert file_sha256(path) == sha256 and file_sha256(raw) == raw_sha256
        calls.append(path)
        return rows[path]

    monkeypatch.setattr(state, 'read_candidate', read)
    return manifest, rows, calls


def _read(path):
    return state.read_candidate_dataset(path, expected_sha256=file_sha256(path))


def test_cross_bundle_physical_and_source_aliases_remain_in_one_split(tmp_path, monkeypatch):
    manifest, _, calls = _dataset(tmp_path, monkeypatch, [
        ('a','family-1','run-a'), ('a','family-2','run-b'), ('b','family-2','run-c'),
        ('c','family-3','run-d'), ('d','family-4','run-e')])
    rows, audit = _read(manifest)
    assert len(calls) == len(rows) == len(audit['sources']) == 5
    assert audit['manifest_sha256'] == file_sha256(manifest)
    assert audit['source_scope'] == 'verified_native_capture_replayed_offline'
    assert audit['successful_demonstration_pool'] is False
    split = state.split_estimator_rows(rows)
    assert split['independent_groups'] == 3
    assert any({0,1,2}.issubset(split[p+'_indices']) for p in state.PARTITIONS)
    reverse = state.split_estimator_rows(rows[::-1])
    for part in state.PARTITIONS:
        assigned = {rows[i]['identity']['run_uuid'] for i in split[part+'_indices']}
        assert assigned == {rows[::-1][i]['identity']['run_uuid'] for i in reverse[part+'_indices']}


def test_duplicate_capture_across_bundles_is_rejected(tmp_path, monkeypatch):
    manifest, _, _ = _dataset(tmp_path, monkeypatch,
        [('a','family-1','same-run'), ('a','family-1','same-run')])
    with pytest.raises(ValueError, match='duplicate'):
        _read(manifest)


@pytest.mark.parametrize('damage', ['calibration','privileged_clock','disparity','reference','processing'])
def test_unbound_source_channel_is_rejected(tmp_path, monkeypatch, damage):
    manifest, rows, _ = _dataset(tmp_path, monkeypatch, [('a','family','run')])
    row = next(iter(rows.values()))[0]
    if damage == 'privileged_clock':
        row['privileged_identity']['physical_initial_condition_sha256'] = 'b'*64
    elif damage == 'calibration':
        row['identity']['calibration_sha256'] = row['privileged_identity']['calibration_sha256'] = 'd'*64
    elif damage == 'disparity':
        row['label_validity']['stereo_disparity_px'].logical_not_()
    elif damage == 'reference':
        row['identity'].pop('state_reference_artifact')
        row['privileged_identity'].pop('state_reference_artifact')
    else:
        row['identity']['observation_preprocessor']['depth_estimator']['method'] = 'reference_depth'
        row['privileged_identity'] = copy.deepcopy(row['identity'])
    with pytest.raises(ValueError, match='binding|calibration|privileged|disparity|reference|processing'):
        _read(manifest)


def test_manifest_bytes_are_bound_before_opening_any_bundle(tmp_path, monkeypatch):
    manifest, _, calls = _dataset(tmp_path, monkeypatch, [('a','family','run')])
    with pytest.raises(ValueError, match='manifest SHA'):
        state.read_candidate_dataset(manifest, expected_sha256='f'*64)
    assert calls == []


def test_public_manifest_cli_records_admission_and_blocks_two_group_training(tmp_path, monkeypatch):
    manifest, _, _ = _dataset(tmp_path, monkeypatch, [('a','family-1','run-a'), ('b','family-2','run-b')])
    output = tmp_path/'formal'
    exit_code = state.main(['--dataset-manifest', str(manifest), '--dataset-manifest-sha256', file_sha256(manifest),
        '--output', str(output), '--mode', 'formal_training'])
    assert exit_code == 2
    report = json.loads((output/'state_supervision_report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'blocked_insufficient_independent_groups'
    assert report['dataset_manifest_sha256'] == file_sha256(manifest)
    assert [source['teacher_checkpoint_sha256'] for source in report['dataset_sources']] == ['d'*64]*2
    assert report['stages']['center_xyz']['eligible_rows'] == 2
    assert not any(stage['enabled'] for stage in report['stages'].values())
    assert not (output/'state_estimator.pt').exists()


def test_formal_training_separates_center_from_other_geometry():
    rows = center_rows()
    extra = copy.deepcopy(rows[0])
    extra['identity']['physical_initial_condition_sha256'] = 'c'*64
    extra['identity']['source_episode_group'] = 'c'
    rows.append(extra)
    report = state.run_training_mode(estimator(), rows, mode='formal_training', batch_size=2)
    assert report['status'] == 'formal_state_fit_completed'
    assert list(report['stages']) == ['center_xyz','wire_geometry','velocity','deformation','contact','slip','force_proxy']
    assert report['stages']['center_xyz']['supervised_indices'] == [61,62,63]
    assert [name for name, entry in report['stages'].items() if entry['enabled']] == ['center_xyz']
