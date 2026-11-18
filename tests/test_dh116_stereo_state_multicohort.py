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


def test_named_offline_mode_requires_reopened_manifest_and_never_promotes(tmp_path, monkeypatch):
    manifest, _, _ = _dataset(tmp_path, monkeypatch, [
        ('a','family-1','run-a'), ('b','family-2','run-b'), ('c','family-3','run-c')])
    rows, _ = _read(manifest)
    report = state.run_training_mode(estimator(), rows, mode='offline_qualified_training')
    assert report['status'] == 'blocked_missing_verified_dataset_manifest'
    assert report['state_training_started'] is False
    output = tmp_path/'offline'
    assert state.main(['--dataset-manifest', str(manifest), '--dataset-manifest-sha256', file_sha256(manifest),
        '--output', str(output), '--mode', 'offline_qualified_training']) == 0
    report = json.loads((output/'state_supervision_report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'offline_qualified_state_fit_completed'
    assert report['native_stereo_source_qualified'] is False
    assert report['actor_deployment_qualified'] is False
    assert report['generalization_qualified'] is False


def test_named_native_mode_missing_proof_does_not_train_or_fall_back(tmp_path, monkeypatch):
    manifest, _, _ = _dataset(tmp_path, monkeypatch, [
        ('a','family-1','run-a'), ('b','family-2','run-b'), ('c','family-3','run-c')])
    output = tmp_path/'native'
    assert state.main(['--dataset-manifest', str(manifest), '--dataset-manifest-sha256', file_sha256(manifest),
        '--output', str(output), '--mode', 'native_qualified_training']) == 2
    report = json.loads((output/'state_supervision_report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'blocked_missing_native_stereo_proof'
    assert report['state_training_started'] is False
    assert not (output/'state_estimator.pt').exists()


def test_native_mode_rejects_claims_and_cross_processing_proofs(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_stereo_promotion as promotion
    from go2_pvcnn.control.dh116 import tabletop_stereo_preprocessing as processing
    manifest, _, _ = _dataset(tmp_path, monkeypatch, [
        ('a','family-1','run-a'), ('b','family-2','run-b'), ('c','family-3','run-c')])
    config = tmp_path/'camera.json'
    config.write_text('{}')
    proof = tmp_path/'proof.json'
    proof.write_text(json.dumps(dict(qualified=True,
        calibration_artifact=dict(path=config.name, sha256=file_sha256(config)))))
    def reject_unverified(_):
        raise ValueError('fixture must not qualify as native')
    monkeypatch.setattr(promotion, 'verify_stereo_qualification', reject_unverified)
    args = ['--dataset-manifest', str(manifest), '--dataset-manifest-sha256', file_sha256(manifest),
        '--mode', 'native_qualified_training', '--native-stereo-proof', str(proof),
        '--native-stereo-proof-sha256', file_sha256(proof)]
    first = tmp_path/'native_rejected'
    assert state.main([*args, '--output', str(first)]) == 2
    rejected = json.loads((first/'state_supervision_report.json').read_text(encoding='utf-8'))
    assert rejected['status'] == 'blocked_native_stereo_proof'
    assert 'must not qualify' in rejected['qualification']['detail']
    monkeypatch.setattr(promotion, 'verify_stereo_qualification', lambda _: dict(
        qualified=True, measurements_verified=True, source_scope='native',
        calibration_sha256='c'*64, depth_estimator_config={'method':'wire_silhouette'}))
    monkeypatch.setattr(processing, 'processing_contract', lambda _: dict(
        calibration_sha256='c'*64, processing_sha256='9'*64, depth_estimator={'method':'wire_silhouette'}))
    second = tmp_path/'native_mismatched'
    assert state.main([*args, '--output', str(second)]) == 2
    mismatch = json.loads((second/'state_supervision_report.json').read_text(encoding='utf-8'))
    assert mismatch['status'] == 'blocked_native_stereo_proof'
    assert 'processing' in mismatch['qualification']['detail']
    assert not (second/'state_estimator.pt').exists()


def test_native_mode_binds_verified_source_and_preserves_model_gates(tmp_path, monkeypatch):
    # The verifier's result is a CPU fixture for admission wiring only.
    from go2_pvcnn.control.dh116 import tabletop_stereo_promotion as promotion
    from go2_pvcnn.control.dh116 import tabletop_stereo_preprocessing as processing
    manifest, _, _ = _dataset(tmp_path, monkeypatch, [
        ('a','family-1','run-a'), ('b','family-2','run-b'), ('c','family-3','run-c')])
    _, evidence = _read(manifest)
    config = tmp_path/'camera.json'
    config.write_text('{}')
    proof = tmp_path/'proof.json'
    proof.write_text(json.dumps(dict(calibration_artifact=dict(
        path=config.name, sha256=file_sha256(config)))))
    monkeypatch.setattr(promotion, 'verify_stereo_qualification', lambda _: dict(
        qualified=True, measurements_verified=True, source_scope='native',
        calibration_sha256=evidence['calibration_sha256'],
        depth_estimator_config={'method':'wire_silhouette','image_processing_batch_size':2}))
    monkeypatch.setattr(processing, 'processing_contract', lambda _: dict(
        calibration_sha256=evidence['calibration_sha256'],
        processing_sha256=evidence['observation_preprocessor_sha256'],
        depth_estimator={'method':'wire_silhouette'}))
    out = tmp_path/'native'
    args = ['--dataset-manifest', str(manifest), '--dataset-manifest-sha256', file_sha256(manifest),
        '--mode', 'native_qualified_training', '--native-stereo-proof', str(proof)]
    assert state.main([*args, '--native-stereo-proof-sha256', file_sha256(proof), '--output', str(out)]) == 0
    report = json.loads((out/'state_supervision_report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'native_qualified_state_fit_completed'
    assert report['fit_execution_scope'] == 'offline_tensor_optimization'
    assert report['native_stereo_source_qualified'] is True
    assert report['generalization_qualified'] is False
    assert report['actor_deployment_qualified'] is False
    model = torch.load(out/'state_estimator.pt', map_location='cpu', weights_only=True)
    assert model['qualification']['native_stereo_proof_sha256'] == file_sha256(proof)
    assert model['successful_demonstration_pool'] is False

    bad_out = tmp_path/'bad_sha'
    assert state.main([*args, '--native-stereo-proof-sha256', 'f'*64, '--output', str(bad_out)]) == 2
    rejected = json.loads((bad_out/'state_supervision_report.json').read_text(encoding='utf-8'))
    assert rejected['status'] == 'blocked_native_stereo_proof'
    assert 'SHA mismatch' in rejected['qualification']['detail']
