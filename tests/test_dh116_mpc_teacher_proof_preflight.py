"""CPU format and fail-closed checks for the Teacher expert proof preflight."""
import json

import torch

from go2_pvcnn.control.dh116 import tabletop_mpc_teacher_proof_preflight as preflight


def test_advisory_outer_exit_zero_does_not_supply_external_native_provenance(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'monitor.json').write_text(json.dumps({'control_steps': 720, 'successful_episodes': 0}))
    observer = tmp_path / 'observer'
    observer.mkdir()
    (observer / 'exit_code').write_text('0\n')
    result = preflight.inspect_run(run, observer=observer)
    assert result['process_evidence']['advisory_observed_exit_code'] == 0
    assert result['process_evidence']['formal_completion_supplied'] is False
    assert result['episodes_ready_for_assembly'] == 0
    assert 'independent_native_episode_proofs_missing' in result['blocked_reasons']


def test_successful_legacy_rows_do_not_make_incomplete_371_values_or_force_qualified(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    mask = torch.zeros(2,371,dtype=torch.bool)
    mask[:, :24] = True
    torch.save(dict(first_episode_outcome='successful', qualified_for_bc=False,
        observations=torch.zeros(2,371), actions=torch.zeros(2,24), privileged_valid=mask,
        sample_metadata=[dict(episode_uid='episode', env_id=0, tick=i) for i in range(2)]),
        run/'mpc_successful_first_episodes_unqualified.pt')
    result = preflight.inspect_run(run)
    assert result['legacy_partitions']['successful']['episodes'] == 1
    assert result['legacy_partitions']['successful']['fully_valid_371_rows'] == 0
    assert result['episodes_ready_for_assembly'] == 0
    assert 'successful_rows_have_unverified_371_state' in result['blocked_reasons']


def test_mpc_gzip_schema_never_substitutes_for_canonical_force_and_hold_replay(tmp_path):
    import gzip
    run = tmp_path / 'run'
    sidecar = run / 'mpc_state_capture'
    sidecar.mkdir(parents=True)
    with gzip.open(sidecar/'states_480hz.jsonl.gz','wt') as stream:
        stream.write(json.dumps(dict(action_owner='legacy_teacher_mpc', issued_action24=[0]*24,
            state=dict(clock={}, fields={})))+'\n')
    with gzip.open(sidecar/'sensor_evidence_480hz.jsonl.gz','wt') as stream:
        stream.write(json.dumps(dict(producer_evidence=dict(native_attested=True)))+'\n')
    result = preflight.inspect_run(run)
    assert result['legacy_raw_schema']['states_480hz.jsonl.gz']['schema_observed'] is True
    assert result['legacy_raw_schema']['sensor_evidence_480hz.jsonl.gz']['schema_observed'] is True
    assert result['episodes_ready_for_assembly'] == 0
    assert 'no_canonical_480hz_episode_and_force_support_proof' in result['blocked_reasons']


def test_proposed_proof_reopens_standard_native_gate_and_reports_failure(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_stereo_promotion as promotion
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    run = tmp_path / 'run'
    run.mkdir()
    proof = tmp_path / 'proof.json'
    proof.write_text('{}')
    def reject(path, **kwargs):
        assert path == proof
        assert kwargs == {'expected_identity': {'test_only': True}, 'subject': 'mpc'}
        raise ValueError('unrecognized native evidence producer; arbitrary file is insufficient')
    monkeypatch.setattr(promotion, 'verify_native_episode', reject)
    result = preflight.inspect_run(run, episode_proofs=[(proof,file_sha256(proof))],
        expected_identity={'test_only':True})
    assert result['episodes_ready_for_assembly'] == 0
    assert result['episode_proofs'][0]['error'].startswith('unrecognized native evidence producer')
    assert 'proposed_native_episode_proof_rejected' in result['blocked_reasons']
