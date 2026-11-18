import hashlib
import json

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_mpc_privileged_capture import PreActionSample


def _sample(tick):
    state = torch.arange(371, dtype=torch.float32).reshape(1, 371).repeat(2, 1) / 13
    valid = torch.zeros_like(state, dtype=torch.bool)
    valid[:, :24] = True
    action = torch.full((2, 24), .2345, dtype=torch.float64)
    before = dict(world_step=tick*8, sdk_step=18+tick*8,
                  world_time_s=tick/60, sdk_time_s=(18+tick*8)/480)
    after = dict(world_step=before['world_step']+8, sdk_step=before['sdk_step']+8,
                 world_time_s=(tick+1)/60, sdk_time_s=(18+(tick+1)*8)/480)
    evidence = dict(samples=dict(
        run_uuid=['actual-native-run', None], env_id=[0, None],
        episode_uid=['actual-native-run:0:first', None], tick=[tick, None],
        policy_timestamp_s=[before['sdk_time_s'], None], first_physics_step=[before['sdk_step'], None],
        terminal=torch.tensor([False, False]), actions=action.clone(),
        valid=torch.tensor([True, False])))
    return PreActionSample(state, valid, action, before), after, evidence


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_completed_step_is_fsynced_and_recoverable_when_finalizer_fails(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_mpc_capture_journal as journal

    path = tmp_path / 'journal.jsonl'
    calls = []
    original = journal.os.fsync

    def checked_fsync(fd):
        original(fd)
        calls.append(path.stat().st_size)

    monkeypatch.setattr(journal.os, 'fsync', checked_fsync)
    sample, after, evidence = _sample(0)
    with pytest.raises(RuntimeError, match='legacy finalize'), journal.capture_journal(path, 2, 2) as writer:
        writer.append(sample, after, evidence)
        assert calls[-1] == path.stat().st_size
        raise RuntimeError('legacy finalize failed')
    recovered = journal.recover_journal(path, _sha(path))
    assert recovered.status == 'quarantined_exception'
    assert recovered.qualified_for_bc is False
    assert recovered.samples[0].state.equal(sample.state)
    assert recovered.samples[0].action.dtype == torch.float64
    assert recovered.samples[0].action.equal(sample.action)
    assert recovered.samples[0].valid.equal(sample.valid)
    assert recovered.identities[0][0]['camera_episode_uid'] == 'actual-native-run:0:first'
    assert not recovered.identities[0][0]['terminal']
    assert len(recovered.identities[0]) == 1


@pytest.mark.parametrize('fault', ['clock', 'action', 'sequence_clock', 'budget', 'unknown_ownership'])
def test_rejects_unproven_or_overbudget_step(tmp_path, fault):
    from go2_pvcnn.control.dh116 import tabletop_mpc_capture_journal as journal

    path = tmp_path / 'journal.jsonl'
    sample, after, evidence = _sample(0)
    if fault == 'clock':
        after['sdk_step'] -= 1
    if fault == 'action':
        evidence['samples']['actions'][0, 2] = 0.
    if fault == 'sequence_clock':
        evidence['samples']['first_physics_step'][0] += 1
    if fault == 'unknown_ownership':
        evidence['samples']['env_id'] = [None, None]
    with pytest.raises(ValueError), journal.capture_journal(path, 2, 1) as writer:
        if fault == 'budget':
            writer.append(sample, after, evidence)
            sample, after, evidence = _sample(1)
        writer.append(sample, after, evidence)
    recovered = journal.recover_journal(path, _sha(path))
    assert len(recovered.samples) == (fault == 'budget')
    assert recovered.status == 'quarantined_exception'


def test_interrupted_final_line_recovers_only_complete_records(tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_mpc_capture_journal as journal

    path = tmp_path / 'journal.jsonl'
    with journal.capture_journal(path, 2, 2) as writer:
        writer.append(*_sample(0))
        writer.finish({'rows': 1, 'qualified_for_bc': False})
    with pytest.raises(ValueError, match='SHA'):
        journal.recover_journal(path, '0' * 64)
    valid = path.read_bytes().splitlines(keepends=True)
    assert json.loads(valid[-1])['kind'] == 'status'
    path.write_bytes(b''.join(valid[:-1]) + b'{"kind":"step",')
    recovered = journal.recover_journal(path, _sha(path))
    assert len(recovered.samples) == 1
    assert recovered.status == 'quarantined_interrupted' and recovered.qualified_for_bc is False
    assert recovered.discarded_partial_line


def test_completed_status_requires_original_joined_rows_and_private_path(tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_mpc_capture_journal as journal

    public = tmp_path / 'public'
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError, match='private existing'):
        with journal.capture_journal(public / 'journal.jsonl', 2, 2):
            pytest.fail('public journal unexpectedly created')
    path = tmp_path / 'journal.jsonl'
    with pytest.raises(ValueError, match='joined rows'), journal.capture_journal(path, 2, 2) as writer:
        writer.append(*_sample(0))
        writer.finish({'rows': 2, 'qualified_for_bc': False})
    assert journal.recover_journal(path, _sha(path)).status == 'quarantined_exception'
