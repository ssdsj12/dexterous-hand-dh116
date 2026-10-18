import json
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.evidence_journal import EvidenceJournal


def test_append_visible_before_close_and_snapshot_not_aliased(tmp_path):
    path = tmp_path / 'trajectory.jsonl'
    with EvidenceJournal(path) as journal:
        row = {'step': 0, 'passed': np.bool_(True), 'q': np.array([1.])}
        journal.append(row)
        row['q'][0] = 2.
        assert json.loads(path.read_text())['q'] == [1.]
        assert journal[-1]['q'] == [1.]
        assert len(journal) == 1
        assert list(journal) == [journal[-1]]
    assert len(path.read_text().splitlines()) == 1


def test_many_records_keep_only_last_and_replay_in_order(tmp_path):
    with EvidenceJournal(tmp_path / 'trajectory.jsonl') as journal:
        for step in range(1000):
            journal.append({'step': step, 'passed': step != 17})
        assert len(journal) == 1000
        assert journal[-1]['step'] == 999
        assert not any(isinstance(v, list) for v in vars(journal).values())
        assert [r['step'] for r in journal] == list(range(1000))
        assert not all(r['passed'] for r in journal)
        with pytest.raises(IndexError):
            journal[0]


def test_reject_overwrite_and_invalid_record_without_losing_history(tmp_path):
    path = tmp_path / 'trajectory.jsonl'
    with EvidenceJournal(path) as journal:
        journal.append({'step': 0})
        saved = path.read_bytes()
        with pytest.raises(ValueError):
            journal.append({'value': float('nan')})
        assert path.read_bytes() == saved
        assert len(journal) == 1
        with pytest.raises(FileExistsError):
            EvidenceJournal(path)
    assert path.read_bytes() == saved


def test_exception_retains_flushed_evidence_and_close_is_idempotent(tmp_path):
    path = tmp_path / 'trajectory.jsonl'
    with pytest.raises(RuntimeError):
        with EvidenceJournal(path) as journal:
            journal.append({'step': 0})
            raise RuntimeError('guard aborted')
    journal.close()
    assert list(journal) == [{'step': 0}]
    with pytest.raises(ValueError):
        journal.append({'step': 1})


def test_empty_has_no_last(tmp_path):
    with EvidenceJournal(tmp_path / 'trajectory.jsonl') as journal:
        assert len(journal) == 0
        assert list(journal) == []
        with pytest.raises(IndexError):
            journal[-1]


def test_runner_streams_without_finally_truncating_trajectory():
    source = (Path(__file__).parents[1] / 'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert "rows=EvidenceJournal(args.output/'trajectory.jsonl')" in source
    assert "(args.output/'trajectory.jsonl').open('w')" not in source
    assert 'rows.close()' in source
