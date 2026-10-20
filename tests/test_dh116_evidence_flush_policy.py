from go2_pvcnn.control.dh116.evidence_journal import EvidenceJournal


def test_buffered_policy_flushes_on_count(tmp_path):
    journal = EvidenceJournal(tmp_path / 'trajectory.jsonl',
                              flush_policy='buffered', flush_every=2)
    journal.append({'step': 0})
    assert journal.flush_count == 0
    journal.append({'step': 1})
    assert journal.flush_count == 1
    journal.close()


def test_phase_transition_forces_flush(tmp_path):
    journal = EvidenceJournal(tmp_path / 'trajectory.jsonl', flush_policy='buffered')
    journal.append({'step': 0}, force_flush=True)
    assert journal.flush_count == 1
    journal.close()


def test_audit_policy_remains_visible_after_each_append(tmp_path):
    path = tmp_path / 'trajectory.jsonl'
    journal = EvidenceJournal(path)
    journal.append({'step': 0})
    assert path.read_text().splitlines() == ['{"step": 0}']
    journal.close()
