"""CPU artifacts exercise the producer only; synthetic files grant no native receipt."""
import hashlib
import json

import pytest

from test_native_teacher_worker_artifacts import artifact_tree, cpu_native_source_gate
from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_artifacts as auditor


GENERATED = ("physics.jsonl", "transitions.jsonl", "ownership.jsonl",
             "training_result.json", "shutdown_complete.json")


def _derive():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_artifacts import derive_actual_artifacts
    return derive_actual_artifacts


@pytest.fixture
def attempt(tmp_path, cpu_native_source_gate):
    # The original validator fixture supplies the core result shape and raw journal.
    spec, output, rows, result = artifact_tree(tmp_path)
    for name in GENERATED:
        (output / name).unlink()
    journal = output / "training_events.jsonl"
    lifecycle = output / "lifecycle"
    lifecycle.mkdir()
    record = dict(acquisition={
        **result["training"], "clean_checkpoint": result["clean_checkpoint"],
        "physical": dict(start_sdk_step=2, end_sdk_step=258, start_rod_tick=0, end_rod_tick=3072),
    }, close_mode="normal", failure=None, process_exit_verified=False, native_qualification=False,
        lifetime_released_before_close=True)
    values = {
        "result.json": record,
        "cleanup_stages.json": dict(close_mode="normal", failure=None, qualification_granted=False,
            stages={stage: dict(attempted=True, returned=True, error=None) for stage in
                    ("stop", "manager_before_close", "ownership_before_close", "close", "close_returned_marker")}),
        "app_close_returned.json": dict(app_close_returned=True, close_mode="normal",
                                       process_exit_verified=False, qualification_granted=False),
        "stop_commit.json": dict(stop_commit_returned=True, explicit_physics_steps_requested=0),
    }
    for name, value in values.items():
        (lifecycle / name).write_text(json.dumps(value))
    (lifecycle / "ownership.jsonl").write_text('{"owner":"cpu R06 fixture"}\n')
    return spec, output, rows, journal, lifecycle


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_derive_preserves_raw_and_separates_r06_ownership(attempt):
    spec, output, rows, journal, lifecycle = attempt
    raw_before, lifecycle_before = journal.read_bytes(), {f.name: f.read_bytes() for f in lifecycle.iterdir()}
    result = _derive()(spec=spec, output=output)
    assert journal.read_bytes() == raw_before
    assert {f.name: f.read_bytes() for f in lifecycle.iterdir()} == lifecycle_before
    assert result["process_exit_verified"] is False and result["native_qualification"] is False
    assert result["raw_journal_sha256"] == _sha(journal)
    assert result["lifecycle_sha256"] == {f.name: _sha(f) for f in lifecycle.iterdir()}
    old = auditor.audit_teacher_artifacts_v2(spec, output)
    assert old["ledger_counts"] == {"physics.jsonl": 256, "transitions.jsonl": 32,
                                   "training_events.jsonl": len(rows), "ownership.jsonl": 1}
    assert old["journal_rows"] == rows
    assert (output / "ownership.jsonl").read_bytes() != (lifecycle / "ownership.jsonl").read_bytes()


@pytest.mark.parametrize("damage", ["fault", "sigsegv_claim", "unreleased", "bad_checkpoint",
                                    "partial_row", "short_rollout", "symlink", "existing_view"])
def test_derive_rejects_incomplete_or_reused_native_output(attempt, damage):
    spec, output, rows, journal, lifecycle = attempt
    if damage in ("fault", "sigsegv_claim", "unreleased"):
        p = lifecycle / "result.json"
        x = json.loads(p.read_text())
        key, value = {"fault": ("failure", "fault"), "sigsegv_claim": ("process_exit_verified", True),
                      "unreleased": ("lifetime_released_before_close", False)}[damage]
        x[key] = value
        p.write_text(json.dumps(x))
    elif damage == "bad_checkpoint":
        (output / "teacher_clean.pt").write_bytes(b"changed after training")
    elif damage == "partial_row":
        journal.write_bytes(journal.read_bytes().rstrip(b"\n"))
    elif damage == "short_rollout":
        journal.write_text("".join(json.dumps(row) + "\n" for row in rows[:12]))
    elif damage == "symlink":
        p = lifecycle / "stop_commit.json"
        copy = p.rename(output / "copy.json")
        p.symlink_to(copy)
    else:
        (output / "physics.jsonl").write_text("preserved output")
    with pytest.raises((ValueError, FileExistsError), match="R06|ownership|checkpoint|newline|journal|output|scope"):
        _derive()(spec=spec, output=output)
    assert not (output / "training_result.json").exists()
    assert set(x.name for x in output.iterdir()) & set(GENERATED) == (
        {"physics.jsonl"} if damage == "existing_view" else set())
