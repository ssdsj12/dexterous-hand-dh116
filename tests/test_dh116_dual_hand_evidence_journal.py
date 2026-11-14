"""Exercise the physical owner and durable recorder with explicit CPU doubles."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import (
    DualHandEvidenceJournal, inspect_evidence,
)
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from test_deformable_r0_replay import setup


def connect(journal, episode, reset):
    for phase, step in (("before_reset", 0), ("after_reset", 2)):
        journal.record(dict(kind="setup", phase=phase, sdk_step=step, sdk_time_s=step / 480))
    journal.record(reset)
    for side in ("left", "right"):
        journal.record(dict(kind="initial_geometry", side=side,
            evidence=dict(side=side, identity=dict(sdk_step=2, sdk_time_s=2 / 480))))
    episode._record = journal.record


def test_roundtrip_real_owner_cpu_rows_with_explicit_unknowns(monkeypatch, tmp_path):
    episode, _, _, sim, rows, _ = setup(monkeypatch, tmp_path, mode="empty")
    path = tmp_path / "owned.jsonl"
    action = torch.zeros((1, 24))
    with DualHandEvidenceJournal(path, physics_budget_end=18, solver_substeps=12) as journal:
        connect(journal, episode, rows[0])
        for _ in range(2):
            journal.begin(episode.pre_action_state(), action)
            journal.end(episode.control(action))
    result = inspect_evidence(path)
    assert result["record_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["complete_control_windows"] == 2
    assert result["measured_physics_rows"] == 16
    assert result["last_sdk_step"] == sim.current_time_step_index == 18
    assert result["teacher_admission"] is result["native_qualification"] is False
    packets = [decode_owned(json.loads(line)["payload"]) for line in path.read_text().splitlines()[1:]]
    pre = next(row for row in packets if row.get("pre_action_state"))["pre_action_state"]["common"]
    assert pre["valid"].sum() == 66 and np.isnan(pre["values"][~pre["valid"]]).all()
    assert "NaN" not in path.read_text()
    with pytest.raises(FileExistsError):
        DualHandEvidenceJournal(path, physics_budget_end=18, solver_substeps=12)


def test_terminal_is_readable_before_settlement_and_disallows_noop(monkeypatch, tmp_path):
    episode, _, hands, sim, rows, _ = setup(monkeypatch, tmp_path)
    path = tmp_path / "terminal.jsonl"
    original = sim.step

    def fail(*, render):
        original(render=render)
        hands[1].measured[0, hands[1].joint_names.index("finger12")] = -.05

    sim.step = fail
    with DualHandEvidenceJournal(path, physics_budget_end=18, solver_substeps=12) as journal:
        connect(journal, episode, rows[0])
        pre = episode.pre_action_state()
        action = torch.zeros((1, 24))
        journal.begin(pre, action)
        control = episode.control(action)
        prefix = inspect_evidence(path)
        assert prefix["terminal_reason"] == "right:joint_limit"
        assert prefix["unsettled_control_window"] is True
        journal.end(control)
        with pytest.raises(ValueError, match="terminal"):
            journal.begin(pre, action)
    assert inspect_evidence(path)["measured_physics_rows"] == 1
    assert sim.current_time_step_index == 3


@pytest.mark.parametrize("damage", ["policy", "step", "rod", "action", "after_terminal", "truncated"])
def test_independent_reader_detects_corruption(monkeypatch, tmp_path, damage):
    episode, _, _, _, rows, _ = setup(monkeypatch, tmp_path)
    path = tmp_path / "original.jsonl"
    with DualHandEvidenceJournal(path, physics_budget_end=18, solver_substeps=12) as journal:
        connect(journal, episode, rows[0])
        action = torch.zeros((1, 24))
        journal.begin(episode.pre_action_state(), action)
        journal.end(episode.control(action))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    first = next(row for row in lines if row.get("operation") == "physical")
    exchange = first["payload"]["exchange"]
    if damage == "policy":
        exchange["policy_step"] += 1
    elif damage == "step":
        exchange["physics_step"] += 1
    elif damage == "rod":
        first["payload"]["rod_boundary"]["rod_tick"] += 12
    elif damage == "action":
        exchange["action_readback"]["request"]["action24"]["__dh116_array__"][0][0] = .5
    elif damage == "after_terminal":
        exchange["guard_failure"] = "right:joint_limit"
    tampered = tmp_path / "tampered.jsonl"
    text = "\n".join(json.dumps(row) for row in lines) + "\n"
    tampered.write_text(text[:-3] if damage == "truncated" else text)
    with pytest.raises(ValueError):
        inspect_evidence(tampered)


def test_rejects_overbudget_and_retain_prefix_after_failed_observation(monkeypatch, tmp_path):
    episode, _, _, _, rows, _ = setup(monkeypatch, tmp_path)
    short = tmp_path / "short.jsonl"
    with DualHandEvidenceJournal(short, physics_budget_end=10, solver_substeps=12) as journal:
        connect(journal, episode, rows[0])
        action = torch.zeros((1, 24))
        journal.begin(episode.pre_action_state(), action)
        journal.end(episode.control(action))
        with pytest.raises(ValueError, match="budget"):
            journal.begin(episode.pre_action_state(), action)
    episode, _, _, _, rows, _ = setup(monkeypatch, tmp_path)
    path = tmp_path / "prefix.jsonl"
    with DualHandEvidenceJournal(path, physics_budget_end=18, solver_substeps=12) as journal:
        connect(journal, episode, rows[0])
        action = torch.zeros((1, 24))
        journal.begin(episode.pre_action_state(), action)
        episode._record = rows.append
        episode.control(action)
        journal.record(rows[1])
        damaged = deepcopy(rows[2])
        damaged["exchange"]["physics_step"] += 1
        with pytest.raises(ValueError, match="physical"):
            journal.record(damaged)
        with pytest.raises(ValueError, match="closed"):
            journal.record(rows[2])
    result = inspect_evidence(path)
    assert result["unsettled_control_window"] is True
    assert result["measured_physics_rows"] == 1
