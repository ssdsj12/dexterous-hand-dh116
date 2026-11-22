"""Explicit priming is physical evidence, never a sampled policy transition."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from test_fresh_supported_episode import episode_case
from test_action24_history_owner import model_setup, MeasuredCommonTeacher, DeformablePPOTrainer, PPOConfig
from go2_pvcnn.control.dh116.tabletop_native_teacher_session import SessionBudget
from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal


def module():
    name = "go2_pvcnn.control.dh116.tabletop_native_teacher_source_priming"
    assert importlib.util.find_spec(name) is not None, "missing explicit source priming transaction"
    return __import__(name, fromlist=["prime_supported_sources"])


def case(monkeypatch, tmp_path):
    m = module()
    c, _, plan, episode, rows, readback = episode_case(monkeypatch, tmp_path)
    history = m.PrimedNativeActionHistory(initial_action24=torch.zeros(1, 24),
        seed_source="explicit new CPU zero-command memory", physics_budget_end=18, solver_substeps=12)
    owners = dict(c.owners)
    owners["controller"] = history.wrap_controller_reset(owners["controller"])
    episode._record = history.wrap_recorder(rows.append)
    episode.reset(c.seed, episode_uid="cpu17:priming", owner_resets=owners,
                  supported_mesh_plan=plan, record_reset_readback=readback.append)
    for side in ("left", "right"):
        history.record(dict(kind="initial_geometry", side=side,
            evidence=dict(side=side, identity=dict(sdk_step=2, sdk_time_s=2/480))))
    runtime = SessionBudget(dict(max_sdk_steps=18, max_solver_calls=16,
        max_resets=1, max_episode_count=1, wall_timeout_s=20.))
    owner = type("CPUScene", (), dict(sim=c.sim, rod=c.rod))()
    runtime.record_setup(owner, [dict(sdk_step=0, sdk_time_s=0.), dict(sdk_step=2, sdk_time_s=2/480)])
    runtime.reserve_reset(owner)
    journal = NativeJournal(tmp_path/"priming.jsonl", scope="cpu_fixture")
    return m, c, episode, history, runtime, journal, rows


def test_one_priming_step_retains_reset_and_first_real_policy_window(monkeypatch, tmp_path):
    m, c, episode, history, runtime, journal, rows = case(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="priming"):
        history.capture_pre_action(episode.pre_action_state())
    result = m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    assert c.sim.current_time_step_index == 3 and c.rope.steps == 1
    assert runtime.sdk_steps == 3 and runtime.solver_calls == 1 and runtime.internal_solver_substeps == 12
    assert history._reset["identity"]["physics_step"] == 2
    assert history._sequence.controls == 0 and history._row_hashes == []
    assert result["ppo_transitions"] == 0 and result["reset_identity"]["physics_step"] == 2
    assert result["primed_identity"]["physics_step"] == 3
    assert rows[-1]["kind"] == "priming_physical"
    pre = episode.pre_action_state()
    assert pre["common"]["identity"]["physics_step"] == 3
    captured = history.capture_pre_action(pre)
    assert captured["priming"]["priming_row_sha256"] == result["priming_row_sha256"]
    config, identity, common, valid, raw, clock, mask = model_setup()
    model = MeasuredCommonTeacher(config, identity)
    obs = {k: v[:1].clone() for k,v in model.pack_measured(common, valid, raw, clock, mask).items()}
    trainer = DeformablePPOTrainer(model, PPOConfig(epochs=1, minibatches=1), rollout_steps=1, num_envs=1)
    sample = trainer.collect_step(obs, obs)
    history.begin(pre, trainer)
    control = episode.control(sample["actions"])
    settled = history.end(control)
    issued = history.issued_for_pending(trainer, settled)
    trainer.record_step(torch.tensor([0.]), torch.tensor([False]), torch.tensor([0.]),
                        executed_actions=issued, collection_token=sample["collection_token"])
    assert c.sim.current_time_step_index == 11 and c.rope.steps == 9
    assert len(trainer.rollout) == 1 and trainer.counters["updates"] == 0
    assert settled["physical_samples"] == 8
    assert history._reset["identity"]["rod_tick"] == 0
    assert not settled["teacher_admission"]


def test_repeat_priming_or_insufficient_next_window_budget_is_rejected_before_sdk(monkeypatch, tmp_path):
    m, c, episode, history, runtime, journal, rows = case(monkeypatch, tmp_path)
    runtime.limits["max_sdk_steps"] = 10
    with pytest.raises((ValueError, RuntimeError), match="budget"):
        m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    assert c.sim.current_time_step_index == 2 and c.rope.steps == 0
    runtime.limits["max_sdk_steps"] = 18
    m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    with pytest.raises((ValueError, RuntimeError), match="priming|fresh|active"):
        m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    assert c.sim.current_time_step_index == 3 and c.rope.steps == 1


@pytest.mark.parametrize("fault", ["nonzero_request", "wrong_kind", "foreign_clock"])
def test_malformed_priming_evidence_is_durable_and_cannot_be_backfilled(monkeypatch, tmp_path, fault):
    m, c, episode, history, runtime, journal, rows = case(monkeypatch, tmp_path)
    step = episode._exchange._step
    def damage(action, substep):
        row = step(action, substep)
        if fault == "nonzero_request":
            row["action_readback"]["request"]["action24"][0,0] = .1
        elif fault == "foreign_clock":
            row["action_readback"]["post"]["identity"][0]["episode_uid"] = "foreign"
        return row
    episode._exchange._step = damage
    if fault == "wrong_kind":
        record = episode._record
        def wrong(row):
            if row["kind"] == "priming_physical":
                row = dict(row, kind="physical")
            record(row)
        episode._record = wrong
    with pytest.raises((ValueError, RuntimeError)):
        m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    assert c.sim.current_time_step_index == 3 and c.rope.steps == 1
    assert runtime.sdk_steps == 3 and runtime.solver_calls == 1
    assert "priming_exchange" in journal.path.read_text()
    assert episode._state == "failed" and history._reset["identity"]["physics_step"] == 2
    with pytest.raises((ValueError, RuntimeError)):
        episode.control(torch.zeros(1,24))
    assert c.rope.steps == 1


def test_step_exception_after_progress_is_charged_and_no_replay(monkeypatch, tmp_path):
    m, c, episode, history, runtime, journal, rows = case(monkeypatch, tmp_path)
    step = episode._exchange._step
    def completed_then_raise(action, substep):
        step(action, substep)
        raise RuntimeError("SDK operation failed after measured progress")
    episode._exchange._step = completed_then_raise
    with pytest.raises(RuntimeError, match="measured progress"):
        m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    assert runtime.sdk_steps == 3 and runtime.solver_calls == 1
    assert runtime.internal_solver_substeps == 12 and c.rope.steps == 1
    assert episode._state == "failed"
    assert '"solver_completion_verified":false' in journal.path.read_text().replace(" ", "")
    with pytest.raises((ValueError, RuntimeError)):
        m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    assert c.rope.steps == 1


def test_recorder_failure_poisoning_keeps_consumed_cost_and_raw_exchange(monkeypatch, tmp_path):
    m, c, episode, history, runtime, journal, rows = case(monkeypatch, tmp_path)
    def failed_recorder(row):
        raise OSError("physical ledger fsync failed")
    pre = episode.pre_action_state()
    episode._record = history.wrap_recorder(failed_recorder)
    with pytest.raises(OSError, match="fsync"):
        m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    assert episode._state == "failed" and c.rope.steps == 1
    assert runtime.sdk_steps == 3 and runtime.internal_solver_substeps == 12
    assert "priming_exchange" in journal.path.read_text()
    with pytest.raises(RuntimeError, match="failed action history"):
        history.capture_pre_action(pre)


def test_receipt_mutation_cannot_rewrite_reset_or_prime(monkeypatch, tmp_path):
    m, c, episode, history, runtime, journal, rows = case(monkeypatch, tmp_path)
    m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    returned = history.priming
    returned["reset_identity"]["physics_step"] = 3
    assert history.priming["reset_identity"]["physics_step"] == 2
    history._reset["identity"]["physics_step"] = 3
    with pytest.raises(ValueError, match="changed"):
        history.capture_pre_action(episode.pre_action_state())


def test_old_unprimed_history_still_rejects_post_physics_as_first_boundary(monkeypatch, tmp_path):
    m, c, episode, history, runtime, journal, rows = case(monkeypatch, tmp_path)
    old = deepcopy(history._sequence)
    m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    old.clock = deepcopy(history._sequence.clock)
    with pytest.raises(ValueError):
        old.accept("begin", dict(pre_action_state=episode.pre_action_state(), action=torch.zeros(1, 24)))


def test_nonzero_initial_memory_is_rejected_before_any_sdk_step(monkeypatch, tmp_path):
    m = module()
    with pytest.raises(ValueError, match="zero"):
        m.PrimedNativeActionHistory(initial_action24=torch.ones(1,24)*.1,
            seed_source="nonzero memory", physics_budget_end=18, solver_substeps=12)


def test_journal_failure_invalidates_history_even_before_record_dispatch(monkeypatch, tmp_path):
    m, c, episode, history, runtime, journal, rows = case(monkeypatch, tmp_path)
    pre = episode.pre_action_state()
    def broken_emit(kind, payload):
        raise OSError("raw journal write failed")
    journal.emit = broken_emit
    with pytest.raises(OSError, match="raw journal"):
        m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    assert history._failed and episode._state == "failed"
    with pytest.raises(RuntimeError, match="failed action history"):
        history.capture_pre_action(pre)
    assert runtime.sdk_steps == 3 and runtime.solver_calls == 1


def test_partial_solver_clock_progress_is_counted_without_claiming_completion(monkeypatch, tmp_path):
    m, c, episode, history, runtime, journal, rows = case(monkeypatch, tmp_path)
    clock = c.rod.clock
    step = episode._exchange._step
    def partial_clock_then_raise(action, substep):
        step(action, substep)
        # CPU fault injector: the failing backend reports only five completed
        # internal ticks. These are observations, never successful solver evidence.
        actual = clock()
        actual["rod_tick"] = 5
        monkeypatch.setattr(c.rod, "clock", lambda: deepcopy(actual))
        raise RuntimeError("reported partial solver progress")
    episode._exchange._step = partial_clock_then_raise
    with pytest.raises(RuntimeError, match="partial solver"):
        m.prime_supported_sources(episode, history=history, runtime=runtime, journal=journal)
    assert runtime.sdk_steps == 3 and runtime.solver_calls == 1
    assert runtime.internal_solver_substeps == 5
    assert episode._state == "failed" and history._failed
    assert '"solver_completion_verified":false' in journal.path.read_text().replace(" ", "")
