"""CPU composition and rejection tests. No App or physical simulator is started."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_process as process
from go2_pvcnn.control.dh116.tabletop_native_teacher_training import VERSION


def provider():
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_provider
    return tabletop_native_teacher_actual_provider


def test_run_a_rejects_caller_object_before_creating_output(tmp_path):
    with pytest.raises(TypeError, match="verified.*lease"):
        provider().run_a(config={}, lease=SimpleNamespace(), output=tmp_path / "attempt")
    assert not list(tmp_path.iterdir())


@pytest.fixture
def lease_fixture(tmp_path, monkeypatch):
    lease = object.__new__(process.ActualWorkerLease)
    config = dict(schema=VERSION, preflight=dict(style="pinch", profile="approach_wiring_smoke_v1"),
                  model_config={}, episode_cap_s=2., rollout_steps=32)
    output = tmp_path / "native"
    lease._config = config
    lease._reply = SimpleNamespace(role="a_fresh", checkpoint_origin=None)
    lease._envelope = SimpleNamespace(registered_intent={}, device_plan={"logical_device": "cuda:0"},
                                      nonce="fixture")
    lease._repository = SimpleNamespace(root=tmp_path)
    lease._package = {"fixture": True}
    monkeypatch.setattr(process.ActualWorkerLease, "revalidate", lambda _: None)
    fake_recipe = dict(spec={"teacher_contract": {}, "budget": {"physics_steps": 258}})
    monkeypatch.setattr(process, "_registered", lambda *_: ("a", fake_recipe, {}, {}, {"output": str(output)}))
    return lease, config, output


@pytest.mark.parametrize("change", ["different_output", "existing_output", "symlink", "config", "role"])
def test_lease_binds_fresh_role_config_and_canonical_output(lease_fixture, change):
    lease, config, output = lease_fixture
    if change == "different_output":
        output = output.with_name("other")
    if change == "existing_output":
        output.mkdir()
    if change == "symlink":
        output.symlink_to(output.with_name("different"), target_is_directory=True)
    if change == "config":
        config = dict(config, rollout_steps=32.)
    if change == "role":
        lease._reply.role = "b_resume"
    with pytest.raises((ValueError, FileExistsError), match="role|output|config"):
        provider()._verify_attempt(config, lease, output)


@pytest.mark.parametrize("failure", [None, "update", "close", "both"])
def test_acquisition_uses_one_journal_and_releases_collector_on_every_return(tmp_path, monkeypatch, failure):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_factories as factories
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_update as update
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_session as session

    import hashlib
    import torch

    events = []
    initial_rng = torch.get_rng_state().clone()
    exploration_state = torch.Generator(device="cpu").manual_seed(2026112002).get_state()
    attempt = SimpleNamespace(project=tmp_path, output=tmp_path, session_uid="cpu-composition",
                              device={"logical_device": "cuda:0"})
    sim, owners, trace, lease, owner, static = (object() for _ in range(6))

    def build(state, **kwargs):
        assert state is static
        assert kwargs["sim"] is sim and kwargs["owners"] is owners and kwargs["ownership"] is trace
        assert kwargs["episode_uid"] == "cpu-composition:0"
        state_journal = kwargs["ledger"]
        assert not state_journal._stream.closed and state_journal.path.name == "training_events.jsonl"
        events.append(("scene", state_journal))
        return owner

    def compose(**kwargs):
        assert kwargs["lease"] is lease
        assert callable(kwargs["prepare_source_pack"])
        return kwargs["episode_factory"]

    def create(config, make_episode):
        assert config.journal_path == tmp_path / "training_events.jsonl"
        assert make_episode(static) is owner
        journal = events[-1][1]
        journal.emit("owned_cpu_fixture", {"native_qualification": False})

        def close():
            events.append(("close", journal))
            assert not journal._stream.closed
            if failure in ("close", "both"):
                raise RuntimeError("collector close failure")

        return SimpleNamespace(owner=owner, journal=journal, close=close, _epoch={"weights": "c" * 64})

    def step(collector, *, checkpoint_contract):
        assert checkpoint_contract == {"fixture": True}
        events.append(("update", collector.journal))
        entries = [json.loads(line) for line in collector.journal.path.read_text().splitlines()]
        seed_records = [entry["payload"] for entry in entries if entry["operation"] == "fresh_policy_rng"]
        assert len(seed_records) == 1
        seed = seed_records[0]
        assert seed["initialization_seed"] == 2026112001
        assert seed["exploration_seed"] == 2026112002
        assert seed["initial_weights_sha256"] == "c" * 64
        assert seed["deterministic_physics_claimed"] is False
        assert seed["exploration_state_sha256"] == hashlib.sha256(exploration_state.numpy().tobytes()).hexdigest()
        assert seed["native_qualification"] is False and seed["teacher_admission"] is False
        assert torch.equal(torch.get_rng_state(), exploration_state)
        torch.rand(5)
        if failure in ("update", "both"):
            raise ValueError("original update failure")
        return {"updates": 1, "native_qualification": False}

    monkeypatch.setattr(session, "build_fresh_pinch_owner", build)
    monkeypatch.setattr(session, "create_native_collector", create)
    monkeypatch.setattr(factories, "build_actual_factories", compose)
    monkeypatch.setattr(update, "run_fresh_update", step)
    monkeypatch.setattr(provider(), "_checkpoint_contract", lambda *_: {"fixture": True})

    def record(collector, given_lease, phase):
        assert given_lease is lease
        events.append((phase, collector.journal))

    monkeypatch.setattr(provider(), "_record_actual_wall_budget", record)
    config = {"preflight": {}, "model_config": {}, "episode_cap_s": 2., "rollout_steps": 32}
    if failure in ("update", "both"):
        with pytest.raises(ValueError, match="original update failure"):
            provider()._acquire(sim, owners, trace, attempt=attempt, config=config, lease=lease)
    elif failure == "close":
        with pytest.raises(RuntimeError, match="collector close failure"):
            provider()._acquire(sim, owners, trace, attempt=attempt, config=config, lease=lease)
    else:
        assert provider()._acquire(sim, owners, trace, attempt=attempt, config=config, lease=lease) == {
            "updates": 1, "native_qualification": False}
    expected = ["scene", "before_update", "update"]
    if failure not in ("update", "both"):
        expected.append("after_update")
    assert [name for name, _ in events] == [*expected, "close"]
    assert all(log is events[0][1] and log._stream.closed for _, log in events)
    saved = [json.loads(line) for line in Path(events[0][1].path).read_text().splitlines()]
    assert [row["operation"] for row in saved] == ["owned_cpu_fixture", "fresh_policy_rng"]
    assert saved[0]["payload"]["native_qualification"] is False
    assert torch.equal(torch.get_rng_state(), initial_rng)


def test_run_a_keeps_python_close_distinct_from_outer_exit(lease_fixture, monkeypatch):
    lease, config, output = lease_fixture
    result = {"updates": 1, "native_qualification": True, "teacher_admission": True}

    def run(attempt, supplied_config, supplied_lease):
        assert attempt.output == output and output.is_dir() and not list(output.iterdir())
        assert supplied_config is config and supplied_lease is lease
        return {"acquisition": result, "close_mode": "normal", "failure": None}

    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_artifacts as artifacts

    def derive(*, spec, output):
        assert spec["budget"]["physics_steps"] == 258
        assert output.is_dir() and not list(output.iterdir())
        return {"native_qualification": False}

    monkeypatch.setattr(provider(), "_run_isaac", run)
    monkeypatch.setattr(artifacts, "derive_actual_artifacts", derive)
    assert provider().run_a(config=config, lease=lease, output=output) == {
        "updates": 1, "process_exit_verified": False,
        "native_qualification": False, "teacher_admission": False,
    }
    assert {path.name for path in output.iterdir()} == {"owned_update.json"}


def test_run_a_writes_no_success_marker_when_lifecycle_raises(lease_fixture, monkeypatch):
    lease, config, output = lease_fixture

    def fail(attempt, config, lease):
        (attempt.output / "runtime_failure.txt").write_text("CPU fault injection")
        raise RuntimeError("R06 close failed")

    monkeypatch.setattr(provider(), "_run_isaac", fail)
    with pytest.raises(RuntimeError, match="R06 close failed"):
        provider().run_a(config=config, lease=lease, output=output)
    assert {path.name for path in output.iterdir()} == {"runtime_failure.txt"}


@pytest.mark.parametrize("mode,failure", [("fault", None), ("normal", "SDK shutdown failure")])
def test_run_a_rejects_an_incomplete_close_record(lease_fixture, monkeypatch, mode, failure):
    lease, config, output = lease_fixture
    monkeypatch.setattr(provider(), "_run_isaac", lambda *_: {
        "acquisition": {"updates": 1}, "close_mode": mode, "failure": failure})
    with pytest.raises(RuntimeError, match="R06.*normal"):
        provider().run_a(config=config, lease=lease, output=output)
    assert not list(output.iterdir())
