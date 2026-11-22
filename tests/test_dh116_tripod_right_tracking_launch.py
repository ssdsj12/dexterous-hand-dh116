from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts import launch_dh116_tripod_right_tracking as launch
from scripts import record_dh116_teacher_numeric_reset as reset


def test_worker_needs_exact_frozen_plan_before_lifecycle(tmp_path, monkeypatch):
    output = tmp_path / "native"
    expected = dict(project="fixture", output=str(output), frozen_files={})
    monkeypatch.setattr(launch, "freeze_plan", lambda path, tracking: expected)
    monkeypatch.setattr(launch.native, "validate_execution_context", lambda *a: None)
    calls = []
    monkeypatch.setattr(reset, "execute", lambda *a, **k: calls.append((a, k)))
    plan = tmp_path / "plan.json"
    args = ["--output", str(output), "--tracking-plan", str(tmp_path / "reference.json"),
            "--plan", str(plan)]
    assert launch.main(args) == 0
    with pytest.raises(ValueError, match="frozen"):
        launch.main([*args, "--approved-frozen-sha256", "0" * 64])
    assert calls == []


def test_claim_is_exclusive_per_output(tmp_path):
    output = tmp_path / "native"
    path = launch.claim_attempt(output, "exact", ["python", "-m", "worker"])
    assert path.is_file()
    with pytest.raises(FileExistsError):
        launch.claim_attempt(output, "exact", ["python", "-m", "worker"])


def test_cache_uses_measured_final_sdk_step_and_old_default(monkeypatch):
    calls = []
    monkeypatch.setattr(reset, "_acquire", lambda *a, **k: {"legacy": True})
    monkeypatch.setattr(reset.native, "clock", lambda sim: {"sdk_step": sim.step})
    monkeypatch.setattr(reset.native, "validate_loaded_project_modules", lambda *a, **k: None)
    cache = lambda sim, *, expected_sdk_step: calls.append(expected_sdk_step)
    plan = dict(budget=dict(sdk_steps=178), frozen_files={})
    old = reset._run_acquisition(SimpleNamespace(step=2), None, None, None, None, plan, cache)
    assert old == {"legacy": True}
    result = {"physics_rows": 176}
    operation = lambda *a, **k: result
    assert reset._run_acquisition(SimpleNamespace(step=178), None, None, None, None,
                                  plan, cache, operation) == result
    with pytest.raises(ValueError, match="clock"):
        reset._run_acquisition(SimpleNamespace(step=177), None, None, None, None,
                               plan, cache, operation)
    assert calls == [2, 178]


@pytest.mark.parametrize("code,elapsed,reason", [
    (1, 40., "worker failed"),
    (-11, 40., "worker failed"),
    (0, 181., "wall"),
    (0, 40., "natural-close"),
])
def test_outer_does_not_promote_close_or_failure(code, elapsed, reason, tmp_path):
    with pytest.raises(ValueError, match=reason):
        launch.validate_outer(tmp_path, exit_code=code, wall_seconds=elapsed)


def test_known_guard_stop_cleans_observed_step_and_preserves_primary_failure(monkeypatch):
    monkeypatch.setattr(reset.native, "clock", lambda sim: {"sdk_step": sim.step})
    monkeypatch.setattr(reset.native, "validate_loaded_project_modules", lambda *a, **k: None)
    plan = dict(budget=dict(sdk_steps=178), frozen_files={})
    result = dict(failure="quiet failed", guarded_stop_sdk_step=10)
    calls = []

    def bad_cache(sim, *, expected_sdk_step):
        calls.append(expected_sdk_step)
        raise RuntimeError("synthetic cache error")

    received = reset._run_acquisition(SimpleNamespace(step=10), None, None, None, None,
                                      plan, bad_cache, lambda *a: result)
    assert calls == [10]
    assert received["failure"].splitlines() == ["quiet failed", "stat cache cleanup: RuntimeError: synthetic cache error"]
    for stopped in (1, 179):
        with pytest.raises(ValueError, match="outside"):
            reset._run_acquisition(SimpleNamespace(step=stopped), None, None, None, None,
                plan, bad_cache, lambda *a: dict(failure="guard stop", guarded_stop_sdk_step=stopped))


@pytest.mark.parametrize("filename,key", [
    ("ag1800-tripod-right-native-preparation.json", "TRACKING_SHA"),
    ("ag1900-tripod-straight-tracking-plan.json", "STRAIGHT_TRACKING_SHA"),
])
def test_frozen_launch_keeps_curved_and_straight_provenance(filename, key, tmp_path, monkeypatch):
    evidence = Path.home() / ".local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119"
    tracking = evidence / filename
    monkeypatch.setattr(reset, "freeze_plan", lambda output: {"frozen_files": {}, "budget": {}})
    plan = launch.freeze_plan(tmp_path / "native", tracking)
    digest = getattr(launch, key)
    assert plan["tracking_plan"] == dict(path=str(tracking.resolve()), sha256=digest)
    assert plan["frozen_files"][str(tracking.resolve())] == digest
    assert plan["budget"]["sdk_steps"] == 178
    rejected = tmp_path / "changed.json"
    rejected.write_bytes(tracking.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA|approved"):
        launch.freeze_plan(tmp_path / "native", rejected)


def test_measured_settled_thumb_stop_admits_early_boundary_for_cache_release(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(reset.native, "clock", lambda sim: {"sdk_step": sim.step})
    monkeypatch.setattr(reset.native, "validate_loaded_project_modules", lambda *a, **k: None)
    plan = dict(version="dh116-wire-settled-left-thumb-approach-v1", quiet_settlement_sdk=1426,
                budget=dict(mode="quiet_wire_then_left_thumb_transfer", sdk_steps=2210,
                            solver_steps=26496, control_steps=98), frozen_files={})
    motion = dict(first_contact_sdk=2167, first_transfer_sdk=2168, contact_reached=True,
                  start_sdk=1426, end_sdk=2168, control_steps=93, teacher_admission=False,
                  grasp_qualified=False,
                  readback=dict(clock=dict(clock_valid=True, physics_step=2168, rod_tick=25992)))
    proof = dict(first_contact_sdk=2167, first_transfer_sdk=2168,
                 teacher_admission=False, grasp_qualified=False)
    (tmp_path/"settled_thumb_transfer.json").write_text(json.dumps(proof))
    result = dict(scope="measured_settled_wire_left_thumb", sdk_steps=2168, solver_substeps=25992,
                  measured_exchanges=2166, control_steps=93, quiet_settlement=dict(qualified=True, last_sdk=1426),
                  approach_feedback=motion)
    calls = []

    def run(actual_sdk=2168):
        return reset._run_acquisition(SimpleNamespace(step=actual_sdk), None, None, tmp_path, None,
            plan, lambda sim, *, expected_sdk_step: calls.append(expected_sdk_step), lambda *a: result)

    assert run() == result
    assert calls == [2168]
    with pytest.raises(ValueError, match="clock"):
        run(2167)
    motion["first_contact_sdk"] = 2166
    with pytest.raises(ValueError, match="transfer"):
        run()
    motion["first_contact_sdk"] = 2167
    motion["readback"]["clock"]["rod_tick"] -= 12
    with pytest.raises(ValueError, match="transfer"):
        run()
    motion["readback"]["clock"]["rod_tick"] += 12
    (tmp_path/"settled_thumb_transfer.json").unlink()
    with pytest.raises(ValueError, match="transfer"):
        run()
    assert calls == [2168, 2167, 2168, 2168, 2168]
