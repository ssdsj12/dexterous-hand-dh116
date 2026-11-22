"""Zero-command counterfactual reuses the real episode driver; CPU fixtures only."""
from copy import deepcopy
import importlib
import importlib.util
import numpy as np
import pytest

from test_fresh_supported_episode import episode_case


def entry():
    name = "scripts.verify_dh116_tripod_zero_action_settling"
    assert importlib.util.find_spec(name) is not None, "missing bounded no-Actor settling diagnostic"
    return importlib.import_module(name)


def case(monkeypatch, tmp_path):
    c, _, plan, episode, rows, readback = episode_case(monkeypatch, tmp_path)
    episode.reset(c.seed, episode_uid="cpu17:settling", owner_resets=c.owners,
                  supported_mesh_plan=plan, record_reset_readback=readback.append)
    return c, episode, rows


def test_budget_separates_diagnostic_prime_from_control_and_policy():
    budget = entry().budget(controls=30, wall_seconds=360)
    assert budget["sdk_steps"] == 243
    assert budget["solver_steps"] == 241 and budget["solver_substeps"] == 2892
    assert budget["diagnostic_control_windows"] == 30 and budget["priming_sdk_steps"] == 1
    for key in ("policy_windows", "teacher_queries", "model_loads", "ppo_updates", "auto_retries"):
        assert budget[key] == 0
    assert budget["auto_kill"] is False


@pytest.mark.parametrize("controls", [0, -1, True, 1.5, 31])
def test_invalid_or_unbounded_diagnostic_controls_rejected(controls):
    with pytest.raises(ValueError, match="control"):
        entry().budget(controls=controls, wall_seconds=360)


def test_real_episode_prime_updates_boundary_without_actor_tokens_and_restores_step(monkeypatch, tmp_path):
    m = entry()
    c, episode, rows = case(monkeypatch, tmp_path)
    raw, enriched = [], []
    before = episode._exchange._step
    result = m.run_diagnostic(episode, controls=1, raw_record=raw.append,
        enrich=lambda row: row.update(cpu_fixture=True), physical_record=enriched.append,
        remaining_s=lambda: 20.)
    assert result["failure"] is None
    assert result["sdk_steps"] == 11 and result["solver_steps"] == 9
    assert result["solver_substeps"] == 108 and result["diagnostic_control_windows"] == 1
    assert len(raw) == len(enriched) == 9
    assert c.rope.steps == 9
    assert episode.pre_action_state()["common"]["identity"]["physics_step"] == 11
    assert episode._exchange._step == before
    for row in raw:
        assert np.count_nonzero(row["action_readback"]["request"]["action24"]) == 0
    assert [row["kind"] for row in rows].count("diagnostic_priming_physical") == 1
    assert result["on_policy"] is False and result["teacher_admission"] is False
    assert result["ppo_updates"] == result["policy_windows"] == 0


@pytest.mark.parametrize("fault", ["boundary", "enrichment", "native_after_progress", "raw_sink"])
def test_failed_advanced_step_is_charged_preserved_and_not_replayed(monkeypatch, tmp_path, fault):
    m = entry()
    c, episode, rows = case(monkeypatch, tmp_path)
    raw, enriched = [], []
    def fail(*args, **kwargs):
        raise RuntimeError("injected " + fault)
    if fault == "boundary":
        monkeypatch.setattr(episode._boundary, "capture", fail)
    if fault == "native_after_progress":
        original = episode._exchange._step
        def step(action, substep):
            original(action, substep)
            fail()
        episode._exchange._step = step
    before = episode._exchange._step
    result = m.run_diagnostic(episode, controls=1,
        raw_record=fail if fault == "raw_sink" else raw.append,
        enrich=fail if fault == "enrichment" else lambda row: None,
        physical_record=enriched.append, remaining_s=lambda: 20.)
    assert fault in result["failure"]
    assert result["sdk_steps"] == 3 and result["solver_steps"] == 1
    assert result["solver_substeps"] == 12 and c.rope.steps == 1
    assert episode._state == "failed" and episode._exchange._step == before
    assert result["teacher_admission"] is False
    if fault not in ("native_after_progress", "raw_sink"):
        assert len(raw) == 1
    if fault == "enrichment":
        assert len(enriched) == 1 and "diagnostic_source_error" in enriched[0]
        assert result["returned_solver_rows"] == 1
        assert result["source_valid_rows"] == 0
        assert result["persisted_physics_rows"] == 1
        assert result["solver_completion_verified"] is True
    if fault == "native_after_progress":
        assert result["returned_solver_rows"] == 0
        assert result["solver_completion_verified"] is False
    again = m.run_diagnostic(episode, controls=1, raw_record=raw.append,
        enrich=lambda row: None, physical_record=enriched.append, remaining_s=lambda: 20.)
    assert again["failure"] and c.rope.steps == 1


def test_original_guard_stops_after_recorded_prime_and_wall_expiry_never_steps(monkeypatch, tmp_path):
    m = entry()
    c, episode, _ = case(monkeypatch, tmp_path)
    raw, enriched = [], []
    result = m.run_diagnostic(episode, controls=1, raw_record=raw.append,
        enrich=lambda row: row.update(guard_failure="original_clearance_guard"),
        physical_record=enriched.append, remaining_s=lambda: 20.)
    assert "original_clearance_guard" in result["failure"]
    assert result["sdk_steps"] == 3 and len(enriched) == c.rope.steps == 1

    c2, episode2, _ = case(monkeypatch, tmp_path)
    result = m.run_diagnostic(episode2, controls=1, raw_record=raw.append,
        enrich=lambda row: None, physical_record=enriched.append, remaining_s=lambda: 0.)
    assert "wall" in result["failure"] and c2.rope.steps == 0
    assert result["sdk_steps"] == 2 and result["solver_steps"] == 0


def test_freeze_retains_real_reset_proof_and_independent_budget_without_native_output(tmp_path):
    from test_fresh_supported_builder import EVIDENCE
    import hashlib
    m = entry()
    manifest = EVIDENCE/"ref902/fresh-supported-tripod-reset-route-order-v3/teacher-reset-manifest.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = m.freeze_plan(tmp_path/"native", manifest_path=manifest, manifest_sha256=digest,
                         controls=30, wall_seconds=360)
    assert plan["diagnostic_scope"] == m.SCOPE and plan["on_policy"] is False
    assert plan["budget"]["mode"] == "tripod_zero_action_settling"
    assert plan["budget"]["sdk_steps"] == 243 and plan["budget"]["solver_substeps"] == 2892
    assert plan["reset_manifest"] == dict(path=str(manifest), sha256=digest)
    assert plan["frozen_files"][str(manifest)] == digest
    assert plan["frozen_files"][str(m.Path(m.__file__).resolve())] == hashlib.sha256(m.Path(m.__file__).read_bytes()).hexdigest()
    for reader in ("tabletop_native_parameter_endpoints.py",
                   "tabletop_native_teacher_policy_tripod_distance.py",
                   "tabletop_native_teacher_solver_health.py",
                   "tabletop_native_teacher_contact_measurement.py"):
        path = m.ROOT/"src/go2_pvcnn/control/dh116"/reader
        assert plan["frozen_files"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not (tmp_path/"native").exists()


def test_approved_settling_cli_cpu_child_keeps_app_process_sdk_free(tmp_path):
    from test_fresh_supported_builder import EVIDENCE
    import hashlib
    import json
    import os
    import subprocess
    import sys
    m = entry()
    manifest = EVIDENCE/"ref902/fresh-supported-tripod-reset-route-order-v3/teacher-reset-manifest.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = tmp_path/"native"
    plan = m.freeze_plan(output, manifest_path=manifest, manifest_sha256=digest, controls=1, wall_seconds=360)
    path = tmp_path/"plan.json"
    path.write_text(json.dumps(plan))
    args = ["--output", str(output), "--plan", str(path), "--reset-manifest", str(manifest),
            "--reset-manifest-sha256", digest, "--controls", "1",
            "--approved-frozen-sha256", m.native.digest(plan)]
    code = """
import importlib.abc,json,sys
from scripts import verify_dh116_tripod_zero_action_settling as m
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split(".")[0] in ("pxr","PhysxSchema","isaaclab","isaacsim","ropesim"):
            raise RuntimeError("pre-App runtime import:"+name)
sys.meta_path.insert(0,Boundary())
def execute(plan,*,acquire_operation):
    assert callable(acquire_operation)
    assert plan["budget"]["model_loads"] == plan["budget"]["ppo_updates"] == 0
    assert not any(n.split(".")[0] in ("pxr","isaaclab","ropesim") for n in sys.modules)
    print("ZERO_ACTION_ENTRY_CLEAN")
m.reset_entry.lifecycle.execute=execute
m.main(json.loads(sys.argv[1]))
"""
    process = subprocess.run([sys.executable,"-B","-c",code,json.dumps(args)],cwd=m.ROOT,
        env=os.environ.copy(),capture_output=True,text=True,timeout=45)
    assert process.returncode == 0, process.stdout+process.stderr
    assert "ZERO_ACTION_ENTRY_CLEAN" in process.stdout and not output.exists()


def test_nonzero_actual_request_is_retained_and_refused_as_zero_counterfactual(monkeypatch, tmp_path):
    m = entry()
    c, episode, _ = case(monkeypatch, tmp_path)
    original = episode._exchange._step
    def changed(action, substep):
        row = original(action, substep)
        row["action_readback"]["request"]["action24"][0, 7] = .1
        return row
    episode._exchange._step = changed
    raw, physical = [], []
    result = m.run_diagnostic(episode, controls=1, raw_record=raw.append,
        enrich=lambda row: None, physical_record=physical.append, remaining_s=lambda: 20.)
    assert "zero" in result["failure"]
    assert len(raw) == c.rope.steps == 1
    assert raw[0]["action_readback"]["request"]["action24"][0, 7] == pytest.approx(.1)
    assert episode._state == "failed"
