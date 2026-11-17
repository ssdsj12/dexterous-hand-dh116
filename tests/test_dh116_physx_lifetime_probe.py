"""CPU-only lifecycle/control/GDB preparation, never Isaac execution."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    assert (ROOT / "scripts/probe_dh116_physx_lifetime.py").exists(), "versioned lifecycle probe missing"
    return importlib.import_module("scripts.probe_dh116_physx_lifetime")


def fixture(output, *, fail=False, close_fail=False):
    probe = load()
    events, refs = [], {}
    class Sim:
        current_time_step_index = 0
        current_time = 0.
        def reset(self):
            self.current_time_step_index += 2
            self.current_time += 2 / 480
    sim = Sim()
    class Owner:
        def reset(self):
            events.append("composer_reset")
    def acquire(sim, owners, trace, cleanup):
        probe.record_reset(sim, output)
        for name in probe.OWNED_NAMES:
            obj = owners.keep(name, Owner())
            refs[name] = weakref.ref(obj)
            trace.watch(name, obj, owner="CPU fixture", release="scope")
        trace.checkpoint("resources_acquired")
        events.append("acquire:" + cleanup)
        if fail:
            raise RuntimeError("CPU acquisition failure")
        return {"rod_steps": 0, "measurement_steps": 0, "ppo_updates": 0, "case": cleanup}
    def close():
        events.append("close")
        if close_fail:
            raise RuntimeError("CPU close failure")
    app = SimpleNamespace(update=lambda: events.append("update"), close=close)
    kwargs = dict(make_sim=lambda: sim, acquire=acquire,
                  synchronize=lambda: events.append("sync"),
                  clear_sim=lambda value: events.append("clear"),
                  stop=lambda: events.append("stop"), manager_is_none=lambda: True,
                  freeze=lambda: events.append("freeze") or False,
                  output=output, cleanup="registry")
    return probe, app, kwargs, events, refs


def test_real_scope_orders_sync_drop_stop_update_close_and_records_two_boundaries(tmp_path):
    p, app, kwargs, events, refs = fixture(tmp_path)
    result = p.run_lifecycle(app, **kwargs)
    assert events.index("sync") < events.index("clear") < events.index("stop") < events.index("update") < events.index("close")
    assert events.count("update") == events.count("close") == 1
    assert all(ref() is None for ref in refs.values())
    rows = [json.loads(line) for line in (tmp_path / "initialization_boundaries.jsonl").read_text().splitlines()]
    assert [row["sdk_step"] for row in rows] == [0, 2]
    assert [row["phase"] for row in rows] == ["before_reset", "after_reset"]
    assert result["physics_steps"] == 2 and result["rod_steps"] == 0
    assert result["process_exit_verified"] is False
    assert (tmp_path / "app_close_returned.json").exists()


def test_acquisition_exception_preserves_failure_and_closes_without_owned_refs(tmp_path):
    p, app, kwargs, events, refs = fixture(tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="CPU acquisition failure"):
        p.run_lifecycle(app, **kwargs)
    assert events[-1] == "close"
    assert all(ref() is None for ref in refs.values())
    assert json.loads((tmp_path / "result.json").read_text())["failure"] is not None


def test_close_error_has_no_success_marker(tmp_path):
    p, app, kwargs, events, refs = fixture(tmp_path, close_fail=True)
    with pytest.raises(RuntimeError, match="CPU close failure"):
        p.run_lifecycle(app, **kwargs)
    assert not (tmp_path / "app_close_returned.json").exists()


def test_reset_budget_rejects_hidden_steps(tmp_path):
    p = load()
    sim = SimpleNamespace(current_time_step_index=0, current_time=0.)
    def reset():
        sim.current_time_step_index = 3
        sim.current_time = 3/480
    sim.reset = reset
    with pytest.raises(RuntimeError, match="two"):
        p.record_reset(sim, tmp_path)


def test_native_cli_rejects_generic_entry_before_mkdir_or_sdk_import(tmp_path):
    p = load()
    output = tmp_path / "native"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop(p.WORKER_ENV, None)
    result = subprocess.run([sys.executable, "-m", "scripts.probe_dh116_physx_lifetime",
                             "--cleanup", "registry", "--output", str(output), "--execute-native", "--headless"],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert "dedicated" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("fatal", [False, True])
def test_real_cpu_gdb_records_registers_maps_and_independent_status(tmp_path, fatal):
    p = load()
    from scripts.dh116_physx_lifetime_debugger import render_gdb_commands
    from scripts.run_dh116_work_packages import debugger_status
    output = tmp_path / "capture"; output.mkdir()
    fixture_script = tmp_path / "fixture.py"
    fixture_script.write_text("import os, signal\n" +
                              ("os.kill(os.getpid(), signal.SIGSEGV)\n" if fatal else "print('CPU fixture only')\n"))
    commands = tmp_path / "debugger.gdb"
    commands.write_text(render_gdb_commands(output))
    done = subprocess.run(["/usr/bin/gdb", "-nx", "-q", "--batch", "--return-child-result",
                           "-x", str(commands), "--args", sys.executable, str(fixture_script)],
                          text=True, capture_output=True, timeout=25)
    log = tmp_path / "native.log"; log.write_text(done.stdout + done.stderr)
    status = debugger_status(log)
    assert status["debugger_status_verified"] is True
    assert status["debugger_inferior_signal"] == (11 if fatal else None)
    assert status["debugger_inferior_exit_code"] == (None if fatal else 0)
    phase = "fatal_1" if fatal else "finalize"
    result = json.loads((output / ("gdb_" + phase + ".json")).read_text())
    assert result["all_commands_succeeded"] is True
    assert "rip" in (output / ("gdb_" + phase + ".registers.txt")).read_text()
    assert "r-xp" in (output / ("gdb_" + phase + ".maps")).read_text()
    assert result["process_exit_verified"] is False
    from scripts.validate_dh116_physx_lifetime import capture_evidence
    reopened=capture_evidence(output,phase)
    assert reopened["program_counter"]["pc"].startswith("0x")


@pytest.mark.parametrize("cleanup", ["registry", "explicit"])
@pytest.mark.parametrize("retain_frame", [False, True])
def test_only_explicit_scope_detaches_a_retained_completed_acquisition_frame(tmp_path, cleanup, retain_frame):
    import gc
    import numpy as np
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
    p = load()
    frames, refs = [], {}
    class Resource:
        def reset(self):
            pass
    class Hand(Resource):
        @property
        def data(self):
            if retain_frame:
                frames.append(sys._getframe(1))
            return SimpleNamespace(joint_pos=np.zeros((1, 6)), joint_vel=np.zeros((1, 6)),
                                   body_link_pos_w=np.zeros((1, 5, 3)))
    def build(sim):
        hand, view, composer = Hand(), Resource(), Resource()
        hand.view, hand.composer = view, composer
        view.getter, composer.asset = lambda: hand, hand
        shared = Resource()
        values = dict(hand=hand, hand_view=view, composer=composer, asset_sim_view=shared,
                      manager_sim_view=shared, runtime=Resource(), provider=Resource(), rope=Resource())
        for name, value in values.items():
            refs[name] = weakref.ref(value)
        return SimpleNamespace(**values, metadata={"rod_steps": 0, "measurement_steps": 0,
                                                   "ppo_updates": 0, "asset_sim_view_is_manager": True})
    trace = p.LifetimeTrace(tmp_path / "ownership.jsonl")
    assert hasattr(p, "acquire_graph"), "shared real-ownership acquisition boundary missing"
    run_owned_call(lambda owners: p.acquire_graph(None, owners, trace, cleanup, build),
                   synchronize=lambda: None, ownership=trace)
    actual_live = any(ref() is not None for ref in refs.values())
    assert actual_live is (retain_frame and cleanup == "registry")
    frames.clear()
    gc.collect()
    assert all(ref() is None for ref in refs.values())


def test_native_builder_has_one_reset_boundary_and_no_measured_solver_calls():
    import ast
    path = ROOT / "scripts/dh116_physx_lifetime_acquisition.py"
    assert path.exists(), "shared native resource builder is missing"
    tree = ast.parse(path.read_text())
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_native")
    names = [n.func.id for n in ast.walk(build) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert names.count("record_reset") == 1
    calls = [n.func.attr for n in ast.walk(build) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not set(calls) & {"step", "step_free", "render", "update_simulation", "play"}
    assert {"Articulation", "FreeRodRuntime", "DeformableRodStateProvider"} <= set(names)
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(not any(word in ast.unparse(n) for word in ("isaac", "omni", "torch", "ropesim")) for n in top_imports)
