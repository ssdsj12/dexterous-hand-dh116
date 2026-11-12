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
MODULE = "scripts.probe_dh116_camera_c5_ownership"
OWNER = "omni.kit.app._impl.telemetry_helpers"


def load_probe():
    assert (ROOT / "scripts/probe_dh116_camera_c5_ownership.py").is_file(), "C5 ownership probe missing"
    return importlib.import_module(MODULE)


def holder_pair(events):
    cls = type("IHookHolder", (), {
        "__module__": "omni.ext._extensions",
        "__del__": lambda self: events.append("holder_released"),
    })
    return (cls(), cls())


def test_observation_does_not_import_or_acquire_or_retain_holder(monkeypatch):
    probe = load_probe()
    monkeypatch.delitem(sys.modules, OWNER, raising=False)
    assert probe.observe_telemetry_hook()["module_present"] is False
    assert OWNER not in sys.modules
    events = []
    owner = SimpleNamespace(_telemetry_hook=holder_pair(events))
    refs = [weakref.ref(item) for item in owner._telemetry_hook]
    monkeypatch.setitem(sys.modules, OWNER, owner)
    observed = probe.observe_telemetry_hook()
    assert observed["hook_types"] == ["omni.ext._extensions.IHookHolder"] * 2
    json.dumps(observed)
    owner._telemetry_hook = None
    assert all(ref() is None for ref in refs)
    assert events == ["holder_released"] * 2


def test_release_changes_only_existing_owner_before_close(monkeypatch, tmp_path):
    probe = load_probe()
    events = []
    untouched = object()
    owner = SimpleNamespace(_telemetry_hook=holder_pair(events), untouched=untouched)
    monkeypatch.setitem(sys.modules, OWNER, owner)

    class App:
        def update(self):
            events.append("update")
        def close(self):
            assert owner._telemetry_hook is None
            events.append("close")

    probe.run_lifecycle(App(), tmp_path, snapshot=lambda app, case, phase: {"phase": phase})
    assert events == ["update", "holder_released", "holder_released", "close"]
    assert owner.untouched is untouched
    before = json.loads((tmp_path / "before_hook_release.json").read_text())
    assert before["hook_present"] is True
    result = json.loads((tmp_path / "app_close_returned.json").read_text())
    assert result["owner_release_applied"] is True
    assert result["hook_absent_after_close"] is True
    assert result["process_exit_verified"] is False
    for phase in ("before_explicit_update", "before_hook_release", "before_app_close", "after_app_close"):
        assert "[stack]" in (tmp_path / (phase + ".maps")).read_text()


@pytest.mark.parametrize("value", [None, (), object(), [object(), object()]])
def test_invalid_owner_never_qualifies_but_closes(monkeypatch, tmp_path, value):
    probe = load_probe()
    owner = SimpleNamespace(_telemetry_hook=value)
    monkeypatch.setitem(sys.modules, OWNER, owner)
    events = []
    app = SimpleNamespace(update=lambda: events.append("update"), close=lambda: events.append("close"))
    with pytest.raises(RuntimeError, match="existing telemetry IHookHolder pair"):
        probe.run_lifecycle(app, tmp_path, snapshot=lambda *args: {})
    assert events == ["update", "close"]
    assert owner._telemetry_hook is value
    result = json.loads((tmp_path / "app_close_returned.json").read_text())
    assert result["owner_release_applied"] is False


def test_reacquired_hook_cannot_qualify(monkeypatch, tmp_path):
    probe = load_probe()
    owner = SimpleNamespace(_telemetry_hook=holder_pair([]))
    monkeypatch.setitem(sys.modules, OWNER, owner)
    def close():
        owner._telemetry_hook = holder_pair([])
    with pytest.raises(RuntimeError, match="reacquired"):
        probe.run_lifecycle(SimpleNamespace(update=lambda: None, close=close),
                            tmp_path, snapshot=lambda *args: {})
    result = json.loads((tmp_path / "app_close_returned.json").read_text())
    assert result["hook_absent_after_close"] is False


def test_close_failure_does_not_write_return_marker(monkeypatch, tmp_path):
    probe = load_probe()
    monkeypatch.setitem(sys.modules, OWNER, SimpleNamespace(_telemetry_hook=holder_pair([])))
    def close():
        raise RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="close failed"):
        probe.run_lifecycle(SimpleNamespace(update=lambda: None, close=close),
                            tmp_path, snapshot=lambda *args: {})
    assert (tmp_path / "before_app_close.maps").is_file()
    assert not (tmp_path / "app_close_returned.json").exists()


def test_plan_keeps_exact_c4_launch_and_never_loads_sdk(tmp_path):
    probe = load_probe()
    from scripts import probe_dh116_simulation_app_lifecycle as base
    output = tmp_path / "plan"
    subprocess.run([sys.executable, "-m", MODULE, "--output", str(output)], check=True)
    plan = json.loads((output / "plan.json").read_text())
    config, options = base.launch_options("C4")
    assert plan["launch_config"] == config
    assert plan["launch_options"] == options
    assert plan["preloaded_runtime_modules"] == []
    assert plan["native_execution_performed"] is False
    assert plan["explicit_application_updates"] == 1
    assert plan["physics_steps"] == plan["ppo_updates"] == 0


def test_plain_scheduler_cannot_launch_c5_without_its_capture(tmp_path):
    probe = load_probe()
    env = dict(os.environ)
    env.pop("DH116_CAMERA_C5_GDB", None)
    result = subprocess.run([sys.executable, "-m", MODULE, "--output", str(tmp_path / "bad"),
                             "--execute-native"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "C5 debugger capture required" in result.stderr


def test_gdb_normal_and_crash_capture_full_maps_without_native_run(tmp_path, monkeypatch):
    script = ROOT / "scripts/dh116_camera_c5_debugger.py"
    assert script.is_file(), "C5 GDB capture missing"
    source = importlib.import_module("scripts.dh116_camera_c5_debugger").GDB_COMMANDS.split("\npython\n", 1)[1].split("\nend\n", 1)[0]
    output = tmp_path / "native"
    output.mkdir()
    (output / "before_app_close.json").write_text('{"case": "C5"}')
    (output / "app_close_returned.json").write_text('{"app_close_returned": true}')
    log, commands, callbacks = [], [], []
    class Breakpoint:
        def __init__(self, symbol, **kwargs):
            assert symbol == "Py_FinalizeEx"
    class SignalEvent:
        stop_signal = "SIGSEGV"
    fake = SimpleNamespace(Breakpoint=Breakpoint, SignalEvent=SignalEvent,
        events=SimpleNamespace(stop=SimpleNamespace(connect=callbacks.append)),
        selected_inferior=lambda: SimpleNamespace(pid=os.getpid()),
        parameter=lambda name: 'probe.py --output ' + str(output) + ' --execute-native',
        write=log.append, execute=commands.append, error=RuntimeError)
    monkeypatch.setitem(sys.modules, "gdb", fake)
    scope = {}
    exec(compile(source, str(script), "exec"), scope)
    assert scope["finalize_breakpoint"].stop() is False
    normal = json.loads((output / "gdb_finalize.json").read_text())
    assert normal["all_commands_succeeded"] is True
    assert normal["before_app_close"]["case"] == "C5"
    assert normal["app_close_returned"]["app_close_returned"] is True
    assert normal["full_proc_maps_bytes"] > 0
    assert "[stack]" in (output / "gdb_finalize.maps").read_text()
    assert commands == ["bt 32", "thread apply all bt 20", "info sharedlibrary", "info proc mappings"]
    callbacks[0](SignalEvent())
    assert (output / "gdb_fatal_1.maps").is_file()
    assert any("DH116_GDB_FATAL_BACKTRACE SIGSEGV" in line for line in log)


def test_gdb_failure_is_not_success_evidence(tmp_path, monkeypatch):
    script = ROOT / "scripts/dh116_camera_c5_debugger.py"
    assert script.is_file(), "C5 GDB capture missing"
    source = importlib.import_module("scripts.dh116_camera_c5_debugger").GDB_COMMANDS.split("\npython\n", 1)[1].split("\nend\n", 1)[0]
    class Breakpoint:
        def __init__(self, *args, **kwargs):
            pass
    def fail(command):
        raise RuntimeError("unavailable")
    fake = SimpleNamespace(Breakpoint=Breakpoint, SignalEvent=type("SignalEvent", (), {}),
        events=SimpleNamespace(stop=SimpleNamespace(connect=lambda fn: None)),
        selected_inferior=lambda: SimpleNamespace(pid=os.getpid()),
        parameter=lambda name: "probe.py --output " + str(tmp_path),
        write=lambda line: None, execute=fail, error=RuntimeError)
    monkeypatch.setitem(sys.modules, "gdb", fake)
    scope = {}
    exec(compile(source, str(script), "exec"), scope)
    scope["finalize_breakpoint"].stop()
    assert json.loads((tmp_path / "gdb_finalize.json").read_text())["all_commands_succeeded"] is False


def test_c5_recipe_requires_normal_capture_and_unchanged_c4_config():
    from scripts import run_dh116_work_packages as scheduler
    path = ROOT / "configs/dh116_execution/deformx_camera_c5.json"
    assert path.is_file(), "C5 recipe missing"
    recipe = json.loads(path.read_text())
    c4 = json.loads((path.parent / "deformx_camera_c4.json").read_text())
    assert recipe["budget"] == c4["budget"]
    assert recipe["environment"] == c4["environment"]
    assert recipe["validator"]["result_file"] == "gdb_finalize.json"
    assert recipe["validator"]["equals"]["all_commands_succeeded"] is True
    assert recipe["validator"]["equals"]["full_proc_maps_present"] is True
    assert recipe["validator"]["equals"]["app_close_returned.owner_release_applied"] is True
    assert recipe["validator"]["equals"]["app_close_returned.hook_absent_after_close"] is True
    package = {"command_spec": str(path.relative_to(ROOT)), "tests": []}
    identity = scheduler.native_recipe(scheduler.Repository(ROOT), package)
    assert identity["code_config_sha256"]
    assert "scripts/run_dh116_camera_c5_work_packages.py" in recipe["source_files"]


def test_c5_wrapper_uses_versioned_capture_without_editing_base(monkeypatch):
    from scripts import run_dh116_camera_c5_work_packages as wrapper
    original = wrapper.scheduler.GDB_COMMANDS
    # monkeypatch restores the shared module after this test.
    monkeypatch.setattr(wrapper.scheduler, "GDB_COMMANDS", original)
    def check():
        assert wrapper.scheduler.GDB_COMMANDS == importlib.import_module("scripts.dh116_camera_c5_debugger").GDB_COMMANDS
        return 17
    monkeypatch.setattr(wrapper.scheduler, "main", check)
    assert wrapper.main() == 17
