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
    output = tmp_path / "native"
    source = importlib.import_module("scripts.dh116_camera_c5_debugger").render_gdb_commands(output).split("\npython\n", 1)[1].split("\nend\n", 1)[0]
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
    source = importlib.import_module("scripts.dh116_camera_c5_debugger").render_gdb_commands(tmp_path).split("\npython\n", 1)[1].split("\nend\n", 1)[0]
    class Breakpoint:
        def __init__(self, *args, **kwargs):
            pass
    def fail(command):
        raise RuntimeError("unavailable")
    fake = SimpleNamespace(Breakpoint=Breakpoint, SignalEvent=type("SignalEvent", (), {}),
        events=SimpleNamespace(stop=SimpleNamespace(connect=lambda fn: None)),
        selected_inferior=lambda: SimpleNamespace(pid=os.getpid()),
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
    original = wrapper.scheduler.native_debugger_wrapper
    original_commands = wrapper.scheduler.GDB_COMMANDS

    def check():
        assert wrapper.scheduler.native_debugger_wrapper.func is wrapper.write_c5_debugger
        assert wrapper.scheduler.GDB_COMMANDS == original_commands
        return 17

    monkeypatch.setattr(wrapper.scheduler, "main", check)
    entry_before = wrapper.scheduler.__file__
    assert wrapper.main() == 17
    assert wrapper.scheduler.__file__ == entry_before
    assert wrapper.scheduler.native_debugger_wrapper is original


def test_real_worker_subprocess_keeps_c5_capture_root_and_private_directory(tmp_path):
    """Run start_native and the actual worker CLI, stubbing only native work."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "__init__.py").touch()
    for name in ("run_dh116_camera_c5_work_packages.py", "dh116_camera_c5_debugger.py"):
        (scripts / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    source = (ROOT / "scripts/run_dh116_work_packages.py").read_text()
    stub = """
def _cpu_worker(self, intent):
    result = {}
    native_debugger_wrapper(intent.parent, result)
    result.update(worker_root=str(self.root), worker_private=str(self.private),
                  worker_pid=os.getpid(), worker_entry=str(Path(__file__).resolve()),
                  worker_cli_entry=str(Path(sys.argv[0]).resolve()),
                  captured_commands=(intent.parent / 'debugger.gdb').read_text())
    atomic_write(intent.parent / 'receipt.json', json_text(result))
    return result
Repository.run_native_worker = _cpu_worker
Repository.verify_local_target = lambda self: None
"""
    marker = "if __name__ == '__main__':"
    assert source.count(marker) == 1
    # Only the native body is replaced in this private CPU fixture. start_native,
    # argument parsing, root derivation, and debugger wrapper remain real code.
    (scripts / "run_dh116_work_packages.py").write_text(source.replace(marker, stub + "\n" + marker))
    private = tmp_path / "separate evidence"
    attempt = private / "attempt"
    attempt.mkdir(parents=True)
    intent = attempt / "intent.json"
    intent.write_text("{}")
    capture_output = private / "native output"
    (attempt / "launch_argv.json").write_text(json.dumps({
        "argv": ["bash", "scripts/python.sh", "fixture.py", "--output", str(capture_output)],
        "dependency_receipt": None,
    }))
    parent = """
import json
import os
from pathlib import Path
from scripts import run_dh116_camera_c5_work_packages as wrapper
scheduler = wrapper.scheduler
repo = scheduler.Repository(Path.cwd(), evidence_dir=Path.cwd() / 'separate evidence')
intent = repo.private / 'attempt' / 'intent.json'
repo.register_native_attempt = lambda name, approval: intent
repo.registry = lambda: {'packages': {'fixture': {}}}
repo.native_recipe = lambda package: {'spec': {'budget': {'timeout_seconds': 10}}}
def parent_main():
    result = repo.start_native('fixture', {})
    print(json.dumps({'parent_root': str(repo.root), 'parent_private': str(repo.private),
                      'parent_pid': os.getpid(), 'child': result}))
    return 0
scheduler.main = parent_main
raise SystemExit(wrapper.main())
"""
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1",
           "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run([sys.executable, "-c", parent], cwd=tmp_path, env=env,
                               check=True, capture_output=True, text=True, timeout=20)
    report = json.loads(completed.stdout)
    child = report["child"]
    assert child is not None
    assert child["worker_pid"] != report["parent_pid"]
    assert child["worker_root"] == report["parent_root"] == str(tmp_path)
    assert child["worker_private"] == report["parent_private"] == str(private)
    assert child["worker_entry"] == str(scripts / "run_dh116_camera_c5_work_packages.py")
    assert child["worker_cli_entry"] == child["worker_entry"]
    expected = importlib.import_module("scripts.dh116_camera_c5_debugger").render_gdb_commands(capture_output)
    assert child["captured_commands"] == expected
    assert "set environment DH116_CAMERA_C5_GDB 1" in child["captured_commands"]
    assert "DH116_GDB_NORMAL_FINALIZE_BACKTRACE" in child["captured_commands"]


@pytest.mark.skipif(not Path("/usr/bin/gdb").is_file(), reason="requires the Linux GDB executable")
@pytest.mark.parametrize("fatal", [False, True])
def test_real_gdb_preserves_capture_and_status_with_launcher_args(tmp_path, monkeypatch, fatal):
    from scripts import run_dh116_camera_c5_work_packages as wrapper

    output = tmp_path / "capture output"
    output.mkdir()
    (output / "before_app_close.json").write_text('{"case":"C5"}')
    (output / "app_close_returned.json").write_text('{"app_close_returned":true}')
    fixture = tmp_path / "fixture.py"
    fixture.write_text(
        "import os, resource, signal\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        + ("os.kill(os.getpid(), signal.SIGSEGV)\n" if fatal else "print('CPU fixture')\n")
    )
    argv = [str(fixture), "--output", str(output)]
    (tmp_path / "launch_argv.json").write_text(
        json.dumps({"argv": ["bash", "scripts/python.sh", *argv], "dependency_receipt": None})
    )
    def run_fixture():
        debugger = wrapper.scheduler.native_debugger_wrapper(tmp_path, {})
        return subprocess.run([debugger, *argv], capture_output=True, text=True, timeout=25)

    monkeypatch.setattr(wrapper.scheduler, "main", run_fixture)
    result = wrapper.main()
    log = tmp_path / "fixture.log"
    log.write_text(result.stdout + result.stderr)
    phase = "fatal_1" if fatal else "finalize"
    assert (output / f"gdb_{phase}.json").is_file(), log.read_text()
    capture = json.loads((output / f"gdb_{phase}.json").read_text())
    assert capture["all_commands_succeeded"], capture
    assert "[stack]" in (output / f"gdb_{phase}.maps").read_text()
    status = wrapper.scheduler.debugger_status(log)
    assert status["debugger_status_verified"]
    assert status["debugger_inferior_exit_code"] == (None if fatal else 0)
    assert status["debugger_inferior_signal"] == (11 if fatal else None)
    assert status["debugger_backtrace_captured"] == fatal
    assert (result.returncode == 0) == (not fatal)
