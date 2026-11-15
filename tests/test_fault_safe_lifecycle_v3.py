"""CPU fault injection into the opt-in lifecycle; no native qualification."""
import importlib
import importlib.util
import json
import sys
import weakref
from types import SimpleNamespace

import pytest
from scripts import dh116_pinch_acquisition_v2 as old


def api():
    name = "scripts.dh116_fault_safe_lifecycle_v3"
    assert importlib.util.find_spec(name) is not None, "fault-safe lifecycle missing"
    return importlib.import_module(name)


def fixture(tmp_path, monkeypatch, damage=None):
    events, refs, frames, held = [], {}, [], []
    counts = {}
    def hit(stage):
        events.append(stage)
        counts[stage] = counts.get(stage, 0) + 1
        if damage == stage or damage == (stage, counts[stage]):
            raise RuntimeError("injected " + stage)
    class Sim:
        current_time_step_index = 2
        current_time = 2 / 480
    class Runtime:
        pass
    class Trace:
        def __init__(self, path):
            hit("trace_init")
            self.references = {}
        def watch(self, name, value, **kwargs):
            try:
                refs[name] = self.references[name] = weakref.ref(value)
                hit("watch")
            finally:
                value = None
        def checkpoint(self, phase):
            hit("checkpoint:" + phase)
        def event(self, phase, **kwargs):
            hit("trace_event")
    def make_sim():
        hit("make_sim")
        return Sim()
    def acquire(sim, owners, trace, cleanup):
        value = None
        try:
            frames.append(sys._getframe())
            value = owners.keep("runtime", Runtime())
            trace.watch("runtime", value)
            if damage == "retained_owner":
                held.append(value)
            hit("acquire")
            if damage == "acquired_failure":
                return {"failure": "injected acquired failure"}
            if damage == "acquired_type":
                return None
            return {"physics_rows": 0, "code_config_sha256": "a" * 64}
        finally:
            value = sim = None
    def clear(sim):
        try:
            frames.append(sys._getframe())
            hit("clear")
        finally:
            sim = None
    def dead():
        assert all(ref() is None for name, ref in refs.items()
                   if name != "runtime" or damage != "retained_owner")
        assert all(frame.f_locals.get("sim") is None for frame in frames)
    def stop():
        dead()
        hit("stop")
    def freeze():
        hit("freeze")
        number = counts["freeze"]
        if damage == ("freeze_true", number):
            return True
        if damage == ("freeze_unknown", number):
            return None
        return False
    def manager():
        hit("manager")
        if damage == "manager_false" or damage == ("manager_false", counts["manager"]):
            return False
        if damage == "manager_unknown":
            return None
        return True
    def update():
        dead()
        assert events.index("clear") < events.index("stop")
        assert "manager" in events and "snapshot" in events
        hit("update")
    def close():
        dead()
        hit("close")
    original_clock = old.clock
    def clock(sim):
        try:
            frames.append(sys._getframe())
            hit("clock")
            return original_clock(sim)
        finally:
            sim = None
    def snapshot(rows):
        hit("snapshot")
        if damage == "ownership_unknown":
            return {"resources": {}}
        return {"resources": {key: {"alive": ref() is not None} for key, ref in rows.items()}}
    original_write = old.write_json
    def write(path, value):
        if damage == "before_marker" and path.name in ("before_app_close.json", "fault_before_app_close.json"):
            raise RuntimeError("injected before marker")
        original_write(path, value)
    monkeypatch.setattr(old, "LifetimeTrace", Trace)
    monkeypatch.setattr(old, "OWNED_NAMES", ("runtime", "simulation"))
    monkeypatch.setattr(old, "snapshot_weakrefs", snapshot)
    monkeypatch.setattr(old, "clock", clock)
    monkeypatch.setattr(old, "write_json", write)
    kwargs = dict(make_sim=make_sim, acquire=acquire, synchronize=lambda:hit("synchronize"),
                  clear_sim=clear, stop=stop, manager_is_none=manager,
                  freeze=freeze, output=tmp_path, cleanup="explicit")
    return SimpleNamespace(app=SimpleNamespace(update=update, close=close), kwargs=kwargs,
                           events=events, refs=refs, frames=frames, held=held)


def read(tmp_path, name):
    return json.loads((tmp_path / name).read_text())


def assert_stage(stages, name, *, returned, error=False, attempted=True):
    row = stages["stages"][name]
    assert row["attempted"] is attempted
    assert row["returned"] is returned
    assert bool(row["error"]) is error


def test_healthy_sequence_releases_aliases_before_stop_and_checks_before_update(tmp_path, monkeypatch):
    module = api()
    f = fixture(tmp_path, monkeypatch)
    result = module.run_lifecycle(f.app, **f.kwargs)
    assert result["failure"] is None
    assert result["close_mode"] == "normal"
    assert result["native_qualification"] is result["teacher_admission"] is False
    assert result["process_exit_verified"] is result["grasp_qualified"] is False
    assert result["physics_steps"] == 2
    assert f.events.count("close") == f.events.count("stop") == f.events.count("update") == 1
    assert f.events.index("clear") < f.events.index("stop") < f.events.index("manager")
    assert f.events.index("manager") < f.events.index("update") < f.events.index("close")
    assert f.events.count("manager") == f.events.count("snapshot") == 2
    assert all(ref() is None for ref in f.refs.values())
    assert (tmp_path / "before_app_close.json").exists()
    assert (tmp_path / "app_close_returned.json").exists()
    assert not (tmp_path / "fault_app_close_returned.json").exists()
    stages = read(tmp_path, "cleanup_stages.json")
    assert stages["version"] == module.VERSION
    assert stages["qualification_granted"] is False
    for name in ("acquire", "clear", "drop_alias", "stop", "manager_readback",
                 "ownership_readback", "post_stop_update", "close"):
        assert_stage(stages, name, returned=True)


@pytest.mark.parametrize("damage", [
    "acquire", "acquired_failure", "acquired_type", "clear", "stop", "close",
    "manager_false", "manager_unknown", "manager", "ownership_unknown", "retained_owner",
    "snapshot", "trace_init", "watch", "make_sim", "synchronize", "clock",
    "checkpoint:owned hand and solver references dropped", "checkpoint:before_stop",
    "checkpoint:after_stop", "checkpoint:before_close_check", "checkpoint:after_app_close",
    "trace_event", "update", "before_marker",
    ("freeze", 1), ("freeze", 2), ("freeze", 3),
    ("freeze_true", 1), ("freeze_true", 2), ("freeze_true", 3),
    ("freeze_unknown", 1), ("freeze_unknown", 2), ("freeze_unknown", 3),
    ("manager_false", 2), ("snapshot", 2),
])
def test_faults_persist_and_rethrow_with_one_close(tmp_path, monkeypatch, damage):
    module = api()
    f = fixture(tmp_path, monkeypatch, damage)
    with pytest.raises(RuntimeError) as caught:
        module.run_lifecycle(f.app, **f.kwargs)
    result = read(tmp_path, "result.json")
    stages = read(tmp_path, "cleanup_stages.json")
    assert result["failure"] and str(caught.value) == result["failure"]
    assert result["native_qualification"] is result["teacher_admission"] is False
    assert stages["qualification_granted"] is False
    assert f.events.count("close") == 1
    assert f.events.count("stop") == 1
    assert_stage(stages, "drop_alias", returned=True)
    assert all(ref() is None for name, ref in f.refs.items()
               if name != "runtime" or damage != "retained_owner")
    if damage not in ("trace_init", "make_sim"):
        assert f.events.index("clear") < f.events.index("stop")
    else:
        assert_stage(stages, "clear", attempted=False, returned=False)
    if damage in ("acquire", "acquired_failure", "acquired_type", "synchronize",
                  "checkpoint:owned hand and solver references dropped",
                  "close", "checkpoint:after_app_close", "before_marker"):
        assert stages["close_mode"] == "normal"
    else:
        assert stages["close_mode"] == "fault"
        assert result["close_mode"] == "fault"
        assert not (tmp_path / "before_app_close.json").exists()
        assert not (tmp_path / "app_close_returned.json").exists()
        assert (tmp_path / "fault_before_app_close.json").exists()
        assert (tmp_path / "fault_app_close_returned.json").exists()
        assert "checkpoint:before_app_close" not in f.events
    if damage == "clear":
        assert_stage(stages, "clear", returned=False, error=True)
        assert_stage(stages, "stop", returned=True)
    if damage in ("clear", "stop", "manager", "manager_false", "manager_unknown",
                  "ownership_unknown", "retained_owner", "snapshot",
                  ("freeze", 1), ("freeze", 2), ("freeze_true", 1), ("freeze_true", 2),
                  ("freeze_unknown", 1), ("freeze_unknown", 2)):
        assert "update" not in f.events
        assert_stage(stages, "post_stop_update", attempted=False, returned=False)
    if damage == "close":
        assert_stage(stages, "close", returned=False, error=True)
        assert not (tmp_path / "app_close_returned.json").exists()
    if damage == "before_marker":
        assert_stage(stages, "before_close_marker", returned=False, error=True)
        assert_stage(stages, "close", returned=True)


def test_original_and_cleanup_errors_are_strings_without_retained_aliases(tmp_path, monkeypatch):
    module = api()
    f = fixture(tmp_path, monkeypatch, "acquire")
    original_clear, original_stop = f.kwargs["clear_sim"], f.kwargs["stop"]
    def clear(sim):
        try:
            original_clear(sim)
            raise ValueError("secondary clear")
        finally:
            sim = None
    def stop():
        original_stop()
        raise ValueError("secondary stop")
    f.kwargs.update(clear_sim=clear, stop=stop)
    with pytest.raises(RuntimeError) as caught:
        module.run_lifecycle(f.app, **f.kwargs)
    result = read(tmp_path, "result.json")
    assert all(text in result["failure"] for text in ("injected acquire", "secondary clear", "secondary stop"))
    assert "update" not in f.events
    assert f.events.count("close") == 1
    stages = read(tmp_path, "cleanup_stages.json")
    for name in ("acquire", "clear", "stop"):
        assert_stage(stages, name, returned=False, error=True)
    assert all(ref() is None for ref in f.refs.values())
    tb = caught.value.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_name == "run_lifecycle":
            local = tb.tb_frame.f_locals
            assert local["sim"] is None
            assert all(not isinstance(value, BaseException) for value in local.values())
        tb = tb.tb_next


@pytest.mark.parametrize("raise_inside", [False, True])
def test_opt_in_scope_restores_legacy_binding_and_nested_scope(raise_inside):
    module = api()
    before = old.run_lifecycle
    with module.repaired_lifecycle_scope():
        assert old.run_lifecycle is module.run_lifecycle
        try:
            with module.repaired_lifecycle_scope():
                assert old.run_lifecycle is module.run_lifecycle
                if raise_inside:
                    raise RuntimeError("fixture")
        except RuntimeError:
            assert raise_inside
        assert old.run_lifecycle is module.run_lifecycle
    assert old.run_lifecycle is before
    try:
        with module.repaired_lifecycle_scope():
            raise RuntimeError("outer fixture")
    except RuntimeError:
        pass
    assert old.run_lifecycle is before


@pytest.mark.parametrize("failed_name", ["cleanup_stages.json", "result.json"])
def test_final_output_failure_keeps_original_errors_and_attempts_both_files(
        tmp_path, monkeypatch, failed_name):
    module = api()
    f = fixture(tmp_path, monkeypatch, "acquire")
    original_clear = f.kwargs["clear_sim"]
    def clear(sim):
        try:
            original_clear(sim)
            raise RuntimeError("secondary clear")
        finally:
            sim = None
    f.kwargs["clear_sim"] = clear
    original_write = old.write_json
    attempts, saved = [], {}
    def write(path, value):
        if path.name in ("cleanup_stages.json", "result.json"):
            attempts.append(path.name)
            if path.name == failed_name:
                raise OSError("injected final write " + path.name)
        original_write(path, value)
        if path.name == "cleanup_stages.json":
            saved["ledger"] = path.read_bytes()
    monkeypatch.setattr(old, "write_json", write)
    with pytest.raises(RuntimeError) as caught:
        module.run_lifecycle(f.app, **f.kwargs)
    error = str(caught.value)
    assert attempts == ["cleanup_stages.json", "result.json"]
    assert all(text in error for text in (
        "injected acquire", "secondary clear", "injected final write " + failed_name))
    assert caught.value.__context__ is None
    assert f.events.count("stop") == f.events.count("close") == 1
    assert "update" not in f.events
    assert all(ref() is None for ref in f.refs.values())
    assert not (tmp_path / failed_name).exists()
    if failed_name == "cleanup_stages.json":
        result = read(tmp_path, "result.json")
        assert result["failure"] == error
        assert result["native_qualification"] is result["teacher_admission"] is False
    else:
        assert (tmp_path / "cleanup_stages.json").read_bytes() == saved["ledger"]
        stages = read(tmp_path, "cleanup_stages.json")
        assert "injected acquire" in stages["stages"]["acquire"]["error"]
        assert "secondary clear" in stages["stages"]["clear"]["error"]
        assert_stage(stages, "stop", returned=True)
        assert_stage(stages, "close", returned=True)


def test_both_final_outputs_unwritable_rethrow_all_causes_without_extra_cleanup(
        tmp_path, monkeypatch):
    module = api()
    f = fixture(tmp_path, monkeypatch, "acquire")
    original_clear = f.kwargs["clear_sim"]
    def clear(sim):
        try:
            original_clear(sim)
            raise RuntimeError("secondary clear")
        finally:
            sim = None
    f.kwargs["clear_sim"] = clear
    original_write = old.write_json
    attempts = []
    def write(path, value):
        if path.name in ("cleanup_stages.json", "result.json"):
            attempts.append(path.name)
            raise OSError("injected final write " + path.name)
        original_write(path, value)
    monkeypatch.setattr(old, "write_json", write)
    with pytest.raises(RuntimeError) as caught:
        module.run_lifecycle(f.app, **f.kwargs)
    error = str(caught.value)
    assert attempts == ["cleanup_stages.json", "result.json"]
    assert all(text in error for text in (
        "injected acquire", "secondary clear",
        "injected final write cleanup_stages.json", "injected final write result.json"))
    assert caught.value.__context__ is None
    assert not (tmp_path / "cleanup_stages.json").exists()
    assert not (tmp_path / "result.json").exists()
    assert f.events.count("stop") == f.events.count("close") == 1
    assert "update" not in f.events
    assert all(ref() is None for ref in f.refs.values())
