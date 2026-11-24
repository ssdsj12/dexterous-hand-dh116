"""CPU ownership/order proofs; no native release or exit qualification."""
import importlib
import json
import weakref

import pytest

def api():
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_runtime_lifecycle")

class Timeline:
    def __init__(self, events):
        self.events = events
        self.stopped = False
    def stop(self):
        self.events.append("stop")
        self.stopped = True
    def commit(self):
        self.events.append("commit")
    def is_stopped(self):
        return self.stopped

def test_stop_commits_original_sequence_and_drops_local_before_close(tmp_path):
    events, refs = [], []
    def acquire():
        timeline = Timeline(events)
        refs.append(weakref.ref(timeline))
        return timeline
    owner = api().TimelineStopOwner(acquire=acquire, freeze=lambda:False,
        read_clock=lambda:dict(sdk_step=2), output=tmp_path)
    result = owner.stop()
    assert result["stop_commit_returned"] is True
    assert events == ["stop", "commit"]
    assert refs[0]() is None
    assert owner.before_close() is True
    audit = json.loads((tmp_path/"timeline_release.json").read_text())
    assert audit["local_timeline_weakref_alive"] is False
    assert audit["local_timeline_release_verified"] is True
    assert audit["native_release_verified"] is False
    assert audit["process_exit_verified"] is False

def test_foreign_timeline_holder_is_rejected_without_destroying_it(tmp_path):
    held = Timeline([])
    owner = api().TimelineStopOwner(acquire=lambda:held, freeze=lambda:False,
        read_clock=lambda:dict(sdk_step=2), output=tmp_path)
    owner.stop()
    with pytest.raises(RuntimeError, match="timeline.*retained"):
        owner.before_close()
    audit = json.loads((tmp_path/"timeline_release.json").read_text())
    assert audit["local_timeline_weakref_alive"] is True
    assert held.is_stopped()

def test_non_weakrefable_native_wrapper_is_unknown_not_released(tmp_path):
    class Unsupported:
        __slots__ = ()
        def stop(self): pass
        def commit(self): pass
        def is_stopped(self): return True
    owner = api().TimelineStopOwner(acquire=Unsupported, freeze=lambda:False,
        read_clock=lambda:dict(sdk_step=2), output=tmp_path)
    with pytest.raises(RuntimeError, match="weakref"):
        owner.stop()
    with pytest.raises(RuntimeError, match="not verified"):
        owner.before_close()
    assert json.loads((tmp_path/"timeline_release.json").read_text())["local_timeline_release_verified"] is False

def test_failed_stop_keeps_no_local_alias_and_cannot_pass(tmp_path):
    refs = []
    class Broken(Timeline):
        def commit(self):
            raise ValueError("injected STOP failure")
    def acquire():
        timeline = Broken([])
        refs.append(weakref.ref(timeline))
        return timeline
    owner = api().TimelineStopOwner(acquire=acquire, freeze=lambda:False,
        read_clock=lambda:dict(sdk_step=2), output=tmp_path)
    with pytest.raises(RuntimeError, match="STOP failure") as caught:
        owner.stop()
    # Exception tracebacks must not retain the native timeline argument.
    assert refs[0]() is None
    with pytest.raises(RuntimeError, match="not verified"):
        owner.before_close()

def test_stop_and_release_are_once_only(tmp_path):
    owner = api().TimelineStopOwner(acquire=lambda:Timeline([]), freeze=lambda:False,
        read_clock=lambda:dict(sdk_step=2), output=tmp_path)
    owner.stop()
    with pytest.raises(RuntimeError, match="once"):
        owner.stop()
    assert owner.before_close() is True
    with pytest.raises(RuntimeError, match="once"):
        owner.before_close()


def test_nested_stop_error_chain_cannot_hold_timeline_across_close(tmp_path):
    refs = []
    def failing_helper(timeline):
        raise ValueError("nested native STOP failure")
    class Broken(Timeline):
        def commit(self):
            try:
                failing_helper(self)
            except ValueError as error:
                raise RuntimeError("wrapped STOP failure") from error
    def acquire():
        timeline = Broken([])
        refs.append(weakref.ref(timeline))
        return timeline
    owner = api().TimelineStopOwner(acquire=acquire, freeze=lambda:False,
        read_clock=lambda:dict(sdk_step=2), output=tmp_path)
    with pytest.raises(RuntimeError, match="nested native STOP failure") as caught:
        owner.stop()
    assert refs[0]() is None
    assert caught.value.__context__ is None and caught.value.__cause__ is None
