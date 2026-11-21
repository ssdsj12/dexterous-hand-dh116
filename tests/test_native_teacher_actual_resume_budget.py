"""Unit coverage for the timing of a CPU-only producer receipt audit."""
from types import SimpleNamespace

import pytest

from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_process as process


@pytest.mark.parametrize(("inner_s", "outer_s", "native_s"), [(180, 240, 180), (137.25, 120, 120)])
def test_native_wall_clock_starts_after_a_receipt_verification(monkeypatch, inner_s, outer_s, native_s):
    clock = SimpleNamespace(now=14000.125)
    monkeypatch.setattr(process.time, "monotonic", lambda: clock.now)
    registered = {"role": "b_resume", "device_plan": {}}
    spec = {"teacher_contract": {"source_sha256": "a" * 64, "semantic_sha256": "b" * 64,
                                 "training_config": {}, "limits": {"inner_wall_timeout_s": inner_s}},
            "budget": {"timeout_seconds": outer_s}}
    identity = {"core_identity": {}, "actual_identity": {}}
    monkeypatch.setattr(process, "_registered", lambda *_: ("teacher.b", {"spec": spec}, registered, identity, {}))
    monkeypatch.setattr(process, "_socket_path", lambda *_: None)
    monkeypatch.setattr(process, "_verify_scheduler_parent", lambda *_: None)
    monkeypatch.setattr(process, "process_identity", lambda *_: SimpleNamespace(pid=3))

    def resolve(launch, *, resume_spec, package_name):
        assert launch == registered and resume_spec == spec and package_name == "teacher.b"
        clock.now += 23.49
        return {"receipt_sha256": "c" * 64}

    repo = SimpleNamespace(resolve_actual_checkpoint_origin=resolve)
    started = clock.now
    envelope = process.make_parent_launch_envelope(repository=repo, package={}, intent_binding={},
        launch=registered, socket_path="/unsupported/cpu-test-only.sock", nonce="d" * 64)
    assert envelope.issued_monotonic_s == pytest.approx(started + 23.49)
    assert envelope.deadline_monotonic_s == pytest.approx(envelope.issued_monotonic_s + native_s)
    assert envelope.deadline_monotonic_s - envelope.issued_monotonic_s <= native_s
    assert envelope.checkpoint_origin == {"receipt_sha256": "c" * 64}
