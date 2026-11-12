import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_rod_contact import (
    command_at_step, schedule_summary,
)


def test_bounded_target_and_one_exchange_lag():
    q0 = np.zeros(6)
    endpoint = np.array([0., .09, 0., 0., 0., 0.])
    q, a = command_at_step(q0, endpoint, step=49, side="right")
    assert q[1] == pytest.approx(.09 / 160)
    assert a[7] == pytest.approx(.9)
    assert np.count_nonzero(a) == 1
    _, hold = command_at_step(q0, endpoint, step=209, side="right")
    assert np.count_nonzero(hold) == 0
    with pytest.raises(ValueError, match="velocity"):
        command_at_step(q0, endpoint * 2, step=49, side="left")


def test_wrench_evidence_refuses_wrong_exchange_clock():
    base = {"phase_step": 1, "sdk_step": 8, "rod_substep": 12, "sdk_time_s": 8 / 480,
            "applied_wrench_com_w": [0.] * 6, "rod_on_hand_com_w": [0.] * 6,
            "pair_force_residual_n": 0., "failure": None}
    with pytest.raises(ValueError, match="clock"):
        schedule_summary([base, {**base, "phase_step": 2, "sdk_step": 10, "rod_substep": 24}])


def test_owned_release_checkpoint_observes_collected_callback_cycle(tmp_path):
    import gc
    import json

    from go2_pvcnn.control.dh116.tabletop_deformable_rod_bridge import NativeOwnershipTrace
    from scripts.verify_dh116_deformx_thumb_contact import _collect_owned_references

    class Owned:
        pass

    evidence = tmp_path / "ownership.jsonl"
    trace = NativeOwnershipTrace(evidence)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        owned = Owned()
        owned.cycle = owned
        trace.watch("probe", owned, owner="test", release="drop before close")
        del owned
        assert trace.references["probe"]() is not None
        _collect_owned_references(trace, "before app close")
        assert trace.references["probe"]() is None
    finally:
        if was_enabled:
            gc.enable()
    last = json.loads(evidence.read_text().splitlines()[-1])
    assert last["phase"] == "before app close"
    assert last["python_liveness"] == {"probe": False}


def test_scheduler_left_receipt_requires_outer_zero_and_locked_ledger(tmp_path):
    import hashlib
    import json

    from scripts.verify_dh116_deformx_thumb_contact import left_output_from_scheduler_receipt

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint = "b" * 64
    attempt = tmp_path / "native-attempts" / "flex.f4-left-contact-native" / fingerprint / "attempt-test"
    attempt.mkdir(parents=True)
    output = tmp_path / "outputs" / "flex.f4-left-contact-native" / "attempt-test" / "native"
    output.mkdir(parents=True)
    log = attempt / "native.log"
    log.write_text("test only")
    (output / "physics.jsonl").write_text("{}\n" * 240)
    (output / "result.json").write_text(json.dumps(
        {"side": "left", "sample_validation_passed": True, "physics_rows": 240, "failure_reasons": []}))
    (output / "shutdown_complete.json").write_text('{"close_returned": true}')
    names = ("physics.jsonl", "result.json", "shutdown_complete.json")
    outer = {"outer_exit_code": 0, "native_lock_observed": True, "log_sha256": digest(log)}
    (attempt / "outer_exit.json").write_text(json.dumps(outer))
    (attempt / "intent.json").write_text(json.dumps({
        "package": "flex.f4-left-contact-native", "output": str(output),
        "code_config_sha256": fingerprint, "command_spec_sha256": "c" * 64,
    }))
    receipt = attempt / "receipt.json"
    receipt.write_text(json.dumps({
        **outer, "executor": "native_bounded", "native_execution_performed": True,
        "native_qualification": True, "output": str(output), "code_config_sha256": fingerprint,
        "command_spec_sha256": "c" * 64, "ledger_counts": {"physics.jsonl": 240},
        "artifact_sha256": {name: digest(output/name) for name in names},
    }))
    receipt_hash = digest(receipt)
    assert left_output_from_scheduler_receipt(receipt, receipt_hash) == output
    (attempt / "outer_exit.json").write_text(json.dumps({**outer, "outer_exit_code": 139}))
    with pytest.raises(ValueError, match="outer"):
        left_output_from_scheduler_receipt(receipt, receipt_hash)
    (attempt / "outer_exit.json").write_text(json.dumps(outer))
    (output / "physics.jsonl").write_text('{"different": true}\n' * 240)
    with pytest.raises(ValueError, match="SHA256"):
        left_output_from_scheduler_receipt(receipt, receipt_hash)
