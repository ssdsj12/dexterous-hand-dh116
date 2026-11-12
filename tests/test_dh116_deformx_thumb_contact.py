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


def test_right_thumb_recipe_binds_real_right_identity():
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    execution = root / "configs/dh116_execution"
    right = json.loads((execution / "deformx_thumb_right_lifecycle_v2.json").read_text())
    left = json.loads((execution / "deformx_thumb_left_lifecycle_v2.json").read_text())
    action = json.loads((root / "docs/contracts/stereo_action_contract.json").read_text())
    lock = json.loads((root / "docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json").read_text())
    assets = root / "assets/fixed_dual_panda_dh116_v1"
    manifest = json.loads((assets / "manifest.json").read_text())
    original = next(row for row in manifest["sources"] if row["kind"] == "dh116" and row["side"] == "right")
    source = assets / "dependencies" / Path(original["path"]).name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original["sha256"]
    inputs_by_filename = {Path(name).name: digest for name, digest in right["inputs"].items()}

    argv = right["argv"]
    assert right["entrypoint"] == left["entrypoint"]
    assert argv[argv.index("--side") + 1] == "right"
    assert argv[argv.index("--steps") + 1] == "240"
    assert argv[argv.index("--device") + 1] == "cuda:0"
    assert argv[argv.index("--left-scheduler-receipt") + 1] == "{left_scheduler_receipt}"
    assert argv[argv.index("--left-receipt-sha256") + 1] == "{left_receipt_sha256}"
    assert "--left-evidence" not in argv
    assert right["dependency_receipt"] == {
        "package": "flex.f4-left-contact-native",
        "qualification_scope": "dh116_left_thumb_deformx_contact_lifecycle",
    }
    assert right["budget"] == left["budget"]
    assert right["debugger"] == left["debugger"] == "gdb_on_segv"
    assert len(right["inputs"]["/usr/bin/gdb"]) == 64
    assert right["inputs"]["/usr/bin/gdb"] == left["inputs"]["/usr/bin/gdb"]
    assert "left_finger14_link.json" not in inputs_by_filename
    assert "left.finger14_link.npz" not in inputs_by_filename
    assert inputs_by_filename[source.name] == original["sha256"]
    assert inputs_by_filename["right_finger14_link.json"] == lock["gpu_rows"]["right.finger14_link"]
    expected = right["validator"]["equals"]
    assert expected["side"] == "right"
    assert expected["body_path"] == "/World/DeformXThumb/rightHand/finger14_link"
    assert expected["source_usd_sha256"] == original["sha256"]
    assert expected["native_hull_sha256"] == lock["gpu_rows"]["right.finger14_link"]
    assert expected["standalone_sdf_sha256"] == inputs_by_filename["right.finger14_link.npz"]
    assert expected["standalone_sdf_sha256"] != left["validator"]["equals"]["standalone_sdf_sha256"]
    assert expected["registered_pad_slot"] == 0
    assert expected["thumb_action_index"] == 7
    assert action["action_contract"]["joint_order"][expected["thumb_action_index"]] == "right.finger12"
    assert expected["physics_rows"] == 240 and expected["sample_validation_passed"] is True
    assert expected["failure_reasons"] == [] and expected["failure"] is None
    assert right["validator"]["line_counts"] == {"physics.jsonl": 240}


def test_right_thumb_runtime_identity_rejects_left_action_or_wrong_pad():
    from scripts.verify_dh116_deformx_thumb_contact import verify_thumb_ledger_identity

    start = np.zeros(6)
    endpoint = np.array([0., -.09, 0., 0., 0., 0.])
    path = "/World/DeformXThumb/rightHand/finger14_link"
    rows = [
        {"phase_step": step, "canonical_action24": command_at_step(start, endpoint, step=step, side="right")[1],
         "side": "right", "body_path": path, "mesh_index": 0, "rod_uid": "rod0"}
        for step in range(1, 241)
    ]
    assert verify_thumb_ledger_identity(rows, side="right", body_path=path, slot=0) == 7

    left_action = command_at_step(start, endpoint, step=49, side="left")[1]
    with pytest.raises(ValueError, match="canonical action"):
        verify_thumb_ledger_identity(
            [*rows[:48], {**rows[48], "canonical_action24": left_action}, *rows[49:]],
            side="right", body_path=path, slot=0,
        )
    for change in ({"mesh_index": 1}, {"body_path": path.replace("rightHand", "leftHand")}, {"rod_uid": "rod1"}):
        with pytest.raises(ValueError, match="pad identity"):
            verify_thumb_ledger_identity([*rows[:-1], {**rows[-1], **change}], side="right", body_path=path, slot=0)
