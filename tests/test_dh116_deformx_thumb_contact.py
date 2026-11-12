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


def test_thumb_sdk_clock_charges_setup_before_bounded_measurement():
    from scripts.verify_dh116_deformx_thumb_contact import reset_thumb_physics, step_thumb_physics

    class FakeSim:
        current_time_step_index = 0
        reset_steps = 2
        next_step_ticks = 1
        playing = True

        def reset(self):
            self.current_time_step_index += self.reset_steps

        def is_playing(self):
            return self.playing

        def step(self, *, render):
            assert not render
            self.current_time_step_index += self.next_step_ticks

    sim = FakeSim()
    assert reset_thumb_physics(sim) == (0, 2)
    for expected in range(2, 240):
        assert step_thumb_physics(sim, expected, budget_end=240) == expected + 1
    with pytest.raises(RuntimeError, match="budget"):
        step_thumb_physics(sim, 240, budget_end=240)
    assert sim.current_time_step_index == 240

    sim = FakeSim()
    sim.reset_steps = 3
    with pytest.raises(RuntimeError, match="setup"):
        reset_thumb_physics(sim)
    sim = FakeSim()
    sim.current_time_step_index = 1
    with pytest.raises(RuntimeError, match="setup"):
        reset_thumb_physics(sim)

    sim = FakeSim()
    reset_thumb_physics(sim)
    sim.next_step_ticks = 2
    with pytest.raises(RuntimeError, match="clock"):
        step_thumb_physics(sim, 2, budget_end=240)
    sim.next_step_ticks = 1
    with pytest.raises(RuntimeError, match="clock"):
        step_thumb_physics(sim, 3, budget_end=240)
    sim.playing = False
    with pytest.raises(RuntimeError, match="timeline"):
        step_thumb_physics(sim, 4, budget_end=240)
    assert sim.current_time_step_index == 4


def test_wrench_evidence_refuses_wrong_exchange_clock():
    base = {"phase_step": 1, "sdk_step": 8, "rod_substep": 12, "sdk_time_s": 8 / 480,
            "applied_wrench_com_w": [0.] * 6, "rod_on_hand_com_w": [0.] * 6,
            "pair_force_residual_n": 0., "failure": None}
    with pytest.raises(ValueError, match="clock"):
        schedule_summary([base, {**base, "phase_step": 2, "sdk_step": 10, "rod_substep": 24}])


def test_thumb_summary_counts_238_measured_rows_without_rescaling_motion():
    start = np.zeros(6)
    end = np.array([0., .09, 0., 0., 0., 0.])
    target, action = command_at_step(start, end, step=238, side="left")
    old_final_target, old_final_action = command_at_step(start, end, step=240, side="left")
    assert np.array_equal(target, old_final_target)
    assert np.array_equal(action, old_final_action)

    rows, previous = [], [0.] * 6
    for phase in range(1, 241):
        force = [0., 0., .01, 0., 0., 0.] if phase >= 173 else [0.] * 6
        rows.append({"phase_step": phase, "sdk_step": phase + 2, "rod_substep": phase * 12,
                     "sdk_time_s": (phase + 2) / 480, "applied_wrench_com_w": previous,
                     "rod_on_hand_com_w": force, "pair_force_residual_n": 0., "failure": None})
        previous = force
    assert schedule_summary(rows[:238])["sample_validation_passed"] is True
    assert schedule_summary(rows)["sample_validation_passed"] is False


def test_right_preflight_requires_receipt_and_steps_mean_total(tmp_path):
    from types import SimpleNamespace

    from scripts.verify_dh116_deformx_thumb_contact import preflight

    with pytest.raises(ValueError, match="240 total"):
        preflight(SimpleNamespace(steps=238, side="left"), tmp_path)
    with pytest.raises(ValueError, match="qualified left scheduler receipt"):
        preflight(SimpleNamespace(steps=240, side="right", left_scheduler_receipt=None,
                                  left_evidence=tmp_path), tmp_path)


@pytest.mark.parametrize("scenario", ["complete", "run_failed", "late_tick", "cannot_freeze"])
def test_thumb_shutdown_freezes_before_clock_release_and_stop_update(tmp_path, monkeypatch, scenario):
    import sys
    from types import ModuleType, SimpleNamespace

    from go2_pvcnn.control.dh116 import tabletop_stereo_contact_calibration as calibration
    from scripts import verify_dh116_deformx_thumb_contact as probe

    sequence = []
    playing = True

    class Settings:
        enabled = True

        def get(self, path):
            if path == "/app/fastShutdown":
                return False
            assert path == "/app/player/playSimulations"
            return self.enabled

        def set_bool(self, path, value):
            assert path == "/app/player/playSimulations" and value is False
            sequence.append("freeze")
            if scenario != "cannot_freeze":
                self.enabled = value

    settings = Settings()

    class Simulation:
        def __init__(self, config):
            pass

        @property
        def current_time_step_index(self):
            assert not settings.enabled
            sequence.append("read_clock")
            return 241 if scenario == "late_tick" else 240

        @classmethod
        def clear_instance(cls):
            assert not settings.enabled
            sequence.append("clear")

    def stop():
        nonlocal playing
        assert not settings.enabled
        sequence.append("stop")
        playing = False

    def update():
        assert not settings.enabled and not playing
        sequence.append("update")

    def close(app, output):
        assert not settings.enabled and not playing
        sequence.append("close")

    def run(*args):
        sequence.append("run")
        if scenario == "run_failed":
            raise RuntimeError("physics run failed")
        return {"physics_clock": {"sdk_final_step": 240, "sdk_last_measurement_step": 240}}

    def install(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    install("carb", settings=install("carb.settings", get_settings=lambda: settings))
    install("isaaclab", sim=install("isaaclab.sim", SimulationContext=Simulation,
                                   SimulationCfg=lambda **kw: kw, PhysxCfg=lambda **kw: kw))
    install("isaacsim.core.simulation_manager", SimulationManager=SimpleNamespace(get_physics_sim_view=lambda: None))
    install("omni", timeline=install("omni.timeline", get_timeline_interface=lambda: SimpleNamespace(stop=stop)))
    monkeypatch.setattr(calibration, "close_calibration_app", close)
    monkeypatch.setattr(probe, "_run", run)
    monkeypatch.setattr(probe, "_collect_owned_references", lambda *_: None)
    ownership = SimpleNamespace(watch=lambda *a, **kw: None, event=lambda *a, **kw: None, checkpoint=lambda *_: None)
    app = SimpleNamespace(update=update)
    args = SimpleNamespace(output=tmp_path, device="cuda:0")
    if scenario == "complete":
        probe._run_thumb_probe(app, args, {}, None, None, ownership)
    else:
        reason = {"run_failed": "physics run failed", "late_tick": "shutdown clock", "cannot_freeze": "freeze"}
        with pytest.raises(RuntimeError, match=reason[scenario]):
            probe._run_thumb_probe(app, args, {}, None, None, ownership)
    if scenario == "cannot_freeze":
        assert "update" not in sequence and "close" not in sequence
    else:
        assert sequence == ["run", "freeze", "read_clock", "clear", "stop", "freeze", "update", "freeze", "close"]


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
    clock = dict(sdk_initial_step=0, sdk_after_setup_step=2, sdk_first_measurement_step=3,
                 sdk_last_measurement_step=240, sdk_final_step=240, setup_steps=2,
                 measured_steps=238, total_steps=240)
    ledger = "".join(json.dumps({"phase_step": step, "sdk_step": step + 2}) + "\n" for step in range(1, 239))
    (output / "physics.jsonl").write_text(ledger)
    (output / "result.json").write_text(json.dumps(
        {"side": "left", "sample_validation_passed": True, "physics_rows": 238, "failure_reasons": [],
         "physics_clock": clock}))
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
        "command_spec_sha256": "c" * 64, "ledger_counts": {"physics.jsonl": 238}, "physics_clock": clock,
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
    (output / "physics.jsonl").write_text(ledger)
    historical = json.loads(receipt.read_text())
    historical["ledger_counts"]["physics.jsonl"] = 240
    historical.pop("physics_clock")
    receipt.write_text(json.dumps(historical))
    with pytest.raises(ValueError, match="ledger count"):
        left_output_from_scheduler_receipt(receipt, digest(receipt))


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
    assert expected["physics_rows"] == 238 and expected["sample_validation_passed"] is True
    assert expected["failure_reasons"] == [] and expected["failure"] is None
    assert right["budget"]["physics_steps"] == 240
    for spec in (left, right):
        assert spec["validator"]["line_counts"] == {"physics.jsonl": 238}
        assert spec["validator"]["physics_clock"] == {"ledger": "physics.jsonl", "setup_steps": 2}


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
