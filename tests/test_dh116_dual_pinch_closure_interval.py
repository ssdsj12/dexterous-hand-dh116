"""Independent no-lift schedule and complete/terminated SDK-shaped CPU replay."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_dh116_dual_pinch_closure_interval as closure
from test_dh116_dual_pinch_builder import prepared, geometry, setup
from test_dh116_dual_pinch_interval import install_sdk_execution

ROOT = Path(__file__).resolve().parents[1]


def test_exact_schedule_does_not_permit_lift_or_other_fingers():
    actions = closure.action_schedule()
    assert tuple(actions.shape) == (2, 24)
    assert not actions[0].any()
    assert actions[1].nonzero().flatten().tolist() == [1, 2, 7, 8]
    assert actions[1, [1, 2, 7, 8]].tolist() == [.1, .1, .05, .1]
    actions[1, 14] = 1
    assert closure.action_schedule()[1, 14] == 0


@pytest.mark.parametrize("stop", [False, True])
def test_actual_episode_with_one_held_nonzero_interval(monkeypatch, tmp_path, prepared, geometry, stop):
    return run_actual_episode(monkeypatch, tmp_path, prepared, geometry, stop)


@pytest.mark.parametrize("source_failure", ["geometry", "table"])
@pytest.mark.parametrize("stop", [False, True])
def test_source_failure_preserves_ninth_physical_row_and_old_guard(monkeypatch, tmp_path, prepared, geometry, stop, source_failure):
    return run_actual_episode(monkeypatch, tmp_path, prepared, geometry, stop, source_failure)


def run_actual_episode(monkeypatch, tmp_path, prepared, geometry, stop, source_failure=None):
    from pxr import Sdf, UsdGeom
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder

    author = scene.author_source_stage
    table = prepared["prefix"] + "/Table"

    def author_nested(project, config):
        stage = author(project, config)
        layer = stage.GetRootLayer()
        temporary = Sdf.Path(table + "Temporary")
        assert Sdf.CopySpec(layer, Sdf.Path(table), layer, temporary)
        assert stage.RemovePrim(table)
        UsdGeom.Xform.Define(stage, table)
        UsdGeom.Xform.Define(stage, table + "/geometry")
        assert Sdf.CopySpec(layer, temporary, layer, Sdf.Path(table + "/geometry/mesh"))
        assert stage.RemovePrim(temporary)
        return stage

    monkeypatch.setattr(scene, "author_source_stage", author_nested)

    sim, trace, owners, sync, record, rows, events, refs, meta, drop = setup(
        monkeypatch, tmp_path, prepared, geometry
    )
    build = builder.build_reset_owned
    from go2_pvcnn.control.dh116 import tabletop_bimanual_table_contacts as table_module
    from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import MeasuredClosureGeometry
    actual_table = table_module.read_bimanual_table_contacts
    actual_capture = MeasuredClosureGeometry.capture

    def table_with_failure(*args, **kwargs):
        if source_failure == "table" and sim.current_time_step_index == 11:
            raise ValueError("CPU injected table read failure")
        return actual_table(*args, **kwargs)

    def geometry_with_failure(self, exchange, **kwargs):
        if source_failure == "geometry" and exchange["physics_step"] == 11:
            raise ValueError("CPU injected geometry read failure")
        return actual_capture(self, exchange, **kwargs)

    monkeypatch.setattr(table_module, "read_bimanual_table_contacts", table_with_failure)
    monkeypatch.setattr(MeasuredClosureGeometry, "capture", geometry_with_failure)

    def bind(sim, owners, **kwargs):
        try:
            result = build(sim, owners, **kwargs)
            install_sdk_execution(sim, owners.resources["runtime"], owners.resources["hand_bindings"], events)
            if stop:
                live = owners.resources["dual_episode"]._exchange
                read = live._read_geometry_after_sdk

                def too_low(clock):
                    row = read(clock)
                    if row["physics_step"] == 11:
                        row["clearance_m"][0, 1] = .002
                    return row

                live._read_geometry_after_sdk = too_low
            return result
        finally:
            sim = owners = None

    monkeypatch.setattr(builder, "build_reset_owned", bind)
    out = tmp_path / "closure"
    out.mkdir()
    callbacks = []
    app = SimpleNamespace(update=lambda: callbacks.append("update"), close=lambda: callbacks.append("close"))

    def clear(value):
        drop()
        value.step = None
        callbacks.append("clear")

    def run():
        return closure.run_existing_app(
            app, make_sim=lambda: sim, synchronize=sync, clear_sim=clear,
            stop=lambda: callbacks.append("stop"), manager_is_none=lambda: True,
            freeze=lambda: False, output=out, project=ROOT, prepared=prepared,
            owner_resets=owners, run_uuid="closure-fixture", episode_uid="closure-fixture:pair",
            code_config_sha256="a" * 64,
        )

    failed = stop or source_failure is not None
    reason = "post_step_source_clearance:right" if stop else (
        "closure_" + source_failure + "_source_invalid" if source_failure else None)
    if failed:
        with pytest.raises(RuntimeError, match=reason):
            run()
    else:
        result = run()
        assert result["measurement_steps"] == 16
        assert result["physics_steps"] == 18
        assert result["ppo_updates"] == 0
        assert result["acquisition"]["journal"] == closure.inspect_interval(out / "physical.jsonl")
    raw = json.loads((out / "result.json").read_text())
    assert raw["failure"] is not None if failed else raw["failure"] is None
    assert sim.current_time_step_index == (11 if failed else 18)
    journal = closure.inspect_interval(out / "physical.jsonl", complete=not failed)
    assert journal["settled_control_windows"] == 2
    assert journal["measured_physics_rows"] == (9 if failed else 16)
    assert journal["terminal_reason"] == reason
    clock_rows = [json.loads(line) for line in (out / "sdk_clock.jsonl").read_text().splitlines()]
    assert len(clock_rows) == (9 if failed else 16)
    assert [row["sdk_step"] for row in clock_rows] == list(range(3, (12 if failed else 19)))
    assert clock_rows[-1]["policy_step"] == 1
    assert (out / "app_close_returned.json").exists()
    assert callbacks == ["clear", "stop", "update", "close"]
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    entries = [json.loads(line) for line in (out / "physical.jsonl").read_text().splitlines()]
    physical_rows = [decode_owned(e["payload"]) for e in entries[1:] if e["operation"] == "physical"]
    final = physical_rows[-1]["exchange"]
    assert final["physics_step"] == sim.current_time_step_index
    assert final["guard_failure"] == reason
    assert final["original_guard_failure"] == ("post_step_source_clearance:right" if stop else None)
    assert all(row["hand_table_contacts"] == row["exchange"]["hand_table_contacts"] for row in physical_rows)
    if source_failure:
        assert final[source_failure + "_capture_error"] == "ValueError: CPU injected " + source_failure + " read failure"
        assert final["closure_geometry" if source_failure == "geometry" else "hand_table_contacts"] is None
    else:
        assert final["geometry_capture_error"] is final["table_capture_error"] is None
        assert final["closure_geometry"]["version"] == "dh116-measured-closure-geometry-v1"
    assert sum(e.get("operation") == "end" for e in entries) == 2


def test_worker_binds_exact_real_parent_and_frozen_cpu_profile():
    from scripts import verify_dh116_dual_pinch_closure_interval as worker

    contract = worker.load_profile()
    assert contract["phases"][1]["action24"] == closure.action_schedule()[1].tolist()
    with pytest.raises(ValueError, match="parent receipt"):
        worker.load_profile(worker.PARENT_RECEIPT, "a" * 64)
    with pytest.raises(ValueError, match="parent receipt"):
        worker.load_profile(ROOT / "README.md", worker.PARENT_SHA)


def test_clock_sample_uses_actual_three_way_readback():
    from copy import deepcopy
    identity = dict(physics_step=7, timestamp_s=7/480)
    row = dict(kind="physical", exchange=dict(physics_step=7, policy_step=0, substep=4,
        action_readback=dict(post=dict(identity=[dict(identity)])), rod=dict(identity),
        post_step_clearance=dict(sdk_time_s=7/480)))
    assert closure.sdk_clock_sample(row, 5) == dict(phase_step=5, sdk_step=7,
        policy_step=0, substep=4, timestamp_s=7/480, sdk_time_s=7/480)
    for path in ("sdk", "rod", "post", "sdk_time"):
        bad = deepcopy(row)
        if path == "sdk":
            bad["exchange"]["physics_step"] = 8
        elif path == "rod":
            bad["exchange"]["rod"]["physics_step"] = 8
        elif path == "sdk_time":
            bad["exchange"]["post_step_clearance"]["sdk_time_s"] += 1
        else:
            bad["exchange"]["action_readback"]["post"]["identity"][0]["physics_step"] = 8
        with pytest.raises(ValueError, match="actual SDK"):
            closure.sdk_clock_sample(bad, 5)


@pytest.mark.parametrize("source", ["geometry", "table"])
@pytest.mark.parametrize("original", [None, "right:mimic_error"])
def test_completed_physical_sample_preserved_when_diagnostic_source_fails(source, original):
    row = dict(physics_step=11, guard_failure=original)
    def failure():
        raise ValueError("test source unavailable")
    def geometry():
        return dict(passed=True, terminal_reason=None)
    def table():
        return {"source": "test only"}
    result = closure.complete_observation(row,
        geometry=failure if source == "geometry" else geometry,
        table=failure if source == "table" else table)
    assert result["physics_step"] == 11
    assert result["original_guard_failure"] == original
    assert result["guard_failure"] == (original or "closure_" + source + "_source_invalid")
    assert result[source + "_capture_error"] == "ValueError: test source unavailable"
    assert result["closure_geometry" if source == "geometry" else "hand_table_contacts"] is None


def test_execute_rebinding_reaches_captured_original_once(monkeypatch):
    from scripts import verify_dh116_dual_pinch_closure_interval as worker
    calls = []
    def sentinel(*args, **kwargs):
        calls.append(kwargs["prepared"])
        return "original-owner-once"
    monkeypatch.setattr(closure, "run_original_existing_app", sentinel, raising=False)
    monkeypatch.setattr(worker.original, "execute", lambda *unused:
        worker.original.interval.run_existing_app(prepared={}, project=ROOT))
    assert worker.execute(None, ROOT, {}) == "original-owner-once"
    assert calls == [{}]


@pytest.mark.parametrize("source", ["geometry", "table"])
@pytest.mark.parametrize("bad", ["missing_packet", "packet_with_error", "empty_error"])
def test_observation_packet_error_pairs_fail_closed(source, bad):
    row = closure.complete_observation(dict(guard_failure=None),
        geometry=lambda: dict(passed=True, terminal_reason=None), table=lambda: {"available": False})
    field = "closure_geometry" if source == "geometry" else "hand_table_contacts"
    if bad == "missing_packet":
        row[field] = None
    elif bad == "packet_with_error":
        row[source + "_capture_error"] = "ValueError: unavailable"
    else:
        row[field] = None
        row[source + "_capture_error"] = ""
    with pytest.raises(ValueError, match=source):
        closure.observation_failure(row)


@pytest.mark.parametrize("old_guard", [None, "right:mimic_error"])
def test_geometry_terminal_precedes_table_source_error_but_not_original_guard(old_guard):
    def failed_table():
        raise ValueError("table unavailable")
    row = closure.complete_observation(dict(guard_failure=old_guard),
        geometry=lambda: dict(passed=False, terminal_reason="closure_geometry_guard"), table=failed_table)
    assert row["guard_failure"] == (old_guard or "closure_geometry_guard")
    assert row["table_capture_error"] == "ValueError: table unavailable"
    assert row["hand_table_contacts"] is None


def test_missing_geometry_packet_becomes_durable_source_failure():
    row = closure.complete_observation(dict(physics_step=11, guard_failure=None),
        geometry=lambda: None, table=lambda: {"available": False})
    assert row["physics_step"] == 11
    assert row["guard_failure"] == "closure_geometry_source_invalid"
    assert row["closure_geometry"] is None
    assert row["geometry_capture_error"] == "ValueError: owned observation packet required"
