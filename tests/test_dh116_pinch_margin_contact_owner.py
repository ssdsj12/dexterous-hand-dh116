"""Actual Python owner with explicitly synthetic SDK buffers; no native evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from test_dh116_dual_pinch_builder import prepared,geometry
from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
ROOT=Path(__file__).resolve().parents[1]
import test_dh116_dual_pinch_closure_interval as prior_test
import test_dh116_dual_pinch_director_interval as director_tests
from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as reader
from scripts import prepare_dh116_pinch_margin_contact as closure
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

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

    sim, trace, owners, sync, record, rows, events, refs, meta, drop = prior_test.setup(
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
            prior_test.install_sdk_execution(sim, owners.resources["runtime"], owners.resources["hand_bindings"], events)
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
        assert result["measurement_steps"] == 232
        assert result["physics_steps"] == 234
        assert result["ppo_updates"] == 0
        assert result["acquisition"]["journal"] == closure.inspect_interval(out / "physical.jsonl")
    raw = json.loads((out / "result.json").read_text())
    assert raw["failure"] is not None if failed else raw["failure"] is None
    assert sim.current_time_step_index == (11 if failed else 234)
    journal = closure.inspect_interval(out / "physical.jsonl", complete=not failed)
    assert journal["settled_control_windows"] == (2 if failed else 29)
    assert journal["measured_physics_rows"] == (9 if failed else 232)
    assert journal["terminal_reason"] == reason
    clock_rows = [json.loads(line) for line in (out / "sdk_clock.jsonl").read_text().splitlines()]
    assert len(clock_rows) == (9 if failed else 232)
    assert [row["sdk_step"] for row in clock_rows] == list(range(3, (12 if failed else 235)))
    assert clock_rows[-1]["policy_step"] == (1 if failed else 28)
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
    assert sum(e.get("operation") == "end" for e in entries) == (2 if failed else 29)
def run_owner(monkeypatch,tmp_path,prepared,geometry,failure=None,old_guard=False):
    from scripts import prepare_dh116_pinch_margin_contact as operation
    calls=director_tests.install_cpu_directors(monkeypatch)
    monkeypatch.setattr(prior_test,"run_actual_episode",run_actual_episode)
    old_install=prior_test.install_sdk_execution
    parameter_calls=[]
    def install(sim,rod,bindings,events):
        old_install(sim,rod,bindings,events)
        rod.rope.num_envs=1;rod.rope.fixed_root_orientation=False
        cfg=json.loads((reader.ROOT/reader.CONFIG).read_text())["rod"]
        rest=np.linalg.norm(np.diff(rod._base.nodes,axis=0),axis=1)[None]
        def capture():
            parameter_calls.append(sim.current_time_step_index)
            if sim.current_time_step_index==failure:raise ValueError("CPU injected parameter failure")
            return dict(schema="dh116.native_parameter_snapshot.v1",num_envs=1,nodes_per_env=41,time_s=rod.rope.time,
                metadata_source="native_host_metadata",segment_source="cuda_device_copy",hardware_measured=False,
                contact_radius_m=cfg["radius_m"],gravity_world_m_s2=np.array(cfg["gravity_world_m_s2"],np.float32),
                contact_mu_static=cfg["mu_s"],contact_mu_kinetic=cfg["mu_k"],contact_v_slip_m_s=cfg["v_slip_m_s"],
                contact_kappa=cfg["kappa"],rest_lengths_m=rest.copy(),stored_k_stretch_n_m=rest*cfg["axial_stiffness_n"],
                rest_quaternion_norm_n_m=4*cfg["bend_stiffness_n_m2"]/rest,
                stretch_ea_n=np.full((1,40),cfg["axial_stiffness_n"],np.float32),
                bend_ei_n_m2=np.full((1,40),cfg["bend_stiffness_n_m2"],np.float32),
                inverse_masses_kg_inv=rod.rope.inv_masses()[None])
        rod.rope.parameter_snapshot=capture
    monkeypatch.setattr(prior_test,"install_sdk_execution",install)
    monkeypatch.setattr(prior_test,"closure",operation)
    # The CPU fixture must instantiate the same explicit backend identity as
    # this independent experiment, before original builder setup.
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    with bindings((builder,"BACKEND",reader.BACKEND),(builder,"BACKEND_SHA",reader.BACKEND_SHA)):
        if failure is not None and not old_guard:
            with pytest.raises(RuntimeError,match="material_director"):
                prior_test.run_actual_episode(monkeypatch,tmp_path,prepared,geometry,False)
        else:prior_test.run_actual_episode(monkeypatch,tmp_path,prepared,geometry,old_guard)
    return tmp_path/"closure",calls,parameter_calls

@pytest.mark.parametrize("failure",[None,11])
def test_actual_owner_full_budget_or_terminal_without_extra_step(monkeypatch,tmp_path,prepared,geometry,failure):
    # The legacy test fixture installs a source table shape after App and actual source FK.
    out,d,p=run_owner(monkeypatch,tmp_path,prepared,geometry,failure=failure)
    r=closure.inspect_interval(out/"physical.jsonl",complete=failure is None)
    assert r["measured_physics_rows"]==(232 if failure is None else 9)
    assert r["last_sdk_step"]==(234 if failure is None else 11)
    assert d==p==list(range(2,235 if failure is None else 12))
    assert r["material_director_audit"]["valid_packets"]==(233 if failure is None else 9)
    assert (out/"app_close_returned.json").exists()
    if failure:
        assert r["terminal_reason"]=="material_director_source_invalid"

def test_exact_18_step_old_ledger_cannot_pass_new_234_step_scope():
    from scripts.verify_dh116_pinch_margin_contact import load_parent
    parent,_=load_parent()
    with pytest.raises(ValueError,match="budget"):
        closure.inspect_interval(Path(parent["output"])/"physical.jsonl")
