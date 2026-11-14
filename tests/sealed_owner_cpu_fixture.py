"""CPU setup copied from the old owner fixture; explicit sealed result contract.
No Isaac, CUDA physics or native qualification is produced.
"""
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
from scripts import prepare_dh116_effective_right_native as closure
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

    assert not stop and source_failure is None
    result=run()
    assert result["physics_steps"]==234 and result["measurement_steps"]==232
    assert result["acquisition"]["journal"]["status"]=="pending_post_exit"
    assert result["acquisition"]["journal"]["full_audit_performed"] is False
    assert sim.current_time_step_index==234
    assert callbacks==["clear","stop","update","close"]
    return result
