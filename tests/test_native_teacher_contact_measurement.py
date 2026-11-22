"""Strict Teacher contact values from intact solver buffers, never geometry-only closure."""
from copy import deepcopy
import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from test_dh116_power_palm_contact_evidence import build
from go2_pvcnn.control.dh116.tabletop_deformx_pair_capture import read_pair_capture


def module():
    name = "go2_pvcnn.control.dh116.tabletop_native_teacher_contact_measurement"
    assert importlib.util.find_spec(name), "missing measured contact source for Teacher"
    return importlib.import_module(name)


def omit_one_substep_pad(rod):
    """Rebuild the full solver fixture; all aggregate reconciliation remains active."""
    pairs = [p for p in rod["rod_mesh_contact_pairs"]["pairs"]
             if not (p["substep"] == 11 and p["mesh_index"] == 1)]
    indices, vectors = [], []
    nodes = np.zeros((41, 3))
    mesh = np.zeros((17, 6))
    for p in pairs:
        indices.append((p["substep"], p["node_index"], p["mesh_index"]))
        force = np.asarray(p["force_on_rod_world_n"])
        torque = np.asarray(p["torque_on_mesh_about_origin_world_nm"])
        vectors.append([*p["contact_point_world_m"], *p["outward_normal_world"], *force, *torque])
        nodes[p["node_index"]] += force/12
        mesh[p["mesh_index"], :3] -= force/12
        mesh[p["mesh_index"], 3:] += torque/12
    raw = dict(substeps=12, mesh_count=17, capacity=41*17*12, attempted=len(pairs),
               overflow=False, indices=np.asarray(indices), vectors=np.asarray(vectors),
               node_forces_on_rod_world_n=nodes, mesh_wrenches_on_mesh_world=mesh)
    cap = read_pair_capture(SimpleNamespace(contact_pair_samples=lambda:raw), dt_s=1/480,
                           physics_step=3, rod_tick=12, mesh_names=rod["mesh_names"])
    cap["solver_buffer"] = {k:raw[k] for k in ("substeps","mesh_count","capacity","attempted","overflow")}
    rod["rod_mesh_contact_pairs"] = cap
    rod["distributed_contact"] = nodes[None]
    rod["named_wrenches"] = dict(zip(rod["mesh_names"], mesh))
    rod["table_wrench_world"] = mesh[5]
    rod["external_support_wrenches_world"] = {k:mesh[rod["mesh_names"].index(k)]
                                            for k in ("table","support_0","support_1")}
    return rod


def test_all_substeps_original_force_thresholds_and_raw_identity_are_retained():
    rod = build(proximal=True, load=.125)
    result = module().capture_supported_contact(rod, grasp="tripod")
    assert result["coverage_left"] == result["coverage_right"] == 1.
    assert result["opposed_left"] and result["opposed_right"] and result["bilateral_support"]
    assert result["off_table"] and result["no_external_support"]
    assert result["identity"]["physics_step"] == 3 and result["identity"]["rod_tick"] == 12
    assert len(result["raw_rod_sha256"]) == 64 and not result["teacher_admission"]
    result["evidence"]["per_hand"]["left"]["required_pads"].clear()
    assert module().capture_supported_contact(rod, grasp="tripod")["coverage_left"] == 1.


def test_mean_load_cannot_hide_one_missing_substep():
    rod = omit_one_substep_pad(build(proximal=True, load=.125))
    result = module().capture_supported_contact(rod, grasp="tripod")
    assert result["coverage_left"] == pytest.approx(2/3)
    assert result["coverage_right"] == 1.
    assert not result["bilateral_support"]


def test_opposition_and_all_pad_coverage_do_not_replace_total_hand_load():
    result = module().capture_supported_contact(build(proximal=True, load=.08), grasp="tripod")
    assert result["coverage_left"] == result["coverage_right"] == 1.
    assert result["opposed_left"] and result["opposed_right"]
    assert not result["bilateral_support"]


def test_wrong_material_side_invalidates_all_contact_progress():
    result = module().capture_supported_contact(build(proximal=True, load=.125, crossed=True), grasp="tripod")
    assert result["coverage_left"] == result["coverage_right"] == 0.
    assert not result["opposed_left"] and not result["opposed_right"] and not result["bilateral_support"]
    assert result["evidence"]["wrong_side_contacts"]


def test_supports_are_reported_separately_from_prohibited_table_load():
    result = module().capture_supported_contact(build(proximal=True, load=.125, table=True), grasp="tripod")
    assert not result["off_table"] and not result["no_external_support"]
    assert result["support_load_n"]["support_0"] == pytest.approx(.125)
    assert len(result["support_substep_load_n"]["table"]) == 12
    assert result["table_contact_failure"]


def test_reset_or_power_cannot_masquerade_as_measured_tripod_source():
    m = module()
    rod = build(proximal=True)
    rod["contact_valid"] = False
    with pytest.raises(ValueError):
        m.capture_supported_contact(rod, grasp="tripod")
    with pytest.raises(ValueError):
        m.capture_supported_contact(build(proximal=True), grasp="power")


def test_pinch_keeps_its_two_pad_denominator_and_own_total_load_test():
    m = module()
    loaded = m.capture_supported_contact(build(proximal=True, load=.2), grasp="pinch")
    assert loaded["bilateral_support"]
    assert loaded["evidence"]["per_hand"]["left"]["required_mesh_ids"] == [0, 1]
    assert loaded["evidence"]["per_hand"]["right"]["required_mesh_ids"] == [6, 7]
    weak = m.capture_supported_contact(build(proximal=True, load=.125), grasp="pinch")
    assert weak["coverage_left"] == weak["coverage_right"] == 1.
    assert not weak["bilateral_support"]
