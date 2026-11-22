"""Verify independently held wire specs and the same-material scaling model."""
import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/dh116_tasks/flexible_pinch_candidate_v1.json"


def api():
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_supported_wire_specs")


def test_same_material_wire_preserves_reference_and_reanchors_supports_and_grasps():
    mod = api()
    rod = json.loads(BASE.read_bytes())["rod"]
    reference = deepcopy(rod)
    y = np.linspace(-.25, .25, 41)
    scene = dict(
        rod_nodes_w_m=np.column_stack(([.44]*41, y, [.365]*41)).tolist(),
        support_scene=dict(wire_radius_m=.005, wire_length_m=.5, lift_reference_bottom_m=.36,
            supports=[dict(name=hand+"_y", center_m=[.44, position, .33], size_m=[.024, .012, .060])
                      for hand, position in (("negative", -.2375), ("positive", .2375))]),
        hands={side: dict(pose_wxyz=[.44, y[index], .43, 1., 0., 0., 0.], q_rad=[.1]*6)
               for side, index in (("left", 35), ("right", 5))})
    baseline_scene = deepcopy(scene)
    variant = mod.build_variant(scene, rod, radius_m=.004, length_m=.45,
                                grasp_material_ids={"left": 35, "right": 5})
    geometry, parameters = variant["scene"], variant["rod"]
    nodes = np.array(geometry["rod_nodes_w_m"])
    assert geometry["support_scene"]["wire_radius_m"] == .004
    assert nodes[:, 2].min() == pytest.approx(.364)
    assert np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum() == pytest.approx(.45)
    assert geometry["hands"]["left"]["pose_wxyz"][1] == pytest.approx(nodes[35, 1])
    assert geometry["hands"]["right"]["pose_wxyz"][1] == pytest.approx(nodes[5, 1])
    assert geometry["hands"]["left"]["q_rad"] == [.1]*6
    assert geometry["support_scene"]["supports"][0]["center_m"][2] == .33
    assert parameters["linear_density_kg_m"] == pytest.approx(.065 * .8**2)
    assert parameters["axial_stiffness_n"] == pytest.approx(800. * .8**2)
    assert parameters["bend_stiffness_n_m2"] == pytest.approx(.018 * .8**4)
    assert parameters["solver_substeps"] == rod["solver_substeps"]
    assert scene == baseline_scene and rod == reference


def test_same_physical_spec_cannot_be_in_training_and_holdout_under_another_name():
    mod = api()
    source = json.loads(BASE.read_bytes())["rod"]
    rows = [
        dict(id="nominal", split="train", radius_m=.005, length_m=.5),
        dict(id="new_name", split="test", radius_m=.005, length_m=.5),
    ]
    with pytest.raises(ValueError, match="physical specification"):
        mod.group_wire_specs(rows, source)
    rows[1]["radius_m"] = float(np.nextafter(.005, 1.))
    with pytest.raises(ValueError, match="physical specification"):
        mod.group_wire_specs(rows, source)
    rows[1]["radius_m"] = .0055
    selected = mod.group_wire_specs(rows, source)
    assert selected["train"][0]["spec_sha256"] != selected["test"][0]["spec_sha256"]


def test_power_feedback_requires_the_correct_material_side_and_matching_native_spec():
    from test_supported_lift_observation import sample
    from test_teacher_lift_progress import row

    mod = api()
    cfg = json.loads(BASE.read_bytes())["rod"]
    spec = dict(rod=cfg, length_m=.5)
    rod = sample(crossed=True)
    rod["rest_lengths_m"] = np.full(40, .5/40)
    packet = row(rod, 3)
    rest = rod["rest_lengths_m"][None]
    mass = np.r_[rest[0, 0]/2, (rest[0, :-1] + rest[0, 1:])/2, rest[0, -1]/2] * cfg["linear_density_kg_m"]
    packet["material_director"] = dict(identity=rod, node_positions_m=rod["node_positions"],
        raw_native_parameters=dict(
            schema="dh116.native_parameter_snapshot.v1", metadata_source="native_host_metadata",
            segment_source="cuda_device_copy", hardware_measured=False, num_envs=1, nodes_per_env=41,
            time_s=rod["solver_time_s"], contact_radius_m=cfg["radius_m"],
            gravity_world_m_s2=cfg["gravity_world_m_s2"], contact_mu_static=cfg["mu_s"],
            contact_mu_kinetic=cfg["mu_k"], contact_v_slip_m_s=cfg["v_slip_m_s"], contact_kappa=cfg["kappa"],
            rest_lengths_m=rest, stored_k_stretch_n_m=rest * cfg["axial_stiffness_n"],
            rest_quaternion_norm_n_m=4 * cfg["bend_stiffness_n_m2"] / rest,
            stretch_ea_n=np.full_like(rest, cfg["axial_stiffness_n"]),
            bend_ei_n_m2=np.full_like(rest, cfg["bend_stiffness_n_m2"]),
            inverse_masses_kg_inv=(1/mass)[None]))
    execute = lambda wire: mod.observe_wire_pose_and_reaction(packet, wire, grasp="power",
                                                              grasp_material_ids={"left": 35, "right": 5})
    result = execute(spec)
    assert result["bilateral_opposition"] is False
    assert not result["reaction_valid"]
    assert any(contact["link"] == "left.finger52_link" for contact in result["wrong_side_contacts"])
    assert "base_link" in result["measured_hands"]["left"]["reaction"]
    assert result["readback"]["weight_n"] == pytest.approx(.0325*9.81)
    wrong = deepcopy(spec)
    wrong["rod"]["radius_m"] = .004
    with pytest.raises(ValueError, match="declared physical specification"):
        execute(wrong)
