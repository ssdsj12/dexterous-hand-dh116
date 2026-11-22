"""Preserve the original Reset contract when binding a resized trial."""
import importlib
import json

import numpy as np
import pytest

from test_supported_wire_specifications import BASE


def test_resized_seed_retains_hand_joints_and_measures_new_rest_length():
    module = importlib.import_module("scripts.verify_dh116_wire_variant_isaac")
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import (
        TRIPOD_SCOPE, prepare_initial_state,
    )

    hand = dict(pose_wxyz=[.44, .18, .44, 1., 0., 0., 0.], q_rad=[.2] * 6)
    scene = dict(scope=TRIPOD_SCOPE, physics_qualified=False, fixed_rod_nodes=[],
        gravity_w_m_s2=[0., 0., -9.81], offline_candidate_sha256="a"*64,
        hands=dict(left=hand, right=dict(hand, pose_wxyz=[.44, -.18, .44, 1., 0., 0., 0.])),
        rod_nodes_w_m=[[.44, float(y), .365] for y in np.linspace(-.25,.25,41)],
        support_scene=dict(wire_radius_m=.005, wire_length_m=.5, lift_reference_bottom_m=.36,
            supports=[dict(name=name, center_m=[.44, y, .33], size_m=[.024,.012,.060])
                      for name,y in (("negative_y",-.2375),("positive_y",.2375))]))
    table = dict(center_m=[.4,0.,.285], size_m=[1.,1.,.03])
    cfg = json.loads(BASE.read_bytes())
    expected, active, _, _ = prepare_initial_state(scene, table, expected_scope=TRIPOD_SCOPE)
    seed, resized_active, nodes, supports, variant = module.prepare_variant(
        scene, table, cfg["rod"], radius_m=.0045, length_m=.45, stations=dict(left=35,right=5))
    assert seed["joint_positions"] == expected["joint_positions"]
    assert np.array_equal(active, resized_active)
    assert variant["rod"]["solver_substeps"] == cfg["rod"]["solver_substeps"]
    assert np.linalg.norm(np.diff(nodes,axis=0), axis=1).sum() == pytest.approx(.45)
    assert nodes[0,2] == pytest.approx(.3645)
    assert supports["support_0"][0][1] == pytest.approx(-.21375)
    assert seed["sha256"] != expected["sha256"]
    assert scene["support_scene"]["wire_radius_m"] == .005


def test_support_settlement_requires_both_real_end_loads_at_terminal_and_no_hand_pairs():
    from scripts.verify_dh116_wire_variant_isaac import validate_settlement

    result = dict(pair_counts={"left.thumb": 0, "right.thumb": 0, "support_0": 12, "support_1": 12},
                  peak_support_force_n={"table": 0., "support_0": .05, "support_1": .04})
    row = dict(mesh_bindings={"left.thumb": {}, "right.thumb": {}},
               external_support_wrenches_world={"table": np.zeros(6),
                   "support_0": np.array([0., 0., -.05, 0., 0., 0.]),
                   "support_1": np.array([0., 0., -.04, 0., 0., 0.])})
    measured = validate_settlement(result, row)
    assert measured["terminal_support_force_n"]["support_1"] == .04
    for body, force in (("support_1", 0.), ("table", .002), ("support_0", .05)):
        loads = dict(row["external_support_wrenches_world"])
        loads[body] = np.array([0., 0., force, 0., 0., 0.])
        with pytest.raises(ValueError, match="settled on both supports"):
            validate_settlement(result, dict(row, external_support_wrenches_world=loads))
    with pytest.raises(ValueError, match="hand pair"):
        validate_settlement(dict(result, pair_counts=dict(result["pair_counts"], **{"left.thumb": 1})), row)
