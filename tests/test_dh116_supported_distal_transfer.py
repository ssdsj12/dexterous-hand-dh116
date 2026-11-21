from copy import deepcopy

import numpy as np
import pytest

from scripts.verify_dh116_supported_distal_transfer import check_reaction_delivery, target_shape_keys, pairs_outside_station


def test_distal_shape_bindings_require_one_exact_owned_body_per_target():
    shape = np.zeros((4, 3))
    bindings = {'right.finger14_link': dict(side='right', body_name='finger14_link',
                                         body_index=4, prim_path='/World/rightHand/finger14_link')}
    target = {'right.finger14_link': [2, 3, 4]}
    assert target_shape_keys(bindings, [(1, 4, shape), (0, 4, shape)], target) == {(1, 4)}
    with pytest.raises(ValueError):
        target_shape_keys(bindings, [(1, 4, shape), (1, 4, shape)], target)
    with pytest.raises(ValueError):
        target_shape_keys(bindings, [(0, 4, shape)], target)


def test_only_the_selected_pad_and_original_material_neighborhood_is_permitted():
    allowed = {'right.finger14_link': [2, 3, 4]}
    pairs = [{'mesh_name':'right.finger14_link', 'node_index':3}, {'mesh_name':'support_0', 'node_index':0}]
    assert pairs_outside_station(pairs, allowed) == []
    assert pairs_outside_station(pairs, {}) == ['right.finger14_link:3']
    assert pairs_outside_station([dict(pairs[0], node_index=9)], allowed) == ['right.finger14_link:9']
    assert pairs_outside_station([{'mesh_name':'table', 'node_index':1}], allowed) == ['table:1']


def test_solver_reaction_must_be_delivered_at_next_sdk_tick_with_same_world_routes():
    def wrench(force):
        return {'body_ids': [1], 'is_global': True, 'forces': np.array([[[force, 0., 0.]]]),
                'torques': np.array([[[0., 0., .1]]])}

    prior = {'physics_step': 529, 'rod': {'hand_wrenches_com_world': {'right': wrench(.25)}}}
    current = {'physics_step': 530, 'reaction_source_physics_step': 529,
               'issued_composer_wrench': deepcopy(prior['rod']['hand_wrenches_com_world'])}
    assert check_reaction_delivery(prior, current)
    current['issued_composer_wrench']['right']['forces'][0, 0, 0] += .01
    with pytest.raises(ValueError):
        check_reaction_delivery(prior, current)
    current['issued_composer_wrench'] = deepcopy(prior['rod']['hand_wrenches_com_world'])
    current['reaction_source_physics_step'] = 528
    with pytest.raises(ValueError):
        check_reaction_delivery(prior, current)


def test_probe_persists_contact_then_transferred_wrench_and_rejects_unselected_contact(monkeypatch):
    from scripts import verify_dh116_supported_distal_transfer as transfer

    binding = {'right.finger14_link': dict(side='right', body_name='finger14_link',
                   body_index=4, prim_path='/World/rightHand/finger14_link')}
    target = {'right.finger14_link': [3]}
    probe = transfer.SupportedDistalProbe(528, target, binding, [(1, 4, np.zeros((4, 3)))],
                grids={'right.finger14_link': None}, archive={}, max_pad_penetration_m=.00025)
    monkeypatch.setattr(transfer, 'measure_supported_distal_contact',
                        lambda rod, grasp: dict(bilateral=False, wrong_side_contacts=[], per_hand={s: {'loaded': False}
                                                         for s in ('left', 'right')}))
    def wrench(force):
        return {'right': {'body_ids': [4], 'is_global': True,
                          'forces': np.array([[[force, 0., 0.]]]), 'torques': np.zeros((1, 1, 3))}}
    def row(sdk, applied, reaction, pairs):
        return dict(physics_step=sdk, reaction_source_physics_step=sdk-1,
                    issued_composer_wrench=wrench(applied), rod=dict(
                        hand_wrenches_com_world=wrench(reaction), rod_mesh_contact_pairs=dict(pairs=pairs)))
    assert probe.observe(row(528, 0., 0., [])) == (None, None)
    accepted, reason = probe.observe(row(529, 0., .05, [dict(mesh_name='right.finger14_link', node_index=3)]))
    assert accepted['intended_pairs'] == 1 and reason is None
    assert not accepted['reaction_from_previous_solver_row_applied']
    observed, reason = probe.observe(row(530, .05, 0., [dict(mesh_name='right.base_link', node_index=3)]))
    assert observed['reaction_from_previous_solver_row_applied']
    assert observed['violations'] == ['right.base_link:3']
    assert reason == 'unexpected_contact_outside_distal_station'
    assert probe.summary()['first_transfer_sdk'] == 530

    wrong = {'link': 'right.finger14_link', 'node_index': 3}
    monkeypatch.setattr(transfer, 'measure_supported_distal_contact',
                        lambda rod, grasp: dict(bilateral=False, wrong_side_contacts=[wrong],
                                                per_hand={s: {'loaded': False} for s in ('left', 'right')}))
    observed, reason = probe.observe(row(531, 0., 0., [dict(mesh_name='right.finger14_link', node_index=3)]))
    assert observed['violations'] == ['right.finger14_link:3']
    assert reason == 'unexpected_contact_outside_distal_station'


def test_pinch_probe_uses_a_two_pad_diagnostic_without_enabling_third_pad(monkeypatch):
    from scripts import verify_dh116_supported_distal_transfer as transfer

    target = {'right.finger14_link': [3], 'right.finger23_link': [2]}
    binding = {name: dict(side='right', body_index=i, body_name=name.split('.')[1],
                         prim_path='/World/rightHand/'+name.split('.')[1])
               for i, name in enumerate(target)}
    def observe(rod, *, grasp):
        assert grasp == 'pinch'
        return dict(bilateral=False, wrong_side_contacts=[], per_hand=dict(left={'loaded': False}, right={'loaded': False}))
    monkeypatch.setattr(transfer, 'measure_supported_distal_contact', observe)
    probe = transfer.SupportedDistalProbe(432, target, binding,
                [(1, i, np.zeros((4, 3))) for i in range(2)], grids=dict.fromkeys(target),
                archive={}, max_pad_penetration_m=.00025, grasp='pinch')
    def row(step, pairs):
        return dict(physics_step=step, reaction_source_physics_step=step-1, issued_composer_wrench={},
                    rod=dict(hand_wrenches_com_world={}, rod_mesh_contact_pairs=dict(pairs=pairs)))
    probe.observe(row(432, []))
    evidence, reason = probe.observe(row(433, [dict(mesh_name='right.finger33_link', node_index=2)]))
    assert evidence['violations'] == ['right.finger33_link:2']
    assert reason == 'unexpected_contact_outside_distal_station'


def test_bilateral_tripod_targets_activate_only_after_the_exact_right_contact_boundary():
    from scripts.verify_dh116_supported_distal_transfer import selected_stations_at

    right = {'right.finger14_link': [3]}
    left = {'left.finger14_link': [35]}
    assert selected_stations_at(528, 528, right, 752, left) == {}
    assert selected_stations_at(529, 528, right, 752, left) == right
    assert selected_stations_at(752, 528, right, 752, left) == right
    assert selected_stations_at(753, 528, right, 752, left) == {**right, **left}


def test_left_midpoint_rejects_position_rotation_and_finger_tracking_errors():
    from scipy.spatial.transform import Rotation
    from scripts.verify_dh116_supported_distal_transfer import midpoint_reached

    pose = np.array([.5, .13, .43, 1., 0., 0., 0.])
    target = {'pose_wxyz':pose,'active_q_rad':np.array([.1,.2,.3,.4,.5,.6])}
    row = {'sdk_hand_geometry':{'poses':{'left.base_link':(pose[:3].copy(),[0.,0.,0.,1.])}},
           'action_readback':{'post':{'hands':{'left':{'q':target['active_q_rad'][None].copy(),
                                                     'active_indices':list(range(6))}}}}}
    assert midpoint_reached(row,target)
    modified = deepcopy(row)
    modified['sdk_hand_geometry']['poses']['left.base_link'][0][0] += .00021
    assert not midpoint_reached(modified,target)
    modified = deepcopy(row)
    modified['sdk_hand_geometry']['poses']['left.base_link'] = (pose[:3], Rotation.from_rotvec([.009,0,0]).as_quat())
    assert not midpoint_reached(modified,target)
    modified = deepcopy(row)
    modified['action_readback']['post']['hands']['left']['q'][0,0] += .0026
    assert not midpoint_reached(modified,target)


def test_live_source_clearance_keeps_support_even_for_selected_contact_shape():
    from scripts.verify_dh116_power_shared_rod import check_supported_shape
    from scripts.screen_dh116_tripod_supported_reset import archived_tube

    y = np.linspace(-.20, .20, 41)
    rod = np.c_[np.zeros(41), y, np.zeros(41)]
    shape = np.array([[x,y,z] for x in (-.001,.001) for y in (-.001,.001) for z in (-.001,.001)])
    tube = archived_tube(rod[None])
    boxes = [('support_0', np.array([.020,-.1,-.1]), np.array([.040,.1,.1]))]
    row = check_supported_shape(shape, boxes, tube, contact_shape=False)
    assert row['conditions'] == ['rod_clearance']
    assert row['rod_gap_lower_m'] <= .00025
    assert check_supported_shape(shape, boxes, tube, contact_shape=True) is None
    row = check_supported_shape(shape+[.03,0,0], boxes, tube, contact_shape=True)
    assert row['conditions'] == ['support_clearance']
    assert list(row['support_gap_m']) == ['support_0']


def test_live_source_guard_emits_the_measured_body_before_stopping(monkeypatch):
    from types import SimpleNamespace

    from scripts import verify_dh116_power_shared_rod as shared

    step = 1178
    monkeypatch.setattr(shared.native, "clock", lambda _: {"sdk_step": step})
    transforms = np.zeros((1, 23, 7))
    transforms[..., 6] = 1.
    transforms[0, 4, :3] = [.40, .13, .30]
    transforms[0, 5, :3] = [.55, -.13, .40]
    names = [f"link_{i}" for i in range(23)]
    names[4] = "finger13_link"
    view = SimpleNamespace(get_link_transforms=lambda: transforms, get_coms=lambda: transforms,
                           link_paths=[[f"/World/left/{name}" for name in names]])
    hand = SimpleNamespace(root_physx_view=view, body_names=names)
    bindings = {side: SimpleNamespace(hand=hand) for side in shared.SIDES}
    verts = np.array([[x, y, z] for x in (-.001, .001) for y in (-.001, .001) for z in (-.001, .001)])
    rods = np.c_[np.full(41, .40), np.linspace(-.20, .20, 41), np.full(41, .30)][None]
    supports = {"supports": [{"name": "support_0", "center_m": [.80, .10, .30], "size_m": [.01, .01, .01]}]}
    records = []

    with pytest.raises(ValueError, match=r"clearance: left\.finger13_link"):
        shared.measure_links(None, bindings, {}, [(0, 4, verts), (1, 5, verts)], sdk=step,
                             support_scene=supports, reference_nodes=rods, record_guard=records.append)
    assert len(records) == 1
    record = records[0]
    assert record["current_sdk_clock"] == {"sdk_step": step}
    assert record["observed_sdk_step"] == step
    (failure,) = record["failures"]
    assert failure["link_prim"] == "/World/left/finger13_link"
    assert failure["conditions"] == ["rod_clearance"]
    assert failure["rod_gap_lower_m"] < .00025
    np.testing.assert_array_equal(failure["pose_world_wxyz"], [.40, .13, .30, 1., 0., 0., 0.])
    np.testing.assert_array_equal(failure["source_vertices_local_m"], verts)


def test_pinch_left_distal_guard_only_opens_after_its_staged_midpoint():
    names = ("finger14_link", "finger23_link")
    bindings = {f"{side}.{name}": dict(side=side, body_name=name, body_index=i,
                    prim_path=f"/World/{side}/{name}")
                for side in ("left","right") for i,name in enumerate(names)}
    shapes = [(lane,i,np.zeros((4,3))) for lane in (0,1) for i in (0,1)]
    from scripts.verify_dh116_supported_distal_transfer import SupportedDistalProbe

    probe = SupportedDistalProbe(432,{f"right.{name}":[3] for name in names},bindings,shapes,
        grids=dict.fromkeys(bindings),archive={},max_pad_penetration_m=.00025,grasp="pinch",
        bilateral={"boundary_sdk":1792,"replay_sdk":848,"stations":{f"left.{name}":[35] for name in names}})
    assert probe.contact_shape(849,(1,0))
    assert not probe.contact_shape(849,(0,0))
    assert not probe.contact_shape(1792,(0,0))
    assert probe.contact_shape(1793,(0,0))
    assert probe.stations_at(1793)=={f"{side}.{name}":[35 if side=="left" else 3]
                                    for side in ("left","right") for name in names}
