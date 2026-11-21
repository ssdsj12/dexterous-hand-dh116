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
                        lambda rod, grasp: dict(per_hand={s: {'loaded': False}
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
