"""Synthetic CPU bundles matching one observed historical tuple field.

These tests are not native trials. Only the known phase-force excluded-source
labels have an approved tuple-to-JSON-array representation; all other types and
the serialized Torch contract digest remain strict.
"""
import copy
import json

import pytest
import torch

from test_vision_experiment_checkpoint import checkpoint
from go2_pvcnn.control.dh116.tabletop_phase_force_reward import CONTRACT as FORCE_CONTRACT
from go2_pvcnn.control.dh116.tabletop_surface_contract import semantic_sha256
from go2_pvcnn.control.dh116.tabletop_vision_action_contract import (
    LEGACY_VERSION, _ADDED, upgrade_action_contract)
from go2_pvcnn.control.dh116.tabletop_vision_experiment_checkpoint import retain_checkpoint_bundle
from go2_pvcnn.control.dh116.tabletop_validation_selection import _bundle


def retained(tmp_path, *, extra_tuple=False):
    path, saved = checkpoint(tmp_path)
    base = copy.deepcopy(saved['contract']); base['version'] = LEGACY_VERSION
    for name in _ADDED: base.pop(name, None)
    base['network'].pop('actor_output', None)
    critic = base['privileged_critic_contract']
    critic['phase_force_reward'] = dict(excluded_sources=FORCE_CONTRACT['excluded_sources'],
        grasp_phases=None, lift_phases=None)
    if extra_tuple: critic['unapproved_physical_tuple'] = (1., 2.)
    critic['surface_contract_sha256'] = semantic_sha256(critic)
    saved['contract'] = upgrade_action_contract(base)
    torch.save(saved, path)
    value = retain_checkpoint_bundle(path, tmp_path/'bundles')
    return path, saved, value


def test_real_historical_excluded_sources_tuple_has_only_json_representation_change(tmp_path):
    path, saved, retained_value = retained(tmp_path)
    original = path.read_bytes()
    assert type(saved['contract']['privileged_critic_contract']['phase_force_reward']['excluded_sources']) is tuple
    assert type(retained_value['contract']['privileged_critic_contract']['phase_force_reward']['excluded_sources']) is tuple
    value = _bundle(retained_value['manifest_path'])
    assert value['contract']['privileged_critic_contract']['phase_force_reward']['excluded_sources'] == list(FORCE_CONTRACT['excluded_sources'])
    assert value['state_sha256']['contract'] == retained_value['state_sha256']['contract']
    assert path.read_bytes() == original


@pytest.mark.parametrize('change', ['order', 'labels', 'physical_bool_type', 'typed_contract_hash'])
def test_tuple_representation_exception_never_weakens_other_contract_or_state_checks(tmp_path, change):
    _, _, value = retained(tmp_path)
    path = value['manifest_path']; data = json.loads(open(path, encoding='utf-8').read())
    critic = data['contract']['privileged_critic_contract']
    if change == 'order': critic['phase_force_reward']['excluded_sources'].reverse()
    if change == 'labels': critic['phase_force_reward']['excluded_sources'][0] = 'measured_sensor_feedback'
    if change == 'physical_bool_type': critic['observation_size'] = True
    if change == 'typed_contract_hash': data['state_sha256']['contract'] = '0'*64
    with open(path, 'w', encoding='utf-8') as stream: json.dump(data, stream)
    with pytest.raises(ValueError): _bundle(path)


def test_unknown_tuple_field_does_not_receive_generic_json_normalization(tmp_path):
    _, _, value = retained(tmp_path, extra_tuple=True)
    with pytest.raises(ValueError, match='metadata mismatch'): _bundle(value['manifest_path'])
