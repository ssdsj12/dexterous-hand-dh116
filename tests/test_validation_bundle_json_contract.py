"""Synthetic CPU regressions for a real historical contract's tuple field.

These fixtures are not Isaac evaluation trials or successful grasps.
"""
from copy import deepcopy
import json

import pytest
import torch

from test_vision_experiment_checkpoint import checkpoint, sha
from go2_pvcnn.control.dh116.tabletop_surface_contract import semantic_sha256
from go2_pvcnn.control.dh116.tabletop_vision_action_contract import (
    LEGACY_VERSION, _ADDED, upgrade_action_contract,
)
from go2_pvcnn.control.dh116.tabletop_vision_experiment_checkpoint import retain_checkpoint_bundle
from go2_pvcnn.control.dh116.tabletop_validation_selection import _bundle


def historical_tuple_bundle(tmp_path):
    source, saved = checkpoint(tmp_path)
    base = deepcopy(saved['contract'])
    base['version'] = LEGACY_VERSION
    for key in _ADDED:
        base.pop(key, None)
    base['network'].pop('actor_output', None)
    critic = base['privileged_critic_contract']
    # This exact tuple exists in the actual iteration-1439 derived checkpoint.
    critic['phase_force_reward'] = dict(excluded_sources=(
        'table_reaction', 'gravity_compensation', 'sensor_clipping'))
    critic['surface_contract_sha256'] = semantic_sha256(critic)
    saved['contract'] = upgrade_action_contract(base)
    torch.save(saved, source)
    return source, saved, retain_checkpoint_bundle(source, tmp_path / 'bundles')


def test_valid_complete_bundle_accepts_real_contract_json_array_roundtrip(tmp_path):
    source, saved, receipt = historical_tuple_bundle(tmp_path)
    manifest = _bundle(receipt['manifest_path'])
    original = saved['contract']['privileged_critic_contract']['phase_force_reward']['excluded_sources']
    view = manifest['contract']['privileged_critic_contract']['phase_force_reward']['excluded_sources']
    assert isinstance(original, tuple) and isinstance(view, list)
    assert list(original) == view
    assert manifest['checkpoint_sha256'] == sha(source)


def test_retaining_same_real_tuple_bundle_twice_is_idempotent(tmp_path):
    source, _, receipt = historical_tuple_bundle(tmp_path)
    before = open(receipt['manifest_path'], 'rb').read()
    again = retain_checkpoint_bundle(source, tmp_path / 'bundles')
    assert again['checkpoint_sha256'] == receipt['checkpoint_sha256']
    assert open(receipt['manifest_path'], 'rb').read() == before


def test_json_view_does_not_allow_contract_metadata_tampering(tmp_path):
    _, _, receipt = historical_tuple_bundle(tmp_path)
    path = tmp_path / 'bundles' / 'iteration_00001439' / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['contract']['privileged_critic_contract']['phase_force_reward']['excluded_sources'][0] = 'changed'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='metadata mismatch'):
        _bundle(path)


def test_typed_checkpoint_contract_hash_remains_strict_after_json_view_fix(tmp_path):
    _, _, receipt = historical_tuple_bundle(tmp_path)
    path = tmp_path / 'bundles' / 'iteration_00001439' / 'manifest.json'
    manifest = json.loads(path.read_text())
    checkpoint_path = manifest['checkpoint_path']
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    phase = saved['contract']['privileged_critic_contract']['phase_force_reward']
    phase['excluded_sources'] = list(phase['excluded_sources'])
    torch.save(saved, checkpoint_path)
    # An attacker updating only the file SHA still cannot bypass the typed
    # contract state hash. Exact training-resume comparison is also unchanged.
    manifest['checkpoint_sha256'] = sha(checkpoint_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='contract hash mismatch'):
        _bundle(path)
