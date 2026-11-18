"""CPU regressions for the qualified mixed reset migration."""
from copy import deepcopy
import importlib
import importlib.util

import pytest

from go2_pvcnn.control.dh116.tabletop_surface_contract import (
    ACTION_NAME, LEGACY_OBSERVATION_VERSION, bind_contract, semantic_sha256, validate_contract,
)
from go2_pvcnn.control.dh116.tabletop_surface_imitation import validate_reset_family


PINCH = 'pinch_thumb_index'
TRIPOD = 'tripod_thumb_index_middle'
MODULE = 'go2_pvcnn.control.dh116.tabletop_mixed_reset_contract'


def member_contract(digit):
    return bind_contract(dict(
        observation_version=LEGACY_OBSERVATION_VERSION, observation_size=351,
        policy_command_size=24, action=ACTION_NAME, num_envs=4,
        rollout_steps=32, reward='surface', physics_hz=480, control_hz=60,
        close_seed_sha256=digit * 64, grasp_initialization_sha256=digit * 64,
        success_bottom_m=[.39] * 4, surface_force_sharing=dict(lift_reference_bottom_m=[.36] * 4),
    ))


def reset_family():
    return [dict(label=label, contract=member_contract(digit),
                 source_dataset_sha256='f' * 64, proof_path=f'proof/{label}.json')
            for label, digit in [(PINCH, 'a'), (TRIPOD, 'b')]]


def make_mixed(module, family, contract=None, secondary_sha256='b' * 64):
    return module.mixed_reset_contract(
        family[0]['contract'] if contract is None else contract, family,
        primary_label=PINCH, secondary_label=TRIPOD,
        secondary_sha256=secondary_sha256, num_envs=4,
    )


def test_declared_layout_keeps_primary_projection_for_bc_and_binds_digest():
    assert importlib.util.find_spec(MODULE), 'mixed reset migration is missing'
    module = importlib.import_module(MODULE)
    family = reset_family()
    primary = deepcopy(family[0]['contract'])
    mixed = make_mixed(module, family)
    groups = mixed['mixed_reset']['groups']
    assert [(row['label'], row['num_envs']) for row in groups] == [(PINCH, 2), (TRIPOD, 2)]
    assert [row['grasp_initialization_sha256'] for row in groups] == ['a' * 64, 'b' * 64]
    assert mixed['mixed_reset']['layout'] == 'contiguous_primary_then_secondary'
    assert mixed['surface_contract_sha256'] != primary['surface_contract_sha256']
    validate_contract(mixed)
    projected = module.single_reset_contract(mixed)
    validate_reset_family(family, primary, projected)
    assert projected == primary == family[0]['contract']


def test_known_single_styles_can_resume_and_existing_mixed_must_match():
    module = importlib.import_module(MODULE)
    family = reset_family()
    mixed = make_mixed(module, family)
    for member in family:
        assert module.mixed_reset_resume_changes(member['contract'], mixed, family) == ['surface_mixed_reset']
    assert module.mixed_reset_resume_changes(mixed, mixed, family) == []
    reordered = deepcopy(mixed)
    reordered['mixed_reset']['groups'].reverse()
    reordered['surface_contract_sha256'] = semantic_sha256(reordered)
    with pytest.raises(ValueError, match='mixed reset'):
        module.mixed_reset_resume_changes(reordered, mixed, family)
    with pytest.raises(ValueError, match='mixed reset'):
        module.mixed_reset_resume_changes(mixed, family[0]['contract'], family)


@pytest.mark.parametrize('key,value', [('reward', 'changed'), ('rollout_steps', 16), ('num_envs', 8)])
def test_resume_rejects_changes_to_non_reset_fields(key, value):
    module = importlib.import_module(MODULE)
    family = reset_family()
    mixed = make_mixed(module, family)
    saved = deepcopy(family[0]['contract'])
    saved[key] = value
    saved['surface_contract_sha256'] = semantic_sha256(saved)
    with pytest.raises(ValueError):
        module.mixed_reset_resume_changes(saved, mixed, family)


def test_declared_secondary_must_match_the_actual_initialization():
    module = importlib.import_module(MODULE)
    family = reset_family()
    with pytest.raises(ValueError, match='secondary'):
        make_mixed(module, family, secondary_sha256='a' * 64)


def test_current_mixed_record_must_match_canonical_balanced_assignment():
    module = importlib.import_module(MODULE)
    family = reset_family()
    primary = family[0]['contract']
    mixed = make_mixed(module, family)
    mixed['mixed_reset']['groups'][0]['num_envs'] = 1
    mixed['surface_contract_sha256'] = semantic_sha256(mixed)
    with pytest.raises(ValueError, match='mixed reset'):
        module.mixed_reset_resume_changes(primary, mixed, family)
