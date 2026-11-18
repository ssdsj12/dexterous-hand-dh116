"""Evaluation physics identity survives batch/reward-only forks, never physics changes."""
from copy import deepcopy
import hashlib
import importlib
import json

import pytest
import torch

from test_vision_action_derivation import source_checkpoint
from go2_pvcnn.control.dh116.tabletop_surface_contract import semantic_sha256
from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import ablation_contract
from go2_pvcnn.control.dh116.tabletop_lift_reward_experiment import experiment_contract
from go2_pvcnn.control.dh116.tabletop_vision_action_contract import (
    SQUASHED_VERSION, upgrade_action_contract, validate_squashed_checkpoint)


def api():
    module = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract')
    assert hasattr(module, 'evaluation_physics_sha256'), 'batch-independent evaluation physics identity missing'
    return module


def critic(n=1024, reward=None):
    value = dict(num_envs=n, minibatches=8, physics_hz=480,
        success_dwell_s=5., success_slip_m=.0005,
        success_bottom_m=[.39]*(n//2)+[.40]*(n//2),
        surface_force_sharing=dict(lift_reference_bottom_m=[.36]*(n//2)+[.37]*(n//2)),
        surface_reward_ablation=ablation_contract('support_deficit'),
        object_contact_material=dict(static_friction=1., dynamic_friction=1.))
    if reward: value['surface_lift_reward_experiment'] = experiment_contract(reward)
    value['surface_contract_sha256'] = semantic_sha256(value)
    return value


def test_batch_and_declared_pure_reward_forks_share_physics_identity_without_mutation():
    a = critic(); b = critic(128, 'R1'); c = critic(200, 'R5')
    before = deepcopy((a,b,c))
    digest = api().evaluation_physics_sha256(a)
    assert digest == api().evaluation_physics_sha256(b) == api().evaluation_physics_sha256(c)
    assert (a,b,c) == before
    assert digest != hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',', ':')).encode()).hexdigest()


@pytest.mark.parametrize('field', ['physics_hz', 'success_dwell_s', 'success_slip_m',
    'material', 'success_bottom', 'reference_bottom', 'base_reward'])
def test_every_original_physical_success_and_base_reward_parameter_remains_bound(field):
    original = critic(); changed = deepcopy(original)
    if field == 'material': changed['object_contact_material']['static_friction'] += .1
    elif field == 'success_bottom': changed['success_bottom_m'][:512] = [.391]*512
    elif field == 'reference_bottom': changed['surface_force_sharing']['lift_reference_bottom_m'][:512] = [.361]*512
    elif field == 'base_reward': changed['surface_reward_ablation']['profile'] = 'other'
    else: changed[field] += 1
    changed['surface_contract_sha256'] = semantic_sha256(changed)
    assert api().evaluation_physics_sha256(changed) != api().evaluation_physics_sha256(original)


@pytest.mark.parametrize('field', ['success_bottom_m', 'lift_reference_bottom_m'])
def test_nonuniform_per_style_references_are_not_silently_flattened(field):
    value = critic(); record = value if field == 'success_bottom_m' else value['surface_force_sharing']
    record[field][3] += .0001
    with pytest.raises(ValueError, match='uniform|reference'):
        api().evaluation_physics_sha256(value)


def test_only_declared_known_pure_reward_metadata_can_be_excluded():
    value = critic(128, 'R1')
    value['surface_lift_reward_experiment']['weights']['lift_quality_height_hold'] += 100.
    with pytest.raises(ValueError, match='reward'):
        api().evaluation_physics_sha256(value)


def test_reward_candidate_loads_against_r0_execution_contract_and_retains_full_source_provenance(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_inference import load_vision_inference
    source, saved = source_checkpoint(tmp_path)
    base = deepcopy(saved['contract']); base['privileged_critic_contract'] = critic(2, 'R1')
    saved.update(version=SQUASHED_VERSION, contract=upgrade_action_contract(base))
    torch.save(saved, source)
    validate_squashed_checkpoint(saved)
    projected = api().evaluation_contract(saved['contract'], 4)
    current = deepcopy(projected)
    # The evaluation scene always uses original R0; only the actor is evaluated.
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import _ADDED, LEGACY_VERSION
    current['version'] = LEGACY_VERSION
    for key in _ADDED: current.pop(key)
    current['network'].pop('actor_output')
    current['privileged_critic_contract'].pop('surface_lift_reward_experiment')
    current['privileged_critic_contract']['surface_contract_sha256'] = semantic_sha256(current['privileged_critic_contract'])
    current = upgrade_action_contract(current)
    original = source.read_bytes()
    loaded = load_vision_inference(source, current)
    assert loaded.provenance['source_contract'] == saved['contract']
    assert loaded.provenance['evaluation_contract'] == current
    assert source.read_bytes() == original
    assert api().evaluation_physics_sha256(projected['privileged_critic_contract']) == api().evaluation_physics_sha256(current['privileged_critic_contract'])
    # Inference permission cannot loosen exact training resume.
    with pytest.raises(ValueError, match='same complete v3'):
        validate_squashed_checkpoint(saved, current)
    bad = deepcopy(current); bad['latent_noise']['max_std'] = 99.
    with pytest.raises(ValueError, match='contract'):
        load_vision_inference(source, bad)
    bad = deepcopy(current); bad['physics_reward_provenance']['privileged_critic_contract_sha256'] = 'a'*64
    with pytest.raises(ValueError, match='provenance|contract'):
        load_vision_inference(source, bad)


def test_native_preflight_uses_same_frozen_manifest_for_reward_candidate(tmp_path):
    from types import SimpleNamespace
    from test_vision_evaluation_trials import manifest
    source, saved = source_checkpoint(tmp_path)
    base = deepcopy(saved['contract']); base['privileged_critic_contract'] = critic(2, 'R1')
    base.update(camera_mode='fixed_per_env',camera_batch_size=2,
        camera_batching='fixed_per_environment_real_RGBD',camera_resolution_wh=[224,96])
    saved.update(version=SQUASHED_VERSION,contract=upgrade_action_contract(base))
    torch.save(saved,source)
    value=manifest(100,'validation')
    value['metadata']['camera_sha256']=base['camera_config_sha256']
    value['metadata']['physics_sha256']=api().evaluation_physics_sha256(critic(1024))
    frozen=tmp_path/'validation.json';frozen.write_text(json.dumps(value))
    args=SimpleNamespace(vision_evaluation_checkpoint=source,vision_evaluation_manifest=frozen,
        vision_evaluation_max_control_steps=1800,output=tmp_path/'evaluation',vision_student=True,
        tensor_pipeline=True,headless=True,num_envs=200,vision_camera_mode='fixed',
        vision_width=224,vision_height=96,vision_visibility_mode='all',resume=None,
        vision_action_distribution='tanh_gaussian',surface_reward_profile='support_deficit')
    api().validate_vision_evaluation_options(args)
    assert not args.output.exists()
    args.vision_lift_reward_experiment='R1'
    with pytest.raises(ValueError,match='exclusive'):
        api().validate_vision_evaluation_options(args)
