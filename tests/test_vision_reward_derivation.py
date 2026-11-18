"""Explicit reward-only forks never mutate an old checkpoint or physics."""
from copy import deepcopy
import json

import pytest
import torch

from test_vision_squashed_training import setup, run
from test_vision_resume import assert_same
from go2_pvcnn.control.dh116.tabletop_surface_reward_ablation import ablation_contract
from go2_pvcnn.control.dh116.tabletop_vision_action_contract import (
    LEGACY_VERSION, _ADDED, upgrade_action_contract, validate_squashed_checkpoint)


def source(tmp_path):
    dw, task, env, args = setup(tmp_path)
    run(dw, task, env, args)
    path = args.output/'vision_student_ppo.pt'
    saved = torch.load(path, weights_only=False)
    base = deepcopy(saved['contract']); base['version'] = LEGACY_VERSION
    for name in _ADDED: base.pop(name, None)
    base['network'].pop('actor_output', None)
    base['privileged_critic_contract']['surface_reward_ablation'] = ablation_contract('support_deficit')
    saved['contract'] = upgrade_action_contract(base)
    torch.save(saved, path)
    return path, saved


@pytest.mark.parametrize('profile', ['R1', 'R2', 'R3', 'R4', 'R5'])
def test_reward_fork_changes_only_declared_reward_contract(tmp_path, profile):
    from go2_pvcnn.control.dh116.tabletop_vision_reward_derivation import derive_reward_experiment_checkpoint
    from go2_pvcnn.control.dh116.tabletop_lift_reward_experiment import experiment_contract
    path, saved = source(tmp_path); original = path.read_bytes()
    output = tmp_path/(profile+'.pt')
    report = derive_reward_experiment_checkpoint(path, output, profile)
    result = torch.load(output, weights_only=False)
    validate_squashed_checkpoint(result)
    for key in saved:
        if key != 'contract': assert_same(saved[key], result[key])
    critic = deepcopy(result['contract']['privileged_critic_contract'])
    assert critic.pop('surface_lift_reward_experiment') == experiment_contract(profile)
    assert critic == saved['contract']['privileged_critic_contract']
    assert report['status'] == 'reward_warm_start_not_exact_resume'
    assert result['reward_experiment_derivation']['source_sha256'] == __import__('hashlib').sha256(original).hexdigest()
    assert json.loads(output.with_suffix('.json').read_text()) == report
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError): derive_reward_experiment_checkpoint(path, output, profile)


@pytest.mark.parametrize('case', ['R0', 'unknown', 'wrong_base', 'already_forked', 'same_path'])
def test_ambiguous_reward_forks_reject_without_touching_source(tmp_path, case):
    from go2_pvcnn.control.dh116.tabletop_vision_reward_derivation import derive_reward_experiment_checkpoint
    path, saved = source(tmp_path)
    output = tmp_path/'derived.pt'; profile = 'R1'
    if case in ('R0', 'unknown'): profile = case
    if case in ('wrong_base', 'already_forked'):
        base = deepcopy(saved['contract']); base['version'] = LEGACY_VERSION
        for name in _ADDED: base.pop(name, None)
        base['network'].pop('actor_output', None)
        if case == 'wrong_base': base['privileged_critic_contract'].pop('surface_reward_ablation')
        else: base['privileged_critic_contract']['surface_lift_reward_experiment'] = {'profile': 'R2'}
        saved['contract'] = upgrade_action_contract(base); torch.save(saved, path)
    if case == 'same_path': output = path
    original = path.read_bytes()
    with pytest.raises(ValueError): derive_reward_experiment_checkpoint(path, output, profile)
    assert path.read_bytes() == original
    assert not (tmp_path/'derived.pt').exists()


def test_existing_sidecar_is_retained_on_failed_reward_derivation(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_reward_derivation import derive_reward_experiment_checkpoint
    path, _ = source(tmp_path)
    target = tmp_path/'derived.pt'; target.with_suffix('.json').write_text('existing')
    with pytest.raises(FileExistsError): derive_reward_experiment_checkpoint(path, target, 'R1')
    assert not target.exists()
    assert target.with_suffix('.json').read_text() == 'existing'


def test_reward_resume_requires_explicit_matching_profile_before_sensors(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_reward_derivation import derive_reward_experiment_checkpoint
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_vision_action_options
    path, _ = source(tmp_path)
    target = tmp_path/'R1.pt'; derive_reward_experiment_checkpoint(path, target, 'R1')
    parent = tmp_path/'launch'; parent.mkdir()
    _, _, _, args = setup(parent, maximum=2)
    args.resume = target; args.surface_reward_profile = 'support_deficit'
    with pytest.raises(ValueError, match='surface_lift_reward_experiment'):
        validate_vision_action_options(args)
    args.vision_lift_reward_experiment = 'R1'
    validate_vision_action_options(args)
    args.vision_lift_reward_experiment = 'R2'
    with pytest.raises(ValueError, match='surface_lift_reward_experiment'):
        validate_vision_action_options(args)
    args.resume = path; args.vision_lift_reward_experiment = 'R1'
    with pytest.raises(ValueError, match='surface_lift_reward_experiment'):
        validate_vision_action_options(args)


@pytest.mark.parametrize('case', ['legacy', 'critic_only', 'wrong_base', 'unknown'])
def test_reward_option_cannot_leak_into_legacy_or_unrelated_training(tmp_path, case):
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_vision_action_options
    _, _, _, args = setup(tmp_path)
    args.vision_lift_reward_experiment = 'R1'; args.surface_reward_profile = 'support_deficit'
    if case == 'legacy': args.vision_action_distribution = 'gaussian_clip'
    if case == 'critic_only': args.vision_student = False
    if case == 'wrong_base': args.surface_reward_profile = 'baseline'
    if case == 'unknown': args.vision_lift_reward_experiment = 'R8'
    with pytest.raises(ValueError): validate_vision_action_options(args)
