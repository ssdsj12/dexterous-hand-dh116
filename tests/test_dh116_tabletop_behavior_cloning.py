"""Synthetic unit fixtures exercise contracts, never certify simulator success."""
import copy
import hashlib

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_behavior_cloning as bc


def contract():
    return dict(observation_version='unit-only-156', observation_size=156,
                action='20D_piecewise_joint_envelope_palm_load_v1', physics_hz=480,
                reward='unit-only-reward', grasp_controller={'version': 'unit-only'},
                num_envs=3, rollout_steps=24, minibatches=3)


def dataset():
    generator = torch.Generator().manual_seed(7)
    episodes = []
    for index in range(5):
        obs = torch.randn(12, 156, generator=generator) * .1
        episodes.append(dict(env_index=0, episode_index=index, success=True,
            observations=obs, next_observations=obs + .01,
            actions=torch.full((12, 20), .3), rewards=torch.ones(12),
            dones=torch.tensor([False]*11+[True]),
            success_mask=torch.tensor([False]*11+[True]), failure=torch.zeros(12, dtype=torch.long),
            physical_evidence=dict(contact_center_relative_m=torch.zeros(12, 2, 3),
                contact_normal=torch.tensor([[[0., 0., 1.]]*2]*12),
                contact_valid=torch.ones(12, 2, dtype=torch.bool),
                normal_vertical_force_n=torch.ones(12), lift_vertical_force_n=torch.ones(12),
                object_bottom_m=torch.full((12,), .33))))
    return dict(version=bc.DATASET_VERSION, observation_dim=156, action_dim=20,
        contract=contract(), qualification=dict(qualified=True, successful_replays=3,
        physical_validation=True, fresh_resets=True, proof_id='synthetic-unit-fixture-only'), episodes=episodes)


def save_dataset(tmp_path, change=None):
    data = dataset()
    if change:
        change(data)
    path = tmp_path/'dataset.pt'
    torch.save(data, path)
    return path


@pytest.mark.parametrize('change', [
    lambda d: d.update(episodes=[]),
    lambda d: d.update(episodes=d['episodes'][:2]),
    lambda d: d.update(observation_dim=155),
    lambda d: d['qualification'].update(qualified=False),
    lambda d: d['qualification'].update(successful_replays=2),
    lambda d: d['qualification'].pop('proof_id'),
    lambda d: d['qualification'].pop('physical_validation'),
    lambda d: d['qualification'].update(fresh_resets=False),
    lambda d: d['episodes'][0].update(success=False),
    lambda d: d['episodes'][0]['observations'].__setitem__((0, 0), float('nan')),
    lambda d: d['episodes'][0]['actions'].__setitem__((0, 0), 1.01),
    lambda d: d['episodes'][0].update(next_observations=torch.zeros(11, 156)),
    lambda d: d['episodes'][0]['dones'].__setitem__(0, True),
    lambda d: d['episodes'][0]['failure'].__setitem__(0, 1),
    lambda d: d['episodes'][1].update(episode_index=0),
])
def test_reject_invalid_dataset_before_writing_artifact(tmp_path, change):
    path = save_dataset(tmp_path, change)
    output = tmp_path/'actor.pt'
    with pytest.raises(ValueError):
        bc.train_imitation(path, output, epochs=1)
    assert not output.exists()


def test_whole_episode_training_reports_bounded_actor_and_source(tmp_path):
    path = save_dataset(tmp_path)
    output = tmp_path/'actor.pt'
    metadata = bc.train_imitation(path, output, epochs=30, seed=42, batch_size=64)
    artifact = torch.load(output, weights_only=True)
    assert artifact['version'] == bc.ACTOR_VERSION
    assert artifact['dataset_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact['contract'] == contract()
    assert metadata['validation_metrics']['mse'] < .01
    assert len(metadata['validation_metrics']['per_action_mae']) == 20
    train_ids = {tuple(x) for x in artifact['split']['train_episode_ids']}
    validation_ids = {tuple(x) for x in artifact['split']['validation_episode_ids']}
    assert train_ids and validation_ids and not (train_ids & validation_ids)
    assert len(train_ids | validation_ids) == 5
    actor = bc.make_actor()
    actor.load_state_dict(artifact['actor'])
    assert set(artifact['actor']) == {'0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias'}
    assert actor(torch.randn(2, 156)).shape == (2, 20)
    assert actor(torch.randn(2, 156)).abs().max() <= 1


def algorithm():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import make_algorithm, observations
    return make_algorithm(observations(torch.zeros(2, 156)), steps=2)


def actor_artifact(tmp_path):
    path = save_dataset(tmp_path)
    output = tmp_path/'actor.pt'
    bc.train_imitation(path, output, epochs=1)
    return output


def test_actor_initialization_preserves_fresh_critic_optimizer_and_counters(tmp_path):
    path = actor_artifact(tmp_path)
    target = algorithm()
    critic = copy.deepcopy(target.policy.critic.state_dict())
    optimizer = copy.deepcopy(target.optimizer.state_dict())
    target.current_learning_iteration = 0
    metadata = bc.load_actor_initialization(target, path, dict(contract(), num_envs=4096,
        rollout_steps=64, minibatches=8))
    assert all(torch.equal(v, target.policy.critic.state_dict()[k]) for k, v in critic.items())
    assert target.optimizer.state_dict() == optimizer
    assert target.current_learning_iteration == 0
    assert torch.allclose(target.policy.log_std.exp(), torch.full((20,), .05))
    assert metadata['actor_only'] is True
    assert metadata['optimizer_state_entries'] == 0
    assert metadata['closed_loop_student_validation'] == 'required_before_optimization'
    saved = torch.load(path, weights_only=True)
    assert all(torch.equal(v, target.policy.actor.state_dict()[k]) for k, v in saved['actor'].items())


@pytest.mark.parametrize('change', [
    lambda a: a['contract'].update(physics_hz=240),
    lambda a: a['contract'].update(reward='different'),
    lambda a: a['contract']['grasp_controller'].update(version='different'),
    lambda a: a.update(observation_dim=155),
    lambda a: a['qualification'].update(qualified=False),
    lambda a: a.update(initial_noise_std=.01),
    lambda a: a['actor']['0.weight'].__setitem__((0, 0), float('inf')),
    lambda a: a['actor'].update({'0.weight': torch.zeros(128, 155)}),
    lambda a: a.update(dataset_sha256='not-a-hash'),
])
def test_reject_invalid_initialization_atomically(tmp_path, change):
    path = actor_artifact(tmp_path)
    artifact = torch.load(path, weights_only=True)
    change(artifact)
    torch.save(artifact, path)
    target = algorithm()
    before = copy.deepcopy(target.policy.state_dict())
    with pytest.raises(ValueError):
        bc.load_actor_initialization(target, path, contract())
    assert all(torch.equal(v, target.policy.state_dict()[k]) for k, v in before.items())
    assert not target.optimizer.state


def test_reject_optimizer_history_and_nonzero_counters(tmp_path):
    path = actor_artifact(tmp_path)
    target = algorithm()
    target.optimizer.state[next(target.policy.actor.parameters())]['step'] = torch.tensor(1.)
    with pytest.raises(ValueError, match='fresh'):
        bc.load_actor_initialization(target, path, contract())
    target = algorithm()
    target.current_learning_iteration = 5
    with pytest.raises(ValueError, match='fresh'):
        bc.load_actor_initialization(target, path, contract())


@pytest.mark.parametrize('change', [
    lambda a: a.update(validation_metrics={}),
    lambda a: a['validation_metrics'].update(mse=-1.),
    lambda a: a['validation_metrics'].update(per_action_mae=[0.]*19),
    lambda a: a.pop('split'),
    lambda a: a['split'].update(validation_episode_ids=a['split']['train_episode_ids']),
    lambda a: a['split'].update(method='random_transition'),
])
def test_reject_incomplete_or_leaking_evaluation_provenance(tmp_path, change):
    path = actor_artifact(tmp_path)
    artifact = torch.load(path, weights_only=True)
    change(artifact)
    torch.save(artifact, path)
    with pytest.raises(ValueError):
        bc.load_actor_initialization(algorithm(), path, contract())


def test_reject_partially_collected_rollout(tmp_path):
    path = actor_artifact(tmp_path)
    target = algorithm()
    target.storage.step = 1
    with pytest.raises(ValueError, match='fresh'):
        bc.load_actor_initialization(target, path, contract())


def test_cli_fails_closed_for_unqualified_dataset(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    path = save_dataset(tmp_path, lambda d: d['qualification'].update(qualified=False))
    script = Path(__file__).resolve().parents[1]/'scripts/train_dh116_tabletop_imitation.py'
    output = tmp_path/'rejected.pt'
    result = subprocess.run([sys.executable, str(script), '--dataset', str(path),
        '--output', str(output), '--epochs', '1'], capture_output=True, text=True, env=os.environ)
    assert result.returncode != 0
    assert 'qualification requires qualified=True' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('field', ['dones', 'success_mask', 'failure', 'physical_evidence'])
def test_qualified_dataset_requires_terminal_trace_and_physical_evidence(tmp_path, field):
    path = save_dataset(tmp_path, lambda d: d['episodes'][0].pop(field))
    with pytest.raises(ValueError):
        bc.train_imitation(path, tmp_path/'rejected.pt', epochs=1)


@pytest.mark.parametrize('field', ['contact_center_relative_m', 'contact_normal',
    'contact_valid', 'normal_vertical_force_n', 'lift_vertical_force_n', 'object_bottom_m'])
def test_qualified_dataset_requires_every_measured_physical_field(tmp_path, field):
    path = save_dataset(tmp_path, lambda d: d['episodes'][0]['physical_evidence'].pop(field))
    with pytest.raises(ValueError, match='physical_evidence'):
        bc.train_imitation(path, tmp_path/'rejected.pt', epochs=1)


@pytest.mark.parametrize('change', [
    lambda e: e.update(contact_center_relative_m=torch.zeros(11, 2, 3)),
    lambda e: e.update(contact_normal=torch.zeros(12, 3)),
    lambda e: e.update(contact_valid=torch.full((12, 2), .5)),
    lambda e: e['object_bottom_m'].__setitem__(0, float('nan')),
    lambda e: e.update(lift_vertical_force_n=torch.zeros(12, 1)),
])
def test_reject_invalid_physical_evidence_before_training(tmp_path, change):
    path = save_dataset(tmp_path, lambda d: change(d['episodes'][0]['physical_evidence']))
    with pytest.raises(ValueError, match='physical_evidence'):
        bc.train_imitation(path, tmp_path/'rejected.pt', epochs=1)
