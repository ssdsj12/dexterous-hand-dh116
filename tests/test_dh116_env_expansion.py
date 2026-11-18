"""CPU-only resume migration tests; synthetic dynamics do not qualify physics."""
import copy
import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from go2_pvcnn.control.dh116 import tabletop_training as checkpoint
from go2_pvcnn.control.dh116.tabletop_vec_training import run_vector_training


class ExpansionToy(gym.Env):
    observation_space = gym.spaces.Box(-1., 1., (4,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)

    def __init__(self):
        self.count = 0

    def set_stage(self, stage):
        pass

    def set_mode(self, mode):
        pass

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.count += 1
        done = self.count == 3
        return np.zeros(4, dtype=np.float32), float(1 - np.mean(action**2)), done, False, dict(
            phase='lift', reached_phases=['close', 'hold', 'lift'], is_success=done,
            failure=None, safety_failure=None)


def vector(count):
    return DummyVecEnv([ExpansionToy for _ in range(count)])


@pytest.fixture
def source(tmp_path):
    contract = dict(num_envs=2, physics_hz=480, control_hz=60, reward='same',
                    rod=dict(dt=1e-5, gravity=(0., 0., -9.81), anchors=[(1., 2.)]))
    output = tmp_path/'source'
    _, model = run_vector_training(vector(2), output, contract, direct_training=True,
        timesteps=16, eval_episodes=2, checkpoint_every=16,
        ppo_kwargs=dict(n_steps=4, batch_size=4, n_epochs=1))
    return output/'ppo_checkpoint.zip', contract, model


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def test_nested_tuple_contract_roundtrip_is_strict_resume_compatible(source):
    path, contract, original = source
    restored, _ = checkpoint.load_checkpoint(path, vector(2), contract)
    assert_state_equal(original.policy.state_dict(), restored.policy.state_dict())
    assert checkpoint.canonical_contract(contract) == json.loads(json.dumps(contract))
    assert isinstance(contract['rod']['gravity'], tuple)


def test_expansion_restores_policy_optimizer_and_allocates_new_rollout(source):
    path, contract, original = source
    before = (path.read_bytes(), path.with_suffix('.json').read_bytes())
    restored, metadata = checkpoint.load_checkpoint(path, vector(4), dict(contract, num_envs=4),
                                                    allow_env_expansion=True)
    assert_state_equal(original.policy.state_dict(), restored.policy.state_dict())
    assert_state_equal(original.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict())
    assert restored.num_timesteps == original.num_timesteps == 16
    assert restored._n_updates == original._n_updates == 2
    assert restored.n_envs == restored.rollout_buffer.n_envs == 4
    assert restored.rollout_buffer.observations.shape == (4, 4, 4)
    assert restored._last_obs is None
    assert metadata['env_expansion'] == dict(source_num_envs=2, target_num_envs=4,
        source_timesteps=16, source_updates=2)
    saved_optimizer = copy.deepcopy(restored.policy.optimizer.state_dict())
    restored.learn(total_timesteps=16, reset_num_timesteps=False)
    assert restored.num_timesteps == 32 and restored._n_updates == 3
    assert all(restored.policy.optimizer.state_dict()['state'][key]['step'] > value['step']
               for key, value in saved_optimizer['state'].items())
    assert before == (path.read_bytes(), path.with_suffix('.json').read_bytes())


def test_expansion_progress_prefix_new_zero_and_truthful_step_history(source, tmp_path, monkeypatch):
    path, contract, original = source
    output = tmp_path/'expanded'
    old_rows = json.loads(path.with_suffix('.json').read_text())['training_progress']['per_env']
    learn = PPO.learn
    initial = []

    def inspect_then_learn(model, *args, **kwargs):
        if not initial:
            progress = json.loads((output/'monitor.json').read_text())
            initial.append(progress)
            assert progress['per_env'][:2] == old_rows
            assert [r['episodes'] for r in progress['per_env']] == [2, 2, 0, 0]
            assert progress['steps_per_env'] == [8, 8, 0, 0]
            assert_state_equal(original.policy.optimizer.state_dict(), model.policy.optimizer.state_dict())
        return learn(model, *args, **kwargs)

    monkeypatch.setattr(PPO, 'learn', inspect_then_learn)
    report, model = run_vector_training(vector(4), output, dict(contract, num_envs=4),
        resume=path, allow_env_expansion=True, direct_training=True, episodes_per_env=3,
        timesteps=16, eval_episodes=2)
    assert initial
    assert report['resumed_completed_episodes_per_env'] == [2, 2, 0, 0]
    assert report['env_expansion']['source_num_envs'] == 2
    progress = json.loads((output/'monitor.json').read_text())
    assert progress['steps_per_env'] == [20, 20, 12, 12]
    assert sum(progress['steps_per_env']) == model.num_timesteps == 64
    assert [r['episodes'] for r in progress['per_env']] == [6, 6, 4, 4]
    assert report['episode_target_reached']


@pytest.mark.parametrize('changes,flag', [
    ({'num_envs': 4}, False), ({'num_envs': 1}, True), ({'num_envs': 2}, True),
    ({'num_envs': 4, 'physics_hz': 240}, True),
    ({'num_envs': 4, 'reward': 'changed'}, True),
    ({'num_envs': 4, 'rod': {'dt': 2e-5}}, True),
])
def test_contract_changes_or_nonexpansion_rejected(source, changes, flag):
    path, contract, _ = source
    with pytest.raises(ValueError, match='contract|expansion'):
        checkpoint.load_checkpoint(path, vector(changes['num_envs']), dict(contract, **changes),
                                   allow_env_expansion=flag)


def test_reward_migration_cannot_be_combined_with_expansion(source):
    path, contract, _ = source
    with pytest.raises(ValueError, match='mutually exclusive'):
        checkpoint.load_checkpoint(path, vector(4), dict(contract, num_envs=4),
                                   allow_env_expansion=True, allow_contact_reward_change=True)


@pytest.mark.parametrize('saved_count', [1, 3, True, 2.0, None])
def test_checkpoint_source_count_must_match_sb3_archive(source, saved_count):
    path, contract, _ = source
    metadata = json.loads(path.with_suffix('.json').read_text())
    metadata['contract']['num_envs'] = saved_count
    path.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='environment count|num_envs'):
        checkpoint.load_checkpoint(path, vector(4), dict(contract, num_envs=4),
                                   allow_env_expansion=True)


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(num_envs=4), lambda p: p['per_env'].pop(),
    lambda p: p['per_env'][1].update(env_index=0),
    lambda p: p['per_env'][0].update(episodes=-1),
    lambda p: p['per_env'][0]['phase_counts'].update(lift=-1),
    lambda p: p['per_env'][0]['phase_counts'].update(lift=1.5),
    lambda p: p['per_env'][0]['phase_counts'].update(lift=7),
])
def test_expansion_rejects_malformed_source_progress(source, tmp_path, mutation):
    path, contract, _ = source
    metadata = json.loads(path.with_suffix('.json').read_text())
    mutation(metadata['training_progress'])
    path.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='progress'):
        run_vector_training(vector(4), tmp_path/'bad', dict(contract, num_envs=4), resume=path,
                            allow_env_expansion=True, direct_training=True, timesteps=16)


def test_expansion_requires_history_even_with_transition_budget(source, tmp_path):
    path, contract, _ = source
    metadata = json.loads(path.with_suffix('.json').read_text())
    metadata.pop('training_progress')
    path.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='progress'):
        run_vector_training(vector(4), tmp_path/'bad', dict(contract, num_envs=4), resume=path,
                            allow_env_expansion=True, direct_training=True, timesteps=16)


def test_default_resume_cannot_silently_change_actual_environment_count(source):
    path, contract, _ = source
    with pytest.raises(ValueError, match='environment count'):
        checkpoint.load_checkpoint(path, vector(4), contract)


def test_expansion_cannot_write_into_source_run(source):
    path, contract, _ = source
    before = (path.read_bytes(), path.with_suffix('.json').read_bytes())
    with pytest.raises(ValueError, match='source|output'):
        run_vector_training(vector(4), path.parent, dict(contract, num_envs=4), resume=path,
                            allow_env_expansion=True, direct_training=True, timesteps=16)
    assert before == (path.read_bytes(), path.with_suffix('.json').read_bytes())


def test_expansion_requires_explicit_source(tmp_path):
    with pytest.raises(ValueError, match='requires resume'):
        run_vector_training(vector(4), tmp_path, dict(num_envs=4), allow_env_expansion=True)


def test_expansion_rejects_actual_target_count_different_from_contract(source):
    path, contract, _ = source
    with pytest.raises(ValueError, match='target environment count'):
        checkpoint.load_checkpoint(path, vector(3), dict(contract, num_envs=4),
                                   allow_env_expansion=True)


def test_external_source_progress_can_supply_expansion_history(source, tmp_path):
    path, contract, _ = source
    metadata = json.loads(path.with_suffix('.json').read_text())
    metadata.pop('training_progress')
    path.with_suffix('.json').write_text(json.dumps(metadata))
    report, model = run_vector_training(vector(4), tmp_path/'expanded', dict(contract, num_envs=4),
        resume=path, resume_episode_progress=path.parent/'training_progress.json',
        allow_env_expansion=True, direct_training=True, timesteps=16, eval_episodes=2)
    assert report['resumed_completed_episodes_per_env'] == [2, 2, 0, 0]
    assert model.num_timesteps == 32
