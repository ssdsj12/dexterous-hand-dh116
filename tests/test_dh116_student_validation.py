"""Closed-loop validation uses actual terminal flags and never PPO updates."""
import importlib
import importlib.util
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def validator():
    name = 'go2_pvcnn.control.dh116.tabletop_student_validation'
    assert importlib.util.find_spec(name) is not None, 'student validator is not implemented'
    return importlib.import_module(name).validate_student


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(156, 20), nn.Tanh())
        self.seen = []

    def act_inference(self, obs):
        assert not self.training and not torch.is_grad_enabled()
        assert list(obs.batch_size) == [2]
        self.seen.append(obs['policy'].clone())
        return self.actor(obs['policy'])

    def act(self, obs):
        raise AssertionError('validation must never sample stochastic actions')


class Algorithm:
    def __init__(self):
        self.policy = Policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters())
        self.counters = {'iteration': 0, 'updates': 0}

    def act(self, obs):
        raise AssertionError('validation must not create rollout transitions')

    def update(self):
        raise AssertionError('validation must not optimize')

    def process_env_step(self, *args):
        raise AssertionError('validation must not write PPO storage')


class Env:
    n = 2
    device = 'cpu'

    def __init__(self, events):
        self.events = events
        self.steps = 0
        self.episodes = torch.full((2,), 100, dtype=torch.long)
        self.successes = torch.full((2,), 100, dtype=torch.long)

    def reset(self):
        raise AssertionError('production step handles reset boundaries')

    def step(self, actions):
        assert not actions.requires_grad
        assert actions.shape == (2, 20)
        event = self.events[min(self.steps, len(self.events) - 1)]
        self.steps += 1
        if isinstance(event, Exception):
            raise event
        done, success, failure = event
        # Returned observations model independently auto-reset slots.
        obs = torch.full((2, 156), float(self.steps))
        return obs, torch.ones(2), torch.tensor(done), {
            'success': torch.tensor(success), 'failure': torch.tensor(failure),
            'terminal_observation': torch.full((2, 156), -float(self.steps)),
        }


def args(tmp_path, budget=10):
    return SimpleNamespace(output=tmp_path, student_validation_steps=budget)


def test_three_fresh_episodes_pass_without_training_or_manual_resets(tmp_path):
    env = Env([([True, True], [True, True], [0, 0])])
    algorithm = Algorithm()
    before = {key: value.clone() for key, value in algorithm.policy.state_dict().items()}
    contract = {'physics_hz': 480, 'reward': {'success': 25}}
    passed, report = validator()(env, algorithm, args(tmp_path), contract, torch.zeros((2, 156)))
    assert passed is True and report['passed'] is True
    assert env.steps == 3
    assert report['episode_counts'] == [3, 3]
    assert report['success_counts'] == [3, 3]
    assert report['success_rate'] == 1.
    assert report['optimization_performed'] is False
    assert report['contract'] == contract
    assert len(report['actor_hash']) == 64
    assert algorithm.policy.training is True
    assert not algorithm.optimizer.state and algorithm.counters == {'iteration': 0, 'updates': 0}
    for key, value in algorithm.policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert json.loads((tmp_path / 'student_validation.json').read_text()) == report
    assert [float(obs[0, 0]) for obs in algorithm.policy.seen] == [0., 1., 2.]


def test_success_without_done_and_done_with_failure_do_not_count(tmp_path):
    env = Env([
        ([False, False], [True, True], [0, 0]),
        ([True, True], [True, False], [4, 0]),
    ])
    passed, report = validator()(env, Algorithm(), args(tmp_path), {}, torch.zeros((2, 156)))
    assert passed is False
    assert report['episode_counts'] == [3, 3]
    assert report['success_counts'] == [0, 0]
    assert report['success_rate'] == 0.
    assert env.steps == 4


def test_every_environment_must_complete_three_episodes(tmp_path):
    env = Env([([True, False], [True, False], [0, 0])])
    passed, report = validator()(env, Algorithm(), args(tmp_path, budget=4), {}, torch.zeros((2, 156)))
    assert passed is False and env.steps == 4
    assert report['episode_counts'] == [4, 0]
    assert report['success_counts'] == [4, 0]
    assert report['status'] == 'budget_exhausted'


def test_at_least_three_actual_successes_are_required(tmp_path):
    env = Env([
        ([True, True], [True, True], [0, 0]),
        ([True, True], [False, False], [2, 2]),
    ])
    passed, report = validator()(env, Algorithm(), args(tmp_path), {}, torch.zeros((2, 156)))
    assert passed is False and report['successes'] == 2
    assert report['episodes'] == 6
    assert report['status'] == 'insufficient_successes'


def test_guard_exception_is_reported_and_reraised_without_optimization(tmp_path):
    fault = RuntimeError('tensor physical safety failure: live_velocity')
    env = Env([([True, True], [True, True], [0, 0]), fault])
    algorithm = Algorithm()
    with pytest.raises(RuntimeError, match='live_velocity') as caught:
        validator()(env, algorithm, args(tmp_path), {}, torch.zeros((2, 156)))
    assert caught.value is fault
    report = json.loads((tmp_path / 'student_validation.json').read_text())
    assert report['passed'] is False and report['status'] == 'error'
    assert report['success_counts'] == [1, 1]
    assert report['optimization_performed'] is False
    assert algorithm.policy.training is True and not algorithm.optimizer.state


def test_terminal_measured_channels_are_retained_when_available(tmp_path):
    class MeasuredEnv(Env):
        def step(self, actions):
            values = super().step(actions)
            values[3]['physical_evidence'] = {
                'object_bottom_m': torch.tensor([.331, .332]),
                'normal_vertical_force_n': torch.tensor([2., 3.]),
                'lift_vertical_force_n': torch.tensor([4., 5.]),
            }
            return values

    env = MeasuredEnv([([True, True], [True, True], [0, 0])])
    passed, report = validator()(env, Algorithm(), args(tmp_path), {}, torch.zeros((2, 156)))
    assert passed
    terminal = report['terminal_events'][0]
    assert terminal['object_bottom_m'] == pytest.approx(.331)
    assert terminal['normal_vertical_force_n'] == 2.
    assert terminal['lift_vertical_force_n'] == 4.


def test_malformed_success_evidence_cannot_pass(tmp_path):
    env = Env([([True, True], [.5, 1.], [0, 0])])
    with pytest.raises(ValueError, match='success'):
        validator()(env, Algorithm(), args(tmp_path), {}, torch.zeros((2, 156)))
    report = json.loads((tmp_path / 'student_validation.json').read_text())
    assert report['passed'] is False


def test_eval_mode_is_restored_and_actor_hash_identifies_weights(tmp_path):
    algorithm = Algorithm()
    algorithm.policy.eval()
    env = Env([([True, True], [True, True], [0, 0])])
    _, first = validator()(env, algorithm, args(tmp_path), {}, torch.zeros((2, 156)))
    assert algorithm.policy.training is False
    with torch.no_grad():
        algorithm.policy.actor[0].weight[0, 0] += .01
    env = Env([([True, True], [True, True], [0, 0])])
    _, second = validator()(env, algorithm, args(tmp_path), {}, torch.zeros((2, 156)))
    assert first['actor_hash'] != second['actor_hash']
