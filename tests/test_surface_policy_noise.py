"""CPU-only noise-contract tests; no training or simulator dependencies."""
import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from go2_pvcnn.control.dh116 import tabletop_policy_noise as noise


LEGACY_CONTRACT = dict(
    version='bounded_policy_noise_v1', min_std=.05, max_std=1.,
    parameter='log_std', projection='after_each_optimizer_step_and_resume',
    projected_adam_moments='zero_only_changed_coordinates',
    action_log_probability='unclipped_gaussian_sample')
MOMENTS = ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')


class DummyActor(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.num_actions = actions
        self.log_std = nn.Parameter(torch.full((actions,), math.log(.08)))
        self.actor = nn.Sequential(nn.Linear(3, actions), nn.Tanh())
        self.critic = nn.Linear(3, 1)

    def act(self, observations):
        self.distribution = Normal(self.actor(observations), self.log_std.exp())
        return self.distribution.sample()

    def log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(-1)


def algorithm(actions=24):
    policy = DummyActor(actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.01, amsgrad=True)
    # Populate real Adam moments for noise, actor and critic parameters.
    sum(parameter.sum() for parameter in policy.parameters()).backward()
    optimizer.step()
    return SimpleNamespace(policy=policy, optimizer=optimizer)


def snapshot(alg):
    return copy.deepcopy(dict(
        policy=alg.policy.state_dict(),
        gradients={name: parameter.grad for name, parameter in alg.policy.named_parameters()},
        optimizer=alg.optimizer.state_dict()))


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            assert_same(value, reference)
    else:
        assert actual == expected


def assert_only_noise_changed(alg, before, changed):
    expected = copy.deepcopy(before)
    expected['policy']['log_std'] = alg.policy.log_std.detach().clone()
    parameters = [p for group in alg.optimizer.param_groups for p in group['params']]
    parameter_ids = [i for group in expected['optimizer']['param_groups'] for i in group['params']]
    noise_id = next(i for p, i in zip(parameters, parameter_ids) if p is alg.policy.log_std)
    for name in MOMENTS:
        expected['optimizer']['state'][noise_id][name][changed] = 0.
    assert_same(snapshot(alg), expected)


@pytest.mark.parametrize('actions', [20, 24, 26])
def test_unconfigured_algorithm_keeps_exact_legacy_contract_and_projection(actions):
    alg = algorithm(actions)
    with torch.no_grad():
        alg.policy.log_std[:2].copy_(torch.tensor([math.log(.001), math.log(2.)]))
    before = snapshot(alg)
    changed = noise.project_policy_noise(alg)
    assert changed.tolist() == [True, True] + [False] * (actions - 2)
    torch.testing.assert_close(alg.policy.log_std[:2].exp(), torch.tensor([.05, 1.]))
    assert noise.CONTRACT == LEGACY_CONTRACT
    assert not hasattr(alg, 'policy_noise_contract')
    assert_only_noise_changed(alg, before, changed)
    assert not noise.project_policy_noise(alg).any()


def test_configure_sets_actual_initial_std_and_an_independent_versioned_contract():
    alg = algorithm()
    parameter = alg.policy.log_std
    with torch.no_grad():
        parameter[0] = math.log(.005)  # Already initialized: preserve this moment.
    before = snapshot(alg)
    contract = noise.configure_surface_noise(alg)
    assert contract is alg.policy_noise_contract
    assert alg.policy.log_std is parameter
    torch.testing.assert_close(parameter.exp(), torch.full_like(parameter, .005))
    assert alg.policy_noise_contract == dict(
        LEGACY_CONTRACT, version='bounded_surface_policy_noise_v2',
        initial_std=.005, min_std=.001, max_std=.3)
    assert alg.policy_noise_contract is not noise.CONTRACT
    assert noise.CONTRACT == LEGACY_CONTRACT
    assert_only_noise_changed(alg, before, torch.tensor([False] + [True] * 23))
    before = snapshot(alg)
    noise.configure_surface_noise(alg)
    assert_same(snapshot(alg), before)
    other = algorithm()
    noise.configure_surface_noise(other)
    assert other.policy_noise_contract is not alg.policy_noise_contract


@pytest.mark.parametrize('initial,minimum,maximum', [(.002, .0005, .02), (.004, .004, .004)])
def test_custom_configuration_accepts_inclusive_bounds(initial, minimum, maximum):
    alg = algorithm()
    noise.configure_surface_noise(alg, initial, minimum, maximum)
    torch.testing.assert_close(alg.policy.log_std.exp(), torch.full_like(alg.policy.log_std, initial))
    assert alg.policy_noise_contract == dict(
        LEGACY_CONTRACT, version='bounded_surface_policy_noise_v2',
        initial_std=initial, min_std=minimum, max_std=maximum)
    assert not noise.project_policy_noise(alg).any()


def test_surface_projection_uses_local_bounds_and_preserves_unrelated_moments():
    alg = algorithm()
    noise.configure_surface_noise(alg)
    # Refresh moments after configuration so selective clearing is observable.
    alg.optimizer.zero_grad()
    alg.policy.log_std.sum().backward()
    alg.optimizer.step()
    with torch.no_grad():
        alg.policy.log_std.copy_(torch.tensor(
            [math.log(.0001), math.log(.6)] + [math.log(.005)] * 22))
    before = snapshot(alg)
    changed = noise.project_policy_noise(alg)
    assert changed.tolist() == [True, True] + [False] * 22
    torch.testing.assert_close(alg.policy.log_std[:2].exp(), torch.tensor([.001, .3]))
    torch.testing.assert_close(alg.policy.log_std[2:].exp(), torch.full((22,), .005))
    assert_only_noise_changed(alg, before, changed)
    before = snapshot(alg)
    assert not noise.project_policy_noise(alg).any()
    assert_same(snapshot(alg), before)
    assert noise.CONTRACT == LEGACY_CONTRACT


@pytest.mark.parametrize('direction,bound', [(1., .001), (-1., .3)])
def test_existing_optimizer_hook_projects_surface_noise_after_each_real_step(direction, bound):
    alg = algorithm()
    # Install before opting in, just as the existing algorithm factory does.
    noise.install_noise_projection(alg)
    noise.configure_surface_noise(alg)
    for group in alg.optimizer.param_groups:
        group['lr'] = 10.
    for _ in range(3):
        alg.optimizer.zero_grad()
        (direction * alg.policy.log_std.sum()).backward()
        alg.optimizer.step()
        torch.testing.assert_close(alg.policy.log_std.exp(), torch.full_like(alg.policy.log_std, bound))
        for name in MOMENTS:
            assert not alg.optimizer.state[alg.policy.log_std][name].any()


@pytest.mark.parametrize('initial,minimum,maximum', [
    (0., .001, .3), (-.005, .001, .3), (.005, 0., .3), (.005, -.001, .3),
    (.0005, .001, .3), (.4, .001, .3), (.005, .3, .001), (.005, .001, 0.),
    *[(value if index == 0 else .005,
       value if index == 1 else .001,
       value if index == 2 else .3)
      for index in range(3) for value in (float('nan'), float('inf'), float('-inf'))],
])
@pytest.mark.parametrize('configured', [False, True])
def test_invalid_settings_reject_before_any_mutation(initial, minimum, maximum, configured):
    alg = algorithm()
    if configured:
        noise.configure_surface_noise(alg)
    old_contract = getattr(alg, 'policy_noise_contract', None)
    contract_before = copy.deepcopy(old_contract)
    before = snapshot(alg)
    with pytest.raises(ValueError):
        noise.configure_surface_noise(alg, initial, minimum, maximum)
    assert_same(snapshot(alg), before)
    assert getattr(alg, 'policy_noise_contract', None) is old_contract
    assert old_contract == contract_before
    assert hasattr(alg, 'policy_noise_contract') == configured
    assert noise.CONTRACT == LEGACY_CONTRACT


@pytest.mark.parametrize('actions', [0, 1, 20, 23, 25, 26])
def test_surface_configuration_requires_exactly_24_actions_without_mutation(actions):
    alg = algorithm(actions)
    before = snapshot(alg)
    with pytest.raises(ValueError, match='24'):
        noise.configure_surface_noise(alg)
    assert_same(snapshot(alg), before)
    assert not hasattr(alg, 'policy_noise_contract')


@pytest.mark.parametrize('configured', [False, True])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_log_std_still_rejected_without_mutation(configured, value):
    alg = algorithm()
    if configured:
        noise.configure_surface_noise(alg)
    with torch.no_grad():
        alg.policy.log_std[0] = value
    before = snapshot(alg)
    with pytest.raises(RuntimeError, match='nonfinite'):
        noise.project_policy_noise(alg)
    assert_same(snapshot(alg), before)


def test_configuration_rejects_nonfinite_existing_noise_without_attaching_contract():
    alg = algorithm()
    with torch.no_grad():
        alg.policy.log_std[0] = float('nan')
    before = snapshot(alg)
    with pytest.raises(RuntimeError, match='nonfinite'):
        noise.configure_surface_noise(alg)
    assert_same(snapshot(alg), before)
    assert not hasattr(alg, 'policy_noise_contract')


@pytest.mark.parametrize('configured', [False, True])
@pytest.mark.parametrize('operation', ['project', 'configure'])
def test_noise_updates_preserve_normal_distribution_raw_samples_and_log_probabilities(configured, operation):
    alg = algorithm()
    if configured:
        noise.configure_surface_noise(alg)
    # A mean near the actuator limit makes unclipped samples observable.
    with torch.no_grad():
        alg.policy.actor[0].weight.zero_()
        alg.policy.actor[0].bias.fill_(10.)
        alg.policy.log_std.fill_(math.log(2.))
    observations = torch.ones(64, 3)
    with torch.no_grad():
        raw = alg.policy.act(observations)
        distribution = alg.policy.distribution
        mean, scale = distribution.loc.clone(), distribution.scale.clone()
        saved_raw, saved_log_prob = raw.clone(), alg.policy.log_prob(raw).clone()
    assert (raw.abs() > 1.).any()
    if operation == 'project':
        assert noise.project_policy_noise(alg).all()
    else:
        noise.configure_surface_noise(alg)
    assert alg.policy.distribution is distribution
    assert_same(distribution.loc, mean)
    assert_same(distribution.scale, scale)
    assert_same(raw, saved_raw)
    assert_same(alg.policy.log_prob(raw), saved_log_prob)
    with torch.no_grad():
        new_raw = alg.policy.act(observations)
    assert type(alg.policy.distribution) is Normal
    expected_scale = alg.policy.log_std.exp().expand_as(new_raw)
    assert_same(alg.policy.distribution.scale, expected_scale)
    assert_same(alg.policy.log_prob(new_raw), Normal(mean, expected_scale).log_prob(new_raw).sum(-1))
