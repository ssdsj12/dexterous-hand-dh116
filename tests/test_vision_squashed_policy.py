"""Sensor-only squashed actor contract and deterministic mode."""
import importlib
import importlib.util

import pytest
import torch
from torch import nn
from tensordict import TensorDict


def policy_module():
    name = 'go2_pvcnn.control.dh116.tabletop_vision_squashed_policy'
    assert importlib.util.find_spec(name), 'separate squashed vision policy is missing'
    return importlib.import_module(name)


def observations(n=3):
    return TensorDict(dict(policy=torch.randn(n, 3844)*.1, critic=torch.randn(n, 371)), [n])


def test_actor_has_one_outer_squash_and_critic_isolation():
    p = policy_module()
    torch.set_num_threads(2)
    obs = observations()
    algorithm = p.make_squashed_vision_algorithm(obs, 2)
    policy = algorithm.policy
    assert isinstance(policy.actor.action_net[-1], nn.Identity)
    assert sum(isinstance(layer, nn.Tanh) for layer in policy.actor.action_net) == 2
    with torch.no_grad():
        policy.actor.action_net[-2].weight.zero_()
        policy.actor.action_net[-2].bias.fill_(2.)
        latent = policy.actor(obs['policy'])
        assert (latent == 2.).all()
        mode = policy.act_inference(obs)
        torch.testing.assert_close(mode, latent.tanh(), atol=0, rtol=0)
        altered = obs.clone()
        altered['critic'] += 10.
        torch.testing.assert_close(policy.act_inference(altered), mode, atol=0, rtol=0)
        assert not torch.equal(policy.evaluate(obs), policy.evaluate(altered))
    assert policy.distribution is None  # Deterministic inference draws no distribution samples.
    assert algorithm.policy_noise_contract['initial_std'] == .1
    assert algorithm.policy_noise_contract['action_log_probability'] == 'tanh_transformed_gaussian_at_saved_latent'


def test_sample_and_evaluate_latents_share_exact_likelihood_and_latent_parameters():
    p = policy_module()
    obs = observations()
    policy = p.make_squashed_vision_algorithm(obs, 1).policy
    sample = policy.sample_actions(obs)
    torch.testing.assert_close(sample['actions'], sample['latent_actions'].tanh(), atol=0, rtol=0)
    evaluated = policy.evaluate_latents(obs, sample['latent_actions'], entropy_epsilon=torch.zeros(3, 24))
    torch.testing.assert_close(sample['log_prob'], evaluated['log_prob'], atol=0, rtol=0)
    torch.testing.assert_close(policy.action_mean, sample['latent_mean'])
    torch.testing.assert_close(policy.action_std, sample['latent_std'])
    with pytest.raises(ValueError, match='latent'):
        policy.get_actions_log_prob(sample['actions'])
    with torch.no_grad():
        policy.actor.action_net[-2].bias[0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        policy.sample_actions(obs)


@pytest.mark.parametrize('actor_dim,critic_dim', [(3864,371), (3844,370)])
def test_factory_rejects_observation_contract_changes(actor_dim, critic_dim):
    p = policy_module()
    obs = TensorDict(dict(policy=torch.zeros(2,actor_dim), critic=torch.zeros(2,critic_dim)), [2])
    with pytest.raises(ValueError, match='3844'):
        p.make_squashed_vision_algorithm(obs, 1)


@pytest.mark.parametrize('log_std',[float('nan'),float('inf'),-1000.])
def test_sampling_rejects_invalid_noise_before_returning_commands(log_std):
    packet = observations()
    policy = policy_module().make_squashed_vision_algorithm(packet,1).policy
    with torch.no_grad():
        policy.log_std[0] = log_std
    with pytest.raises(ValueError,match='finite|positive'):
        policy.sample_actions(packet)


def test_actor_point_gradients_do_not_reach_privileged_critic():
    packet = observations()
    frames = packet['policy'].reshape(3,4,961)
    frames[...,768:896] = 1.
    policy = policy_module().make_squashed_vision_algorithm(packet,1).policy
    (policy.act_inference(packet)-.25).square().mean().backward()
    assert sum(p.grad.abs().sum() for p in policy.actor.point_net.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in policy.critic.parameters())


def test_unbounded_actor_preserves_legacy_weights_and_initialization_exactly():
    from go2_pvcnn.control.dh116.tabletop_vision_student import VisionStudentActor
    p = policy_module()
    torch.manual_seed(313)
    legacy = VisionStudentActor()
    torch.manual_seed(313)
    unbounded = p.UnboundedVisionStudentActor()
    for name,value in legacy.state_dict().items():
        torch.testing.assert_close(unbounded.state_dict()[name],value,atol=0,rtol=0)
    packet = observations()
    torch.testing.assert_close(unbounded(packet['policy']).tanh(),legacy(packet['policy']),atol=0,rtol=0)


@pytest.mark.parametrize('num_envs,steps,batches',[(1,8,1),(3,4,3),(10,2,8)])
def test_factory_preserves_legacy_logical_minibatch_step_budget(num_envs,steps,batches):
    algorithm = policy_module().make_squashed_vision_algorithm(observations(num_envs),steps)
    assert algorithm.num_learning_epochs == 5
    assert algorithm.num_mini_batches == batches
