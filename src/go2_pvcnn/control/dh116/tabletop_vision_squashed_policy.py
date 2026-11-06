"""Sensor-only 3844D actor / separate 371D critic, with one output tanh."""
import math

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic

from .tabletop_vision_student import POLICY_DIM, VisionStudentActor
from .tabletop_policy_noise import configure_surface_noise, install_noise_projection
from .tabletop_squashed_gaussian import (
    DISTRIBUTION_CONTRACT, require_finite, validate_parameters,
    squashed_log_prob, squashed_entropy,
)


class UnboundedVisionStudentActor(VisionStudentActor):
    def __init__(self, *, wrist_quaternion_preprocessing='raw'):
        # Preserve the existing actor's initialization (which indexes [-2]).
        super().__init__(wrist_quaternion_preprocessing=wrist_quaternion_preprocessing)
        self.action_net[-1] = nn.Identity()


class SquashedVisionActorCritic(ActorCritic):
    """action_mean/action_std always describe the latent Normal for PPO KL."""
    def _update_distribution(self, obs):
        mu = self.actor(obs)
        std = self.log_std.exp().expand_as(mu)
        validate_parameters(mu, std)
        self.distribution = Normal(mu, std)

    def _distribution_for(self, obs):
        self._update_distribution(self.actor_obs_normalizer(self.get_actor_obs(obs)))
        return self.distribution

    def sample_actions(self, obs):
        distribution = self._distribution_for(obs)
        z = distribution.sample()
        log_prob = squashed_log_prob(z, distribution.mean, distribution.stddev)
        return dict(latent_actions=z, actions=z.tanh(), log_prob=log_prob,
                    latent_mean=distribution.mean, latent_std=distribution.stddev)

    def evaluate_latents(self, obs, z, entropy_epsilon=None):
        distribution = self._distribution_for(obs)
        mu, std = distribution.mean, distribution.stddev
        if z.shape != mu.shape:
            raise ValueError('saved latent actions must match policy output shape')
        log_prob = squashed_log_prob(z, mu, std)
        if entropy_epsilon is None:
            entropy_epsilon = torch.randn_like(mu)
        return dict(log_prob=log_prob, entropy=squashed_entropy(mu, std, entropy_epsilon),
                    latent_entropy=distribution.entropy().sum(-1),
                    latent_mean=mu, latent_std=std)

    def act(self, obs, **kwargs):
        return self.sample_actions(obs)['actions']

    def act_inference(self, obs):
        mu = self.actor(self.actor_obs_normalizer(self.get_actor_obs(obs)))
        require_finite('latent mean', mu)
        return mu.tanh()

    def get_actions_log_prob(self, actions):
        raise ValueError('use evaluate_latents with saved latent actions; never invert bounded commands')

    @property
    def entropy(self):
        mu, std = self.action_mean, self.action_std
        return squashed_entropy(mu, std, torch.randn_like(mu))


def make_squashed_vision_algorithm(obs, steps, device='cpu', *,
                                    microbatch_size=128, inference_batch_size=128,
                                    wrist_quaternion_preprocessing='raw'):
    from .tabletop_vision_squashed_ppo import SquashedVisionPPO

    if (type(steps) is not int or steps < 1 or len(obs.batch_size) != 1
            or obs.batch_size[0] < 1 or obs['policy'].shape != (obs.batch_size[0], POLICY_DIM)
            or obs['critic'].shape != (obs.batch_size[0], 371)):
        raise ValueError('squashed vision PPO requires [N,3844] policy and [N,371] critic observations')
    obs = obs.to(device)
    policy = SquashedVisionActorCritic(
        obs, {'policy': ['policy'], 'critic': ['critic']}, 24,
        actor_hidden_dims=[128,128], critic_hidden_dims=[128,128], activation='tanh',
        init_noise_std=.1, noise_std_type='log',
    )
    policy.actor = UnboundedVisionStudentActor(wrist_quaternion_preprocessing=wrist_quaternion_preprocessing)
    policy = policy.to(device)
    for layer in policy.critic:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, math.sqrt(2.))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(policy.critic[-1].weight, 1.)
    algorithm = SquashedVisionPPO(
        policy, microbatch_size=microbatch_size, inference_batch_size=inference_batch_size,
        num_learning_epochs=5, num_mini_batches=min(8, len(obs)),
        learning_rate=3e-4, gamma=.995, lam=.95, clip_param=.2, entropy_coef=.005,
        value_loss_coef=.5, max_grad_norm=.5, schedule='adaptive', desired_kl=.02,
        device=str(device),
    )
    algorithm.init_storage('rl', len(obs), steps, obs, [24])
    configure_surface_noise(algorithm, initial_std=.1, min_std=.001, max_std=.3)
    algorithm.policy_noise_contract = dict(
        algorithm.policy_noise_contract,
        action_log_probability=DISTRIBUTION_CONTRACT['action_log_probability'])
    algorithm.distribution_contract = dict(DISTRIBUTION_CONTRACT)
    install_noise_projection(algorithm)
    return algorithm
