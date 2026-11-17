"""Independent privileged actor/critic; no simulator or legacy policy imports."""
import math

import torch
from torch import nn

from .tabletop_squashed_gaussian import squashed_log_prob, squashed_entropy

VERSION = 'dh116-privileged-teacher-tanh-v1'
OBSERVATION_SIZE, ACTION_SIZE = 371, 24


def _observation(value):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[-1] != OBSERVATION_SIZE:
        raise ValueError('finite [N,371] privileged observation required')
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError('finite floating [N,371] observation required')
    return value


class PrivilegedTeacher(nn.Module):
    def __init__(self, obs_dim=371, action_dim=24, hidden=(256, 256), initial_std=.1):
        super().__init__()
        if obs_dim != OBSERVATION_SIZE or action_dim != ACTION_SIZE or tuple(hidden) != (256, 256):
            raise ValueError('this version requires 371D/24D and hidden256/256')
        if not math.isfinite(initial_std) or not .001 <= initial_std <= .3:
            raise ValueError('initial std must be in [.001,.3]')
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), math.log(initial_std)))
        self.actor = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 256),
            nn.ELU(), nn.Linear(256, action_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 256),
            nn.ELU(), nn.Linear(256, 1))

    def latent_mean(self, observations):
        return self.actor(_observation(observations))

    def deterministic(self, observations):
        return self.latent_mean(observations).tanh()

    def value(self, observations):
        return self.critic(_observation(observations)).squeeze(-1)

    def sample(self, observations):
        mu = self.latent_mean(observations)
        std = self.log_std.exp().expand_as(mu)
        z = mu + std * torch.randn_like(mu)
        return dict(actions=z.tanh(), latent_actions=z, log_prob=squashed_log_prob(z, mu, std),
            latent_mean=mu, latent_std=std)

    def evaluate_latents(self, actor_obs, critic_obs, latent_actions):
        mu = self.latent_mean(actor_obs)
        if latent_actions.shape != mu.shape:
            raise ValueError('saved latent actions must match [N,24] policy output')
        std = self.log_std.exp().expand_as(mu)
        return dict(log_prob=squashed_log_prob(latent_actions, mu, std),
            entropy=squashed_entropy(mu, std, torch.randn_like(mu)),
            value=self.value(critic_obs), latent_mean=mu, latent_std=std)
