"""Bound the Gaussian mean in the same normalized range as finger actions."""
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class BoundedMeanPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Reuse the original layer/parameters already registered with the
        # optimizer. Tanh introduces no parameters and is serialized normally.
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())
