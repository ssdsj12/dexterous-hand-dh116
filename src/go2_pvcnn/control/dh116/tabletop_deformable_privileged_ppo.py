"""CPU opt-in structured Critic for the frozen legacy PPO implementation."""

from pathlib import Path
import uuid

import torch

from .tabletop_deformable_privileged_teacher import DeformablePrivilegedTeacher
from .tabletop_privileged_ppo import OnPolicyRollout, PPOTrainer, _cpu, _device, _finite, _index
from .tabletop_squashed_gaussian import squashed_log_prob
from .tabletop_stereo_contract import file_sha256


VERSION = "dh116-deformable-structured-critic-ppo-v1"
BASE_PPO_SHA256 = "f0a96f4f8c9548b5153d004013bef9d77b97250aa0bc43272626cffb15416854"


def verify_base_ppo_source():
    path = Path(__file__).with_name("tabletop_privileged_ppo.py")
    if file_sha256(path) != BASE_PPO_SHA256:
        raise ValueError("frozen base PPO source SHA256 differs")


class StructuredCriticBatch(dict):
    """An owned Tensor tree that supports the base trainer's minibatch syntax."""

    def __getitem__(self, key):
        if isinstance(key, torch.Tensor):
            return StructuredCriticBatch(_index(self, key))
        return super().__getitem__(key)

    def to(self, device):
        return _device(self, device)


class _StructuredRollout(OnPolicyRollout):
    def batch(self, *, resolve_actor=False):
        batch = super().batch(resolve_actor=resolve_actor)
        batch["critic_obs"] = StructuredCriticBatch(batch["critic_obs"])
        return batch


class DeformablePPOTrainer(PPOTrainer):
    """Independent CPU trainer using the original update, GAE and KL guard."""

    def __init__(self, model, config=None, *, rollout_steps=32, num_envs=1):
        if not isinstance(model, DeformablePrivilegedTeacher):
            raise ValueError("independent deformable Teacher model required")
        verify_base_ppo_source()
        super().__init__(model, config, rollout_steps=rollout_steps, num_envs=num_envs, device="cpu")

    def _new_rollout(self):
        self.rollout = _StructuredRollout(self.rollout_steps, self.num_envs,
            session_id=self.session_id, policy_version=self.policy_version, capability=self._capability)

    @torch.no_grad()
    def collect_step(self, actor_obs, critic_obs):
        if self._pending is not None:
            raise RuntimeError("pending transition must be recorded before another sample")
        if self.rollout.full:
            raise RuntimeError("complete rollout must be updated before collection")
        for name, observations in (("actor", actor_obs), ("critic", critic_obs)):
            common, nodes, _ = self.model._parts(observations)
            if len(common) != self.num_envs:
                raise ValueError(f"{name} observation environment count differs")
            if len(self.rollout) and self.rollout._rows[0][f"{name}_obs"]["rod_nodes"].shape != nodes.shape:
                raise ValueError(f"{name} rod node shape changed within this rollout")
        owned_actor, owned_critic = _cpu(actor_obs), _cpu(critic_obs)
        mu = self._actor_mean(owned_actor, self.num_envs)
        _finite("latent mean", mu)
        if mu.shape != (self.num_envs, 24):
            raise ValueError("policy latent mean must be [N,24]")
        std = self.model.log_std.exp().expand_as(mu)
        z = mu + std * torch.randn_like(mu)
        actions, log_prob = z.tanh(), squashed_log_prob(z, mu, std)
        value = self.model.value(_device(owned_critic, self.device))
        _finite("critic value", value)
        if value.shape != (self.num_envs,):
            raise ValueError("critic value must be [N]")
        token = uuid.uuid4().hex
        self._pending = dict(actor_obs=owned_actor, critic_obs=owned_critic,
            actions=_cpu(actions), latent_actions=_cpu(z), old_log_prob=_cpu(log_prob),
            old_mu=_cpu(mu), old_std=_cpu(std), values=_cpu(value), collection_token=token,
            session_id=self.session_id, policy_version=self.policy_version)
        return dict(actions=actions.clone(), latent_actions=z.clone(), log_prob=log_prob.clone(),
            latent_mean=mu.clone(), latent_std=std.clone(), value=value.clone(),
            collection_token=token, policy_version=self.policy_version)
