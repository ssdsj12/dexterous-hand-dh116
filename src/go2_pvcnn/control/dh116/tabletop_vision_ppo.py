"""FP32 CPU rollouts and bounded microbatch PPO for the visual student.

The eight logical PPO minibatches and optimizer-step frequency are preserved.
Only the forward/backward work inside each logical minibatch is split. Global
GAE and advantage normalization use the installed RSL implementation on CPU.
"""

import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage


class VisionPPO(PPO):
    """Feedforward PPO with CPU storage and sample-weighted gradient accumulation."""

    def __init__(self, policy, *, microbatch_size=128, inference_batch_size=128, **kwargs):
        for name, value in (('microbatch_size', microbatch_size),
                            ('inference_batch_size', inference_batch_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if (policy.is_recurrent or policy.state_dependent_std
                or kwargs.get('rnd_cfg') or kwargs.get('symmetry_cfg') or kwargs.get('multi_gpu_cfg')):
            raise ValueError('vision CPU PPO supports feedforward fixed-std single-GPU training only')
        super().__init__(policy, **kwargs)
        self.microbatch_size = microbatch_size
        self.inference_batch_size = inference_batch_size
        self._pin_storage = torch.device(self.device).type == 'cuda'

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        if training_type != 'rl':
            raise ValueError('vision CPU rollout storage is for reinforcement learning')
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, device='cpu')
        if self._pin_storage:
            self.storage.observations = self.storage.observations.pin_memory()
            for name, value in vars(self.storage).items():
                if isinstance(value, torch.Tensor):
                    setattr(self.storage, name, value.pin_memory())

    @torch.no_grad()
    def act(self, obs):
        collected = {name: [] for name in ('actions', 'values', 'actions_log_prob',
                                           'action_mean', 'action_sigma')}
        for start in range(0, len(obs), self.inference_batch_size):
            batch = obs[start:start+self.inference_batch_size]
            actions = self.policy.act(batch)
            collected['actions'].append(actions)
            collected['values'].append(self.policy.evaluate(batch))
            collected['actions_log_prob'].append(self.policy.get_actions_log_prob(actions))
            collected['action_mean'].append(self.policy.action_mean)
            collected['action_sigma'].append(self.policy.action_std)
        for name, values in collected.items():
            setattr(self.transition, name, torch.cat(values, dim=0).detach())
        self.transition.observations = obs
        return self.transition.actions

    @torch.no_grad()
    def compute_returns(self, obs):
        last_values = torch.cat([
            self.policy.evaluate(obs[start:start+self.inference_batch_size])
            for start in range(0, len(obs), self.inference_batch_size)
        ], dim=0).cpu()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch)
        # RSL creates a fresh normalized tensor rather than filling its old buffer.
        if self._pin_storage:
            self.storage.advantages = self.storage.advantages.pin_memory()

    def _adapt_learning_rate(self, kl_mean):
        if self.desired_kl is None or self.schedule != 'adaptive':
            return
        if kl_mean > self.desired_kl * 2.:
            self.learning_rate = max(1e-5, self.learning_rate/1.5)
        elif 0. < kl_mean < self.desired_kl/2.:
            self.learning_rate = min(1e-2, self.learning_rate*1.5)
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate

    def update(self):
        storage = self.storage
        if storage.step != storage.num_transitions_per_env:
            raise RuntimeError('vision PPO requires a complete rollout before updating')
        batch_size = storage.num_envs * storage.num_transitions_per_env
        if not 1 <= self.num_mini_batches <= batch_size:
            raise ValueError('logical minibatch count exceeds the collected sample count')
        flat = storage.observations.flatten(0, 1)
        fields = {name: getattr(storage, name).flatten(0, 1) for name in (
            'actions', 'values', 'advantages', 'returns', 'actions_log_prob', 'mu', 'sigma')}
        # Match RSL's fixed permutation across epochs. Keep remainder samples too.
        groups = torch.tensor_split(torch.randperm(batch_size), self.num_mini_batches)
        totals = dict(value_function=0., surrogate=0., entropy=0.)
        for _ in range(self.num_learning_epochs):
            for indices in groups:
                logical_size = len(indices)
                advantage = fields['advantages'][indices]
                if self.normalize_advantage_per_mini_batch:
                    advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
                metrics = torch.zeros(4, device=self.device)
                self.optimizer.zero_grad()
                for start in range(0, logical_size, self.microbatch_size):
                    stop = min(start+self.microbatch_size, logical_size)
                    selected = indices[start:stop]
                    obs = TensorDict(
                        {key: value[selected].to(self.device) for key, value in flat.items()},
                        batch_size=[len(selected)], device=self.device)
                    data = {name: value[selected].to(self.device) for name, value in fields.items()}
                    # Gaussian samples stay unclipped in storage and in old log probabilities.
                    self.policy.act(obs)
                    log_prob = self.policy.get_actions_log_prob(data['actions'])
                    values = self.policy.evaluate(obs)
                    entropy = self.policy.entropy.mean()
                    ratio = torch.exp(log_prob-data['actions_log_prob'].flatten())
                    adv = advantage[start:stop].to(self.device).flatten()
                    unclipped = -adv*ratio
                    clipped = -adv*ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
                    surrogate = torch.maximum(unclipped, clipped).mean()
                    if self.use_clipped_value_loss:
                        bounded = data['values']+(values-data['values']).clamp(
                            -self.clip_param, self.clip_param)
                        value_loss = torch.maximum(
                            (values-data['returns']).square(),
                            (bounded-data['returns']).square()).mean()
                    else:
                        value_loss = (values-data['returns']).square().mean()
                    weight = len(selected)/logical_size
                    loss = surrogate+self.value_loss_coef*value_loss-self.entropy_coef*entropy
                    (loss*weight).backward()
                    with torch.no_grad():
                        sigma, mu = self.policy.action_std, self.policy.action_mean
                        kl = (torch.log(sigma/data['sigma']+1e-5)
                              +(data['sigma'].square()+(data['mu']-mu).square())
                              /(2.*sigma.square())-.5).sum(-1).mean()
                        metrics += torch.stack((value_loss.detach(), surrogate.detach(),
                                                entropy.detach(), kl))*weight
                # LR only affects the Adam step, so reducing full-batch KL here is
                # equivalent to RSL adapting it before the loss calculation.
                self._adapt_learning_rate(float(metrics[3]))
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                for name, value in zip(totals, metrics[:3].tolist()):
                    totals[name] += value
        updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {name: value/updates for name, value in totals.items()}
