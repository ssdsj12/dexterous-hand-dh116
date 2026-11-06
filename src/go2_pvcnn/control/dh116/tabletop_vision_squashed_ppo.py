"""FP32 CPU saved-latent rollouts and sample-weighted squashed-policy PPO."""
import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage

from .tabletop_vision_ppo import VisionPPO
from .tabletop_squashed_gaussian import (
    require_finite, validate_parameters, latent_gaussian_kl,
)


STORAGE_CONTRACT = dict(
    version='dh116-tanh-latent-cpu-rollout-v1',
    actions='tanh_of_saved_latent', latent_actions='pre_tanh_normal_sample',
    actions_log_prob='tanh_transformed_gaussian_at_saved_latent',
    mu='latent_normal_mean', sigma='latent_normal_std',
    device='cpu', floating_dtype='float32',
)


class SquashedRolloutStorage(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.latent_actions = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_actions = torch.zeros_like(self.actions)

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError('Rollout buffer overflow; clear before reuse')
        z, actions = transition.latent_actions, transition.actions
        if z is None or z.shape != actions.shape:
            raise ValueError('transition requires matching saved latent actions')
        require_finite('saved latent actions', z)
        require_finite('bounded actions', actions)
        validate_parameters(transition.action_mean, transition.action_sigma)
        require_finite('old transformed log probability', transition.actions_log_prob)
        if not bool((actions.abs() <= 1.).all()) or not torch.equal(actions, z.tanh()):
            raise ValueError('bounded commands must equal tanh(saved latent actions)')
        self.latent_actions[self.step].copy_(z)
        super().add_transitions(transition)

    def clear(self):
        # Buffers are reused and completely overwritten in the next rollout.
        super().clear()


class SquashedVisionPPO(VisionPPO):
    def __init__(self, policy, **kwargs):
        super().__init__(policy, **kwargs)
        self.transition = SquashedRolloutStorage.Transition()
        self.last_update_diagnostics = {}
        self.storage_contract = dict(STORAGE_CONTRACT)

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        if training_type != 'rl':
            raise ValueError('squashed rollout storage requires reinforcement learning')
        self.storage = SquashedRolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, device='cpu')
        # Pin after allocating *all* buffers, including the saved latents.
        if self._pin_storage:
            self.storage.observations = self.storage.observations.pin_memory()
            for name, value in vars(self.storage).items():
                if isinstance(value, torch.Tensor):
                    setattr(self.storage, name, value.pin_memory())

    @torch.no_grad()
    def act(self, obs):
        collected = {name: [] for name in ('actions','latent_actions','values','actions_log_prob',
                                           'action_mean','action_sigma')}
        for start in range(0, len(obs), self.inference_batch_size):
            batch = obs[start:start+self.inference_batch_size]
            sample = self.policy.sample_actions(batch)
            for name, key in (('actions','actions'), ('latent_actions','latent_actions'),
                              ('actions_log_prob','log_prob'), ('action_mean','latent_mean'),
                              ('action_sigma','latent_std')):
                collected[name].append(sample[key])
            values = self.policy.evaluate(batch)
            require_finite('critic values', values)
            collected['values'].append(values)
        for name, values in collected.items():
            setattr(self.transition, name, torch.cat(values, 0).detach())
        self.transition.observations = obs
        return self.transition.actions

    def _observations(self, flat, indices):
        return TensorDict({key: value[indices].to(self.device) for key, value in flat.items()},
                          batch_size=[len(indices)], device=self.device)

    @torch.no_grad()
    def compute_returns(self, obs):
        # Preserve RSL's global GAE/sample-std normalization for ordinary
        # rollouts. A single sample has zero centered advantage and no sample
        # variance; explicitly handle that supported factory boundary.
        last_values = torch.cat([
            self.policy.evaluate(obs[start:start+self.inference_batch_size])
            for start in range(0,len(obs),self.inference_batch_size)
        ],0).cpu()
        singleton = self.storage.num_envs*self.storage.num_transitions_per_env == 1
        normalize = not self.normalize_advantage_per_mini_batch
        self.storage.compute_returns(last_values,self.gamma,self.lam,
                                     normalize_advantage=normalize and not singleton)
        if normalize and singleton:
            self.storage.advantages.zero_()
        if self._pin_storage:
            self.storage.advantages = self.storage.advantages.pin_memory()

    @torch.no_grad()
    def _frozen_log_prob_error(self, flat, fields, count):
        maximum = 0.
        # Evaluate the entire rollout before the *first* optimizer step.
        for start in range(0, count, self.microbatch_size):
            indices = torch.arange(start, min(start+self.microbatch_size, count))
            data = self.policy.evaluate_latents(
                self._observations(flat, indices), fields['latent_actions'][indices].to(self.device),
                entropy_epsilon=torch.zeros_like(fields['latent_actions'][indices], device=self.device))
            error = (data['log_prob']-fields['actions_log_prob'][indices].flatten().to(self.device)).abs()
            maximum = max(maximum, float(error.max()))
        return maximum

    def update(self):
        storage = self.storage
        if storage.step != storage.num_transitions_per_env:
            raise RuntimeError('squashed PPO requires a complete rollout before updating')
        count = storage.num_envs*storage.num_transitions_per_env
        if not 1 <= self.num_mini_batches <= count:
            raise ValueError('logical minibatch count exceeds collected samples')
        flat = storage.observations.flatten(0, 1)
        fields = {name: getattr(storage, name).flatten(0, 1) for name in (
            'actions','latent_actions','values','advantages','returns','actions_log_prob','mu','sigma')}
        for name, value in fields.items():
            require_finite(f'rollout {name}', value)
        diagnostics = self._rollout_diagnostics(fields)
        diagnostics['frozen_log_prob_abs_error_max'] = self._frozen_log_prob_error(flat, fields, count)
        # A single permutation is reused across epochs, including every remainder.
        groups = torch.tensor_split(torch.randperm(count), self.num_mini_batches)
        names = ('value_function','surrogate','entropy','latent_gaussian_entropy',
                 'approx_kl','analytic_kl_old_new','clip_fraction','new_log_prob_mean')
        totals = {name: 0. for name in names}
        total_samples = 0
        grad_total = 0.
        logp_min, logp_max = float('inf'), -float('inf')
        for _ in range(self.num_learning_epochs):
            for indices in groups:
                size = len(indices)
                # The full logical epsilon tensor is generated once, independently
                # of microbatch shape, and follows the same row order as saved z.
                epsilon = torch.randn(size, fields['latent_actions'].shape[-1], device='cpu')
                advantage = fields['advantages'][indices]
                if self.normalize_advantage_per_mini_batch:
                    # Match RSL's sample std except its undefined singleton case.
                    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=size>1)+1e-8)
                metrics = torch.zeros(len(names), device=self.device)
                self.optimizer.zero_grad()
                for start in range(0, size, self.microbatch_size):
                    stop = min(start+self.microbatch_size, size)
                    selected = indices[start:stop]
                    obs = self._observations(flat, selected)
                    data = {name: value[selected].to(self.device) for name, value in fields.items()}
                    evaluated = self.policy.evaluate_latents(
                        obs, data['latent_actions'], entropy_epsilon=epsilon[start:stop].to(self.device))
                    logp = evaluated['log_prob']
                    logratio = logp-data['actions_log_prob'].flatten()
                    ratio = logratio.exp()
                    require_finite('PPO likelihood ratio', ratio)
                    values = self.policy.evaluate(obs)
                    entropy = evaluated['entropy'].mean()
                    adv = advantage[start:stop].to(self.device).flatten()
                    surrogate = torch.maximum(-adv*ratio,
                        -adv*ratio.clamp(1.-self.clip_param,1.+self.clip_param)).mean()
                    if self.use_clipped_value_loss:
                        clipped_value = data['values']+(values-data['values']).clamp(-self.clip_param,self.clip_param)
                        value_loss = torch.maximum((values-data['returns']).square(),
                                                   (clipped_value-data['returns']).square()).mean()
                    else:
                        value_loss = (values-data['returns']).square().mean()
                    loss = surrogate+self.value_loss_coef*value_loss-self.entropy_coef*entropy
                    require_finite('PPO loss', loss)
                    weight = len(selected)/size
                    (loss*weight).backward()
                    with torch.no_grad():
                        kl = latent_gaussian_kl(data['mu'], data['sigma'],
                                                evaluated['latent_mean'], evaluated['latent_std']).mean()
                        approximate = (torch.expm1(logratio)-logratio).mean()
                        values_for_metrics = torch.stack((value_loss.detach(), surrogate.detach(),
                            entropy.detach(), evaluated['latent_entropy'].mean(), approximate, kl,
                            ((ratio-1.).abs()>self.clip_param).float().mean(), logp.mean()))
                        require_finite('PPO diagnostics', values_for_metrics)
                        metrics += values_for_metrics*weight
                        logp_min, logp_max = min(logp_min,float(logp.min())),max(logp_max,float(logp.max()))
                for parameter in self.policy.parameters():
                    if parameter.grad is not None:
                        require_finite('PPO gradient', parameter.grad)
                # One gradient clip, LR adaptation and Adam step per logical batch.
                norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm,
                                                error_if_nonfinite=True)
                self._adapt_learning_rate(float(metrics[names.index('analytic_kl_old_new')]))
                self.optimizer.step()
                for name, value in zip(names, metrics.tolist()):
                    totals[name] += value*size
                grad_total += float(norm)*size
                total_samples += size
        means = {name: value/total_samples for name,value in totals.items()}
        diagnostics.update({key:value for key,value in means.items()
                            if key not in ('value_function','surrogate','entropy')})
        diagnostics.update(bounded_policy_entropy=means['entropy'],
                           preclip_gradient_norm=grad_total/total_samples,
                           new_log_prob_min=logp_min, new_log_prob_max=logp_max)
        self.last_update_diagnostics = diagnostics
        storage.clear()
        return {name: means[name] for name in ('value_function','surrogate','entropy')}

    @staticmethod
    def _rollout_diagnostics(fields):
        actions, z = fields['actions'], fields['latent_actions']
        logp, mu, std = fields['actions_log_prob'], fields['mu'], fields['sigma']
        return dict(action_raw_abs_max=float(z.abs().max()),
            action_saturation_rate=float((actions.abs()>=.99).float().mean()),
            action_saturation_env_any_rate=float((actions.abs()>=.99).any(-1).float().mean()),
            bounded_action_mean=actions.mean(0).tolist(),
            bounded_action_std=actions.std(0,unbiased=False).tolist(),
            latent_mu_mean=float(mu.mean()), latent_mu_min=float(mu.min()), latent_mu_max=float(mu.max()),
            latent_std_mean=float(std.mean()), latent_std_min=float(std.min()), latent_std_max=float(std.max()),
            old_log_prob_mean=float(logp.mean()), old_log_prob_min=float(logp.min()),
            old_log_prob_max=float(logp.max()))
