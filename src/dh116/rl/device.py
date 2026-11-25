from dh116.env.contracts import source_file
"""Device rollout/PPO overrides plus exact Teacher policy/Adam/RNG checkpoint state."""
import copy
import math
from copy import deepcopy as devic_ppo__deepcopy
import torch as devic_rollo__torch
from copy import deepcopy as tenso_check__deepcopy
import math as check_state__math
from pathlib import Path as devic_ppo__Path
from dh116.rl.ppo import OnPolicyRollout
from dataclasses import asdict
import random as check_state__random
import random as devic_ppo__random
from dh116.rl.ppo import _tree
import math as tenso_check__math
import numpy as check_state__np
import uuid as devic_ppo__uuid
from dh116.rl.ppo import _join
import os
import torch as check_state__torch
import numpy as devic_ppo__np
from dh116.rl.ppo import _index as devic_rollo___index
from pathlib import Path as tenso_check__Path

def check_state___sha(value):
    return isinstance(value, str) and len(value) == 64 and all((c in '0123456789abcdef' for c in value))
import torch as devic_ppo__torch
from dh116.rl.ppo import _device
import random as tenso_check__random

def check_state___clean(trainer, history_reset):
    if history_reset != 'clear':
        raise ValueError('explicit history_reset=clear required; no history reconstruction')
    if not trainer.clean_boundary:
        raise RuntimeError('checkpoint requires clean PPO update boundary; no pending rollout')
from dh116.env.contracts import runti_io__file_sha256 as devic_ppo__file_sha256
devic_rollo__VERSION = 'dh116-device-on-policy-rollout-v1'
from tempfile import NamedTemporaryFile

def check_state___finite_tree(value):
    if isinstance(value, check_state__torch.Tensor):
        if not bool(check_state__torch.isfinite(value).all()):
            raise ValueError('checkpoint contains nonfinite tensor')
    elif isinstance(value, dict):
        for item in value.values():
            check_state___finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            check_state___finite_tree(item)
from dh116.rl.teacher import TensorNativeTeacher as devic_ppo__TensorNativeTeacher
_NAMES = ('actor_obs', 'critic_obs', 'actions', 'latent_actions', 'old_log_prob', 'old_mu', 'old_std', 'values', 'rewards', 'done', 'next_value', 'bootstrap_mask')
import uuid as tenso_check__uuid

def check_state___cpu_tree(value):
    if isinstance(value, check_state__torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: check_state___cpu_tree(v) for (k, v) in value.items()}
    if isinstance(value, list):
        return [check_state___cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple((check_state___cpu_tree(v) for v in value))
    return copy.deepcopy(value)

def own_device_tree(value, device):
    """Detach+clone on an already matching device; never stage a CPU copy."""
    device = devic_rollo__torch.device(device)

    def own(tensor):
        if tensor.device != device:
            raise ValueError('rollout tensors must already be on the collection device')
        return tensor.detach().to(dtype=devic_rollo__torch.float32 if tensor.is_floating_point() else tensor.dtype).clone()
    return _tree(value, own)
import numpy as tenso_check__np

def check_state___validate_adam(optimizer, model, config, updates):
    groups = optimizer.get('param_groups')
    parameters = list(model.parameters())
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError('one complete Adam parameter group required')
    ids = groups[0].get('params')
    if ids != list(range(len(parameters))) or groups[0].get('lr') != config.lr:
        raise ValueError('Adam parameter order/config mismatch')
    state = optimizer['state']
    if any((type(identifier) is not int or identifier not in ids for identifier in state)):
        raise ValueError('unknown Adam parameter identity')
    if updates > 0 and set(state) != set(ids):
        raise ValueError('updated checkpoint must retain all Adam states')
    for (identifier, entry) in state.items():
        if not isinstance(entry, dict) or not {'step', 'exp_avg', 'exp_avg_sq'}.issubset(entry):
            raise ValueError('incomplete Adam moments')
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if name in entry:
                moment = entry[name]
                parameter = parameters[identifier]
                if not isinstance(moment, check_state__torch.Tensor) or moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                    raise ValueError('Adam moment shape/dtype mismatch')
        step = entry['step']
        if not isinstance(step, check_state__torch.Tensor) or step.numel() != 1 or (not check_state__math.isfinite(step.item())) or (step.item() < 0):
            raise ValueError('invalid Adam step counter')

class StructuredCriticBatch(dict):
    """An owned Tensor tree that supports the base trainer's minibatch syntax."""

    def __getitem__(self, key):
        if isinstance(key, devic_rollo__torch.Tensor):
            return StructuredCriticBatch(devic_rollo___index(self, key))
        return super().__getitem__(key)

    def to(self, device):
        return _device(self, device)
import torch as tenso_check__torch

def check_state___validate_rng(rng):
    if not isinstance(rng, dict) or not {'torch', 'python', 'numpy', 'cuda'}.issubset(rng):
        raise ValueError('complete RNG checkpoint required')
    try:
        check_state__torch.Generator(device='cpu').set_state(rng['torch'])
        check_state__random.Random().setstate(rng['python'])
        check_state__np.random.RandomState().set_state(rng['numpy'])
    except (ValueError, RuntimeError, TypeError) as error:
        raise ValueError('invalid checkpoint RNG state') from error
    if not isinstance(rng['cuda'], list):
        raise ValueError('CUDA RNG list required')
    if rng['cuda']:
        if not check_state__torch.cuda.is_available() or len(rng['cuda']) != check_state__torch.cuda.device_count():
            raise ValueError('checkpoint CUDA RNG device contract unavailable')
        for state in rng['cuda']:
            if not isinstance(state, check_state__torch.Tensor) or state.dtype != check_state__torch.uint8 or state.ndim != 1:
                raise ValueError('invalid CUDA RNG state')

class DeviceOnPolicyRollout(OnPolicyRollout):
    """Same append/batch/GAE protocol; time-major flatten and terminal math.

    The append capability/session/policy gate is unchanged. Append additionally
    owns all tensor leaves, so a native reader reusing its frame cannot rewrite
    an earlier transition. ``batch`` also returns owned tensors. Structured
    critic batches preserve the original trainer's ``batch[indices].to`` API.
    """

    def __init__(self, rollout_steps, num_envs, *, session_id, policy_version, capability, actor_storage=None, device='cpu'):
        if any((type(v) is not int or v < 1 for v in (rollout_steps, num_envs))):
            raise ValueError('positive integer rollout_steps/num_envs required')
        self.device = devic_rollo__torch.empty(0, device=device).device
        super().__init__(rollout_steps, num_envs, session_id=session_id, policy_version=policy_version, capability=capability, actor_storage=actor_storage)

    def _append(self, row, capability):
        if capability is not self._capability:
            raise ValueError('on-policy collection capability required')
        if self.full:
            raise RuntimeError('rollout is already complete')
        if row['session_id'] != self.session_id or row['policy_version'] != self.policy_version:
            raise ValueError('collection session/policy version mismatch')
        owned = dict(row)
        for name in _NAMES:

            def check(tensor):
                if tensor.ndim < 1 or tensor.shape[0] != self.num_envs:
                    raise ValueError('rollout tensor environment count differs: ' + name)
                return tensor
            _tree(row[name], check)
            owned[name] = own_device_tree(row[name], self.device)
            if self._rows:
                previous = self._rows[0][name]
                current = owned[name]
                if isinstance(previous, dict) != isinstance(current, dict):
                    raise ValueError('rollout tensor schema changed: ' + name)
                before = previous if isinstance(previous, dict) else {name: previous}
                after = current if isinstance(current, dict) else {name: current}
                if before.keys() != after.keys() or any((before[k].shape != after[k].shape or before[k].dtype != after[k].dtype for k in before)):
                    raise ValueError('rollout tensor shape/dtype schema changed: ' + name)
        for name in ('done', 'bootstrap_mask'):
            if not isinstance(owned[name], devic_rollo__torch.Tensor) or owned[name].shape != (self.num_envs,) or owned[name].dtype != devic_rollo__torch.bool:
                raise ValueError(name + ' must be boolean [N]')
        for name in ('rewards', 'values', 'next_value', 'old_log_prob'):
            if not isinstance(owned[name], devic_rollo__torch.Tensor) or owned[name].shape != (self.num_envs,) or owned[name].dtype != devic_rollo__torch.float32:
                raise ValueError(name + ' must be float32 [N]')
        super()._append(owned, capability)

    def batch(self, *, resolve_actor=False):
        if not self._rows:
            raise RuntimeError('empty rollout')
        result = {name: _tree(_join([row[name] for row in self._rows]), lambda t: t.detach().clone()) for name in _NAMES}
        if resolve_actor and self.actor_storage is not None:
            result['actor_obs'] = own_device_tree(self.actor_storage.resolve(result['actor_obs'], self.device), self.device)
        if isinstance(result['critic_obs'], dict):
            result['critic_obs'] = StructuredCriticBatch(result['critic_obs'])
        return result

    def returns_and_advantages(self, config):
        if not self.full:
            raise RuntimeError('complete rollout required for GAE')
        advantages = []
        carry = devic_rollo__torch.zeros(self.num_envs, device=self.device, dtype=devic_rollo__torch.float32)
        for row in reversed(self._rows):
            delta = row['rewards'] + config.gamma * row['next_value'] * row['bootstrap_mask'] - row['values']
            carry = delta + config.gamma * config.lam * (~row['done']).float() * carry
            advantages.append(carry.clone())
        advantage = devic_rollo__torch.stack(list(reversed(advantages))).flatten()
        values = devic_rollo__torch.stack([row['values'] for row in self._rows]).flatten()
        return ((advantage + values).detach(), advantage.detach())

def check_state___rng_state():
    (algorithm, keys, position, has_gaussian, cached) = check_state__np.random.get_state()
    return dict(torch=check_state__torch.get_rng_state(), python=check_state__random.getstate(), numpy=(algorithm, check_state__torch.as_tensor(keys.astype(check_state__np.int64)), position, has_gaussian, cached), cuda=[])
devic_ppo__DeviceOnPolicyRollout = DeviceOnPolicyRollout
devic_ppo__own_device_tree = own_device_tree

def check_state___decode_rng(value):
    if not isinstance(value, dict) or set(value) != {'torch', 'python', 'numpy', 'cuda'} or value['cuda'] != []:
        raise ValueError('complete CPU RNG checkpoint required')
    numpy = value['numpy']
    if not isinstance(numpy, (tuple, list)) or len(numpy) != 5:
        raise ValueError('complete NumPy RNG state required')
    (algorithm, keys, position, has_gaussian, cached) = numpy
    if algorithm != 'MT19937' or type(position) is not int or (not 0 <= position <= 624) or (type(has_gaussian) is not int) or (has_gaussian not in (0, 1)) or (type(cached) not in (int, float)) or (not check_state__math.isfinite(cached)) or (not isinstance(keys, check_state__torch.Tensor)) or (keys.dtype != check_state__torch.int64) or (keys.shape != (624,)) or bool(((keys < 0) | (keys > 2 ** 32 - 1)).any()):
        raise ValueError('invalid NumPy RNG key array')
    rng = dict(value, numpy=(algorithm, keys.numpy().astype(check_state__np.uint32), position, has_gaussian, cached))
    check_state___validate_rng(rng)
    return rng
from dh116.rl.ppo import PPOTrainer

def check_state___same_adam_setting(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(value) == len(expected) and all((check_state___same_adam_setting(left, right) for (left, right) in zip(value, expected)))
    return value == expected
from dh116.rl.ppo import _finite
from dh116.rl.ppo import _index as devic_ppo___index
tenso_check___rng_state = check_state___rng_state
tenso_check___decode_rng = check_state___decode_rng
tenso_check___same_adam_setting = check_state___same_adam_setting
tenso_check___clean = check_state___clean
tenso_check___cpu_tree = check_state___cpu_tree
tenso_check___finite_tree = check_state___finite_tree
tenso_check___sha = check_state___sha
tenso_check___validate_adam = check_state___validate_adam
tenso_check___validate_rng = check_state___validate_rng
from dh116.rl.ppo import squashed_log_prob as squashed_log_prob
from dh116.rl.ppo import squashed_entropy as squashed_entropy
from dh116.rl.ppo import latent_gaussian_kl as latent_gaussian_kl
devic_ppo__VERSION = 'dh116-device-deformable-ppo-v1'
BASE_PPO_SHA256 = '2175a41dde5560137bb1f1856049ec46c112c454f7acbe9e84e5e6a7654687e0'

def verify_base_ppo_source():
    path = source_file('rl/ppo.py')
    if devic_ppo__file_sha256(path) != BASE_PPO_SHA256:
        raise ValueError('frozen base PPO source SHA256 differs')

def _statistics(value):
    """Materialize only the final returned telemetry, never a training batch."""
    if isinstance(value, devic_ppo__torch.Tensor):
        if value.ndim == 0:
            return float(value)
        return [_statistics(item) for item in value]
    if isinstance(value, dict):
        return {key: _statistics(item) for (key, item) in value.items()}
    if isinstance(value, list):
        return [_statistics(item) for item in value]
    return value

class DeviceDeformablePPOTrainer(PPOTrainer):
    """Use the original model and public trainer API with device-owned samples.

    The audit path retains the exact full-rollout KL checks. Production
    collectors opt into a sampled KL guard before each Adam step.
    """

    def __init__(self, model, config=None, *, rollout_steps=32, num_envs=1, device='cpu', fast_update=False):
        if not isinstance(model, devic_ppo__TensorNativeTeacher):
            raise ValueError('production tensor Teacher model required')
        if type(fast_update) is not bool:
            raise TypeError('fast_update must be a boolean')
        self._fast_update = fast_update
        verify_base_ppo_source()
        super().__init__(model, config, rollout_steps=rollout_steps, num_envs=num_envs, device=device)

    def _new_rollout(self):
        self.device = self.model.log_std.device
        self.rollout = devic_ppo__DeviceOnPolicyRollout(self.rollout_steps, self.num_envs, session_id=self.session_id, policy_version=self.policy_version, capability=self._capability, device=self.device)

    def _actor_mean(self, actor_obs, count):
        return self.model.latent_mean(self._actor_input(actor_obs))

    @devic_ppo__torch.no_grad()
    def collect_step(self, actor_obs, critic_obs):
        if self._pending is not None:
            raise RuntimeError('pending transition must be recorded before another sample')
        if self.rollout.full:
            raise RuntimeError('complete rollout must be updated before collection')
        for (name, observations) in (('actor', actor_obs), ('critic', critic_obs)):
            (common, nodes, _) = self.model._parts(observations)
            if len(common) != self.num_envs:
                raise ValueError(f'{name} observation environment count differs')
            if len(self.rollout) and self.rollout._rows[0][f'{name}_obs']['rod_nodes'].shape != nodes.shape:
                raise ValueError(f'{name} rod node shape changed within this rollout')
        owned_actor = devic_ppo__own_device_tree(actor_obs, self.device)
        owned_critic = devic_ppo__own_device_tree(critic_obs, self.device)
        mu = self._actor_mean(owned_actor, self.num_envs)
        _finite('latent mean', mu)
        if mu.shape != (self.num_envs, 24):
            raise ValueError('policy latent mean must be [N,24]')
        std = self.model.log_std.exp().expand_as(mu)
        z = mu + std * devic_ppo__torch.randn_like(mu)
        (actions, log_prob) = (z.tanh(), squashed_log_prob(z, mu, std))
        value = self.model.value(owned_critic)
        _finite('critic value', value)
        if value.shape != (self.num_envs,):
            raise ValueError('critic value must be [N]')
        token = devic_ppo__uuid.uuid4().hex
        self._pending = dict(actor_obs=owned_actor, critic_obs=owned_critic, actions=devic_ppo__own_device_tree(actions, self.device), latent_actions=devic_ppo__own_device_tree(z, self.device), old_log_prob=devic_ppo__own_device_tree(log_prob, self.device), old_mu=devic_ppo__own_device_tree(mu, self.device), old_std=devic_ppo__own_device_tree(std, self.device), values=devic_ppo__own_device_tree(value, self.device), collection_token=token, session_id=self.session_id, policy_version=self.policy_version)
        return dict(actions=actions.clone(), latent_actions=z.clone(), log_prob=log_prob.clone(), latent_mean=mu.clone(), latent_std=std.clone(), value=value.clone(), collection_token=token, policy_version=self.policy_version)

    def record_step(self, reward, done, next_value, *, truncated=None, bootstrap_mask=None, executed_actions=None, action_owner='policy', collection_token=None):
        if self._pending is None:
            raise RuntimeError('no pending policy sample')
        if action_owner != 'policy':
            raise ValueError('action owner must be the current policy')
        if collection_token is not None and collection_token != self._pending['collection_token']:
            raise ValueError('collection token mismatch')
        for (name, value) in (('reward', reward), ('next_value', next_value)):
            _finite(name, value)
            if value.shape != (self.num_envs,):
                raise ValueError(name + ' must be [N]')

        def boolean(name, value):
            if not isinstance(value, devic_ppo__torch.Tensor) or value.dtype != devic_ppo__torch.bool or value.shape != (self.num_envs,):
                raise ValueError(name + ' must be boolean [N]')
            return devic_ppo__own_device_tree(value, self.device)
        dones = boolean('done', done)
        trunc = devic_ppo__torch.zeros_like(dones) if truncated is None else boolean('truncated', truncated)
        if bool((trunc & ~dones).any()):
            raise ValueError('truncated rows must also be done')
        if bool(trunc.any()) and bootstrap_mask is None:
            raise ValueError('truncation requires explicit terminal-observation bootstrap mask')
        bootstrap = ~dones if bootstrap_mask is None else boolean('bootstrap_mask', bootstrap_mask)
        if bool((bootstrap & dones & ~trunc).any()):
            raise ValueError('actual terminal success/failure cannot bootstrap')
        if executed_actions is not None:
            _finite('issued actions', executed_actions)
            issued = devic_ppo__own_device_tree(executed_actions, self.device)
            if not devic_ppo__torch.equal(issued, self._pending['actions']):
                raise ValueError('issued normalized actions differ from sampled policy actions')
        row = dict(self._pending, rewards=devic_ppo__own_device_tree(reward, self.device), done=dones, next_value=devic_ppo__own_device_tree(next_value, self.device), bootstrap_mask=bootstrap)
        self.rollout._append(row, self._capability)
        self._pending = None
        self.counters['environment_steps'] += self.num_envs

    @devic_ppo__torch.no_grad()
    def _check_frozen_policy(self, batch):
        mu = self.model.latent_mean(self._actor_input(batch['actor_obs']))
        std = self.model.log_std.exp().expand_as(mu)
        z = batch['latent_actions']
        actual = squashed_log_prob(z, mu, std)
        errors = {}
        for (name, current, old) in (('mean', mu, batch['old_mu']), ('std', std, batch['old_std']), ('log_probability', actual, batch['old_log_prob'])):
            errors[name] = (current - old).abs().max().double()
            if not devic_ppo__torch.allclose(current, old, atol=2e-05, rtol=2e-05):
                label = 'likelihood' if name == 'log_probability' else name
                raise RuntimeError('on-policy frozen ' + label + ' changed before PPO update')
        value = self.model.value(batch['critic_obs'])
        errors['value'] = (value - batch['values']).abs().max().double()
        if not devic_ppo__torch.allclose(value, batch['values'], atol=2e-05, rtol=2e-05):
            raise RuntimeError('critic policy changed during on-policy rollout')
        if not devic_ppo__torch.equal(z.tanh(), batch['actions']):
            raise RuntimeError('saved action/latent sample mismatch')
        return {f'max_abs_{key}_error': value for (key, value) in errors.items()}

    @devic_ppo__torch.no_grad()
    def _minibatch_exact_kl(self, batch, indices):
        if indices.device != self.device:
            raise ValueError('exact KL indices must remain on the policy device')
        mu = self.model.latent_mean(self._actor_input(devic_ppo___index(batch['actor_obs'], indices)))
        std = self.model.log_std.exp().expand_as(mu)
        total = latent_gaussian_kl(batch['old_mu'][indices], batch['old_std'][indices], mu, std).clamp_min(0.0).sum()
        return total.double() / len(indices)

    def _optimizer_step(self, batch):
        if not self.config.kl_backtrack_steps:
            self.optimizer.step()
            self.project_noise()
            return (self.config.lr, [])
        policy = devic_ppo__deepcopy(self.model.state_dict())
        optimizer = devic_ppo__deepcopy(self.optimizer.state_dict())
        rng = dict(torch=devic_ppo__torch.get_rng_state().clone(), python=devic_ppo__random.getstate(), numpy=devic_ppo__deepcopy(devic_ppo__np.random.get_state()), cuda=devic_ppo__torch.cuda.get_rng_state_all() if devic_ppo__torch.cuda.is_initialized() else None)

        def rollback():
            self.model.load_state_dict(policy)
            self.optimizer.load_state_dict(devic_ppo__deepcopy(optimizer))
            devic_ppo__torch.set_rng_state(rng['torch'])
            devic_ppo__random.setstate(rng['python'])
            devic_ppo__np.random.set_state(rng['numpy'])
            if rng['cuda'] is not None:
                devic_ppo__torch.cuda.set_rng_state_all(rng['cuda'])
        all_rows = devic_ppo__torch.arange(len(batch['values']), device=self.device)
        attempts = []
        for attempt in range(self.config.kl_backtrack_steps + 1):
            lr = self.config.lr * 0.1 ** attempt
            self.optimizer.param_groups[0]['lr'] = lr
            try:
                self.optimizer.step()
                self.project_noise()
                kl = self._minibatch_exact_kl(batch, all_rows)
            except Exception:
                rollback()
                raise
            attempts.append(dict(lr=lr, exact_full_rollout_kl=kl))
            if kl <= self.config.target_kl:
                self.optimizer.param_groups[0]['lr'] = self.config.lr
                return (lr, attempts)
            rollback()
        return (None, attempts)

    def _minibatch_policy(self, batch, indices):
        torch = devic_ppo__torch
        entropy_noise = torch.randn(len(indices), 24, device=self.device, dtype=batch['old_mu'].dtype)
        actor_obs = self._actor_input(devic_ppo___index(batch['actor_obs'], indices))
        mu = self.model.latent_mean(actor_obs)
        std = self.model.log_std.exp().expand_as(mu)
        try:
            log_prob = squashed_log_prob(batch['latent_actions'][indices], mu, std)
            entropy = squashed_entropy(mu, std, entropy_noise).mean()
        except ValueError:
            if self._fast_update:
                return None
            raise
        return mu, std, log_prob - batch['old_log_prob'][indices], entropy

    def _minibatch_losses(self, batch, indices, returns, advantage, ratio, entropy):
        torch = devic_ppo__torch
        clip = self.config.clip
        adv = advantage[indices]
        policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
        value = self.model.value(batch['critic_obs'][indices])
        old_value, target = batch['values'][indices], returns[indices]
        clipped_value = old_value + (value - old_value).clamp(-clip, clip)
        value_loss = torch.maximum((value - target).square(), (clipped_value - target).square()).mean()
        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        return loss, (policy_loss, value_loss, entropy)

    def _finish_update(self, accepted_updates):
        if accepted_updates:
            self.policy_version += 1
            self.counters['updates'] += 1
        self._new_rollout()

    def _update_fast(self, batch, returns, advantage):
        torch = devic_ppo__torch
        total = len(advantage)
        target_kl = self.config.target_kl
        minibatch_count = min(total, self.config.minibatches)
        loss_sum = torch.zeros(3, dtype=torch.float64, device=self.device)
        kl_sum = 0.0
        accepted_updates = optimized_samples = 0
        stop_reason = early_stop_kl = None
        for _ in range(self.config.epochs):
            for indices in torch.randperm(total, device=self.device).tensor_split(minibatch_count):
                self.optimizer.zero_grad(set_to_none=True)
                policy = self._minibatch_policy(batch, indices)
                if policy is None:
                    stop_reason = 'nonfinite_policy'
                    break
                _, _, log_ratio, entropy = policy
                frozen_log_ratio = log_ratio.detach()
                approx_kl = float((frozen_log_ratio.expm1() - frozen_log_ratio).mean().clamp_min(0))
                if not math.isfinite(approx_kl):
                    stop_reason = 'nonfinite_kl'
                    break
                if target_kl is not None and approx_kl > target_kl:
                    stop_reason = 'target_kl'
                    early_stop_kl = approx_kl
                    break
                loss, losses = self._minibatch_losses(batch, indices, returns, advantage, log_ratio.exp(), entropy)
                if not bool(torch.isfinite(loss)):
                    stop_reason = 'nonfinite_loss'
                    break
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                if not bool(torch.isfinite(grad_norm)):
                    stop_reason = 'nonfinite_gradient'
                    break
                self.optimizer.step()
                self.project_noise()
                loss_sum += torch.stack(losses).detach().double()
                kl_sum += approx_kl
                accepted_updates += 1
                optimized_samples += len(indices)
            if stop_reason is not None:
                break
        self.optimizer.zero_grad(set_to_none=True)
        self._finish_update(accepted_updates)
        policy_loss, value_loss, entropy = (loss_sum / accepted_updates).tolist() if accepted_updates else (None, None, None)
        return dict(
            policy_loss=policy_loss, value_loss=value_loss, entropy=entropy,
            kl=kl_sum / accepted_updates if accepted_updates else None,
            ppo_update_mode='sampled_kl', samples=total, optimized_samples=optimized_samples,
            minibatch_updates=accepted_updates, policy_version=self.policy_version,
            target_kl=target_kl, early_stopped=stop_reason is not None,
            stop_reason=stop_reason, early_stop_kl=early_stop_kl,
        )

    def update(self):
        if self._pending is not None:
            raise RuntimeError('cannot update with pending transition')
        if not self.rollout.full:
            raise RuntimeError('complete nonempty rollout required')
        batch = self.rollout.batch(resolve_actor=False)
        frozen_check = None if self._fast_update else self._check_frozen_policy(batch)
        (returns, advantage) = self.rollout.returns_and_advantages(self.config)
        advantage = ((advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-08)).detach()
        if self._fast_update:
            return self._update_fast(batch, returns, advantage)
        total = len(advantage)
        (trace, epoch_trace) = ([], [])
        early_stopped = False
        (accepted_updates, optimized_samples) = (0, 0)
        aggregate = {name: devic_ppo__torch.zeros((), dtype=devic_ppo__torch.float64, device=self.device) for name in ('policy_loss', 'value_loss', 'entropy', 'kl')}
        for epoch in range(self.config.epochs):
            epoch_start = len(trace)
            minibatches = devic_ppo__torch.randperm(total, device=self.device).tensor_split(min(total, self.config.minibatches))
            for (minibatch, indices) in enumerate(minibatches):
                self.optimizer.zero_grad(set_to_none=True)
                mu, std, log_ratio, entropy = self._minibatch_policy(batch, indices)
                ratio = log_ratio.exp()
                loss, (policy_loss, value_loss, entropy) = self._minibatch_losses(batch, indices, returns, advantage, ratio, entropy)
                _finite('PPO loss', loss)
                loss.backward()
                with devic_ppo__torch.no_grad():
                    kl = latent_gaussian_kl(batch['old_mu'][indices], batch['old_std'][indices], mu, std).mean()
                    sampled_approx_kl = (devic_ppo__torch.expm1(log_ratio) - log_ratio).mean().double()
                    ratio_clipped_fraction = ((ratio - 1.0).abs() > self.config.clip).float().mean().double()
                metrics = {name: item.detach().double() for (name, item) in (('policy_loss', policy_loss), ('value_loss', value_loss), ('entropy', entropy), ('kl', kl))}
                norm = devic_ppo__torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
                (accepted_lr, guard_attempts) = self._optimizer_step(batch)
                target = self.config.target_kl
                post_kl = None if target is None or accepted_lr is None else self._minibatch_exact_kl(batch, indices)
                trace.append(dict(epoch=epoch, minibatch=minibatch, samples=len(indices), pre_step_exact_kl=metrics['kl'], post_step_exact_kl=post_kl, sampled_approx_kl=sampled_approx_kl, entropy=metrics['entropy'], probability_ratio_clip_fraction=ratio_clipped_fraction, gradient_norm_before_clip=norm.double(), accepted_lr=accepted_lr, kl_guard_attempts=guard_attempts))
                if accepted_lr is None:
                    early_stopped = True
                    break
                accepted_updates += 1
                optimized_samples += len(indices)
                for (name, item) in metrics.items():
                    aggregate[name] += item
                if target is not None and (not guard_attempts) and (post_kl > target):
                    early_stopped = True
                    break
            epoch_rows = trace[epoch_start:]
            epoch_samples = sum((row['samples'] for row in epoch_rows))
            epoch_trace.append(dict(policy_epoch_index=epoch, attempted_minibatches=len(epoch_rows), accepted_minibatches=sum((row['accepted_lr'] is not None for row in epoch_rows)), exact_full_rollout_kl=self._minibatch_exact_kl(batch, devic_ppo__torch.arange(total, device=self.device)), sample_weighted_pre_step_entropy=sum((row['entropy'] * row['samples'] for row in epoch_rows)) / epoch_samples, early_stopped=early_stopped, action_groups={name: dict(indices=list(range(lo, hi)), log_std=self.model.log_std.detach()[lo:hi].clone(), latent_std=self.model.log_std.detach().exp()[lo:hi]) for (name, lo, hi) in (('fingers12', 0, 12), ('translation6', 12, 18), ('rotation6', 18, 24))}))
            if early_stopped:
                break
        self._finish_update(accepted_updates)
        return _statistics(dict({key: value / accepted_updates if accepted_updates else None for (key, value) in aggregate.items()}, samples=total, optimized_samples=optimized_samples, minibatch_updates=accepted_updates, policy_version=self.policy_version, target_kl=self.config.target_kl, early_stopped=early_stopped, frozen_policy_check=frozen_check, minibatch_trace=trace, epoch_trace=epoch_trace))
tenso_check__verify_base_ppo_source = verify_base_ppo_source
tenso_check__DeviceDeformablePPOTrainer = DeviceDeformablePPOTrainer
from dh116.env.contracts import runti_io__ROOT as ROOT
from dh116.env.contracts import runti_io__digest as digest
from dh116.env.contracts import runti_io__file_sha256 as tenso_check__file_sha256
from dh116.physics.solver import TensorBackendProfile
from dh116.rl.teacher import TensorNativeTeacher as tenso_check__TensorNativeTeacher
tenso_check__VERSION = 'dh116-tensor-native-checkpoint-v2'
_KEYS = {'version', 'identity', 'policy', 'optimizer', 'policy_version', 'counters', 'rng'}
_SOURCES = ('rl/device.py','rl/teacher.py','rl/ppo.py','rl/reward.py','physics/runtime.py','physics/solver.py','physics/geometry.py','hand/articulation.py','hand/kinematics.py','hand/safety.py','env/contracts.py','env/scene.py','env/lifecycle.py')

def identity(trainer, *, recipe_sha256, receipt_sha256, backend_profile, seed_sha256, policy_seed):
    """Bind the exact tensor namespace, execution recipe and current source bytes."""
    if type(trainer) is not tenso_check__DeviceDeformablePPOTrainer or type(trainer.model) is not tenso_check__TensorNativeTeacher:
        raise ValueError('actual tensor Teacher and device PPO trainer required')
    if any((not tenso_check___sha(value) for value in (recipe_sha256, receipt_sha256, seed_sha256))) or type(policy_seed) is not int or (not 0 <= policy_seed < 2 ** 64):
        raise ValueError('complete recipe/receipt/seed SHA and policy seed required')
    if type(backend_profile) is not TensorBackendProfile or trainer.model._backend_profile != backend_profile:
        raise ValueError('same actual flat backend profile required')
    backend_profile.verify()
    tenso_check__verify_base_ppo_source()
    model = trainer.model.model_identity
    grasp = model.get('grasp')
    if grasp not in ('pinch', 'tripod', 'power') or model.get('checkpoint_namespace') != 'tensor_native_teacher_' + grasp + '_v1' or model.get('policy_version') != 'dh116-tensor-native-context-teacher-v1' or (model.get('model_identity_sha256') != digest({k: v for (k, v) in model.items() if k != 'model_identity_sha256'})) or (model.get('parent_num_envs') != trainer.num_envs) or (model.get('backend_profile_sha256') != backend_profile.digest) or (model.get('backend_sha256') != backend_profile.binary_sha256) or any((model.get(key) is not False for key in ('native_ready', 'teacher_admission', 'grasp_qualified'))):
        raise ValueError('tensor model namespace/profile identity differs')
    for (name, sha) in model['dependency_source_sha256'].items():
        if tenso_check__file_sha256(source_file(name)) != sha:
            raise ValueError('tensor model dependency source SHA differs: ' + name)
    sources = {name: tenso_check__file_sha256(source_file(name)) for name in _SOURCES}
    sources['rl/training.py'] = tenso_check__file_sha256(source_file('rl/training.py'))
    return dict(version=tenso_check__VERSION, model=model, recipe_sha256=recipe_sha256, receipt_sha256=receipt_sha256, seed_sha256=seed_sha256, policy_seed=policy_seed, backend_profile=asdict(backend_profile), ppo_config=asdict(trainer.config), ppo_update_mode='sampled_kl' if trainer._fast_update else 'audited', rollout_steps=trainer.rollout_steps, num_envs=trainer.num_envs, device_type=trainer.device.type, cuda_rng_devices=tenso_check__torch.cuda.device_count() if tenso_check__torch.cuda.is_initialized() else 0, source_sha256=sources, torch_version=str(tenso_check__torch.__version__), history_reset='clear', physical_state_saved=False, restart_requires_verified_reset=True, teacher_admission=False, grasp_qualified=False)

def _same_tree(left, right):
    if isinstance(left, tenso_check__torch.Tensor):
        return isinstance(right, tenso_check__torch.Tensor) and left.shape == right.shape and (left.dtype == right.dtype) and tenso_check__torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all((_same_tree(v, right[k]) for (k, v) in left.items()))
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all((_same_tree(a, b) for (a, b) in zip(left, right)))
    return type(left) is type(right) and left == right

def _validated_rng(encoded, expected_devices):
    if not isinstance(encoded, dict) or set(encoded) != {'torch', 'python', 'numpy', 'cuda'}:
        raise ValueError('complete tensor checkpoint RNG required')
    cuda = encoded['cuda']
    if not isinstance(cuda, list) or len(cuda) != expected_devices:
        raise ValueError('tensor checkpoint CUDA RNG device contract differs')
    rng = tenso_check___decode_rng(dict(encoded, cuda=[]))
    rng['cuda'] = cuda
    tenso_check___validate_rng(rng)
    for (index, state) in enumerate(cuda):
        try:
            tenso_check__torch.Generator(device='cuda:' + str(index)).set_state(state)
        except (ValueError, RuntimeError, TypeError) as error:
            raise ValueError('invalid tensor checkpoint CUDA RNG state') from error
    return rng

def _validate(packet, trainer, expected_identity):
    if not isinstance(packet, dict) or set(packet) != _KEYS or packet['version'] != tenso_check__VERSION or (packet['identity'] != expected_identity):
        raise ValueError('tensor checkpoint source/recipe/model namespace identity differs')
    tenso_check___finite_tree(packet)
    (counts, version) = (packet['counters'], packet['policy_version'])
    samples = trainer.num_envs * trainer.rollout_steps
    if not isinstance(counts, dict) or set(counts) != {'updates', 'environment_steps'} or any((type(v) is not int or v < 0 for v in counts.values())) or (type(version) is not int) or (version != counts['updates']) or counts['environment_steps'] % samples or (counts['environment_steps'] < counts['updates'] * samples):
        raise ValueError('complete consistent tensor PPO counters required')
    (saved, current) = (packet['policy'], trainer.model.state_dict())
    if not isinstance(saved, dict) or saved.keys() != current.keys():
        raise ValueError('complete tensor policy topology required')
    for (key, value) in current.items():
        if isinstance(value, tenso_check__torch.Tensor):
            if not isinstance(saved[key], tenso_check__torch.Tensor) or saved[key].shape != value.shape or saved[key].dtype != value.dtype:
                raise ValueError('tensor policy topology differs')
        elif saved[key] != value:
            raise ValueError('tensor policy model metadata differs')
    noise = saved['log_std']
    if bool(((noise < tenso_check__math.log(trainer.config.min_std)) | (noise > tenso_check__math.log(trainer.config.max_std))).any()):
        raise ValueError('tensor policy noise outside declared bounds')
    adam = packet['optimizer']
    if not isinstance(adam, dict) or set(adam) != {'state', 'param_groups'} or (not isinstance(adam['state'], dict)):
        raise ValueError('complete tensor Adam checkpoint required')
    tenso_check___validate_adam(adam, trainer.model, trainer.config, counts['updates'])
    for entry in adam['state'].values():
        if set(entry) != {'step', 'exp_avg', 'exp_avg_sq'} or entry['step'].ndim != 0 or entry['step'].item() % 1 or (entry['step'].item() < counts['updates']) or bool((entry['exp_avg_sq'] < 0).any()):
            raise ValueError('complete consistent tensor Adam moments/steps required')
    rng = _validated_rng(packet['rng'], expected_identity['cuda_rng_devices'])
    model = tenso_check__deepcopy(trainer.model)
    model.load_state_dict(saved, strict=True)
    optimizer = tenso_check__torch.optim.Adam(model.parameters(), lr=trainer.config.lr)
    (stored_group,) = adam['param_groups']
    (expected_group,) = optimizer.param_groups
    if stored_group.keys() != expected_group.keys() or any((not tenso_check___same_adam_setting(stored_group[k], v) for (k, v) in expected_group.items() if k != 'params')):
        raise ValueError('tensor Adam parameter-group settings differ')
    optimizer.load_state_dict(adam)
    return rng

def _report(path, packet):
    return dict(path=str(path), sha256=tenso_check__file_sha256(path), version=tenso_check__VERSION, identity=tenso_check__deepcopy(packet['identity']), counters=tenso_check__deepcopy(packet['counters']), policy_version=packet['policy_version'], physical_state_saved=False, restart_requires_verified_reset=True, teacher_admission=False, grasp_qualified=False)

def save_checkpoint(path, trainer, **context):
    """Publish an atomic exclusive snapshot without changing live state or RNG."""
    tenso_check___clean(trainer, 'clear')
    path = tenso_check__Path(path)
    if path.exists():
        raise FileExistsError(path)
    expected = identity(trainer, **context)
    if trainer.rollout.session_id != trainer.session_id or trainer.rollout.policy_version != trainer.policy_version or trainer.rollout._capability is not trainer._capability:
        raise ValueError('active tensor rollout session/policy identity differs')
    rng = tenso_check___rng_state()
    rng['cuda'] = tenso_check__torch.cuda.get_rng_state_all() if tenso_check__torch.cuda.is_initialized() else []
    packet = dict(version=tenso_check__VERSION, identity=expected, policy=tenso_check___cpu_tree(trainer.model.state_dict()), optimizer=tenso_check___cpu_tree(trainer.optimizer.state_dict()), policy_version=trainer.policy_version, counters=tenso_check__deepcopy(trainer.counters), rng=rng)
    _validate(packet, trainer, expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = tenso_check__Path(stream.name)
        try:
            tenso_check__torch.save(packet, stream)
            stream.flush()
            os.fsync(stream.fileno())
            if not trainer._fast_update:
                serialized = tenso_check__torch.load(temporary, map_location='cpu', weights_only=True)
                if not _same_tree(packet, serialized):
                    raise ValueError('serialized tensor checkpoint changed policy/Adam/counters/RNG')
                _validate(serialized, trainer, expected)
            os.link(temporary, path)
        finally:
            temporary.unlink()
    return _report(path, packet)

def load_checkpoint(path, trainer, *, expected_sha256, **context):
    """Restore validated policy/Adam/counters and all RNG; require physical Reset.

    A new session, append capability and empty rollout invalidate old sample
    identities. Native/source/reward histories are intentionally not restored.
    """
    tenso_check___clean(trainer, 'clear')
    path = tenso_check__Path(path)
    if not tenso_check___sha(expected_sha256) or tenso_check__file_sha256(path) != expected_sha256:
        raise ValueError('actual complete tensor checkpoint SHA differs')
    expected = identity(trainer, **context)
    packet = tenso_check__torch.load(path, map_location='cpu', weights_only=True)
    rng = _validate(packet, trainer, expected)
    trainer.model.load_state_dict(packet['policy'], strict=True)
    trainer.optimizer.load_state_dict(packet['optimizer'])
    trainer.counters = tenso_check__deepcopy(packet['counters'])
    trainer.policy_version = packet['policy_version']
    (trainer.session_id, trainer._capability) = (tenso_check__uuid.uuid4().hex, object())
    trainer._pending = None
    trainer._new_rollout()
    tenso_check__torch.set_rng_state(rng['torch'])
    tenso_check__random.setstate(rng['python'])
    tenso_check__np.random.set_state(rng['numpy'])
    if rng['cuda']:
        tenso_check__torch.cuda.set_rng_state_all(rng['cuda'])
    return dict(_report(path, packet), session_id=trainer.session_id)
