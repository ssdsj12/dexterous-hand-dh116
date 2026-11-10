"""On-policy tanh PPO with owned samples and separate actor/critic observations."""
from dataclasses import asdict, dataclass
import math
import uuid

import torch

from .tabletop_squashed_gaussian import squashed_log_prob, squashed_entropy, latent_gaussian_kl

VERSION = 'dh116-privileged-on-policy-ppo-v1'


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = .995
    lam: float = .95
    clip: float = .2
    lr: float = 3e-4
    epochs: int = 5
    minibatches: int = 8
    entropy_coef: float = .005
    value_coef: float = .5
    max_grad_norm: float = .5
    min_std: float = .001
    max_std: float = .3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError('finite numeric PPO config required: ' + name)
        if not (0 <= self.gamma <= 1 and 0 <= self.lam <= 1 and 0 < self.clip < 1
                and self.lr > 0 and self.entropy_coef >= 0 and self.value_coef >= 0
                and self.max_grad_norm > 0 and 0 < self.min_std <= self.max_std):
            raise ValueError('invalid PPO coefficient or noise bounds')
        if type(self.epochs) is not int or type(self.minibatches) is not int or min(self.epochs, self.minibatches) < 1:
            raise ValueError('positive integer epochs/minibatches required')


def _tree(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict) and value and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in value.items()):
        return {key: fn(item) for key, item in value.items()}
    raise ValueError('actor observations must be Tensor or nonempty dict[str,Tensor]')


def _cpu(value):
    return _tree(value, lambda t: t.detach().to(device='cpu',
        dtype=torch.float32 if t.is_floating_point() else t.dtype).clone())


def _device(value, device):
    return _tree(value, lambda t: t.to(device=device,
        dtype=torch.float32 if t.is_floating_point() else t.dtype))


def _index(value, indices):
    return _tree(value, lambda t: t.index_select(0, indices))


def _join(values):
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values).flatten(0, 1)
    keys = values[0].keys()
    if any(value.keys() != keys for value in values):
        raise ValueError('actor history keys changed during rollout')
    return {key: torch.stack([value[key] for value in values]).flatten(0, 1) for key in keys}


def _finite(name, value):
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError(name + ' must be finite tensor')
    return value


class OnPolicyRollout:
    """Only PPOTrainer owns its write capability; no external transition add API."""
    def __init__(self, rollout_steps, num_envs, *, session_id, policy_version, capability):
        self.rollout_steps, self.num_envs = rollout_steps, num_envs
        self.session_id, self.policy_version = session_id, policy_version
        self._capability, self._rows = capability, []

    def __len__(self):
        return len(self._rows)

    @property
    def full(self):
        return len(self) == self.rollout_steps

    def _append(self, row, capability):
        if capability is not self._capability:
            raise ValueError('on-policy collection capability required')
        if self.full:
            raise RuntimeError('rollout is already complete')
        if row['session_id'] != self.session_id or row['policy_version'] != self.policy_version:
            raise ValueError('collection session/policy version mismatch')
        self._rows.append(row)

    def batch(self):
        if not self._rows:
            raise RuntimeError('empty rollout')
        # Return owned clones so consumers cannot alter the update ledger.
        names = ('actor_obs', 'critic_obs', 'actions', 'latent_actions', 'old_log_prob',
            'old_mu', 'old_std', 'values', 'rewards', 'done', 'next_value', 'bootstrap_mask')
        return {name: _tree(_join([row[name] for row in self._rows]), lambda t: t.detach().clone()) for name in names}

    def returns_and_advantages(self, config):
        if not self.full:
            raise RuntimeError('complete rollout required for GAE')
        advantages = []
        carry = torch.zeros(self.num_envs)
        for row in reversed(self._rows):
            delta = row['rewards'] + config.gamma * row['next_value'] * row['bootstrap_mask'] - row['values']
            carry = delta + config.gamma * config.lam * (~row['done']).float() * carry
            advantages.append(carry.clone())
        advantage = torch.stack(list(reversed(advantages))).flatten()
        values = torch.stack([row['values'] for row in self._rows]).flatten()
        return (advantage + values).detach(), advantage.detach()


class PPOTrainer:
    def __init__(self, model, config=None, *, rollout_steps=32, num_envs=1, device='cpu'):
        if any(type(v) is not int or v < 1 for v in (rollout_steps, num_envs)):
            raise ValueError('positive integer rollout_steps/num_envs required')
        self.config = config or PPOConfig()
        self.model, self.device = model.to(device), torch.device(device)
        self.rollout_steps, self.num_envs = rollout_steps, num_envs
        if model.log_std.shape != (24,):
            raise ValueError('generic policy must expose 24D log_std')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        self.policy_version, self.session_id = 0, uuid.uuid4().hex
        self.counters = dict(environment_steps=0, updates=0)
        self._capability, self._pending = object(), None
        self._new_rollout()
        self.project_noise()

    def _new_rollout(self):
        self.rollout = OnPolicyRollout(self.rollout_steps, self.num_envs,
            session_id=self.session_id, policy_version=self.policy_version, capability=self._capability)

    @property
    def clean_boundary(self):
        return self._pending is None and len(self.rollout) == 0

    @torch.no_grad()
    def project_noise(self):
        parameter = self.model.log_std
        _finite('log_std', parameter)
        bounded = parameter.clamp(math.log(self.config.min_std), math.log(self.config.max_std))
        changed = parameter != bounded
        parameter.copy_(bounded)
        state = self.optimizer.state.get(parameter, {})
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if name in state:
                state[name].masked_fill_(changed, 0.)
        return changed

    @torch.no_grad()
    def collect_step(self, actor_obs, critic_obs):
        if self._pending is not None:
            raise RuntimeError('pending transition must be recorded before another sample')
        if self.rollout.full:
            raise RuntimeError('complete rollout must be updated before collection')
        def check(value):
            _finite('actor observation', value)
            if value.ndim < 2 or value.shape[0] != self.num_envs:
                raise ValueError('batched actor observation count mismatch')
            return value
        _tree(actor_obs, check)
        _finite('critic observation', critic_obs)
        if critic_obs.shape != (self.num_envs, 371):
            raise ValueError('critic observations must be [N,371]')
        owned_actor, owned_critic = _cpu(actor_obs), _cpu(critic_obs)
        mu = self.model.latent_mean(_device(owned_actor, self.device))
        _finite('latent mean', mu)
        if mu.shape != (self.num_envs, 24):
            raise ValueError('policy latent mean must be [N,24]')
        std = self.model.log_std.exp().expand_as(mu)
        z = mu + std * torch.randn_like(mu)
        actions, log_prob = z.tanh(), squashed_log_prob(z, mu, std)
        value = self.model.value(owned_critic.to(self.device))
        _finite('critic value', value)
        if value.shape != (self.num_envs,):
            raise ValueError('critic value must be [N]')
        token = uuid.uuid4().hex
        self._pending = dict(actor_obs=owned_actor, critic_obs=owned_critic,
            actions=_cpu(actions), latent_actions=_cpu(z), old_log_prob=_cpu(log_prob),
            old_mu=_cpu(mu), old_std=_cpu(std), values=_cpu(value), collection_token=token,
            session_id=self.session_id, policy_version=self.policy_version)
        return dict(actions=actions.clone(), latent_actions=z.clone(), log_prob=log_prob.clone(),
            latent_mean=mu.clone(), latent_std=std.clone(), value=value.clone(),
            collection_token=token, policy_version=self.policy_version)

    def record_step(self, reward, done, next_value, *, truncated=None, bootstrap_mask=None,
                    executed_actions=None, action_owner='policy', collection_token=None):
        if self._pending is None:
            raise RuntimeError('no pending policy sample')
        if action_owner != 'policy':
            raise ValueError('action owner must be the current policy')
        if collection_token is not None and collection_token != self._pending['collection_token']:
            raise ValueError('collection token mismatch')
        for name, value in (('reward', reward), ('next_value', next_value)):
            _finite(name, value)
            if value.shape != (self.num_envs,):
                raise ValueError(name + ' must be [N]')
        def boolean(name, value):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.shape != (self.num_envs,):
                raise ValueError(name + ' must be boolean [N]')
            return value.detach().cpu().clone()
        dones = boolean('done', done)
        trunc = torch.zeros_like(dones) if truncated is None else boolean('truncated', truncated)
        if bool((trunc & ~dones).any()):
            raise ValueError('truncated rows must also be done')
        if bool(trunc.any()) and bootstrap_mask is None:
            raise ValueError('truncation requires explicit terminal-observation bootstrap mask')
        bootstrap = ~dones if bootstrap_mask is None else boolean('bootstrap_mask', bootstrap_mask)
        if bool((bootstrap & dones & ~trunc).any()):
            raise ValueError('actual terminal success/failure cannot bootstrap')
        if executed_actions is not None:
            _finite('issued actions', executed_actions)
            issued = executed_actions.detach().to(device='cpu', dtype=torch.float32)
            if not torch.equal(issued, self._pending['actions']):
                raise ValueError('issued normalized actions differ from sampled policy actions')
        row = dict(self._pending, rewards=_cpu(reward), done=dones,
            next_value=_cpu(next_value), bootstrap_mask=bootstrap)
        self.rollout._append(row, self._capability)
        self._pending = None
        self.counters['environment_steps'] += self.num_envs

    @torch.no_grad()
    def _check_frozen_policy(self, batch):
        mu = self.model.latent_mean(_device(batch['actor_obs'], self.device))
        std = self.model.log_std.exp().expand_as(mu)
        z = batch['latent_actions'].to(self.device)
        actual = squashed_log_prob(z, mu, std)
        for name, current, old in (('mean', mu, batch['old_mu']), ('std', std, batch['old_std']),
                ('likelihood', actual, batch['old_log_prob'])):
            if not torch.allclose(current.cpu(), old, atol=2e-5, rtol=2e-5):
                raise RuntimeError('on-policy frozen ' + name + ' changed before PPO update')
        value = self.model.value(batch['critic_obs'].to(self.device))
        if not torch.allclose(value.cpu(), batch['values'], atol=2e-5, rtol=2e-5):
            raise RuntimeError('critic policy changed during on-policy rollout')
        # Check on the collection device; CPU/GPU tanh kernels may round differently.
        if not torch.equal(z.tanh().cpu(), batch['actions']):
            raise RuntimeError('saved action/latent sample mismatch')

    def update(self):
        if self._pending is not None:
            raise RuntimeError('cannot update with pending transition')
        if not self.rollout.full:
            raise RuntimeError('complete nonempty rollout required')
        batch = self.rollout.batch()
        self._check_frozen_policy(batch)
        returns, advantage = self.rollout.returns_and_advantages(self.config)
        advantage = ((advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)).detach()
        total, count = len(advantage), 0
        aggregate = dict(policy_loss=0., value_loss=0., entropy=0., kl=0.)
        for _ in range(self.config.epochs):
            for indices in torch.randperm(total).tensor_split(min(total, self.config.minibatches)):
                actor_obs = _device(_index(batch['actor_obs'], indices), self.device)
                critic_obs = batch['critic_obs'][indices].to(self.device)
                mu = self.model.latent_mean(actor_obs)
                std = self.model.log_std.exp().expand_as(mu)
                z = batch['latent_actions'][indices].to(self.device)
                log_prob = squashed_log_prob(z, mu, std)
                old_log = batch['old_log_prob'][indices].to(self.device)
                ratio = (log_prob - old_log).exp()
                adv = advantage[indices].to(self.device)
                policy_loss = -torch.minimum(ratio * adv,
                    ratio.clamp(1-self.config.clip, 1+self.config.clip) * adv).mean()
                value = self.model.value(critic_obs)
                old_value = batch['values'][indices].to(self.device)
                target = returns[indices].to(self.device)
                clipped_value = old_value + (value-old_value).clamp(-self.config.clip, self.config.clip)
                value_loss = torch.maximum((value-target).square(), (clipped_value-target).square()).mean()
                entropy = squashed_entropy(mu, std, torch.randn_like(mu)).mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                _finite('PPO loss', loss)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm,
                    error_if_nonfinite=True)
                self.optimizer.step()
                self.project_noise()
                with torch.no_grad():
                    kl = latent_gaussian_kl(batch['old_mu'][indices].to(self.device),
                        batch['old_std'][indices].to(self.device), mu, std).mean()
                for name, value in (('policy_loss', policy_loss), ('value_loss', value_loss),
                        ('entropy', entropy), ('kl', kl)):
                    aggregate[name] += float(value.detach())
                count += 1
        self.policy_version += 1
        self.counters['updates'] += 1
        self._new_rollout()
        return dict({key: value/count for key, value in aggregate.items()}, samples=total,
            optimized_samples=total*self.config.epochs, minibatch_updates=count,
            policy_version=self.policy_version)
