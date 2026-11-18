"""Exclusive checkpoints for the new policy, with explicit contract identity."""
import copy
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .tabletop_surface_contract import ACTION_CONTRACT
from .tabletop_privileged_ppo import VERSION as PPO_VERSION
from .tabletop_privileged_teacher import VERSION as POLICY_VERSION

VERSION = 'dh116-privileged-teacher-checkpoint-v1'


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def build_identity(config, *, state_contract_sha256, source_sha256):
    if not _sha(state_contract_sha256) or not isinstance(source_sha256, dict):
        raise ValueError('371D state manifest SHA and source identities required')
    if any(not isinstance(k, str) or not _sha(v) for k, v in source_sha256.items()):
        raise ValueError('invalid source SHA identity')
    source = {}
    for name in ('teacher', 'ppo', 'reward', 'dataset', 'checkpoint'):
        path = Path(__file__).with_name('tabletop_privileged_' + name + '.py')
        key = 'src/go2_pvcnn/control/dh116/' + path.name
        source[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        if key in source_sha256 and source_sha256[key] != source[key]:
            raise ValueError('current implementation source SHA mismatch')
    source.update(source_sha256)
    return dict(policy_version=POLICY_VERSION, ppo_version=PPO_VERSION,
        observation_size=371, action_size=24, distribution='tanh_latent_gaussian',
        action_contract_sha256=canonical_sha256(ACTION_CONTRACT),
        state_contract_sha256=state_contract_sha256, ppo_config=asdict(config),
        source_sha256=copy.deepcopy(source))


def _clean(trainer, history_reset):
    if history_reset != 'clear':
        raise ValueError('explicit history_reset=clear required; no history reconstruction')
    if not trainer.clean_boundary:
        raise RuntimeError('checkpoint requires clean PPO update boundary; no pending rollout')


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('checkpoint contains nonfinite tensor')
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite_tree(item)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def _validate_adam(optimizer, model, config, updates):
    groups = optimizer.get('param_groups')
    parameters = list(model.parameters())
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError('one complete Adam parameter group required')
    ids = groups[0].get('params')
    if ids != list(range(len(parameters))) or groups[0].get('lr') != config.lr:
        raise ValueError('Adam parameter order/config mismatch')
    state = optimizer['state']
    if any(type(identifier) is not int or identifier not in ids for identifier in state):
        raise ValueError('unknown Adam parameter identity')
    if updates > 0 and set(state) != set(ids):
        raise ValueError('updated checkpoint must retain all Adam states')
    for identifier, entry in state.items():
        if not isinstance(entry, dict) or not {'step', 'exp_avg', 'exp_avg_sq'}.issubset(entry):
            raise ValueError('incomplete Adam moments')
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if name in entry:
                moment = entry[name]
                parameter = parameters[identifier]
                if not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                    raise ValueError('Adam moment shape/dtype mismatch')
        step = entry['step']
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not math.isfinite(step.item()) or step.item() < 0:
            raise ValueError('invalid Adam step counter')


def _validate_rng(rng):
    if not isinstance(rng, dict) or not {'torch', 'python', 'numpy', 'cuda'}.issubset(rng):
        raise ValueError('complete RNG checkpoint required')
    try:
        torch.Generator(device='cpu').set_state(rng['torch'])
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
    except (ValueError, RuntimeError, TypeError) as error:
        raise ValueError('invalid checkpoint RNG state') from error
    if not isinstance(rng['cuda'], list):
        raise ValueError('CUDA RNG list required')
    if rng['cuda']:
        if not torch.cuda.is_available() or len(rng['cuda']) != torch.cuda.device_count():
            raise ValueError('checkpoint CUDA RNG device contract unavailable')
        for state in rng['cuda']:
            if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
                raise ValueError('invalid CUDA RNG state')


def save_checkpoint(path, trainer, identity, *, history_reset):
    _clean(trainer, history_reset)
    expected = build_identity(trainer.config, state_contract_sha256=identity['state_contract_sha256'],
        source_sha256=identity['source_sha256'])
    if expected != identity:
        raise ValueError('checkpoint identity/config mismatch')
    payload = dict(version=VERSION, identity=copy.deepcopy(identity),
        policy=_cpu_tree(trainer.model.state_dict()), optimizer=_cpu_tree(trainer.optimizer.state_dict()),
        policy_version=trainer.policy_version, counters=copy.deepcopy(trainer.counters),
        rollout_steps=trainer.rollout_steps, num_envs=trainer.num_envs, history_reset=history_reset,
        rng=dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []))
    _finite_tree(payload)
    stream = io.BytesIO()
    torch.save(payload, stream)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as file:
        file.write(stream.getvalue())
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), version=VERSION)


def load_checkpoint(path, trainer, expected_identity, *, history_reset, restore_rng=True):
    _clean(trainer, history_reset)
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('version') != VERSION:
        raise ValueError('unsupported checkpoint version; legacy visual/surface transfer prohibited')
    if payload.get('identity') != expected_identity:
        raise ValueError('checkpoint contract identity mismatch')
    actual = build_identity(trainer.config, state_contract_sha256=expected_identity['state_contract_sha256'],
        source_sha256=expected_identity['source_sha256'])
    if actual != expected_identity:
        raise ValueError('checkpoint identity/config mismatch')
    if (payload.get('history_reset') != history_reset or payload.get('rollout_steps') != trainer.rollout_steps
            or payload.get('num_envs') != trainer.num_envs):
        raise ValueError('checkpoint rollout/history contract mismatch')
    _finite_tree(payload)
    current = trainer.model.state_dict()
    saved = payload.get('policy', {})
    if set(saved) != set(current) or any(saved[k].shape != current[k].shape or saved[k].dtype != current[k].dtype for k in current):
        raise ValueError('checkpoint policy topology mismatch')
    optimizer = payload.get('optimizer')
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get('state'), dict):
        raise ValueError('complete Adam checkpoint required')
    if type(payload.get('policy_version')) is not int or payload['policy_version'] < 0:
        raise ValueError('checkpoint policy counter required')
    counters = payload.get('counters')
    if (not isinstance(counters, dict) or set(counters) != {'environment_steps', 'updates'}
            or any(type(v) is not int or v < 0 for v in counters.values())):
        raise ValueError('complete nonnegative checkpoint counters required')
    _validate_adam(optimizer, trainer.model, trainer.config, counters['updates'])
    noise = saved.get('log_std')
    if noise is None or bool(((noise < math.log(trainer.config.min_std)) | (noise > math.log(trainer.config.max_std))).any()):
        raise ValueError('checkpoint noise outside declared policy bounds')
    # Validate state_dict loading on disposable copies before mutating live state.
    candidate_model = copy.deepcopy(trainer.model)
    candidate_model.load_state_dict(saved, strict=True)
    candidate_optimizer = torch.optim.Adam(candidate_model.parameters(), lr=trainer.config.lr)
    candidate_optimizer.load_state_dict(optimizer)
    rng = payload.get('rng', {})
    _validate_rng(rng)
    trainer.model.load_state_dict(saved, strict=True)
    trainer.optimizer.load_state_dict(optimizer)
    trainer.policy_version = payload['policy_version']
    trainer.counters = copy.deepcopy(counters)
    trainer._new_rollout()
    trainer.project_noise()
    if restore_rng:
        torch.set_rng_state(rng['torch'])
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        if rng['cuda']:
            if not torch.cuda.is_available():
                raise ValueError('checkpoint CUDA RNG cannot be restored without CUDA')
            torch.cuda.set_rng_state_all(rng['cuda'])
    return dict(version=VERSION, identity=copy.deepcopy(expected_identity), history_reset=history_reset,
        policy_version=trainer.policy_version, counters=copy.deepcopy(counters))
