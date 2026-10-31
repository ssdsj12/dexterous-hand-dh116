"""Explicit immutable 351D surface PPO repair, without PPO or Isaac imports."""
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import torch
from torch import nn

from .tabletop_policy_noise import configure_surface_noise
from .tabletop_surface_exploration import STABLE_HOLD_REWARD_CONTRACT
from .tabletop_surface_imitation import OBSERVATION_VERSION, make_actor, validate_dataset


VERSION = 'native_surface_repair_resume_v1'
CHECKPOINT_VERSION = 'dh116-rsl-tensor-ppo-v1'
MOMENTS = ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')
METADATA_KEYS = frozenset(('dataset_sha256', 'initial_weight', 'decay', 'batch_size', 'update',
    'source', 'holding_weight', 'wrist_weight', 'auxiliary_learning_rate', 'weight_semantics'))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value, path='checkpoint'):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite or unsupported tensor at ' + path)
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite(item, path + '.' + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, path + '[' + str(index) + ']')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite scalar at ' + path)


def _same(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict):
        return (isinstance(right, dict) and list(left) == list(right)
                and all(_same(value, right[key]) for key, value in left.items()))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_same(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and pickle.dumps(left) == pickle.dumps(right)


class _SurfacePolicy(nn.Module):
    """Only the exact make_algorithm surface parameter topology; no algorithm."""

    def __init__(self):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(24))
        self.actor = make_actor(351)
        self.critic = nn.Sequential(nn.Linear(351, 128), nn.Tanh(), nn.Linear(128, 128),
                                    nn.Tanh(), nn.Linear(128, 1))


def _validated_model(saved):
    required = {'version', 'contract', 'policy', 'optimizer', 'iteration', 'timesteps', 'learning_rate',
                'episodes', 'successes', 'recovered', 'reached_counts', 'transition_counts',
                'torch_rng', 'cuda_rng', 'surface_imitation'}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError('complete surface PPO checkpoint and counters required')
    if saved['version'] != CHECKPOINT_VERSION:
        raise ValueError('unsupported surface PPO checkpoint version')
    contract = saved['contract']
    if (not isinstance(contract, dict) or contract.get('observation_size') != 351
            or contract.get('observation_version') != OBSERVATION_VERSION
            or not any(key in contract for key in ('action_size', 'policy_command_size'))
            or any(contract[key] != 24 for key in ('action_size', 'policy_command_size') if key in contract)):
        raise ValueError('repair requires the original 351-observation / 24-action surface contract')
    if STABLE_HOLD_REWARD_CONTRACT.get('version') != 'surface_stable_hold_reward_v2':
        raise ValueError('current stable hold reward v2 required')
    _finite(saved)
    # Model initialization must not consume the caller's CPU RNG. No CUDA RNG,
    # distribution, rollout storage, simulator or PPO instance is constructed.
    with torch.random.fork_rng(devices=[]):
        model = _SurfacePolicy().float()
    parameters = list(model.named_parameters())
    names = [name for name, _ in parameters]
    state = saved['policy']
    if not isinstance(state, dict) or list(state) != names or len(names) != 13:
        raise ValueError('policy parameter order must be log_std, six actor, six critic parameters')
    for name, parameter in parameters:
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                or value.dtype != parameter.dtype):
            raise ValueError('policy parameter shape/dtype mismatch: ' + name)
    optimizer = saved['optimizer']
    if (not isinstance(optimizer, dict) or not isinstance(optimizer.get('state'), dict)
            or not isinstance(optimizer.get('param_groups'), list) or len(optimizer['param_groups']) != 1):
        raise ValueError('one validated PPO Adam parameter group required')
    group = optimizer['param_groups'][0]
    identifiers = group.get('params') if isinstance(group, dict) else None
    if (not isinstance(identifiers, list) or any(type(value) is not int for value in identifiers)
            or identifiers != list(range(len(parameters)))
            or ('param_names' in group and group['param_names'] != names)):
        raise ValueError('optimizer parameter mapping/order does not match the surface model')
    if any(type(key) is not int or key not in identifiers for key in optimizer['state']):
        raise ValueError('unknown optimizer parameter mapping identifier')
    for identifier, (name, parameter) in zip(identifiers, parameters):
        entry = optimizer['state'].get(identifier, {})
        if not isinstance(entry, dict):
            raise ValueError('invalid optimizer parameter state: ' + name)
        required_moments = ('step', 'exp_avg', 'exp_avg_sq') + (('max_exp_avg_sq',) if group.get('amsgrad') else ())
        if entry and not all(field in entry for field in required_moments):
            raise ValueError('incomplete Adam/AMSGrad moments: ' + name)
        for field in MOMENTS:
            if field in entry and (not isinstance(entry[field], torch.Tensor)
                    or entry[field].shape != parameter.shape or entry[field].dtype != parameter.dtype):
                raise ValueError('optimizer moment shape/dtype mismatch: ' + name + '.' + field)
        if 'step' in entry and (entry['step'].numel() != 1 if isinstance(entry['step'], torch.Tensor)
                                else type(entry['step']) not in (int, float)):
            raise ValueError('invalid optimizer Adam step: ' + name)
    model.load_state_dict(state, strict=True)
    return model, names, dict(zip(names, identifiers))


def _metadata(previous, demonstration_sha):
    if (not isinstance(previous, dict) or not set(previous).issubset(METADATA_KEYS)
            or not isinstance(previous.get('source'), str) or not previous['source']):
        raise ValueError('known saved surface imitation metadata/source required')
    if previous.get('dataset_sha256') != demonstration_sha:
        raise ValueError('demonstrations exact file SHA256 must match saved surface_imitation.dataset_sha256')
    return dict(dataset_sha256=demonstration_sha, initial_weight=.1, decay=1., batch_size=256,
        update='independent_actor_adam_stable_hold_v2', source=previous['source'],
        holding_weight=20., wrist_weight=4., auxiliary_learning_rate=.001,
        weight_semantics='multiply_independent_Adam_learning_rate_not_its_gradient')


def _verify(before, after, noise_id, changed):
    mutable = {'contract', 'policy', 'optimizer', 'surface_imitation',
               'surface_imitation_optimizer', 'surface_repair_migration'}
    checks = dict(
        all_top_level_state_retained=set(after) == set(before) | {'surface_imitation_optimizer'},
        counters_rng_and_other_ppo_state_unchanged=all(_same(value, after[key]) for key, value in before.items()
                                                     if key not in mutable),
        actor_critic_weights_unchanged=all(_same(value, after['policy'][name]) for name, value in before['policy'].items()
                                         if name != 'log_std'),
        policy_noise_is_exact_target=torch.equal(after['policy']['log_std'],
                                                torch.full_like(before['policy']['log_std'], math.log(.005))),
        contract_only_approved_fields=_same(
            {k: v for k, v in before['contract'].items() if k not in ('policy_noise', 'surface_training_objective')},
            {k: v for k, v in after['contract'].items() if k not in ('policy_noise', 'surface_training_objective')}),
        ppo_param_groups_unchanged=_same(before['optimizer']['param_groups'], after['optimizer']['param_groups']),
        ppo_nonnoise_state_unchanged=True, noise_moments_only_changed_coordinates=True,
        auxiliary_optimizer_fresh_actor_order=(after['surface_imitation_optimizer']['state'] == {}
            and len(after['surface_imitation_optimizer']['param_groups']) == 1
            and after['surface_imitation_optimizer']['param_groups'][0]['params'] == list(range(6))
            and after['surface_imitation_optimizer']['param_groups'][0]['lr'] == .001))
    old, new = before['optimizer'], after['optimizer']
    checks['ppo_nonnoise_state_unchanged'] &= list(old) == list(new) and list(old['state']) == list(new['state'])
    for key in old:
        if key != 'state':
            checks['ppo_nonnoise_state_unchanged'] &= _same(old[key], new[key])
    for identifier, state in old['state'].items():
        if identifier != noise_id:
            checks['ppo_nonnoise_state_unchanged'] &= _same(state, new['state'][identifier])
            continue
        checks['noise_moments_only_changed_coordinates'] &= list(state) == list(new['state'][identifier])
        for field, value in state.items():
            expected = value.clone().masked_fill_(changed, 0.) if field in MOMENTS else value
            checks['noise_moments_only_changed_coordinates'] &= _same(expected, new['state'][identifier][field])
    _finite(after)
    checks['all_tensors_finite'] = True
    if not all(checks.values()):
        raise RuntimeError('surface repair immutability verification failed: ' +
                           ', '.join(key for key, value in checks.items() if not value))
    return checks


def prepare_repair_resume(input_path, output_path, demonstrations):
    """Create output and output.with_suffix('.json') exclusively; return manifest.

    The source and dataset remain immutable. SHA256 binds their exact file
    bytes. Only the external manifest contains the derived checkpoint's hash.
    """
    source, output, demo_path = (Path(path).absolute() for path in (input_path, output_path, demonstrations))
    sidecar = output.with_suffix('.json')
    destinations = (output, sidecar)
    if (output.resolve() == sidecar.resolve()
            or any(path.resolve() in (source.resolve(), demo_path.resolve()) for path in destinations)):
        raise ValueError('derived checkpoint and JSON sidecar must be distinct from inputs and each other')
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    source_sha, demo_sha = _sha256(source), _sha256(demo_path)
    saved = torch.load(source, map_location='cpu', weights_only=False)
    model, names, mapping = _validated_model(saved)
    metadata = _metadata(saved['surface_imitation'], demo_sha)
    data = torch.load(demo_path, map_location='cpu', weights_only=False)
    _finite(data, 'demonstrations')
    derived = copy.deepcopy(saved)
    noise_id = mapping['log_std']
    adapter = SimpleNamespace(policy=model, optimizer=SimpleNamespace(
        state={model.log_std: derived['optimizer']['state'].get(noise_id, {})}))
    derived['contract']['policy_noise'] = dict(configure_surface_noise(adapter))
    derived['contract']['surface_training_objective'] = copy.deepcopy(STABLE_HOLD_REWARD_CONTRACT)
    validate_dataset(data, derived['contract'])
    derived['policy']['log_std'] = model.log_std.detach().clone()
    changed = saved['policy']['log_std'] != derived['policy']['log_std']
    derived['surface_imitation'] = metadata
    derived['surface_imitation_optimizer'] = torch.optim.Adam(model.actor.parameters(), lr=.001).state_dict()
    checks = _verify(saved, derived, noise_id, changed)
    migration = dict(version=VERSION, source_checkpoint=str(source), source_sha256=source_sha,
        demonstrations=str(demo_path), demonstrations_sha256=demo_sha,
        parameter_order=names, optimizer_parameter_mapping=mapping,
        changes=dict(
            policy_log_std=dict(before=saved['policy']['log_std'].tolist(), after=derived['policy']['log_std'].tolist()),
            ppo_noise_moments=dict(parameter_id=noise_id, changed_coordinates=changed.tolist(),
                zeroed_fields=[field for field in MOMENTS if field in saved['optimizer']['state'].get(noise_id, {})],
                adam_step_retained=True, all_actor_critic_moments_retained=True),
            contract={key: dict(present_before=key in saved['contract'], before=saved['contract'].get(key),
                               after=derived['contract'][key]) for key in ('policy_noise', 'surface_training_objective')},
            surface_imitation=dict(before=saved['surface_imitation'], after=metadata),
            surface_imitation_optimizer=dict(previously_present='surface_imitation_optimizer' in saved,
                previous_state_entries=len(saved.get('surface_imitation_optimizer', {}).get('state', {})),
                new_state_entries=0, learning_rate=.001,
                parameter_order=[name.removeprefix('actor.') for name in names if name.startswith('actor.')])),
        invariant_checks=checks)
    if 'surface_repair_migration' in saved:
        migration['previous_migration'] = copy.deepcopy(saved['surface_repair_migration'])
    derived['surface_repair_migration'] = migration
    manifest = dict(version=VERSION, status='surface_repair_resume_prepared',
        source_checkpoint=str(source), source_sha256=source_sha,
        derived_checkpoint=str(output), demonstrations_sha256=demo_sha,
        native_student_proof_present=bool(saved.get('surface_student_validation')),
        native_student_validation_created=False, native_success_claimed=False, migration=migration)
    json.dumps(manifest, allow_nan=False)  # Reject non-JSON provenance before creating either output.
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open('xb') as checkpoint_stream:
            created.append(output)
            with sidecar.open('x', encoding='utf-8') as manifest_stream:
                created.append(sidecar)
                torch.save(derived, checkpoint_stream)
                checkpoint_stream.flush()
                if _sha256(source) != source_sha or _sha256(demo_path) != demo_sha:
                    raise RuntimeError('source or demonstrations changed during repair preparation')
                manifest['derived_sha256'] = _sha256(output)
                json.dump(manifest, manifest_stream, indent=2, allow_nan=False)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return manifest
