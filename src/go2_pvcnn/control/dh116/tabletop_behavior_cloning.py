"""Success-only offline imitation and strict, fresh PPO actor initialization.

Qualification is supplied by the real teacher runner. This module validates its
explicit evidence fields; offline regression metrics never establish lift success.
"""
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

DATASET_VERSION = 'dh116-tabletop-demonstrations-v1'
ACTOR_VERSION = 'dh116-tabletop-bc-actor-v1'
OBSERVATION_DIM = 156
ACTION_DIM = 20
INITIAL_NOISE_STD = .05
_BATCH_ONLY_FIELDS = {'num_envs', 'rollout_steps', 'minibatches'}


def make_actor():
    """Match make_algorithm's actor exactly, including state-dict layer keys."""
    return nn.Sequential(nn.Linear(156, 128), nn.Tanh(), nn.Linear(128, 128),
                         nn.Tanh(), nn.Linear(128, 20), nn.Tanh())


def _json_copy(value, name):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError(f'{name} must contain finite JSON-compatible metadata') from error


def _qualification(value):
    if not isinstance(value, Mapping):
        raise ValueError('missing physical/replay qualification')
    for field in ('qualified', 'physical_validation', 'fresh_resets'):
        if value.get(field) is not True:
            raise ValueError(f'qualification requires {field}=True')
    replays = value.get('successful_replays')
    if isinstance(replays, bool) or not isinstance(replays, int) or replays < 3:
        raise ValueError('qualification requires at least three successful replays')
    if not any(isinstance(value.get(key), str) and value[key].strip()
               for key in ('proof_path', 'proof_id')):
        raise ValueError('qualification requires a nonempty proof_path or proof_id')
    return _json_copy(dict(value), 'qualification')


def _contract(value):
    required = {'observation_version', 'observation_size', 'action', 'physics_hz', 'reward'}
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError('full task contract must include observations, actions, physics and reward')
    if value['observation_size'] != OBSERVATION_DIM:
        raise ValueError('contract observation size must be 156')
    return _json_copy(dict(value), 'contract')


def _dimensions(data):
    if data.get('observation_dim') != OBSERVATION_DIM or data.get('action_dim') != ACTION_DIM:
        raise ValueError('imitation requires 156 observations and 20 actions')


def _finite_tensor(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f'{name} shape must be {shape}')
    if value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite and real')
    result = value.detach().to(device='cpu', dtype=torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f'{name} must remain finite as float32')
    return result


def _load(path):
    try:
        data = torch.load(Path(path), map_location='cpu', weights_only=True)
    except (RuntimeError, TypeError, EOFError) as error:
        raise ValueError('invalid imitation artifact') from error
    if not isinstance(data, dict):
        raise ValueError('imitation artifact must be a mapping')
    return data


def _physical_evidence(value, steps):
    """Require the measured channels emitted by the production teacher trace.

    Missing contacts remain explicit false masks. Their presence is not itself
    success; the terminal predicate and fresh-reset proof qualify the episode.
    """
    shapes = dict(contact_center_relative_m=(steps, 2, 3),
        contact_normal=(steps, 2, 3), contact_valid=(steps, 2),
        normal_vertical_force_n=(steps,), lift_vertical_force_n=(steps,),
        object_bottom_m=(steps,))
    if not isinstance(value, Mapping) or not set(shapes).issubset(value):
        raise ValueError('qualified episodes require complete measured physical_evidence')
    for key, shape in shapes.items():
        measured = _finite_tensor(value[key], shape, f'physical_evidence.{key}')
        if key == 'contact_valid' and not bool(((measured == 0) | (measured == 1)).all()):
            raise ValueError('physical_evidence.contact_valid must contain boolean values')
    for key in value.keys() - shapes.keys():
        field = value[key]
        if not isinstance(field, torch.Tensor) or field.ndim < 1 or field.shape[0] != steps:
            raise ValueError(f'physical_evidence.{key} must have one row per transition')
        _finite_tensor(field, tuple(field.shape), f'physical_evidence.{key}')


def _validated_dataset(data):
    if data.get('version') != DATASET_VERSION:
        raise ValueError('unsupported demonstration dataset version')
    _dimensions(data)
    contract = _contract(data.get('contract'))
    qualification = _qualification(data.get('qualification'))
    episodes = data.get('episodes')
    if not isinstance(episodes, list) or len(episodes) < 3:
        raise ValueError('at least three complete successful episodes are required')
    seen = set()
    validated = []
    for episode in episodes:
        if not isinstance(episode, dict) or episode.get('success') is not True:
            raise ValueError('dataset may contain only explicit successful episodes')
        identity = (episode.get('env_index'), episode.get('episode_index'))
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in identity):
            raise ValueError('invalid episode identity')
        if identity in seen:
            raise ValueError('duplicate episode identity')
        seen.add(identity)
        observations = episode.get('observations')
        if not isinstance(observations, torch.Tensor) or observations.ndim != 2:
            raise ValueError('observations must have shape (T, 156)')
        steps = observations.shape[0]
        if not steps:
            raise ValueError('empty successful episode')
        obs = _finite_tensor(observations, (steps, 156), 'observations')
        actions = _finite_tensor(episode.get('actions'), (steps, 20), 'actions')
        if bool((actions.abs() > 1.).any()):
            raise ValueError('demonstrated actions must be normalized to [-1, 1]')
        _finite_tensor(episode.get('next_observations'), (steps, 156), 'next_observations')
        _finite_tensor(episode.get('rewards'), (steps,), 'rewards')
        for name in ('dones', 'success_mask'):
            flags = _finite_tensor(episode.get(name), (steps,), name)
            if not bool(((flags == 0) | (flags == 1)).all()):
                raise ValueError(f'{name} must contain boolean values')
            if not bool(flags[-1]) or bool(flags[:-1].any()):
                raise ValueError(f'{name} must mark only the terminal successful boundary')
        failures = _finite_tensor(episode.get('failure'), (steps,), 'failure')
        if bool(failures.any()):
            raise ValueError('failed episodes cannot be imitation labels')
        _physical_evidence(episode.get('physical_evidence'), steps)
        if 'phase' in episode:
            phases = _finite_tensor(episode['phase'], (steps,), 'phase')
            if bool(((phases < 0) | (phases != phases.floor())).any()):
                raise ValueError('phase must contain nonnegative integer codes')
        validated.append((identity, obs, actions))
    return contract, qualification, validated


@torch.no_grad()
def _metrics(actor, obs, actions):
    errors = actor(obs) - actions
    if not bool(torch.isfinite(errors).all()):
        raise ValueError('nonfinite imitation predictions')
    mae = errors.abs().mean(0)
    return dict(mse=float(errors.square().mean()), mae=float(mae.mean()),
                maximum_action_error=float(errors.abs().max()),
                per_action_mae=mae.cpu().tolist(),
                channel_mae=dict(joints=float(mae[:12].mean()),
                                 palms=float(mae[12:18].mean()), force=float(mae[18:].mean())))


def _evaluation_provenance(artifact):
    metrics = _json_copy(artifact.get('validation_metrics'), 'validation metrics')
    if not isinstance(metrics, dict):
        raise ValueError('actor artifact requires held-out validation metrics')
    def nonnegative(value):
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and value >= 0)
    if any(not nonnegative(metrics.get(key)) for key in ('mse', 'mae', 'maximum_action_error')):
        raise ValueError('validation metrics must contain nonnegative finite action errors')
    per_action = metrics.get('per_action_mae')
    if (not isinstance(per_action, list) or len(per_action) != 20
            or not all(nonnegative(value) for value in per_action)):
        raise ValueError('validation metrics must report all 20 action channels')
    split = artifact.get('split')
    if not isinstance(split, dict) or split.get('method') != 'whole_episode':
        raise ValueError('actor artifact requires a whole_episode evaluation split')
    groups = []
    for field in ('train_episode_ids', 'validation_episode_ids'):
        identities = split.get(field)
        if not isinstance(identities, list) or not identities:
            raise ValueError('training and evaluation require distinct complete episodes')
        seen = set()
        for identity in identities:
            if (not isinstance(identity, list) or len(identity) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in identity)
                    or tuple(identity) in seen):
                raise ValueError('invalid or repeated evaluation episode identity')
            seen.add(tuple(identity))
        groups.append(seen)
    if groups[0] & groups[1] or len(groups[0] | groups[1]) < 3:
        raise ValueError('episode evaluation split leaks training labels or has too few episodes')
    return metrics


def train_imitation(dataset_path, output, device='cpu', epochs=200, seed=42,
                    batch_size=256, learning_rate=1e-3, validation_fraction=.2):
    """Fit a bounded actor; hold out entire episodes and emit no PPO state."""
    for name, value in (('epochs', epochs), ('batch_size', batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    dataset_path, output = Path(dataset_path), Path(output)
    if dataset_path.resolve() == output.resolve():
        raise ValueError('output must not overwrite the source dataset')
    contract, qualification, episodes = _validated_dataset(_load(dataset_path))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(episodes), generator=generator).tolist()
    count = min(len(episodes)-1, max(1, round(len(episodes)*validation_fraction)))
    validation_ids, train_ids = indices[:count], indices[count:]
    device = torch.device(device)
    train_obs = torch.cat([episodes[i][1] for i in train_ids]).to(device)
    train_actions = torch.cat([episodes[i][2] for i in train_ids]).to(device)
    validation_obs = torch.cat([episodes[i][1] for i in validation_ids]).to(device)
    validation_actions = torch.cat([episodes[i][2] for i in validation_ids]).to(device)
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        actor = make_actor().to(device)
        optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
        initial_metrics = _metrics(actor, validation_obs, validation_actions)
        for _ in range(epochs):
            for batch in torch.randperm(len(train_obs), generator=generator).split(batch_size):
                batch = batch.to(device)
                loss = (actor(train_obs[batch])-train_actions[batch]).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError('nonfinite imitation loss')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        metrics = _metrics(actor, validation_obs, validation_actions)
        train_metrics = _metrics(actor, train_obs, train_actions)
    actor_state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in actor_state.values()):
        raise ValueError('nonfinite trained actor weights')
    artifact = dict(version=ACTOR_VERSION, actor=actor_state,
        observation_dim=156, action_dim=20, contract=contract, qualification=qualification,
        dataset_path=str(dataset_path.resolve()), dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        initial_noise_std=INITIAL_NOISE_STD, initial_validation_metrics=initial_metrics,
        validation_metrics=metrics, train_metrics=train_metrics,
        split=dict(method='whole_episode', train_episode_ids=[list(episodes[i][0]) for i in train_ids],
                   validation_episode_ids=[list(episodes[i][0]) for i in validation_ids]),
        training=dict(epochs=epochs, seed=seed, batch_size=batch_size, learning_rate=learning_rate),
        closed_loop_student_validation='required_before_optimization')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    return {key: value for key, value in artifact.items() if key != 'actor'}


def load_actor_initialization(algorithm, path, current_contract):
    """Validate completely, then load actor and permitted noise into fresh PPO.

    This does not certify the student's closed-loop behavior. The trainer must
    evaluate that behavior before allowing PPO optimization.
    """
    artifact = _load(path)
    if artifact.get('version') != ACTOR_VERSION:
        raise ValueError('expected a BC actor artifact, not a PPO resume checkpoint')
    _dimensions(artifact)
    saved_contract = _contract(artifact.get('contract'))
    current_contract = _contract(current_contract)
    meaningful = lambda value: {k: v for k, v in value.items() if k not in _BATCH_ONLY_FIELDS}
    if meaningful(saved_contract) != meaningful(current_contract):
        raise ValueError('actor initialization contract mismatch; physics/reward/control migration is forbidden')
    qualification = _qualification(artifact.get('qualification'))
    digest = artifact.get('dataset_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('actor initialization requires dataset SHA256 provenance')
    if artifact.get('initial_noise_std') != INITIAL_NOISE_STD:
        raise ValueError('actor initialization noise must equal the existing .05 floor')
    if algorithm.optimizer.state:
        raise ValueError('actor initialization requires a fresh optimizer')
    if getattr(getattr(algorithm, 'storage', None), 'step', 0) != 0:
        raise ValueError('actor initialization requires fresh rollout storage')
    for owner in (algorithm, algorithm.policy):
        for name in ('current_learning_iteration', 'iteration', 'timesteps', 'total_timesteps', 'tot_timesteps'):
            if hasattr(owner, name) and getattr(owner, name) != 0:
                raise ValueError('actor initialization requires fresh zero counters')
    weights = artifact.get('actor')
    target = algorithm.policy.actor.state_dict()
    if not isinstance(weights, dict) or set(weights) != set(target):
        raise ValueError('actor initialization state keys mismatch')
    for key, reference in target.items():
        value = weights[key]
        if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError(f'actor initialization nonfinite or mismatched weight {key}')
        converted = value.to(reference)
        if not bool(torch.isfinite(converted).all()):
            raise ValueError(f'actor initialization weight {key} overflows target dtype')
    # Check both expected architecture and target loading without mutating PPO.
    try:
        make_actor().load_state_dict(weights, strict=True)
        candidate = copy.deepcopy(algorithm.policy.actor)
        candidate.load_state_dict(weights, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError('actor initialization architecture mismatch') from error
    log_std = algorithm.policy.log_std
    if log_std.shape != (20,) or not bool(torch.isfinite(log_std).all()):
        raise ValueError('fresh policy must expose finite 20D log_std')
    validation_metrics = _evaluation_provenance(artifact)
    algorithm.policy.actor.load_state_dict(weights, strict=True)
    with torch.no_grad():
        log_std.fill_(math.log(INITIAL_NOISE_STD))
    return dict(version=ACTOR_VERSION, path=str(Path(path).resolve()), actor_only=True,
        dataset_sha256=digest, qualification=qualification, validation_metrics=validation_metrics,
        optimizer_state_entries=0, initial_noise_std=INITIAL_NOISE_STD,
        source_contract=saved_contract,
        closed_loop_student_validation='required_before_optimization')
