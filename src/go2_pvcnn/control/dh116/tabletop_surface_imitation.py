"""Surface-only behavior cloning and post-PPO demonstration regression.

No simulator or rollout collector is imported here. Qualification records are
supplied by the native replay collector; offline MSE is not grasp evidence.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from .tabletop_policy_noise import CONTRACT as NOISE_CONTRACT, project_policy_noise


DATASET_VERSION = 'dh116-surface-demonstrations-v1'
ACTOR_VERSION = 'dh116-surface-actor-v1'
OBSERVATION_VERSION = 'dh116-surface-sharing-lift-objective-v2'
HISTORY_OBSERVATION_VERSION = 'dh116-surface-sharing-history-v3'


def validate_student_admission(actor_path, reports, current_contract):
    from .tabletop_surface_admission import validate_student_admission as validate
    return validate(actor_path, reports, current_contract)
# Deliberately top-level and narrow: physics, initialization, observations and
# action semantics stay exact. Only the two known uniform height arrays below
# may discard their repetition count during comparison.
_BATCH_FIELDS = frozenset(('num_envs', 'rollout_steps', 'minibatches'))
_RESET_FIELDS = ('close_seed_sha256', 'grasp_initialization_sha256')


def _canonical_height_array(values):
    if (isinstance(values, list) and values
            and all(type(value) in (int, float) and math.isfinite(value)
                    and value == values[0] for value in values)):
        return {'uniform_env_value': values[0]}
    # Tag unchanged values too, so literal metadata cannot impersonate the
    # uniform marker. Nonuniform lists retain every value and its order.
    return {'exact_env_values': values}


def _canonical_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError('surface contract must be a dictionary')
    legacy=(contract.get('observation_size')==351
            and contract.get('observation_version')==OBSERVATION_VERSION)
    history=False
    if (contract.get('observation_size')==401
            and contract.get('observation_version')==HISTORY_OBSERVATION_VERSION):
        from .tabletop_surface_contact_history import CONTRACT as HISTORY_CONTRACT
        history=contract.get('surface_contact_history')==HISTORY_CONTRACT
    if not (legacy or history):
        raise ValueError('surface contract requires explicit 351D or causal-history 401D observations')
    dimensions = [contract[key] for key in ('action_size', 'policy_command_size')
                  if key in contract]
    if not dimensions or any(value != 24 for value in dimensions):
        raise ValueError('surface contract requires 24D actions')
    canonical = {k: v for k, v in contract.items() if k not in _BATCH_FIELDS}
    # BC never consumes replay rewards. This exact new training objective uses
    # unchanged observations/actions/physics and unchanged physical SUCCESS.
    from .tabletop_surface_exploration import STABLE_HOLD_REWARD_CONTRACT
    if canonical.get('surface_training_objective') == STABLE_HOLD_REWARD_CONTRACT:
        canonical.pop('surface_training_objective')
    # Demonstrations supervise submitted actuator commands, independently of
    # the student's Gaussian exploration variance. Admit only this explicit
    # alternate sampling contract; every physics/reward/input field stays exact.
    noise = canonical.get('policy_noise')
    if (isinstance(noise, dict) and noise.get('version') == 'bounded_surface_policy_noise_v2'
            and set(noise) == set(NOISE_CONTRACT) | {'initial_std'}
            and all(noise[k] == value for k, value in NOISE_CONTRACT.items()
                    if k not in ('version', 'min_std', 'max_std'))
            and all(type(noise[k]) in (int, float) and math.isfinite(noise[k])
                    for k in ('initial_std', 'min_std', 'max_std'))
            and 0 < noise['min_std'] <= noise['initial_std'] <= noise['max_std'] <= 1.):
        canonical['policy_noise'] = dict(NOISE_CONTRACT)
    if 'success_bottom_m' in canonical:
        canonical['success_bottom_m'] = _canonical_height_array(canonical['success_bottom_m'])
    sharing = canonical.get('surface_force_sharing')
    if isinstance(sharing, dict) and 'lift_reference_bottom_m' in sharing:
        canonical['surface_force_sharing'] = dict(sharing,
            lift_reference_bottom_m=_canonical_height_array(sharing['lift_reference_bottom_m']))
    try:
        return json.dumps(canonical, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('surface contract must contain finite JSON metadata') from error


def _check_contract(saved, current):
    if _canonical_contract(saved) != _canonical_contract(current):
        raise ValueError('surface contract mismatch; only batch fields and repetition '
                         'counts of the two known uniform height arrays may differ')


def _sha256_reference(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdefABCDEF' for char in value))


def _qualification(value):
    if not isinstance(value, dict) or value.get('qualified') is not True:
        raise ValueError('qualified native replay evidence is required')
    successes = value.get('successful_replays')
    proof = value.get('proof_path')
    if (type(successes) is not int or successes < 1
            or not isinstance(proof, str) or not proof.strip()):
        raise ValueError('qualification requires successful_replays >= 1 and nonempty proof_path')


def _family_physics(contract):
    """Compare full canonical contracts except two explicit reset SHA fields."""
    canonical = json.loads(_canonical_contract(contract))
    for field in _RESET_FIELDS:
        if not _sha256_reference(canonical.get(field)):
            raise ValueError(f'reset family contract requires valid {field}')
        del canonical[field]
    for container, key in ((contract, 'success_bottom_m'),
                           (contract.get('surface_force_sharing', {}), 'lift_reference_bottom_m')):
        if isinstance(container, dict) and key in container:
            if 'uniform_env_value' not in _canonical_height_array(container[key]):
                raise ValueError('reset family requires uniform finite numeric height arrays')
    return json.dumps(canonical, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_reset_family(family, saved_contract, current_contract):
    """Validate a declared family and require BOTH full contracts to be members.

    Labels do not establish a new initialization. At least two different reset
    SHA pairs must be present. Provenance paths are portable references; native
    success is supplied by replay qualification, never inferred from BC loss.
    """
    if not isinstance(family, list) or len(family) < 2:
        raise ValueError('reset family requires at least two declared members')
    physics = None
    members = set()
    initializations = set()
    for member in family:
        if not isinstance(member, dict):
            raise ValueError('reset family members must be dictionaries')
        if (not isinstance(member.get('label'), str) or not member['label'].strip()
                or not isinstance(member.get('proof_path'), str) or not member['proof_path'].strip()
                or not _sha256_reference(member.get('source_dataset_sha256'))):
            raise ValueError('reset family requires member label, proof_path and source dataset SHA256')
        if 'qualification' in member:
            _qualification(member['qualification'])
            if member['qualification']['proof_path'] != member['proof_path']:
                raise ValueError('reset family proof_path disagrees with member qualification')
        if 'source_dataset_path' in member and (
                not isinstance(member['source_dataset_path'], str) or not member['source_dataset_path'].strip()):
            raise ValueError('reset family source_dataset_path must be nonempty')
        candidate = member.get('contract')
        physical = _family_physics(candidate)
        if physics is None:
            physics = physical
        elif physical != physics:
            raise ValueError('reset family contract mismatch outside the two reset SHA fields')
        members.add(_canonical_contract(candidate))
        initializations.add(tuple(candidate[field].lower() for field in _RESET_FIELDS))
    if len(initializations) < 2:
        raise ValueError('reset family requires at least two different initializations')
    if any(_canonical_contract(value) not in members for value in (saved_contract, current_contract)):
        raise ValueError('saved and current contracts must both be declared reset family members')
    return family


def _validate_sample_weights(weights, samples):
    if (not isinstance(weights, torch.Tensor) or weights.shape != (samples,)
            or weights.layout != torch.strided or not weights.is_floating_point()
            or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())
            or not bool(weights.sum(dtype=torch.float64).isfinite())
            or not bool(weights.sum(dtype=torch.float64) > 0)):
        raise ValueError('sample_weights must be finite nonnegative floating [N] with positive total')


def _sampling_weights(data, device):
    if 'sample_weights' not in data:
        return None
    return data['sample_weights'].detach().to(device=device).clone()


def _sample_indices(samples, batch_size, device, weights=None, generator=None):
    if weights is not None:
        return torch.multinomial(weights, batch_size, replacement=True, generator=generator)
    return torch.randint(samples, (batch_size,), device=device, generator=generator)


def validate_dataset(data, contract=None):
    """Validate in-memory replay data without modifying it.

    Accept collector diagnostics and additional qualification evidence. The
    proof_path is a required provenance reference, not an offline success test.
    Compare every non-batch contract field when a target contract is supplied.
    """
    if not isinstance(data, dict) or data.get('version') != DATASET_VERSION:
        raise ValueError(f'expected {DATASET_VERSION} dataset dictionary')
    _canonical_contract(data.get('contract'))
    if 'reset_family' in data:
        validate_reset_family(data['reset_family'], data['contract'],
                              data['contract'] if contract is None else contract)
    elif contract is not None:
        _check_contract(data['contract'], contract)
    observations = data.get('observations')
    actions = data.get('actions')
    if (not isinstance(observations, torch.Tensor) or observations.ndim != 2
            or observations.shape[1] != data['contract']['observation_size'] or len(observations) == 0):
        raise ValueError('observations must match the nonempty versioned surface input shape')
    n = len(observations)
    if not isinstance(actions, torch.Tensor) or actions.shape != (n, 24):
        raise ValueError('actions must have shape [N,24] matching observations')
    for name, values in (('observations', observations), ('actions', actions)):
        if (not values.is_floating_point() or values.layout != torch.strided
                or not bool(torch.isfinite(values).all())):
            raise ValueError(f'{name} must be finite floating point tensors')
    if bool((actions.abs() > 1).any()):
        raise ValueError('actions must lie within [-1,1]')
    ids = data.get('episode_ids')
    if (not isinstance(ids, torch.Tensor) or ids.shape != (n,)
            or ids.dtype != torch.long or bool((ids < 0).any())):
        raise ValueError('episode_ids must be nonnegative long tensor [N]')
    _qualification(data.get('qualification'))
    if 'sample_weights' in data:
        _validate_sample_weights(data['sample_weights'], n)
    return data


def make_actor(observation_size=351):
    """Return the bounded actor for an explicitly versioned surface input."""
    if observation_size not in (351,401):raise ValueError('unsupported surface actor input size')
    return nn.Sequential(nn.Linear(observation_size, 128), nn.Tanh(), nn.Linear(128, 128),
                         nn.Tanh(), nn.Linear(128, 24), nn.Tanh())


def _dataset_sha256(data):
    """Hash supervised content, its full contract and qualification evidence.

    Collector-only diagnostics (next observations/rewards/traces) are excluded;
    dtype, shape and every byte of all three training columns are included.
    """
    digest = hashlib.sha256()
    try:
        metadata = {key: data[key] for key in ('version', 'contract', 'qualification')}
        if 'reset_family' in data:
            metadata['reset_family'] = data['reset_family']
        header = json.dumps(metadata,
                            sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('dataset provenance must contain finite JSON metadata') from error
    digest.update(header.encode('utf-8'))
    columns = ['observations', 'actions', 'episode_ids']
    if 'sample_weights' in data:
        columns.append('sample_weights')
    for key in columns:
        tensor = data[key].detach().cpu().contiguous()
        digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode('utf-8'))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _training_tensors(data, device):
    columns = [data[key].detach().to(device=device, dtype=torch.float32).clone()
               for key in ('observations', 'actions')]
    if any(not bool(torch.isfinite(value).all()) for value in columns):
        raise ValueError('demonstrations must remain finite in actor float32 dtype')
    return columns


def _positive_int(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def action_fit_loss(prediction, target, *, holding_weight=0., wrist_weight=1.):
    """Penalize persistent hold velocities explicitly, without runtime latches.

    Teacher all-zero commands mean retain current motor/wrist targets. Their
    errors must not disappear inside an average over mostly static fingers.
    Default arguments preserve the original reconstruction objective exactly.
    """
    if (not math.isfinite(holding_weight) or holding_weight < 0
            or not math.isfinite(wrist_weight) or wrist_weight <= 0):
        raise ValueError('finite nonnegative holding and positive wrist weights required')
    if (prediction.shape != target.shape or prediction.ndim != 2
            or prediction.shape[-1] != 24 or not len(prediction)
            or not bool(torch.isfinite(prediction).all())
            or not bool(torch.isfinite(target).all())):
        raise ValueError('finite nonempty matching [N,24] actions required')
    error = (prediction - target).square()
    if wrist_weight != 1.:
        axes = error.new_ones(24)
        axes[12:] = wrist_weight
        error = error * axes
    if holding_weight:
        holding = target.abs().amax(-1) < .001
        error = error * (1. + holding_weight * holding.to(error.dtype))[:, None]
    return error.mean()


def pretrain(data, output, *, steps=2000, batch_size=256, seed=42, device='cpu',
             learning_rate=1e-3, warm_start=None, holding_weight=0., wrist_weight=1.,
             extend_contact_history=False):
    """Fit only a new actor, save its surface checkpoint, and return metrics."""
    validate_dataset(data)
    _positive_int(steps, 'steps')
    _positive_int(batch_size, 'batch_size')
    _positive_int(seed, 'seed', minimum=0)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('finite positive learning rate required')
    identity = _dataset_sha256(data)
    observations, actions = _training_tensors(data, device)
    weights = _sampling_weights(data, device)
    torch.manual_seed(seed)
    actor = make_actor(data['contract']['observation_size']).to(device)
    warm_metadata = None
    input_extension = None
    if warm_start is not None:
        source = Path(warm_start).resolve()
        if source == Path(output).resolve():
            raise ValueError('warm start actor must remain immutable')
        previous = torch.load(source, map_location='cpu', weights_only=True)
        if not isinstance(previous, dict) or previous.get('version') != ACTOR_VERSION:
            raise ValueError('current surface actor checkpoint required')
        target_contract=data['contract']
        extending=(extend_contact_history and previous.get('contract',{}).get('observation_size')==351
                   and data['contract']['observation_size']==401)
        if extending:
            target_contract=copy.deepcopy(data['contract'])
            target_contract.update(observation_size=351,observation_version=OBSERVATION_VERSION)
            target_contract.pop('surface_contact_history')
            input_extension=dict(old_observation_size=351,new_observation_size=401,new_columns='zero')
        if 'reset_family' in previous:
            validate_reset_family(previous['reset_family'], previous.get('contract'), target_contract)
        else:
            _check_contract(previous.get('contract'), target_contract)
        state = previous.get('actor')
        if extending and isinstance(state,dict) and isinstance(state.get('0.weight'),torch.Tensor):
            if state['0.weight'].shape!=(128,351):raise ValueError('invalid old surface actor shape')
            state=dict(state)
            state['0.weight']=torch.cat((state['0.weight'],state['0.weight'].new_zeros(128,50)),dim=1)
        expected = actor.state_dict()
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or any(not isinstance(v, torch.Tensor) or v.shape != expected[k].shape
                       or v.dtype != expected[k].dtype or not bool(torch.isfinite(v).all())
                       for k, v in state.items())):
            raise ValueError('finite compatible warm start parameters required')
        actor.load_state_dict(state, strict=True)
        warm_metadata = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        initial_mse = float((actor(observations) - actions).square().mean())
        action_fit_loss(actor(observations[:1]), actions[:1],
                        holding_weight=holding_weight, wrist_weight=wrist_weight)
    for _ in range(steps):
        indices = _sample_indices(len(observations), batch_size, device, weights, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = action_fit_loss(actor(observations[indices]), actions[indices],
                               holding_weight=holding_weight, wrist_weight=wrist_weight)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite surface BC loss')
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_mse = float((actor(observations) - actions).square().mean())
    if not math.isfinite(initial_mse) or not math.isfinite(final_mse):
        raise RuntimeError('nonfinite surface BC metrics')
    metrics = dict(initial_mse=initial_mse, final_mse=final_mse, samples=len(observations),
                   steps=steps, batch_size=batch_size, seed=seed, device=str(device),
                   learning_rate=learning_rate, warm_start=warm_metadata,input_extension=input_extension,
                   holding_weight=holding_weight, wrist_weight=wrist_weight)
    payload = dict(version=ACTOR_VERSION, contract=copy.deepcopy(data['contract']),
                   dataset_version=DATASET_VERSION, dataset_sha256=identity,
                   qualification=copy.deepcopy(data['qualification']), metrics=metrics,
                   actor={key: value.detach().cpu().clone() for key, value in actor.state_dict().items()})
    if 'reset_family' in data:
        payload['reset_family'] = copy.deepcopy(data['reset_family'])
    if weights is not None:
        payload['sample_weights'] = data['sample_weights'].detach().cpu().clone()
        payload['sample_weights_source'] = dict(dataset_sha256=identity,
            method='torch.multinomial', replacement=True)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return metrics


def load_actor(algorithm, path, contract):
    """Load surface actor parameters; retain critic and the entire optimizer.

    Initialize exploration at max(0.03, existing noise-contract minimum) before
    projection, so projection never clears the optimizer's noise moments.
    Return initialization metadata, including the actual per-action std.
    """
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('version') != ACTOR_VERSION:
        raise ValueError(f'expected {ACTOR_VERSION} actor checkpoint')
    if 'reset_family' in payload:
        validate_reset_family(payload['reset_family'], payload.get('contract'), contract)
    else:
        _check_contract(payload.get('contract'), contract)
    if 'sample_weights' in payload:
        _validate_sample_weights(payload['sample_weights'], payload.get('metrics', {}).get('samples'))
    state = payload.get('actor')
    target = algorithm.policy.actor.state_dict()
    if (not isinstance(state, dict) or state.keys() != target.keys()
            or any(not isinstance(value, torch.Tensor) or value.shape != target[key].shape
                   or value.dtype != target[key].dtype or not bool(torch.isfinite(value).all())
                   for key, value in state.items())):
        raise ValueError('surface actor checkpoint has incompatible or nonfinite parameters')
    # Check shapes against the fixed surface architecture before any mutation.
    shapes = {'0.weight': (128, contract['observation_size']), '0.bias': (128,), '2.weight': (128, 128),
              '2.bias': (128,), '4.weight': (24, 128), '4.bias': (24,)}
    if {key: tuple(value.shape) for key, value in state.items()} != shapes:
        raise ValueError('surface actor architecture mismatch')
    algorithm.policy.actor.load_state_dict(state, strict=True)
    with torch.no_grad():
        noise = getattr(algorithm, 'policy_noise_contract', NOISE_CONTRACT)
        initial_std = noise.get('initial_std', max(.03, noise['min_std']))
        algorithm.policy.log_std.fill_(math.log(initial_std))
        project_policy_noise(algorithm)
    metadata = dict(version=ACTOR_VERSION, path=str(path), contract=copy.deepcopy(payload['contract']),
                dataset_version=payload.get('dataset_version'),
                dataset_sha256=payload.get('dataset_sha256'),
                metrics=copy.deepcopy(payload.get('metrics', {})),
                actual_std=algorithm.policy.log_std.detach().exp().cpu().tolist())
    for key in ('reset_family', 'sample_weights_source'):
        if key in payload:
            metadata[key] = copy.deepcopy(payload[key])
    return metadata


class DemonstrationRegularizer:
    """Independent supervised step, called after algorithm.update() completes.

    Demonstrations never enter PPO storage or its sampled actions/logprobs.
    Legacy mode reuses PPO's optimizer. Explicit stable-hold mode has its own
    serializable actor Adam, so opposing PPO moments cannot undo supervision.
    Non-actor gradients are always cleared and never updated by this step.
    """

    def __init__(self, data, contract, device, *, independent_optimizer=False,
                 holding_weight=0., wrist_weight=1., learning_rate=1e-3):
        validate_dataset(data, contract)
        self.contract = copy.deepcopy(data['contract'])
        self.dataset_sha256 = _dataset_sha256(data)
        self.observations, self.actions = _training_tensors(data, device)
        self.reset_family = copy.deepcopy(data.get('reset_family'))
        self.sample_weights = _sampling_weights(data, device)
        action_fit_loss(self.actions[:1], self.actions[:1],
                        holding_weight=holding_weight, wrist_weight=wrist_weight)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('finite positive auxiliary learning rate required')
        self.independent_optimizer = bool(independent_optimizer)
        self.holding_weight, self.wrist_weight = holding_weight, wrist_weight
        self.learning_rate = learning_rate
        self._optimizer = None

    def optimizer_for(self, algorithm):
        if not self.independent_optimizer:
            return algorithm.optimizer
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=self.learning_rate)
        algorithm.surface_imitation_optimizer = self._optimizer
        return self._optimizer

    def update(self, algorithm, iteration, initial_weight=.1, decay=.995, batch_size=256):
        _positive_int(iteration, 'iteration', minimum=0)
        _positive_int(batch_size, 'batch_size')
        if not math.isfinite(initial_weight) or initial_weight < 0:
            raise ValueError('initial_weight must be finite and nonnegative')
        if not math.isfinite(decay) or not 0 <= decay <= 1:
            raise ValueError('decay must be finite within [0,1]')
        weight = initial_weight * decay ** iteration
        actor = algorithm.policy.actor
        indices = _sample_indices(len(self.observations), batch_size,
                                  self.observations.device, self.sample_weights)
        # Clear gradients left by the completed PPO update, including critic.
        algorithm.optimizer.zero_grad(set_to_none=True)
        optimizer = self.optimizer_for(algorithm)
        optimizer.zero_grad(set_to_none=True)
        loss = action_fit_loss(actor(self.observations[indices]), self.actions[indices],
                               holding_weight=self.holding_weight, wrist_weight=self.wrist_weight)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite surface demonstration loss')
        if weight > 0:
            if self.independent_optimizer:
                # Apply the scheduled weight to the ACTUAL step size. Scaling
                # Adam's gradient alone largely cancels inside its moments.
                for group in optimizer.param_groups:
                    group['lr'] = self.learning_rate * weight
                loss.backward(inputs=list(actor.parameters()))
            else:
                (weight * loss).backward(inputs=list(actor.parameters()))
            optimizer.step()
        return dict(loss=float(loss.detach()), weight=weight,
                    optimizer=('independent_actor_adam' if self.independent_optimizer else 'legacy_shared_ppo_adam'),
                    learning_rate=(self.learning_rate * weight if self.independent_optimizer else algorithm.learning_rate))
