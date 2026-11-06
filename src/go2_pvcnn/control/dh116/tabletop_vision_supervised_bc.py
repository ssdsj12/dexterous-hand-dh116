"""Independent sensor-actor fitting to actually issued Teacher/MPC commands.

No PPO object, rollout, latent/log probability, critic or noise parameter is
created or optimized here. The separate BC artifact is not a PPO checkpoint.
An explicit second operation derives a new v3 warm start, preserving the base
PPO state and recording that actor fitting is not an exact PPO resume.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch

from .tabletop_vision_action_contract import validate_squashed_checkpoint
from .tabletop_vision_student import FRAME_DIM, HISTORY, POLICY_DIM
from .tabletop_vision_squashed_policy import UnboundedVisionStudentActor
from .tabletop_vision_quaternion import (
    RAW_WRIST_QUATERNIONS, HEMISPHERE_WRIST_QUATERNIONS,
    validate_wrist_quaternion_preprocessing,
)


VERSION = 'dh116-vision-supervised-bc-v2-visual-gradient-verified'
DERIVATION_VERSION = 'dh116-visual-supervision-derivation-v2'
QUATERNION_VERSION = 'dh116-vision-supervised-bc-v3-wrist-quaternion-hemisphere-v1'
QUATERNION_DERIVATION_VERSION = 'dh116-visual-supervision-derivation-v3-wrist-quaternion-hemisphere-v1'
PREPROCESSING_FIELD = 'actor_wrist_quaternion_preprocessing'
UNIFORM_SAMPLING = 'torch_randint_private_cpu_generator_v1'
WEIGHTED_SAMPLING = 'torch_multinomial_replacement_private_cpu_generator_v1'
GROUPS = dict(fingers=(0, 12), world_xyz=(12, 18), intrinsic_xyz_rotation=(18, 24))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def _same_state(left, right):
    if type(left) is not type(right): return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_same_state(a, b) for a, b in zip(left, right))
    return left == right


def _validate_rows(observations, actions):
    if (not isinstance(observations, torch.Tensor)
            or observations.ndim != 2 or observations.shape[1] != POLICY_DIM
            or len(observations) == 0 or observations.dtype != torch.float32
            or not bool(torch.isfinite(observations).all())):
        raise ValueError('BC requires nonempty finite float32 [S,3844] sensor observations')
    if (not isinstance(actions, torch.Tensor) or actions.shape != (len(observations), 24)
            or actions.dtype != torch.float32 or not bool(torch.isfinite(actions).all())
            or not bool((actions.abs() <= 1).all())):
        raise ValueError('BC requires aligned finite float32 bounded [S,24] issued actions')


def _tensor_sha256(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sampling_weights(weights, count):
    if weights is None: return None
    if (not isinstance(weights, torch.Tensor) or weights.shape != (count,)
            or weights.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(weights).all() & (weights > 0).all())):
        raise ValueError('BC sampling weights require finite positive floating [S] values')
    weights = weights.detach().to(device='cpu', dtype=torch.float64).clone()
    total = weights.sum()
    if not bool(torch.isfinite(total)):
        raise ValueError('BC sampling weights must have a finite positive sum')
    weights /= total
    if not bool((weights > 0).all()):
        raise ValueError('BC sampling probabilities must remain positive')
    return weights


def _validate_startup_sampling(fraction, control_steps, mode):
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 <= fraction <= .5):
        raise ValueError('startup sampling fraction must be finite in [0,0.5]')
    if type(control_steps) is not int or control_steps < 1:
        raise ValueError('startup control steps must be a positive integer')
    if fraction > 0 and mode != HEMISPHERE_WRIST_QUATERNIONS:
        raise ValueError('startup sampling requires explicit canonical quaternion BC')


def _startup_distribution(data, train, fraction, control_steps):
    count = len(train)
    startup = torch.tensor([data['sample_metadata'][i]['tick'] < control_steps
                            for i in train.tolist()], dtype=torch.bool)
    startup_count = int(startup.sum())
    if fraction > 0 and not startup_count:
        raise ValueError('startup sampling requires startup rows in the training subset')
    probabilities = torch.full((count,), (1-fraction)/count, dtype=torch.float64)
    if fraction > 0: probabilities[startup] += fraction/startup_count
    actual = _sampling_weights(probabilities, count)
    return probabilities, dict(fraction=float(fraction), control_steps=control_steps,
        train_sample_count=count, startup_sample_count=startup_count,
        distribution_sha256=_tensor_sha256(actual),
        algorithm=WEIGHTED_SAMPLING if fraction > 0 else UNIFORM_SAMPLING,
        distribution_order='training_subset_in_dataset_order',
        rule='p_i=(1-f)/N+f/K_if_tick_less_than_control_steps',
        probability_dtype='torch.float64', replacement=True)


def _validate_sampling_artifact(artifact, report, mode):
    """Bind weighted derivations to the immutable training split, never heldout rows."""
    identity = report.get('startup_sampling')
    if identity is None:
        if 'startup_sampling' in artifact or 'train_sampling_probabilities' in artifact:
            raise ValueError('BC sampling provenance is incomplete')
        return None  # Historical unweighted artifacts did not carry sampler metadata.
    if not isinstance(identity, dict) or not _same_state(identity, artifact.get('startup_sampling')):
        raise ValueError('BC sampling artifact and report identities differ')
    fraction, steps = identity.get('fraction'), identity.get('control_steps')
    _validate_startup_sampling(fraction, steps, mode)
    count, startup = identity.get('train_sample_count'), identity.get('startup_sample_count')
    if (type(count) is not int or count < 1 or count != report.get('train_sample_count')
            or type(startup) is not int or not 0 <= startup <= count):
        raise ValueError('BC sampling row counts are invalid')
    algorithm = WEIGHTED_SAMPLING if fraction > 0 else UNIFORM_SAMPLING
    probabilities = artifact.get('train_sampling_probabilities')
    if (identity.get('algorithm') != algorithm or report.get('sampling_algorithm') != algorithm
            or not isinstance(probabilities, torch.Tensor) or probabilities.shape != (count,)
            or probabilities.dtype != torch.float64
            or not bool(torch.isfinite(probabilities).all() & (probabilities > 0).all())
            or _tensor_sha256(probabilities) != identity.get('distribution_sha256')):
        raise ValueError('BC sampling distribution does not match its identity')
    if fraction == 0:
        expected = _sampling_weights(torch.full((count,), 1/count, dtype=torch.float64), count)
        if not torch.equal(probabilities, expected):
            raise ValueError('zero startup sampling fraction requires a uniform distribution')
        return None
    if report.get('sampling_distribution_sha256') != identity['distribution_sha256']:
        raise ValueError('BC sampling primitive and training-split distributions differ')
    from .tabletop_vision_supervision_dataset import validate_dataset, split_by_source
    path = Path(report['dataset'])
    if _sha256(path) != report.get('dataset_sha256'):
        raise ValueError('BC sampling dataset SHA differs')
    data = torch.load(path, map_location='cpu', weights_only=False)
    validate_dataset(data, require_successful=True)
    train = split_by_source(data, report['heldout_source_trajectory_ids'])['train_indices']
    raw_expected, expected_identity = _startup_distribution(data, train, fraction, steps)
    if (identity != expected_identity or not torch.equal(probabilities, _sampling_weights(raw_expected, len(train)))
            or sorted({data['sample_metadata'][i]['source_trajectory_id'] for i in train.tolist()})
            != report['train_source_trajectory_ids']):
        raise ValueError('BC sampling provenance differs from the original training subset')
    return path, report['dataset_sha256']


def _initialize_actor(actor_state, mode, seed, wrist_quaternion_preprocessing=RAW_WRIST_QUATERNIONS):
    if mode not in ('source', 'reset_pointnet'):
        raise ValueError('unknown actor initialization; choose source or explicit reset_pointnet')
    actor = UnboundedVisionStudentActor(wrist_quaternion_preprocessing=wrist_quaternion_preprocessing)
    actor.load_state_dict(deepcopy(actor_state), strict=True)
    if any(not bool(torch.isfinite(p).all()) for p in actor.parameters()):
        raise ValueError('BC actor weights must be finite')
    reinitialized = []
    if mode == 'reset_pointnet':
        # A private CPU generator also avoids torch.manual_seed's CUDA-wide
        # side effects when this CPU utility is called from a GPU process.
        generator = torch.Generator(device='cpu').manual_seed(seed)
        for layer in actor.point_net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, math.sqrt(2.), generator=generator)
                torch.nn.init.zeros_(layer.bias)
        reinitialized = [name for name in actor.state_dict() if name.startswith('point_net.')]
    initialized = actor.state_dict()
    if any(not torch.equal(value, initialized[name]) for name, value in actor_state.items()
           if name.startswith('action_net.')):
        raise RuntimeError('encoder initialization must preserve the source action MLP')
    return actor, dict(mode=mode, seed=seed if mode == 'reset_pointnet' else None,
        reinitialized_parameters=reinitialized,
        initialization_scope='explicit_new_actor_only_not_PPO_or_historical_source',
        parameter_sha256_before_BC={name: _tensor_sha256(value) for name, value in initialized.items()},
        source_parameter_sha256={name: _tensor_sha256(value) for name, value in actor_state.items()},
        source_action_MLP_initialization_preserved=True)


@torch.no_grad()
def _visual_branch_audit(actor, observations, batch_size):
    """Audit real coordinate use without altering mask/proprio/history flags.

    This is a numerical encoder/actor test, not a native grasp test. Only the
    point-coordinate slice is zeroed for sensitivity; masks and all 62D
    proprioception, previous action and timing values remain byte-for-byte.
    Counts cover the full supplied subset, not a few favorable observations.
    """
    valid_points = nonzero_coordinates = nonzero_features = 0
    maximum_delta = 0.
    delta_sum = 0.
    sensitive_rows = 0
    rows = len(observations)
    for start in range(0, rows, batch_size):
        obs = observations[start:start+batch_size]
        frames = obs.reshape(-1, HISTORY, FRAME_DIM)
        mask = frames[..., 768:896] > .5
        points = frames[..., :768].reshape(-1, HISTORY, 2, 128, 3)
        points = torch.where(mask[:, :, None, :, None], points, 0.)
        features = actor.point_net(points)
        valid_points += int(mask.sum()) * 2
        nonzero_coordinates += int(((points != 0.) & mask[:, :, None, :, None]).sum())
        nonzero_features += int(((features != 0.) & mask[:, :, None, :, None]).sum())
        zero_points = obs.clone()
        zero_points.reshape(-1, HISTORY, FRAME_DIM)[..., :768] = 0.
        delta = (actor(obs).tanh()-actor(zero_points).tanh()).abs()
        if not bool(torch.isfinite(features).all() & torch.isfinite(delta).all()):
            raise ValueError('nonfinite visual branch audit')
        maximum_delta = max(maximum_delta, float(delta.max()))
        delta_sum += float(delta.double().sum())
        sensitive_rows += int((delta.amax(-1) > 0.).sum())
    return dict(sample_count=rows, valid_hand_point_count=valid_points,
        nonzero_valid_coordinates=nonzero_coordinates,
        nonzero_valid_point_features=nonzero_features,
        zero_points_max_action_delta=maximum_delta,
        zero_points_mean_action_delta=delta_sum/(rows*24),
        zero_points_sensitive_rows=sensitive_rows,
        intervention='zero_coordinates_only_keep_mask_proprio_previous_action_and_timing',
        validation_scope='CPU_encoder_activation_and_action_sensitivity_not_native_grasp')


def audit_actor_visual_branch(actor_state, observations, *, batch_size=64,
                              wrist_quaternion_preprocessing=RAW_WRIST_QUATERNIONS):
    """Read-only, full-subset source-encoder preflight with isolated RNG."""
    validate_wrist_quaternion_preprocessing(wrist_quaternion_preprocessing)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    _validate_rows(observations, torch.zeros(len(observations), 24))
    with torch.random.fork_rng(devices=[]):
        actor, _ = _initialize_actor(actor_state, 'source', 0, wrist_quaternion_preprocessing)
        return _visual_branch_audit(actor, observations.detach().cpu(), batch_size)


def _require_visual_evidence(report):
    audit = report.get('visual_branch_after', {})
    gradient = report.get('pointnet_gradient_norm_max', 0.)
    steps = report.get('pointnet_gradient_nonzero_steps', 0)
    if (not isinstance(audit, dict) or audit.get('valid_hand_point_count', 0) <= 0
            or audit.get('nonzero_valid_coordinates', 0) <= 0
            or audit.get('nonzero_valid_point_features', 0) <= 0
            or audit.get('zero_points_max_action_delta', 0.) <= 0
            or not isinstance(gradient, (int, float)) or not math.isfinite(gradient) or gradient <= 0
            or type(steps) is not int or steps <= 0):
        raise ValueError('visual BC verification failed: nonzero encoder gradients and coordinate sensitivity required')


@torch.no_grad()
def _metrics(actor, observations, actions, batch_size):
    count = len(observations)
    if not count: return None
    squared = torch.zeros(24, dtype=torch.float64)
    absolute = torch.zeros(24, dtype=torch.float64)
    saturated = torch.zeros(24, dtype=torch.float64)
    maximum = 0.
    object_stats = {True: [0., 0], False: [0., 0]}
    for start in range(0, count, batch_size):
        obs, target = observations[start:start+batch_size], actions[start:start+batch_size]
        predicted = actor(obs).tanh()  # Exactly one tanh; labels may include +/-1.
        error = (predicted-target).double()
        squared += error.square().sum(0)
        absolute += error.abs().sum(0)
        saturated += (predicted.abs() >= .99).double().sum(0)
        maximum = max(maximum, error.abs().max().item())
        object_valid = obs.reshape(-1, HISTORY, FRAME_DIM)[:, -1, -1] > .5
        for valid in (True, False):
            selected = error[object_valid == valid]
            object_stats[valid][0] += selected.square().sum().item()
            object_stats[valid][1] += selected.numel()
    return dict(mse=squared.sum().item()/(count*24),
        mae_per_action=(absolute/count).tolist(), max_absolute_error=maximum,
        action_group_mae={name: absolute[lo:hi].sum().item()/(count*(hi-lo))
                          for name, (lo, hi) in GROUPS.items()},
        predicted_saturation_per_action=(saturated/count).tolist(),
        label_saturation_fraction=(actions.abs() >= .99).float().mean().item(),
        object_valid_mse={str(valid).lower(): (total/n if n else None)
                          for valid, (total, n) in object_stats.items()})


def _validate_bc_optimizer(state, actor, learning_rate):
    if not isinstance(state, dict) or set(state) != {'state', 'param_groups'}:
        raise ValueError('BC continuation requires its independent complete Adam state')
    groups = state['param_groups']
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError('BC continuation requires one Adam group')
    group = groups[0]
    parameters = list(actor.parameters())
    if (group.get('params') != list(range(len(parameters)))
            or group.get('lr') != learning_rate or group.get('maximize', False)
            or group.get('amsgrad', False)
            or state['state'].keys() != set(range(len(parameters)))):
        raise ValueError('BC continuation Adam parameter mapping or learning rate differs')
    for identifier, parameter in enumerate(parameters):
        item = state['state'][identifier]
        if set(item) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('BC Adam moment state is incomplete')
        step = item['step']
        if (not isinstance(step, torch.Tensor) or step.ndim != 0
                or not bool(torch.isfinite(step)) or step.item() < 0):
            raise ValueError('BC Adam step is invalid')
        for field in ('exp_avg', 'exp_avg_sq'):
            value = item[field]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                    or (field == 'exp_avg_sq' and not bool((value >= 0).all()))):
                raise ValueError('BC Adam moment shape or values are invalid')


def fit_actor(actor_state, observations, actions, *, steps=100, batch_size=64,
              learning_rate=1e-4, seed=0, optimizer_state=None, sampling_rng_state=None,
              heldout_observations=None, heldout_actions=None, actor_initialization='source',
              wrist_quaternion_preprocessing=RAW_WRIST_QUATERNIONS, sampling_weights=None):
    """Pure CPU actor fitting primitive; qualification is the caller's duty.

    The public file workflow below validates native-success demonstration
    eligibility before calling this primitive. This function is also useful
    for explicitly synthetic CPU tests and continuation of independent BC.
    Process Torch RNG and all caller tensors/states remain unchanged.
    """
    validate_wrist_quaternion_preprocessing(wrist_quaternion_preprocessing)
    _validate_rows(observations, actions)
    probabilities = _sampling_weights(sampling_weights, len(observations))
    for name, value in [('steps', steps), ('batch_size', batch_size)]:
        if type(value) is not int or value < 1: raise ValueError(name+' must be a positive integer')
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('BC learning rate must be finite and positive')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('BC seed must be a nonnegative int64')
    if actor_initialization not in ('source', 'reset_pointnet'):
        raise ValueError('unknown actor initialization; choose source or explicit reset_pointnet')
    if actor_initialization == 'reset_pointnet' and optimizer_state is not None:
        raise ValueError('PointNet initialization requires fresh independent BC Adam, not continuation moments')
    if heldout_observations is not None or heldout_actions is not None:
        _validate_rows(heldout_observations, heldout_actions)
    obs, target = observations.detach().cpu(), actions.detach().cpu()
    held_obs = heldout_observations.detach().cpu() if heldout_observations is not None else None
    held_target = heldout_actions.detach().cpu() if heldout_actions is not None else None
    with torch.random.fork_rng(devices=[]):
        # Module initialization consumes a private scope, not the PPO RNG.
        actor, initialization = _initialize_actor(actor_state, actor_initialization, seed,
                                                  wrist_quaternion_preprocessing)
        optimizer = torch.optim.Adam(actor.parameters(), lr=float(learning_rate))
        if optimizer_state is not None:
            _validate_bc_optimizer(optimizer_state, actor, learning_rate)
            optimizer.load_state_dict(deepcopy(optimizer_state))
        generator = torch.Generator(device='cpu').manual_seed(seed)
        if sampling_rng_state is not None:
            try: generator.set_state(sampling_rng_state.clone())
            except (RuntimeError, AttributeError, TypeError) as error:
                raise ValueError('BC sampling RNG state is invalid') from error
        initial = _metrics(actor, obs, target, batch_size)
        visual_before = _visual_branch_audit(actor, obs, batch_size)
        held_before = _metrics(actor, held_obs, held_target, batch_size) if held_obs is not None else None
        gradient_max = 0.
        pointnet_gradient_max = 0.
        pointnet_nonzero_steps = 0
        point_parameters = {name: value for name, value in actor.named_parameters()
                            if name.startswith('point_net.')}
        gradient_seen = {name: torch.zeros_like(value, dtype=torch.bool)
                         for name, value in point_parameters.items()}
        coverage = {name: dict(parameter_elements=value.numel(), nonzero_gradient_steps=0,
                    l2_norm_max=0., max_absolute_gradient=0.)
                    for name, value in point_parameters.items()}
        for _ in range(steps):
            if probabilities is None:
                indices = torch.randint(len(obs), (min(batch_size, len(obs)),), generator=generator)
            else:
                indices = torch.multinomial(probabilities, min(batch_size, len(obs)),
                                            replacement=True, generator=generator)
            predicted = actor(obs[indices]).tanh()
            loss = (predicted-target[indices]).square().mean()
            if not bool(torch.isfinite(loss)): raise ValueError('nonfinite BC loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            point_squared_norm = 0.
            for name, parameter in point_parameters.items():
                gradient = parameter.grad
                if gradient is None: continue
                if not bool(torch.isfinite(gradient).all()): raise ValueError('nonfinite PointNet BC gradient')
                nonzero = gradient != 0.
                gradient_seen[name] |= nonzero
                coverage[name]['nonzero_gradient_steps'] += int(bool(nonzero.any()))
                norm = float(gradient.double().norm())
                point_squared_norm += norm*norm
                coverage[name]['l2_norm_max'] = max(coverage[name]['l2_norm_max'], norm)
                coverage[name]['max_absolute_gradient'] = max(
                    coverage[name]['max_absolute_gradient'], float(gradient.abs().max()))
            point_norm = math.sqrt(point_squared_norm)
            pointnet_gradient_max = max(pointnet_gradient_max, point_norm)
            pointnet_nonzero_steps += int(point_norm > 0.)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), .5, error_if_nonfinite=True)
            gradient_max = max(gradient_max, gradient_norm.item())
            optimizer.step()
        final = _metrics(actor, obs, target, batch_size)
        visual_after = _visual_branch_audit(actor, obs, batch_size)
        for name, seen in gradient_seen.items():
            count = int(seen.sum())
            coverage[name].update(ever_nonzero_elements=count,
                fraction_ever_nonzero=count/seen.numel())
        held_after = _metrics(actor, held_obs, held_target, batch_size) if held_obs is not None else None
        return dict(actor={k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
            optimizer=deepcopy(optimizer.state_dict()), sampling_rng=generator.get_state().clone(),
            steps=steps, learning_rate=float(learning_rate), batch_size=batch_size, seed=seed,
            loss_before=initial['mse'], loss_after=final['mse'],
            train_metrics_before=initial, train_metrics_after=final,
            heldout_metrics_before=held_before, heldout_metrics_after=held_after,
            gradient_norm_max=gradient_max, physical_success_unverified=True,
            actor_wrist_quaternion_preprocessing=wrist_quaternion_preprocessing,
            sampling_algorithm=UNIFORM_SAMPLING if probabilities is None else WEIGHTED_SAMPLING,
            sampling_distribution_sha256=None if probabilities is None else _tensor_sha256(probabilities),
            actor_initialization=initialization,
            visual_branch_before=visual_before, visual_branch_after=visual_after,
            pointnet_gradient_norm_max=pointnet_gradient_max,
            pointnet_gradient_nonzero_steps=pointnet_nonzero_steps,
            pointnet_gradient_coverage=coverage)


def _new_paths(output_path, inputs):
    output = Path(output_path).absolute()
    sidecar = output.with_suffix('.json')
    if output.suffix != '.pt' or output.resolve() == sidecar.resolve():
        raise ValueError('BC output requires distinct new .pt and .json paths')
    if any(Path(source).resolve() in (output.resolve(), sidecar.resolve()) for source in inputs):
        raise ValueError('BC output must be distinct from every input source')
    for path in (output, sidecar):
        if path.exists() or path.is_symlink(): raise FileExistsError(path)
    return output, sidecar


def _write_exclusive(output, sidecar, saved, report, guard):
    # Serialize metadata and run guards before any destination creation.
    json.dumps(report, allow_nan=False)
    guard()
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open('xb') as stream:
            created.append(output)
            with sidecar.open('x', encoding='utf-8') as manifest:
                created.append(sidecar)
                torch.save(saved, stream)
                stream.flush()
                guard()
                report['output_sha256'] = _sha256(output)
                json.dump(report, manifest, indent=2, allow_nan=False)
    except BaseException:
        for path in reversed(created): path.unlink(missing_ok=True)
        raise
    return report


def fit_supervision(source_checkpoint, dataset_path, artifact_output, *, steps=100,
                    batch_size=64, learning_rate=1e-4, seed=0,
                    heldout_trajectories=(), allow_fit_only=False, actor_initialization='source',
                    wrist_quaternion_preprocessing=RAW_WRIST_QUATERNIONS,
                    startup_sampling_fraction=0., startup_control_steps=4):
    """Validate synchronized labels, then save separate actor-only BC state.

    A full visual contract match is required. Use the existing explicit
    experiment projection if environment count/batching must be changed.
    Source-family heldout is compulsory unless fit-only is explicitly chosen.
    """
    from .tabletop_vision_supervision_dataset import validate_dataset, split_by_source
    validate_wrist_quaternion_preprocessing(wrist_quaternion_preprocessing)
    _validate_startup_sampling(startup_sampling_fraction, startup_control_steps, wrist_quaternion_preprocessing)
    if actor_initialization not in ('source', 'reset_pointnet'):
        raise ValueError('unknown actor initialization; choose source or explicit reset_pointnet')
    source, dataset = Path(source_checkpoint).absolute(), Path(dataset_path).absolute()
    output, sidecar = _new_paths(artifact_output, (source, dataset))
    source_sha, dataset_sha = _sha256(source), _sha256(dataset)
    saved = torch.load(source, map_location='cpu', weights_only=False)
    validate_squashed_checkpoint(saved, saved.get('contract'))
    source_mode = saved['contract'].get(PREPROCESSING_FIELD, RAW_WRIST_QUATERNIONS)
    if source_mode != RAW_WRIST_QUATERNIONS and source_mode != wrist_quaternion_preprocessing:
        raise ValueError('BC cannot silently remove source actor quaternion preprocessing')
    trained_contract = deepcopy(saved['contract'])
    canonical = wrist_quaternion_preprocessing == HEMISPHERE_WRIST_QUATERNIONS
    if canonical:
        trained_contract[PREPROCESSING_FIELD] = wrist_quaternion_preprocessing
    artifact_version = QUATERNION_VERSION if canonical else VERSION
    data = torch.load(dataset, map_location='cpu', weights_only=False)
    validate_dataset(data, require_successful=True)
    if data['visual_contract'] != saved['contract']:
        raise ValueError('BC requires an exact full visual contract; explicitly project the base first')
    split = split_by_source(data, heldout_trajectories)
    fit_only = bool(split['fit_only'])
    if fit_only and not allow_fit_only:
        raise ValueError('no independent source holdout: explicitly allow fit-only; no generalization claim')
    train, held = split['train_indices'], split['heldout_indices']
    if len(train) == 0: raise ValueError('BC requires a nonempty training source subset')
    probabilities, sampling_identity = _startup_distribution(data, train,
        startup_sampling_fraction, startup_control_steps)
    obs, actions = data['observations'], data['actions']
    actor_state = {name.removeprefix('actor.'): value for name, value in saved['policy'].items()
                   if name.startswith('actor.')}
    source_visual = audit_actor_visual_branch(actor_state, obs, batch_size=batch_size,
        wrist_quaternion_preprocessing=source_mode)
    if (source_visual['valid_hand_point_count'] <= 0
            or source_visual['nonzero_valid_coordinates'] <= 0):
        raise ValueError('visual BC requires real valid nonzero point coordinates')
    if actor_initialization == 'source' and source_visual['nonzero_valid_point_features'] == 0:
        raise ValueError('inactive PointNet on all valid dataset points; choose explicit reset_pointnet')
    fitted = fit_actor(actor_state, obs[train], actions[train], steps=steps,
        batch_size=batch_size, learning_rate=learning_rate, seed=seed,
        heldout_observations=obs[held] if len(held) else None,
        heldout_actions=actions[held] if len(held) else None,
        actor_initialization=actor_initialization,
        wrist_quaternion_preprocessing=wrist_quaternion_preprocessing,
        sampling_weights=probabilities if startup_sampling_fraction > 0 else None)
    _require_visual_evidence(fitted)
    held_ids = sorted({data['sample_metadata'][i]['source_trajectory_id'] for i in held.tolist()})
    train_ids = sorted({data['sample_metadata'][i]['source_trajectory_id'] for i in train.tolist()})
    styles = set(data['grasp_styles'])
    train_styles = {data['grasp_styles'][i] for i in train.tolist()}
    held_styles = {data['grasp_styles'][i] for i in held.tolist()}
    report = dict(version=artifact_version, status='offline_visual_gradient_verified_not_native_validation',
        artifact=str(output), source_checkpoint=str(source),
        source_checkpoint_sha256=source_sha, dataset=str(dataset), dataset_sha256=dataset_sha,
        teacher_source_sha256=data['teacher_source_sha256'],
        source_iteration=saved['iteration'], source_timesteps=saved['timesteps'],
        train_sample_count=len(train), heldout_sample_count=len(held), fit_only=fit_only,
        train_source_trajectory_ids=train_ids, heldout_source_trajectory_ids=held_ids,
        independent_source_trajectory_count=split['source_trajectory_count'],
        per_style_independent_holdout_complete=(styles <= train_styles and styles <= held_styles),
        split='whole_source_trajectory_never_random_rows',
        action_target='actual_issued_bounded_24D_not_latent_or_reference',
        loss='mean_squared_error_of_single_tanh_actor_output',
        actor_observation_size=3844, critic_observation_size=371, action_size=24,
        native_success_claimed=False, physical_success_unverified=True,
        visual_gradient_verified=True,
        source_visual_branch_audit=source_visual,
        startup_sampling=deepcopy(sampling_identity),
        visual_verification_scope='nonzero_encoder_gradient_and_coordinate_sensitivity_only_not_grasp',
        ppo_updates_performed=0, ppo_rollout_modified=False,
        optimized_parameters=sorted(actor_state),
        preserved='critic;log_std;PPO_Adam;learning_rate;counters;saved_RNG;existing_provenance',
        **{key: deepcopy(value) for key, value in fitted.items()
           if key not in ('actor', 'optimizer', 'sampling_rng', 'physical_success_unverified')})
    artifact = dict(version=artifact_version, source_checkpoint_state=deepcopy(saved),
        trained_actor=fitted['actor'], bc_optimizer=fitted['optimizer'],
        bc_sampling_rng=fitted['sampling_rng'], visual_contract=deepcopy(saved['contract']),
        actor_wrist_quaternion_preprocessing=wrist_quaternion_preprocessing,
        trained_actor_contract=trained_contract,
        startup_sampling=deepcopy(sampling_identity),
        train_sampling_probabilities=_sampling_weights(probabilities, len(train)),
        report=deepcopy(report))
    def guard():
        if _sha256(source) != source_sha or _sha256(dataset) != dataset_sha:
            raise RuntimeError('source checkpoint or dataset changed during BC fitting')
    return _write_exclusive(output, sidecar, artifact, report, guard)


def derive_supervised_checkpoint(bc_artifact, output_path):
    """Explicitly derive a new complete v3 PPO warm start from a BC artifact.

    Retained PPO Adam moments are deliberate warm-start state, not fresh BC
    moments. A later explicit Adam-reset experiment remains a separate choice.
    """
    artifact_path = Path(bc_artifact).absolute()
    output, sidecar = _new_paths(output_path, (artifact_path,))
    artifact_sha = _sha256(artifact_path)
    artifact = torch.load(artifact_path, map_location='cpu', weights_only=False)
    if not isinstance(artifact, dict) or artifact.get('version') not in (VERSION, QUATERNION_VERSION):
        raise ValueError('supervised derivation requires the independent BC artifact version')
    before = artifact.get('source_checkpoint_state')
    validate_squashed_checkpoint(before, artifact.get('visual_contract'))
    old_report = artifact.get('report')
    if not isinstance(old_report, dict) or old_report.get('version') != artifact['version']:
        raise ValueError('BC artifact requires source and fitting provenance')
    mode = artifact.get(PREPROCESSING_FIELD, RAW_WRIST_QUATERNIONS)
    validate_wrist_quaternion_preprocessing(mode)
    canonical = artifact['version'] == QUATERNION_VERSION
    expected_mode = HEMISPHERE_WRIST_QUATERNIONS if canonical else RAW_WRIST_QUATERNIONS
    if mode != expected_mode or old_report.get(PREPROCESSING_FIELD, RAW_WRIST_QUATERNIONS) != mode:
        raise ValueError('BC artifact version and quaternion preprocessing metadata differ')
    trained_contract = deepcopy(before['contract'])
    source_mode = trained_contract.get(PREPROCESSING_FIELD, RAW_WRIST_QUATERNIONS)
    if canonical:
        trained_contract[PREPROCESSING_FIELD] = mode
    elif source_mode != RAW_WRIST_QUATERNIONS:
        raise ValueError('legacy BC artifact cannot remove source actor quaternion preprocessing')
    declared_contract = artifact.get('trained_actor_contract', None if canonical else trained_contract)
    if not _same_state(declared_contract, trained_contract):
        raise ValueError('trained actor contract must change only the explicit quaternion preprocessing')
    sampling_source = _validate_sampling_artifact(artifact, old_report, mode)
    if old_report.get('visual_gradient_verified') is not True:
        raise ValueError('BC artifact requires verified visual gradient and sensitivity evidence')
    _require_visual_evidence(old_report)
    source = Path(old_report['source_checkpoint'])
    if source.resolve() in (output.resolve(), sidecar.resolve()):
        raise ValueError('derived output cannot replace the original source checkpoint')
    source_sha = old_report['source_checkpoint_sha256']
    if _sha256(source) != source_sha: raise ValueError('original source checkpoint SHA differs')
    original = torch.load(source, map_location='cpu', weights_only=False)
    if not _same_state(original, before):
        raise ValueError('BC base state differs from the immutable source checkpoint')
    trained = artifact.get('trained_actor')
    with torch.random.fork_rng(devices=[]):
        check = UnboundedVisionStudentActor(wrist_quaternion_preprocessing=mode)
        check.load_state_dict(trained, strict=True)
        if any(not bool(torch.isfinite(p).all()) for p in check.parameters()):
            raise ValueError('trained BC actor must be finite')
    derived = deepcopy(before)
    derived['contract'] = deepcopy(trained_contract)
    for name, value in trained.items(): derived['policy']['actor.'+name] = value.detach().cpu().clone()
    derivation_version = QUATERNION_DERIVATION_VERSION if canonical else DERIVATION_VERSION
    meta = dict(version=derivation_version, status='warm_start_not_exact_resume',
        source_checkpoint=str(source), source_checkpoint_sha256=source_sha,
        bc_artifact=str(artifact_path), bc_artifact_sha256=artifact_sha,
        source_iteration=before['iteration'], source_timesteps=before['timesteps'],
        dataset_sha256=old_report['dataset_sha256'],
        teacher_source_sha256=old_report['teacher_source_sha256'],
        fit_report=deepcopy(old_report),
        ppo_Adam='preserved_exactly_not_the_independent_BC_Adam',
        actor_wrist_quaternion_preprocessing=mode,
        physics_and_action_contract='unchanged',
        physical_success_unverified=True, native_success_claimed=False)
    history = deepcopy(derived.get('visual_supervision_history', []))
    if not isinstance(history, list): raise ValueError('supervision history must be a list')
    if ('visual_supervision_derivation' in derived
            and (not history or history[-1] != derived['visual_supervision_derivation'])):
        history.append(deepcopy(derived['visual_supervision_derivation']))
    history.append(deepcopy(meta))
    derived.update(visual_supervision_derivation=meta, visual_supervision_history=history)
    validate_squashed_checkpoint(derived, trained_contract)
    report = dict(version=derivation_version, status='derived_offline',
        output_checkpoint=str(output), visual_supervision_derivation=deepcopy(meta),
        complete_ppo_update_performed=False, physical_success_unverified=True)
    def guard():
        if _sha256(artifact_path) != artifact_sha or _sha256(source) != source_sha:
            raise RuntimeError('BC artifact or original checkpoint changed during derivation')
        if sampling_source is not None and _sha256(sampling_source[0]) != sampling_source[1]:
            raise RuntimeError('BC sampling dataset changed during derivation')
    return _write_exclusive(output, sidecar, derived, report, guard)
