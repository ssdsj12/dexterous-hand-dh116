"""Independent Student training checkpoints and actor-only deployment packages.

No Teacher/PPO runtime is imported by actor export/loading.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import math
from pathlib import Path
import random

import numpy as np
import torch

from .tabletop_stereo_contract import (action_sha256, digest, file_sha256, require_sha, validate_contract)
from .tabletop_stereo_student_ppo import (INPUT_KEYS, make_stereo_actor, normalize_network_config)

VERSION = 'dh116-stereo-rgb-student-checkpoint-v1'
DEPLOYMENT_VERSION = 'dh116-stereo-rgb-deployment-actor-v1'
ACTOR_CONTRACT_VERSION = 'dh116-stereo-rgb-actor-contract-v1'


def _cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list): return [_cpu(item) for item in value]
    if isinstance(value, tuple): return tuple(_cpu(item) for item in value)
    return deepcopy(value)


def _finite(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()): raise ValueError('nonfinite checkpoint tensor')
    elif isinstance(value, dict):
        for item in value.values(): _finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value: _finite(item)


def tensor_state_sha256(state):
    if not isinstance(state, dict) or not state: raise ValueError('nonempty model tensor state required')
    hash_ = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not isinstance(value, torch.Tensor): raise ValueError('named tensor state required')
        _finite(value)
        cpu = value.detach().cpu().contiguous()
        hash_.update((key+'|'+str(cpu.dtype)+'|'+str(tuple(cpu.shape))+'|').encode())
        hash_.update(cpu.numpy().tobytes())
    return hash_.hexdigest()


def _source_sha():
    names = ('checkpoint', 'student_ppo', 'state_estimator', 'observation', 'encoder', 'temporal', 'contract')
    return {f'tabletop_stereo_{name}.py': hashlib.sha256(Path(__file__).with_name(
        f'tabletop_stereo_{name}.py').read_bytes()).hexdigest() for name in names}


def _training_source_sha():
    names = ('tabletop_privileged_ppo.py', 'tabletop_squashed_gaussian.py')
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def _validate_ppo_configuration(contract, actual):
    from .tabletop_privileged_ppo import PPOConfig
    configured = asdict(PPOConfig(**contract['config'].get('ppo', {})))
    if configured != actual: raise ValueError('experiment PPO config differs from actual optimizer configuration')


def _actor_contract(contract, network, *, training_provenance=None):
    from .tabletop_stereo_training_lineage import validate_training_provenance
    calibration = require_sha(contract.get('calibration_sha256'), 'calibration SHA')
    provenance=validate_training_provenance(contract['config'].get('provenance') if training_provenance is None
        else training_provenance,require_artifacts=True)
    processing=deepcopy(contract['config'].get('preprocessor'))
    if processing is not None:
        from .tabletop_stereo_preprocessing import validate_processing_contract
        validate_processing_contract(processing)
        if processing['calibration_sha256']!=calibration:raise ValueError('actor preprocessing/calibration contract mismatch')
    result = dict(version=ACTOR_CONTRACT_VERSION, input_keys=list(INPUT_KEYS),
        actor_input='StereoHistoryBatch_only', action_size=24, action_contract_sha256=action_sha256(),
        training_kind=contract['kind'],native_qualified=False,
        network_config_sha256=digest(network), calibration_sha256=calibration,
        training_contract_sha256=contract['contract_sha256'], state_schema_sha256=contract['state_schema_sha256'],
        training_provenance=provenance,
        preprocessor=dict(binding='bound_raw_stereo' if processing is not None else 'history_input_only_unspecified',
            processing=processing,rgb='synchronized_left_right_RGB', depth='computed_from_synchronized_left_right_RGB',
            points='calibrated_stereo_world_points', proprio='measured62_with_previous_issued24',
            missing='explicit_boolean_masks;no_ground_truth_fallback'), source_sha256=_source_sha())
    result['contract_sha256'] = digest(result)
    return result


def _validate_actor_contract(value, network, calibration):
    if not isinstance(value, dict) or value.get('version') != ACTOR_CONTRACT_VERSION:
        raise ValueError('independent actor input contract version required')
    if (value.get('input_keys') != list(INPUT_KEYS) or value.get('actor_input') != 'StereoHistoryBatch_only'
            or value.get('action_size') != 24 or value.get('action_contract_sha256') != action_sha256()):
        raise ValueError('actor input/action contract mismatch; truth keys forbidden')
    if value.get('training_kind') not in ('student','bc','dagger') or value.get('native_qualified') is not False:
        raise ValueError('explicit independent actor training role required; export does not grant native qualification')
    from .tabletop_stereo_training_lineage import validate_training_provenance
    if not isinstance(value.get('training_provenance'),dict):raise ValueError('explicit training provenance metadata required')
    validate_training_provenance(value['training_provenance'])
    if value.get('network_config_sha256') != digest(network): raise ValueError('network config SHA mismatch')
    if value.get('calibration_sha256') != calibration: raise ValueError('calibration contract mismatch')
    require_sha(calibration, 'calibration SHA')
    preprocessor=value.get('preprocessor')
    if not isinstance(preprocessor,dict) or preprocessor.get('binding') not in ('bound_raw_stereo','history_input_only_unspecified'):
        raise ValueError('explicit raw-stereo or history-input-only preprocessor binding required')
    if preprocessor['binding']=='bound_raw_stereo':
        from .tabletop_stereo_preprocessing import validate_processing_contract
        processing=validate_processing_contract(preprocessor.get('processing'))
        if processing['calibration_sha256']!=calibration:raise ValueError('processing/calibration contract mismatch')
    elif preprocessor.get('processing') is not None:
        raise ValueError('unspecified history package cannot claim a raw processing contract')
    if value.get('source_sha256') != _source_sha(): raise ValueError('deployment implementation source SHA mismatch')
    if value.get('contract_sha256') != digest({k: v for k, v in value.items() if k != 'contract_sha256'}):
        raise ValueError('actor contract SHA mismatch')
    return value


def _write_new(path, payload):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    _finite(payload)
    stream = io.BytesIO(); torch.save(payload, stream)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as out: out.write(stream.getvalue())
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), version=payload['version'])


def _boundary(trainer, history_reset):
    if history_reset != 'clear': raise ValueError('explicit history_reset=clear required')
    if not trainer.clean_boundary: raise RuntimeError('clean PPO boundary required; pending/nonempty rollout forbidden')


def _validate_state(state, model):
    current = model.state_dict()
    if (not isinstance(state, dict) or set(state) != set(current)
            or any(not isinstance(state[k], torch.Tensor) or state[k].shape != current[k].shape
                or state[k].dtype != current[k].dtype for k in current)):
        raise ValueError('model topology/input version mismatch')
    _finite(state)


def _training_payload(path):
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('version') != VERSION:
        raise ValueError('independent Student checkpoint required; legacy visual3844 conversion forbidden')
    contract, network = payload.get('contract'), payload.get('network_config')
    validate_contract(contract, kind='student')
    if network != normalize_network_config(network): raise ValueError('normalized network config required')
    configured = contract['config'].get('network')
    if configured is None or normalize_network_config(configured) != network:
        raise ValueError('experiment/network config mismatch')
    _validate_actor_contract(payload.get('actor_contract'), network, contract.get('calibration_sha256'))
    _validate_ppo_configuration(contract, payload.get('ppo_config'))
    if payload.get('training_source_sha256') != _training_source_sha(): raise ValueError('PPO training source SHA mismatch')
    if payload.get('policy_sha256') != tensor_state_sha256(payload.get('policy')):
        raise ValueError('Student policy tensor SHA mismatch')
    _finite(payload)
    return payload


def save_student_checkpoint(path, trainer, contract, *, network_config, history_reset='clear'):
    _boundary(trainer, history_reset)
    validate_contract(contract, kind='student')
    _validate_ppo_configuration(contract, asdict(trainer.config))
    network = normalize_network_config(network_config)
    if network != trainer.model.network_config or normalize_network_config(contract['config'].get('network')) != network:
        raise ValueError('Student model/network/contract mismatch')
    policy = _cpu(trainer.model.state_dict())
    payload = dict(version=VERSION, contract=deepcopy(contract), network_config=network,
        actor_contract=_actor_contract(contract, network,training_provenance=getattr(trainer,'training_provenance',None)),
        policy=policy, policy_sha256=tensor_state_sha256(policy),
        training_source_sha256=_training_source_sha(),
        optimizer=_cpu(trainer.optimizer.state_dict()), ppo_config=asdict(trainer.config),
        policy_version=trainer.policy_version, counters=deepcopy(trainer.counters),
        rollout_steps=trainer.rollout_steps, num_envs=trainer.num_envs, history_reset=history_reset,
        rng=dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []),
        auxiliary_bc_metadata=deepcopy(getattr(getattr(trainer, 'auxiliary_bc', None), 'metadata', None)))
    return _write_new(path, payload)


def _validate_optimizer(state, trainer, updates):
    groups = state.get('param_groups') if isinstance(state, dict) else None
    parameters = list(trainer.model.parameters())
    if not isinstance(groups, list) or len(groups) != 1 or groups[0].get('params') != list(range(len(parameters))):
        raise ValueError('complete named-order Student Adam group required')
    if groups[0].get('lr') != trainer.config.lr or not isinstance(state.get('state'), dict):
        raise ValueError('Student Adam config mismatch')
    expected = trainer.optimizer.state_dict()['param_groups'][0]
    if any(groups[0].get(key) != value for key, value in expected.items() if key != 'params'):
        raise ValueError('Student Adam hyperparameter contract mismatch')
    if updates and not state['state']: raise ValueError('updated Student Adam state missing')
    for index, entry in state['state'].items():
        if type(index) is not int or not 0 <= index < len(parameters) or not isinstance(entry, dict):
            raise ValueError('unknown Adam parameter identity')
        if not {'step', 'exp_avg', 'exp_avg_sq'} <= entry.keys(): raise ValueError('incomplete Student Adam moments')
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key in entry and (not isinstance(entry[key], torch.Tensor)
                    or entry[key].shape != parameters[index].shape or entry[key].dtype != parameters[index].dtype):
                raise ValueError('Student Adam moment shape/dtype mismatch')
        if not isinstance(entry['step'], torch.Tensor) or entry['step'].numel() != 1 or entry['step'].item() < 0:
            raise ValueError('Student Adam counter mismatch')


def load_student_checkpoint(path, trainer, expected_contract, *, history_reset='clear', restore_rng=True):
    _boundary(trainer, history_reset)
    payload = _training_payload(path)
    validate_contract(payload['contract'], kind='student', expected=expected_contract)
    if payload['network_config'] != trainer.model.network_config or payload['ppo_config'] != asdict(trainer.config):
        raise ValueError('Student optimizer/network configuration mismatch')
    if (payload.get('history_reset') != history_reset or payload.get('num_envs') != trainer.num_envs
            or payload.get('rollout_steps') != trainer.rollout_steps):
        raise ValueError('Student rollout/history contract mismatch')
    _validate_state(payload['policy'], trainer.model)
    counters = payload.get('counters')
    if (not isinstance(counters, dict) or set(counters) != {'environment_steps', 'updates'}
            or any(type(v) is not int or v < 0 for v in counters.values())):
        raise ValueError('complete Student counters required')
    if type(payload.get('policy_version')) is not int or payload['policy_version'] < 0:
        raise ValueError('Student policy version counter required')
    _validate_optimizer(payload.get('optimizer'), trainer, counters['updates'])
    noise = payload['policy']['actor.log_std']
    if bool(((noise < math.log(trainer.config.min_std)) | (noise > math.log(trainer.config.max_std))).any()):
        raise ValueError('Student noise contract mismatch')
    # Lazy training-only import: deployment loading never imports Teacher or PPO.
    from .tabletop_privileged_checkpoint import _validate_rng
    _validate_rng(payload.get('rng'))
    candidate = deepcopy(trainer.model)
    candidate.load_state_dict(payload['policy'], strict=True)
    candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=trainer.config.lr)
    candidate_optimizer.load_state_dict(payload['optimizer'])
    trainer.model.load_state_dict(payload['policy'], strict=True)
    trainer.optimizer.load_state_dict(payload['optimizer'])
    trainer.policy_version, trainer.counters = payload['policy_version'], deepcopy(counters)
    trainer.training_provenance=deepcopy(payload['actor_contract']['training_provenance'])
    trainer._new_rollout(); trainer.project_noise()
    if restore_rng:
        rng = payload['rng']; torch.set_rng_state(rng['torch']); random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        if rng['cuda']: torch.cuda.set_rng_state_all(rng['cuda'])
    return dict(version=VERSION, contract=deepcopy(payload['contract']), network_config=deepcopy(payload['network_config']),
        history_reset=history_reset, policy_version=trainer.policy_version, counters=deepcopy(counters),
        auxiliary_bc_metadata=deepcopy(payload.get('auxiliary_bc_metadata')))


def export_actor(checkpoint_path, output_path):
    payload = _training_payload(checkpoint_path)
    state = {key[len('actor.'):]: value for key, value in payload['policy'].items() if key.startswith('actor.')}
    actor = make_stereo_actor(payload['network_config']); _validate_state(state, actor)
    input_contract=deepcopy(payload['actor_contract'])
    input_contract['training_provenance']['exported_from_checkpoint_artifact']=dict(
        path=str(Path(checkpoint_path).resolve()),sha256=file_sha256(checkpoint_path))
    input_contract['contract_sha256']=digest({key:value for key,value in input_contract.items() if key!='contract_sha256'})
    result = dict(version=DEPLOYMENT_VERSION, actor=state, actor_sha256=tensor_state_sha256(state),
        actor_contract=input_contract, network_config=deepcopy(payload['network_config']),
        calibration_sha256=payload['contract']['calibration_sha256'],
        preprocessor=deepcopy(payload['actor_contract']['preprocessor']))
    return _write_new(output_path, result)


def save_actor_package(path,actor,contract,*,network_config):
    """Export actual BC/DAgger/Student actor weights without inventing PPO state."""
    validate_contract(contract)
    if contract.get('kind') not in ('student','bc','dagger'):
        raise ValueError('only independent Student/BC/DAgger actor contracts may deploy')
    from .tabletop_stereo_state_estimator import StereoStudentActor
    if type(actor) is not StereoStudentActor:
        raise ValueError('actual independent StereoStudentActor required; legacy visual3844 conversion forbidden')
    network=normalize_network_config(network_config)
    if normalize_network_config(contract['config'].get('network'))!=network:
        raise ValueError('BC/DAgger actor network config differs from complete source contract')
    # Constructing the topology validator must not perturb ongoing training RNG.
    with torch.random.fork_rng(devices=[]): candidate=make_stereo_actor(network)
    state=_cpu(actor.state_dict());_validate_state(state,candidate)
    if actor.action_contract_sha256!=action_sha256():raise ValueError('actor action semantics SHA mismatch')
    input_contract=_actor_contract(contract,network)
    payload=dict(version=DEPLOYMENT_VERSION,actor=state,actor_sha256=tensor_state_sha256(state),
        actor_contract=input_contract,network_config=network,calibration_sha256=contract['calibration_sha256'],
        preprocessor=deepcopy(input_contract['preprocessor']))
    return _write_new(path,payload)


def load_deployment_actor(path, expected_calibration_sha256=None):
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('version') != DEPLOYMENT_VERSION:
        raise ValueError('independent stereo deployment version required; legacy visual3844 forbidden')
    allowed = {'version', 'actor', 'actor_sha256', 'actor_contract', 'network_config', 'calibration_sha256', 'preprocessor'}
    if set(payload) != allowed: raise ValueError('actor-only package must contain no critic/Teacher/optimizer/env')
    network = payload['network_config']; calibration = payload['calibration_sha256']
    _validate_actor_contract(payload['actor_contract'], network, calibration)
    if payload['preprocessor'] != payload['actor_contract']['preprocessor']: raise ValueError('preprocessor contract mismatch')
    if expected_calibration_sha256 is not None and calibration != expected_calibration_sha256:
        raise ValueError('deployment calibration SHA mismatch')
    if payload['actor_sha256'] != tensor_state_sha256(payload['actor']): raise ValueError('actor tensor SHA mismatch')
    actor = make_stereo_actor(network); _validate_state(payload['actor'], actor)
    actor.load_state_dict(payload['actor'], strict=True); actor.eval()
    actor.deployment_contract = deepcopy(payload['actor_contract'])
    actor.network_config = deepcopy(network)
    actor.calibration_sha256 = calibration
    actor.deployment_package_artifact=dict(path=str(Path(path).resolve()),sha256=file_sha256(path))
    return actor
