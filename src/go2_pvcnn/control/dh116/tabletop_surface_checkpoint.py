"""Immutable, offline derivation of the one audited 351D surface PPO topology.

Only the contract and two input matrices (including their Adam moments) change.
Provenance lives in the JSON sidecar so every other checkpoint field survives
exactly. No policy, optimizer, simulator or training runtime is instantiated.
"""
import copy
import hashlib
import json
import math
import numbers
from pathlib import Path
import pickle

import numpy as np
import torch

from .tabletop_surface_contract import (
    ACTION_CONTRACT, OBSERVATION_VERSION, bind_contract, semantic_sha256, validate_contract,
)


VERSION = 'dh116-surface-checkpoint-351-to-371-v1'
CHECKPOINT_VERSION = 'dh116-rsl-tensor-ppo-v1'
LEGACY_OBSERVATION_VERSION = 'dh116-surface-sharing-lift-objective-v2'
MOMENTS = ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')
EXPANDED = ('actor.0.weight', 'critic.0.weight')
# Canonical named_parameters() order: log_std precedes child modules. Never
# infer an optimizer mapping from shape: several biases have identical shapes.
PARAMETERS = (
    ('log_std', (24,)),
    ('actor.0.weight', (128, 351)), ('actor.0.bias', (128,)),
    ('actor.2.weight', (128, 128)), ('actor.2.bias', (128,)),
    ('actor.4.weight', (24, 128)), ('actor.4.bias', (24,)),
    ('critic.0.weight', (128, 351)), ('critic.0.bias', (128,)),
    ('critic.2.weight', (128, 128)), ('critic.2.bias', (128,)),
    ('critic.4.weight', (1, 128)), ('critic.4.bias', (1,)),
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value, path='checkpoint'):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized or not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite or unsupported tensor at ' + path)
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in 'biufc' and not bool(np.isfinite(value).all()):
            raise ValueError('nonfinite array at ' + path)
        if value.dtype.hasobject:
            for index, item in enumerate(value.flat):
                _finite(item, path + '[' + str(index) + ']')
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite(item, path + '.' + str(key))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            _finite(item, path + '[' + str(index) + ']')
    elif isinstance(value, numbers.Complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ValueError('nonfinite scalar at ' + path)


def _same(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict):
        equal = list(left) == list(right) and all(_same(value, right[key]) for key, value in left.items())
        return equal and (not hasattr(left, '__dict__') or _same(vars(left), vars(right)))
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return pickle.dumps(left) == pickle.dumps(right)


def _optimizer_mapping(optimizer, parameters, label):
    if (not isinstance(optimizer, dict) or not isinstance(optimizer.get('state'), dict)
            or not isinstance(optimizer.get('param_groups'), list) or len(optimizer['param_groups']) != 1):
        raise ValueError(label + ': one validated Adam parameter group required')
    names = [name for name, _ in parameters]
    group = optimizer['param_groups'][0]
    identifiers = group.get('params') if isinstance(group, dict) else None
    if (not isinstance(identifiers, list) or any(type(item) is not int for item in identifiers)
            or identifiers != list(range(len(names)))
            or ('param_names' in group and group['param_names'] != names)):
        raise ValueError(label + ': ambiguous optimizer parameter mapping/order')
    if any(type(key) is not int or key not in identifiers for key in optimizer['state']):
        raise ValueError(label + ': unknown optimizer parameter identifier')
    for identifier, (name, shape) in zip(identifiers, parameters):
        entry = optimizer['state'].get(identifier, {})
        if not isinstance(entry, dict):
            raise ValueError(label + ': invalid Adam parameter state: ' + name)
        required = ('step', 'exp_avg', 'exp_avg_sq') + (('max_exp_avg_sq',) if group.get('amsgrad') else ())
        if entry and not all(field in entry for field in required):
            raise ValueError(label + ': incomplete Adam/AMSGrad moments: ' + name)
        for field in MOMENTS:
            if field in entry:
                moment = entry[field]
                if (not isinstance(moment, torch.Tensor) or tuple(moment.shape) != shape
                        or moment.dtype != torch.float32):
                    raise ValueError(label + ': Adam moment shape/dtype mismatch: ' + name + '.' + field)
        if 'step' in entry:
            count = entry['step']
            if isinstance(count, torch.Tensor):
                if count.numel() != 1 or count.is_complex() or count.dtype == torch.bool:
                    raise ValueError(label + ': invalid Adam step: ' + name)
                count = count.item()
            if type(count) not in (int, float) or count < 0 or not math.isfinite(count) or count != int(count):
                raise ValueError(label + ': invalid Adam step: ' + name)
    return dict(zip(names, identifiers))


def _validate_source(saved):
    required = {'version', 'contract', 'policy', 'optimizer', 'iteration', 'timesteps', 'learning_rate',
                'episodes', 'successes', 'recovered', 'reached_counts', 'transition_counts',
                'torch_rng', 'cuda_rng', 'surface_imitation'}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError('complete surface PPO checkpoint and counters required')
    if saved['version'] != CHECKPOINT_VERSION:
        raise ValueError('unsupported surface PPO checkpoint version')
    _finite(saved)
    contract = saved['contract']
    if (not isinstance(contract, dict) or type(contract.get('observation_size')) is not int
            or contract['observation_size'] != 351
            or contract.get('observation_version') != LEGACY_OBSERVATION_VERSION):
        raise ValueError('only the original versioned 351D / 24D surface checkpoint is supported')
    # bind_contract audits the historical action name, dimensions and nested
    # controller semantics. Explicit saved schemas must not contradict it.
    for section in (None, 'policy_control', 'grasp_controller', 'free_wrist_grasp', 'surface_force_sharing'):
        declaration = contract if section is None else contract.get(section, {})
        if not isinstance(declaration, dict):
            raise ValueError('saved action declaration must be a dictionary: ' + str(section))
        for key in ('action_size', 'policy_command_size'):
            if key in declaration and (type(declaration[key]) is not int or declaration[key] != 24):
                raise ValueError('saved action declaration requires 24D: ' + str(section) + '.' + key)
        for key in ('action_contract', 'action_schema', 'surface_action_contract'):
            if key in declaration and declaration[key] != ACTION_CONTRACT:
                raise ValueError('saved explicit action schema mismatch: ' + str(section) + '.' + key)
    new_contract = bind_contract(copy.deepcopy(contract))
    validate_contract(new_contract)
    if new_contract['observation_size'] != 371 or new_contract['observation_version'] != OBSERVATION_VERSION:
        raise ValueError('canonical contract must describe 371D feedback observations')
    policy = saved['policy']
    if not isinstance(policy, dict) or list(policy) != [name for name, _ in PARAMETERS]:
        raise ValueError('policy parameter order must be log_std, six actor, six critic parameters')
    for name, shape in PARAMETERS:
        value = policy[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.float32):
            raise ValueError('policy parameter shape/dtype mismatch: ' + name)
    mappings = {'optimizer': _optimizer_mapping(saved['optimizer'], PARAMETERS, 'PPO')}
    if 'surface_imitation_optimizer' in saved:
        actor = tuple((name.removeprefix('actor.'), shape) for name, shape in PARAMETERS if name.startswith('actor.'))
        mappings['surface_imitation_optimizer'] = _optimizer_mapping(
            saved['surface_imitation_optimizer'], actor, 'surface imitation')
    return new_contract, mappings


def _append(value):
    return torch.cat((value, value.new_zeros((value.shape[0], 20))), dim=1)


def _verify(before, after, contract, mappings):
    checks = dict(
        all_checkpoint_fields_retained=list(before) == list(after),
        counters_rng_metadata_and_other_state_unchanged=all(_same(value, after[key])
            for key, value in before.items() if key not in ('contract', 'policy', *mappings)),
        canonical_contract_bound=_same(contract, after['contract']),
        policy_exact_except_zero_input_columns=True,
        optimizer_exact_except_zero_input_moments=True)
    checks['policy_exact_except_zero_input_columns'] = list(before['policy']) == list(after['policy'])
    for name, value in before['policy'].items():
        checks['policy_exact_except_zero_input_columns'] &= _same(
            _append(value) if name in EXPANDED else value, after['policy'][name])
    for key, mapping in mappings.items():
        expected = copy.deepcopy(before[key])
        for name in EXPANDED:
            local = name.removeprefix('actor.') if key == 'surface_imitation_optimizer' else name
            if local not in mapping:
                continue
            entry = expected['state'].get(mapping[local], {})
            for field in MOMENTS:
                if field in entry:
                    entry[field] = _append(entry[field])
        checks['optimizer_exact_except_zero_input_moments'] &= _same(expected, after[key])
    _finite(after)
    checks['all_numeric_state_finite'] = True
    if not all(checks.values()):
        raise RuntimeError('surface checkpoint preservation verification failed: ' +
                           ', '.join(key for key, valid in checks.items() if not valid))
    return checks


def derive_surface_checkpoint(input_path, output_path):
    """Exclusively create a new .pt and its .json sidecar; return the manifest.

    The source is a trusted full torch checkpoint, loaded on CPU. Canonical
    sequential parameter IDs/order are mandatory for both Adam instances;
    shapes alone cannot resolve same-shaped parameter-order ambiguity.
    """
    source, output = (Path(path).absolute() for path in (input_path, output_path))
    sidecar = output.with_suffix('.json')
    destinations = (output, sidecar)
    if (output.suffix != '.pt' or output.resolve() == sidecar.resolve()
            or any(path.resolve() == source.resolve() for path in destinations)):
        raise ValueError('new .pt checkpoint and .json sidecar must be distinct from source and each other')
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    source_sha = _sha256(source)
    saved = torch.load(source, map_location='cpu', weights_only=False)
    contract, mappings = _validate_source(saved)
    derived = copy.deepcopy(saved)
    derived['contract'] = contract
    for name in EXPANDED:
        derived['policy'][name] = _append(saved['policy'][name])
        for key, mapping in mappings.items():
            local = name.removeprefix('actor.') if key == 'surface_imitation_optimizer' else name
            if local not in mapping:
                continue
            entry = derived[key]['state'].get(mapping[local], {})
            for field in MOMENTS:
                if field in entry:
                    entry[field] = _append(entry[field])
    checks = _verify(saved, derived, contract, mappings)
    manifest = dict(version=VERSION, status='derived_offline', source_checkpoint=str(source),
        derived_checkpoint=str(output), source_sha256_before=source_sha,
        old_contract_sha256=semantic_sha256(saved['contract']), new_contract_sha256=semantic_sha256(contract),
        parameter_order=[name for name, _ in PARAMETERS], optimizer_parameter_mappings=mappings,
        changes=dict(input_columns_before=351, input_columns_after=371, appended_zero_columns=20,
            policy_parameters=list(EXPANDED), adam_moment_fields=list(MOMENTS)), invariant_checks=checks)
    json.dumps(manifest, allow_nan=False)  # Fail before creating any output.
    if _sha256(source) != source_sha:
        raise RuntimeError('source checkpoint changed during derivation')
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open('xb') as checkpoint_stream:
            created.append(output)
            with sidecar.open('x', encoding='utf-8') as manifest_stream:
                created.append(sidecar)
                torch.save(derived, checkpoint_stream)
                checkpoint_stream.flush()
                source_after = _sha256(source)
                if source_after != source_sha:
                    raise RuntimeError('source checkpoint changed during derivation')
                manifest['source_sha256_after'] = source_after
                manifest['derived_sha256'] = _sha256(output)
                json.dump(manifest, manifest_stream, indent=2, allow_nan=False)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return manifest
