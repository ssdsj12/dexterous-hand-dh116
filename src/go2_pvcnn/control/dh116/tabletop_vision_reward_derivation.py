"""Exclusive, explicit reward-only warm-start forks of a complete v3 model."""
from copy import deepcopy
import json
import os
from pathlib import Path

import torch

from .tabletop_lift_reward_experiment import FACTORS, experiment_contract
from .tabletop_surface_reward_ablation import ablation_contract
from .tabletop_surface_contract import semantic_sha256
from .tabletop_vision_action_contract import (
    LEGACY_VERSION, _ADDED, upgrade_action_contract, validate_squashed_checkpoint)
from .tabletop_vision_action_derivation import _sha256
from .tabletop_vision_resume import _same

VERSION = 'dh116-vision-reward-experiment-derivation-v1'


def reward_experiment_contract(contract, profile):
    """Change the new reward metadata only; retain every physical field."""
    if profile not in FACTORS or profile == 'R0':
        raise ValueError('an explicit non-R0 reward experiment is required')
    base = deepcopy(contract)
    critic = base['privileged_critic_contract']
    if (not _same(critic.get('surface_reward_ablation'), ablation_contract('support_deficit'))
            or 'surface_lift_reward_experiment' in critic):
        raise ValueError('reward fork requires the unchanged R0 support_deficit checkpoint')
    critic['surface_lift_reward_experiment'] = experiment_contract(profile)
    if 'surface_contract_sha256' in critic:
        critic['surface_contract_sha256'] = semantic_sha256(critic)
    base['version'] = LEGACY_VERSION
    for key in _ADDED: base.pop(key, None)
    base['network'].pop('actor_output', None)
    return upgrade_action_contract(base)


def derive_reward_experiment_checkpoint(source_path, output_path, profile):
    source, output = (Path(p).absolute() for p in (source_path, output_path))
    sidecar = output.with_suffix('.json')
    if output.suffix != '.pt' or source.resolve() in (output.resolve(), sidecar.resolve()):
        raise ValueError('reward fork requires new distinct .pt and .json paths')
    for path in (output, sidecar):
        if path.exists() or path.is_symlink(): raise FileExistsError(path)
    source_sha = _sha256(source)
    saved = validate_squashed_checkpoint(torch.load(source, map_location='cpu', weights_only=False))
    contract = reward_experiment_contract(saved['contract'], profile)
    result = deepcopy(saved)
    metadata = dict(version=VERSION, status='reward_warm_start_not_exact_resume',
        source_path=str(source), source_sha256=source_sha, source_iteration=saved['iteration'],
        profile=profile, physical_action_camera_and_reset_parameters='unchanged',
        state='policy_Adam_LR_Torch_RNG_global_counters_preserved',
        original_action_derivation=deepcopy(saved.get('action_derivation')))
    result.update(contract=contract, reward_experiment_derivation=metadata)
    validate_squashed_checkpoint(result, contract)
    report = dict(metadata, output_path=str(output))
    json.dumps(report, allow_nan=False)
    if _sha256(source) != source_sha: raise RuntimeError('reward source changed while loading')
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open('xb') as stream:
            created.append(output)
            with sidecar.open('x', encoding='utf-8') as meta:
                created.append(sidecar)
                torch.save(result, stream); stream.flush(); os.fsync(stream.fileno())
                if _sha256(source) != source_sha: raise RuntimeError('reward source changed while deriving')
                report['checkpoint_sha256'] = _sha256(output)
                json.dump(report, meta, indent=2, allow_nan=False)
                meta.flush(); os.fsync(meta.fileno())
    except BaseException:
        for path in reversed(created): path.unlink(missing_ok=True)
        raise
    return report
