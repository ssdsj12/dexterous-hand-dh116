"""Explicit offline v2 Gaussian/clamp to v3 tanh-Gaussian warm start.

The source is a trusted, complete visual PPO checkpoint, loaded only on CPU.
Removing the actor's final parameterless Tanh changes no weight or Adam shape.
The distribution does change, so this operation is never an exact v2 resume.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from .tabletop_surface_contract import ACTION_CONTRACT
from .tabletop_vision_action_contract import (
    LEGACY_VERSION, SQUASHED_VERSION, assert_finite_state,
    upgrade_action_contract, validate_squashed_checkpoint,
)
from .tabletop_vision_resume import _same, _validate_saved
from .tabletop_vision_checkpoint_state import validate_visual_checkpoint_state


VERSION = 'dh116-visual-action-derivation-v1'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(saved):
    _validate_saved(saved)
    assert_finite_state(saved)
    if not _same(saved['contract'].get('action_contract'), ACTION_CONTRACT):
        raise ValueError('visual action derivation requires the unchanged physical 24D action contract')
    if 'action_derivation' in saved:
        raise ValueError('v2 source cannot already contain an action derivation')
    validate_visual_checkpoint_state(saved)


def derive_action_checkpoint(source_path, output_path):
    """Exclusively create a new .pt and same-name .json derivation report.

    Model/Adam/LR, saved Torch RNG, counters and historical migration metadata
    are copied exactly. Python/NumPy RNG absent from v2 is not invented.
    """
    source, output = (Path(path).absolute() for path in (source_path, output_path))
    sidecar = output.with_suffix('.json')
    if (output.suffix != '.pt' or output.resolve() == sidecar.resolve()
            or source.resolve() in (output.resolve(), sidecar.resolve())):
        raise ValueError('new .pt and .json paths must be distinct from the source')
    for path in (output, sidecar):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    source_sha = _sha256(source)
    saved = torch.load(source, map_location='cpu', weights_only=False)
    _validate_source(saved)
    contract = upgrade_action_contract(saved['contract'])
    meta = dict(version=VERSION, status='warm_start_not_exact_resume',
        source_checkpoint=str(source), source_sha256=source_sha,
        source_version=LEGACY_VERSION, source_iteration=saved['iteration'],
        source_timesteps=saved['timesteps'], target_version=SQUASHED_VERSION,
        source_distribution='gaussian_clip', target_distribution='tanh_gaussian',
        distribution_change='Gaussian_about_bounded_mean_plus_actuator_clamp_to_tanh_of_latent_Gaussian',
        deterministic_action='legacy_actor_output_equals_tanh_of_new_unbounded_latent_mean',
        parameters_and_Adam='unchanged;final_actor_Tanh_has_no_parameters',
        rng_coverage=dict(preserved=['torch_cpu', 'torch_cuda_saved_states'],
            unavailable_in_source=['python', 'numpy'], initialization='none'))
    derived = deepcopy(saved)
    derived.update(version=SQUASHED_VERSION, contract=contract, action_derivation=meta)
    validate_squashed_checkpoint(derived, contract)
    report = dict(version=VERSION, status='derived_offline', source_checkpoint=str(source),
        derived_checkpoint=str(output), action_derivation=deepcopy(meta),
        source_sha256_before=source_sha,
        copied_checkpoint_fields=[key for key in saved if key not in ('version', 'contract')])
    json.dumps(report, allow_nan=False)  # Reject unserializable metadata before creating paths.
    if _sha256(source) != source_sha:
        raise RuntimeError('source checkpoint changed during action derivation')
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        # Exclusive opens are also the race guards. Only files successfully
        # opened by this process enter cleanup; a competing writer is retained.
        with output.open('xb') as checkpoint_stream:
            created.append(output)
            with sidecar.open('x', encoding='utf-8') as manifest_stream:
                created.append(sidecar)
                torch.save(derived, checkpoint_stream)
                checkpoint_stream.flush()
                source_after = _sha256(source)
                if source_after != source_sha:
                    raise RuntimeError('source checkpoint changed during action derivation')
                report['source_sha256_after'] = source_after
                report['derived_sha256'] = _sha256(output)
                json.dump(report, manifest_stream, indent=2, allow_nan=False)
    except BaseException:
        for path in reversed(created): path.unlink(missing_ok=True)
        raise
    return report
