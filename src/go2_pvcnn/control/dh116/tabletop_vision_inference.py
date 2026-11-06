"""Deterministic visual actor only: no PPO, optimizer, critic or rollout storage."""
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn

from .tabletop_vision_action_contract import LEGACY_VERSION, SQUASHED_VERSION, validate_squashed_checkpoint
from .tabletop_vision_action_derivation import _validate_source
from .tabletop_vision_student import VisionStudentActor, POLICY_DIM
from .tabletop_vision_evaluation_contract import (
    evaluation_contract, validate_evaluation_contract, file_sha256)


class VisionInference:
    def __init__(self, actor, version, checkpoint_sha256, source_contract, current_contract):
        self.actor = actor.eval().requires_grad_(False)
        self.version = version
        self.checkpoint_sha256 = checkpoint_sha256
        self.provenance = dict(mode='evaluation_only', training_resume=False,
            source_contract=deepcopy(source_contract), evaluation_contract=deepcopy(current_contract),
            deterministic_action=('bounded_v2_actor_output' if version == LEGACY_VERSION else 'tanh_latent_mean'))

    @torch.inference_mode()
    def act(self, observations):
        # Deliberately read only policy. Even a nonfinite privileged critic is
        # outside the actor's call graph and cannot modify issued commands.
        obs = observations if isinstance(observations, torch.Tensor) else observations['policy']
        if obs.ndim != 2 or obs.shape[1] != POLICY_DIM or not bool(torch.isfinite(obs).all()):
            raise ValueError('visual inference requires finite [N,3844] sensor observations')
        obs = obs.to(device=next(self.actor.parameters()).device, dtype=torch.float32)
        action = self.actor(obs)
        if self.version == SQUASHED_VERSION: action = action.tanh()
        if action.shape != (len(obs),24) or not bool(torch.isfinite(action).all()):
            raise ValueError('invalid deterministic visual action')
        return action


def load_vision_inference(checkpoint, current_contract, *, device='cpu'):
    """Validate trusted complete historical checkpoint; load only its actor.

    Actor construction runs on CPU inside fork_rng(devices=[]), so neither
    CPU global RNG nor saved CUDA generators are consumed or restored.
    """
    path = Path(checkpoint)
    digest = file_sha256(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('version') == LEGACY_VERSION: _validate_source(saved)
    elif saved.get('version') == SQUASHED_VERSION: validate_squashed_checkpoint(saved)
    else: raise ValueError('unsupported visual inference checkpoint version')
    current = validate_evaluation_contract(saved['contract'], current_contract)
    with torch.random.fork_rng(devices=[]):
        actor = VisionStudentActor()
        if saved['version'] == SQUASHED_VERSION: actor.action_net[-1] = nn.Identity()
        actor.load_state_dict({key[6:]: value for key,value in saved['policy'].items()
                               if key.startswith('actor.')}, strict=True)
    actor = actor.to(device)
    if file_sha256(path) != digest: raise RuntimeError('checkpoint changed during inference load')
    return VisionInference(actor, saved['version'], digest, saved['contract'], current)
