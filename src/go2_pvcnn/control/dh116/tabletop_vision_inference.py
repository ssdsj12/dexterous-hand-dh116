"""Deterministic visual actor only: no PPO, optimizer, critic or rollout storage."""
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn

from .tabletop_vision_action_contract import LEGACY_VERSION, SQUASHED_VERSION
from .tabletop_vision_action_derivation import _validate_source
from .tabletop_vision_student import VisionStudentActor, POLICY_DIM
from .tabletop_vision_evaluation_contract import (
    evaluation_contract, validate_evaluation_contract, file_sha256)


class VisionInference:
    def __init__(self, actor, version, checkpoint_sha256, source_contract, current_contract, *,
                 source_version=None, physical_adapter_contract=None, native_qualification=None):
        self.actor = actor.eval().requires_grad_(False)
        self.version = version
        self.source_version = version if source_version is None else source_version
        self.checkpoint_sha256 = checkpoint_sha256
        self._physical_adapter_contract = deepcopy(physical_adapter_contract)
        self._physical_adapter_verified = physical_adapter_contract is None
        self.provenance = dict(mode='evaluation_only', training_resume=False,
            source_contract=deepcopy(source_contract), evaluation_contract=deepcopy(current_contract),
            deterministic_action=('bounded_v2_actor_output' if version == LEGACY_VERSION else 'tanh_latent_mean'))
        if physical_adapter_contract is not None:
            self.provenance.update(source_checkpoint_version=self.source_version,
                physical_action_adapter=deepcopy(physical_adapter_contract),
                physical_adapter_native_qualification=deepcopy(native_qualification),
                physical_action_adapter_runtime_verified=False,
                physical_action_adapter_execution='original_SDK24_identity_mapping_after_runtime_verification')

    @property
    def requires_physical_adapter_verification(self):
        return self._physical_adapter_contract is not None

    def verify_physical_adapter(self, actual_contract):
        """Bind current native code/assets before any V4 Actor invocation."""
        if self._physical_adapter_contract is None:
            raise ValueError('this visual checkpoint has no physical adapter binding')
        from .tabletop_bimanual_checkpoint_contract import require_physical_adapter_runtime_match
        self._physical_adapter_verified = False
        self.provenance['physical_action_adapter_runtime_verified'] = False
        verified = require_physical_adapter_runtime_match(self._physical_adapter_contract, actual_contract)
        self._physical_adapter_verified = True
        self.provenance['physical_action_adapter_runtime_verified'] = True
        return verified

    @torch.inference_mode()
    def act(self, observations):
        if not self._physical_adapter_verified:
            raise RuntimeError('V4 visual inference requires actual runtime physical adapter verification before act')
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


def load_vision_inference(checkpoint, current_contract, *, device='cpu', allow_physical_adapter=False):
    """Validate trusted complete historical checkpoint; load only its actor.

    Actor construction runs on CPU inside fork_rng(devices=[]), so neither
    CPU global RNG nor saved CUDA generators are consumed or restored.
    """
    path = Path(checkpoint)
    digest = file_sha256(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    source_version = saved.get('version')
    original_contract = deepcopy(saved.get('contract'))
    adapter = qualification = None
    from .tabletop_bimanual_checkpoint_contract import (
        VERSION as PHYSICAL_VERSION, ADAPTER_KEY, QUALIFICATION_KEY, unwrap_physical_adapter_checkpoint)
    if source_version == PHYSICAL_VERSION:
        saved = unwrap_physical_adapter_checkpoint(saved, allow_explicit=allow_physical_adapter)
        adapter = original_contract[ADAPTER_KEY]
        qualification = original_contract[QUALIFICATION_KEY]
    elif any(key in (original_contract or {}) for key in (ADAPTER_KEY, QUALIFICATION_KEY)):
        raise ValueError('physical adapter metadata requires independent V4 version')
    if saved.get('version') == LEGACY_VERSION: _validate_source(saved)
    elif saved.get('version') == SQUASHED_VERSION: saved = unwrap_physical_adapter_checkpoint(saved)
    else: raise ValueError('unsupported visual inference checkpoint version')
    current = validate_evaluation_contract(saved['contract'], current_contract)
    with torch.random.fork_rng(devices=[]):
        preprocessing=saved['contract'].get('actor_wrist_quaternion_preprocessing','raw')
        from .tabletop_vision_point_activation import point_activation_from_contract
        point_activation = point_activation_from_contract(saved['contract'])
        if saved['version']==LEGACY_VERSION and preprocessing!='raw':
            raise ValueError('versioned quaternion preprocessing requires a derived v3 model')
        if saved['version']==LEGACY_VERSION and point_activation!='relu':
            raise ValueError('versioned PointNet activation requires a derived v3 model')
        actor = VisionStudentActor(wrist_quaternion_preprocessing=preprocessing,
                                  point_activation=point_activation)
        if saved['version'] == SQUASHED_VERSION: actor.action_net[-1] = nn.Identity()
        actor.load_state_dict({key[6:]: value for key,value in saved['policy'].items()
                               if key.startswith('actor.')}, strict=True)
    actor = actor.to(device)
    if file_sha256(path) != digest: raise RuntimeError('checkpoint changed during inference load')
    return VisionInference(actor, saved['version'], digest, original_contract, current,
                           source_version=source_version, physical_adapter_contract=adapter,
                           native_qualification=qualification)
