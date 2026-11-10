"""Separate expert success admission and actor-only supervised optimization."""
import hashlib
import json
import math
from pathlib import Path

import torch

from .tabletop_surface_contract import ACTION_CONTRACT

DATASET_VERSION = 'dh116-privileged-native-expert-v1'


def masked_group_normalized_loss(predictions, targets, valid_mask):
    if (predictions.ndim != 2 or predictions.shape[-1] != 24 or targets.shape != predictions.shape
            or valid_mask.shape != predictions.shape or valid_mask.dtype != torch.bool
            or valid_mask.device != predictions.device or targets.device != predictions.device):
        raise ValueError('matching actions/targets and boolean mask [N,24] required')
    if not bool(torch.isfinite(predictions).all()) or bool((valid_mask & ~torch.isfinite(targets)).any()):
        raise ValueError('finite predictions and valid targets required')
    safe_targets = torch.where(valid_mask, targets, predictions.detach())
    error = (predictions-safe_targets).square()
    group_losses, active = [], []
    for start, stop in ((0, 12), (12, 18), (18, 24)):
        count = valid_mask[:, start:stop].sum(-1)
        group_losses.append((error[:, start:stop]*valid_mask[:, start:stop]).sum(-1)/count.clamp_min(1))
        active.append(count > 0)
    losses, enabled = torch.stack(group_losses, -1), torch.stack(active, -1)
    per_row = losses.sum(-1)/enabled.sum(-1).clamp_min(1)
    rows = enabled.any(-1)
    return (per_row*rows).sum()/rows.sum().clamp_min(1)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_expert_dataset(payload, *, expected_identity):
    if not isinstance(payload, dict) or payload.get('version') != DATASET_VERSION:
        raise ValueError('versioned privileged expert dataset required')
    if payload.get('schema_only') is not False:
        raise ValueError('schema-only fixture cannot qualify native success dataset')
    if not isinstance(expected_identity, dict) or payload.get('identity') != expected_identity:
        raise ValueError('expert action/state/source identity mismatch')
    for key in ('state_contract_sha256', 'action_contract_sha256'):
        if not _sha(expected_identity.get(key)):
            raise ValueError('expert current contract SHA required')
    action_sha = hashlib.sha256(json.dumps(ACTION_CONTRACT, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if expected_identity['action_contract_sha256'] != action_sha:
        raise ValueError('expert requires canonical actual 24D action semantics')
    observations, actions, mask = (payload.get(k) for k in ('observations', 'actions', 'valid_mask'))
    if (not isinstance(observations, torch.Tensor) or observations.ndim != 2 or observations.shape[1] != 371
            or not observations.is_floating_point() or not bool(torch.isfinite(observations).all())):
        raise ValueError('finite privileged observations [T,371] required')
    n = len(observations)
    if (n < 1 or not isinstance(actions, torch.Tensor) or actions.shape != (n, 24)
            or not actions.is_floating_point() or not bool(torch.isfinite(actions).all())
            or bool((actions.abs() > 1).any())):
        raise ValueError('finite bounded native actions [T,24] required')
    if not isinstance(mask, torch.Tensor) or mask.shape != actions.shape or mask.dtype != torch.bool:
        raise ValueError('explicit expert action validity mask required')
    episodes = payload.get('episodes')
    if not isinstance(episodes, list) or not episodes:
        raise ValueError('complete native episode evidence required')
    cursor, identities = 0, set()
    for episode in episodes:
        if (not isinstance(episode, dict) or episode.get('complete') is not True
                or episode.get('first_episode') is not True or episode.get('native_success') is not True
                or episode.get('recovered') is not False or episode.get('restored') is not False):
            raise ValueError('only complete first native success without recovery/restore is admitted')
        name, start, end = episode.get('episode_id'), episode.get('start'), episode.get('end')
        if (not isinstance(name, str) or not name or name in identities or type(start) is not int
                or type(end) is not int or start != cursor or not start < end <= n):
            raise ValueError('unique contiguous native episode identities/ranges required')
        identities.add(name)
        cursor = end
        for key in ('source_sha256', 'physical_config_sha256', 'proof_sha256'):
            if not _sha(episode.get(key)):
                raise ValueError('source/physical/proof SHA required')
        source_path = episode.get('source_path')
        if not isinstance(source_path, str) or hashlib.sha256(Path(source_path).read_bytes()).hexdigest() != episode['source_sha256']:
            raise ValueError('actual source artifact path/SHA binding required')
        path = episode.get('proof_path')
        if not isinstance(path, str):
            raise ValueError('native success proof file required')
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != episode['proof_sha256']:
            raise ValueError('native success proof SHA mismatch')
        proof = json.loads(raw)
        if proof.get('evidence_kind') != 'native_isaac_physx' or proof.get('schema_only') is not False:
            raise ValueError('actual native evidence required; fixture cannot qualify success')
        for key in ('episode_id', 'source_sha256', 'physical_config_sha256', 'complete', 'first_episode',
                    'recovered', 'restored', 'native_success'):
            if proof.get(key) != episode.get(key):
                raise ValueError('native proof/episode binding mismatch: ' + key)
        if proof.get('identity') != expected_identity:
            raise ValueError('native success proof contract mismatch')
        qualification = episode.get('qualification_evidence')
        if not isinstance(qualification, (dict, str, Path)):
            raise ValueError('raw native episode qualification_evidence required; metadata cannot admit success')
        try:
            from .tabletop_stereo_promotion import verify_native_episode
        except ImportError as error:
            raise ValueError('raw native episode qualification verifier unavailable; success pool is closed') from error
        verified = verify_native_episode(qualification, expected_identity=expected_identity)
        if not isinstance(verified, dict) or verified.get('episode_uid', verified.get('episode_id')) != name:
            raise ValueError('raw native episode identity differs from expert episode')
        metrics = proof.get('success_metrics', {})
        metric_names = ('net_lift_m', 'continuous_hold_s', 'formal_anchor_slip_m')
        finite_metrics = (isinstance(metrics, dict) and all(type(metrics.get(k)) in (float, int)
            and math.isfinite(metrics[k]) for k in metric_names))
        if not (finite_metrics and metrics.get('net_lift_m', -1) >= .03
                and metrics.get('continuous_hold_s', -1) >= 5.
                and 0 <= metrics.get('formal_anchor_slip_m', 1) <= .0005
                and metrics.get('native_task_success') is True):
            raise ValueError('native 30mm/5s/0.5mm proof required')
    if cursor != n:
        raise ValueError('every expert transition must belong to a complete episode')
    return payload


class ActorBCTrainer:
    def __init__(self, model, *, lr=3e-4, ppo_trainer=None):
        self.model, self.ppo_trainer = model, ppo_trainer
        if not hasattr(model, 'actor'):
            raise ValueError('actor-only model parameter group required')
        self.optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr)
        self.steps = 0

    def step(self, observations, actions, valid_mask, *, schema_only=False, dataset=None):
        if self.ppo_trainer is not None and not self.ppo_trainer.clean_boundary:
            raise RuntimeError('BC cannot modify a pending/nonempty PPO rollout')
        if not schema_only:
            if dataset is None:
                raise ValueError('qualified expert dataset required for production BC')
            validate_expert_dataset(dataset, expected_identity=dataset.get('identity'))
            # A validated pool cannot be used to bless unrelated caller tensors.
            if (not torch.equal(observations, dataset['observations'])
                    or not torch.equal(actions, dataset['actions']) or not torch.equal(valid_mask, dataset['valid_mask'])):
                raise ValueError('BC input must be the admitted expert dataset')
        device = next(self.model.parameters()).device
        predictions = self.model.deterministic(observations.to(device))
        loss = masked_group_normalized_loss(predictions, actions.to(device), valid_mask.to(device))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), .5, error_if_nonfinite=True)
        self.optimizer.step()
        self.steps += 1
        if self.ppo_trainer is not None:
            self.ppo_trainer.policy_version += 1
            self.ppo_trainer._new_rollout()
        return dict(loss=float(loss.detach()), steps=self.steps, schema_only=bool(schema_only),
            native_qualification=False if schema_only else 'supplied_native_proof_validated')
