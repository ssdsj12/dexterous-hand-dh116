"""Versioned, state-aligned RGB-D supervision, independent of PPO storage.

This module validates schema and supplied provenance, not native physics.
Successful replay evidence must be produced by the native collector. A saved
historical action at tick t is a trajectory reference, not an arbitrary-state
expert oracle. Nothing here imports Kit, steps a simulator or evaluates PPO.
"""
import copy
import hashlib
import io
import json
import math
from pathlib import Path

import torch

from .tabletop_surface_contract import ACTION_CONTRACT

VERSION = 'dh116-vision-supervised-action-dataset-v1'
STYLES = ('pinch_thumb_index', 'tripod_thumb_index_middle')
POLICY_DIM = 3844
_SCOPE = 'schema_only_not_native_verification'


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdefABCDEF' for c in value))


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('finite JSON metadata required') from error


def _visual_contract(value):
    if (not isinstance(value, dict) or value.get('actor_observation_size') != POLICY_DIM
            or value.get('critic_observation_size') != 371
            or value.get('actor_uses_privileged_state') is not False
            or value.get('action_contract') != ACTION_CONTRACT):
        raise ValueError('visual contract requires pure 3844D actor, 371D critic and canonical 24D actions')
    for name in ('camera_config_sha256', 'primary_reset_sha256', 'secondary_reset_sha256'):
        if not _sha(value.get(name)):
            raise ValueError(f'visual contract requires valid {name}')
    _json(value)
    return value


def _canonical_visual_contract(value):
    canonical = copy.deepcopy(_visual_contract(value))
    # These change collection batch/storage only; camera geometry, sensor inputs,
    # reset identities, action semantics and physics remain exact.
    for key in ('num_envs', 'rollout_steps', 'camera_batch_size'):
        canonical.pop(key, None)
    for group in canonical.get('groups', []):
        group.pop('num_envs', None)
    for key in ('microbatch_size', 'inference_batch_size', 'logical_minibatches'):
        canonical.get('memory_contract', {}).pop(key, None)
    canonical.get('ppo', {}).pop('minibatches', None)
    if 'privileged_critic_contract' in canonical:
        from .tabletop_surface_imitation import _canonical_contract
        canonical['privileged_critic_contract'] = _canonical_contract(canonical['privileged_critic_contract'])
    return _json(canonical)


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')


def _matrix(value, name, rows, width):
    if (not isinstance(value, torch.Tensor) or value.shape != (rows, width)
            or value.layout != torch.strided):
        raise ValueError(f'{name} must have shape [{rows},{width}]')
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite floating tensor')
    if width == 24 and bool((value.abs() > 1).any()):
        raise ValueError(f'{name} outside canonical normalized range [-1,1]')


def validate_dataset(data, visual_contract=None, *, require_successful=True):
    """Validate a nonempty pool without mutating it or certifying a grasp.

    ``require_successful=False`` is for a separate failed/unfinished audit pool.
    It never changes failed episodes into successful training demonstrations.
    All qualification flags are provenance supplied by the collector; reading
    this file alone does not re-run or prove native contact/lift success.
    """
    if not isinstance(data, dict) or data.get('version') != VERSION:
        raise ValueError(f'expected {VERSION} dictionary')
    contract = _visual_contract(data.get('visual_contract'))
    if visual_contract is not None and _canonical_visual_contract(contract) != _canonical_visual_contract(visual_contract):
        raise ValueError('visual contract mismatch beyond collection batch fields')
    if not _sha(data.get('teacher_source_sha256')):
        raise ValueError('teacher_source_sha256 must bind the exact source bytes')
    if data.get('validation_scope') != _SCOPE:
        raise ValueError('dataset must explicitly distinguish schema validation from native verification')
    observations = data.get('observations')
    if not isinstance(observations, torch.Tensor) or observations.ndim != 2 or len(observations) == 0:
        raise ValueError('nonempty [S,3844] observations required')
    n = len(observations)
    _matrix(observations, 'observations', n, POLICY_DIM)
    for key in ('actions', 'reference_actions'):
        _matrix(data.get(key), key, n, 24)
    metadata, episodes = data.get('sample_metadata'), data.get('episodes')
    if not isinstance(metadata, list) or len(metadata) != n or not isinstance(episodes, dict) or not episodes:
        raise ValueError('sample_metadata and episodes must bind every sample')
    groups, styles, previous, used, sample_counts = [], [], {}, set(), {}
    for index, row in enumerate(metadata):
        if not isinstance(row, dict):
            raise ValueError('sample_metadata rows must be dictionaries')
        for name in ('env_id', 'episode_id', 'tick', 'frame_id', 'source_episode_id'):
            _integer(row.get(name), name)
        ep = episodes.get(str(row['episode_id']))
        if not isinstance(ep, dict):
            raise ValueError('sample episode identity missing from episodes')
        for name in ('env_id', 'grasp_style', 'source_trajectory_id', 'source_episode_id'):
            if row.get(name) != ep.get(name):
                raise ValueError(f'sample {name} disagrees with episode provenance')
        if row['grasp_style'] not in STYLES or not isinstance(row['source_trajectory_id'], str) or not row['source_trajectory_id'].strip():
            raise ValueError('known grasp style and nonempty source_trajectory_id required')
        when = row.get('simulation_time_s')
        if type(when) not in (int, float) or not math.isfinite(when) or when < 0:
            raise ValueError('finite nonnegative simulation_time_s required')
        for name in ('first_episode', 'success', 'recovered', 'native_physics'):
            if type(ep.get(name)) is not bool:
                raise ValueError(f'episode {name} must be an explicit boolean')
        if not isinstance(ep.get('proof_path'), str) or not ep['proof_path'].strip():
            raise ValueError('episode proof_path provenance is required')
        if require_successful and not (ep['first_episode'] and ep['success']
                                      and ep['native_physics'] and not ep['recovered']):
            raise ValueError('successful first native episodes without recovery required')
        key = (row['env_id'], row['episode_id'])
        old = previous.get(key)
        if row['tick'] != (0 if old is None else old[0]['tick'] + 1):
            raise ValueError('episode ticks must start at zero and remain ordered and unique')
        if old is not None:
            for name in ('frame_id', 'simulation_time_s'):
                if row[name] <= old[0][name]:
                    raise ValueError(f'{name} must be strictly ordered within each episode')
        if 'physics_step' in row:
            _integer(row['physics_step'], 'physics_step')
            if old is not None and 'physics_step' in old[0] and row['physics_step'] <= old[0]['physics_step']:
                raise ValueError('physics_step must be strictly ordered')
        if 'previous_action' in row:
            prior = row['previous_action']
            if not isinstance(prior, list) or len(prior) != 24:
                raise ValueError('previous_action must be the actual prior issued 24D command')
            prior = torch.tensor(prior, dtype=data['actions'].dtype, device=data['actions'].device)
            expected = torch.zeros_like(prior) if old is None else data['actions'][old[1]]
            if not torch.equal(prior, expected):
                raise ValueError('previous_action differs from the actual prior issued action')
        previous[key] = (row, index)
        used.add(str(row['episode_id']))
        identity = str(row['episode_id'])
        sample_counts[identity] = sample_counts.get(identity, 0) + 1
        groups.append(row['source_trajectory_id'])
        styles.append(row['grasp_style'])
    if set(episodes) != used:
        raise ValueError('episodes must exactly match the pool sample identities')
    if require_successful:
        for identity, ep in episodes.items():
            # Pure schema fixtures may omit terminal evidence, and never prove
            # native physics. Native collector episodes always provide it. In
            # particular a late success after source exhaustion must not admit
            # a prefix missing the executed zero-action tail into the BC pool.
            if 'terminal' not in ep:
                continue
            terminal = ep['terminal']
            if (not isinstance(terminal, dict) or terminal.get('success') is not True
                    or terminal.get('recovered') is not False
                    or type(terminal.get('failure')) is not int or terminal['failure'] != 0
                    or type(terminal.get('control_steps')) is not int
                    or terminal['control_steps'] < 1):
                raise ValueError('successful first native terminal evidence without recovery/failure required')
            count = sample_counts[identity]
            last = previous[(ep['env_id'], int(identity))][0]
            if terminal['control_steps'] != count or last['tick'] != terminal['control_steps'] - 1:
                raise ValueError('successful terminal control_steps must match every recorded episode sample')
    if data.get('source_trajectory_ids') != groups or data.get('grasp_styles') != styles:
        raise ValueError('source_trajectory_ids/grasp_styles columns disagree with sample metadata')
    _json(metadata)
    _json(episodes)
    return data


def build_dataset(observations, actions, reference_actions, sample_metadata, episodes, *,
                  visual_contract, teacher_source_sha256, require_successful=True):
    """Clone pre-action observations and actually issued labels into a new pool."""
    data = dict(version=VERSION, visual_contract=copy.deepcopy(visual_contract),
        teacher_source_sha256=teacher_source_sha256, validation_scope=_SCOPE,
        observations=observations.detach().cpu().clone(), actions=actions.detach().cpu().clone(),
        reference_actions=reference_actions.detach().cpu().clone(),
        sample_metadata=copy.deepcopy(sample_metadata), episodes=copy.deepcopy(episodes),
        source_trajectory_ids=[row.get('source_trajectory_id') for row in sample_metadata],
        grasp_styles=[row.get('grasp_style') for row in sample_metadata])
    return validate_dataset(data, require_successful=require_successful)


def split_by_source(data, heldout_source_trajectory_ids=()):
    """Split complete source trajectory groups; never random neighboring frames."""
    validate_dataset(data)
    groups = set(data['source_trajectory_ids'])
    heldout = set(heldout_source_trajectory_ids)
    if heldout - groups:
        raise ValueError('unknown heldout source trajectory IDs')
    train = [i for i, source in enumerate(data['source_trajectory_ids']) if source not in heldout]
    validation = [i for i, source in enumerate(data['source_trajectory_ids']) if source in heldout]
    if not train:
        raise ValueError('at least one complete source trajectory must remain in train')
    return dict(train_indices=torch.tensor(train, dtype=torch.long),
        heldout_indices=torch.tensor(validation, dtype=torch.long), source_trajectory_count=len(groups),
        heldout_independent_source_count=len(heldout), fit_only=not bool(validation),
        split_unit='source_trajectory;all_replay_copies_and_overlapping_histories_bound_together')


def save_dataset(data, path):
    """Save only a validated success pool, exclusively, without replacing files."""
    validate_dataset(data)
    target = Path(path).resolve()
    if data.get('teacher_source_path') and target == Path(data['teacher_source_path']).resolve():
        raise ValueError('source teacher data must remain immutable')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as output:
        torch.save(data, output)
    return dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                samples=len(data['observations']), validation_scope=_SCOPE)


def load_teacher_reference_dataset(path, critic_contract=None):
    """Load canonical 371D replay actions, not a learned Teacher or state oracle.

    The merged corpus contains replay copies. All episodes from one reset-family
    source dataset deliberately share one conservative source-trajectory ID.
    Current server success must be re-established during synchronized collection.
    """
    from .tabletop_surface_imitation import validate_dataset as validate_surface
    source = Path(path).resolve()
    raw = source.read_bytes()
    data = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    validate_surface(data, critic_contract)
    if (data['contract'].get('observation_size') != 371
            or data['contract'].get('surface_contract', {}).get('action') != ACTION_CONTRACT):
        raise ValueError('Teacher references require the audited 371D / canonical 24D contract')
    style_map = data.get('episode_grasp_styles')
    if not isinstance(style_map, dict):
        raise ValueError('Teacher source requires episode_grasp_styles')
    family = {record['label']: record for record in data.get('reset_family', [])}
    episodes, references = [], {}
    for identity in sorted(torch.unique(data['episode_ids']).tolist()):
        style = style_map.get(identity, style_map.get(str(identity)))
        if style not in STYLES or not _sha(family.get(style, {}).get('source_dataset_sha256')):
            raise ValueError('Teacher source requires known styles and reset-family source SHA')
        actions = data['actions'][data['episode_ids'] == identity].detach().float().clone()
        _matrix(actions, 'Teacher actions', len(actions), 24)
        record = dict(source_episode_id=identity, grasp_style=style,
            source_trajectory_id=family[style]['source_dataset_sha256'], actions=actions,
            grouping_basis='reset_family_source_dataset;replay_copies_not_independent')
        episodes.append(record)
        references.setdefault(style, record)
    if set(references) != set(STYLES):
        raise ValueError('Teacher references require both pinch and tripod source trajectories')
    return dict(source_path=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
        action_contract=copy.deepcopy(ACTION_CONTRACT), episodes=episodes, references=references,
        independent_source_trajectory_count=len({record['source_trajectory_id'] for record in episodes}),
        native_success_revalidated=False, state_query_oracle=False)
