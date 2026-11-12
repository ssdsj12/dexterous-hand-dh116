"""Assemble admitted Teacher BC rows from completed, independently proven episodes."""
import argparse
import io
import json
from pathlib import Path

import torch

from . import tabletop_stereo_promotion as promotion
from .tabletop_privileged_dataset import DATASET_VERSION, validate_expert_dataset
from .tabletop_privileged_warm_start import _checked_bytes, verify_supervised_rows
from .tabletop_stereo_contract import file_sha256, new_run, require_sha, write_json_new
from .tabletop_surface_contract import ACTION_CONTRACT, validate_contract


def _episode_source_rows(source, episode_uid, physical):
    if not isinstance(source, dict) or source.get('action_contract') != ACTION_CONTRACT:
        raise ValueError('canonical source ledger action contract required')
    validate_contract(source.get('privileged_observation_contract'))
    observations, actions, valid, metadata = (source.get(key) for key in (
        'observations', 'actions', 'privileged_valid', 'sample_metadata'))
    if (not isinstance(observations, torch.Tensor) or observations.ndim != 2 or observations.shape[1] != 371
            or not isinstance(actions, torch.Tensor) or actions.shape != (len(observations), 24)
            or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.shape != observations.shape
            or not isinstance(metadata, list) or len(metadata) != len(observations)
            or not all(isinstance(row, dict) for row in metadata)):
        raise ValueError('synchronized source requires 371D observations, validity, and issued 24D actions')
    indices = [index for index, row in enumerate(metadata) if row.get('episode_uid') == episode_uid]
    if not indices or not bool(valid[indices].all()):
        raise ValueError('all 371 source fields must have actual pre-action validity in every selected row')
    ticks = [metadata[index].get('tick') for index in indices]
    if any(type(tick) is not int or tick < 0 for tick in ticks) or ticks != list(range(ticks[0], ticks[0] + len(ticks))):
        raise ValueError('selected source requires contiguous policy ticks')
    cursor = 0
    for slot, index in enumerate(indices):
        record = metadata[index]
        count = record.get('physical_substeps')
        step = record.get('policy_clock_before', {}).get('world_step')
        terminal = slot == len(indices) - 1
        if (type(count) is not int or not 1 <= count <= 8 or (not terminal and count != 8)
                or type(step) is not int or record.get('terminal_in_interval') is not terminal):
            raise ValueError('source interval requires the original eight-substep clock and exact first terminal')
        interval = physical[cursor:cursor + count]
        if len(interval) != count or any(
                (row.get('run_uuid'), row.get('env_id'), row.get('episode_uid'), row.get('policy_tick'),
                 row.get('physics_step'), row.get('terminal'), row.get('actor_issued_action')) !=
                (record.get('run_uuid'), record.get('env_id'), episode_uid, record['tick'],
                 step + offset, terminal and offset == count, actions[index].tolist())
                for offset, row in enumerate(interval, 1)):
            raise ValueError('source action or original policy/physics clock differs from qualified native execution')
        cursor += count
    if cursor != len(physical):
        raise ValueError('a complete successful native episode must bind every live physical sample')
    return observations[indices], actions[indices]


def _artifact_path(value, base):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve(strict=True)


def assemble_experts(manifest, identity, output, *, base=Path('.')):
    if (not isinstance(manifest, dict) or manifest.get('version') != 'dh116-privileged-expert-assembly-v1'
            or not isinstance(manifest.get('episodes'), list) or not manifest['episodes']):
        raise ValueError('versioned reviewed expert assembly manifest with complete episodes required')
    episodes, receipts, observations, actions = [], [], [], []
    seen = set()
    source_cache = {}
    cursor = 0
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for entry in manifest['episodes']:
        uid = entry['episode_uid']
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError('unique native episode identities required')
        seen.add(uid)
        qualification = _artifact_path(entry['qualification_path'], base)
        proof = json.loads(_checked_bytes(qualification, entry['qualification_sha256'], 'native qualification'))
        verified = promotion.verify_native_episode(qualification, expected_identity=identity)
        if (verified.get('episode_uid') != uid or verified.get('native_evidence_verified') is not True
                or verified.get('qualified_success') is not True or verified.get('owner') not in ('mpc', 'teacher')):
            raise ValueError('exact independently qualified Teacher or MPC success required')
        raw, _ = promotion._artifact(proof['raw_artifact'], qualification.parent)
        physical = [row for row in promotion._jsonl(raw) if row.get('kind', 'sample') == 'sample' and row.get('live', True)]
        if len(physical) != verified['sample_count']:
            raise ValueError('qualified raw physical sample count differs')
        source_binding = dict(path=str(_artifact_path(entry['source_path'], base)), sha256=entry['source_sha256'])
        raw_source, source_path = promotion._process_output(proof, qualification.parent, source_binding)
        if file_sha256(qualification) != entry['qualification_sha256']:
            raise ValueError('native qualification changed during assembly')
        if source_binding['sha256'] not in source_cache:
            source_cache[source_binding['sha256']] = torch.load(io.BytesIO(raw_source), map_location='cpu', weights_only=True)
        source = source_cache[source_binding['sha256']]
        owner = 'legacy_teacher_mpc' if verified['owner'] == 'mpc' else 'teacher'
        if source.get('action_owner') != owner:
            raise ValueError('issued action owner differs from independently qualified native episode')
        state, issued = _episode_source_rows(source, uid, physical)
        metrics = dict(net_lift_m=physical[-1]['object_net_lift_m'], continuous_hold_s=physical[-1]['formal_hold_s'],
            formal_anchor_slip_m=physical[-1]['formal_drift_m'], native_task_success=True)
        require_sha(proof['producer']['config_artifact']['sha256'], 'verified physical config SHA256')
        episode = dict(episode_id=uid, start=cursor, end=cursor + len(state), complete=True,
            first_episode=True, recovered=False, restored=False, native_success=True,
            source_path=str(source_path), source_sha256=source_binding['sha256'],
            physical_config_sha256=proof['producer']['config_artifact']['sha256'])
        receipts.append(dict(**episode, identity=identity, evidence_kind='native_isaac_physx',
            schema_only=False, success_metrics=metrics))
        episodes.append(dict(**episode, qualification_evidence=str(qualification)))
        observations.append(state)
        actions.append(issued)
        cursor += len(state)
    issued = torch.cat(actions)
    payload = dict(version=DATASET_VERSION, schema_only=False, identity=identity,
        observations=torch.cat(observations), actions=issued, valid_mask=torch.ones_like(issued, dtype=torch.bool),
        episodes=episodes)
    # Inputs have passed the original verifier and the synchronized source/physics join.
    verify_supervised_rows(payload)
    output = new_run(output)
    for index, (episode, receipt) in enumerate(zip(episodes, receipts)):
        path = output / f'episode_{index:05d}.json'
        write_json_new(path, receipt)
        episode.update(proof_path=str(path), proof_sha256=file_sha256(path))
    validate_expert_dataset(payload, expected_identity=identity)
    target = output / 'privileged_native_expert.pt'
    with target.open('xb') as stream:
        torch.save(payload, stream)
    result = dict(version='dh116-privileged-expert-assembly-receipt-v1', episodes=len(episodes), rows=cursor,
        dataset_file=target.name, dataset_sha256=file_sha256(target), actor_warm_start_executed=False)
    write_json_new(output / 'assembly.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'model-identity', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--model-identity-sha256', required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(_checked_bytes(args.manifest, args.manifest_sha256, 'reviewed expert manifest'))
    identity = json.loads(_checked_bytes(args.model_identity, args.model_identity_sha256, 'target Teacher model identity'))
    print(json.dumps(assemble_experts(manifest, identity, args.output, base=args.manifest.resolve().parent), indent=2))
    return 0
