"""Fit a qualified privileged Teacher actor on CPU before native R0 PPO."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from . import tabletop_privileged_dataset as expert
from .tabletop_privileged_checkpoint import save_checkpoint
from .tabletop_privileged_cli import load_config, training_identity
from .tabletop_privileged_ppo import PPOTrainer
from .tabletop_privileged_teacher import PrivilegedTeacher
from .tabletop_stereo_contract import ROOT, file_sha256, new_run, require_sha, write_json_new
from .tabletop_surface_contract import validate_contract


def _checked_bytes(path, expected_sha256, label):
    require_sha(expected_sha256, label + ' SHA256')
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(label + ' SHA256 differs from reviewed artifact')
    return content


def verify_supervised_rows(payload):
    """Reopen actual pre-action state/issued-action rows for every admitted episode."""
    sources = {}
    bindings = []
    for episode in payload['episodes']:
        source_sha = episode['source_sha256']
        key = str(Path(episode['source_path']).resolve()), source_sha
        if key not in sources:
            raw = _checked_bytes(*key, 'expert source ledger')
            sources[key] = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        source = sources[key]
        if not isinstance(source, dict):
            raise ValueError('expert source must be a synchronized ledger')
        observations, actions, valid, metadata = (source.get(name) for name in
            ('observations', 'actions', 'privileged_valid', 'sample_metadata'))
        if (not isinstance(observations, torch.Tensor) or observations.ndim != 2 or observations.shape[1] != 371
                or not isinstance(actions, torch.Tensor) or actions.shape != (len(observations), 24)
                or not isinstance(valid, torch.Tensor) or valid.shape != observations.shape or valid.dtype != torch.bool
                or not isinstance(metadata, list) or len(metadata) != len(observations)
                or not all(isinstance(row, dict) and isinstance(row.get('episode_uid'), str) for row in metadata)):
            raise ValueError('source ledger requires aligned 371D/24D rows and explicit per-field validity')
        episode_id = episode['episode_id']
        indices = [i for i, row in enumerate(metadata) if row['episode_uid'] == episode_id]
        ticks = [metadata[i].get('tick') for i in indices]
        start, stop = episode['start'], episode['end']
        if (not ticks or len(indices) != stop - start or any(type(tick) is not int or tick < 0 for tick in ticks)
                or ticks != list(range(ticks[0], ticks[0] + len(ticks)))):
            raise ValueError('source episode must contain the same contiguous pre-action rows')
        if not bool(valid[indices].all()):
            raise ValueError('unmasked 371D Teacher cannot warm start from unverified privileged fields')
        for name, value in (('observations', observations), ('actions', actions)):
            target = payload[name][start:stop]
            if value.dtype != target.dtype or not torch.equal(value[indices], target):
                raise ValueError('expert rows differ from measured/issued source ' + name)
        if not bool(payload['valid_mask'][start:stop].any(dim=1).all()):
            raise ValueError('every supervised pre-action row needs an eligible issued action')
        bindings.append(dict(episode_id=episode_id, source_sha256=source_sha, rows=len(indices),
                             first_policy_tick=ticks[0], last_policy_tick=ticks[-1]))
    return bindings


def read_expert(expert_dataset, expert_sha256, identity):
    raw = _checked_bytes(expert_dataset, expert_sha256, 'expert dataset')
    payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    expert.validate_expert_dataset(payload, expected_identity=identity)
    return payload, verify_supervised_rows(payload)


def warm_start_teacher(config, ppo, identity, expert_dataset, *, expert_sha256, output, steps, num_envs, seed):
    """Save an update-zero checkpoint with the BC actor and fresh PPO optimizer."""
    if type(steps) is not int or steps < 1 or type(num_envs) is not int or num_envs < 1:
        raise ValueError('positive BC step and eventual PPO environment budgets required')
    if config['reward_profile'] != 'teacher_r0_existing' or ppo.target_kl is None:
        raise ValueError('warm-start target must be the KL-guarded R0 trainer')
    payload, rows = read_expert(expert_dataset, expert_sha256, identity)
    torch.manual_seed(seed)
    trainer = PPOTrainer(PrivilegedTeacher(**config['model']), ppo, rollout_steps=config['rollout_steps'],
        num_envs=num_envs, device='cpu')
    bc = expert.ActorBCTrainer(trainer.model, lr=config['bc']['lr'])
    protected = {key: value.clone() for key, value in trainer.model.state_dict().items()
                 if not key.startswith('actor.')}
    observations = payload['observations'].float()
    actions = payload['actions'].float()
    valid = payload['valid_mask']
    output = new_run(output)
    with (output / 'bc_iterations.jsonl').open('x', encoding='utf-8') as log:
        for step in range(1, steps + 1):
            predictions = trainer.model.deterministic(observations)
            loss = expert.masked_group_normalized_loss(predictions, actions, valid)
            bc.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainer.model.actor.parameters(), .5, error_if_nonfinite=True)
            bc.optimizer.step()
            log.write(json.dumps(dict(step=step, loss=float(loss.detach()), grad_norm=float(norm)), allow_nan=False) + '\n')
            log.flush()
    critic_and_noise_unchanged = all(torch.equal(trainer.model.state_dict()[key], value)
                                     for key, value in protected.items())
    if not critic_and_noise_unchanged or trainer.optimizer.state:
        raise RuntimeError('actor-only warm start altered the critic, exploration noise or PPO Adam state')
    trainer.policy_version += steps
    trainer._new_rollout()
    checkpoint = output / 'teacher_bc_actor_warm_start.pt'
    saved = save_checkpoint(checkpoint, trainer, identity, history_reset='clear')
    with (output / 'expert_bc_optimizer.pt').open('xb') as stream:
        torch.save(bc.optimizer.state_dict(), stream)
    report = dict(version='dh116-privileged-teacher-bc-warm-start-v1', seed=seed, bc_steps=steps,
        expert_dataset_sha256=expert_sha256, expert_sources=rows, identity=identity, checkpoint_file=checkpoint.name,
        checkpoint_sha256=saved['sha256'], native_grasp_qualification=False, r0_ppo_updates=0,
        expected_ppo_environments=num_envs, expected_ppo_rollout_steps=config['rollout_steps'],
        ppo_optimizer_state_empty=True, critic_and_noise_unchanged=critic_and_noise_unchanged)
    write_json_new(output / 'warm_start.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--native-contract', required=True, type=Path)
    parser.add_argument('--native-contract-sha256', required=True)
    parser.add_argument('--native-task-config', type=Path,
        default=ROOT / 'configs/dh116_training/stereo_native_task_v1.json')
    parser.add_argument('--expert-dataset', required=True, type=Path)
    parser.add_argument('--expert-dataset-sha256', required=True)
    parser.add_argument('--num-envs', required=True, type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int, default=116)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args(argv)
    config, ppo = load_config(args.config)
    from .tabletop_stereo_native import native_task_identity
    contract_document = json.loads(_checked_bytes(args.native_contract, args.native_contract_sha256, 'native task contract'))
    contract, task = contract_document['inherited_contract'], contract_document['task_identity']
    validate_contract(contract)
    if (task != native_task_identity(json.loads(args.native_task_config.read_text(encoding='utf-8')))
            or contract_document.get('task_identity_sha256') != task['task_identity_sha256']
            or contract.get('num_envs') != args.num_envs
            or contract_document.get('inherited_launcher_sha256') != file_sha256(ROOT / 'scripts/train_dh116_tabletop_gpu_ppo.py')
            or contract.get('physics_hz') != 480 or contract.get('control_hz') != 60):
        raise ValueError('verified task, launcher, 480Hz/60Hz and eventual environment count must match the original R0 run')
    identity = training_identity(ppo, contract, task)
    if args.check_config:
        _, episodes = read_expert(args.expert_dataset, args.expert_dataset_sha256, identity)
        print(json.dumps(dict(status='reviewed_bc_inputs_verified', native_started=False,
                              native_grasp_qualification=False, episodes=episodes), indent=2))
        return 0
    if args.output is None or args.steps is None:
        parser.error('explicit --output and --steps required to fit the Teacher actor')
    result = warm_start_teacher(config, ppo, identity, args.expert_dataset,
        expert_sha256=args.expert_dataset_sha256, output=args.output, steps=args.steps,
        num_envs=args.num_envs, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0
