"""Explicit new-model entry points; imports Isaac only after native opt-in.

No native job runs on import or with --check-config. Local CPU tests of this
module do not qualify a Teacher, an expert dataset or a successful grasp.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .tabletop_stereo_contract import (
    ROOT, digest, file_sha256, new_run, require_opt_in, state_contract, write_json_new,
)
from .tabletop_privileged_teacher import PrivilegedTeacher
from .tabletop_privileged_ppo import PPOConfig, PPOTrainer
from .tabletop_privileged_checkpoint import build_identity, save_checkpoint, load_checkpoint
from .tabletop_privileged_reward import compose_native_reward


def load_config(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('version') != 'dh116-privileged-teacher-ppo-config-v1':
        raise ValueError('new privileged Teacher config version required')
    model = value.get('model', {})
    if (model.get('obs_dim') != 371 or model.get('action_dim') != 24
            or model.get('hidden') != [256, 256]):
        raise ValueError('audited 371D/24D Teacher topology required')
    config = PPOConfig(**value['ppo'])
    if value.get('reward_profile') != 'teacher_r0_existing':
        raise ValueError('native CLI defaults to audited R0; R1/R2 need separate qualified-source experiment')
    if type(value.get('rollout_steps')) is not int or value['rollout_steps'] < 1:
        raise ValueError('positive rollout_steps required')
    return value, config


def training_identity(config, bridge_contract):
    from .tabletop_surface_contract import semantic_sha256
    source = {name: file_sha256(Path(__file__).with_name(name + '.py')) for name in (
        'tabletop_privileged_teacher', 'tabletop_privileged_ppo', 'tabletop_privileged_reward',
        'tabletop_privileged_dataset', 'tabletop_privileged_checkpoint')}
    source['native_task_contract'] = semantic_sha256(bridge_contract)
    return build_identity(config, state_contract_sha256=digest(state_contract()), source_sha256=source)


def train_teacher(bridge, config, ppo_config, *, output, updates, checkpoint=None,
                  expert_dataset=None, bc_steps=0, checkpoint_interval=10):
    """Train only the independent model using the unchanged env.step reward."""
    output = Path(output)
    model = PrivilegedTeacher(**config['model'])
    trainer = PPOTrainer(model, ppo_config, rollout_steps=config['rollout_steps'],
        num_envs=bridge.n, device=bridge.device)
    identity = training_identity(ppo_config, bridge.contract)
    if checkpoint is not None:
        load_checkpoint(checkpoint, trainer, identity, history_reset='clear')
    if expert_dataset is not None:
        from .tabletop_privileged_dataset import ActorBCTrainer, validate_expert_dataset
        payload = torch.load(expert_dataset, map_location='cpu', weights_only=False)
        validate_expert_dataset(payload, expected_identity=identity)
        bc = ActorBCTrainer(model, ppo_trainer=trainer, lr=config['bc']['lr'])
        for _ in range(bc_steps):
            bc.step(payload['observations'], payload['actions'], payload['valid_mask'], dataset=payload)
        with (output / 'expert_bc_optimizer.pt').open('xb') as stream:
            torch.save(bc.optimizer.state_dict(), stream)
    elif bc_steps:
        raise ValueError('qualified expert dataset required for BC warm start')
    write_json_new(output / 'model_identity.json', identity)
    write_json_new(output / 'resolved_config.json', config)
    save_checkpoint(output / f'teacher_update_{trainer.counters["updates"]:06d}.pt',
        trainer, identity, history_reset='clear')
    obs = bridge.critic
    started = time.monotonic()
    with (output / 'iterations.jsonl').open('x', encoding='utf-8') as log:
        for update_index in range(updates):
            term_totals = {}
            for _ in range(config['rollout_steps']):
                sample = trainer.collect_step(obs, obs)
                next_obs, reward, done, extra = bridge.step(sample['actions'])
                reward, trace = compose_native_reward(reward, {
                    k: v.float() for k, v in extra['terms'].items()}, profile=config['reward_profile'])
                for name, term in trace['terms'].items():
                    term_totals[name] = term_totals.get(name, 0.) + float(term['total_contribution'])
                with torch.no_grad():
                    next_value = model.value(next_obs)
                trainer.record_step(reward, done, next_value, executed_actions=sample['actions'],
                    collection_token=sample['collection_token'])
                obs = next_obs
            metrics = trainer.update()
            update = trainer.counters['updates']
            row = dict(update=update, **metrics, reward_term_totals=term_totals,
                native_successes=int(bridge.env.successes.sum()),
                completed_episodes=int(bridge.env.episodes.sum()),
                elapsed_wall_s=time.monotonic() - started,
                qualification='pending_independent_full_episode_validation')
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            if update % checkpoint_interval == 0 or update_index == updates - 1:
                save_checkpoint(output / f'teacher_update_{update:06d}.pt', trainer, identity, history_reset='clear')
    return dict(status='independent_teacher_updates_completed', updates=trainer.counters['updates'],
        checkpoint_family='privileged_teacher_only', native_grasp_qualification=False)


def _policy_for_evaluation(bridge, config, ppo_config, checkpoint):
    # Evaluation may use a different collection batch; load the saved topology
    # on a disposable trainer with its declared storage geometry, then use only
    # deterministic model inference at the current env count.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = training_identity(ppo_config, bridge.contract)
    trainer = PPOTrainer(PrivilegedTeacher(**config['model']), ppo_config,
        rollout_steps=payload.get('rollout_steps', config['rollout_steps']),
        num_envs=payload.get('num_envs', bridge.n), device=bridge.device)
    load_checkpoint(checkpoint, trainer, identity, history_reset='clear', restore_rng=False)
    trainer.model.eval()
    return trainer.model, identity


@torch.no_grad()
def collect_or_evaluate(bridge, config, ppo_config, *, checkpoint, output, max_steps, collect=False):
    model, identity = _policy_for_evaluation(bridge, config, ppo_config, checkpoint)
    output = Path(output)
    initial = [bridge.identities.uid(i) for i in range(bridge.n)]
    completed, rows = set(), []
    for _ in range(max_steps):
        obs = bridge.critic.clone()
        before = bridge.pre_action_identities('0'*64)
        sample = model.sample(obs) if collect else None
        action = model.deterministic(obs)
        next_obs, reward, done, extra = bridge.step(action)
        if collect:
            for i in range(bridge.n):
                if i not in completed:
                    rows.append(dict(identity=before[i], privileged_state=obs[i].cpu(),
                        teacher_latent_mean=sample['latent_mean'][i].cpu(),
                        teacher_latent_std=sample['latent_std'][i].cpu(),
                        teacher_sampled_action=sample['actions'][i].cpu(),
                        teacher_mean_action=action[i].cpu(), actually_issued_action=action[i].cpu(),
                        applied_action_source='native_physics.jsonl;not_assumed_equal_to_actor',
                        reward=float(reward[i]), terminal=bool(done[i]),
                        native_success=bool(extra['success'][i]), failure_code=int(extra['failure'][i])))
        completed.update(done.nonzero().flatten().tolist())
        if len(completed) == bridge.n:
            break
    terminal = [bridge._terminal_rows.get(key) for key in initial]
    if collect:
        with (output / 'privileged_candidate_trajectories.pt').open('xb') as stream:
            torch.save(dict(version='dh116-privileged-candidate-collection-v1',
                identity=identity, samples=rows, terminal_rows=terminal,
                successful_expert_pool=False, qualification='requires_independent_native_evidence_review'), stream)
    report = dict(status='bounded_teacher_collection' if collect else 'bounded_teacher_evaluation',
        checkpoint_sha256=file_sha256(checkpoint), identity=identity,
        environments=bridge.n, first_episodes_complete=sum(t is not None for t in terminal),
        native_successes=sum(bool(t and t['native_success']) for t in terminal),
        steps=bridge.policy_tick, terminated_before_budget=len(completed) == bridge.n,
        native_force_qualification=None, qualified_for_student_dagger=False,
        terminal_evidence='native/native_physics.jsonl',
        limitation='clones are not independent initial conditions; this report does not qualify a Teacher')
    write_json_new(output / 'teacher_evaluation.json', report)
    return report


def main(mode='train', argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=116)
    parser.add_argument('--num-envs', type=int, default=2)
    parser.add_argument('--rollout-steps', type=int)
    parser.add_argument('--updates', type=int, default=1)
    parser.add_argument('--checkpoint', '--resume', dest='checkpoint', type=Path)
    parser.add_argument('--expert-dataset', type=Path)
    parser.add_argument('--bc-steps', type=int, default=0)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--max-control-steps', type=int, default=1200)
    parser.add_argument('--native-task-config', type=Path,
        default=ROOT / 'configs/dh116_training/stereo_native_task_v1.json')
    parser.add_argument('--experiment-profile', default='teacher_r0_existing')
    parser.add_argument('--enable-experiment', action='store_true')
    parser.add_argument('--execute-native', action='store_true')
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args(argv)
    config, ppo = load_config(args.config)
    if args.rollout_steps is not None:
        config['rollout_steps'] = args.rollout_steps
    if args.experiment_profile != config['reward_profile']:
        parser.error('experiment profile must match the explicit config')
    if any(type(v) is not int or v < 1 for v in (args.num_envs, config['rollout_steps'],
            args.updates, args.checkpoint_interval, args.max_control_steps)) or args.bc_steps < 0:
        parser.error('positive environment/rollout/update/checkpoint/step budgets required')
    if args.check_config:
        print(json.dumps(dict(mode=mode, validated=True, native_started=False,
            ppo=asdict(ppo), config=config), indent=2))
        return 0
    require_opt_in(args.enable_experiment)
    if not args.execute_native or args.output is None:
        parser.error('explicit --execute-native and new --output required; use --check-config for CPU inspection')
    if mode != 'train' and args.checkpoint is None:
        parser.error('collection/evaluation requires a new privileged Teacher checkpoint')
    torch.manual_seed(args.seed)
    run = new_run(args.output)
    write_json_new(run / 'launch.json', dict(mode=mode, seed=args.seed, num_envs=args.num_envs,
        input_config_sha256=file_sha256(args.config), native_task_config_sha256=file_sha256(args.native_task_config),
        new_model_only=True, old_models_modified=False))
    from .tabletop_stereo_native import launch_native

    def callback(bridge):
        if mode == 'train':
            return train_teacher(bridge, config, ppo, output=run, updates=args.updates,
                checkpoint=args.checkpoint, expert_dataset=args.expert_dataset, bc_steps=args.bc_steps,
                checkpoint_interval=args.checkpoint_interval)
        return collect_or_evaluate(bridge, config, ppo, checkpoint=args.checkpoint,
            output=run, max_steps=args.max_control_steps, collect=mode == 'collect')

    result = launch_native(callback, native_arguments=args.native_task_config,
        output=run / 'native', num_envs=args.num_envs, seed=args.seed,
        rollout_steps=config['rollout_steps'], enabled=True)
    write_json_new(run / 'completion.json', result)
    return 0
