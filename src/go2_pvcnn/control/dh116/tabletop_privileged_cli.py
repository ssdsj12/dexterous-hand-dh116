"""Explicit new-model entry points; imports Isaac only after native opt-in.

No native job runs on import or with --check-config. Local CPU tests of this
module do not qualify a Teacher, an expert dataset or a successful grasp.
"""
import argparse
from copy import deepcopy
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
from .tabletop_squashed_gaussian import squashed_entropy,latent_gaussian_kl


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


def training_identity(config, bridge_contract, task_identity=None):
    from .tabletop_surface_contract import semantic_sha256
    source = {name: file_sha256(Path(__file__).with_name(name + '.py')) for name in (
        'tabletop_privileged_teacher', 'tabletop_privileged_ppo', 'tabletop_privileged_reward',
        'tabletop_privileged_dataset', 'tabletop_privileged_checkpoint')}
    source['native_task_contract'] = semantic_sha256(bridge_contract)
    if task_identity is not None:source['native_task_identity']=digest(task_identity)
    return build_identity(config, state_contract_sha256=digest(state_contract()), source_sha256=source)


_ACTION_GROUPS=dict(left_fingers=(0,6),right_fingers=(6,12),left_translation=(12,15),
    right_translation=(15,18),left_rotation=(18,21),right_rotation=(21,24))


def _same_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(_same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(_same_tree(x,y) for x,y in zip(a,b))
    return a==b


def _checkpoint_readback(path,trainer,identity):
    """Read into a disposable PPO instance; never perturb collection RNG."""
    candidate=PPOTrainer(deepcopy(trainer.model),trainer.config,rollout_steps=trainer.rollout_steps,
        num_envs=trainer.num_envs,device=trainer.device)
    load_checkpoint(path,candidate,identity,history_reset='clear',restore_rng=False)
    report=dict(checkpoint_sha256=file_sha256(path),policy_equal=_same_tree(trainer.model.state_dict(),candidate.model.state_dict()),
        optimizer_equal=_same_tree(trainer.optimizer.state_dict(),candidate.optimizer.state_dict()),
        counters_equal=trainer.counters==candidate.counters,policy_version_equal=trainer.policy_version==candidate.policy_version)
    if not all(report[k] for k in ('policy_equal','optimizer_equal','counters_equal','policy_version_equal')):
        raise RuntimeError('Teacher checkpoint readback differs from saved policy/Adam/counters')
    return report


def _write_diagnostic_rollout(output, trainer, checkpoint):
    """Record the frozen training inputs and RNG before the first optimizer step."""
    returns, advantages = trainer.rollout.returns_and_advantages(trainer.config)
    artifact = dict(version='dh116-privileged-bounded-diagnostic-rollout-v1',
        config=asdict(trainer.config), policy_version=trainer.policy_version,
        initial_checkpoint_file=checkpoint.name, initial_checkpoint_sha256=file_sha256(checkpoint),
        layout='step_major_then_environment', rollout_steps=trainer.rollout_steps, num_envs=trainer.num_envs,
        batch=trainer.rollout.batch(), returns=returns, advantages=advantages,
        rng=dict(torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []))
    with (output / 'diagnostic_rollout.pt').open('xb') as stream:
        torch.save(artifact, stream)


def train_teacher(bridge, config, ppo_config, *, output, updates, checkpoint=None,
                  expert_dataset=None, bc_steps=0, checkpoint_interval=10,
                  record_diagnostic_rollout=False):
    """Train only the independent model using the unchanged env.step reward."""
    if record_diagnostic_rollout and (bridge.n, config['rollout_steps'], updates) != (2, 32, 1):
        raise ValueError('diagnostic rollout recording requires 2 environments, 32 steps and 1 update')
    output = Path(output)
    model = PrivilegedTeacher(**config['model'])
    trainer = PPOTrainer(model, ppo_config, rollout_steps=config['rollout_steps'],
        num_envs=bridge.n, device=bridge.device)
    identity = training_identity(ppo_config, bridge.contract,getattr(bridge,'task_identity',None))
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
    if getattr(bridge,'task_identity',None) is not None:
        write_json_new(output / 'task_identity.json',bridge.task_identity)
    write_json_new(output / 'resolved_config.json', config)
    initial_checkpoint = output / f'teacher_update_{trainer.counters["updates"]:06d}.pt'
    save_checkpoint(initial_checkpoint,
        trainer, identity, history_reset='clear')
    native_identity = callable(getattr(bridge, 'bind_teacher_checkpoint', None))
    if native_identity:
        bridge.bind_teacher_checkpoint(initial_checkpoint, policy_version=trainer.policy_version)
    obs = bridge.critic
    started = time.monotonic()
    with (output / 'iterations.jsonl').open('x', encoding='utf-8') as log:
        for update_index in range(updates):
            term_totals = {}
            stats={k:torch.zeros(24,dtype=torch.float64) for k in ('sum','square','noise_square','std_sum','saturation')}
            count=0;closure=0.;joint_failures=0;failure_samples=0
            terminal_count = 0
            group_entropy={name:0. for name in _ACTION_GROUPS}
            policy_samples=[]
            for _ in range(config['rollout_steps']):
                if native_identity:
                    bridge.teacher_pre_action_identities()
                    obs = bridge.critic.clone()
                sample = trainer.collect_step(obs, obs)
                action_cpu=sample['actions'].detach().cpu().double()
                noise=action_cpu-sample['latent_mean'].detach().cpu().double().tanh()
                stats['sum']+=action_cpu.sum(0);stats['square']+=action_cpu.square().sum(0)
                stats['noise_square']+=noise.square().sum(0)
                stats['std_sum']+=sample['latent_std'].detach().cpu().double().expand_as(action_cpu).sum(0)
                stats['saturation']+=(action_cpu.abs()>=.999).double().sum(0);count+=len(action_cpu)
                policy_samples.append((obs.detach().cpu().clone(),sample['latent_mean'].detach().cpu().clone(),
                    sample['latent_std'].detach().cpu().clone()))
                # Existing sampled epsilon gives a diagnostic current-policy MC
                # entropy estimate without drawing from or changing PPO RNG.
                for name,(lo,hi) in _ACTION_GROUPS.items():
                    mu=sample['latent_mean'][:,lo:hi];std_=sample['latent_std'][:,lo:hi]
                    epsilon=(sample['latent_actions'][:,lo:hi]-mu)/std_
                    group_entropy[name]+=float(squashed_entropy(mu,std_,epsilon).sum())
                if native_identity:
                    with torch.no_grad():
                        mean_action = model.deterministic(obs)
                    bridge.set_teacher_action_diagnostics(sampled_action=sample['actions'],
                        mean_action=mean_action, issued_action=sample['actions'])
                next_obs, reward, done, extra = bridge.step(sample['actions'])
                original_reward=reward.clone()
                reward, trace = compose_native_reward(reward, {
                    k: v.float() for k, v in extra['terms'].items()}, profile=config['reward_profile'])
                for name, term in trace['terms'].items():
                    term_totals[name] = term_totals.get(name, 0.) + float(term['total_contribution'])
                closure=max(closure,float((reward-original_reward).abs().max()))
                if 'failure' in extra:
                    joint_failures+=int((extra['failure']==3).sum());failure_samples+=len(extra['failure'])
                with torch.no_grad():
                    next_value = model.value(next_obs)
                    # The native task ends the episode on success/failure,
                    # including its task timeout. next_obs can already be reset.
                    next_value = next_value.masked_fill(done, 0.)
                terminal_count += int(done.sum())
                trainer.record_step(reward, done, next_value, executed_actions=sample['actions'],
                    bootstrap_mask=~done, collection_token=sample['collection_token'])
                obs = next_obs
            if record_diagnostic_rollout:
                _write_diagnostic_rollout(output, trainer, initial_checkpoint)
            metrics = trainer.update()
            group_kl={name:0. for name in _ACTION_GROUPS}
            with torch.no_grad():
                for observation,old_mu,old_std in policy_samples:
                    new_mu=model.latent_mean(observation.to(bridge.device))
                    new_std=model.log_std.exp().expand_as(new_mu)
                    old_mu,old_std=old_mu.to(new_mu),old_std.to(new_mu)
                    for name,(lo,hi) in _ACTION_GROUPS.items():
                        group_kl[name]+=float(latent_gaussian_kl(old_mu[:,lo:hi],old_std[:,lo:hi],
                            new_mu[:,lo:hi],new_std[:,lo:hi]).sum())
            update = trainer.counters['updates']
            mean=stats['sum']/count;std=(stats['square']/count-mean.square()).clamp_min(0).sqrt()
            noise_rms=(stats['noise_square']/count).sqrt();latent_std=stats['std_sum']/count
            action_diagnostics=dict(execution_mode='sampled_on_policy',issued_equals_sampled=True,
                sampled_action_mean24=mean.tolist(),sampled_action_std24=std.tolist(),
                sampled_minus_mean_rms24=noise_rms.tolist(),latent_std24=latent_std.tolist(),
                saturation_fraction24=(stats['saturation']/count).tolist(),
                normalized_action_clipping_fraction=0.,
                normalized_action_clipping_source='issued sample equals bounded tanh sample; any out-of-range command is rejected',
                joint_safety_projection=None,
                joint_safety_projection_source='pre_projection_joint_target_not_exposed; native raw accepted/SDK targets and pressure mask are separate',
                joint_limit_failure_fraction=joint_failures/failure_samples if failure_samples else None,
                final_old_to_new_kl=sum(group_kl.values())/count,
                entropy_definition='collection-policy MC from already sampled epsilon; no additional RNG draw',
                kl_definition='exact Normal(old||final_updated_policy) on collected states; distinct from optimizer mean KL',
                actual_execution_evidence='native/native_physics.jsonl' if native_identity else None,
                groups={name:dict(indices=list(range(lo,hi)),sampled_mean=mean[lo:hi].tolist(),
                    sampled_std=std[lo:hi].tolist(),noise_rms=noise_rms[lo:hi].tolist(),latent_std=latent_std[lo:hi].tolist(),
                    sampled_tanh_entropy=group_entropy[name]/count,final_old_to_new_kl=group_kl[name]/count,
                    normalized_saturation_fraction=(stats['saturation'][lo:hi]/count).tolist(),
                    native_joint_limit_hand_attribution=None)
                    for name,(lo,hi) in _ACTION_GROUPS.items()})
            row = dict(update=update, **metrics, reward_term_totals=term_totals,
                optimizer_update_accepted=metrics['minibatch_updates'] > 0,
                gae_diagnostics=dict(terminal_count=terminal_count, terminal_bootstrap_count=0,
                    terminal_timeout_semantics='task_timeout_is_terminal', reset_observation_bootstrap=False),
                native_successes=int(bridge.env.successes.sum()),
                completed_episodes=int(bridge.env.episodes.sum()),
                elapsed_wall_s=time.monotonic() - started,
                action_diagnostics=action_diagnostics,reward_diagnostics=dict(profile=config['reward_profile'],
                    r0_closure_max_abs=closure,reward_owner='native_env_once'),
                qualification='pending_independent_full_episode_validation')
            if not row['optimizer_update_accepted']:
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                return dict(status='teacher_optimizer_update_rejected', updates=update,
                    checkpoint_family='privileged_teacher_only', native_grasp_qualification=False)
            if native_identity or update % checkpoint_interval == 0 or update_index == updates - 1:
                current_checkpoint = output / f'teacher_update_{update:06d}.pt'
                save_checkpoint(current_checkpoint, trainer, identity, history_reset='clear')
                row['checkpoint_roundtrip']=_checkpoint_readback(current_checkpoint,trainer,identity)
                if native_identity and update_index < updates - 1:
                    bridge.bind_teacher_checkpoint(current_checkpoint, policy_version=trainer.policy_version)
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
    return dict(status='independent_teacher_updates_completed', updates=trainer.counters['updates'],
        checkpoint_family='privileged_teacher_only', native_grasp_qualification=False)


def _policy_for_evaluation(bridge, config, ppo_config, checkpoint):
    # Evaluation may use a different collection batch; load the saved topology
    # on a disposable trainer with its declared storage geometry, then use only
    # deterministic model inference at the current env count.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = training_identity(ppo_config, bridge.contract,getattr(bridge,'task_identity',None))
    trainer = PPOTrainer(PrivilegedTeacher(**config['model']), ppo_config,
        rollout_steps=payload.get('rollout_steps', config['rollout_steps']),
        num_envs=payload.get('num_envs', bridge.n), device=bridge.device)
    load_checkpoint(checkpoint, trainer, identity, history_reset='clear', restore_rng=False)
    trainer.model.eval()
    return trainer.model, identity


@torch.no_grad()
def collect_or_evaluate(bridge, config, ppo_config, *, checkpoint, output, max_steps, collect=False):
    checkpoint_sha256 = file_sha256(checkpoint)
    model, identity = _policy_for_evaluation(bridge, config, ppo_config, checkpoint)
    output = Path(output)
    bridge.bind_teacher_checkpoint(checkpoint)
    if bridge.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError('Teacher checkpoint bytes changed between policy load and native identity binding')
    initial = [bridge.identities.uid(i) for i in range(bridge.n)]
    completed, rows = set(), []
    for _ in range(max_steps):
        before = bridge.teacher_pre_action_identities()
        obs = bridge.critic.clone()
        sample = model.sample(obs) if collect else None
        action = model.deterministic(obs)
        bridge.set_teacher_action_diagnostics(sampled_action=sample['actions'] if sample is not None else None,
            mean_action=action, issued_action=action)
        next_obs, reward, done, extra = bridge.step(action)
        if collect:
            for i in range(bridge.n):
                if i not in completed:
                    rows.append(dict(identity=before[i], privileged_state=obs[i].cpu(),
                        teacher_latent_mean=sample['latent_mean'][i].cpu(),
                        teacher_latent_std=sample['latent_std'][i].cpu(),
                        teacher_sampled_action=sample['actions'][i].cpu(),
                        teacher_mean_action=action[i].cpu(), actually_issued_action=action[i].cpu(),
                        applied_action=None,
                        applied_action_source='native/native_physics.jsonl:controller_and_SDK_targets;normalized_applied_action_unavailable',
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
        checkpoint_sha256=checkpoint_sha256, identity=identity,
        calibration_sha256=bridge.camera.calibration.sha256,
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
    parser.add_argument('--camera-config', type=Path,
        default=ROOT / 'configs/dh116_training/stereo_camera_640x480_v2.json')
    parser.add_argument('--expert-dataset', type=Path)
    parser.add_argument('--bc-steps', type=int, default=0)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--record-diagnostic-rollout', action='store_true',
        help='save private pre-optimizer tensors/RNG for an explicit 2-env, 32-step, 1-update diagnosis')
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
    if args.record_diagnostic_rollout and (mode != 'train' or
            (args.num_envs, config['rollout_steps'], args.updates) != (2, 32, 1)):
        parser.error('diagnostic rollout recording requires train with 2 environments, 32 steps and 1 update')
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
        from .tabletop_stereo_camera import StereoCamera
        bridge.bind_camera(StereoCamera.from_native(bridge.world.sim, bridge.device_world.origins,
            args.camera_config, device=bridge.device))
        if mode == 'train':
            return train_teacher(bridge, config, ppo, output=run, updates=args.updates,
                checkpoint=args.checkpoint, expert_dataset=args.expert_dataset, bc_steps=args.bc_steps,
                checkpoint_interval=args.checkpoint_interval, record_diagnostic_rollout=args.record_diagnostic_rollout)
        return collect_or_evaluate(bridge, config, ppo, checkpoint=args.checkpoint,
            output=run, max_steps=args.max_control_steps, collect=mode == 'collect')

    result = launch_native(callback, native_arguments=args.native_task_config,
        output=run / 'native', num_envs=args.num_envs, seed=args.seed,
        rollout_steps=config['rollout_steps'], enabled=True)
    write_json_new(run / 'completion.json', result)
    return 0
