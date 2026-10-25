"""Serial SB3 lifecycle. Admission evidence belongs to the supplied environment.

The helpers can be tested without Isaac; tests using a synthetic Gym environment
prove software wiring only, never physics qualification.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STAGES = ('approach', 'close', 'lift', 'full')
PHYSICAL_FAILURES = frozenset({
    'table_penetration', 'excess_force', 'object_dropped', 'joint_limit',
    'mimic_error', 'nonfinite_state', 'nonfinite_joint_state', 'invalid_action',
})


def report_exit_code(report):
    """Unsafe/incomplete execution is an error; a safe unsolved task is data."""
    failed = (report.get('error') or report.get('interrupted')
        or report.get('status') in ('error', 'interrupted') or report.get('training_skipped')
        or report.get('training_stopped') == 'physical_safety_failure'
        or report.get('admission', {}).get('allowed') is False
        or report.get('evaluation', {}).get('physically_safe') is False
        or any(s.get('evaluation', {}).get('physically_safe') is False
               for s in report.get('stages', []))
        or report.get('approach', {}).get('safety_failure'))
    return 2 if failed else 0


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def admission_report(stage, probes):
    """Task failure is learnable; missing/unstable physical evidence is not."""
    if stage not in STAGES:
        raise ValueError('unknown curriculum stage')
    required = 'lift' if stage == 'full' else stage
    reasons = []
    if len(probes) < 3:
        reasons.append('at_least_three_probes_required')
    for i, row in enumerate(probes):
        if row.get('steps', 0) <= 0 or row.get('finite') is not True:
            reasons.append(f'probe_{i}_invalid_measurements')
        if row.get('reset_repeatable') is not True:
            reasons.append(f'probe_{i}_reset_not_repeatable')
        if 'safety_failure' not in row or row['safety_failure'] is not None:
            reasons.append(f'probe_{i}_physical_safety')
        if row.get('failure') in PHYSICAL_FAILURES:
            reasons.append(f'probe_{i}_physical_failure')
    # The zero baseline may fail the task. At least two exploration probes
    # must nevertheless reach this stage through real preceding physics.
    if sum(required in row.get('reached_phases', []) for row in probes) < 2:
        reasons.append('stage_unreachable_in_two_probes')
    return dict(allowed=not reasons, stage=stage, reasons=reasons, probes=probes)


def run_episode(env, model=None, *, seed=42, random_actions=False,
                check_reset=False, max_steps=1000, probe_stage=None):
    obs, _ = env.reset(seed=seed)
    finite = bool(np.isfinite(obs).all())
    reset_repeatable = None
    if check_reset:
        again, _ = env.reset(seed=seed)
        reset_repeatable = bool(np.allclose(obs, again, atol=1e-4, rtol=0))
        obs = again
    rng = np.random.default_rng(seed)
    total = 0.
    info = {}
    steps = 0
    for _ in range(max_steps):
        if not finite:
            info = dict(failure='nonfinite_state', safety_failure='nonfinite_state')
            break
        phase = getattr(getattr(env, 'task', None), 'phase', info.get('phase', 'approach'))
        in_stage = (probe_stage in (None, 'full') or phase == probe_stage
                    or (probe_stage == 'close' and phase == 'hold'))
        if random_actions and in_stage:
            action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        elif model is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total += float(reward)
        finite = finite and bool(np.isfinite(obs).all() and np.isfinite(reward))
        if terminated or truncated:
            break
    else:
        info = dict(info, failure='runner_step_limit', stage_success=False, is_success=False)
    safety = info.get('safety_failure')
    if info.get('failure') in PHYSICAL_FAILURES:
        safety = info['failure']
    if not finite:
        safety = 'nonfinite_state'
    return dict(steps=steps, finite=finite, reset_repeatable=reset_repeatable,
                safety_failure=safety, failure=info.get('failure'),
                reached_phases=info.get('reached_phases', []),
                stage_success=bool(info.get('stage_success', False)),
                is_success=bool(info.get('is_success', False)),
                return_=total if np.isfinite(total) else None, seed=seed,
                random_action_stage=probe_stage if random_actions else None,
                controller='random' if random_actions else ('policy' if model else 'zero'))


def evaluate(env, model, stage, *, episodes=5, seed=1000):
    if episodes < 2:
        raise ValueError('at least two evaluation episodes required')
    env.set_stage(stage)
    env.set_mode('evaluation')
    rows = [run_episode(env, model, seed=seed+i) for i in range(episodes)]
    return dict(stage=stage, episodes=rows,
                stage_success_rate=float(np.mean([r['stage_success'] for r in rows])),
                grasp_success_rate=float(np.mean([r['is_success'] for r in rows])),
                physically_safe=all(r['finite'] and r['safety_failure'] is None for r in rows))


def save_checkpoint(model, path, contract, *, stage, completed_stages=()):
    path = Path(path).with_suffix('.zip')
    temporary = path.with_name(path.stem + '.tmp.zip')
    model.save(temporary)
    temporary.replace(path)
    write_json(path.with_suffix('.json'), dict(
        contract=contract, stage=stage, completed_stages=list(completed_stages),
        timesteps=int(model.num_timesteps), updates=int(model._n_updates)))


def canonical_contract(value):
    """Compare in-memory nested tuples/lists in their checkpoint JSON form."""
    return json.loads(json.dumps(value, allow_nan=False))


def load_checkpoint(path, env, contract, *, device='cpu', allow_contact_reward_change=False,
                    allow_env_expansion=False):
    from stable_baselines3 import PPO
    if allow_env_expansion and allow_contact_reward_change:
        raise ValueError('environment expansion and reward change are mutually exclusive')
    path = Path(path).with_suffix('.zip')
    metadata = json.loads(path.with_suffix('.json').read_text())
    contract = canonical_contract(contract)
    previous = canonical_contract(metadata.get('contract', {}))
    if allow_env_expansion:
        from zipfile import ZipFile
        source_count, target_count = previous.get('num_envs'), contract.get('num_envs')
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
               for n in (source_count, target_count)):
            raise ValueError('environment expansion requires positive integer num_envs')
        if target_count <= source_count:
            raise ValueError('environment expansion requires strictly increasing num_envs')
        if env is None or env.num_envs != target_count:
            raise ValueError('target environment count differs from expansion contract')
        # PPO.load replaces n_envs with the supplied environment count. Inspect
        # the original archive first so inconsistent sidecars cannot hide that.
        with ZipFile(path) as archive:
            saved_count = json.loads(archive.read('data')).get('n_envs')
        if type(saved_count) is not int or saved_count != source_count:
            raise ValueError('checkpoint metadata environment count differs from saved model')
        if dict(previous, num_envs=target_count) != contract:
            raise ValueError('checkpoint contract differs beyond permitted environment expansion')
    elif previous != contract:
        from .tabletop_contact_reward import REWARD_VERSION
        compatible = (previous.get('reward') == 'object_support_progress_v3'
            and contract.get('reward') == REWARD_VERSION
            and dict(previous, reward=REWARD_VERSION) == contract)
        if not allow_contact_reward_change or not compatible:
            raise ValueError('checkpoint contract differs from current observation/action/physics contract')
        metadata['reward_change'] = dict(previous=previous['reward'], current=REWARD_VERSION)
    if env is not None and not allow_env_expansion:
        from zipfile import ZipFile
        with ZipFile(path) as archive:
            saved_count = json.loads(archive.read('data')).get('n_envs')
        if saved_count != getattr(env, 'num_envs', 1):
            raise ValueError('checkpoint environment count differs; explicit expansion is required')
    model = PPO.load(path, env=env, device=device)
    if metadata.get('timesteps') != model.num_timesteps or metadata.get('updates') != model._n_updates:
        raise ValueError('checkpoint metadata does not match model')
    if allow_env_expansion:
        metadata['env_expansion'] = dict(source_num_envs=source_count, target_num_envs=target_count,
            source_timesteps=int(model.num_timesteps), source_updates=int(model._n_updates))
    return model, metadata


def run_training(env, output, contract, *, stages, timesteps=1024,
                 eval_episodes=5, probe_episodes=3, checkpoint_every=512,
                 resume=None, initialize_from=None, ppo_kwargs=None, seed=42):
    """Train each requested curriculum stage; promotion requires >=80% success.

    Episodes use the environment's declared reset phase. Direct-close seeds
    qualify close-to-lift only. A physical safety failure during learning stops
    learning and saves a diagnostic checkpoint.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    if resume and initialize_from:
        raise ValueError('resume and initialize_from are mutually exclusive')
    if (not stages or any(s not in STAGES for s in stages) or timesteps <= 0
            or eval_episodes < 2 or probe_episodes < 3 or checkpoint_every <= 0):
        raise ValueError('positive budget/checkpoint interval, valid stages, >=2 evals and >=3 probes required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(passed=False, grasp_qualified=False, ppo_updates=0, ppo_timesteps=0,
                  stages=[], status='running', pipeline_scope='close_to_lift'
                  if contract.get('reset_phase') == 'close' else 'approach_to_lift')
    model = None
    completed = []
    current_stage = stages[0]
    parameters = dict(learning_rate=3e-4, n_steps=128, batch_size=64, n_epochs=5,
        gamma=.995, gae_lambda=.95, clip_range=.2, ent_coef=.005, vf_coef=.5,
        max_grad_norm=.5, target_kl=.02, seed=seed, device='cpu',
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))
    parameters.update(ppo_kwargs or {})
    policy = 'MlpPolicy'
    if contract.get('action') == '12D_direct_joint_target_v1':
        from .tabletop_policy import BoundedMeanPolicy
        policy = BoundedMeanPolicy
        parameters['policy_kwargs'] = dict(parameters['policy_kwargs'], log_std_init=-1.2)
    result['ppo_parameters'] = parameters

    class SafetyCheckpoint(BaseCallback):
        def __init__(self):
            super().__init__()
            self.failure = None

        def _on_step(self):
            for info in self.locals.get('infos', []):
                failure = info.get('safety_failure')
                if failure or info.get('failure') in PHYSICAL_FAILURES:
                    self.failure = failure or info['failure']
            if self.n_calls % checkpoint_every == 0:
                save_checkpoint(self.model, output/f'checkpoint_{self.num_timesteps}_steps.zip',
                    contract, stage=current_stage, completed_stages=completed)
                write_json(output/'training_progress.json', dict(
                    stage=current_stage, timesteps=self.num_timesteps,
                    updates=self.model._n_updates, safety_stop=self.failure))
            return self.failure is None

    try:
        if initialize_from:
            metadata = json.loads(Path(initialize_from).with_suffix('.json').read_text())
            # Explicit weights-only migration across a pregrasp handoff change.
            # Observation, action, source assets and physics must still match.
            def transfer_contract(value):
                return {k: v for k, v in value.items() if k != 'approach_pregrasp_radius_m'}
            if transfer_contract(metadata['contract']) != transfer_contract(contract):
                raise ValueError('initialization contract differs beyond pregrasp handoff')
            source, _ = load_checkpoint(initialize_from, env, metadata['contract'])
            model = PPO('MlpPolicy', env, verbose=1, **parameters)
            model.policy.load_state_dict(source.policy.state_dict())
            del source
            result['initialized_from'] = str(initialize_from)
            result['initialized_from_timesteps'] = metadata['timesteps']
            result['initialization_mode'] = 'policy_weights_only_fresh_optimizer_no_qualifications'
        if resume:
            model, metadata = load_checkpoint(resume, env, contract)
            current_stage = metadata['stage']
            completed = list(metadata['completed_stages'])
            result['resumed_from'] = str(resume)
            # PPO.load restores these settings; CLI defaults do not replace
            # them. Report the actual loaded optimizer/rollout configuration.
            for key in ('learning_rate', 'n_steps', 'batch_size', 'n_epochs',
                        'gamma', 'gae_lambda', 'ent_coef', 'vf_coef',
                        'max_grad_norm', 'target_kl', 'seed', 'policy_kwargs'):
                result['ppo_parameters'][key] = getattr(model, key)
            result['ppo_parameters']['clip_range'] = float(model.clip_range(1.))
            result['resume_mode'] = 'restart_episode_keep_model_and_optimizer'
            if len(stages) > 1:
                if metadata['stage'] not in stages:
                    raise ValueError('saved stage is not in requested curriculum')
                start = stages.index(metadata['stage'])
                if metadata['stage'] in completed and start + 1 < len(stages):
                    start += 1
                stages = stages[start:]
            elif stages[0] != metadata['stage']:
                # Explicit forward overrides may only select the next stage
                # after a qualified stage, never skip an unlearned prefix.
                order = ('approach', 'close', 'lift')
                permitted = (metadata['stage'] in order and stages[0] in order
                    and metadata['stage'] in completed
                    and order.index(stages[0]) == order.index(metadata['stage']) + 1)
                if not permitted:
                    raise ValueError('explicit resume stage is not saved stage or qualified successor')
        for current_stage in stages:
            env.set_stage(current_stage)
            env.set_mode('admission')
            # Baseline plus full-range bounded random residuals in the actual
            # current-stage episode. Earlier stages are never skipped.
            probes = [run_episode(env, None, seed=seed, check_reset=True)]
            probes += [run_episode(env, model, seed=seed+i, random_actions=True,
                                   probe_stage=current_stage, check_reset=True)
                       for i in range(1, probe_episodes)]
            admission = admission_report(current_stage, probes)
            record = dict(stage=current_stage, admission=admission)
            result['stages'].append(record)
            write_json(output/'training_report.json', result)
            if not admission['allowed']:
                result['training_skipped'] = 'exploration_not_qualified'
                break
            if model is None:
                model = PPO(policy, env, verbose=1, **parameters)
            model.set_logger(configure(str(output/current_stage), ['csv', 'stdout']))
            env.set_mode('training')
            # Updating a previously qualified stage invalidates that stage's
            # promotion until this newly updated policy is assessed again.
            completed = [s for s in completed if STAGES.index(s) < STAGES.index(current_stage)]
            # Evaluation/probes reset the same single simulator, invalidating
            # SB3's cached rollout observation before a continued learn call.
            model._last_obs = None
            callback = SafetyCheckpoint()
            model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=False)
            record['safety_stop'] = callback.failure
            result.update(ppo_updates=int(model._n_updates), ppo_timesteps=int(model.num_timesteps))
            save_checkpoint(model, output/'ppo_checkpoint.zip', contract,
                            stage=current_stage, completed_stages=completed)
            if callback.failure:
                result['training_stopped'] = 'physical_safety_failure'
                break
            assessment = evaluate(env, model, current_stage, episodes=eval_episodes, seed=seed+1000)
            record['evaluation'] = assessment
            promoted = assessment['physically_safe'] and assessment['stage_success_rate'] >= .8
            if promoted and current_stage not in completed:
                completed.append(current_stage)
            record['qualified_for_promotion'] = promoted
            if current_stage in ('lift', 'full') or contract.get('grasp_mode') == 'object_stability':
                # Close/hold precede lift; approach is also exercised only
                # when declared by pipeline_scope. Stage entry is not success.
                result['passed'] = assessment['physically_safe'] and assessment['grasp_success_rate'] >= .8
                result['grasp_qualified'] = result['passed']
            if not promoted:
                result['training_stopped'] = ('curriculum_stage_not_mastered'
                    if assessment['physically_safe'] else 'physical_safety_failure')
                break
        result['status'] = ('not_admitted' if result.get('training_skipped') else
                            'stopped' if result.get('training_stopped') else 'completed_budget')
        return result, model
    except BaseException as exc:
        result['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
        result['exception'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if model is not None:
            result.update(ppo_updates=int(model._n_updates), ppo_timesteps=int(model.num_timesteps))
            save_checkpoint(model, output/'ppo_checkpoint.zip', contract,
                            stage=current_stage, completed_stages=completed)
        result['completed_stages'] = completed
        write_json(output/'training_report.json', result)
