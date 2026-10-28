"""Single-process vector PPO lifecycle with physical admission per environment.

The supplied SB3 VecEnv owns one shared simulator. This module never launches
workers or a second simulator. Unsafe states stop globally unless an explicit,
validated training-only isolated-reset contract accepts the failed episode.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .tabletop_gpu_recovery import validated_recovery

from .tabletop_training import (
    PHYSICAL_FAILURES, canonical_contract, load_checkpoint, save_checkpoint, write_json,
)


def _safety(infos, observations, rewards=None):
    if not np.isfinite(observations).all() or (rewards is not None and not np.isfinite(rewards).all()):
        return 'nonfinite_state'
    for info in infos:
        if info.get('safety_failure'):
            return info['safety_failure']
        if info.get('failure') in PHYSICAL_FAILURES:
            return info['failure']
        terminal = info.get('terminal_observation')
        if terminal is not None and not np.isfinite(terminal).all():
            return 'nonfinite_state'
    return None


def _validated_recovered_mimic(info, done, contract, observation_shape):
    """Accept only the explicit adapter's terminal/reset validation protocol.

    This never changes the original failed transition or assessment semantics.
    The adapter independently checks all joint/geometry guards, including those
    a pre-existing task failure may have prevented the phase machine checking.
    """
    if (contract.get('training_failure_handling') != 'validated_mimic_reset_v1'
            or not bool(done) or info.get('recovered_mimic') is not True
            or info.get('global_stop') is not False
            or info.get('global_stop_failure') is not None
            or info.get('TimeLimit.truncated', False) is not False
            or info.get('failure') != 'mimic_error' or info.get('safety_failure') != 'mimic_error'
            or info.get('is_success', False) or info.get('stage_success', False)):
        return False
    validation = info.get('reset_validation')
    physical = info.get('physical')
    if not isinstance(validation, dict) or not isinstance(physical, dict):
        return False
    if (validation.get('version') != 'validated_mimic_reset_v1'
            or any(validation.get(field) is not True for field in ('verified', 'passed',
                'other_guards_safe', 'reset_guards_safe', 'post_reset_guards_safe'))):
        return False
    checked = validation.get('checked_physics_substeps')
    terminal_step = validation.get('terminal_physics_substep')
    remaining = validation.get('remaining_physics_substeps')
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
           for value in (checked, terminal_step, remaining)):
        return False
    try:
        decimation = int(contract['physics_hz'] / contract['control_hz'])
        values = [float(physical[key]) for key in
                  ('object_bottom_m', 'slip_m', 'max_force_n', 'max_mimic_error_rad')]
        error = float(validation['terminal_mimic_abs_rad'])
        terminal = np.asarray(info['terminal_observation'])
        expected_size = contract.get('observation_size', observation_shape[0])
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False
    if (tuple(observation_shape) != (expected_size,) or terminal.shape != (expected_size,)
            or terminal.dtype.kind not in 'fiu'):
        return False
    expected_scope = 'immediate_reset_only' if remaining == 0 else 'reset_and_remaining_substeps'
    return bool(1 <= terminal_step <= decimation and remaining == decimation-terminal_step
        and checked == remaining and validation.get('validation_scope') == expected_scope
        and np.isfinite(values+[error]).all() and np.isfinite(terminal).all() and .02 < error < .03
        and np.isclose(error, values[3], rtol=0., atol=1e-9)
        and values[0] >= .29 and 0. <= values[2] <= 30.)


def collect_vector_episodes(env, model=None, *, episodes=None, seed=42,
                            random_actions=False, check_reset=False,
                            max_steps=1000, progress=None):
    """Collect fixed per-env quotas, inspecting safety even after a quota is met.

    Auto-resets of faster environments do not inflate evaluation success rates.
    Terminal observations are inspected separately from reset observations.
    """
    count = int(env.num_envs)
    episodes = max(count, count if episodes is None else int(episodes))
    quotas = np.array([(episodes + i) // count for i in range(count)])
    env.seed(seed)
    obs = env.reset()
    failure = _safety(getattr(env, 'reset_infos', []), obs)
    repeatable = [None] * count
    if check_reset and failure is None:
        env.seed(seed)
        again = env.reset()
        failure = _safety(getattr(env, 'reset_infos', []), again)
        repeatable = [bool(np.allclose(obs[i], again[i], atol=1e-4, rtol=0)) for i in range(count)]
        obs = again
    completed = np.zeros(count, dtype=int)
    lengths = np.zeros(count, dtype=int)
    returns = np.zeros(count)
    phases = [set() for _ in range(count)]
    rows = []
    vector_steps = 0
    rng = np.random.default_rng(seed)
    while failure is None and np.any(completed < quotas):
        if random_actions:
            actions = rng.uniform(-1, 1, (count, *env.action_space.shape)).astype(np.float32)
        elif model is None:
            actions = np.zeros((count, *env.action_space.shape), dtype=np.float32)
        else:
            actions, _ = model.predict(obs, deterministic=True)
        if not np.isfinite(actions).all():
            failure = 'invalid_action'
            break
        obs, rewards, dones, infos = env.step(actions)
        vector_steps += 1
        failure = _safety(infos, obs, rewards)
        for i, info in enumerate(infos):
            if completed[i] >= quotas[i]:
                continue
            lengths[i] += 1
            returns[i] += float(rewards[i])
            phases[i].update(info.get('reached_phases', []))
            if info.get('phase'):
                phases[i].add(info['phase'])
            if dones[i]:
                rows.append(dict(env_index=i, steps=int(lengths[i]),
                    return_=float(returns[i]) if np.isfinite(returns[i]) else None,
                    finite=bool(np.isfinite(obs[i]).all() and np.isfinite(returns[i])
                        and np.isfinite(info.get('terminal_observation', obs[i])).all()),
                    reset_repeatable=repeatable[i], safety_failure=info.get('safety_failure'),
                    failure=info.get('failure'), reached_phases=sorted(phases[i]),
                    is_success=bool(info.get('is_success', False)),
                    seed=seed+i if completed[i] == 0 else None,
                    reset_seed=seed+i, episode_index=int(completed[i]),
                    controller='random' if random_actions else ('policy' if model is not None else 'zero')))
                completed[i] += 1
                lengths[i] = 0
                returns[i] = 0
                phases[i].clear()
        if progress is not None:
            progress(dict(vector_steps=vector_steps, completed=completed.tolist(), quotas=quotas.tolist(),
                          safety_failure=failure, phases=[info.get('phase') for info in infos]))
        if failure is None and np.any((completed < quotas) & (lengths >= max_steps)):
            failure = 'runner_step_limit'
    safe = failure is None and len(rows) == episodes
    return dict(episodes=rows, vector_steps=vector_steps, transitions=vector_steps*count,
                physically_safe=safe, safety_failure=failure,
                reset_repeatable=repeatable, quotas=quotas.tolist(),
                grasp_success_rate=float(np.mean([r['is_success'] for r in rows])) if rows else 0.)


def evaluate_vector(env, model, *, episodes=5, seed=1000, progress=None):
    env.env_method('set_mode', 'evaluation')
    return collect_vector_episodes(env, model, episodes=episodes, seed=seed, progress=progress)


def run_vector_training(env, output, contract, *, timesteps=1024, eval_episodes=5,
                        checkpoint_every=512, seed=42, initialize_from=None,
                        resume=None, ppo_kwargs=None, probe_only=False, direct_training=False,
                        episodes_per_env=None, resume_episode_progress=None, allow_contact_reward_change=False,
                        allow_physics_refinement=False, allow_mimic_refinement=False,
                        allow_protected_mimic_init=False, allow_env_expansion=False):
    """Run admission, real PPO updates and deterministic vector evaluation.

    Budgets and checkpoint intervals count transitions across all environments.
    Direct training skips full probe episodes after a safe initial reset;
    runtime physical guards and the final evaluation still apply.
    episodes_per_env replaces the transition budget with a minimum completed
    training-episode count for every environment, checked after full PPO updates.
    Resume requires the entire contract to match. Explicit environment expansion
    permits only num_envs growth and retains model, optimizer and original rows'
    history; added environments start with zero history and all episodes restart.
    Explicit weights-only imports
    allow vector layout changes, but preserve reset, observation, action, reward
    and physics semantics and never copy optimizer history or qualifications.
    The explicit refinement flag instead permits only 240->480 Hz physics
    at unchanged 60 Hz control and vector layout, retaining a 240 Hz reset seed.
    Alternatively, mimic refinement permits only frequency 200->400 at fixed
    480 Hz physics, 60 Hz control and damping ratio 1; every other field matches.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from .tabletop_policy import BoundedMeanPolicy

    contract = canonical_contract(contract)
    count = int(env.num_envs)
    if count < 1 or timesteps <= 0 or checkpoint_every <= 0 or eval_episodes < 2:
        raise ValueError('positive env count, budget/checkpoint interval and >=2 evaluation episodes required')
    if resume and initialize_from:
        raise ValueError('resume and initialize_from are mutually exclusive')
    if direct_training and probe_only:
        raise ValueError('direct_training and probe_only are mutually exclusive')
    if episodes_per_env is not None and (isinstance(episodes_per_env, bool)
            or not isinstance(episodes_per_env, int) or episodes_per_env < 1):
        raise ValueError('episodes_per_env must be a positive integer')
    if episodes_per_env is not None and probe_only:
        raise ValueError('episodes_per_env cannot be combined with probe_only')
    if resume_episode_progress is not None and not resume:
        raise ValueError('resume_episode_progress requires resume')
    if allow_contact_reward_change and not resume:
        raise ValueError('contact reward change requires resume')
    if allow_env_expansion and not resume:
        raise ValueError('environment expansion requires resume')
    if allow_env_expansion and allow_contact_reward_change:
        raise ValueError('environment expansion and reward change are mutually exclusive')
    if sum(bool(flag) for flag in (allow_physics_refinement, allow_mimic_refinement,
                                  allow_protected_mimic_init)) > 1:
        raise ValueError('initialization migrations are mutually exclusive')
    if allow_physics_refinement and (not initialize_from or resume):
        raise ValueError('physics refinement requires initialize_from and cannot resume')
    if allow_mimic_refinement and (not initialize_from or resume):
        raise ValueError('mimic refinement requires initialize_from and cannot resume')
    if allow_protected_mimic_init and (not initialize_from or resume):
        raise ValueError('protected mimic initialization requires initialize_from and cannot resume')
    if contract.get('num_envs', count) != count:
        raise ValueError('contract num_envs differs from VecEnv')
    output = Path(output)
    if allow_env_expansion and output.resolve() == Path(resume).resolve().parent:
        raise ValueError('environment expansion requires a separate output directory from source run')
    output.mkdir(parents=True, exist_ok=True)
    parameters = dict(learning_rate=3e-4, n_steps=128, batch_size=min(256, 128*count),
        n_epochs=5, gamma=.995, gae_lambda=.95, clip_range=.2, ent_coef=.005,
        vf_coef=.5, max_grad_norm=.5, target_kl=.02, seed=seed, device='cpu',
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), log_std_init=-1.2))
    parameters.update(ppo_kwargs or {})
    result = dict(status='admission', passed=False, grasp_qualified=False,
                  num_envs=count, pipeline_scope='close_to_lift', ppo_updates=0,
                  ppo_timesteps=0, ppo_parameters=parameters, recovered_mimic_episodes=0, recovered_physical_episodes=0)
    model = None
    callback = None
    live_writer = None
    started = time.monotonic()
    training_started = None
    training_finished = None
    start_timesteps = 0
    last_monitor = 0.
    per_env = [dict(env_index=i, episodes=0, successes=0, return_sum=0.,
                    last_return=None, last_failure=None, phase_counts={}, reached_phase_counts={},
                    recovered_mimic_episodes=0, recovered_physical_episodes=0, failure_counts={})
               for i in range(count)]
    episode_returns = np.zeros(count)
    episode_lengths = np.zeros(count, dtype=int)

    def episode_budget():
        completed = [row['episodes'] for row in per_env]
        return dict(episodes_per_env_target=episodes_per_env,
            completed_training_episodes=sum(completed),
            min_completed_episodes_per_env=min(completed),
            max_completed_episodes_per_env=max(completed),
            episode_target_reached=bool(episodes_per_env is not None
                and min(completed) >= episodes_per_env))

    def save_vector_checkpoint(path, *, completed_stages=()):
        save_checkpoint(model, path, contract, stage='full', completed_stages=completed_stages)
        metadata_path = Path(path).with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        metadata['training_progress'] = dict(version=1, num_envs=count, per_env=per_env)
        if 'env_expansion' in result:
            metadata['env_expansion'] = result['env_expansion']
        write_json(metadata_path, metadata)

    def monitor(extra=None, force=False):
        nonlocal last_monitor
        now = time.monotonic()
        if not force and now-last_monitor < 2.:
            return
        last_monitor = now
        result.update(episode_budget())
        steps = int(model.num_timesteps) if model is not None else 0
        # Only model.learn contributes to throughput; admission and evaluation
        # run the same simulator but do not collect training transitions.
        elapsed = ((training_finished if training_finished is not None else now)-training_started
                   if training_started is not None else 0.)
        additional = steps-start_timesteps
        value = dict(status=result['status'], num_envs=count, timesteps=steps,
            updates=int(model._n_updates) if model is not None else 0,
            steps_per_env=[sum(row['phase_counts'].values()) for row in per_env],
            transitions_this_run=additional,
            transitions_per_second=additional/max(elapsed, 1e-9),
            training_elapsed_seconds=elapsed, elapsed_seconds=now-started,
            per_env=per_env, safety_stop=result.get('safety_failure'),
            recovered_mimic_episodes=result['recovered_mimic_episodes'],
            recovered_physical_episodes=result['recovered_physical_episodes'],
            training_stopped=result.get('training_stopped'),
            passed=result['passed'], evaluation=result.get('evaluation'))
        if 'env_expansion' in result:
            value['env_expansion'] = result['env_expansion']
        value.update(episode_budget())
        value.update({key:result[key] for key in ('telemetry_status','logging_errors') if key in result})
        value.update(extra or {})
        write_json(output/'monitor.json', value)
        write_json(output/'training_progress.json', value)

    class SafetyCheckpoint(BaseCallback):
        def __init__(self):
            super().__init__()
            self.failure = None
            self.last_checkpoint = start_timesteps
            self.last_progress = start_timesteps

        def _on_step(self):
            infos = self.locals.get('infos', [])
            rewards = self.locals.get('rewards')
            recovered = [_validated_recovered_mimic(info, done, contract, env.observation_space.shape)
                         or validated_recovery(info, done, contract, env.observation_space.shape)
                         for info, done in zip(infos, self.locals.get('dones', []))]
            # Only training may continue after a validated isolated reset. Keep
            # original infos/rewards/dones intact for SB3 and diagnostic counts.
            safety_infos = [dict(info, failure=None, safety_failure=None) if accepted else info
                            for info, accepted in zip(infos, recovered)]
            self.failure = _safety(safety_infos, self.locals.get('new_obs'), rewards)
            for i, info in enumerate(infos):
                if 'force_feedback' in info:
                    from .tabletop_tensorboard import finite_diagnostics
                    per_env[i]['force_feedback'] = finite_diagnostics(info['force_feedback'])
                physical = info.get('physical')
                if physical:
                    # Non-finite diagnostics must remain JSON-serializable so
                    # writing a safety-stop report cannot mask the actual fault.
                    per_env[i]['last_physical'] = {
                        key: float(value) if value is not None and np.isfinite(value) else None
                        for key, value in physical.items()}
                    for source, target in (('object_bottom_m', 'max_object_bottom_m'),
                            ('max_mimic_error_rad', 'max_mimic_error_rad'),
                            ('slip_m', 'max_slip_m'), ('max_force_n', 'max_force_n')):
                        value = per_env[i]['last_physical'].get(source)
                        if value is not None:
                            per_env[i][target] = max(value, per_env[i].get(target, value))
                episode_lengths[i] += 1
                if rewards is not None and np.isfinite(rewards[i]):
                    episode_returns[i] += float(rewards[i])
                phase = info.get('phase', 'unknown')
                counts = per_env[i]['phase_counts']
                counts[phase] = counts.get(phase, 0)+1
                if self.locals.get('dones')[i]:
                    row = per_env[i]
                    row['episodes'] += 1
                    row['successes'] += int(bool(info.get('is_success', False)))
                    row['return_sum'] += float(episode_returns[i])
                    row['last_return'] = float(episode_returns[i])
                    row['last_episode_steps'] = int(episode_lengths[i])
                    row['last_failure'] = info.get('failure')
                    if info.get('failure'):
                        failures = row['failure_counts']
                        failures[info['failure']] = failures.get(info['failure'], 0)+1
                    if recovered[i]:
                        row['recovered_physical_episodes'] = row.get('recovered_physical_episodes',0)+1
                        result['recovered_physical_episodes'] += 1
                        if info.get('failure') == 'mimic_error':
                            row['recovered_mimic_episodes'] += 1
                            result['recovered_mimic_episodes'] += 1
                    reached = Counter(row['reached_phase_counts'])
                    reached.update(set(info.get('reached_phases', [])))
                    row['reached_phase_counts'] = dict(reached)
                    episode_returns[i] = 0.
                    episode_lengths[i] = 0
            if self.failure:
                result['safety_failure'] = self.failure
                result['training_stopped'] = 'physical_safety_failure'
            if self.num_timesteps-self.last_checkpoint >= checkpoint_every:
                save_vector_checkpoint(output/f'checkpoint_{self.num_timesteps}_steps.zip')
                self.last_checkpoint = self.num_timesteps
            if self.failure or self.num_timesteps-self.last_progress >= min(256, checkpoint_every):
                monitor(force=True)
                self.last_progress = self.num_timesteps
            else:
                monitor()
            live_writer.write(self.num_timesteps, infos, rewards, per_env, self.failure,
                              force=bool(self.failure))
            return self.failure is None

        def _on_training_end(self):
            live_writer.write(self.num_timesteps, self.locals.get('infos', []),
                              self.locals.get('rewards'), per_env, self.failure, force=True)
            # The last PPO update otherwise stays buffered until the next rollout.
            if self.logger.name_to_value:
                self.logger.dump(step=self.num_timesteps)

    try:
        env.env_method('set_stage', 'full')
        if resume:
            options = dict(allow_contact_reward_change=True) if allow_contact_reward_change else {}
            if allow_env_expansion:
                options['allow_env_expansion'] = True
            model, metadata = load_checkpoint(resume, env, contract, device=parameters['device'], **options)
            if 'reward_change' in metadata:
                result['reward_change'] = metadata['reward_change']
            if 'env_expansion' in metadata:
                result['env_expansion'] = metadata['env_expansion']
            if metadata.get('stage') != 'full':
                raise ValueError('vector checkpoint must use full close-to-lift stage')
            result['resumed_from'] = str(resume)
            result['resume_mode'] = 'restart_episodes_keep_model_and_optimizer'
            progress = metadata.get('training_progress')
            if resume_episode_progress is not None:
                progress = json.loads(Path(resume_episode_progress).read_text())
                if (progress.get('timesteps') != int(model.num_timesteps)
                        or progress.get('updates') != int(model._n_updates)):
                    raise ValueError('episode progress does not match checkpoint counters')
                result['resumed_episode_progress_from'] = str(resume_episode_progress)
            if progress is None and (episodes_per_env is not None or allow_env_expansion):
                raise ValueError('resumed episode goal requires saved episode progress')
            if progress is not None:
                source_count = (metadata['env_expansion']['source_num_envs']
                                if allow_env_expansion else count)
                rows = progress.get('per_env')
                if (type(progress.get('num_envs')) is not int
                        or progress['num_envs'] != source_count or not isinstance(rows, list)
                        or len(rows) != source_count):
                    raise ValueError('episode progress environment count differs from checkpoint')
                for i, row in enumerate(rows):
                    if not isinstance(row, dict) or row.get('env_index') != i:
                        raise ValueError('episode progress environment indices are invalid')
                    for key in ('episodes', 'successes', 'recovered_mimic_episodes'):
                        v = row.get(key)
                        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                            raise ValueError('episode progress counters must be nonnegative integers')
                    recovered_physical = row.get('recovered_physical_episodes',0)
                    if type(recovered_physical) is not int or not 0 <= recovered_physical <= row['episodes']:
                        raise ValueError('recovered physical episode count invalid')
                    if row['successes'] > row['episodes']:
                        raise ValueError('episode progress successes exceed completed episodes')
                    if (not isinstance(row.get('return_sum'), (int, float))
                            or not np.isfinite(row['return_sum'])):
                        raise ValueError('episode progress return sum must be finite')
                    for key in ('phase_counts', 'reached_phase_counts', 'failure_counts'):
                        if not isinstance(row.get(key), dict):
                            raise ValueError('episode progress statistics must be mappings')
                        if any(type(v) is not int or v < 0 for v in row[key].values()):
                            raise ValueError('episode progress statistics must be nonnegative integers')
                if (allow_env_expansion and sum(sum(r['phase_counts'].values()) for r in rows)
                        != int(model.num_timesteps)):
                    raise ValueError('episode progress historical steps do not match checkpoint counters')
                per_env[:source_count] = rows
                result['recovered_mimic_episodes'] = sum(r['recovered_mimic_episodes'] for r in rows)
                result['recovered_physical_episodes'] = sum(r.get('recovered_physical_episodes',0) for r in rows)
                result['resumed_completed_episodes_per_env'] = [r['episodes'] for r in per_env]
            for key in ('learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
                        'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl',
                        'seed', 'policy_kwargs'):
                result['ppo_parameters'][key] = getattr(model, key)
            result['ppo_parameters']['clip_range'] = float(model.clip_range(1.))
        if initialize_from:
            source_path = Path(initialize_from).with_suffix('.zip')
            metadata = json.loads(source_path.with_suffix('.json').read_text())
            refining = allow_physics_refinement or allow_mimic_refinement or allow_protected_mimic_init
            vector_keys = (set() if refining else
                           {'num_envs', 'env_spacing_m', 'vector_backend'})
            comparable = lambda value: {k: v for k, v in value.items() if k not in vector_keys}
            source_contract = metadata['contract']
            source_comparable = comparable(source_contract)
            if allow_physics_refinement:
                if not (source_contract.get('physics_hz') == 240
                        and contract.get('physics_hz') == 480
                        and source_contract.get('control_hz') == contract.get('control_hz') == 60
                        and source_contract.get('seed_physics_hz', 240) == 240
                        and contract.get('seed_physics_hz') == 240):
                    raise ValueError('physics refinement requires 240->480 Hz, 60 Hz control and 240 Hz seed')
                # Normalize only this declared refinement on a copy. Preserve
                # the original source contract and checkpoint metadata intact.
                source_comparable.update(physics_hz=480, seed_physics_hz=240)
            elif allow_mimic_refinement:
                if not (source_contract.get('physics_hz') == contract.get('physics_hz') == 480
                        and source_contract.get('control_hz') == contract.get('control_hz') == 60
                        and source_contract.get('mimic_natural_frequency_s_inv') == 200.
                        and contract.get('mimic_natural_frequency_s_inv') == 400.
                        and source_contract.get('mimic_damping_ratio') == contract.get('mimic_damping_ratio') == 1.):
                    raise ValueError('mimic refinement requires 200->400 frequency, 480 Hz physics, '
                                     '60 Hz control and damping ratio 1')
                source_comparable['mimic_natural_frequency_s_inv'] = 400.
            elif allow_protected_mimic_init:
                if not (source_contract.get('physics_hz') == 480
                        and source_contract.get('control_hz') == 60
                        and source_contract.get('mimic_natural_frequency_s_inv') == 200.
                        and source_contract.get('mimic_damping_ratio') == 1.
                        and 'training_failure_handling' not in source_contract
                        and contract.get('training_failure_handling') == 'validated_mimic_reset_v1'):
                    raise ValueError('protected mimic initialization requires unchanged 480 Hz/60 Hz, '
                                     'frequency 200, damping 1 and newly enabled validated reset handling')
                source_comparable['training_failure_handling'] = 'validated_mimic_reset_v1'
            if source_comparable != comparable(contract):
                raise ValueError('initialization contract differs beyond permitted refinement'
                    if refining else 'initialization contract differs beyond vector layout')
            source, _ = load_checkpoint(source_path, None, source_contract)
            if type(source.policy) is not BoundedMeanPolicy:
                raise ValueError('initialization policy must have bounded Gaussian mean')
            model = PPO(BoundedMeanPolicy, env, verbose=1, **parameters)
            model.policy.load_state_dict(source.policy.state_dict())
            result['initialized_from'] = str(source_path)
            result['initialized_from_sha256'] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            result['source_ppo_timesteps'] = int(source.num_timesteps)
            result['source_updates'] = int(source._n_updates)
            result['source_contract'] = source_contract
            result['initialization_mode'] = 'policy_weights_only_fresh_optimizer_no_qualifications'
            if allow_physics_refinement:
                result['physics_refinement'] = dict(source_physics_hz=240, physics_hz=480,
                    control_hz=60, seed_physics_hz=240)
            elif allow_mimic_refinement:
                result['mimic_refinement'] = dict(physics_hz=480, control_hz=60,
                    source_mimic_natural_frequency_s_inv=200., mimic_natural_frequency_s_inv=400.,
                    mimic_damping_ratio=1.)
            elif allow_protected_mimic_init:
                result['protected_mimic_initialization'] = 'validated_mimic_reset_v1'
            del source
        start_timesteps = int(model.num_timesteps) if model is not None else 0
        monitor(force=True)
        batches = []
        reasons = []
        if direct_training:
            # Startup contact/drive checks belong to the simulator entry point.
            # No complete diagnostic episodes here: failed grasps may teach PPO.
            env.env_method('set_mode', 'training')
            env.seed(seed)
            reset_obs = env.reset()
            failure = _safety(getattr(env, 'reset_infos', []), reset_obs)
            if failure:
                reasons.append('initial_reset_physical_safety_failure')
                result['safety_failure'] = failure
        else:
            env.env_method('set_mode', 'admission')
            for index in range(3):
                batch = collect_vector_episodes(env, episodes=count, seed=seed+index*count,
                    random_actions=index > 0, check_reset=True,
                    progress=lambda update: monitor(dict(admission_batch=len(batches), admission_progress=update)))
                batches.append(batch)
                if not batch['physically_safe']:
                    reasons.append(f'batch_{index}_physical_safety_or_incomplete')
                    result['safety_failure'] = batch['safety_failure']
                    break
                for i, repeated in enumerate(batch['reset_repeatable']):
                    if repeated is not True:
                        reasons.append(f'batch_{index}_env_{i}_reset_not_repeatable')
            for i in range(count):
                rows = [row for batch in batches for row in batch['episodes'] if row['env_index'] == i]
                if len(rows) != 3:
                    reasons.append(f'env_{i}_three_probes_required')
                # With a measured-contact gate, reaching lift is a learned outcome.
                # Require safe repeatable close exploration, not pre-training grasp.
                required_phase = ('close' if contract.get('lift_gate') in ('measured_hand_load_stability_v1', 'measured_hand_load_stability_v2',
                                                                  'measured_hand_load_stability_v3')
                                  else 'lift')
                if sum(required_phase in row['reached_phases'] for row in rows) < 2:
                    reasons.append(f'env_{i}_stage_unreachable_in_two_probes')
        result['admission'] = dict(allowed=not reasons, reasons=reasons, batches=batches,
            mode='direct_training' if direct_training else 'three_full_probes',
            probes_skipped=bool(direct_training))
        write_json(output/'training_report.json', result)
        if reasons:
            result.update(status='not_admitted', training_skipped='exploration_not_qualified')
            return result, model
        if probe_only:
            result['status'] = 'probe_complete'
            return result, None
        if model is None:
            model = PPO(BoundedMeanPolicy, env, verbose=1, **parameters)
            if contract.get('grip_control') == 'measured_reaction_force_v1':
                from .tabletop_force_feedback import initialize_force_policy
                initialize_force_policy(model.policy)
                result['initial_force_action_mean'] = -.95
        model.set_logger(configure(str(output/'ppo'), ['csv', 'stdout', 'tensorboard']))
        from .tabletop_tensorboard import LiveTrainingWriter
        live_writer = LiveTrainingWriter(output/'ppo'/'live')
        env.env_method('set_mode', 'training')
        model._last_obs = None
        callback = SafetyCheckpoint()
        result['status'] = 'training'
        training_started = time.monotonic()
        monitor(force=True)
        try:
            if episodes_per_env is None:
                model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=False)
            else:
                # Keep training terminal transitions through the end of their
                # rollout, including the final update. Fast envs may exceed quota.
                while not episode_budget()['episode_target_reached']:
                    model.learn(total_timesteps=model.n_steps*count, callback=callback,
                                reset_num_timesteps=False)
                    result.update(episode_budget())
                    monitor(force=True)
                    if callback.failure:
                        break
        finally:
            training_finished = time.monotonic()
        result.update(ppo_updates=int(model._n_updates), ppo_timesteps=int(model.num_timesteps))
        result.update(episode_budget())
        save_vector_checkpoint(output/'ppo_checkpoint.zip')
        if callback.failure:
            result['status'] = 'stopped'
            return result, model
        result['status'] = 'evaluation'
        monitor(force=True)
        assessment = evaluate_vector(env, model, episodes=eval_episodes, seed=seed+10000,
            progress=lambda update: monitor(dict(evaluation_progress=update)))
        result['evaluation'] = assessment
        result['passed'] = assessment['physically_safe'] and assessment['grasp_success_rate'] >= .8
        result['grasp_qualified'] = result['passed']
        if not assessment['physically_safe']:
            result['training_stopped'] = 'physical_safety_failure'
            result['safety_failure'] = assessment['safety_failure']
            result['status'] = 'stopped'
        else:
            result['status'] = 'completed_budget'
        return result, model
    except BaseException as exc:
        failure = getattr(exc, 'failure', None)
        if (isinstance(exc, Exception) and getattr(exc, 'global_stop', None) is True
                and isinstance(failure, str) and failure):
            def diagnostic(value):
                if isinstance(value, dict):
                    return {str(key): diagnostic(item) for key, item in value.items()}
                if isinstance(value, np.ndarray):
                    return diagnostic(value.tolist())
                if isinstance(value, (list, tuple)):
                    return [diagnostic(item) for item in value]
                if isinstance(value, np.generic):
                    return diagnostic(value.item())
                if isinstance(value, float) and not np.isfinite(value):
                    return str(value)
                if value is None or isinstance(value, (str, int, float, bool)):
                    return value
                return repr(value)
            # No transition is fabricated for a simulator step that aborted.
            # Finally saves the last completed PPO updates and truthful count.
            result.update(status='stopped', training_stopped='physical_safety_failure',
                safety_failure=failure, passed=False, grasp_qualified=False,
                failure_env_index=diagnostic(getattr(exc, 'env_index', None)),
                failure_info=diagnostic(getattr(exc, 'info', {})),
                exception=f'{type(exc).__name__}: {exc}')
            return result, model
        result['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
        result['exception'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        cleanups=[]
        if live_writer is not None:
            cleanups.append(('live_tensorboard',live_writer.close))
        if model is not None and getattr(model, '_logger', None) is not None:
            cleanups.append(('ppo_logger',model.logger.close))
        for name,cleanup in cleanups:
            try:
                cleanup()
            except OSError as error:
                # An optional logger failure must not replace a physics stop or
                # prevent saving the model and its original failure report.
                result.setdefault('logging_errors',[]).append(dict(
                    operation=name+'.close',error=f'{type(error).__name__}: {error}'))
                result['telemetry_status']='degraded'
        result.update(episode_budget())
        if model is not None:
            result.update(ppo_updates=int(model._n_updates), ppo_timesteps=int(model.num_timesteps))
            save_vector_checkpoint(output/'ppo_checkpoint.zip',
                                   completed_stages=['full'] if result['passed'] else [])
        write_json(output/'training_report.json', result)
        monitor(force=True)
