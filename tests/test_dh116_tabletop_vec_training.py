"""Software-only vector PPO checks; these do not qualify Isaac physics."""
import json

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from go2_pvcnn.control.dh116.tabletop_vec_training import (
    collect_vector_episodes, run_vector_training,
)


class ToyEnv(gym.Env):
    observation_space = gym.spaces.Box(-10., 10., (4,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (12,), dtype=np.float32)

    def __init__(self, length=4, fail_mode=None, terminal_nan=False, reaches_lift=True):
        self.length = length
        self.fail_mode = fail_mode
        self.terminal_nan = terminal_nan
        self.reaches_lift = reaches_lift
        self.mode = 'test'
        self.count = 0
        self.completed = 0

    def set_mode(self, mode):
        self.mode = mode

    def set_stage(self, stage):
        self.stage = stage

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        self.initial = self.np_random.uniform(-.1, .1, 4).astype(np.float32)
        return self.initial.copy(), {}

    def step(self, action):
        self.count += 1
        done = self.count >= self.length
        self.completed += int(done)
        safety = 'mimic_error' if self.mode == self.fail_mode else None
        obs = self.initial.copy()
        if done and self.terminal_nan:
            obs[0] = np.nan
        return obs, float(1 - np.mean(action ** 2)), done, False, dict(
            phase='lift' if self.reaches_lift else 'close',
            is_success=done and safety is None,
            reached_phases=['close', 'hold', 'lift'] if self.reaches_lift else ['close'],
            failure=safety, safety_failure=safety,
        )


def vector(*args):
    return DummyVecEnv([lambda kw=kw: ToyEnv(**kw) for kw in args])


def test_vector_episode_quotas_do_not_overcount_fast_auto_resets():
    env = vector(dict(length=2), dict(length=5))
    report = collect_vector_episodes(env, episodes=5, seed=7, check_reset=True)
    assert len(report['episodes']) == 5
    assert report['vector_steps'] == 15
    assert [sum(r['env_index'] == i for r in report['episodes']) for i in range(2)] == [2, 3]
    assert all(r['reset_repeatable'] for r in report['episodes'])
    assert all(r['steps'] == [2, 5][r['env_index']] for r in report['episodes'])
    assert report['physically_safe']


def test_vector_collector_detects_nonfinite_terminal_observation_hidden_by_reset():
    report = collect_vector_episodes(vector(dict(terminal_nan=True), {}), episodes=2)
    assert not report['physically_safe']
    assert report['safety_failure'] == 'nonfinite_state'


def test_all_envs_must_pass_admission_even_if_other_envs_reach_lift(tmp_path):
    env = vector({}, dict(reaches_lift=False))
    report, model = run_vector_training(env, tmp_path, {}, timesteps=32)
    assert model is None
    assert not report['admission']['allowed']
    assert 'env_1_stage_unreachable_in_two_probes' in report['admission']['reasons']


def test_real_vector_ppo_updates_aggregate_steps_saves_and_resumes(tmp_path):
    contract = dict(action='12D_direct_joint_target_v1', num_envs=2)
    env = vector(dict(length=3), dict(length=5))
    kwargs = dict(timesteps=32, eval_episodes=5, checkpoint_every=16,
                  ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    report, model = run_vector_training(env, tmp_path, contract, **kwargs)
    assert model.num_timesteps == report['ppo_timesteps'] == 32
    assert report['ppo_updates'] > 0
    assert report['evaluation']['grasp_success_rate'] == 1.
    assert report['passed']
    assert len(report['evaluation']['episodes']) == 5
    assert all(len(r['episodes']) == 2 for r in report['admission']['batches'])
    assert (tmp_path / 'checkpoint_16_steps.zip').is_file()
    progress = json.loads((tmp_path / 'monitor.json').read_text())
    assert progress['timesteps'] == 32
    assert progress['steps_per_env'] == [16, 16]
    assert len(progress['per_env']) == 2
    assert progress['per_env'][0]['episodes'] > 0
    assert progress['transitions_per_second'] > 0
    report2, model2 = run_vector_training(env, tmp_path / 'resume', contract,
        resume=tmp_path / 'ppo_checkpoint.zip', **kwargs)
    assert model2.num_timesteps == report2['ppo_timesteps'] == 64
    assert report2['ppo_parameters']['n_steps'] == 8
    with pytest.raises(ValueError, match='contract'):
        run_vector_training(env, tmp_path / 'bad', dict(contract, num_envs=3),
            resume=tmp_path / 'ppo_checkpoint.zip', **kwargs)


def test_one_env_training_safety_failure_stops_all_and_saves_checkpoint(tmp_path):
    env = vector({}, dict(fail_mode='training'))
    report, model = run_vector_training(env, tmp_path, {}, timesteps=32,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert report['training_stopped'] == 'physical_safety_failure'
    assert report['safety_failure'] == 'mimic_error'
    assert model.num_timesteps == 2
    assert 'evaluation' not in report
    assert (tmp_path / 'ppo_checkpoint.zip').is_file()


def test_vector_evaluation_failure_does_not_report_qualification(tmp_path):
    env = vector({}, dict(fail_mode='evaluation'))
    report, _ = run_vector_training(env, tmp_path, {}, timesteps=16,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert report['training_stopped'] == 'physical_safety_failure'
    assert not report['passed']
    assert not report['evaluation']['physically_safe']


def test_reset_repeatability_failure_blocks_ppo(tmp_path):
    class Nonrepeatable(ToyEnv):
        resets = 0

        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            self.resets += 1
            return obs + self.resets * .01, info

    env = DummyVecEnv([Nonrepeatable, ToyEnv])
    report, model = run_vector_training(env, tmp_path, {}, timesteps=16)
    assert model is None
    assert not report['admission']['allowed']
    assert 'batch_0_env_0_reset_not_repeatable' in report['admission']['reasons']


def test_safety_is_checked_for_env_that_has_completed_its_evaluation_quota():
    class FastEventuallyUnsafe(ToyEnv):
        def step(self, action):
            obs, reward, done, truncated, info = super().step(action)
            if self.completed >= 2:
                info['safety_failure'] = 'mimic_error'
            return obs, reward, done, truncated, info

    env = DummyVecEnv([lambda: FastEventuallyUnsafe(length=2), lambda: ToyEnv(length=9)])
    report = collect_vector_episodes(env, episodes=2)
    assert report['safety_failure'] == 'mimic_error'
    assert not report['physically_safe']
    assert len(report['episodes']) == 1  # The fast env's second episode does not count.
    assert report['vector_steps'] == 4


def test_training_checks_terminal_observation_before_auto_reset_hides_nan(tmp_path):
    class TerminalNanInTraining(ToyEnv):
        def step(self, action):
            self.terminal_nan = self.mode == 'training'
            return super().step(action)

    env = DummyVecEnv([TerminalNanInTraining, ToyEnv])
    report, model = run_vector_training(env, tmp_path, {}, timesteps=32,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert report['safety_failure'] == 'nonfinite_state'
    assert report['training_stopped'] == 'physical_safety_failure'
    assert model.num_timesteps == 8
    assert (tmp_path / 'ppo_checkpoint.zip').is_file()


def test_interrupted_vector_training_persists_current_checkpoint(tmp_path):
    class InterruptDuringTraining(ToyEnv):
        steps_training = 0

        def step(self, action):
            if self.mode == 'training':
                self.steps_training += 1
                if self.steps_training == 12:
                    raise KeyboardInterrupt()
            return super().step(action)

    env = DummyVecEnv([InterruptDuringTraining, ToyEnv])
    with pytest.raises(KeyboardInterrupt):
        run_vector_training(env, tmp_path, {}, timesteps=64, checkpoint_every=16,
            ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    report = json.loads((tmp_path / 'training_report.json').read_text())
    assert report['status'] == 'interrupted'
    assert report['ppo_timesteps'] == 22
    assert report['ppo_updates'] > 0
    assert (tmp_path / 'ppo_checkpoint.zip').is_file()


def test_weights_import_only_permits_vector_layout_change_and_resets_counters(tmp_path):
    contract = dict(num_envs=2, reset_settle_steps=0, reward='same', vector_backend='shared_physics_cpu_v1')
    kwargs = dict(timesteps=32, checkpoint_every=16,
        ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    report, _ = run_vector_training(vector({}, {}), tmp_path / 'source', contract, **kwargs)
    source = tmp_path / 'source/ppo_checkpoint.zip'
    report2, model2 = run_vector_training(vector({}, {}, {}), tmp_path / 'target',
        dict(contract, num_envs=3), initialize_from=source, **kwargs)
    assert report2['initialization_mode'] == 'policy_weights_only_fresh_optimizer_no_qualifications'
    assert model2.num_timesteps == 48  # Two whole 24-transition rollouts, fresh counter.
    with pytest.raises(ValueError, match='beyond vector layout'):
        run_vector_training(vector({}, {}), tmp_path / 'bad_import',
            dict(contract, reset_settle_steps=12), initialize_from=source, **kwargs)


def test_probe_only_validates_all_envs_without_creating_model(tmp_path):
    result, model = run_vector_training(vector({}, {}), tmp_path, {},
        timesteps=32, probe_only=True)
    assert model is None
    assert result['status'] == 'probe_complete'
    assert result['admission']['allowed']
    assert result['ppo_timesteps'] == 0
    assert not (tmp_path / 'ppo_checkpoint.zip').exists()
    assert json.loads((tmp_path / 'monitor.json').read_text())['status'] == 'probe_complete'


def test_direct_training_learns_from_failed_grasps_without_admission_steps(tmp_path):
    class FailedGrasp(ToyEnv):
        def step(self, action):
            assert self.mode != 'admission'
            obs, reward, done, truncated, info = super().step(action)
            info.update(is_success=False, failure='grasp_not_established')
            return obs, reward, done, truncated, info

    env = DummyVecEnv([lambda: FailedGrasp(reaches_lift=False), FailedGrasp])
    result, model = run_vector_training(env, tmp_path, {}, direct_training=True,
        timesteps=32, eval_episodes=2, checkpoint_every=16,
        ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert model.num_timesteps == 32
    assert result['ppo_updates'] > 0
    assert result['admission']['mode'] == 'direct_training'
    assert result['admission']['probes_skipped'] is True
    assert result['admission']['batches'] == []
    assert result['evaluation']['grasp_success_rate'] == 0
    assert not result['passed']


def test_direct_training_retains_per_step_physical_stop(tmp_path):
    env = vector({}, dict(fail_mode='training'))
    result, model = run_vector_training(env, tmp_path, {}, direct_training=True,
        timesteps=32, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert result['safety_failure'] == 'mimic_error'
    assert result['training_stopped'] == 'physical_safety_failure'
    assert model.num_timesteps == 2
    assert result['ppo_updates'] == 0
    assert (tmp_path / 'ppo_checkpoint.zip').exists()


@pytest.mark.parametrize('bad_reset', ['nan', 'safety'])
def test_direct_training_rejects_invalid_reset(tmp_path, bad_reset):
    class BadReset(ToyEnv):
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            if bad_reset == 'nan':
                obs[0] = np.nan
            else:
                info['safety_failure'] = 'mimic_error'
            return obs, info

    result, model = run_vector_training(DummyVecEnv([BadReset, ToyEnv]),
        tmp_path, {}, direct_training=True, timesteps=32)
    assert model is None
    assert result['status'] == 'not_admitted'
    assert result['safety_failure'] == ('nonfinite_state' if bad_reset == 'nan' else 'mimic_error')


def test_direct_training_cannot_be_combined_with_probe_only(tmp_path):
    with pytest.raises(ValueError, match='direct_training.*probe_only'):
        run_vector_training(vector({}, {}), tmp_path, {},
                            direct_training=True, probe_only=True)


def test_episode_budget_waits_for_each_env_and_finishes_rollout_update(tmp_path):
    result, model = run_vector_training(vector(dict(length=2), dict(length=5)),
        tmp_path, {}, direct_training=True, episodes_per_env=3, timesteps=1,
        eval_episodes=2, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    progress = json.loads((tmp_path / 'monitor.json').read_text())
    assert [r['episodes'] for r in progress['per_env']] == [8, 3]
    assert model.num_timesteps == 32
    assert model._n_updates == 2
    assert progress['episodes_per_env_target'] == 3
    assert result['episode_target_reached'] is True
    assert progress['episode_target_reached'] is True


def test_episode_budget_resume_preserves_counts_and_optimizer_updates(tmp_path):
    kwargs = dict(direct_training=True, eval_episodes=2,
                  ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    _, first = run_vector_training(vector(dict(length=2), dict(length=5)),
        tmp_path / 'a', {}, episodes_per_env=3, **kwargs)
    result, resumed = run_vector_training(vector(dict(length=2), dict(length=5)),
        tmp_path / 'b', {}, episodes_per_env=4,
        resume=tmp_path / 'a/ppo_checkpoint.zip', **kwargs)
    progress = json.loads((tmp_path / 'b/monitor.json').read_text())
    assert [r['episodes'] for r in progress['per_env']] == [12, 4]
    assert resumed.num_timesteps == first.num_timesteps + 16
    assert resumed._n_updates == first._n_updates + 1
    assert result['episode_target_reached']
    metadata = json.loads((tmp_path / 'b/ppo_checkpoint.json').read_text())
    assert metadata['training_progress']['per_env'] == progress['per_env']


def test_episode_budget_stops_on_physical_failure_before_quota(tmp_path):
    result, model = run_vector_training(vector({}, dict(fail_mode='training')),
        tmp_path, {}, direct_training=True, episodes_per_env=1024,
        ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert result['training_stopped'] == 'physical_safety_failure'
    assert not result['episode_target_reached']
    assert model.num_timesteps == 2


def test_episode_budget_already_met_resume_does_not_train_more(tmp_path):
    kwargs = dict(direct_training=True, eval_episodes=2,
                  ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    _, first = run_vector_training(vector({}, {}), tmp_path / 'a', {},
                                   episodes_per_env=2, **kwargs)
    result, resumed = run_vector_training(vector({}, {}), tmp_path / 'b', {},
        episodes_per_env=2, resume=tmp_path / 'a/ppo_checkpoint.zip', **kwargs)
    assert resumed.num_timesteps == first.num_timesteps
    assert resumed._n_updates == first._n_updates
    assert result['episode_target_reached']


def test_episode_budget_resume_uses_requested_device(tmp_path, monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_vec_training as training
    original = training.load_checkpoint
    seen = []

    def recorded_load(*args, **kwargs):
        seen.append(kwargs['device'])
        return original(*args, **kwargs)

    kwargs = dict(direct_training=True, eval_episodes=2,
                  ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1, device='cpu'))
    run_vector_training(vector({}, {}), tmp_path / 'a', {}, episodes_per_env=2, **kwargs)
    monkeypatch.setattr(training, 'load_checkpoint', recorded_load)
    run_vector_training(vector({}, {}), tmp_path / 'b', {}, episodes_per_env=3,
                        resume=tmp_path / 'a/ppo_checkpoint.zip', **kwargs)
    assert seen == ['cpu']


@pytest.mark.parametrize('target', [0, -1, 1.5, True])
def test_episode_budget_rejects_invalid_target(tmp_path, target):
    with pytest.raises(ValueError, match='episodes_per_env'):
        run_vector_training(vector({}, {}), tmp_path, {}, episodes_per_env=target)


def test_episode_budget_migrates_only_matching_old_checkpoint_progress(tmp_path):
    kwargs = dict(direct_training=True, eval_episodes=2,
                  ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    run_vector_training(vector({}, {}), tmp_path / 'a', {}, timesteps=16, **kwargs)
    metadata_path = tmp_path / 'a/ppo_checkpoint.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.pop('training_progress', None)
    metadata_path.write_text(json.dumps(metadata))
    source = tmp_path / 'a/ppo_checkpoint.zip'
    with pytest.raises(ValueError, match='episode progress'):
        run_vector_training(vector({}, {}), tmp_path / 'missing', {},
            resume=source, episodes_per_env=3, **kwargs)
    progress_path = tmp_path / 'a/monitor.json'
    good = json.loads(progress_path.read_text())
    bad = dict(good, timesteps=good['timesteps'] + 2)
    (tmp_path / 'bad_progress.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='progress.*checkpoint'):
        run_vector_training(vector({}, {}), tmp_path / 'bad', {},
            resume=source, episodes_per_env=3,
            resume_episode_progress=tmp_path / 'bad_progress.json', **kwargs)
    report, _ = run_vector_training(vector({}, {}), tmp_path / 'good', {},
        resume=source, episodes_per_env=3,
        resume_episode_progress=progress_path, **kwargs)
    p = json.loads((tmp_path / 'good/monitor.json').read_text())
    assert [x['episodes'] for x in p['per_env']] == [4, 4]
    assert report['episode_target_reached']


def test_evaluation_seed_labels_distinguish_seeded_reset_from_rng_continuation():
    result = collect_vector_episodes(vector(dict(length=2), dict(length=3)),
        episodes=5, seed=200)
    for env_index in range(2):
        rows = [row for row in result['episodes'] if row['env_index'] == env_index]
        assert [row['episode_index'] for row in rows] == list(range(len(rows)))
        assert all(row['reset_seed'] == 200 + env_index for row in rows)
        assert rows[0]['seed'] == 200 + env_index
        assert all(row['seed'] is None for row in rows[1:])


def test_training_throughput_excludes_evaluation_time(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116 import tabletop_vec_training
    clock = [0.]
    monkeypatch.setattr(tabletop_vec_training, 'time', SimpleNamespace(monotonic=lambda: clock[0]))

    class TimedEnv(ToyEnv):
        def step(self, action):
            clock[0] += 1. if self.mode == 'training' else (100. if self.mode == 'evaluation' else 0.)
            return super().step(action)

    env = DummyVecEnv([TimedEnv, TimedEnv])
    run_vector_training(env, tmp_path, {}, timesteps=16, checkpoint_every=8,
        ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    progress = json.loads((tmp_path / 'monitor.json').read_text())
    assert progress['training_elapsed_seconds'] == 16.
    assert progress['transitions_per_second'] == 1.
    assert progress['elapsed_seconds'] > 1000.


def test_monitor_records_last_physics_and_training_extrema_per_env(tmp_path):
    class PhysicalEnv(ToyEnv):
        def step(self, action):
            obs, reward, done, truncated, info = super().step(action)
            info['physical'] = dict(object_bottom_m=.32 if self.count == 2 else .30,
                slip_m=.001*self.count, max_force_n=.1*self.count,
                max_mimic_error_rad=.012 if self.count == 1 else .003)
            return obs, reward, done, truncated, info

    env = DummyVecEnv([PhysicalEnv, PhysicalEnv])
    run_vector_training(env, tmp_path, {}, timesteps=16, checkpoint_every=8,
        ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    progress = json.loads((tmp_path / 'monitor.json').read_text())
    for row in progress['per_env']:
        assert row['last_physical'] == dict(object_bottom_m=.30, slip_m=.004,
            max_force_n=.4, max_mimic_error_rad=.003)
        assert row['max_object_bottom_m'] == .32
        assert row['max_mimic_error_rad'] == .012
        assert row['max_slip_m'] == .004
        assert row['max_force_n'] == .4


@pytest.fixture
def refinement_source(tmp_path):
    contract = dict(num_envs=2, physics_hz=240, control_hz=60, reset_settle_steps=0,
        reward='object_support_progress_v3', action='12D_direct_joint_target_v1',
        coupling_warning='follower_consistent_recovery_v2', mimic_natural_frequency_s_inv=200.)
    kwargs = dict(timesteps=32, checkpoint_every=16,
        ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    report, model = run_vector_training(vector({}, {}), tmp_path/'source', contract, **kwargs)
    return tmp_path/'source/ppo_checkpoint.zip', contract, report, model


def test_explicit_physics_refinement_transfers_weights_with_fresh_optimizer_and_provenance(
        tmp_path, refinement_source):
    import hashlib
    source, contract, source_report, source_model = refinement_source
    original_metadata = source.with_suffix('.json').read_bytes()
    target = dict(contract, physics_hz=480, seed_physics_hz=240)
    report, model = run_vector_training(vector({}, {}), tmp_path/'refined', target,
        initialize_from=source, allow_physics_refinement=True, timesteps=16,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert model.num_timesteps == report['ppo_timesteps'] == 16
    assert model._n_updates < source_model._n_updates
    assert all(float(row['step']) == 2 for row in model.policy.optimizer.state.values())
    assert report['initialization_mode'] == 'policy_weights_only_fresh_optimizer_no_qualifications'
    assert report['physics_refinement'] == dict(source_physics_hz=240, physics_hz=480,
        control_hz=60, seed_physics_hz=240)
    assert report['source_ppo_timesteps'] == source_report['ppo_timesteps'] == 32
    assert report['source_updates'] == source_report['ppo_updates']
    assert report['source_contract'] == contract
    assert report['initialized_from'] == str(source)
    assert report['initialized_from_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert source.with_suffix('.json').read_bytes() == original_metadata


@pytest.mark.parametrize('changed', [
    {'reward': 'different_reward'}, {'action': 'different_action'},
    {'coupling_warning': 'new_recovery'}, {'mimic_natural_frequency_s_inv': 400.},
    {'physics_hz': 960}, {'physics_hz': 240}, {'control_hz': 120},
    {'seed_physics_hz': 480}, {'seed_physics_hz': None},
])
def test_refinement_rejects_unrelated_or_invalid_contract_changes(tmp_path, refinement_source, changed):
    source, contract, _, _ = refinement_source
    target = dict(contract, physics_hz=480, seed_physics_hz=240)
    target.update(changed)
    with pytest.raises(ValueError, match='refinement'):
        run_vector_training(vector({}, {}), tmp_path/'rejected', target,
            initialize_from=source, allow_physics_refinement=True, timesteps=16)


def test_physics_refinement_requires_explicit_flag(tmp_path, refinement_source):
    source, contract, _, _ = refinement_source
    with pytest.raises(ValueError, match='beyond vector layout'):
        run_vector_training(vector({}, {}), tmp_path/'rejected',
            dict(contract, physics_hz=480, seed_physics_hz=240), initialize_from=source, timesteps=16)


def test_reverse_physics_refinement_is_rejected(tmp_path, refinement_source):
    source, contract, _, _ = refinement_source
    metadata = json.loads(source.with_suffix('.json').read_text())
    metadata['contract'].update(physics_hz=480, seed_physics_hz=240)
    source.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='refinement'):
        run_vector_training(vector({}, {}), tmp_path/'rejected',
            dict(contract, physics_hz=240, seed_physics_hz=240), initialize_from=source,
            allow_physics_refinement=True, timesteps=16)


@pytest.mark.parametrize('argument', [{}, {'resume': 'not-a-real-checkpoint.zip'}])
def test_refinement_flag_requires_weights_initialization_not_resume(tmp_path, argument):
    with pytest.raises(ValueError, match='initialize_from'):
        run_vector_training(vector({}, {}), tmp_path, {}, timesteps=16,
            allow_physics_refinement=True, **argument)


def test_refinement_preserves_policy_weights_but_not_optimizer_or_qualification(tmp_path, refinement_source):
    import torch
    from stable_baselines3 import PPO
    source, contract, source_report, source_model = refinement_source
    assert source_report['passed']
    output = tmp_path/'probed_transfer'
    report, returned_model = run_vector_training(vector({}, {}), output,
        dict(contract, physics_hz=480, seed_physics_hz=240), initialize_from=source,
        allow_physics_refinement=True, probe_only=True, timesteps=16)
    assert returned_model is None
    assert not report['passed'] and not report['grasp_qualified']
    restored = PPO.load(output/'ppo_checkpoint.zip', device='cpu')
    assert restored.num_timesteps == restored._n_updates == 0
    assert not restored.policy.optimizer.state
    assert all(torch.equal(value, restored.policy.state_dict()[key])
        for key, value in source_model.policy.state_dict().items())
    assert json.loads((output/'ppo_checkpoint.json').read_text())['completed_stages'] == []


@pytest.mark.parametrize('changed,env_count', [
    ({'num_envs': 3}, 3), ({'env_spacing_m': 3.}, 2),
    ({'vector_backend': 'different_backend'}, 2),
])
def test_physics_refinement_cannot_also_change_vector_layout(
        tmp_path, refinement_source, changed, env_count):
    source, contract, _, _ = refinement_source
    target = dict(contract, physics_hz=480, seed_physics_hz=240)
    target.update(changed)
    with pytest.raises(ValueError, match='refinement'):
        run_vector_training(vector(*[{} for _ in range(env_count)]), tmp_path/'rejected',
            target, initialize_from=source, allow_physics_refinement=True, timesteps=16,
            ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))


@pytest.fixture
def mimic_refinement_source(tmp_path):
    contract = dict(num_envs=2, env_spacing_m=2., vector_backend='shared_physics_cpu_v1',
        physics_hz=480, control_hz=60, seed_physics_hz=240, reset_settle_steps=0,
        reward='object_support_progress_v3', action='12D_direct_joint_target_v1',
        coupling_warning='follower_consistent_recovery_v2',
        mimic_natural_frequency_s_inv=200., mimic_damping_ratio=1.)
    report, model = run_vector_training(vector({}, {}), tmp_path/'source', contract,
        timesteps=32, checkpoint_every=16,
        ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    return tmp_path/'source/ppo_checkpoint.zip', contract, report, model


def test_explicit_mimic_refinement_preserves_weights_and_provenance_with_fresh_state(
        tmp_path, mimic_refinement_source):
    import hashlib
    import torch
    from stable_baselines3 import PPO
    source, contract, source_report, source_model = mimic_refinement_source
    original_metadata = source.with_suffix('.json').read_bytes()
    target = dict(contract, mimic_natural_frequency_s_inv=400.)
    output = tmp_path/'refined'
    report, _ = run_vector_training(vector({}, {}), output, target,
        initialize_from=source, allow_mimic_refinement=True, probe_only=True, timesteps=16)
    restored = PPO.load(output/'ppo_checkpoint.zip', device='cpu')
    assert restored.num_timesteps == restored._n_updates == 0
    assert not restored.policy.optimizer.state
    assert all(torch.equal(value, restored.policy.state_dict()[key])
        for key, value in source_model.policy.state_dict().items())
    assert not report['passed'] and not report['grasp_qualified']
    assert report['mimic_refinement'] == dict(physics_hz=480, control_hz=60,
        source_mimic_natural_frequency_s_inv=200., mimic_natural_frequency_s_inv=400.,
        mimic_damping_ratio=1.)
    assert report['source_ppo_timesteps'] == source_report['ppo_timesteps'] == 32
    assert report['source_updates'] == source_report['ppo_updates']
    assert report['source_contract'] == contract
    assert report['initialized_from_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert source.with_suffix('.json').read_bytes() == original_metadata
    trained, model = run_vector_training(vector({}, {}), tmp_path/'trained', target,
        initialize_from=source, allow_mimic_refinement=True, timesteps=16,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert model.num_timesteps == trained['ppo_timesteps'] == 16
    assert model._n_updates == 1
    assert all(float(row['step']) == 2 for row in model.policy.optimizer.state.values())


@pytest.mark.parametrize('changed,env_count', [
    ({'reward': 'different'}, 2), ({'action': 'different'}, 2),
    ({'coupling_warning': 'new_recovery'}, 2), ({'physics_hz': 240}, 2),
    ({'control_hz': 120}, 2), ({'mimic_damping_ratio': .8}, 2),
    ({'mimic_natural_frequency_s_inv': 800.}, 2),
    ({'mimic_natural_frequency_s_inv': 200.}, 2),
    ({'num_envs': 3}, 3), ({'env_spacing_m': 3.}, 2),
    ({'vector_backend': 'shared_physics_kinematic_palm_v2'}, 2),
    ({'seed_physics_hz': 480}, 2),
])
def test_mimic_refinement_rejects_all_other_changes(
        tmp_path, mimic_refinement_source, changed, env_count):
    source, contract, _, _ = mimic_refinement_source
    target = dict(contract, mimic_natural_frequency_s_inv=400.)
    target.update(changed)
    with pytest.raises(ValueError, match='refinement'):
        run_vector_training(vector(*[{} for _ in range(env_count)]), tmp_path/'rejected',
            target, initialize_from=source, allow_mimic_refinement=True, timesteps=16)


def test_mimic_refinement_requires_explicit_flag(tmp_path, mimic_refinement_source):
    source, contract, _, _ = mimic_refinement_source
    with pytest.raises(ValueError, match='beyond vector layout'):
        run_vector_training(vector({}, {}), tmp_path/'rejected',
            dict(contract, mimic_natural_frequency_s_inv=400.), initialize_from=source, timesteps=16)


def test_reverse_mimic_refinement_is_rejected(tmp_path, mimic_refinement_source):
    source, contract, _, _ = mimic_refinement_source
    metadata = json.loads(source.with_suffix('.json').read_text())
    metadata['contract']['mimic_natural_frequency_s_inv'] = 400.
    source.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='refinement'):
        run_vector_training(vector({}, {}), tmp_path/'rejected', contract,
            initialize_from=source, allow_mimic_refinement=True, timesteps=16)


@pytest.mark.parametrize('argument', [
    {}, {'resume': 'not-a-real-checkpoint.zip'},
    {'initialize_from': 'not-a-real-checkpoint.zip', 'allow_physics_refinement': True},
])
def test_mimic_refinement_rejects_resume_missing_initialization_and_combined_flags(tmp_path, argument):
    with pytest.raises(ValueError, match='initialize_from|mutually exclusive'):
        run_vector_training(vector({}, {}), tmp_path, {}, timesteps=16,
            allow_mimic_refinement=True, **argument)


def test_protected_mimic_initialization_preserves_weights_and_resets_qualification(
        tmp_path, mimic_refinement_source):
    import torch
    from stable_baselines3 import PPO
    source, contract, _, source_model = mimic_refinement_source
    target = dict(contract, training_failure_handling='validated_mimic_reset_v1')
    report, _ = run_vector_training(vector({}, {}), tmp_path/'protected', target,
        initialize_from=source, allow_protected_mimic_init=True, probe_only=True, timesteps=16)
    restored = PPO.load(tmp_path/'protected/ppo_checkpoint.zip', device='cpu')
    assert restored.num_timesteps == restored._n_updates == 0
    assert not restored.policy.optimizer.state
    assert all(torch.equal(value, restored.policy.state_dict()[key])
        for key, value in source_model.policy.state_dict().items())
    assert not report['passed'] and not report['grasp_qualified']
    assert report['source_contract'] == contract
    assert report['protected_mimic_initialization'] == 'validated_mimic_reset_v1'


@pytest.mark.parametrize('changed,env_count', [
    ({'mimic_natural_frequency_s_inv': 400.}, 2), ({'physics_hz': 240}, 2),
    ({'reward': 'changed'}, 2), ({'action': 'changed'}, 2),
    ({'num_envs': 3}, 3), ({'env_spacing_m': 3.}, 2),
    ({'vector_backend': 'different'}, 2), ({'seed_physics_hz': 480}, 2),
    ({'training_failure_handling': 'unvalidated_reset'}, 2),
])
def test_protected_mimic_initialization_rejects_any_other_change(
        tmp_path, mimic_refinement_source, changed, env_count):
    source, contract, _, _ = mimic_refinement_source
    target = dict(contract, training_failure_handling='validated_mimic_reset_v1')
    target.update(changed)
    with pytest.raises(ValueError, match='protected|refinement'):
        run_vector_training(vector(*[{} for _ in range(env_count)]), tmp_path/'rejected', target,
            initialize_from=source, allow_protected_mimic_init=True, timesteps=16)


@pytest.mark.parametrize('argument', [
    {}, {'resume': 'fake.zip'},
    {'initialize_from': 'fake.zip', 'allow_physics_refinement': True},
    {'initialize_from': 'fake.zip', 'allow_mimic_refinement': True},
])
def test_protected_mimic_initialization_rejects_resume_and_combined_migrations(tmp_path, argument):
    with pytest.raises(ValueError, match='initialize_from|mutually exclusive'):
        run_vector_training(vector({}, {}), tmp_path, {}, timesteps=16,
            allow_protected_mimic_init=True, **argument)


def validated_reset_info():
    return dict(version='validated_mimic_reset_v1', verified=True, passed=True,
        other_guards_safe=True, reset_guards_safe=True, post_reset_guards_safe=True,
        terminal_mimic_abs_rad=.021, checked_physics_substeps=3,
        terminal_physics_substep=5, remaining_physics_substeps=3,
        validation_scope='reset_and_remaining_substeps')


class RecoveredMimicToy(ToyEnv):
    def __init__(self, mode='training', mutation=None):
        super().__init__()
        self.recovery_mode = mode
        self.mutation = mutation
        self.failure_infos = []

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if self.mode == self.recovery_mode and done:
            reward = -25.
            info.update(failure='mimic_error', safety_failure='mimic_error',
                is_success=False, stage_success=False, recovered_mimic=True, global_stop=False,
                physical=dict(object_bottom_m=.30, slip_m=.01, max_force_n=.8,
                              max_mimic_error_rad=.021), reset_validation=validated_reset_info())
            if self.mutation:
                self.mutation(info)
            self.failure_infos.append(info)
        return obs, reward, done, truncated, info


def protected_contract():
    return dict(training_failure_handling='validated_mimic_reset_v1',
                physics_hz=480, control_hz=60)


def test_validated_mimic_failure_enters_real_ppo_buffer_and_training_continues(tmp_path):
    env = DummyVecEnv([RecoveredMimicToy, RecoveredMimicToy])
    report, model = run_vector_training(env, tmp_path, protected_contract(), timesteps=32,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert model.num_timesteps == 32 and model._n_updates == 2
    assert 'training_stopped' not in report
    assert np.count_nonzero(model.rollout_buffer.rewards == -25.) == 4
    assert np.count_nonzero(model.rollout_buffer.episode_starts[4]) == 2
    assert all(info['failure'] == info['safety_failure'] == 'mimic_error'
        and not info['is_success'] for e in env.envs for info in e.failure_infos)
    assert report['recovered_mimic_episodes'] == 8
    monitor = json.loads((tmp_path/'monitor.json').read_text())
    assert monitor['recovered_mimic_episodes'] == 8
    for row in monitor['per_env']:
        assert row['recovered_mimic_episodes'] == 4
        assert row['failure_counts'] == {'mimic_error': 4}
        assert row['successes'] == 0


@pytest.mark.parametrize('mutation', [
    lambda i: i.pop('reset_validation'),
    lambda i: i['reset_validation'].update(verified=False),
    lambda i: i['reset_validation'].update(passed=False),
    lambda i: i['reset_validation'].update(version='unvalidated'),
    lambda i: i['reset_validation'].update(other_guards_safe=False),
    lambda i: i['reset_validation'].update(reset_guards_safe=False),
    lambda i: i['reset_validation'].update(post_reset_guards_safe=False),
    lambda i: i['reset_validation'].update(checked_physics_substeps=-1),
    lambda i: i['reset_validation'].update(checked_physics_substeps=8),
    lambda i: i['reset_validation'].update(checked_physics_substeps=True),
    lambda i: i['reset_validation'].update(terminal_mimic_abs_rad=.022),
    lambda i: i['physical'].update(max_force_n=31.),
    lambda i: i['physical'].update(object_bottom_m=.28),
    lambda i: i['physical'].update(slip_m=float('nan')),
    lambda i: i['physical'].update(max_mimic_error_rad=.03),
    lambda i: i.update(recovered_mimic=False),
    lambda i: i.update(safety_failure='joint_limit'),
    lambda i: i.update(is_success=True),
])
def test_missing_or_invalid_recovery_ack_cannot_bypass_global_safety(tmp_path, mutation):
    env = DummyVecEnv([lambda: RecoveredMimicToy(mutation=mutation), ToyEnv])
    report, model = run_vector_training(env, tmp_path, protected_contract(), timesteps=32,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert report['training_stopped'] == 'physical_safety_failure'
    assert model.num_timesteps == 8 and model._n_updates == 0
    assert report.get('recovered_mimic_episodes', 0) == 0


def test_old_contract_still_stops_even_with_complete_recovery_ack(tmp_path):
    env = DummyVecEnv([RecoveredMimicToy, ToyEnv])
    report, _ = run_vector_training(env, tmp_path, {}, timesteps=16,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert report['training_stopped'] == 'physical_safety_failure'


@pytest.mark.parametrize('mode', ['admission', 'evaluation'])
def test_recovery_ack_does_not_relax_admission_or_evaluation(tmp_path, mode):
    env = DummyVecEnv([lambda: RecoveredMimicToy(mode=mode), ToyEnv])
    report, model = run_vector_training(env, tmp_path, protected_contract(), timesteps=16,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    if mode == 'admission':
        assert not report['admission']['allowed'] and model is None
    else:
        assert report['training_stopped'] == 'physical_safety_failure'
        assert not report['evaluation']['physically_safe'] and not report['passed']


@pytest.mark.parametrize('changed', [
    {'mimic_natural_frequency_s_inv': 400.}, {'physics_hz': 240},
    {'control_hz': 30}, {'mimic_damping_ratio': .8},
    {'training_failure_handling': None},
])
def test_protected_initialization_rejects_non_v30_source_even_if_other_target_fields_match(
        tmp_path, mimic_refinement_source, changed):
    source, contract, _, _ = mimic_refinement_source
    metadata = json.loads(source.with_suffix('.json').read_text())
    metadata['contract'].update(changed)
    source.with_suffix('.json').write_text(json.dumps(metadata))
    target = dict(metadata['contract'], training_failure_handling='validated_mimic_reset_v1')
    with pytest.raises(ValueError, match='protected'):
        run_vector_training(vector({}, {}), tmp_path/'rejected', target,
            initialize_from=source, allow_protected_mimic_init=True, timesteps=16)


def test_protected_initialization_cannot_be_implicitly_enabled(tmp_path, mimic_refinement_source):
    source, contract, _, _ = mimic_refinement_source
    with pytest.raises(ValueError, match='beyond vector layout'):
        run_vector_training(vector({}, {}), tmp_path/'rejected',
            dict(contract, training_failure_handling='validated_mimic_reset_v1'),
            initialize_from=source, timesteps=16)


def test_recovery_ack_cannot_hide_fatal_failure_in_another_env(tmp_path):
    env = DummyVecEnv([RecoveredMimicToy, lambda: RecoveredMimicToy(
        mutation=lambda i: i.update(safety_failure='joint_limit'))])
    report, model = run_vector_training(env, tmp_path, protected_contract(), timesteps=32,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert report['training_stopped'] == 'physical_safety_failure'
    assert report['safety_failure'] == 'joint_limit'
    assert model.num_timesteps == 8 and model._n_updates == 0


def test_unrecoverable_physics_exception_stops_and_saves_updated_model_without_retry(tmp_path):
    class FatalPhysicsToy(ToyEnv):
        training_calls = 0

        def step(self, action):
            if self.mode == 'training':
                self.training_calls += 1
                if self.training_calls == 12:
                    exc = RuntimeError('shared physics stopped')
                    exc.global_stop = True
                    exc.failure = 'mimic_error'
                    exc.env_index = 0
                    exc.info = dict(failure='mimic_error', recovered_mimic=False,
                        physical=dict(max_mimic_error_rad=np.float64(.04)),
                        terminal_observation=np.array([1., np.nan, 3., 4.]))
                    raise exc
            return super().step(action)

    env = DummyVecEnv([FatalPhysicsToy, ToyEnv])
    report, model = run_vector_training(env, tmp_path, protected_contract(), timesteps=64,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert report['status'] == 'stopped'
    assert report['training_stopped'] == 'physical_safety_failure'
    assert report['safety_failure'] == 'mimic_error'
    assert report['failure_env_index'] == 0
    assert report['failure_info']['physical']['max_mimic_error_rad'] == .04
    assert report['failure_info']['terminal_observation'] == [1., 'nan', 3., 4.]
    assert model.num_timesteps == 22 and model._n_updates == 1
    assert env.envs[0].training_calls == 12
    assert not report['passed'] and 'evaluation' not in report
    metadata = json.loads((tmp_path/'ppo_checkpoint.json').read_text())
    assert metadata['timesteps'] == 22 and metadata['updates'] == 1
    assert json.loads((tmp_path/'training_report.json').read_text())['status'] == 'stopped'
    assert json.loads((tmp_path/'monitor.json').read_text())['safety_stop'] == 'mimic_error'


@pytest.mark.parametrize('global_stop,failure', [(False, 'mimic_error'), (1, 'mimic_error'),
                                                (True, None), (True, 123), (True, '')])
def test_plain_runtime_bugs_or_invalid_global_stop_protocol_still_raise(tmp_path, global_stop, failure):
    class BuggyToy(ToyEnv):
        def step(self, action):
            if self.mode == 'training':
                exc = RuntimeError('ordinary bug')
                exc.global_stop, exc.failure = global_stop, failure
                raise exc
            return super().step(action)

    with pytest.raises(RuntimeError, match='ordinary bug'):
        run_vector_training(DummyVecEnv([BuggyToy, ToyEnv]), tmp_path, protected_contract(),
            timesteps=16, checkpoint_every=16,
            ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert json.loads((tmp_path/'training_report.json').read_text())['status'] == 'error'


@pytest.mark.parametrize('mutation', [
    lambda i: i.update(global_stop=True),
    lambda i: i.update(global_stop=0),
    lambda i: i.pop('global_stop'),
    lambda i: i.update(global_stop_failure='mimic_error'),
    lambda i: i['reset_validation'].pop('terminal_physics_substep'),
    lambda i: i['reset_validation'].update(terminal_physics_substep=True),
    lambda i: i['reset_validation'].update(terminal_physics_substep=0),
    lambda i: i['reset_validation'].update(terminal_physics_substep=9),
    lambda i: i['reset_validation'].update(terminal_physics_substep=4),
    lambda i: i['reset_validation'].pop('remaining_physics_substeps'),
    lambda i: i['reset_validation'].update(remaining_physics_substeps=True),
    lambda i: i['reset_validation'].update(remaining_physics_substeps=2),
    lambda i: i['reset_validation'].update(checked_physics_substeps=2),
    lambda i: i['reset_validation'].update(validation_scope='immediate_reset_only'),
    lambda i: i['reset_validation'].pop('validation_scope'),
])
def test_recovery_ack_requires_consistent_tail_validation_and_explicit_no_global_stop(tmp_path, mutation):
    env = DummyVecEnv([lambda: RecoveredMimicToy(mutation=mutation), ToyEnv])
    report, model = run_vector_training(env, tmp_path, protected_contract(), timesteps=16,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert report['training_stopped'] == 'physical_safety_failure'
    assert model.num_timesteps == 8 and model._n_updates == 0


@pytest.mark.parametrize('terminal_observation', [
    np.zeros(3), np.zeros((1, 4)), np.array([1, 2, 3, '4']),
    np.array([1., 2., 3., np.nan]), np.array([True, False, True, False]),
])
def test_recovery_ack_rejects_malformed_terminal_observation(tmp_path, terminal_observation):
    class InvalidTerminalVec(DummyVecEnv):
        def step_wait(self):
            obs, rewards, dones, infos = super().step_wait()
            if self.envs[0].mode == 'training' and dones[0]:
                infos[0]['terminal_observation'] = terminal_observation
            return obs, rewards, dones, infos

    env = InvalidTerminalVec([RecoveredMimicToy, ToyEnv])
    report, model = run_vector_training(env, tmp_path, protected_contract(), timesteps=16,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert report['training_stopped'] == 'physical_safety_failure'
    assert model.num_timesteps == 8 and model._n_updates == 0


def test_recovery_ack_cannot_claim_failure_as_time_limit_truncation(tmp_path):
    class TruncatedTerminalVec(DummyVecEnv):
        def step_wait(self):
            obs, rewards, dones, infos = super().step_wait()
            if self.envs[0].mode == 'training' and dones[0]:
                infos[0]['TimeLimit.truncated'] = True
            return obs, rewards, dones, infos

    report, model = run_vector_training(TruncatedTerminalVec([RecoveredMimicToy, ToyEnv]),
        tmp_path, protected_contract(), timesteps=16, checkpoint_every=16,
        ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert report['training_stopped'] == 'physical_safety_failure'
    assert model.num_timesteps == 8


def test_last_substep_ack_must_truthfully_report_immediate_reset_only(tmp_path):
    def last_substep(info):
        info['reset_validation'].update(terminal_physics_substep=8,
            remaining_physics_substeps=0, checked_physics_substeps=0,
            validation_scope='immediate_reset_only')

    env = DummyVecEnv([lambda: RecoveredMimicToy(mutation=last_substep), ToyEnv])
    report, model = run_vector_training(env, tmp_path,
        dict(protected_contract(), observation_size=4), timesteps=16,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert model.num_timesteps == 16 and model._n_updates == 1
    assert report['recovered_mimic_episodes'] == 2


def test_recovery_observation_must_match_declared_contract_size(tmp_path):
    report, model = run_vector_training(DummyVecEnv([RecoveredMimicToy, ToyEnv]), tmp_path,
        dict(protected_contract(), observation_size=108), timesteps=16,
        checkpoint_every=16, ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    assert report['training_stopped'] == 'physical_safety_failure'
    assert model.num_timesteps == 8
