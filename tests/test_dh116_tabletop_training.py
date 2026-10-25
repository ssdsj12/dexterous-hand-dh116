"""Training lifecycle tests; the toy environment is NOT physical evidence."""
import json

import gymnasium as gym
import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_training import (
    admission_report, evaluate, load_checkpoint, run_training, save_checkpoint, run_episode,
    report_exit_code,
)


def evidence(**overrides):
    row = dict(steps=8, finite=True, reset_repeatable=True, safety_failure=None,
               failure='timeout', reached_phases=['approach', 'close'],
               stage_success=False, is_success=False)
    row.update(overrides)
    return row


def test_admission_allows_safe_task_failure_but_requires_complete_evidence():
    probes = [evidence() for _ in range(3)]
    assert admission_report('close', probes)['allowed']
    assert not admission_report('lift', probes)['allowed']
    assert not admission_report('approach', [{}] * 3)['allowed']
    assert not admission_report('approach', probes[:1])['allowed']
    for field, value in [('finite', False), ('reset_repeatable', False),
                         ('safety_failure', 'mimic_error'), ('steps', 0)]:
        assert not admission_report('approach', [evidence(**{field: value})] + probes[1:])['allowed']


class SyntheticEnv(gym.Env):
    """Eight-step synthetic episodes, solely exercising real SB3 wiring."""
    observation_space = gym.spaces.Box(-1., 1., (4,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (12,), dtype=np.float32)

    def __init__(self, fail=False):
        self.stage = 'approach'
        self.fail = fail
        self.mode = 'test'

    def set_stage(self, stage):
        self.stage = stage

    def set_mode(self, mode):
        self.mode = mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return np.zeros(4, np.float32), {}

    def step(self, action):
        self.count += 1
        done = self.count == 8
        full = self.stage in ('lift', 'full')
        failure = 'mimic_error' if self.fail and done else None
        return np.zeros(4, np.float32), float(1 - np.mean(action**2)), done, False, dict(
            is_success=done and full and not failure,
            stage_success=done and not failure, failure=failure,
            safety_failure=failure,
            reached_phases=['approach', 'close', 'hold', 'lift'],
        )


def test_evaluation_does_not_confuse_stage_completion_with_grasp():
    result = evaluate(SyntheticEnv(), None, 'approach', episodes=2, seed=12)
    assert result['stage_success_rate'] == 1
    assert result['grasp_success_rate'] == 0
    assert len(result['episodes']) == 2


def test_real_sb3_update_checkpoint_resume_and_contract_rejection(tmp_path):
    env = SyntheticEnv()
    contract = {'observation_version': 'synthetic-test-only', 'action': 'bounded-12d'}
    result, model = run_training(env, tmp_path, contract, stages=['approach'], timesteps=32,
        eval_episodes=2, probe_episodes=3, checkpoint_every=16,
        ppo_kwargs=dict(n_steps=16, batch_size=8, n_epochs=1))
    assert result['ppo_updates'] > 0
    assert result['ppo_timesteps'] == 32
    assert result['passed'] is False
    assert result['stages'][0]['evaluation']['stage_success_rate'] == 1
    checkpoint = tmp_path / 'ppo_checkpoint.zip'
    assert checkpoint.is_file() and checkpoint.with_suffix('.json').is_file()
    restored, metadata = load_checkpoint(checkpoint, env, contract)
    assert restored.num_timesteps == 32
    assert metadata['stage'] == 'approach'
    np.testing.assert_allclose(model.predict(np.zeros(4, np.float32), deterministic=True)[0],
                               restored.predict(np.zeros(4, np.float32), deterministic=True)[0])
    with pytest.raises(ValueError, match='contract'):
        load_checkpoint(checkpoint, env, {'observation_version': 'wrong'})
    result2, model2 = run_training(env, tmp_path / 'resume', contract, stages=['close'],
        timesteps=16, eval_episodes=2, probe_episodes=3, checkpoint_every=16,
        resume=checkpoint)
    assert model2.num_timesteps == 48
    assert result2['ppo_parameters']['n_steps'] == model2.n_steps == 16
    assert result2['stages'][0]['stage'] == 'close'
    assert list(tmp_path.glob('checkpoint_*_steps.zip'))


def test_unsafe_probe_never_creates_model(tmp_path):
    result, model = run_training(SyntheticEnv(fail=True), tmp_path, {}, stages=['approach'],
        timesteps=16, eval_episodes=2, probe_episodes=3, checkpoint_every=16)
    assert model is None
    assert result['training_skipped'] == 'exploration_not_qualified'
    assert result['ppo_updates'] == 0
    assert not (tmp_path/'ppo_checkpoint.zip').exists()


def test_curriculum_resume_skips_qualified_stage_and_keeps_zero_probe(tmp_path):
    contract = {'test_only': True}
    kwargs = dict(timesteps=16, eval_episodes=2, probe_episodes=3, checkpoint_every=16,
                  ppo_kwargs=dict(n_steps=16, batch_size=8, n_epochs=1))
    run_training(SyntheticEnv(), tmp_path/'first', contract, stages=['approach'], **kwargs)
    result, _ = run_training(SyntheticEnv(), tmp_path/'next', contract,
        stages=['approach', 'close', 'lift'], resume=tmp_path/'first/ppo_checkpoint.zip', **kwargs)
    assert [s['stage'] for s in result['stages']] == ['close', 'lift']
    for stage in result['stages']:
        assert stage['admission']['probes'][0]['controller'] == 'zero'


def test_interrupted_learning_persists_truthful_status_and_updated_checkpoint(tmp_path):
    class InterruptedEnv(SyntheticEnv):
        training_steps = 0
        def step(self, action):
            if self.mode == 'training':
                self.training_steps += 1
                if self.training_steps == 25:
                    raise KeyboardInterrupt()
            return super().step(action)
    with pytest.raises(KeyboardInterrupt):
        run_training(InterruptedEnv(), tmp_path, {}, stages=['approach'], timesteps=64,
            eval_episodes=2, probe_episodes=3, checkpoint_every=16,
            ppo_kwargs=dict(n_steps=16, batch_size=8, n_epochs=1))
    report = json.loads((tmp_path/'training_report.json').read_text())
    assert report['status'] == 'interrupted'
    assert report['ppo_updates'] > 0 and report['ppo_timesteps'] == 24
    assert (tmp_path/'ppo_checkpoint.zip').is_file()


def test_full_admission_requires_lift_access_but_zero_baseline_need_not_succeed():
    assert not admission_report('full', [evidence()] * 3)['allowed']
    assert admission_report('lift', [evidence(), evidence(reached_phases=['approach','close','hold','lift']),
        evidence(reached_phases=['approach','close','hold','lift'])])['allowed']


@pytest.mark.parametrize('stage,random_indices', [('close', range(2,6)), ('lift', range(6,8))])
def test_random_probe_uses_retained_policy_only_on_preceding_phases(stage, random_indices):
    from types import SimpleNamespace
    class PhaseEnv(SyntheticEnv):
        def reset(self, **kwargs):
            self.task = SimpleNamespace(phase='approach')
            self.actions = []
            return super().reset(**kwargs)
        def step(self, action):
            self.actions.append(action.copy())
            result = super().step(action)
            self.task.phase = ('approach','close','hold','lift')[min(self.count//2,3)]
            return result
    class PrefixPolicy:
        def predict(self, obs, deterministic):
            return np.full(12,.25), None
    env = PhaseEnv()
    run_episode(env, PrefixPolicy(), seed=3, random_actions=True, probe_stage=stage)
    for i, action in enumerate(env.actions):
        assert np.all(action == .25) == (i not in random_indices)


@pytest.mark.parametrize('report', [
    {'error': 'failure'}, {'status': 'interrupted'},
    {'training_skipped': 'exploration_not_qualified'},
    {'training_stopped': 'physical_safety_failure'}, {'admission': {'allowed': False}},
    {'evaluation': {'physically_safe': False}},
    {'stages': [{'evaluation': {'physically_safe': False}}]},
    {'approach': {'safety_failure': 'mimic_error'}},
])
def test_all_execution_modes_propagate_unsafe_results(report):
    assert report_exit_code(report) != 0


def test_safe_unsolved_task_is_not_a_process_crash():
    assert report_exit_code({'evaluation': {'physically_safe': True, 'grasp_success_rate': 0}}) == 0


def test_unsafe_post_training_assessment_is_reported_as_physical_stop(tmp_path):
    class UnsafeEvaluation(SyntheticEnv):
        def step(self, action):
            self.fail = self.mode == 'evaluation'
            return super().step(action)
    report, _ = run_training(UnsafeEvaluation(), tmp_path, {}, stages=['approach'],
        timesteps=16, eval_episodes=2, probe_episodes=3, checkpoint_every=16,
        ppo_kwargs=dict(n_steps=16, batch_size=8, n_epochs=1))
    assert report['training_stopped'] == 'physical_safety_failure'
    assert report_exit_code(report) == 2


def test_retraining_stage_invalidates_previous_promotion_until_reassessed(tmp_path):
    kwargs = dict(timesteps=16, eval_episodes=2, probe_episodes=3, checkpoint_every=16,
                  ppo_kwargs=dict(n_steps=16, batch_size=8, n_epochs=1))
    run_training(SyntheticEnv(), tmp_path/'first', {}, stages=['approach'], **kwargs)
    class FailedEvaluation(SyntheticEnv):
        def step(self, action):
            obs, rew, done, truncated, info = super().step(action)
            if self.mode == 'evaluation':
                info['stage_success'] = False
            return obs, rew, done, truncated, info
    report, _ = run_training(FailedEvaluation(), tmp_path/'next', {}, stages=['approach'],
        resume=tmp_path/'first/ppo_checkpoint.zip', **kwargs)
    assert report['completed_stages'] == []


def test_explicit_weight_initialization_drops_old_qualification_and_optimizer_history(tmp_path):
    contract = {'observation_version': 'synthetic', 'action': 'bounded-12d'}
    small = dict(n_steps=16, batch_size=8, n_epochs=1)
    run_training(SyntheticEnv(), tmp_path/'source', contract, stages=['approach'],
        timesteps=32, eval_episodes=2, probe_episodes=3, checkpoint_every=16, ppo_kwargs=small)
    updated = dict(contract, approach_pregrasp_radius_m=.005)
    result, model = run_training(SyntheticEnv(), tmp_path/'transfer', updated, stages=['close'],
        timesteps=16, eval_episodes=2, probe_episodes=3, checkpoint_every=16, ppo_kwargs=small,
        initialize_from=tmp_path/'source/ppo_checkpoint.zip')
    assert model.num_timesteps == 16
    assert result['initialized_from_timesteps'] == 32
    assert result['completed_stages'] == ['close']
    with pytest.raises(ValueError, match='contract'):
        run_training(SyntheticEnv(), tmp_path/'bad', dict(updated, action='changed'), stages=['close'],
            timesteps=16, initialize_from=tmp_path/'source/ppo_checkpoint.zip')


def test_direct_target_training_uses_bounded_policy_and_restores_it(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_policy import BoundedMeanPolicy
    contract=dict(action='12D_direct_joint_target_v1',reset_phase='close')
    report,model=run_training(SyntheticEnv(),tmp_path,contract,stages=['full'],timesteps=16,
        eval_episodes=2,checkpoint_every=16,ppo_kwargs=dict(n_steps=16,batch_size=8,n_epochs=1))
    assert isinstance(model.policy,BoundedMeanPolicy)
    assert report['pipeline_scope']=='close_to_lift'
    restored,_=load_checkpoint(tmp_path/'ppo_checkpoint.zip',SyntheticEnv(),contract)
    assert isinstance(restored.policy,BoundedMeanPolicy)
    np.testing.assert_allclose(model.predict(np.zeros(4),deterministic=True)[0],
                               restored.predict(np.zeros(4),deterministic=True)[0])


def test_object_mode_close_reports_actual_object_success(tmp_path):
    class ObjectSuccessEnv(SyntheticEnv):
        def step(self, action):
            obs, reward, done, truncated, info = super().step(action)
            info['is_success'] = done
            return obs, reward, done, truncated, info
    report, _ = run_training(ObjectSuccessEnv(), tmp_path,
        dict(action='12D_direct_joint_target_v1', reset_phase='close',
             grasp_mode='object_stability'), stages=['close'], timesteps=16,
        eval_episodes=2, checkpoint_every=16,
        ppo_kwargs=dict(n_steps=16, batch_size=8, n_epochs=1))
    assert report['passed'] and report['grasp_qualified']
    assert report['pipeline_scope'] == 'close_to_lift'
