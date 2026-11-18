"""Inspect real event files: live simulation metrics and PPO optimizer logs."""
import numpy as np
import json
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from test_dh116_tabletop_vec_training import ToyEnv
from go2_pvcnn.control.dh116.tabletop_vec_training import run_vector_training


def events(path):
    accumulator=EventAccumulator(str(path));accumulator.Reload()
    return {tag:accumulator.Scalars(tag) for tag in accumulator.Tags()['scalars']}


class MeasuredToy(ToyEnv):
    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        info['force_feedback']={'measured_hand_load_n':[.2,.3]}
        return obs,reward,done,truncated,info


def test_live_events_and_final_ppo_loss_survive_one_update(tmp_path):
    env=DummyVecEnv([lambda:MeasuredToy(length=3),lambda:MeasuredToy(length=5)])
    report,model=run_vector_training(env,tmp_path,dict(num_envs=2),direct_training=True,
        timesteps=16,eval_episodes=2,ppo_kwargs=dict(n_steps=8,batch_size=8,n_epochs=1))
    assert model._n_updates==1
    live=events(tmp_path/'ppo'/'live');training=events(tmp_path/'ppo')
    assert live['live/reward_mean']
    assert live['live/hand_load_left_n'][-1].value==np.float32(.2)
    assert live['live/hand_load_right_n'][-1].value==np.float32(.3)
    assert live['live/phase_lift_fraction'][-1].value==1.
    assert live['live/completed_episodes_total'][-1].value==report['completed_training_episodes']
    assert live['live/success_rate'][-1].value==1.
    assert training['train/loss'][-1].step==16
    assert all(np.isfinite(e.value) for values in live.values() for e in values)
    env.close()


def test_safety_stop_flushes_live_finite_flags_without_false_success_rate(tmp_path):
    class BadMeasurement(MeasuredToy):
        def step(self,action):
            obs,reward,done,truncated,info=super().step(action)
            if self.mode=='training':
                reward=float('nan');info['force_feedback']['measured_hand_load_n'][0]=float('nan')
            return obs,reward,done,truncated,info
    env=DummyVecEnv([lambda:BadMeasurement(length=99)])
    report,_=run_vector_training(env,tmp_path,{},direct_training=True,timesteps=8,
        eval_episodes=2,ppo_kwargs=dict(n_steps=8,batch_size=8,n_epochs=1))
    live=events(tmp_path/'ppo'/'live')
    assert report['safety_failure']=='nonfinite_state'
    assert live['live/safety_stop'][-1].value==1.
    assert live['live/rewards_finite_fraction'][-1].value==0.
    assert live['live/hand_load_finite_fraction'][-1].value==.5
    assert 'live/success_rate' not in live
    assert 'live/reward_mean' not in live
    assert all(np.isfinite(e.value) for values in live.values() for e in values)
    env.close()


@pytest.mark.parametrize('failure_kind', ['physical', 'exception'])
def test_logger_close_failures_preserve_stop_and_checkpoint(tmp_path, monkeypatch, failure_kind):
    from stable_baselines3.common.logger import Logger
    from go2_pvcnn.control.dh116.tabletop_tensorboard import LiveTrainingWriter
    from go2_pvcnn.control.dh116.tabletop_protected_vec_env import UnrecoverablePhysicsError
    original_error = (UnrecoverablePhysicsError('joint_velocity', {}, 0)
                      if failure_kind == 'physical' else RuntimeError('original training error'))
    class StopsAfterUpdate(MeasuredToy):
        def step(self, action):
            if self.mode == 'training' and self.count == 9: raise original_error
            return super().step(action)
    closed = []
    original_live_close = LiveTrainingWriter.close
    original_logger_close = Logger.close
    def live_close(self):
        original_live_close(self); closed.append('live')
        raise OSError('injected live close failure')
    def logger_close(self):
        original_logger_close(self); closed.append('logger')
        raise OSError('injected logger close failure')
    monkeypatch.setattr(LiveTrainingWriter, 'close', live_close)
    monkeypatch.setattr(Logger, 'close', logger_close)
    env = DummyVecEnv([lambda: StopsAfterUpdate(length=99)])
    kwargs = dict(direct_training=True, timesteps=32, eval_episodes=2, checkpoint_every=1000,
                  ppo_kwargs=dict(n_steps=8, batch_size=8, n_epochs=1))
    try:
        if failure_kind == 'physical':
            report, model = run_vector_training(env, tmp_path, {}, **kwargs)
            assert report['status'] == 'stopped' and report['safety_failure'] == 'joint_velocity'
            assert model._n_updates == 1
        else:
            with pytest.raises(RuntimeError, match='original training error') as raised:
                run_vector_training(env, tmp_path, {}, **kwargs)
            assert raised.value is original_error
        assert closed == ['live', 'logger']
        assert (tmp_path / 'ppo_checkpoint.zip').exists()
        report = json.loads((tmp_path / 'training_report.json').read_text())
        monitor = json.loads((tmp_path / 'monitor.json').read_text())
        assert report['ppo_updates'] == 1
        assert report['telemetry_status'] == monitor['telemetry_status'] == 'degraded'
        assert len(report['logging_errors']) == len(monitor['logging_errors']) == 2
        assert 'live close failure' in report['logging_errors'][0]['error']
        assert 'logger close failure' in report['logging_errors'][1]['error']
    finally:
        env.close()
