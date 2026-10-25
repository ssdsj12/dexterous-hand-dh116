"""Replay orchestration tests; toy transitions are not physical evidence."""
import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_replay import replay_episode


class ToyEnv:
    reset_phase = 'close'
    initial_offset = .04
    reset_jitter = .001

    def __init__(self, *, terminal_at=None, success=False, failure=None,
                 nonfinite_at=None, reset_nonfinite=False, truncated=False):
        self.terminal_at, self.success, self.failure = terminal_at, success, failure
        self.nonfinite_at, self.reset_nonfinite = nonfinite_at, reset_nonfinite
        self.truncated = truncated
        self.actions = []
        self.resets = []
        self.measurement = {'mimic_errors': np.zeros(10)}

    def reset(self, *, seed=None):
        self.resets.append((seed, self.initial_offset, self.reset_jitter))
        self.actions = []
        return np.array([np.nan if self.reset_nonfinite else 0.]), {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        step = len(self.actions)
        self.measurement['mimic_errors'] = np.full(10, -.001*step)
        done = step == self.terminal_at
        obs = np.array([np.nan if step == self.nonfinite_at else 0.])
        return obs, 0., done and not self.truncated, done and self.truncated, dict(
            is_success=self.success, failure=self.failure if done else None,
            safety_failure=self.failure if done and self.failure == 'mimic_error' else None)

    def flush_episode(self):
        raise AssertionError('caller owns flushing')


def rows(count=3):
    return [dict(phase='close' if i == 0 else 'lift', root_offset_m=.005,
                 residual_action=np.full(12, .1*i).tolist(),
                 pad_forces_n=[99.]*4, is_success=True) for i in range(count)]


def test_replay_uses_logged_actions_and_offset_but_not_logged_success_or_contacts():
    env = ToyEnv(success=True)
    source = rows()
    result = replay_episode(env, source, seed=7)
    assert env.resets == [(7, .005, 0.)]
    np.testing.assert_array_equal(env.actions, [r['residual_action'] for r in source])
    assert result['frames_used'] == result['source_frames_used'] == 3
    assert result['source_exhausted'] and result['partial']
    assert not result['is_success'] and result['failure'] is None
    assert result['finite'] and result['safety_failure'] is None
    assert result['max_mimic'] == pytest.approx(.003)


def test_replay_stops_on_measured_physical_failure_before_source_ends():
    env = ToyEnv(terminal_at=2, failure='mimic_error')
    result = replay_episode(env, rows(4))
    assert result['frames_used'] == 2 and not result['source_exhausted']
    assert result['safety_failure'] == result['failure'] == 'mimic_error'
    assert not result['partial'] and not result['is_success']


def test_zero_extension_requires_real_terminal_success_and_stops_there():
    env = ToyEnv(terminal_at=4, success=True)
    result = replay_episode(env, rows(2), extend_zero_steps=9)
    assert result['frames_used'] == 4 and result['source_frames_used'] == 2
    assert result['zero_extension_steps'] == 2 and result['source_exhausted']
    np.testing.assert_array_equal(env.actions[2:], np.zeros((2, 12)))
    assert result['is_success'] and not result['partial']


def test_exhausted_extension_and_truncation_never_claim_grasp_success():
    for env in (ToyEnv(success=True), ToyEnv(terminal_at=3, success=True, truncated=True)):
        result = replay_episode(env, rows(2), extend_zero_steps=1)
        assert result['partial'] and not result['is_success']


@pytest.mark.parametrize('mutate', [
    lambda r: r.clear(),
    lambda r: r[0].update(phase='approach'),
    lambda r: r[1].update(phase='approach'),
    lambda r: r[0].update(root_offset_m=0.),
    lambda r: r[0].update(root_offset_m=np.nan),
    lambda r: r[0].update(root_offset_m=[.005]),
    lambda r: r[1].update(residual_action=[0.]*11),
    lambda r: r[1].update(residual_action=[1.01]*12),
    lambda r: r[1].update(residual_action=[np.nan]*12),
])
def test_invalid_source_is_rejected_before_any_physics_reset(mutate):
    env = ToyEnv()
    source = rows()
    mutate(source)
    with pytest.raises(ValueError):
        replay_episode(env, source)
    assert not env.resets


@pytest.mark.parametrize('extension', [-1, .5, True])
def test_invalid_extension_is_rejected_before_reset(extension):
    env = ToyEnv()
    with pytest.raises(ValueError):
        replay_episode(env, rows(), extend_zero_steps=extension)
    assert not env.resets


def test_approach_environment_cannot_silently_replay_close_trajectory():
    env = ToyEnv()
    env.reset_phase = 'approach'
    with pytest.raises(ValueError, match='close'):
        replay_episode(env, rows())
    assert not env.resets


@pytest.mark.parametrize('env,expected_steps', [
    (ToyEnv(reset_nonfinite=True), 0), (ToyEnv(nonfinite_at=2), 2),
])
def test_nonfinite_measurement_stops_replay_and_cannot_qualify(env, expected_steps):
    result = replay_episode(env, rows(), extend_zero_steps=10)
    assert result['frames_used'] == expected_steps
    assert not result['finite'] and not result['is_success']
    assert result['safety_failure'] == result['failure'] == 'nonfinite_state'
