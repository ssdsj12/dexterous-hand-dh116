"""Measured synthetic wiring for adaptive grasp; no Isaac qualification."""
import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_env import TabletopEnv


class AdaptiveBackend:
    dt = 1/480
    decimation = 8

    def __init__(self):
        low, high = active_limits()
        self.grasp = np.tile((low+high)/2, 2)
        self.initial_command = self.grasp-.02
        self.calls = []
        self.bottom = .30
        self.forces = np.zeros(4)
        self.mimic = np.zeros(10)
        self.override = {}

    def reset(self, offset):
        self.offset = offset
        self.q = self.initial_command.copy()
        self.palm = np.array([[.001, -.002, .003], [-.001, .002, -.003]])
        self.stiffness = np.array([10., 20.])
        self.pad_relative = np.arange(12).reshape(4, 3)*.001

    def advance(self, offset, command, **kwargs):
        self.calls.append((offset, command.copy(), kwargs))
        self.offset = offset
        self.q = command.copy()

    def measure(self):
        result = dict(q=self.q.copy(), qd=np.zeros(12), forces=self.forces.copy(),
            clearance=.01, object_relative=np.array([0., 0., -.02]),
            object_quat=np.array([1., 0., 0., 0.]), object_velocity=np.zeros(6),
            bottom=self.bottom, approach_error=self.offset, joint_failure=None,
            mimic_errors=self.mimic.copy(), diagnostics=[], lowest_link='synthetic',
            palm_adjustment=np.zeros(4), adaptive_palm_offset=self.palm.copy(),
            grip_stiffness=self.stiffness.copy(), pad_relative=self.pad_relative.copy())
        for key, value in self.override.items():
            if value is None:
                result.pop(key, None)
            else:
                result[key] = value
        return result


def make_env(backend=None, mode='adaptive_grasp', **kwargs):
    return TabletopEnv(backend or AdaptiveBackend(), stage='full', reset_phase='close',
        control_mode=mode, grasp_mode='object_stability', reset_jitter=0, **kwargs)


def test_adaptive_observation_contains_measured_125d_state_with_log_force_scaling():
    backend = AdaptiveBackend()
    backend.forces[:] = [0., .1, .2, .3]
    env = make_env(backend)
    observation, info = env.reset(seed=1)
    assert env.action_space.shape == (20,)
    assert env.observation_space.shape == observation.shape == (125,)
    np.testing.assert_allclose(observation[48:52], np.log1p(backend.forces/.1), atol=1e-7)
    np.testing.assert_allclose(observation[104:110], (backend.palm/[.003,.003,.004]).ravel())
    np.testing.assert_allclose(observation[110:112], [0., 1.])
    assert observation[112] == 0.
    np.testing.assert_allclose(observation[113:], backend.pad_relative.ravel()/.05, atol=1e-7)
    assert info['phase'] == 'close' and not info['is_success']


def test_every_physics_request_preserves_all_twenty_actions_and_logs_measured_state():
    backend = AdaptiveBackend()
    env = make_env(backend)
    env.reset()
    action = np.linspace(-1., 1., 20)
    _, _, done, _, _ = env.step(action)
    assert not done and len(backend.calls) == 8
    for _, _, kwargs in backend.calls:
        np.testing.assert_allclose(kwargs['palm_action'], action[12:18].reshape(2,3))
        np.testing.assert_allclose(kwargs['grip_action'], action[18:])
    np.testing.assert_allclose(env.log[-1]['residual_action'], action)
    np.testing.assert_allclose(env.log[-1]['adaptive_palm_offset'], backend.palm)
    np.testing.assert_allclose(env.log[-1]['grip_stiffness'], backend.stiffness)
    np.testing.assert_allclose(env.log[-1]['pad_relative'], backend.pad_relative)


@pytest.mark.parametrize('action', [np.zeros(12), np.zeros(19), np.zeros(21),
    np.zeros((2,10)), np.r_[np.zeros(19), np.nan], np.r_[np.zeros(19), 1.01],
    None, ['not-numeric']*20])
def test_bad_adaptive_action_safely_terminates_without_advancing_physics(action):
    env = make_env()
    initial, _ = env.reset()
    obs, reward, done, truncated, info = env.step(action)
    assert done and not truncated and info['safety_failure'] == 'invalid_action'
    assert reward == -25. and np.isfinite(obs).all()
    assert not env.backend.calls
    np.testing.assert_array_equal(obs, initial)


def test_old_joint_target_mode_and_joint_dynamics_remain_compatible():
    legacy = make_env(mode='joint_target')
    adaptive = make_env()
    old_obs, _ = legacy.reset(seed=4)
    new_obs, _ = adaptive.reset(seed=4)
    assert legacy.action_space.shape == (12,) and old_obs.shape == (108,)
    np.testing.assert_allclose(new_obs[:104], old_obs[:104])
    action = np.linspace(-1.,1.,12)
    for _ in range(4):
        _, old_reward, _, _, old_info = legacy.step(action)
        _, new_reward, _, _, new_info = adaptive.step(np.r_[action,np.zeros(8)])
        np.testing.assert_allclose(adaptive.command, legacy.command)
        assert new_reward == old_reward and new_info == old_info
    assert 'palm_action' not in legacy.backend.calls[-1][2]


@pytest.mark.parametrize('field,value', [
    ('adaptive_palm_offset', None), ('grip_stiffness', None), ('pad_relative', None),
    ('adaptive_palm_offset', np.zeros(6)), ('grip_stiffness', np.ones((2,1))),
    ('pad_relative', np.zeros(12)), ('grip_stiffness', np.array([10.,np.nan])),
    ('pad_relative', np.full((4,3),np.nan)),
])
def test_adaptive_reset_never_hides_missing_or_invalid_measurements(field, value):
    backend = AdaptiveBackend()
    backend.override[field] = value
    with pytest.raises(ValueError):
        make_env(backend).reset()


def test_runtime_missing_measurement_returns_finite_failure_transition():
    backend = AdaptiveBackend()
    env = make_env(backend)
    obs, _ = env.reset()
    backend.override['grip_stiffness'] = None
    observation, reward, done, _, info = env.step(np.zeros(20))
    assert done and info['safety_failure'] == 'nonfinite_state'
    assert np.isfinite(observation).all() and np.isfinite(reward)
    np.testing.assert_array_equal(observation, obs)


def test_absolute_height_palm_position_stiffness_and_pad_geometry_are_observable():
    backend = AdaptiveBackend()
    env = make_env(backend)
    initial, _ = env.reset()
    backend.bottom += .015
    backend.palm[1,2] += .001
    backend.stiffness[0] += 5.
    backend.pad_relative[2,0] += .01
    env._measure()
    changed = env.observe()
    np.testing.assert_allclose(initial[:104], changed[:104])
    assert changed[109]-initial[109] == pytest.approx(.25)
    assert changed[110]-initial[110] == pytest.approx(.5)
    assert changed[112]-initial[112] == pytest.approx(.5)
    assert changed[119]-initial[119] == pytest.approx(.2)


def test_adaptive_coupling_warning_preserves_finger_recovery_and_pauses_palm():
    backend = AdaptiveBackend()
    backend.mimic[1] = .005
    env = make_env(backend)
    env.reset()
    previous = env.command.copy()
    env.step(np.ones(20))
    assert all(call[2]['palm_motion_allowed'] is False for call in backend.calls)
    assert np.max(abs(env.command-previous)) <= .05/60+1e-12
    assert env.command[0] > previous[0]  # Other fingers are not frozen.
    assert env.log[-1]['motion_paused'] is True


@pytest.mark.parametrize('reset_phase,grasp_mode', [('approach','object_stability'),('close','four_pad')])
def test_adaptive_mode_rejects_incompatible_reset_or_grasp_gate(reset_phase, grasp_mode):
    with pytest.raises(ValueError):
        TabletopEnv(AdaptiveBackend(), stage='full', control_mode='adaptive_grasp',
            reset_phase=reset_phase, grasp_mode=grasp_mode, reset_jitter=0)


def test_finite_sensor_value_that_overflows_float32_observation_is_rejected():
    backend = AdaptiveBackend()
    backend.override['adaptive_palm_offset'] = np.full((2,3), 1e50)
    with pytest.raises(ValueError):
        make_env(backend).reset()


def test_adaptive_uses_object_stability_success_without_prescribed_pad_contacts():
    backend = AdaptiveBackend()
    backend.bottom = .331
    saved = []
    env = make_env(backend, episode_sink=lambda summary, rows: saved.extend(rows))
    env.reset()
    env.task.phase = 'lift'
    env.task.success_time = .49
    _, reward, done, _, info = env.step(np.zeros(20))
    assert done and info['is_success'] and not info['failure']
    assert np.all(backend.forces == 0.)
    assert reward > 24. and saved[-1]['terms']['success'] == 25.
