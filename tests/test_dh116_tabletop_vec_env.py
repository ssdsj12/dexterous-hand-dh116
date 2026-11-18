"""Shared-world scheduling tests; no Isaac physics evidence."""
import gymnasium as gym
import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_control import prepared_approach_reference
from go2_pvcnn.control.dh116.tabletop_env import TabletopEnv


class World:
    def __init__(self, count=2):
        self.ticks = 0
        self.events = []
        self.backends = [Backend(self, i) for i in range(count)]

    def advance_all(self, requests):
        for backend, request in zip(self.backends, requests):
            if request is not None:
                offset, command, kwargs = request
                assert set(kwargs) == {'phase', 'palm_motion_allowed'}
                backend.offset, backend.q = offset, command.copy()
                self.events.append(('command', self.ticks, backend.index))
        self.events.append(('physics', self.ticks))
        self.ticks += 1


class Backend:
    dt = 1 / 240
    decimation = 4

    def __init__(self, world, index):
        self.world, self.index = world, index
        low, high = active_limits()
        self.grasp = np.tile((low + high) / 2, 2)
        self.clearance = .01
        self.resets = 0
        self.measure_ticks = []

    def reset(self, offset):
        self.resets += 1
        self.world.events.append(('reset', self.world.ticks, self.index))
        self.offset = offset
        self.q = np.concatenate([prepared_approach_reference(g) for g in self.grasp.reshape(2, 6)])
        if self.resets > 1:
            self.clearance = .01

    def advance(self, *args, **kwargs):
        raise AssertionError('vector stepping must not advance each backend separately')

    def measure(self):
        self.measure_ticks.append(self.world.ticks)
        return dict(q=self.q.copy(), qd=np.zeros(12), forces=np.zeros(4),
                    clearance=self.clearance, object_relative=np.array([0., 0., -.02]),
                    object_quat=np.array([1., 0., 0., 0.]), object_velocity=np.zeros(6),
                    bottom=.30, approach_error=abs(self.offset), joint_failure=None,
                    diagnostics=[], lowest_link='synthetic')


def make_vec(count=2, **env_kwargs):
    from go2_pvcnn.control.dh116.tabletop_vec_env import SharedPhysicsVecEnv
    world = World(count)
    envs = [TabletopEnv(backend, stage='full', reset_jitter=0, **env_kwargs)
            for backend in world.backends]
    return world, SharedPhysicsVecEnv(envs, world.advance_all)


def test_step_iterator_defers_physics_and_returns_single_transition():
    assert hasattr(TabletopEnv, 'step_iter'), 'shared physics requires cooperative stepping'
    world = World(1)
    env = TabletopEnv(world.backends[0], stage='full', reset_jitter=0)
    env.reset()
    iterator = env.step_iter(np.zeros(12))
    for _ in range(4):
        request = next(iterator)
        assert len(request) == 3
        world.advance_all([request])
    with pytest.raises(StopIteration) as ended:
        next(iterator)
    obs, reward, terminated, truncated, info = ended.value.value
    assert obs.shape == (104,) and np.isfinite(reward)
    assert not terminated and not truncated
    assert info['phase'] == 'approach'
    assert world.ticks == 4


def test_all_commands_precede_one_shared_step_and_measurements_follow():
    world, vec = make_vec()
    obs = vec.reset()
    assert obs.shape == (2, 104) and world.ticks == 0
    world.events.clear()
    obs, rewards, dones, infos = vec.step(np.stack([np.ones(12), -np.ones(12)]))
    assert world.ticks == 4 and not dones.any()
    assert rewards.shape == (2,) and obs.shape == (2, 104)
    assert world.events == [event for tick in range(4) for event in
                            [('command', tick, 0), ('command', tick, 1), ('physics', tick)]]
    assert all(b.measure_ticks == [0, 1, 2, 3, 4] for b in world.backends)
    assert not np.array_equal(world.backends[0].q, world.backends[1].q)


def test_early_terminal_is_preserved_and_only_done_slot_resets_after_shared_tick():
    world, vec = make_vec()
    vec.reset()
    world.backends[0].clearance = .0019
    # The initial measurement is the observation from which this action starts.
    vec.envs[0]._measure()
    obs, rewards, dones, infos = vec.step(np.zeros((2, 12)))
    assert dones.tolist() == [True, False]
    assert infos[0]['failure'] == 'approach_blocked'
    assert infos[0]['terminal_observation'].shape == (104,)
    assert not np.array_equal(obs[0], infos[0]['terminal_observation'])
    assert world.backends[0].resets == 2 and world.backends[1].resets == 1
    assert [event for event in world.events if event[0] == 'reset'][-1] == ('reset', 4, 0)
    assert [event for event in world.events if event[0] == 'command' and event[2] == 0] == [('command', 0, 0)]
    assert world.backends[0].measure_ticks == [0, 0, 1, 4]
    assert infos[0]['TimeLimit.truncated'] is False
    assert vec.envs[1].task.elapsed == pytest.approx(4 / 240)


def test_all_early_done_still_get_one_full_shared_control_tick():
    world, vec = make_vec()
    vec.reset()
    for env in vec.envs:
        env.task.elapsed = 14.999
    _, _, dones, infos = vec.step(np.zeros((2, 12)))
    assert dones.all() and world.ticks == 4
    assert all(info['failure'] == 'timeout' for info in infos)
    assert len([event for event in world.events if event[0] == 'command']) == 2


def test_seed_options_attributes_and_pending_action_contract():
    world, vec = make_vec()
    assert vec.seed(42) == [42, 43]
    vec.set_options({'example': True})
    vec.reset()
    assert vec._seeds == [None, None] and vec._options == [{}, {}]
    vec.env_method('set_mode', 'training', indices=1)
    assert vec.get_attr('mode') == ['evaluation', 'training']
    vec.set_attr('mode', 'probe', indices=[0])
    assert vec.get_attr('mode', indices=0) == ['probe']
    assert vec.env_is_wrapped(gym.Wrapper) == [False, False]
    with pytest.raises(RuntimeError, match='pending'):
        vec.step_wait()
    with pytest.raises(ValueError, match='shape'):
        vec.step_async(np.zeros(12))
    vec.step_async(np.zeros((2, 12)))
    with pytest.raises(RuntimeError, match='pending'):
        vec.step_async(np.zeros((2, 12)))
    vec.reset()  # Cancel work that has not touched the world.
    assert world.ticks == 0
    with pytest.raises(RuntimeError, match='pending'):
        vec.step_wait()


def test_mismatched_physics_clocks_and_duplicate_env_are_rejected():
    from go2_pvcnn.control.dh116.tabletop_vec_env import SharedPhysicsVecEnv
    world, vec = make_vec()
    with pytest.raises(ValueError, match='distinct'):
        SharedPhysicsVecEnv([vec.envs[0], vec.envs[0]], world.advance_all)
    world.backends[1].decimation = 2
    with pytest.raises(ValueError, match='clock'):
        SharedPhysicsVecEnv(vec.envs, world.advance_all)


def test_real_sb3_collects_multi_env_rollout_with_single_shared_clock():
    from stable_baselines3 import PPO
    world, vec = make_vec()
    model = PPO('MlpPolicy', vec, n_steps=8, batch_size=8, n_epochs=1,
                policy_kwargs={'net_arch': [16]}, seed=7, device='cpu')
    model.learn(total_timesteps=16)
    assert model.num_timesteps == 16
    assert world.ticks == 8 * 4
    assert model.rollout_buffer.observations.shape == (16, 104)


def test_physical_monitoring_preserves_terminal_metrics_before_auto_reset():
    world, vec = make_vec()
    backend = world.backends[0]
    original_measure = backend.measure

    def measured():
        result = original_measure()
        if backend.resets == 1:
            result.update(bottom=.309, forces=np.array([1., 2., 3., 4.]),
                          object_relative=np.array([world.ticks * .001, 0., -.02]),
                          mimic_errors=np.array([.001, -.003] + [0.] * 8))
        return result

    backend.measure = measured
    vec.reset()
    vec.envs[0].grip_relative = np.array([0., 0., -.02])
    backend.clearance = .0019
    vec.envs[0]._measure()
    _, _, dones, infos = vec.step(np.zeros((2, 12)))
    assert dones.tolist() == [True, False]
    assert infos[0]['physical'] == pytest.approx(dict(
        object_bottom_m=.309, slip_m=.001, max_force_n=4., max_mimic_error_rad=.003))
    assert infos[1]['physical'] == pytest.approx(dict(
        object_bottom_m=.30, slip_m=0., max_force_n=0., max_mimic_error_rad=0.))
    assert vec.envs[0].metrics['object_bottom_m'] == pytest.approx(.30)
    assert vec.envs[0].metrics['max_force_n'] == 0.
    assert backend.resets == 2
