"""Adapter integration with explicit synthetic measurements, not Isaac evidence."""
import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_env import TabletopEnv
from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_control import prepared_approach_reference


class MeasuredBackend:
    dt = 1/240
    decimation = 4

    def __init__(self, forces=None, clearance=.01):
        low, high = active_limits()
        self.grasp = np.tile((low+high)/2, 2)
        self.forces = np.array([0., 0., .7, 0.] if forces is None else forces)
        self.clearance = clearance
        self.steps = 0

    def reset(self, offset):
        self.offset = offset
        self.q = np.concatenate([prepared_approach_reference(g) for g in self.grasp.reshape(2, 6)])

    def advance(self, offset, command, *, phase, palm_motion_allowed=True):
        self.steps += 1
        self.offset = offset
        self.q = command.copy()

    def measure(self):
        return dict(q=self.q.copy(), qd=np.zeros(12), forces=self.forces.copy(),
            clearance=self.clearance, object_relative=np.array([0, 0, -.02]),
            object_quat=np.array([1, 0, 0, 0]), object_velocity=np.zeros(6),
            bottom=.30, approach_error=abs(self.offset), joint_failure=None,
            diagnostics=[], lowest_link='synthetic')


def test_contact_handoff_reaches_close_without_claiming_grasp():
    backend = MeasuredBackend()
    env = TabletopEnv(backend, stage='approach', initial_offset=.010083, reset_jitter=0)
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    obs, reward, done, truncated, info = env.step(np.zeros(12))
    assert done and not truncated
    assert info['stage_success'] and not info['is_success']
    assert env.task.phase == 'close'
    assert env.offset == pytest.approx(.010083)
    assert backend.steps == 1  # Stop at handoff; do not apply stale approach for three substeps.


def test_clearance_block_never_promotes_and_terminal_cannot_step_again():
    env = TabletopEnv(MeasuredBackend(forces=[0]*4, clearance=.0019), stage='approach', reset_jitter=0)
    env.reset()
    _, _, done, _, info = env.step(np.zeros(12))
    assert done and not info['stage_success']
    assert info['failure'] == 'approach_blocked'
    with pytest.raises(RuntimeError, match='reset'):
        env.step(np.zeros(12))


def test_hold_residual_does_not_integrate_and_phase_control_updates_each_substep():
    env = TabletopEnv(MeasuredBackend(forces=[1]*4), stage='full', reset_jitter=0)
    env.reset()
    for _ in range(80):
        _, _, done, _, info = env.step(np.ones(12))
        assert np.max(np.abs(env.command-env.reference)) <= .100001
        if done:
            break
    assert 'lift' in info['reached_phases']
    assert env.offset > .04  # root lift really began after hold


def test_nonfinite_sensor_returns_finite_terminal_observation():
    backend = MeasuredBackend()
    env = TabletopEnv(backend, reset_jitter=0)
    env.reset()
    backend.forces[0] = np.nan
    obs, reward, done, _, info = env.step(np.zeros(12))
    assert done and np.isfinite(obs).all() and np.isfinite(reward)
    assert info['safety_failure'] == 'nonfinite_state'


def test_full_success_is_not_truncation_and_time_is_observed():
    env = TabletopEnv(MeasuredBackend(forces=[0]*4), stage='full', reset_jitter=0)
    env.reset()
    env.task.elapsed = 14.999
    obs, reward, done, truncated, info = env.step(np.zeros(12))
    assert done and not truncated  # deadline is part of the finite-horizon task
    assert info['failure'] == 'timeout'


def test_partial_training_episode_retains_mode_when_evaluation_resets_scene():
    saved = []
    env = TabletopEnv(MeasuredBackend(forces=[0]*4), stage='close', reset_jitter=0,
                     episode_sink=lambda summary, rows: saved.append(summary))
    env.set_mode('training')
    env.reset()
    env.step(np.zeros(12))
    env.set_mode('evaluation')
    env.reset()
    assert len(saved) == 1
    assert saved[0]['mode'] == 'training' and not saved[0]['complete']


def test_explicit_exploration_radius_bounds_targets_and_is_logged():
    backend = MeasuredBackend(forces=[1]*4)
    env = TabletopEnv(backend, stage='full', reset_jitter=0, residual_scale=.02)
    env.reset()
    for _ in range(70):
        env.step(np.ones(12))
        assert np.max(np.abs(env.command-env.reference)) <= .020001
    assert env.log[-1]['residual_action'] == [1.]*12
    with pytest.raises(ValueError):
        TabletopEnv(backend, residual_scale=.101)


def test_residual_change_rate_is_bounded_and_filter_state_is_observed():
    env = TabletopEnv(MeasuredBackend(forces=[0]*4), stage='full', reset_jitter=0)
    env.reset()
    old = np.zeros(12)
    for i in range(12):
        obs, _, _, _, _ = env.step(np.full(12, 1. if i < 6 else -1.))
        current = env.filtered_residual
        assert np.max(np.abs(current-old)) <= .05/60 + 1e-12
        np.testing.assert_allclose(obs[-12:], current/.1, atol=1e-7)
        old = current.copy()


def test_close_reacquires_when_latched_contact_has_disappeared():
    backend = MeasuredBackend(forces=[0]*4)
    env = TabletopEnv(backend, stage='close', initial_offset=.01, reset_jitter=0)
    env.reset()
    env.task.phase = 'close'
    env.descent_stop = 'pad_contact'
    env.reference = backend.grasp.copy()
    env.command = env.reference.copy()
    backend.q = env.reference.copy()
    env._measure()
    env.step(np.zeros(12))
    assert .0099 < env.offset < .01
    assert env.descent_stop is None


def test_palm_waits_for_actual_joint_tracking():
    backend = MeasuredBackend(forces=[0]*4)
    env = TabletopEnv(backend, stage='close', initial_offset=.02, reset_jitter=0)
    env.reset()
    env.command = env.command + .08
    env.reference = env.command.copy()
    backend.decimation = 1
    env.step(np.zeros(12))
    assert env.offset == pytest.approx(.02)


def test_coupling_warning_pauses_motion_without_relaxing_hard_failure():
    backend = MeasuredBackend(forces=[0]*4)
    original = backend.measure
    def measured():
        return dict(original(), mimic_errors=np.array([-.012]+[0.]*9))
    backend.measure = measured
    env = TabletopEnv(backend, stage='full', initial_offset=.02, reset_jitter=0)
    obs, _ = env.reset()
    command = env.command.copy()
    _, _, done, _, info = env.step(np.ones(12))
    assert not done
    assert env.offset == pytest.approx(.02)
    np.testing.assert_allclose(env.command, command)
    assert env.log[-1]['motion_paused'] is True
    assert obs.shape == (104,)
    assert obs[-22] == pytest.approx(-.6)


def test_reset_uses_near_grasp_preparation_before_palm_descent():
    backend = MeasuredBackend(forces=[0]*4)
    env = TabletopEnv(backend, stage='full', reset_jitter=0)
    env.reset()
    assert np.max(np.abs(env.reference-backend.grasp)) <= .040001
    np.testing.assert_allclose(env.command, env.reference)


def test_coupling_pause_is_explicitly_forwarded_to_backend_palm_tracking():
    backend = MeasuredBackend(forces=[0]*4)
    original_measure, original_advance = backend.measure, backend.advance
    backend.measure = lambda: dict(original_measure(), mimic_errors=np.array([.012]+[0.]*9))
    permissions = []
    def advance(offset, command, *, phase, palm_motion_allowed=True):
        permissions.append(palm_motion_allowed)
        original_advance(offset, command, phase=phase)
    backend.advance = advance
    env = TabletopEnv(backend, stage='full', reset_jitter=0)
    env.reset()
    env.step(np.zeros(12))
    assert permissions and not any(permissions)


def test_pregrasp_handoff_precedes_clearance_boundary_with_no_fake_contact():
    backend = MeasuredBackend(forces=[0]*4, clearance=.004)
    env = TabletopEnv(backend, stage='close', initial_offset=.0048, reset_jitter=0)
    env.reset()
    _, _, done, _, info = env.step(np.zeros(12))
    assert env.task.phase == 'close'
    assert not done and not info['is_success'] and not info['stage_success']
    assert env.metrics['pad_forces_n'].tolist() == [0.]*4


@pytest.mark.parametrize('new_problem', ['coupling', 'tracking'])
def test_pregrasp_handoff_uses_readiness_after_physics_step(new_problem):
    backend = MeasuredBackend(forces=[0]*4, clearance=.004)
    backend.decimation = 1
    original_advance, original_measure = backend.advance, backend.measure
    def advance(offset, command, **kwargs):
        original_advance(offset, command, **kwargs)
        if new_problem == 'tracking':
            backend.q[0] += .05
    def measure():
        errors = np.zeros(10)
        if backend.steps and new_problem == 'coupling':
            errors[0] = .012
        return dict(original_measure(), mimic_errors=errors)
    backend.advance, backend.measure = advance, measure
    env = TabletopEnv(backend, stage='close', initial_offset=.0048, reset_jitter=0)
    env.reset()
    env.step(np.zeros(12))
    assert env.task.phase == 'approach'


def test_direct_close_reset_has_no_approach_credit_or_fake_contact():
    backend = MeasuredBackend(forces=[0]*4)
    backend.initial_command = backend.grasp.copy()
    env = TabletopEnv(backend, stage='close', initial_offset=.005, reset_jitter=0,
                     reset_phase='close', control_mode='joint_target')
    obs, info = env.reset()
    assert env.task.phase == 'close' and env.reached_phases == ['close']
    assert env.task.contact_time == 0 and not info['stage_success']
    assert obs.shape == (108,)
    command = env.command.copy()
    env.step(-np.ones(12))
    assert np.all(env.command <= command)
    assert np.max(np.abs(env.command-command)) <= .05/60+1e-8
    np.testing.assert_allclose(env.reference, backend.grasp)


def test_direct_close_cannot_claim_approach_training():
    with pytest.raises(ValueError):
        TabletopEnv(MeasuredBackend(), stage='approach', reset_phase='close', control_mode='joint_target')


def test_direct_target_away_from_center_does_not_falsely_block_palm_tracking():
    backend=MeasuredBackend(forces=[0]*4)
    backend.initial_command=backend.grasp+.04
    permissions=[]
    advance=backend.advance
    def checked(offset, command, **kwargs):
        permissions.append(kwargs['palm_motion_allowed']);advance(offset,command,**kwargs)
    backend.advance=checked
    env=TabletopEnv(backend,stage='close',initial_offset=.005,reset_jitter=0,
                   reset_phase='close',control_mode='joint_target')
    env.reset();backend.q=env.command.copy();env._measure()
    env.step(np.full(12,.5))
    assert permissions and all(permissions)


def test_direct_success_discounted_return_exceeds_subthreshold_timeout():
    def outcome(force):
        backend=MeasuredBackend(forces=[force]*4)
        backend.initial_command=backend.grasp.copy()
        env=TabletopEnv(backend,stage='close',initial_offset=.005,reset_jitter=0,
                       reset_phase='close',control_mode='joint_target')
        env.reset();total=0.
        for i in range(901):
            _,reward,done,_,info=env.step(np.zeros(12));total += .995**i*reward
            if done: return total,info['stage_success']
    failed,f=outcome(.099);success,s=outcome(.1)
    assert s and not f
    assert success > failed


def test_direct_warning_unloads_only_affected_master_and_keeps_other_fingers_live():
    backend=MeasuredBackend(forces=[0]*4)
    backend.initial_command=backend.grasp.copy()
    measured=backend.measure
    backend.measure=lambda:dict(measured(),mimic_errors=np.array([-.012]+[0.]*9))
    env=TabletopEnv(backend,stage='close',initial_offset=.005,reset_jitter=0,
                   reset_phase='close',control_mode='joint_target')
    env.reset();backend.q=env.command.copy();backend.q[1]-=.01;env._measure()
    old=env.command.copy();backend.decimation=1
    env.step(np.ones(12))
    assert env.command[1] < old[1]  # relieve negative follower error by opening master
    assert env.command[2] > old[2]  # another finger can still explore
    assert not env.task.failure


def test_object_still_on_table_is_not_charged_postgrasp_slip():
    backend=MeasuredBackend(forces=[0]*4)
    backend.initial_command=backend.grasp.copy()
    env=TabletopEnv(backend,stage='full',initial_offset=.005,reset_jitter=0,
        reset_phase='close',control_mode='joint_target',grasp_mode='object_stability')
    env.reset();env.task.phase='lift';env.grip_relative=np.array([.05,0,-.02]);env._measure()
    env.step(np.zeros(12))
    assert env.log[-1]['terms']['slip']==0


def test_object_success_allowed_slip_is_not_penalized():
    backend=MeasuredBackend(forces=[0]*4);backend.initial_command=backend.grasp.copy()
    original=backend.measure
    backend.measure=lambda:dict(original(),bottom=.335)
    env=TabletopEnv(backend,stage='full',initial_offset=.005,reset_jitter=0,
        reset_phase='close',control_mode='joint_target',grasp_mode='object_stability')
    env.reset();env.task.phase='lift';env.grip_relative=np.array([.015,0,-.02]);env._measure()
    env.step(np.zeros(12))
    assert env.log[-1]['terms']['slip']==0


def test_direct_warning_reacts_before_previous_lift_instability_threshold():
    backend=MeasuredBackend(forces=[0]*4);backend.initial_command=backend.grasp.copy()
    original=backend.measure
    backend.measure=lambda:dict(original(),mimic_errors=np.array([-.005]+[0.]*9))
    env=TabletopEnv(backend,stage='full',initial_offset=.005,reset_jitter=0,
        reset_phase='close',control_mode='joint_target',grasp_mode='object_stability')
    env.reset();env.task.phase='lift';offset=env.offset
    env.step(np.zeros(12))
    assert env.offset==offset


def test_object_pickup_with_allowed_slip_beats_no_pickup_without_four_contacts():
    def outcome(pickup):
        backend = MeasuredBackend(forces=[0]*4)
        backend.initial_command = backend.grasp.copy()
        measured = backend.measure
        def measurement():
            t = backend.steps * backend.dt
            return dict(measured(),
                bottom=.30 + (min(.033, max(0., t-2.3)*.005) if pickup else 0.),
                object_relative=np.array([.015 if t > 2.4 else 0., 0., -.02]))
        backend.measure = measurement
        env = TabletopEnv(backend, stage='full', initial_offset=.005, reset_jitter=0,
            reset_phase='close', control_mode='joint_target', grasp_mode='object_stability')
        env.reset()
        total = 0.
        for i in range(902):
            _, reward, done, _, info = env.step(np.zeros(12))
            total += .995**i * reward
            if done:
                return total, info
        pytest.fail('finite-horizon episode did not end')
    success_return, success = outcome(True)
    failure_return, failure = outcome(False)
    assert success['is_success'] and success['stage_success']
    assert not failure['is_success'] and failure['failure'] == 'timeout'
    assert success_return > failure_return


def test_summary_only_trace_preserves_episode_minimum_and_return():
    saved = []
    backend = MeasuredBackend(forces=[0]*4, clearance=.01)
    env = TabletopEnv(backend,stage='full',reset_jitter=0,trace_steps=False,
        episode_sink=lambda summary,rows: saved.append((summary,rows.copy())))
    env.reset()
    env.step(np.zeros(12))
    backend.clearance=.009
    env.step(np.zeros(12))
    backend.clearance=.012
    env.step(np.zeros(12))
    env.flush_episode()
    assert saved[0][0]['min_clearance_m'] == pytest.approx(.009)
    assert saved[0][0]['return_'] == pytest.approx(env.return_)
    assert len(saved[0][1]) == 1
