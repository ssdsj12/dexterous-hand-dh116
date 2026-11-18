"""Offline contracts for measured tabletop control; no simulator is started."""
from dataclasses import replace

import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
from go2_pvcnn.control.dh116.tabletop_task import TabletopTask


def metrics(**overrides):
    values = dict(approach_error_m=.01008, pad_forces_n=np.zeros(4),
                  clearance_m=.004, object_bottom_m=.30,
                  object_speed_m_s=0., slip_m=0., max_force_n=1.)
    values.update(overrides)
    return values


def test_measured_pad_contact_finishes_real_10mm_approach_deadlock():
    task = TabletopTask()
    assert not task.update(.01, **metrics(pad_forces_n=[.1, 0., 0., 0.]),
                           approach_contact_ready=True)
    assert task.phase == 'close'
    assert not task.success


@pytest.mark.parametrize('forces,clearance,ready', [
    ([0., 0., 0., 0.], .004, True),
    ([.09, 0., 0., 0.], .004, True),
    ([.1, 0., 0., 0.], .0019, True),
    ([.1, 0., 0., 0.], .004, False),
])
def test_contact_latch_alone_cannot_finish_approach(forces, clearance, ready):
    task = TabletopTask()
    task.update(.01, **metrics(pad_forces_n=forces, clearance_m=clearance),
                approach_contact_ready=ready)
    assert task.phase == 'approach'
    assert not task.success


def test_clearance_blocker_wins_over_contact_and_original_approach_goal():
    task = TabletopTask()
    task.update(.01, **metrics(approach_error_m=0., pad_forces_n=np.ones(4)),
                approach_contact_ready=True, approach_blocked=True)
    assert task.failure == 'approach_blocked'
    assert task.phase == 'approach'
    assert not task.success


@pytest.mark.parametrize('overrides,failure', [
    ({'clearance_m': -.0011}, 'table_penetration'),
    ({'max_force_n': 30.1}, 'excess_force'),
    ({'object_bottom_m': .2899}, 'object_dropped'),
])
def test_contact_stop_preserves_original_physical_safety(overrides, failure):
    task = TabletopTask()
    task.update(.01, **metrics(pad_forces_n=np.ones(4), **overrides),
                approach_contact_ready=True)
    assert task.failure == failure
    assert not task.success


@pytest.mark.parametrize('forces', [[.1], [1., 1., 1., float('nan')], [1.]*5,
                                    np.ones((2, 2))])
def test_invalid_measured_contact_cannot_finish_approach(forces):
    task = TabletopTask()
    task.update(.01, **metrics(pad_forces_n=forces), approach_contact_ready=True)
    assert task.failure == 'nonfinite_state'
    assert task.phase == 'approach'


def test_contact_transition_preserves_four_pad_continuous_hold_and_lift_gate():
    task = TabletopTask()
    task.update(.01, **metrics(pad_forces_n=[.1, 0., 0., 0.]),
                approach_contact_ready=True)
    for _ in range(50):
        task.update(.01, **metrics(pad_forces_n=[1., 1., 1., 0.]))
    assert task.phase == 'close'
    for _ in range(80):
        task.update(.01, **metrics(pad_forces_n=np.ones(4)))
    assert task.phase == 'lift'
    for _ in range(60):
        task.update(.01, **metrics(pad_forces_n=np.ones(4), object_bottom_m=.329))
    assert not task.success
    for _ in range(50):
        task.update(.01, **metrics(pad_forces_n=np.ones(4), object_bottom_m=.33))
    assert task.success


def nominal():
    low, high = active_limits()
    return np.tile((low+high)/2., 2)


def test_repeated_residual_targets_do_not_accumulate_on_independent_nominal():
    from go2_pvcnn.control.dh116.tabletop_control import bounded_residual_target
    reference = nominal()
    residual = np.ones(12)
    expected = reference+.1
    for _ in range(100):
        target = bounded_residual_target(reference, residual)
        assert np.allclose(target, expected)
    assert np.array_equal(reference, nominal())
    assert np.allclose(bounded_residual_target(reference, np.zeros(12)), reference)


def test_residual_targets_respect_both_hands_coupled_limits():
    from go2_pvcnn.control.dh116.tabletop_control import bounded_residual_target
    low, high = active_limits()
    for reference, residual, expected in [(np.tile(low, 2), -np.ones(12), low),
                                          (np.tile(high, 2), np.ones(12), high)]:
        target = bounded_residual_target(reference, residual)
        assert np.allclose(target, np.tile(expected, 2))
        expand(target[:6])
        expand(target[6:])


@pytest.mark.parametrize('field,value', [
    ('reference', np.zeros(11)), ('reference', np.full(12, np.nan)),
    ('residual', np.zeros((2, 6))), ('residual', np.full(12, np.inf)),
    ('residual', np.full(12, 1.01)), ('scale', -.1),
    ('scale', np.nan), ('scale', .1001),
])
def test_residual_target_rejects_invalid_or_unbounded_commands(field, value):
    from go2_pvcnn.control.dh116.tabletop_control import bounded_residual_target
    kwargs = dict(reference=nominal(), residual=np.zeros(12))
    kwargs[field] = value
    with pytest.raises(ValueError):
        bounded_residual_target(**kwargs)


@pytest.mark.parametrize('stage,phase,success,expected', [
    ('approach', 'approach', False, False), ('approach', 'close', False, True),
    ('close', 'hold', False, False), ('close', 'lift', False, True),
    ('lift', 'lift', False, False), ('full', 'lift', False, False),
    ('lift', 'lift', True, True), ('full', 'lift', True, True),
])
def test_curriculum_completion_does_not_fabricate_grasp_success(stage, phase, success, expected):
    from go2_pvcnn.control.dh116.tabletop_control import curriculum_complete
    task = TabletopTask(phase=phase, success=success)
    assert curriculum_complete(stage, task) is expected
    task.failure = 'excess_force'
    assert not curriculum_complete(stage, task)


def observation_inputs():
    return dict(q=nominal(), qd=np.zeros(12), previous=nominal(), reference=nominal(),
                forces=np.zeros(4), task=TabletopTask(), offset=.04,
                close_progress=0., descent_stop=None, clearance=.004,
                object_relative=np.array([0., 0., -.01]), object_quat=np.array([1., 0., 0., 0.]),
                object_velocity=np.zeros(6), grip_relative=None)


def test_observation_exposes_nominal_reference_latch_timers_and_grip_capture():
    from go2_pvcnn.control.dh116.tabletop_control import (
        OBSERVATION_SIZE, OBSERVATION_VERSION, build_observation)
    kwargs = observation_inputs()
    baseline = build_observation(**kwargs)
    assert OBSERVATION_SIZE == 104
    assert isinstance(OBSERVATION_VERSION, str) and OBSERVATION_VERSION
    assert baseline.shape == (OBSERVATION_SIZE,)
    assert baseline.dtype == np.float32 and np.isfinite(baseline).all()
    changes = [dict(reference=nominal()+.01), dict(descent_stop='pad_contact'),
               dict(descent_stop='table_margin'), dict(task=replace(kwargs['task'], elapsed=1.)),
               dict(task=replace(kwargs['task'], contact_time=.1)),
               dict(task=replace(kwargs['task'], success_time=.1)),
               dict(grip_relative=np.zeros(3)), dict(clearance=.002),
               dict(object_relative=np.array([.01, 0., -.01])),
               dict(object_velocity=np.ones(6)*.01)]
    for change in changes:
        observed = build_observation(**(kwargs | change))
        assert not np.array_equal(observed, baseline), change


@pytest.mark.parametrize('field,value', [
    ('q', np.zeros(11)), ('qd', np.full(12, np.nan)),
    ('previous', np.zeros((2, 6))), ('reference', np.full(12, np.inf)),
    ('forces', np.zeros(3)), ('object_relative', [0., 0.]),
    ('object_quat', np.zeros(3)), ('object_velocity', np.zeros(3)),
    ('grip_relative', np.full(3, np.nan)), ('clearance', np.nan),
    ('close_progress', np.inf), ('descent_stop', 'unknown'),
    ('task', TabletopTask(elapsed=np.nan)), ('task', TabletopTask(phase='unknown')),
])
def test_observation_rejects_invalid_and_nonfinite_inputs(field, value):
    from go2_pvcnn.control.dh116.tabletop_control import build_observation
    with pytest.raises(ValueError):
        build_observation(**(observation_inputs() | {field: value}))


def reward_inputs(**overrides):
    values = dict(phase='approach', previous_metrics=metrics(), metrics=metrics(),
                  action=nominal(), reference=nominal(), previous=nominal(),
                  dt=.02, success_event=False, failed=False)
    values.update(overrides)
    return values


@pytest.mark.parametrize('phase', ['approach', 'close', 'hold', 'lift'])
def test_progress_reward_cannot_reward_idle_tracking_or_existing_contact(phase):
    from go2_pvcnn.control.dh116.tabletop_control import progress_reward
    measured = metrics(pad_forces_n=np.ones(4), object_bottom_m=.34)
    reward, terms = progress_reward(**reward_inputs(phase=phase,
        previous_metrics=measured, metrics=measured))
    assert reward < 0.
    assert terms['mpc_tracking'] <= 0.
    assert terms['approach'] == terms['contact'] == terms['lift'] == 0.
    assert terms['time'] == pytest.approx(-.1*.02)


def test_progress_reward_rewards_measured_approach_and_contact_changes():
    from go2_pvcnn.control.dh116.tabletop_control import progress_reward
    _, approach = progress_reward(**reward_inputs(metrics=metrics(approach_error_m=.009)))
    assert approach['approach'] > 0.
    _, close = progress_reward(**reward_inputs(phase='close',
        metrics=metrics(pad_forces_n=np.ones(4)*.1)))
    assert close['contact'] > 0.


@pytest.mark.parametrize('phase,forces,expected_positive', [
    ('approach', [1.]*4, False), ('close', [1.]*4, False),
    ('lift', [1., 1., 1., 0.], False), ('lift', [1.]*4, True),
])
def test_lift_progress_requires_lift_phase_and_measured_four_pad_grasp(phase, forces, expected_positive):
    from go2_pvcnn.control.dh116.tabletop_control import progress_reward
    _, terms = progress_reward(**reward_inputs(phase=phase,
        previous_metrics=metrics(pad_forces_n=forces, object_bottom_m=.31),
        metrics=metrics(pad_forces_n=forces, object_bottom_m=.32)))
    assert (terms['lift'] > 0.) is expected_positive


def test_stage_bonus_is_separate_from_grasp_terminal_bonus_and_failure_wins():
    from go2_pvcnn.control.dh116.tabletop_control import progress_reward
    _, stage = progress_reward(**reward_inputs(stage_event=True))
    assert stage['stage'] == 5. and stage['success'] == 0.
    _, success = progress_reward(**reward_inputs(success_event=True))
    assert success['success'] == 25. and success['stage'] == 0.
    _, failed = progress_reward(**reward_inputs(success_event=True, stage_event=True, failed=True))
    assert failed['failure'] == -25.
    assert failed['stage'] == failed['success'] == 0.


@pytest.mark.parametrize('field,value', [
    ('action', np.zeros(11)), ('reference', np.full(12, np.nan)),
    ('previous', np.full(12, np.inf)), ('dt', 0.),
    ('metrics', metrics(pad_forces_n=np.ones(3))),
    ('previous_metrics', metrics(slip_m=np.nan)),
])
def test_progress_reward_rejects_invalid_input(field, value):
    from go2_pvcnn.control.dh116.tabletop_control import progress_reward
    with pytest.raises(ValueError):
        progress_reward(**reward_inputs(**{field: value}))


def test_progress_reward_rejects_missing_measured_state():
    from go2_pvcnn.control.dh116.tabletop_control import progress_reward
    incomplete = metrics()
    del incomplete['object_bottom_m']
    with pytest.raises(ValueError):
        progress_reward(**reward_inputs(metrics=incomplete))


def test_moving_object_goal_uses_current_translation_not_spawn_position():
    from go2_pvcnn.control.dh116.tabletop_control import moving_goal_error
    goals = np.array([[.55, .15, .37], [.54, -.15, .39]])
    roots = goals + [.06, 0, .0019]
    assert moving_goal_error(roots, goals, [.49, 0, .308], [.43, 0, .308]) == pytest.approx(.0019)
    # Real lateral tracking lag must still contribute to the approach gate.
    assert moving_goal_error(roots, goals, [.50, 0, .308], [.43, 0, .308]) > .01


def test_scripted_object_tracking_is_bounded_and_frozen_during_lift():
    from go2_pvcnn.control.dh116.tabletop_control import track_root_xy
    xy = np.zeros(2)
    moved = track_root_xy(xy, [.49, 0, .308], [.43, 0, .308], 'close', 1/240)
    np.testing.assert_allclose(moved, [.01/240, 0])
    np.testing.assert_allclose(track_root_xy(moved, [.50, 0, .308], [.43, 0, .308], 'lift', 1/240), moved)
    np.testing.assert_allclose(xy, [0, 0])
    np.testing.assert_allclose(track_root_xy(xy, [.49, 0, .308], [.43, 0, .308],
                                           'close', 1/240, allowed=False), xy)
