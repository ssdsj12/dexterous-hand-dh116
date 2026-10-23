import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_task import TabletopTask, task_reward


def test_approach_reference_opens_and_blends_within_coupled_limits():
    from go2_pvcnn.control.dh116.tabletop_task import approach_joint_reference
    from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
    low, high = active_limits()
    grasp = low + .4*(high-low)
    opened = approach_joint_reference(grasp, .04)
    assert np.all(opened < grasp)
    assert np.allclose(approach_joint_reference(grasp, 0.), grasp)
    for offset in np.linspace(0, .04, 50):
        q = approach_joint_reference(grasp, offset)
        expand(q)
        assert np.all(q >= opened) and np.all(q <= grasp)
    with pytest.raises(ValueError):
        approach_joint_reference(grasp, float('nan'))


def test_descent_guard_stops_on_contact_and_preserves_table_margin():
    from go2_pvcnn.control.dh116.tabletop_task import guarded_descent
    assert guarded_descent(.04, 1/240, .03, [0, 0, 0, 0]) == pytest.approx(.04-.02/240)
    assert guarded_descent(.01, 1/240, .01, [.11, 0, 0, 0]) == .01
    assert guarded_descent(.01, 1/240, .002, [0]*4) == .01
    assert guarded_descent(.01, 1/240, .00201, [0]*4) == pytest.approx(.00999)
    with pytest.raises(ValueError):
        guarded_descent(.01, 1/240, float('nan'), [0]*4)


def test_root_motion_is_substep_rate_limited():
    from go2_pvcnn.control.dh116.tabletop_task import advance_root_offset
    offset = .04
    for _ in range(16):
        new = advance_root_offset(offset, 'approach', 1/960)
        assert abs(new-offset) <= .02/960 + 1e-12
        offset = new
    assert offset == pytest.approx(.04-.02/60)
    assert advance_root_offset(0., 'approach', 1/960) == 0.
    assert advance_root_offset(.05, 'lift', 1/960) == .05
    assert advance_root_offset(.01, 'close', 1/960) == .01


def test_joint_safety_rejects_divergence_and_mimic_drift():
    from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure
    q = np.array([0.2, 0.3])
    limits = np.array([[0., 1.], [0., 1.]])
    assert joint_safety_failure(q, limits, [0.001]) is None
    assert joint_safety_failure([0.2, 30430.], limits, [0.]) == 'joint_limit'
    assert joint_safety_failure(q, limits, [0.1]) == 'mimic_error'
    assert joint_safety_failure([np.nan, 0.3], limits, [0.]) == 'nonfinite_joint_state'


def test_training_gate_rejects_failed_or_missing_baseline():
    from go2_pvcnn.control.dh116.tabletop_task import baseline_allows_training
    assert not baseline_allows_training({})
    assert not baseline_allows_training(dict(is_success=False, failure='joint_limit'))
    assert not baseline_allows_training(dict(is_success=True, failure='table_penetration'))
    assert baseline_allows_training(dict(is_success=True, failure=None))


def observation(**overrides):
    data = dict(approach_error_m=0.0, pad_forces_n=np.ones(4),
                clearance_m=0.004, object_bottom_m=0.30, object_speed_m_s=0.0,
                slip_m=0.0, max_force_n=1.0)
    data.update(overrides)
    return data


def test_sequence_requires_continuous_four_pad_contact_before_lift():
    task = TabletopTask()
    task.update(0.1, **observation())
    assert task.phase == 'close'
    for _ in range(15):
        task.update(0.1, **observation(pad_forces_n=np.array([1, 1, 1, 0])))
    assert task.phase == 'close'
    for _ in range(9):
        task.update(0.1, **observation())
    assert task.phase == 'lift'
    task.update(0.1, **observation(pad_forces_n=np.zeros(4)))
    assert task.failure == 'contact_lost'


def test_success_requires_whole_object_height_and_hold_and_is_one_shot():
    task = TabletopTask()
    for _ in range(10):
        task.update(0.1, **observation())
    assert task.phase == 'lift'
    for _ in range(4):
        assert not task.update(0.1, **observation(object_bottom_m=0.335))
    assert task.update(0.1, **observation(object_bottom_m=0.335))
    assert not task.update(0.1, **observation(object_bottom_m=0.335))


def test_penetration_aborts_even_when_contact_and_lift_look_good():
    task = TabletopTask()
    task.update(0.1, **observation(clearance_m=-0.002, object_bottom_m=0.34))
    assert task.failure == 'table_penetration'
    assert not task.success


def test_reward_no_free_lift_credit_and_no_nan():
    kwargs = dict(phase='approach', action=np.zeros(12), reference=np.zeros(12),
                  previous=np.zeros(12), dt=0.02, success_event=False, failed=False,
                  **observation(pad_forces_n=np.zeros(4), object_bottom_m=0.34))
    reward, terms = task_reward(**kwargs)
    assert terms['lift'] == 0 and terms['success'] == 0
    assert reward == pytest.approx(sum(terms.values()))
    kwargs['slip_m'] = float('nan')
    with pytest.raises(ValueError):
        task_reward(**kwargs)
