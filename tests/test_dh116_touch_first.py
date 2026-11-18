import numpy as np
import pytest

from go2_pvcnn.control.dh116.touch_first import ApproachReadiness, TouchFirst


def measurement(**changes):
    values = dict(position_errors_m=np.zeros(4), normal_errors_rad=np.zeros(4),
                  joint_speeds_rad_s=np.zeros((2, 18)))
    values.update(changes)
    return values


def test_ready_depends_on_measured_state_not_four_seconds():
    gate = ApproachReadiness()
    for i in range(5):
        assert not gate.update(i * .01, **measurement())
    assert gate.update(.05, **measurement())


def test_fast_motion_or_position_error_resets_dwell():
    gate = ApproachReadiness()
    for i in range(5):
        gate.update(i * .01, **measurement())
    speed = np.zeros((2, 18)); speed[1, 8] = -.021
    assert not gate.update(.05, **measurement(joint_speeds_rad_s=speed))
    for i in range(6, 11):
        assert not gate.update(i * .01, **measurement())
    assert gate.update(.11, **measurement())
    assert not gate.update(.12, **measurement(position_errors_m=[0, 0, .003, 0]))


def test_timeout_does_not_authorize_grasp():
    gate = ApproachReadiness()
    for i in range(1000):
        assert not gate.update(i * .01, **measurement(normal_errors_rad=np.ones(4)))
    with pytest.raises(TimeoutError):
        gate.update(10., **measurement(normal_errors_rad=np.ones(4)))


@pytest.mark.parametrize('speed', [None, np.zeros(2), np.full((2, 18), np.nan)])
def test_missing_or_invalid_measured_speed_rejected(speed):
    with pytest.raises(ValueError):
        ApproachReadiness().update(0., **measurement(joint_speeds_rad_s=speed))


def test_no_contact_advances_only_thumbs():
    control = TouchFirst(np.ones((2, 2)))
    r = control.update(0., np.zeros((2, 2)), np.zeros((2, 2), dtype=bool))
    assert r['phases'] == ['touch', 'touch']
    assert np.all(r['normal_speeds_m_s'][:, 0] > 0)
    np.testing.assert_array_equal(r['normal_speeds_m_s'][:, 1], 0.)
    assert not r['ready_for_hold']


def test_hands_transition_independently_after_valid_touch_not_load_alone():
    control = TouchFirst(np.ones((2, 2)))
    loads = np.array([[.05, 0], [.05, 0]])
    valid = np.array([[True, False], [False, False]])
    for i in range(5):
        assert control.update(i * .01, loads, valid)['phases'] == ['touch', 'touch']
    r = control.update(.05, loads, valid)
    assert r['phases'] == ['oppose', 'touch']
    assert r['normal_speeds_m_s'][0, 1] > 0
    assert r['normal_speeds_m_s'][1, 1] == 0
    valid[0, 0] = False
    r = control.update(.06, loads, valid)
    assert r['phases'][0] == 'touch'
    assert r['normal_speeds_m_s'][0, 1] == 0


def test_preload_ramps_and_contact_loss_prevents_hold():
    control = TouchFirst(np.ones((2, 2)))
    loads = np.full((2, 2), .05); valid = np.ones((2, 2), dtype=bool)
    for i in range(11):
        r = control.update(i * .01, loads, valid)
    assert r['phases'] == ['preload', 'preload']
    assert not r['ready_for_hold']
    before = r['target_loads_n'].copy()
    r = control.update(.11, loads, valid)
    np.testing.assert_allclose(r['target_loads_n'] - before, .01)
    for i in range(12, 111):
        r = control.update(i * .01, loads, valid)
    assert r['ready_for_hold']
    np.testing.assert_allclose(r['target_loads_n'], 1.)
    valid[1, 1] = False
    r = control.update(1.11, loads, valid)
    assert not r['ready_for_hold'] and r['phases'][1] == 'oppose'


def test_external_controller_targets_skip_only_the_unused_ramp_not_contact_dwell():
    control=TouchFirst(np.ones((2,2)))
    loads=np.ones((2,2));valid=np.ones((2,2),dtype=bool)
    for i in range(12):
        r=control.update(i*.01,loads,valid,controller_target_loads_n=np.ones((2,2)))
        if i<10:assert not r['ready_for_hold']
        np.testing.assert_array_equal(r['target_loads_n'],1.)
    assert r['phases']==['preload','preload'] and r['ready_for_hold']
    assert r['preload_target_source']=='controller'
    np.testing.assert_array_equal(r['target_loads_n'],1.)
    valid[1,1]=False
    r=control.update(.12,loads,valid,controller_target_loads_n=np.ones((2,2)))
    assert not r['ready_for_hold'] and r['phases'][1]=='oppose'


def test_external_target_is_not_evidence_of_contact():
    control=TouchFirst(np.ones((2,2)))
    for i in range(30):
        r=control.update(i*.01,np.zeros((2,2)),np.zeros((2,2),dtype=bool),
                         controller_target_loads_n=np.ones((2,2)))
    assert r['phases']==['touch','touch'] and not r['ready_for_hold']


@pytest.mark.parametrize('target',[np.ones(4),np.full((2,2),np.nan),np.full((2,2),4.)])
def test_invalid_external_target_does_not_advance_evidence_time(target):
    control=TouchFirst(np.ones((2,2)))
    with pytest.raises(ValueError):
        control.update(0.,np.zeros((2,2)),np.zeros((2,2),dtype=bool),controller_target_loads_n=target)
    assert control.last_time is None


def test_overloaded_thumb_retreats_while_index_stays_open():
    control = TouchFirst(np.ones((2, 2)))
    r = control.update(0., np.ones((2, 2)), np.zeros((2, 2), dtype=bool))
    assert np.all(r['normal_speeds_m_s'][:, 0] < 0)
    np.testing.assert_array_equal(r['normal_speeds_m_s'][:, 1], 0.)


@pytest.mark.parametrize('bad', ['negative', 'nan', 'shape', 'valid', 'time'])
def test_bad_contact_evidence_rejected_without_advancing(bad):
    control = TouchFirst(np.ones((2, 2)))
    loads = np.zeros((2, 2)); valid = np.zeros((2, 2), dtype=bool); t = 0.
    if bad == 'negative': loads[0, 0] = -.01
    if bad == 'nan': loads[1, 1] = np.nan
    if bad == 'shape': loads = np.zeros(4)
    if bad == 'valid': valid = np.zeros((2, 2))
    if bad == 'time': t = np.nan
    with pytest.raises(ValueError): control.update(t, loads, valid)
    assert control.update(0., np.zeros((2, 2)), np.zeros((2, 2), dtype=bool))['phases'] == ['touch'] * 2
