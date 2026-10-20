import importlib
import importlib.util
import numpy as np
import pytest
from go2_pvcnn.control.dh116.touch_first import TouchFirst


def api():
    name = 'go2_pvcnn.control.dh116.speed_profile'
    assert importlib.util.find_spec(name), 'staged speed governor is missing'
    return importlib.import_module(name)


def evidence(**changes):
    data = dict(force_speeds=np.full((2, 2), .0005), loads=np.zeros((2, 2)),
                valid=np.zeros((2, 2), dtype=bool), gaps=np.full((2, 2), .02),
                active=np.ones((2, 2), dtype=bool))
    data.update(changes)
    return data


def test_free_space_acceleration_is_bounded_and_brakes_by_measured_gap():
    module = api(); profile = module.GraspSpeedProfile()
    governor = module.ContactSpeedGovernor(profile)
    previous = np.zeros((2, 2))
    for i in range(101):
        speed = governor.update(i * .01, **evidence())
        assert np.all(speed >= 0)
        assert np.max(np.abs(speed - previous)) <= profile.acceleration_m_s2 * .01 + 1e-12
        previous = speed
    np.testing.assert_allclose(speed, profile.far_m_s)
    gaps = np.array([[.002, .003], [.004, .02]])
    speed = governor.update(1.01, **evidence(gaps=gaps))
    expected = np.minimum(profile.far_m_s, np.sqrt(profile.near_m_s**2 +
                          2 * profile.acceleration_m_s2 * np.maximum(gaps - .002, 0)))
    np.testing.assert_allclose(speed, expected)


def test_contact_load_and_inactive_channel_override_free_space_speed():
    module = api(); governor = module.ContactSpeedGovernor(module.GraspSpeedProfile())
    for i in range(51):
        governor.update(i * .01, **evidence())
    speed = governor.update(.51, **evidence(loads=np.full((2, 2), .01),
        force_speeds=np.array([[-.002, .002], [.0004, .0005]]),
        active=np.array([[True, True], [True, False]])))
    np.testing.assert_allclose(speed, [[-.001, .001], [.0004, 0]])


@pytest.mark.parametrize('bad', [None, np.zeros(4), np.full((2, 2), np.nan)])
def test_invalid_gap_does_not_advance_touch_state(bad):
    module = api(); control = TouchFirst(np.ones((2, 2)), speed_profile=module.GraspSpeedProfile())
    with pytest.raises(ValueError, match='gap'):
        control.update(0., np.zeros((2, 2)), np.zeros((2, 2), dtype=bool), surface_gaps_m=bad)
    assert control.last_time is None
    assert control.phases == ['touch', 'touch']
    control.update(0., np.zeros((2, 2)), np.zeros((2, 2), dtype=bool), surface_gaps_m=np.ones((2, 2)))


def test_touch_first_profile_moves_only_thumb_and_legacy_stays_unchanged():
    module = api(); control = TouchFirst(np.ones((2, 2)), speed_profile=module.GraspSpeedProfile())
    for i in range(101):
        result = control.update(i * .01, np.zeros((2, 2)), np.zeros((2, 2), dtype=bool),
                                surface_gaps_m=np.full((2, 2), .02))
    np.testing.assert_allclose(result['normal_speeds_m_s'], [[.02, 0], [.02, 0]])
    legacy = TouchFirst(np.ones((2, 2))).update(0., np.zeros((2, 2)), np.zeros((2, 2), dtype=bool))
    np.testing.assert_allclose(legacy['normal_speeds_m_s'], [[.0005, 0], [.0005, 0]])


@pytest.mark.parametrize('change', [dict(far_m_s=np.nan), dict(near_m_s=-1),
    dict(acceleration_m_s2=0), dict(slowdown_gap_m=-1), dict(lift_m_s=.1),
    dict(near_m_s=.03), dict(loaded_m_s=.1)])
def test_invalid_profile_rejected(change):
    with pytest.raises(ValueError):
        api().GraspSpeedProfile(**change)


@pytest.mark.parametrize('change', [dict(loads=np.full((2, 2), -1.)),
    dict(force_speeds=np.full((2, 2), np.nan)), dict(valid=np.ones((2, 2))),
    dict(active=np.ones(4, dtype=bool)), dict(gaps=np.zeros(4))])
def test_bad_governor_evidence_does_not_mutate_state(change):
    module = api(); governor = module.ContactSpeedGovernor(module.GraspSpeedProfile())
    governor.update(0., **evidence())
    before = governor.last_speed.copy()
    with pytest.raises(ValueError):
        governor.update(.01, **evidence(**change))
    assert governor.last_time == 0.
    np.testing.assert_array_equal(governor.last_speed, before)
    assert np.all(governor.update(.01, **evidence()) > 0)


def test_governor_rejects_repeated_or_stale_time_without_mutating_state():
    module = api(); governor = module.ContactSpeedGovernor(module.GraspSpeedProfile())
    governor.update(0., **evidence())
    for time in (0., -.01, .02, np.nan):
        with pytest.raises(ValueError):
            governor.update(time, **evidence())
        assert governor.last_time == 0.


def test_force_stop_and_retreat_override_positive_free_space_gap():
    module = api(); governor = module.ContactSpeedGovernor(module.GraspSpeedProfile())
    for i in range(51):
        governor.update(i * .01, **evidence())
    force = np.array([[0., -.0004], [-.002, 0.]])
    np.testing.assert_allclose(governor.update(.51, **evidence(force_speeds=force)),
                               [[0., -.0004], [-.001, 0.]])
