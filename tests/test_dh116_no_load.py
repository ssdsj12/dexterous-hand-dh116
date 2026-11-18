import numpy as np
import pytest


def test_slow_reference_closed_cycle():
    from go2_pvcnn.control.dh116.no_load import reference
    times = np.linspace(0, 8, 801)
    qs = np.array([reference(t) for t in times])
    np.testing.assert_allclose(qs[0], qs[-1])
    assert qs[:, 2].max() == pytest.approx(.3)
    assert np.max(np.abs(np.diff(qs, axis=0))/.01) < .08
    np.testing.assert_allclose(qs[:, [0,1,3,4,5]], .2)
    for t in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            reference(t)


def test_guard_rejects_tracking_coupling_and_nan():
    from go2_pvcnn.control.dh116.no_load import metrics
    from go2_pvcnn.control.dh116.linear_coupling import expand
    q = np.full(6,.2)
    full = expand(q)
    assert metrics(full, np.zeros(11), q)['passed']
    bad = full.copy(); bad[4] += .02
    assert not metrics(bad, np.zeros(11), q)['passed']
    bad = full.copy(); bad[3] += .03
    assert not metrics(bad, np.zeros(11), q)['passed']
    bad[0] = np.nan
    assert not metrics(bad, np.zeros(11), q)['passed']


def test_each_active_joint_reference():
    from go2_pvcnn.control.dh116.no_load import reference
    for i in range(6):
        q = reference(4, active_index=i)
        assert q[i] == pytest.approx(.3)
        np.testing.assert_allclose(np.delete(q,i), .2)
    for bad in (-1, 6, 1.5, True):
        with pytest.raises(ValueError):
            reference(4, active_index=bad)


def test_contact_guard_fails_closed():
    from go2_pvcnn.control.dh116.no_load import contact_metrics
    assert contact_metrics(np.zeros((17,3)), 17)['passed']
    f = np.zeros((17,3)); f[3,1]=.02
    assert not contact_metrics(f,17)['passed']
    assert not contact_metrics(np.zeros((16,3)),17)['passed']
    f[3,1]=np.nan
    assert not contact_metrics(f,17)['passed']
