import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import expand, fold_jacobian, JOINTS, ACTIVE


def test_named_linear_expansion_and_chain_rule():
    q = np.full(6, .3)
    full = expand(q)
    assert full.shape == (11,)
    assert full[JOINTS.index('finger13')] == pytest.approx(.88178789*.3-.00697725445)
    j = np.arange(33, dtype=float).reshape(3, 11)
    analytic = fold_jacobian(j, JOINTS)
    numeric = np.column_stack([(j@expand(q+np.eye(6)[i]*1e-6)-j@expand(q-np.eye(6)[i]*1e-6))/2e-6 for i in range(6)])
    np.testing.assert_allclose(analytic, numeric, atol=1e-8)
    reverse = tuple(reversed(JOINTS))
    np.testing.assert_allclose(fold_jacobian(j[:, ::-1], reverse), analytic)
    assert len(ACTIVE) == 6


@pytest.mark.parametrize('q', [np.zeros(6), np.ones(6)*2, np.ones(5), np.ones(6)*np.nan, np.ones(6)*np.inf])
def test_reject_invalid_or_infeasible_state(q):
    with pytest.raises(ValueError):
        expand(q)


def test_reject_unknown_or_duplicate_jacobian_columns():
    with pytest.raises(ValueError):
        fold_jacobian(np.zeros((3, 11)), JOINTS[:-1]+(JOINTS[0],))
    with pytest.raises(ValueError):
        fold_jacobian(np.zeros((3, 10)), JOINTS)
    with pytest.raises(ValueError):
        fold_jacobian(np.full((3,11), np.nan), JOINTS)


@pytest.mark.parametrize('index,degrees', [(0, 80.01), (2, 70.01), (3, 70.01), (4, 70.01), (5, 70.01)])
def test_v2_manual_limits_reject_states_allowed_by_source_urdf(index, degrees):
    q = np.full(6, .3)
    q[index] = np.deg2rad(degrees)
    before = q.copy()
    with pytest.raises(ValueError, match='limits'):
        expand(q)
    np.testing.assert_array_equal(q, before)


def test_v2_effective_limits_and_all_followers():
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    lo, hi = active_limits()
    np.testing.assert_allclose(lo, [0., .00697725445/.88178789, 0., 0., 0., 0.])
    np.testing.assert_allclose(hi, [np.deg2rad(80), 1.030, *([np.deg2rad(70)]*4)])
    for q in (lo, hi):
        full = expand(q)
        assert 0 <= full[2] <= 1.13
        assert np.all(full[[4,6,8,10]] >= .011072)
        assert np.all(full[[4,6,8,10]] <= 1.950714)
    for i in range(6):
        for bound, delta in ((lo, -1e-9), (hi, 1e-9)):
            q = bound.copy()
            q[i] += delta
            with pytest.raises(ValueError):
                expand(q)
    lo[:] = 99
    hi[:] = -99
    assert active_limits()[0][0] == 0
    assert active_limits()[1][1] == 1.030


def test_chain_rule_near_effective_boundaries():
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    lo, hi = active_limits()
    j = np.arange(66, dtype=float).reshape(6,11) / 66
    for q in (lo+1e-4, hi-1e-4):
        fd = np.column_stack([(j@expand(q+np.eye(6)[i]*1e-6)-j@expand(q-np.eye(6)[i]*1e-6))/2e-6 for i in range(6)])
        np.testing.assert_allclose(fold_jacobian(j, JOINTS), fd, atol=1e-8, rtol=0)
