import numpy as np
import pytest


def test_axis_rejects_empty_instead_of_string_index_zero():
    from go2_pvcnn.control.dh116.usd_contract import axis_vector
    for token in ('', 'XY', 'x', None):
        with pytest.raises(ValueError):
            axis_vector(token)
    np.testing.assert_array_equal(axis_vector('Z'), [0, 0, 1])


def test_full_limits_are_affine_image_in_degrees():
    from go2_pvcnn.control.dh116.usd_contract import full_limits_degrees
    from go2_pvcnn.control.dh116.linear_coupling import active_limits, expand
    lo, hi = full_limits_degrees()
    a, b = active_limits()
    np.testing.assert_allclose(lo, np.rad2deg(expand(a)))
    np.testing.assert_allclose(hi, np.rad2deg(expand(b)))
    assert hi[0] == pytest.approx(80)
    assert hi[3] == pytest.approx(70)
