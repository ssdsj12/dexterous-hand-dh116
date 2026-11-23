import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts.certify_dh116_power_bilateral_detour import left_detour
from scripts.certify_dh116_power_thumb_route import maximum_joint_rates


def test_reversing_thumb_is_bounded_even_with_zero_net_motion():
    states = np.zeros((3, 12))
    states[:, 6:][1, 1] = .04
    rates = maximum_joint_rates(states)
    assert rates["finger12"] == pytest.approx(.08)
    assert rates["finger13"] == pytest.approx(.08 * .88178789)
    assert rates["finger21"] == 0.


def test_left_detour_keeps_right_target_and_respects_original_command_rates():
    low, high = active_limits()
    source = np.zeros((2, 12))
    source[:, 6:] = (low + high) / 2.
    target = source.copy()
    target[0, 2] -= .008
    target[0, 7] += .01
    axis = np.array([0., 0., -1.])
    states, actions = left_detour(source, target, axis)
    np.testing.assert_array_equal(states[0], source)
    np.testing.assert_array_equal(states[-1], target)
    np.testing.assert_array_equal(states[:, 1], np.broadcast_to(source[1], (len(states), 12)))
    assert not np.any(actions[:, [6, 7, 8, 9, 10, 11, 15, 16, 17, 21, 22, 23]])
    assert states[:, 0, 7].min() < source[0, 7]
    assert np.any(np.diff(states[:, 0, 7]) > 0.)
    assert np.abs(actions[:, :12]).max() <= 1.
    assert np.linalg.norm(actions[:, 12:].reshape(len(actions), 4, 3), axis=-1).max() <= 1.
    np.testing.assert_allclose(
        actions[:, :6] * ACTION_CONTRACT["joint_rate_rad_s"] / 60.,
        np.diff(states[:, 0, 6:], axis=0), atol=1e-14,
    )
    with pytest.raises(ValueError, match="right endpoint"):
        left_detour(source, target + [[0.] * 12, [1.] * 12], axis)



@pytest.mark.parametrize(("gap", "guard", "default_ok", "detour_ok"), [
    (.0004, True, True, True),
    (.0012, True, False, True),
    (-.000251, True, False, False),
    (None, True, False, False),
    (.0012, False, False, False),
])
def test_transit_endpoint_only_omits_maximum_positive_palm_gap(monkeypatch, gap, guard, default_ok, detour_ok):
    from scripts import screen_dh116_power_partial_opposition as screen

    monkeypatch.setattr(screen, "_wrist_clearance", lambda *a: {"passed": guard})
    monkeypatch.setattr(screen, "fk_poses", lambda *a: {"base_link": np.eye(4)})
    monkeypatch.setattr(screen, "sample_contact", lambda *a, **kw: {"minimum_gap_m": gap})
    monkeypatch.setattr(screen, "shapes_world", lambda *a: {"source:base_link": np.zeros((3, 3))})
    data = (None, None, None, np.zeros((2, 3)), {"wire_radius_m": .005}, None)
    assert screen.endpoint(*data)["passed"] is default_ok
    assert screen.endpoint(*data, require_palm_reach=False)["passed"] is detour_ok
