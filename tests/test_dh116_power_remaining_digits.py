import numpy as np
import pytest


def test_cooked_clearance_deducts_motion_between_nearest_samples():
    from tools import audit_dh116_power_remaining_digits as audit

    result = audit.continuous_reach_bound([.2, .3, .6], [.0011, .0012, .0014], point_travel_m=.0008)
    assert result["between_samples_reserve_m"] == pytest.approx(.0003)
    assert result["continuous_lower_m"] == pytest.approx(.0008)
    assert result["outside_witness_shell"]
    unresolved = audit.continuous_reach_bound([.2, .6], [.0007, .001], point_travel_m=.0008)
    assert not unresolved["outside_witness_shell"]


def test_candidate_roots_use_only_finite_brackets_and_eligible_material_nodes():
    from tools import audit_dh116_power_remaining_digits as audit

    def query(angle):
        return np.array([np.nan if angle < .2 else -angle, .5 - angle, .7 - angle, .3 - angle])

    result = audit.station_crossings(query, np.array([.0, .25, .6, 1.]), [True, False, True, True])
    assert sorted({item["node"] for item in result["candidates"]}) == [2, 3]
    node_2 = [item for item in result["candidates"] if item["node"] == 2]
    assert len(node_2) == 3
    assert node_2[0]["angle_rad"] == pytest.approx(.700006)
    assert all(item["gap_m"] == pytest.approx(item["target_gap_m"]) for item in node_2)


def test_material_nodes_require_correct_measured_half_and_no_support_adjacency():
    from tools import audit_dh116_power_remaining_digits as audit

    nodes = np.zeros((41, 3))
    nodes[:, 1] = np.linspace(-.25, .25, 41)
    nodes[35, 1] = -.01
    blocked = np.zeros(41, dtype=bool)
    blocked[38] = True
    mask = audit.eligible_nodes(nodes, blocked, "left")
    assert mask[37]
    assert not any(mask[[0, 20, 35, 38, 40]])
    assert not audit.eligible_nodes(nodes, blocked, "right")[35]


def test_compiled_single_digit_is_exactly_continuous_and_retains_other_commands():
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    from tools import audit_dh116_power_remaining_digits as audit

    lower, upper = active_limits()
    start = np.concatenate((np.zeros((2, 6)), np.stack((lower, lower))), axis=1)
    waypoints, actions = audit.compile_extension(start, "right", "finger53_link", upper[5])
    assert np.array_equal(waypoints[0], start)
    assert waypoints[-1, 1, 11] == upper[5]
    assert np.count_nonzero(actions[:, :11]) == np.count_nonzero(actions[:, 12:]) == 0
    assert np.all(actions[:, 11] > 0.)
    assert np.max(actions[:, 11]) <= 1.
    assert np.array_equal(waypoints[:, 0], np.broadcast_to(start[0], waypoints[:, 0].shape))
    with pytest.raises(ValueError, match="closing"):
        audit.compile_extension(start, "right", "finger53_link", lower[5] - .1)
