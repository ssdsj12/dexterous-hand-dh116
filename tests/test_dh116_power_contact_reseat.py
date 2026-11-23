import numpy as np
import pytest


def test_retained_site_cannot_open_or_be_replaced_by_a_foreign_contact():
    from tools import audit_dh116_power_contact_reseat as audit

    gaps = np.array([.001, -.00005, .002, np.nan])
    passed = audit.owned_site(gaps, node=1, limit=.00015, must_touch=True)
    assert passed["failure"] is None
    assert passed["negative_nodes"] == [1]
    opened = gaps.copy()
    opened[1] = .0006
    assert audit.owned_site(opened, node=1, limit=.00015, must_touch=False)["failure"] is None
    assert audit.owned_site(opened, node=1, limit=.00015, must_touch=True)["failure"] == "retained_site_open"
    opened[2] = -.00001
    assert audit.owned_site(opened, node=1, limit=.00015, must_touch=False)["failure"] == "foreign_material_station"


def test_original_penetration_limit_is_not_a_preload_success():
    from tools import audit_dh116_power_contact_reseat as audit

    value = audit.owned_site(np.array([.003, -.000151, .01]), node=1, limit=.00015, must_touch=True)
    assert value["failure"] == "original_penetration_limit"
    assert value["owned_gap_m"] == pytest.approx(-.000151)


def test_fixed_wrist_little_goal_changes_only_its_coupled_slot():
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    from tools import audit_dh116_power_contact_reseat as audit

    low, high = active_limits()
    start = np.r_[np.zeros(6), low]
    goal = audit.digit_goal(start, "finger53_link", high[5])
    assert goal[11] == high[5]
    assert np.array_equal(goal[:11], start[:11])
    with pytest.raises(ValueError, match="strictly closing"):
        audit.digit_goal(start, "finger53_link", low[5]-.01)


def test_matched_thumb_hold_keeps_identical_wrist_and_ring_command_timing():
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    from tools import audit_dh116_power_contact_reseat as audit

    low, _ = active_limits()
    start = np.stack((np.r_[np.zeros(6), low], np.r_[np.zeros(6), low]))
    goal = start.copy()
    goal[1, :3] += [.001, .002, .001]
    goal[1, 6:8] += [.02, .03]
    goal[1, 10] += .07
    states, commands, _ = audit.compile_schedule(start, [goal])
    fixed, held_commands = audit.hold_thumb(states, commands)
    assert np.array_equal(fixed[:, 1, :6], states[:, 1, :6])
    assert np.array_equal(fixed[:, 1, 8:], states[:, 1, 8:])
    assert np.array_equal(fixed[:, 1, 6:8], np.broadcast_to(start[1, 6:8], fixed[:, 1, 6:8].shape))
    assert np.array_equal(held_commands[:, :6], commands[:, :6])
    assert np.array_equal(held_commands[:, 8:], commands[:, 8:])
    assert np.count_nonzero(held_commands[:, 6:8]) == 0


def test_fixed_wrist_ring_reacquisition_can_only_change_its_actuation_slot():
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    from tools import audit_dh116_power_contact_reseat as audit

    low, _ = active_limits()
    start = np.r_[np.zeros(6), low]
    ring = audit.digit_goal(start, "finger43_link", low[4]+.01)
    assert ring[10] == pytest.approx(low[4]+.01)
    assert np.array_equal(np.delete(ring, 10), np.delete(start, 10))
    with pytest.raises(ValueError, match="ring or little"):
        audit.digit_goal(start, "finger14_link", low[1]+.01)
