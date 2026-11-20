import numpy as np
import pytest

from scripts import prepare_dh116_pinch_precontact_hold as hold


def test_pause_rolls_back_live_left_targets_without_adding_right_actions():
    actual = np.linspace(.1, .3, 12)
    issued = actual + .0001
    schedule, target = hold.left_hold_windows(actual, issued)
    assert schedule.shape == (2, 24)
    np.testing.assert_array_equal(schedule[:, 6:12], 0.)
    np.testing.assert_array_equal(schedule[:, 15:18], 0.)
    np.testing.assert_array_equal(schedule[:, 21:24], 0.)
    np.testing.assert_allclose(target[1:3], actual[1:3] + .12 * (issued[1:3] - actual[1:3]))
    np.testing.assert_array_equal(target[6:12], actual[6:12])
    command = hold.prelift.global_channel_order(issued, np.zeros(12))
    goal = hold.prelift.global_channel_order(target, np.zeros(12))
    np.testing.assert_allclose(command + schedule.sum(axis=0) * hold.prelift.WINDOW_SCALE, goal)


def test_admission_uses_distal_pads_as_required_clearance():
    row = {"rod": {"node_positions": np.zeros((1, 41, 3))}}
    calls = []

    def prove(received, actual, issued, goal, envelope, *, nodes, **kwargs):
        np.testing.assert_array_equal(issued, goal)
        np.testing.assert_array_equal(nodes, row["rod"]["node_positions"][0])
        assert received is row
        assert kwargs["side"] == "left"
        assert kwargs["require_pad_clearance"]
        calls.append(kwargs["phase_names"])
        return {"left_hold": {"proved": True}}, None

    assert hold.certify_left_hold(row, np.zeros(12), np.ones(12) * .0001, proof=prove)["proved"]
    assert calls == [("left_hold",)]


def test_admitted_boundaries_stop_before_suspect_thumb_reach():
    hold.validate_boundary(18)
    hold.validate_boundary(26)
    with pytest.raises(ValueError, match="early precontact boundary"):
        hold.validate_boundary(42)
