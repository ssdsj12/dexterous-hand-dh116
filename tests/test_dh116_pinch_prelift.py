import numpy as np
import pytest

from scripts import prepare_dh116_pinch_prelift as prelift


def test_two_windows_spend_original_channel_limits_and_keep_preload():
    measured = np.zeros(24)
    command = measured.copy()
    command[2], command[8], command[22] = .006266, .005393, .016232
    target = measured.copy()
    target[[2, 8]] = [.001, .001]
    schedule = prelift.bounded_transition(command, target)
    np.testing.assert_allclose(command + schedule.sum(axis=0) * prelift.WINDOW_SCALE, target, atol=1e-12)
    assert schedule[0, 2] == -1. and schedule[1, 2] < 0.
    assert schedule[0, 22] == -1. and schedule[1, 22] < 0.
    assert np.all(schedule[2:] == 0.)


def test_required_third_window_and_nonfinite_targets_are_rejected():
    command = np.zeros(24)
    target = command.copy()
    target[21:24] = [.009, .016, 0.]
    with pytest.raises(ValueError, match="two original windows"):
        prelift.bounded_transition(command, target)
    target[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prelift.bounded_transition(command, target)


def test_retained_target_preserves_nonclosing_joints_and_needs_measured_preload():
    low, high = prelift.active_limits()
    actual = np.r_[(low + high) / 2, np.zeros(6)]
    command = actual + .006
    right = prelift.retained_target(actual, command, "right")
    np.testing.assert_array_equal(right[[0, 3, 4, 5, 6, 7, 8]], actual[[0, 3, 4, 5, 6, 7, 8]])
    np.testing.assert_allclose(right[1:3] - actual[1:3], [.0006, .0006])
    np.testing.assert_allclose(right[9:12], [.0015, .0015, .0015])
