import numpy as np
import pytest

from scripts import run_dh116_pinch_fixed_force_hold_control as control


def inputs():
    actions = np.zeros((45, 24))
    actions[44, 3] = .02
    issued = np.array([.28, .47, .7, 1.15, 1.15, 1.15, 0., 0., 0., 0., 0., 0.])
    return actions, {side: (issued.copy(), issued.copy()) for side in ("left", "right")}


def test_same_frozen_prefix_and_all_ten_extra_windows_are_noop():
    prefix, measured = inputs()
    planned, targets = control.compile_noop(prefix, measured)
    assert planned.shape == (55, 24)
    np.testing.assert_array_equal(planned[:45], prefix)
    np.testing.assert_array_equal(planned[45:], 0.)
    for side in ("left", "right"):
        np.testing.assert_array_equal(targets[side], measured[side][1])


def test_sham_rejects_changed_issued_target():
    prefix, measured = inputs()
    _, goals = control.compile_noop(prefix, measured)
    measured["left"][1][6] += 1e-6
    with pytest.raises(ValueError, match="archived"):
        control.compile_noop(prefix, measured, nominal_targets=goals)
