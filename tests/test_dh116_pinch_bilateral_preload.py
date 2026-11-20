import numpy as np
import pytest

from scripts import run_dh116_pinch_bilateral_preload as preload


def test_preload_keeps_archived_windows_and_does_not_move_wrists():
    previous = np.zeros((25, 24))
    previous[0, 12] = 0.5
    issued = np.zeros(12)
    issued[1:3] = [0.4, 0.6]
    states = {side: (issued.copy(), issued.copy()) for side in ("left", "right")}
    goals = {"left": [0.4008, 0.6006], "right": [0.4009, 0.6004]}

    actions, targets = preload.compile_preload(previous, states, goals)

    np.testing.assert_array_equal(actions[:25], previous)
    np.testing.assert_array_equal(actions[25:, preload.closure.UNCHANGED], 0)
    np.testing.assert_array_equal(actions[27:], 0)
    np.testing.assert_allclose(
        actions[25:27].sum(0)[[1, 2, 7, 8]] * preload.prelift.WINDOW_SCALE[[1, 2, 7, 8]],
        [0.0008, 0.0006, 0.0009, 0.0004],
    )
    np.testing.assert_array_equal(targets["right"][1:3], goals["right"])
    goals["right"][1] = issued[2]
    with pytest.raises(ValueError, match="positive"):
        preload.compile_preload(previous, states, goals)


def test_boundary_requires_current_signed_opposition_from_both_hands():
    positive = {"signed_opposition": True}
    missing = {"signed_opposition": False}
    reaction = [{"substep": j, "left": positive, "right": positive} for j in range(12)]
    row = {"physics_step": 202, "guard_failure": None, "stage_reaction": {"per_substep": reaction}}
    assert preload.boundary_has_opposition(row)
    reaction[-1]["right"] = missing
    assert not preload.boundary_has_opposition(row)
    reaction[-1]["right"] = positive
    reaction[-1]["substep"] = 10
    assert not preload.boundary_has_opposition(row)
    reaction[-1]["substep"] = 11
    row["guard_failure"] = "pinch_unselected_contact_pair"
    assert not preload.boundary_has_opposition(row)
