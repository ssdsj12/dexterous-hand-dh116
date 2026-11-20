import numpy as np
import pytest

from scripts import run_dh116_pinch_asymmetric_preload as preload


def test_asymmetric_ramp_keeps_left_index_and_wrists_fixed():
    previous = np.zeros((35, 24))
    issued = np.zeros(12)
    issued[1:3] = [0.4, 0.6]
    states = {side: (issued.copy(), issued.copy()) for side in ("left", "right")}
    goals = {"left": [0.40075, 0.6], "right": [0.40099, 0.60048]}

    actions, targets = preload.compile_asymmetric(previous, states, goals)

    np.testing.assert_array_equal(actions[:35], previous)
    np.testing.assert_array_equal(actions[37:], 0)
    scaled = actions[35:37].sum(0) * preload.prelift.WINDOW_SCALE
    np.testing.assert_array_equal(np.flatnonzero(scaled), [1, 7, 8])
    np.testing.assert_allclose(scaled[[1, 7, 8]], [0.00075, 0.00099, 0.00048])
    assert targets["left"][2] == issued[2]


@pytest.mark.parametrize("left_index", [0.59999, 0.60001])
def test_asymmetric_ramp_rejects_moved_left_index(left_index):
    issued = np.zeros(12)
    issued[1:3] = [0.4, 0.6]
    states = {side: (issued.copy(), issued.copy()) for side in ("left", "right")}
    with pytest.raises(ValueError, match="left index"):
        preload.compile_asymmetric(np.zeros((35, 24)), states,
                                   {"left": [0.40075, left_index], "right": [0.40099, 0.60048]})
