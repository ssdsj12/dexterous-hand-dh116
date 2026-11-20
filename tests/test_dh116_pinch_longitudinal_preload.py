import numpy as np
import pytest

from scripts import run_dh116_pinch_longitudinal_preload as preload


def test_nominal_pinch_ramp_uses_only_one_bounded_wrist_axis():
    old = np.zeros((35, 24))
    issued = np.zeros(12)
    issued[1:3] = [.4, .6]
    issued[7] = .24
    states = {side:(issued.copy(), issued.copy()) for side in ("left", "right")}
    actions, targets = preload.compile_longitudinal(
        old, states, {"left": [.4008, .24008], "right": [.401, .6005]},
    )

    np.testing.assert_array_equal(actions[:35], old)
    np.testing.assert_array_equal(actions[37:], 0)
    delta = actions[35:37].sum(axis=0) * preload.prelift.WINDOW_SCALE
    np.testing.assert_array_equal(np.flatnonzero(delta), [1, 7, 8, 13])
    np.testing.assert_allclose(delta[[1, 7, 8, 13]], [.0008, .001, .0005, .00008])
    assert targets["left"][2] == issued[2]


@pytest.mark.parametrize("left_y", [-.0001, 0., .00017])
def test_pinch_ramp_rejects_unproved_axis_direction_or_magnitude(left_y):
    issued = np.zeros(12)
    issued[1:3] = [.4, .6]
    states = {side:(issued.copy(), issued.copy()) for side in ("left", "right")}
    with pytest.raises(ValueError, match="longitudinal"):
        preload.compile_longitudinal(np.zeros((35, 24)), states,
                                    {"left": [.4008, left_y], "right": [.401, .6005]})
