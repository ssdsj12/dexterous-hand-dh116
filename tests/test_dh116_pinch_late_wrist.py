import numpy as np
import pytest

from scripts.run_dh116_pinch_late_wrist import variant_schedule, compare_prefix


def test_only_final_left_wrist_is_modified():
    original = np.zeros((9, 24))
    original[1:, [1, 2, 7, 8, 21]] = .25
    next_actions = variant_schedule(original, [.00001, 0., 0.], [0., .002, 0.])
    expected = original.copy()
    expected[8, 12] += .00001 / (.01 * 8 / 480)
    expected[8, 19] += .002 / (.5 * 8 / 480)
    np.testing.assert_allclose(next_actions, expected)
    assert np.all(original[:, [12, 19]] == 0.)


def test_variant_requires_original_angular_rate_ball():
    with pytest.raises(ValueError, match="rate"):
        variant_schedule(np.zeros((9, 24)), [0., 0., 0.], [.006, .006, .006])


def test_inconsistent_current_state_refuses_the_last_window():
    fixture = {
        "rod_nodes_m": np.zeros((41,3)),
        "left_q": np.zeros(17), "right_q": np.zeros(17),
        "left_pose": np.r_[np.zeros(3), [0., 0., 0., 1.]],
        "right_pose": np.r_[np.zeros(3), [0., 0., 0., 1.]],
    }
    assert compare_prefix(fixture, fixture)["passed"]
    changed = dict(fixture, rod_nodes_m=np.full((41,3), .001))
    assert not compare_prefix(changed, fixture)["passed"]
