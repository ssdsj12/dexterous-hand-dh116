import numpy as np

from scripts import run_dh116_pinch_precontact_pair as pair


def test_left_pause_preserves_original_right_prefix_until_rotation():
    actions = np.full((9, 24), .125)
    hold = np.zeros((2, 24))
    hold[0, 1] = -.2
    rotation = np.array([.1, -.3, .01])
    out = pair.pair_schedule(actions, hold, rotation)
    assert out.shape == (7, 24)
    np.testing.assert_array_equal(out[:2], actions[:2])
    np.testing.assert_array_equal(out[2:6, pair.RIGHT], actions[2:6, pair.RIGHT])
    np.testing.assert_array_equal(out[2:4, pair.LEFT], hold[:, pair.LEFT])
    np.testing.assert_array_equal(out[4:, pair.LEFT], 0.)
    np.testing.assert_array_equal(out[-1, 6:12], 0.)
    np.testing.assert_array_equal(out[-1, 15:18], 0.)
    np.testing.assert_array_equal(out[-1, 21:24], rotation)


def test_precontact_guard_rejects_solver_pair_on_an_unselected_pad():
    row = {"guard_failure": None,
           "rod": {"rod_mesh_contact_pairs": {"pairs": [{"mesh_name": "left.ring"}]}}}
    pair.guard_hand_contact(row)
    assert row["guard_failure"] == "pinch_precontact_unexpected_hand_load"
    row = {"guard_failure": None,
           "rod": {"rod_mesh_contact_pairs": {"pairs": [{"mesh_name": "table"}]}}}
    pair.guard_hand_contact(row)
    assert row["guard_failure"] is None
