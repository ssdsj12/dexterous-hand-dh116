import numpy as np
import pytest

from scripts.run_dh116_tripod_right_axis_calibration import make_tracking_packet, tracking_actions


def test_right_axis_calibration_keeps_left_parked_and_quiet_windows():
    motion = np.zeros((20, 24))
    motion[:, 6] = -.1
    motion[:, 22] = .2
    actions = tracking_actions(motion)
    np.testing.assert_array_equal(actions[[0, -1]], 0.)
    np.testing.assert_array_equal(actions[1:-1], motion)
    motion[:, 18] = .01
    with pytest.raises(ValueError, match="right-only"):
        tracking_actions(motion)


def test_right_axis_calibration_preserves_wrist_vector_bound():
    motion = np.zeros((20, 24))
    motion[:, 15:18] = [.8, .8, 0.]
    with pytest.raises(ValueError, match="vector"):
        tracking_actions(motion)


def test_current_calibration_packet_passes_the_existing_tracking_recorder():
    from scripts import verify_dh116_tripod_right_tracking as tracker

    prior = {"version": tracker.VERSION, "budget": tracker.BUDGET,
             "original_clearance_floor_m": .0021}
    motion = np.zeros((20, 24))
    motion[:, 6] = -.1
    motion[:, 22] = .2
    packet = make_tracking_packet(prior, {"actions": motion.tolist()}, {}, {"proved": True})

    assert packet["version"] == tracker.VERSION
    assert packet["diagnostic_version"] != tracker.VERSION
    assert len(packet["actions24"]) == 22
