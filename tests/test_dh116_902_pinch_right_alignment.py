"""Regression checks for alternate measured right Pinch alignment."""
from pathlib import Path
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts import search_dh116_902_pinch_right_alignment as alternate


def test_adjacent_station_seeds_follow_the_archived_wire():
    nodes = np.zeros((41, 3))
    nodes[:, 1] = .01 * np.arange(41)
    nodes[:, 2] = -.005 * np.arange(41)
    options = alternate.station_seeds(nodes, 4, Rotation.identity(), radius=1)
    assert {option.station for option in options} == {3, 4, 5}
    for option in options:
        np.testing.assert_array_equal(option.translation, nodes[option.station] - nodes[4])
    tangents = [option.delta[3:6] for option in options if option.station == 4]
    np.testing.assert_allclose(tangents[1], -tangents[2])
    assert np.dot(tangents[1], nodes[5] - nodes[3]) > 0
    assert tangents[1][0] == 0


def test_adjacent_station_seeds_stay_on_the_right_side():
    nodes = np.column_stack((np.zeros(41), np.arange(41), np.zeros(41)))
    options = alternate.station_seeds(nodes, 19, Rotation.identity(), radius=1)
    assert {option.station for option in options} == {18, 19}



def test_warm_seeds_preserve_the_previous_guarded_posture_and_vary_its_roll():
    nodes = np.column_stack((np.zeros(41), .01 * np.arange(41), np.zeros(41)))
    initial = np.array([.001, .002, .003, .04, .05, .06, .02, .03, .04])
    warm, positive, negative = alternate.station_seeds(
        nodes, 4, Rotation.identity(), radius=0, initial_delta=initial)
    np.testing.assert_array_equal(warm.delta, initial)
    np.testing.assert_allclose(positive.delta - initial, [0., 0., 0., 0., .06, 0., 0., .015, 0.])
    np.testing.assert_allclose(negative.delta - initial, [0., 0., 0., 0., -.06, 0., 0., -.015, 0.])


def test_replay_rechecks_all_alternate_endpoints_against_all_right_hulls():
    receipt = os.environ.get("DH116_PINCH_ALIGNMENT_RECEIPT")
    if not receipt:
        pytest.skip("sealed 902 Pinch approach receipt is external")
    result = alternate.search(Path(receipt), radius=1, max_evaluations=3)
    assert result["sdk_range"] == [424, 432]
    assert result["baseline"]["pad_gaps_m"][0] == pytest.approx(.000644681245256404)
    assert result["measured"]["pad_stations"] == [4, 4]
    assert len(result["candidates"]) == 9
    assert {row["search_station"] for row in result["candidates"]} == {3, 4, 5}
    for candidate in result["candidates"]:
        assert len(candidate["rod_envelopes"]) == 22
        assert candidate["limiting_nonpad_hull"] in candidate["rod_envelopes"]
        assert len(candidate["pad_gaps_m"]) == 2
        assert candidate["pose_wxyz"][1] < 0.
        assert candidate["active_q_rad"][3:] == result["measured"]["active_q_rad"][3:]
    assert result["command_path_qualified"] is False
    assert result["native_contact_verified"] is False
