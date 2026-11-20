import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_dh116_tripod_right_tracking import build_tracking_plan

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path.home() / ".local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119"


@pytest.fixture
def original():
    endpoint = json.loads((EVIDENCE / "oakink-tripod-material-roll-final.json").read_bytes())
    reference = json.loads((EVIDENCE / "dc6-tripod-band-real.json").read_bytes())
    return endpoint, reference


def test_right_tracking_plan_keeps_left_parked_and_proves_noncontact(original):
    endpoint, reference = original
    result = build_tracking_plan(endpoint, reference, project=ROOT)
    a = np.asarray(result["actions24"])
    assert a.shape == (22, 24)
    assert np.count_nonzero(a[:, np.r_[0:6, 12:15, 18:21]]) == 0
    assert np.count_nonzero(a[0]) == np.count_nonzero(a[-1]) == 0
    assert np.all(a[1:-1, 7:10] > 0) and np.all(a[1:-1, 17] < 0)
    assert result["budget"]["sdk_steps"] == 178
    assert result["proved_right_sweep"]["proved"] is True
    assert result["minimum_rod_gap_m"]["right"] >= .002
    assert result["minimum_rod_gap_m"]["left"] >= .002
    assert result["minimum_table_gap_m"]["right"] >= .0029
    assert result["bilateral_y_gap_m"] >= .04
    assert result["measured_right_tracking"] is False and result["native_qualification"] is False


def test_unsafe_change_to_right_wrist_is_rejected(original):
    endpoint, reference = original
    reference["hands"]["right"]["executed_target_end_pose_wxyz"][2] -= .05
    with pytest.raises(ValueError, match="right.*clearance|right.*projection|right.*pose"):
        build_tracking_plan(endpoint, reference, project=ROOT)


def test_straight_control_changes_only_rod_x_and_recertifies(original):
    endpoint, reference = original
    curved = build_tracking_plan(endpoint, reference, project=ROOT)
    straight = build_tracking_plan(endpoint, reference, project=ROOT, straight_control=True)
    a, b = (np.asarray(row["rod_nodes_m"]) for row in (curved, straight))
    assert np.ptp(a[:, 0]) > .003
    np.testing.assert_array_equal(b[:, 0], np.full(len(b), .432))
    np.testing.assert_array_equal(a[:, 1:], b[:, 1:])
    assert straight["reset_seed"] == curved["reset_seed"]
    assert straight["actions24"] == curved["actions24"]
    assert straight["budget"] == curved["budget"]
    assert straight["rod_initial_state"] == "straight_free_rod_x_432mm_control"
    assert straight["proved_right_sweep"]["proved"] is True
    assert straight["minimum_rod_gap_m"] != curved["minimum_rod_gap_m"]
    assert min(straight["minimum_rod_gap_m"].values()) >= .002
    assert straight["native_qualification"] is False
