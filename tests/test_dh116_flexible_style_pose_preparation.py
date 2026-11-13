"""Independent Tripod / Power CPU geometry preparation; no simulator or training."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_dh116_flexible_style_poses import (
    TRIPOD_SHA, load_tripod_seed, rotate_seed_pose, close_samples,
    clearance_corrected_pose, tripod_candidates, fixed_root_gap_lower_bounds,
)
from scripts.audit_dh116_pinch_initial_pose import load_geometry
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, active_limits

ROOT = Path(__file__).resolve().parents[1]


def test_rotation_about_pivot_preserves_distance_and_normalizes_quaternion():
    pose = [1, 2, 3, 1, 0, 0, 0]
    result = rotate_seed_pose(pose, [0, 0, 0], 20, 30)
    assert np.linalg.norm(result[:3]) == pytest.approx(np.linalg.norm(pose[:3]))
    assert np.linalg.norm(result[3:]) == pytest.approx(1)


def test_closure_grid_honors_original_upper_limits_and_fixed_count():
    lower, upper = active_limits()
    q = (lower+upper)/2
    q[2] = upper[2]-.017
    values = close_samples(q, 2)
    assert len(values) == 9
    assert values[0] == 0
    assert values[-1] == pytest.approx(.017)
    assert np.all(np.diff(values) >= 0)
    with pytest.raises(ValueError):
        close_samples(q, 6)


@pytest.fixture(scope="module")
def left_geometry():
    return load_geometry(ROOT, side="left")


def test_tripod_requires_independent_pinned_canonical_seed(tmp_path):
    path = tmp_path/"seed.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        load_tripod_seed(path)
    seed = load_tripod_seed(ROOT/"configs/dh116_training/stereo_tripod_reset_20261105_v1.json")
    assert seed["source_sha256"] == TRIPOD_SHA
    assert seed["hands"]["left"]["active_q_rad"][0] == pytest.approx(.509811095111543)


def test_clearance_construction_accounts_for_proximal_and_cooked(left_geometry):
    seed = load_tripod_seed(ROOT/"configs/dh116_training/stereo_tripod_reset_20261105_v1.json")["hands"]["left"]
    result = clearance_corrected_pose(left_geometry, seed["base_pose_wxyz"], seed["active_q_rad"])
    assert result["original_minimum_clearance_m"] == pytest.approx(-.005856137363027758)
    assert result["vertical_shift_m"] > .0078
    assert result["minimum_clearance_m"] == pytest.approx(.002, abs=1e-10)


def test_tripod_enumeration_is_finite_independent_and_deterministic(left_geometry):
    seed = load_tripod_seed(ROOT/"configs/dh116_training/stereo_tripod_reset_20261105_v1.json")["hands"]["left"]
    rows = list(tripod_candidates(left_geometry, seed))
    assert len(rows) == 225
    assert len({r["candidate_id"] for r in rows}) == 225
    assert all(r["independent_style"] == "tripod" for r in rows)
    assert rows[0] == next(tripod_candidates(left_geometry, seed))
    assert all(r["active_joint_names"] == list(ACTIVE) for r in rows)


def test_fixed_root_distance_bound_cannot_be_a_qualification():
    report = fixed_root_gap_lower_bounds(
        {"thumb": .058, "index": .020}, {"thumb": .012, "index": .008})
    assert report["gap_lower_bound_m"]["thumb"] == pytest.approx(.045998)
    assert report["all_required_pass_necessary_reach_bound"] is False
    assert report["native_ready"] is False
    assert report["teacher_admitted"] is False


def test_selected_seed_prefers_initial_clearance_over_unsafe_low_gap():
    from scripts.prepare_dh116_flexible_style_poses import select_prepared_candidate
    unsafe = {"candidate": {"candidate_id": "a", "max_required_endpoint_abs_gap_m": 0.},
              "path": {"geometry_preparation_passed": False, "reset_geometry_passed": False,
                       "blockers": ["initial_overlap"]}}
    clear = {"candidate": {"candidate_id": "b", "max_required_endpoint_abs_gap_m": .0004},
             "path": {"geometry_preparation_passed": False, "reset_geometry_passed": True,
                      "blockers": ["approach_overlap", "opposition_unknown"]}}
    assert select_prepared_candidate([unsafe, clear]) is clear
    assert select_prepared_candidate([unsafe])["path"]["reset_geometry_passed"] is False
