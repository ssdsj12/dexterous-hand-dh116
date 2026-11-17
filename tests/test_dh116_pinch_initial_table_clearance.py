"""CPU regression for exact locked cooked hull pose/clearance; never starts Isaac."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_dh116_pinch_initial_pose import (
    box_relation, segment_hull_distance, load_geometry, audit_initial_pose,
    evaluate_candidate, default_failed_output,
)

ROOT = Path(__file__).resolve().parents[1]


def cube():
    vertices = np.array([[x, y, z] for x in (-1., 1.)
                         for y in (-1., 1.) for z in (-1., 1.)])
    return vertices


def test_finite_table_separation_and_face_crossing():
    assert box_relation(cube(), [0, 0, 0], [1, 1, 1])["intersects"]
    assert not box_relation(cube()+[4, 0, -3], [0, 0, 0], [1, 1, 1])["intersects"]
    assert not box_relation(cube()+[0, 0, 2], [0, 0, 0], [1, 1, 1])["intersects"]


def test_segment_distance_is_continuous_not_rod_node_sampling():
    # The endpoints miss the cube; the continuous segment crosses it.
    assert segment_hull_distance(cube(), [-2, 0, 0], [2, 0, 0]) < 1e-8
    assert segment_hull_distance(cube(), [-2, 3, 0], [2, 3, 0]) == pytest.approx(2, abs=1e-8)
    assert segment_hull_distance(cube(), [3, 3, 0], [3, 3, 1]) == pytest.approx(np.sqrt(8), abs=1e-8)


@pytest.fixture(scope="module")
def geometry():
    return load_geometry(ROOT)


def test_failed_row_uses_exact_cooked_hulls_and_real_link_poses(geometry):
    result = audit_initial_pose(geometry, default_failed_output())
    pads = result["actual_pre_state"]["cooked_pads"]
    assert len(pads) == 5
    np.testing.assert_allclose([r["table"]["top_clearance_m"] for r in pads.values()],
        [.00240081083026454, .00000633341569954, -.01689385971596986,
         -.04109692465010524, -.06322230214950065], atol=1e-10, rtol=0)
    assert [name for name, r in pads.items() if r["table"]["intersects"]] == [
        "finger33_link", "finger43_link", "finger53_link"]
    assert result["source_fk_vs_actual"]["max_position_delta_m"] < 1e-6
    assert result["initial_geometry_passed"] is False
    assert result["native_execution_performed"] is False
    assert result["joint_readback"]["joint_names"] is None
    assert result["joint_readback"]["max_abs_delta_q_rad"] == pytest.approx(.480955, abs=1e-6)
    assert result["joint_readback"]["max_abs_post_qd_rad_s"] == pytest.approx(85.435, abs=.001)
    assert len(result["source_transform_chain"]["collision_shapes"]) == 17
    assert len(result["actual_pre_state"]["T_world_link_column_vector"]) == 5


def test_all_source_colliders_are_accounted_for(geometry):
    assert len(geometry["source_colliders"]) == 17
    assert len(geometry["cooked_pads"]) == 5
    assert geometry["all_17_live_cooked_available"] is False


def test_capture_mutation_rejected(tmp_path):
    from scripts.audit_dh116_pinch_initial_pose import checked_json
    p = tmp_path/"capture.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        checked_json(p, "0"*64)


def test_candidate_approach_is_bounded_and_cannot_qualify_native(geometry):
    path = ROOT/"configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json"
    candidate = json.loads(path.read_text())
    result = evaluate_candidate(geometry, candidate)
    assert result["native_execution_performed"] is False
    assert result["native_recipe_ready"] is False
    assert result["grasp_qualified"] is False
    assert result["geometry_candidate_passed"]
    assert result["source_geometry_initial_clearance_passed"]
    assert result["approach"]["max_active_delta_rad"] <= .025 + 1e-12
    assert result["approach"]["all_source_colliders_above_table"]
    assert result["approach"]["sample_count"] >= 161


def test_finite_box_edge_face_intersection_without_vertex_containment():
    # Two thin crossing boxes: no vertex of either is inside the other.
    a = cube()*[2, .1, .1]
    assert box_relation(a, [0, 0, 0], [.2, 4, .2])["intersects"]


def test_settling_envelope_includes_between_height_collision():
    from scripts.audit_dh116_pinch_initial_pose import convex_patch_closest
    v = cube()*.1+[0, 0, 1]
    result = convex_patch_closest(v, [0, -2, 0], [[0, 4, 0], [0, 0, 2]])
    assert result["distance_m"] < 1e-8


def test_style_missing_seed_never_falls_back_to_pinch():
    from scripts.audit_dh116_pinch_initial_pose import audit_style_seed
    for style in ("tripod", "power"):
        result = audit_style_seed(ROOT, style)
        assert not result["geometry_seed_passed"]
        assert result["blocked_reasons"] == ["missing_independent_"+style+"_seed"]
        assert not result["native_recipe_ready"]


def test_tripod_uses_its_own_canonical_seed():
    from scripts.audit_dh116_pinch_initial_pose import audit_style_seed
    result = audit_style_seed(ROOT, "tripod",
        ROOT/"configs/dh116_training/stereo_tripod_reset_20261105_v1.json")
    assert set(result["hands"]) == {"left", "right"}
    assert result["hands"]["left"]["active_q_rad"][0] == pytest.approx(.509811095111543)
    assert len(result["hands"]["right"]["source_all_17"]) == 17
    assert not result["geometry_seed_passed"]
    assert not result["native_recipe_ready"]


def test_candidate_source_and_limit_mutations_fail_closed(geometry):
    from copy import deepcopy
    candidate = json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json").read_text())
    changed = deepcopy(candidate)
    changed["source_sha256"]["usd"] = "0"*64
    with pytest.raises(ValueError, match="SHA256"):
        evaluate_candidate(geometry, changed)
    changed = deepcopy(candidate)
    changed["approach"]["active_delta_rad"][1] = .026
    with pytest.raises(ValueError, match="bounded limits"):
        evaluate_candidate(geometry, changed)


def test_joint_path_bound_covers_between_sample_geometry(geometry):
    from scripts.audit_dh116_pinch_initial_pose import world_vertices, path_displacement_bound
    q = np.asarray(geometry["active_q"])
    delta = np.array([0, .025, .025, 0, 0, 0])
    bounds = path_displacement_bound(geometry, q, delta)
    before = world_vertices(geometry, geometry["pose"], q)
    after = world_vertices(geometry, geometry["pose"], q+delta)
    for name in before:
        motion = np.linalg.norm(after[name]-before[name], axis=1).max()
        assert motion <= bounds[name]+1e-12
    assert bounds["base_link"] == 0
    assert bounds["finger14_link"] > 0


def test_cooked_table_overlap_cannot_hide_behind_clear_source(geometry):
    from scripts.audit_dh116_pinch_initial_pose import pose_matrix
    candidate = json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json").read_text())
    changed = dict(geometry)
    changed["cooked_pads"] = dict(geometry["cooked_pads"])
    name = "finger33_link"
    shape = dict(geometry["cooked_pads"][name])
    delta = pose_matrix(candidate["base_pose_wxyz"]) @ np.linalg.inv(pose_matrix(geometry["pose"]))
    transforms = geometry["fk"].poses(candidate["active_q_rad"])
    t = next(delta@value for key, value in transforms.items() if key.endswith("/"+name))
    shape["vertices"] = shape["vertices"] + t[:3, :3].T@np.array([0, 0, -.1])
    changed["cooked_pads"][name] = shape
    result = evaluate_candidate(changed, candidate)
    assert result["source_geometry_initial_clearance_passed"]
    assert not result["geometry_candidate_passed"]
    assert not result["cooked_geometry_initial_clearance_passed"]
