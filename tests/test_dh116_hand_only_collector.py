from pathlib import Path
import json
import numpy as np


def test_tabletop_mpc_collector_can_execute_student_and_log_mpc_reference():
    source = (Path(__file__).parents[1] / "scripts/collect_dh116_hand_only_mpc_isaac.py").read_text()
    assert 'parser.add_argument("--student-checkpoint"' in source
    assert "LinearImitationPolicy.load" in source
    assert "student_policy.predict(obs_student)" in source
    assert "mpc_action=mpc_action" in source
    assert 'role="student"' in source
    assert "planner.plan_trajectory(" in source
    assert "trajectory_action_index(" in source


def test_collector_scores_each_action_against_mpc_and_logs_reward_terms():
    source = (Path(__file__).parents[1] / "scripts/collect_dh116_hand_only_mpc_isaac.py").read_text()
    assert "compute_hand_only_reward(" in source
    assert "action=action" in source and "mpc_action=mpc_action" in source
    assert 'reward_terms' in source
    assert "lift_reference_m=lift_reference_m" in source


def test_hand_only_viewer_supports_trajectory_checkpoint_and_visible_gui():
    source = (Path(__file__).parents[1] / "scripts/verify_dh116_hand_only_wire.py").read_text()
    assert 'LinearImitationPolicy.load(args.checkpoint, expected_role="student")' in source
    assert "policy.predict(observation)" in source
    assert "sim.set_camera_view" in source
    assert "DomeLightCfg" in source
    assert "app.update()" in source


def test_candidate3_pose_maps_to_hand_only_roots_joints_and_vertical_wire():
    from go2_pvcnn.control.dh116.hand_only_scene import candidate3_hand_only_layout

    root = Path(__file__).parents[1]
    pose = json.loads((root / "runs/forward_feasibility3/pose.json").read_text())
    layout = candidate3_hand_only_layout(pose)
    np.testing.assert_allclose(layout["hand_roots"]["left"],
                               [0.3815475, 0.23732965, 0.36120843,
                                -0.00524514, 0.99977878, -0.00643838, -0.01932409], atol=1e-6)
    np.testing.assert_allclose(layout["hand_roots"]["right"],
                               [0.3793522, -0.23721493, 0.36181091,
                                0.00606007, 0.99966999, 0.00472094, -0.02451318], atol=1e-6)
    np.testing.assert_allclose(layout["active_joints"]["left"], pose["states"]["left"][7:])
    np.testing.assert_allclose(layout["active_joints"]["right"], pose["states"]["right"][7:])
    assert layout["wire_center"] == (0.43, 0.0, 0.308)
    assert layout["wire_axis"] == "Y"


def test_table_clearance_uses_all_collision_body_heights_and_reports_block_reason():
    from go2_pvcnn.control.dh116.hand_only_scene import assess_table_clearance

    report = assess_table_clearance({"left": [0.36, 0.39], "right": [0.34, 0.41]}, 0.30, 0.01)
    assert np.isclose(report["minimum_clearance_m"], 0.04)
    assert report["passed"] is True
    blocked = assess_table_clearance({"left": [0.299, 0.4], "right": [0.35]}, 0.30, 0.01)
    assert blocked["passed"] is False
    assert blocked["failure_reason"] == "tabletop_penetration"


def test_native_pose_summary_maps_world_hand_roots_for_hand_only_scene():
    from go2_pvcnn.control.dh116.hand_only_scene import native_pose_hand_only_layout

    root = Path(__file__).parents[1]
    payload = json.loads((root / "runs/reference_aligned_native_20261023_fix1/summary.json").read_text())
    layout = native_pose_hand_only_layout(payload)
    np.testing.assert_allclose(layout["hand_roots"]["left"],
                               payload["resolved_poses"]["left"]["dh116_base_pose_w"])
    np.testing.assert_allclose(layout["hand_roots"]["right"],
                               payload["resolved_poses"]["right"]["dh116_base_pose_w"])
    np.testing.assert_allclose(layout["active_joints"]["left"], [0.33324617, 0.47555462, 0.66254604, 0.2, 0.2, 0.2])


def test_native_pose_roll_keeps_grasp_axis_fixed_and_raises_low_side():
    from go2_pvcnn.control.dh116.hand_only_scene import rotate_roots_about_grasp_axes

    roots = {"left": np.array([0.5, 0.2, 0.27, 1, 0, 0, 0], dtype=float),
             "right": np.array([0.5, -0.2, 0.36, 1, 0, 0, 0], dtype=float)}
    contacts = {"left": np.array([[0.43, 0.2, 0.309], [0.42, 0.2, 0.309]]),
                "right": np.array([[0.43, -0.2, 0.309], [0.42, -0.2, 0.309]])}
    rotated = rotate_roots_about_grasp_axes(roots, contacts, {"left": 76, "right": 82})
    for side in ("left", "right"):
        assert np.isclose(np.linalg.norm(rotated[side][3:]), 1.0)
        assert not np.allclose(rotated[side][3:], roots[side][3:])
        old_rotation = np.eye(3)
        new_rotation = _quat_matrix_for_test(rotated[side][3:])
        local_points = (contacts[side] - roots[side][:3]) @ old_rotation
        reconstructed = local_points @ new_rotation.T + rotated[side][:3]
        np.testing.assert_allclose(reconstructed, contacts[side], atol=1e-10)


def _quat_matrix_for_test(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])


def test_hand_only_viewer_exposes_candidate3_layout_option():
    source = (Path(__file__).parents[1] / "scripts/verify_dh116_hand_only_wire.py").read_text()
    assert 'parser.add_argument("--candidate3-pose"' in source
    assert "candidate3_hand_only_layout" in source
    assert "native_pose_hand_only_layout" in source
    assert 'parser.add_argument("--native-pose-summary"' in source
    assert 'axis="Y"' in source
    assert 'sim.set_camera_view' in source
    assert "reset_positions[:, reset_ids]" in source
    assert 'candidate["active_joints"][side]' in source
    assert "collision_enabled=False" not in source
    assert source.count('RigidObjectCfg(prim_path="/World/WireProxy"') == 1
    assert "single_candidate3_reference_orange" in source
    assert source.index("hand.reset(); sensor.reset()") < source.index("hand.write_joint_state_to_sim(reset_positions")
    assert "hand.set_joint_position_target(reset_positions[:, reset_active_ids]" in source
    assert "minimum_table_clearance_m" in source
    assert "candidate_pose_table_penetration" in source
    assert 'clearance_report["failure_reason"]' in source
    assert "table_clearance_passed" in source
    assert 'result["failure_reason"] = "wire_initially_below_table_surface"' in source
    assert "action_rows = () if args.pose_check_only or not result.get(\"table_clearance_passed\", True) else rows" in source
