from pathlib import Path


def test_tabletop_mpc_collector_can_execute_student_and_log_mpc_reference():
    source = (Path(__file__).parents[1] / "scripts/collect_dh116_hand_only_mpc_isaac.py").read_text()
    assert 'parser.add_argument("--student-checkpoint"' in source
    assert "LinearImitationPolicy.load" in source
    assert "student_policy.predict(obs_student)" in source
    assert "mpc_action=mpc_action" in source
    assert 'role="student"' in source
    assert "planner.plan_trajectory(" in source
    assert "trajectory_action_index(" in source
