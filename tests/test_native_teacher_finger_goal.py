"""Regressions for source-bound finger-goal shaping in the native approach smoke."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_training_reward import (
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dh116_training/approach_wiring_smoke_v1.json"
REFERENCE = ROOT / "configs/dh116_training/pose_reference.json"


def goal_config(tmp_path):
    data = json.loads(CONFIG.read_bytes())
    data["finger_goal"] = dict(
        schema="dh116-teacher-finger-goal-v1", grasp="pinch",
        reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
    )
    p = tmp_path / "reward.json"
    p.write_text(json.dumps(data))
    return p


def test_source_bound_goal_exact_finger_names(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_teacher_pose_goal import measured_active_angles

    goal = load_config(goal_config(tmp_path)).finger_goal
    names = ("wrist_x", *reversed(ACTIVE))
    targets = goal.target_rad
    hands = {
        side: dict(joint_names=names, q=np.array([[9.0, *reversed(q)]]))
        for side, q in zip(("left", "right"), targets)
    }
    assert measured_active_angles(hands) == targets
    assert goal.max_errors_rad(targets) == (0.0, 0.0)
    moved = (targets[0], tuple(v + 0.1 for v in targets[1]))
    assert goal.max_errors_rad(moved) == pytest.approx((0.0, 0.1))


def test_corrupt_reference_sha_rejected(tmp_path):
    path = goal_config(tmp_path)
    data = json.loads(path.read_bytes())
    data["finger_goal"]["reference_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reference SHA"):
        load_config(path)


def test_arrival_paid_once_with_physical_terminal_precedence(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_training_reward import (
        Measurement, RewardEndpoint, SourceStamp, SubmittedRequest, TrainingReward,
    )

    reward = TrainingReward(load_config(goal_config(tmp_path)))

    def endpoint(tick, errors, *, guard=False):
        stamp = SourceStamp("episode-goal", tick, tick / 480, "cpu_fixture", "pose-test",
                            env_id=0, reset_uid="reset-goal", rod_uid="rod-goal")
        fields = {
            "distance": Measurement(0.01, "m", stamp),
            "numerical_health": Measurement(True, "dimensionless", stamp),
            "constraint_failure": Measurement(guard, "dimensionless", stamp),
            "finger_goal_max_error_left": Measurement(errors[0], "rad", stamp),
            "finger_goal_max_error_right": Measurement(errors[1], "rad", stamp),
        }
        return RewardEndpoint(stamp, ("material-0",), fields)

    def request(token, state):
        return SubmittedRequest(token, "policy", state.stamp, (0.0,) * 24,
                                "normalized_submitted_command24_preprojection")

    def window(last, index, error, *, guard=False):
        token = f"window-{index}"
        reward.begin(request(token, last), remaining_substeps=8)
        state = last
        for offset in range(1, 9):
            state = endpoint(last.stamp.physics_tick + offset, error, guard=guard and offset == 8)
            reward.observe(state)
        return state, reward.settle_once(token)

    reset = endpoint(0, (0.3, 0.3))
    reward.reset_from_verified_state(reset, previous_request=request("reset", reset))
    state, enter = window(reset, 0, (0.02, 0.04))
    assert enter.event == 0
    assert enter.phi_post_raw > enter.phi_pre
    state, arrive = window(state, 1, (0.02, 0.04))
    assert arrive.event == 2.0
    assert arrive.event_terms["finger_goal_arrived_once"]["raw"] is True
    assert not arrive.terminated
    state, leave = window(state, 2, (0.02, 0.08))
    assert leave.event == 0
    state, _ = window(state, 3, (0.02, 0.04))
    state, reenter = window(state, 4, (0.02, 0.04))
    assert reenter.event == 0
    _, failure = window(state, 5, (0.02, 0.04), guard=True)
    assert failure.event == -40
    assert failure.terminated


def test_only_identical_pinch_task_can_relocate_reset_profile(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import relocate_pinch_task_source

    relative = Path("configs/dh116_tasks/flexible_pinch_candidate_v1.json")
    origin = tmp_path / "producer" / relative
    consumer = tmp_path / "worktree" / relative
    for path in (origin, consumer):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"exact authored task")
    sha = hashlib.sha256(origin.read_bytes()).hexdigest()
    archived = {"backend_sha256": "backend", "sources": {str(origin): sha, "/same/solver": "same"}}
    current = {"backend_sha256": "backend", "sources": {str(consumer): sha, "/same/solver": "same"}}
    assert relocate_pinch_task_source(archived, current, consumer, relative) == current
    assert str(origin) in archived["sources"]
    origin.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        relocate_pinch_task_source(archived, current, consumer, relative)
