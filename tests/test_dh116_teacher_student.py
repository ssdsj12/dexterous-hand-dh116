import json
import numpy as np
import pytest

from go2_pvcnn.control.dh116.teacher_student import make_transition, write_replay


def test_replay_schema_and_hash(tmp_path):
    item = make_transition(obs_student=np.zeros(13), obs_teacher_privileged=np.zeros(14),
                           previous_action=np.zeros(12), mpc_action=np.ones(12),
                           action=np.ones(12), reward=1.0, done=False, phase="grasp")
    manifest = write_replay(tmp_path / "teacher.jsonl", [item], role="teacher")
    assert manifest["action_dim"] == 12
    row = json.loads((tmp_path / "teacher.jsonl").read_text())
    assert row["phase"] == "grasp"


def test_replay_rejects_arm_state_and_bad_phase():
    with pytest.raises(ValueError, match="Panda"):
        make_transition(obs_student={"arm": [0]}, obs_teacher_privileged=np.zeros(1),
                        previous_action=np.zeros(12), mpc_action=np.zeros(12),
                        action=np.zeros(12), reward=0, done=False, phase="lift")
    with pytest.raises(ValueError, match="phase"):
        make_transition(obs_student=np.zeros(1), obs_teacher_privileged=np.zeros(1),
                        previous_action=np.zeros(12), mpc_action=np.zeros(12),
                        action=np.zeros(12), reward=0, done=False, phase="approach")
