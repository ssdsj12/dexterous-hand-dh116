import json

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_vision_policy_diagnostics import (
    VisionPolicyDiagnosticRecorder,
)


def test_policy_diagnostics_preserves_env_episode_and_ppo_fields(tmp_path):
    recorder = VisionPolicyDiagnosticRecorder(tmp_path / "policy.jsonl")
    rows = recorder.record(
        env_id=[2, 7], episode_id=[11, 13], iteration=42, policy_step=3,
        timestamp_s=[1.25, 1.25], action24=np.zeros((2, 24)),
        latent_actions=np.ones((2, 24)), latent_mean=np.zeros((2, 24)),
        log_std=np.full((2, 24), -2.),
        log_prob=[-1.2, -1.3], value=[.1, .2], advantage=[.4, -.5],
        returns=[.5, -.3], reward=[.2, -.1], done=[False, True],
        reward_terms={"lift": [0.3, 0.0], "failure": [0.0, -0.1]},
    )
    recorder.close()
    assert rows[1]["env_id"] == 7
    assert rows[1]["episode_id"] == 13
    assert rows[1]["advantage"] == pytest.approx(-.5)
    assert rows[1]["reward_terms"] == {"lift": 0.0, "failure": -0.1}
    saved = [json.loads(line) for line in (tmp_path / "policy.jsonl").read_text().splitlines()]
    assert saved == rows


def test_policy_diagnostics_reports_left_right_action_stats(tmp_path):
    recorder = VisionPolicyDiagnosticRecorder(tmp_path / "policy.jsonl")
    recorder.record(
        env_id=[0, 1], episode_id=[0, 0], iteration=1, policy_step=0,
        timestamp_s=[0., 0.], action24=np.vstack((np.ones(24), -np.ones(24))),
        latent_actions=None, latent_mean=None, log_std=np.zeros((2, 24)), log_prob=[0., 0.],
        value=[0., 0.], advantage=[0., 0.], returns=[0., 0.], reward=[0., 0.],
        done=[False, False], reward_terms={},
    )
    summary = recorder.summary()
    recorder.close()
    assert summary["rows"] == 2
    assert summary["by_hand"]["left"]["action_mean"] == [0.] * 12
    assert summary["by_hand"]["right"]["action_std"] == [1.] * 12
