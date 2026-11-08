"""Regression tests for the offline policy/physics analysis."""
import json

import pytest

from tools.analyze_vision_stage3_ppo_bias import analyze


def _fixture(run):
    with (run / "policy_ppo_diagnostics.jsonl").open("w") as policy, (
        run / "action_diagnostics.jsonl"
    ).open("w") as physics:
        for step, z in enumerate((-1.0, 0.0, 1.0)):
            for env in (0, 512):
                action = [0.0] * 24
                action[17] = z
                terms = {"up": 2 * z, "down": -3 * z, "constant": 7}
                if step != 1:
                    terms["sparse"] = 4 + z
                sample = dict(env_id=env, episode_id=0, timestamp_s=step / 60)
                policy.write(json.dumps(dict(
                    sample, policy_step=step, action24=action, log_std=[-1.0] * 24,
                    advantage=0, returns=1, value=z * z, reward=1, done=False,
                    reward_terms=terms,
                )) + "\n")
                physics.write(json.dumps(dict(sample, actual_palm_pose_m=[[0.0] * 3] * 2)) + "\n")


def test_each_reward_term_uses_its_own_values(tmp_path):
    _fixture(tmp_path)
    for style in analyze(tmp_path)["by_style"].values():
        corr = style["correlations_with_right_palm_z_action"]
        assert corr["value"] == pytest.approx(0)
        assert corr["reward_term:up"] == pytest.approx(1)
        assert corr["reward_term:down"] == pytest.approx(-1)
        assert corr["reward_term:constant"] is None
        assert corr["reward_term:sparse"] == pytest.approx(1)
        assert style["reward_terms"]["sparse"]["count"] == 2


def test_mismatched_policy_and_physics_cannot_be_combined(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / "action_diagnostics.jsonl"
    rows = path.read_text().splitlines()
    rows[0], rows[1] = rows[1], rows[0]
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="identity"):
        analyze(tmp_path)
