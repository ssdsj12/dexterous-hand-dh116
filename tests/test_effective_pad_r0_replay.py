from copy import deepcopy
import pytest
from scripts.replay_dh116_effective_pad_r0 import configuration, summary_gate, prefix_scope

def test_configuration_cannot_start_native_or_training():
    c = configuration()
    assert c["new_physics_steps"] == c["ppo_updates"] == 0
    assert c["enabled_by_default"] is False and c["teacher_admission"] is False
    assert c["training_reward_enabled"] is False
    assert c["expected_measured_steps"] == 232

def test_missing_stability_still_blocks_r0_without_inventing_a_reward():
    result = summary_gate([dict(training_reward=None, diagnostic_reward=None,
        blocked_terms=["hold_progress", "instability"])])
    assert result["full_reward_ready"] is False
    assert result["finite_diagnostic_reward_controls"] == 0
    assert result["blocked_control_term_counts"] == {"hold_progress": 1, "instability": 1}

def test_short_complete_prefix_is_not_success_or_new_native():
    r = prefix_scope(29, 232, 2)
    assert r["full_episode_evaluated"] is False
    assert r["actual_record_duration_s"] < 5
    assert r["native_qualification"] is False and r["teacher_admission"] is False

@pytest.mark.parametrize("counts", [(29, 231, 2), (29, 232, 3), (30, 232, 2)])
def test_prefix_never_expands_consumed_budget(counts):
    with pytest.raises(ValueError): prefix_scope(*counts)
