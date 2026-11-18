"""Immutable actual prefix replay, not synthetic native qualification."""
import hashlib,json
from pathlib import Path
import pytest

RESULT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/bilateral-r0-evidence-20261114-run01/result.json")
SHA="9720b12aecd6aeb98524d42e2ea6e04a795b06b4b1b08211b1fc253dbb0158e3"

@pytest.fixture(scope="module")
def actual():
    assert not RESULT.is_symlink()
    raw=RESULT.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SHA
    return json.loads(raw)

def test_actual232_samples_29_complete_controls_without_reexecution(actual):
    assert actual["status"]=="complete"
    assert actual["source_receipt_sha256"]=="8ff5b25afb5a46a1cb57899b5aedf42c1e0e766a5bf30d4628e229bd62850db8"
    assert actual["approach_valid_samples"]==actual["recorded_physics_rows"]==232
    assert actual["settled_control_count"]==29
    assert actual["new_physics_steps"]==actual["ppo_updates"]==0
    assert actual["native_imports"]==[]
    assert actual["approach_last_m"]==[0.,0.]
    assert actual["approach_first_m"]==pytest.approx([.0011492206249386072,.001067609991878271])

def test_missing_stability_and_slip_block_every_complete_reward(actual):
    assert actual["blocked_physical_term_counts"]["instability"]==232
    assert actual["blocked_physical_term_counts"]["hold_progress"]==232
    assert actual["blocked_physical_term_counts"]["material_anchor_slip"]==45
    assert actual["blocked_control_term_counts"]["material_anchor_slip"]==6
    assert actual["finite_diagnostic_reward_controls"]==actual["training_reward_controls"]==0
    assert all(c["training_reward"] is None and c["diagnostic_reward"] is None for c in actual["controls"])
    assert actual["teacher_admission"] is False and actual["full_reward_ready"] is False

def test_partial_state_and_readonly_reset_are_not_native_teacher_contract(actual):
    assert actual["common_valid_count"]==66 and actual["common_total"]==371
    assert actual["common_missing_policy"]=="NaN_with_false_validity"
    assert not actual["r0_native_reset_order_qualified"]
    assert not actual["full_episode_evaluated"] and not actual["native_qualification"]
    assert actual["actual_source_native_execution_qualified"]
    assert actual["original_frozen_task_coefficients_unchanged"]
