from copy import deepcopy
import pytest
from scripts import audit_dh116_bilateral_native_evidence as native

def module():
    from scripts import replay_dh116_bilateral_contact_r0 as replay
    return replay

def receipt():
    return dict(duration_seconds=166.28976226993836,native_qualification=True,
        outer_exit_code=0,debugger_inferior_signal=None,qualification_scope=native.SCOPE,
        command_spec_sha256=native.RECIPE_SHA,code_config_sha256=native.CODE_SHA)

def test_new_native_scope_is_exact_and_does_not_grant_r0():
    assert module().require_source_scope(receipt()) is None
    result=module().replay_scope(29,232,2)
    assert result["new_physics_steps"]==0 and result["ppo_updates"]==0
    assert result["training_reward"] is None
    assert not result["teacher_admission"] and not result["native_qualification"]
    assert result["actual_source_native_execution_qualified"]
    assert not result["r0_native_reset_order_qualified"]

@pytest.mark.parametrize("key,value",[
    ("qualification_scope","right_owned_pinch_2_setup_232_guarded_contact_diagnostic_only"),
    ("command_spec_sha256","0"*64),("code_config_sha256","0"*64),
    ("outer_exit_code",-11),("duration_seconds",181),("native_qualification",False)])
def test_wrong_or_failed_execution_cannot_enter_bilateral_reward_replay(key,value):
    r=receipt();r[key]=value
    with pytest.raises(ValueError):module().require_source_scope(r)

@pytest.mark.parametrize("counts",[(29,231,2),(30,240,2),(29,232,1)])
def test_no_missing_or_expanded_source_steps(counts):
    with pytest.raises(ValueError):module().replay_scope(*counts)

def test_unsupported_contact_is_not_zero_filled_or_admitted():
    r=module().summary_gate([dict(diagnostic_reward=None,training_reward=None,
        blocked_terms=["grasp_geometry","material_anchor_slip"]),
        dict(diagnostic_reward=None,training_reward=None,blocked_terms=["instability"])])
    assert r["finite_diagnostic_reward_controls"]==0
    assert r["training_reward_controls"]==0
    assert r["full_reward_ready"] is False
    assert r["blocked_control_term_counts"]=={"grasp_geometry":1,"material_anchor_slip":1,"instability":1}

def test_unexpected_training_reward_fails_closed():
    with pytest.raises(ValueError):
        module().summary_gate([dict(diagnostic_reward=1.,training_reward=1.,blocked_terms=[])])
