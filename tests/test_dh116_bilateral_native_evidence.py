from pathlib import Path
import importlib
from copy import deepcopy
import numpy as np
import pytest

def module():
    p=Path(__file__).parents[1]/"scripts/audit_dh116_bilateral_native_evidence.py"
    assert p.exists(),"same-episode bilateral native analysis missing"
    return importlib.import_module("scripts.audit_dh116_bilateral_native_evidence")

def test_only_new_bilateral_scope_qualifies():
    m=module()
    good=dict(native_qualification=True,outer_exit_code=0,debugger_inferior_signal=None,
        duration_seconds=166.3,qualification_scope=m.SCOPE)
    m.validate_receipt_scope(good)
    for field,value in (("qualification_scope","right_owned_pinch_2_setup_232_guarded_contact_diagnostic_only"),
                        ("native_qualification",False),("duration_seconds",180.1),("outer_exit_code",139)):
        row=deepcopy(good);row[field]=value
        with pytest.raises(ValueError):m.validate_receipt_scope(row)

@pytest.mark.parametrize("duration",[float("nan"),float("inf"),-1,True])
def test_duration_must_be_a_real_bounded_elapsed_measurement(duration):
    m=module()
    with pytest.raises(ValueError):
        m.validate_receipt_scope(dict(native_qualification=True,outer_exit_code=0,
          debugger_inferior_signal=None,duration_seconds=duration,qualification_scope=m.SCOPE))

def test_diagnostic_cannot_accept_a_self_attested_fixture_receipt(tmp_path):
    m=module()
    with pytest.raises(ValueError,match="exact consumed"):
        m.run(tmp_path,tmp_path/"fabricated.json","a"*64)

def test_actual_reset_positions_do_not_use_first_policy_measurement():
    m=module()
    hand=dict(body_names=["finger11_link","base_link"],body_poses_xyzw=np.array([[[0,0,.8,0,0,0,1],[.1,.2,.3,0,0,0,1]]]))
    assert np.array_equal(m.reset_base_position(hand),np.array([.1,.2,.3]))
    with pytest.raises(ValueError):m.reset_base_position(dict(hand,body_names=["x","x"]))

def test_incomplete_reset_transform_is_not_zero_filled():
    m=module()
    with pytest.raises(ValueError):m.reset_base_position(dict(body_names=["base_link"],body_poses_xyzw=np.zeros((1,1,6))))
    with pytest.raises(ValueError):m.reset_base_position(dict(body_names=["base_link"],body_poses_xyzw=np.full((1,1,7),np.nan)))

def test_same_episode_identity_is_required_for_bilateral_summary():
    m=module()
    identity=dict(run_uuid="r",env_id=0,episode_uid="e:one",reset_id=1,rod_uid="rod:one")
    m.require_episode(identity,dict(identity,physics_step=4))
    for k in identity:
        changed=dict(identity);changed[k]="wrong"
        with pytest.raises(ValueError):m.require_episode(identity,changed)
