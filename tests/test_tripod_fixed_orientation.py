"""CPU reference candidates only; no SDK/physics or actuator changes."""
import numpy as np
import pytest
from scripts.prepare_dh116_tripod_fixed_orientation import compile_candidate, delayed_state, readiness

def parent():
    return dict(end_pose_wxyz=[.4,.1,.38,1.,0.,0.,0.],
        q0_rad=[.4,.3,.6,.7,.8,.9],q1_rad=[.36,.32,.64,.708,.8,.9])
def test_candidate_uses_original24d_with_no_rotation_or_right_motion():
    p=compile_candidate(parent(),raise_m=.0015,steps=200)
    a=np.array(p["actions"])
    assert a.shape==(25,24)
    assert (a[:,18:]==0).all() and (a[:,6:12]==0).all() and (a[:,15:18]==0).all()
    assert abs(a[:,14][0]+.36)<1e-12
    assert p["seed_pose_wxyz"][2]==pytest.approx(.3815)
    assert p["native_ready"] is False
@pytest.mark.parametrize("raise_m",[0,-.001,float("nan"),.1])
def test_invalid_or_rate_exceeding_offset_rejected(raise_m):
    with pytest.raises(ValueError):compile_candidate(parent(),raise_m=raise_m,steps=200)
def test_finger_and_translation_delay_are_independent():
    p=compile_candidate(parent(),raise_m=.0015,steps=200)
    pose,q=delayed_state(p,100,[10,0,5,0,0,0],6)
    np.testing.assert_allclose(q,np.array(p["q0_rad"])+np.array([.45,.5,.475,.5,.5,.5])*(np.array(p["q1_rad"])-p["q0_rad"]))
    assert pose[2]==pytest.approx(.3815-.0015*.47)
    assert pose[3:].tolist()==p["end_pose_wxyz"][3:]
@pytest.mark.parametrize("step,delay",[(-1,0),(0,-1),(300,0),(10,float("inf"))])
def test_delay_not_a_budget_extension(step,delay):
    p=compile_candidate(parent(),raise_m=.0015,steps=200)
    with pytest.raises(ValueError):delayed_state(p,step,[delay]*6,0)
def test_sampled_lag_screen_never_grants_native_qualification():
    r=readiness(nominal=True,sampled=True)
    assert r["native_ready"] is r["teacher_admission"] is False
    assert r["continuous_tracking_envelope_proven"] is False
    assert "tracking" in r["blocked_by"]

def test_proof_float_sample_keys_survive_owned_serialization():
    from scripts.prepare_dh116_tripod_fixed_orientation import pack_proof
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    proof=dict(samples={0.0:{"gap":.01},.125:{"gap":.02}},proved=True)
    packet=pack_proof(proof)
    assert encode_owned(packet)["samples"]==[
        {"fraction":0.0,"measurements":{"gap":.01}},
        {"fraction":.125,"measurements":{"gap":.02}}]
    assert set(proof["samples"])=={0.,.125}
def test_cpu_candidate_config_does_not_silently_ignore_guard_edits():
    from pathlib import Path
    import json
    from scripts.prepare_dh116_tripod_fixed_orientation import validate_config,CONFIG
    config=json.loads((Path(__file__).resolve().parents[1]/CONFIG).read_text())
    validate_config(config)
    for key,value in [("table_clearance_m",0.0),("physics_hz",240),("teacher_admission",True),
                      ("native_execution_enabled",True),("maximum_new_physics_steps",1)]:
        wrong=dict(config);wrong[key]=value
        with pytest.raises(ValueError):validate_config(wrong)
