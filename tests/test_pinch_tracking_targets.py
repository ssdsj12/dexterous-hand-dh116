"""Empirical endpoint robustness is CPU prediction, never new native evidence."""
from copy import deepcopy
import importlib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

@pytest.fixture(scope="module")
def m():
    return importlib.import_module("scripts.solve_dh116_pinch_tracking_targets")

@pytest.fixture(scope="module")
def packet(m):
    return m.load_observed_scenarios()

def test_source_preserves_correlated_clock_and_all_hold_steps(m,packet):
    assert [s["identity"]["physics_step"] for s in packet["scenarios"]]==list(range(211,235))
    assert packet["source_sha256"]["receipt"]==m.RECEIPT_SHA
    assert len({s["identity"]["episode_uid"] for s in packet["scenarios"]})==1
    for s in packet["scenarios"]:
        assert np.shape(s["rod_nodes_m"])==(41,3)
        assert np.shape(s["active_error_rad"])==(6,)
        assert s["empirical_only"] is True

def test_wrong_source_sha_is_rejected(m):
    with pytest.raises(ValueError,match="SHA"):
        m.load_observed_scenarios(expected_sha="0"*64)

def test_apply_reproduces_measured_root_and_active_joints(m,packet):
    for s in packet["scenarios"]:
        p,q=m.apply_scenario(packet["anchor"],s)
        np.testing.assert_allclose(p[:3],s["measured_root_wxyz"][:3],atol=1e-12,rtol=0)
        np.testing.assert_allclose(Rotation.from_quat(np.array(p)[[4,5,6,3]]).as_matrix(),
            Rotation.from_quat(np.array(s["measured_root_wxyz"])[[4,5,6,3]]).as_matrix(),atol=1e-12,rtol=0)
        np.testing.assert_allclose(q,s["measured_active_q_rad"],atol=1e-12,rtol=0)

def test_mixed_episode_or_tick_is_rejected_before_prediction(m,packet):
    row=deepcopy(packet["scenarios"][0])
    row["rod_identity"]["physics_step"]+=1
    with pytest.raises(ValueError,match="clock"):m.apply_scenario(packet["anchor"],row)
    row=deepcopy(packet["scenarios"][0])
    row["joint_identity"]["episode_uid"]="different"
    with pytest.raises(ValueError,match="clock"):m.apply_scenario(packet["anchor"],row)

def test_rotation_composition_uses_world_residual_not_component_addition(m,packet):
    row=deepcopy(packet["scenarios"][0])
    row["world_rotation_error_xyzw"]=Rotation.from_euler("x",.13).as_quat().tolist()
    p,q=m.apply_scenario(packet["anchor"],row)
    wanted=Rotation.from_euler("x",.13)*Rotation.from_quat(np.array(packet["anchor"]["root_pose_wxyz"])[[4,5,6,3]])
    np.testing.assert_allclose(Rotation.from_quat(np.array(p)[[4,5,6,3]]).as_matrix(),wanted.as_matrix(),atol=1e-12)

def test_candidate_keeps_same_finger_motion_and_original_bounds(m,packet):
    for x in (np.zeros(9),np.ones(9),-np.ones(9)):
        end=m.decode_candidate(packet["anchor"],x,packet["scenarios"])
        start=m.original_start(end)
        np.testing.assert_allclose(np.array(end["q_rad"])-start["q_rad"],
            packet["original_joint_delta_rad"],atol=1e-12)
        for s in packet["scenarios"]:m.apply_scenario(end,s)
    with pytest.raises(ValueError,match="bounded"):
        m.decode_candidate(packet["anchor"],np.ones(9)*1.01,packet["scenarios"])

def test_actual_baseline_exposes_original_pad_failure(m,packet):
    engine=m.Geometry(packet)
    row=engine.evaluate(np.zeros(9),[packet["scenarios"][-1]])
    assert row["admission"]["static_constraints_passed"] is False
    for p in row["scenarios"]["234"]["pads"].values():
        assert p["ellipse_value"]>1
    assert row["actual_contact_observed"] is False
    assert row["native_ready"] is False
    assert row["teacher_admission"] is False

def test_nominal_success_cannot_replace_all_observed_scenarios(m):
    nominal={"constraints":{"region":.1}}
    observed=[{"constraints":{"region":.01}},{"constraints":{"region":-.001}}]
    assert m.admission(nominal,observed)["empirical_endpoint_passed"] is False
    assert m.admission(nominal,[])["empirical_endpoint_passed"] is False
    assert m.admission(nominal,observed[:1])["native_ready"] is False

def test_explicit_native_region_and_touch_feasibility_do_not_relax_safety(m):
    row={"constraints":{"table:x":-.00001,"rod:y":-.00002,"sdf_depth:finger14_link":.0001,
        "sdf_target:finger14_link":-.00005,"region:finger14_link":-.00004},
        "pads":{"finger14_link":{"sdf_gap_m":-.00005,"ellipse_value":.99}}}
    original=deepcopy(row)
    revised=m.physical_target_constraints(row)
    assert revised["sdf_target:finger14_link"]==.00005
    assert revised["region:finger14_link"]==pytest.approx(.00001)
    for key in ("table:x","rod:y","sdf_depth:finger14_link"):
        assert revised[key]==row["constraints"][key]
    assert row==original
    assert m.admission({"constraints":revised},[{"constraints":revised}])["empirical_endpoint_passed"] is False

def test_constraint_refinement_keeps_failed_intermediate_native_ticks(m):
    prior={"all_observed_validation":{"scenarios":{
        "211":{"constraints":{"contact":1e-7}},
        "212":{"constraints":{"contact":-1.2e-7}},
        "214":{"constraints":{"contact":-5.7e-8}},
        "222":{"constraints":{"contact":1e-7}},
        "234":{"constraints":{"contact":1e-7}}}}}
    assert m.refinement_steps(prior)==[211,212,214,222,234]
    prior["all_observed_validation"]["scenarios"]["213"]={"constraints":{"contact":float("nan")}}
    with pytest.raises(ValueError,match="finite"):m.refinement_steps(prior)
