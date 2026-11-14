"""Own LEFT source and candidate contract; no new native execution."""
from copy import deepcopy
import importlib
import numpy as np
import pytest

@pytest.fixture(scope="module")
def m():return importlib.import_module("scripts.solve_dh116_left_pinch_tracking")
@pytest.fixture(scope="module")
def packet(m):return m.load_packet()

def test_source_is_new_native_left_and_distinguishes_held_from_planned(m,packet):
    assert packet["side"]=="left"
    assert packet["source_sha256"]["receipt"]=="83a8758210fd6aa35da239c32819b91b9b6967307d30f7e380c67f965381cf11"
    assert packet["executed_left_actions_zero"] is True
    assert [s["identity"]["physics_step"] for s in packet["scenarios"]]==list(range(211,235))
    assert packet["original_joint_delta_rad"]==[0.,0.01855539082010854,0.0234063279676594,0.,0.,0.]
    np.testing.assert_allclose(np.array(packet["anchor"]["q_rad"])-packet["executed_anchor"]["q_rad"],packet["original_joint_delta_rad"],atol=1e-15)
    assert all(s["source_hand"]=="left" for s in packet["scenarios"])

def test_source_errors_reproduce_actual_own_hand(m,packet):
    for s in packet["scenarios"]:
        p,q=m.observed.apply_scenario(packet["executed_anchor"],s)
        np.testing.assert_allclose(p[:3],s["measured_root_wxyz"][:3],atol=1e-12)
        np.testing.assert_allclose(q,s["measured_active_q_rad"],atol=1e-12)
    bad=deepcopy(packet["scenarios"][0]);bad["joint_identity"]["env_id"]+=1
    with pytest.raises(ValueError):m.observed.apply_scenario(packet["executed_anchor"],bad)

def test_original_closure_delta_and_coupled_limits_preserved(m,packet):
    for x in (np.zeros(9),np.ones(9),-np.ones(9)):
        end=m.decode_candidate(packet,x);start=m.start_for(packet,end)
        np.testing.assert_allclose(np.array(end["q_rad"])-start["q_rad"],packet["original_joint_delta_rad"],atol=1e-12)
        for row in packet["scenarios"]:m.observed.apply_scenario(end,row)
    with pytest.raises(ValueError):m.decode_candidate(packet,np.ones(9)*1.01)

def test_uses_own_usd_and_original_geometric_guards(m,packet):
    g=m.Geometry(packet)
    row=g.evaluate(np.zeros(9),packet["scenarios"][-1:])
    assert g.side=="left"
    assert row["actual_contact_observed"] is row["native_ready"] is row["teacher_admission"] is False
    assert "nominal:table:source:finger13_link" in row["constraints"]
    assert "nominal:rod:source:finger13_link" in row["constraints"]
    assert "nominal:sdf_depth:finger14_link" in row["constraints"]
    assert len(g.grids)==5
