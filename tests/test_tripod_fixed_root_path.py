import importlib
import numpy as np
import pytest

def module():
    name="scripts.audit_dh116_tripod_fixed_root_path"
    assert importlib.util.find_spec(name) is not None, "fixed-root Tripod proof missing"
    return importlib.import_module(name)

def test_adaptive_lipschitz_proves_small_positive_endpoint_without_dense_grid():
    calls=[]
    def query(t):
        calls.append(t)
        return {"proximal":.001*(1-t)+.000005,"table":.003}
    r=module().certify_path(query,{"proximal":.001,"table":0},
                           {"proximal":0,"table":.0021},max_depth=12,max_samples=1025)
    assert r["proved"] is True
    assert r["minimum_clearance_margin_m"]>0
    assert len(calls)<100
    assert r["native_ready"] is False
    assert sorted(r["samples"])==sorted(set(calls))

def test_negative_measured_sample_rejects_path_without_being_hidden():
    r=module().certify_path(lambda t:{"x":(t-.5)**2-.01},{"x":1},{"x":0})
    assert r["proved"] is False
    assert r["status"]=="measured_violation"
    assert r["violations"][0]["fraction"]==.5

def test_budget_expiry_is_unresolved_not_clear():
    r=module().certify_path(lambda t:{"x":1e-10},{"x":1},{"x":0},max_depth=2,max_samples=9)
    assert r["proved"] is False
    assert r["status"]=="unresolved"
    assert r["unresolved_intervals"]

def test_changing_shape_identity_is_rejected():
    def query(t):return {"a" if t==0 else "b":1}
    with pytest.raises(ValueError):module().certify_path(query,{"a":1},{"a":0})

def test_original_fixed_root_three_digit_closure():
    p=dict(base_pose_wxyz=[0,0,.4,1,0,0,0],active_q_rad=[.4,.4,.5,.5,.7,.95],
           independent_style="tripod",side="left",
           independent_seed_sha256=module().TRIPOD_SHA)
    r=module().candidate(p)
    assert r["initial_q_rad"]==pytest.approx([.4,.37,.47,.47,.7,.95])
    assert r["endpoint_q_rad"]==p["active_q_rad"]
    assert r["active_velocity_rad_s"]==pytest.approx([0,.09,.09,.09,0,0])
    assert r["base_pose_wxyz"]==p["base_pose_wxyz"]
    assert r["native_ready"] is False

def test_wrong_grasp_is_not_allowed_to_reuse_initial_pose():
    p=dict(base_pose_wxyz=[0,0,.4,1,0,0,0],active_q_rad=[.4,.4,.5,.5,.7,.95],
           independent_style="pinch",side="left",independent_seed_sha256=module().TRIPOD_SHA)
    with pytest.raises(ValueError):module().candidate(p)
