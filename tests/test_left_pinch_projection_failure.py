"""Failed-native evidence is retained without promoting a partial trajectory."""
import importlib
from copy import deepcopy
import numpy as np
import pytest

def audit():
    return importlib.import_module("scripts.audit_dh116_left_pinch_projection_failure")

def projection():
    request=np.zeros((1,2,3));request[0,0,2]=-.00118125
    applied=request.copy();applied[0,0,2]=-.001180440967108056
    roots=np.zeros((1,2,7));roots[...,0:3]=applied;roots[...,3]=1
    return dict(palm_offsets_m=applied,pre_projection_palm_offsets_m=request,
        safe_z_offset_m=np.array([[-.001180440967108056,-.002]]),
        safe_z_clipped=np.array([[True,False]]),projection_reasons=np.array([[2,0]]),
        projected_root_poses=roots,projection_source="tabletop_command_kernels.palm_transport+per_hand_safe_z:policy_owned_default_clearance")

def test_safe_projection_is_not_numerical_roundoff_or_table_penetration():
    m=audit();p=projection()
    r=m.projection_decomposition(p,requested=p["pre_projection_palm_offsets_m"],
        submitted=p["projected_root_poses"])
    assert r["classification"]=="original_safe_z_projection"
    assert r["delta_mm"][0][2]==pytest.approx(.000809032891944)
    assert r["table_penetration_proven"] is False
    assert r["independent_clearance_recomputed"] is False

@pytest.mark.parametrize("bad",["flag","reason","source","submitted","xy","nan"])
def test_projection_inconsistent_evidence_rejected(bad):
    m=audit();p=projection();submitted=p["projected_root_poses"].copy()
    if bad=="flag":p["safe_z_clipped"][0,0]=False
    if bad=="reason":p["projection_reasons"][0,0]=0
    if bad=="source":p["projection_source"]="invented"
    if bad=="submitted":submitted[0,0,2]+=.001
    if bad=="xy":p["palm_offsets_m"][0,0,0]=.001
    if bad=="nan":p["safe_z_offset_m"][0,0]=float("nan")
    with pytest.raises(ValueError):m.projection_decomposition(p,
        requested=p["pre_projection_palm_offsets_m"],submitted=submitted)

def test_completed_app_close_does_not_promote_nonzero_process_exit():
    m=audit()
    r=m.lifecycle(dict(outer_exit_code=1,debugger_inferior_exit_code=1,
        debugger_inferior_signal=None,debugger_status_verified=True),True,True,True)
    assert r["app_close_returned"]
    assert r["native_views_dead_after_stop"]
    assert not r["natural_exit_zero"]
    assert not r["native_qualified"]

def test_fixture_interrupted_window_cannot_be_extended():
    m=audit()
    r=m.region_windows([0,1,3],1/5760)
    assert r["longest_samples"]==2
    assert r["longest_duration_s"]==pytest.approx(2/5760)
    with pytest.raises(ValueError):m.region_windows([1,1],1/5760)

def test_locked_failed_native_has_exact_first_projection_and_terminal():
    m=audit();r=m.run()
    assert r["physics_steps"]==176 and r["measurement_rows"]==174
    assert r["last_geometry_step"]==175 and r["terminal_geometry_valid"] is False
    assert r["first_projection"]["identity"]["physics_step"]==176
    assert r["first_projection"]["safe_projection"]["delta_mm"][0][2]==pytest.approx(.000809032891944)
    assert r["failure_reason"]=="closure_geometry_source_invalid"
    assert r["original_guard_failure"] is None
    assert r["lifecycle"]["app_close_returned"] and not r["lifecycle"]["natural_exit_zero"]
    assert r["completed_full_budget"] is False
    assert not r["teacher_admission"] and r["new_physics_steps"]==0

@pytest.mark.parametrize("corruption",["partial_line","duplicate","missing_end","timestamp","episode","after_terminal"])
def test_prefix_integrity_faults_rejected(corruption):
    m=audit();raw=m.load_frozen()["physical_bytes"]
    rows=m.parse_jsonl(raw)
    if corruption=="partial_line":
        with pytest.raises(ValueError):m.parse_jsonl(raw[:-1])
        return
    data=deepcopy(rows)
    loc=[i for i,r in enumerate(data) if r.get("operation")=="physical"]
    if corruption=="duplicate":data.insert(loc[-1],deepcopy(data[loc[-1]]))
    if corruption=="missing_end":data.pop()
    if corruption in ("timestamp","episode"):
        x=data[loc[1]]["payload"]["exchange"]["action_readback"]["post"]["identity"][0]
        if corruption=="timestamp":x["timestamp_s"]+=.1
        else:x["episode_uid"]="other"
    if corruption=="after_terminal":data.insert(len(data)-1,deepcopy(data[loc[-1]]))
    with pytest.raises(ValueError):m.validate_prefix(data)
