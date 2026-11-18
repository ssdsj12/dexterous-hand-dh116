"""Eight measured steps gate the new approach; historical quiet data is not authorization."""
import importlib.util
from copy import deepcopy
import numpy as np
import pytest
from test_rod_quiet_window import rows

def module():
    name="go2_pvcnn.control.dh116.tabletop_tripod_quiet_gate"
    assert importlib.util.find_spec(name) is not None,"same-instance quiet admission missing"
    return __import__(name,fromlist=["TripodQuietGate"])

def make():
    m=module();r=rows()[:8]
    for x in r:
        x["velocities"][:]=[0,0,3e-5]
        for h in x["hands"].values():h["qd"][:]=[1e-6,1e-6]
    return m.TripodQuietGate(reset_nodes_m=r[0]["positions"]),r

def test_cannot_move_before_current_eight_samples_and_new_instance_starts_empty():
    g,r=make()
    with pytest.raises(RuntimeError,match="quiet"):g.require_ready()
    for row in r[:7]:assert g.observe_measurement(row)["status"]=="pending"
    with pytest.raises(RuntimeError):g.require_ready()
    p=g.observe_measurement(r[7]);assert p["status"]=="passed"
    g.require_ready()
    assert p["teacher_admission"] is False and p["grasp_qualified"] is False
    h,_=make()
    with pytest.raises(RuntimeError):h.require_ready()

@pytest.mark.parametrize("fault",["rod_speed","joint_angular","joint_linear","table_missing","hand_load","reset_drift"])
def test_each_physical_quiet_failure_blocks_motion_without_extra_steps(fault):
    g,r=make()
    if fault=="rod_speed":r[3]["velocities"][0,0]=.0002
    elif fault=="joint_angular":r[4]["hands"]["right"]["qd"][0]=.002
    elif fault=="joint_linear":r[5]["hands"]["left"]["qd"][1]=.0002
    elif fault=="table_missing":r[0]["table_loaded_substeps"]=[]
    elif fault=="hand_load":r[2]["hand_loaded_substeps"]=[0]
    else:r[0]["positions"][1,0]+=.00002
    for row in r:p=g.observe_measurement(row)
    assert p["status"]=="failed" and p["terminal_reason"]=="tripod_quiet_admission_failed"
    with pytest.raises(RuntimeError):g.require_ready()
    with pytest.raises(RuntimeError):g.observe_measurement(r[-1])

def test_cross_episode_and_missing_ticks_poison_gate():
    g,r=make();g.observe_measurement(r[0]);r[1]["episode_uid"]="other"
    with pytest.raises(ValueError):g.observe_measurement(r[1])
    with pytest.raises(RuntimeError):g.require_ready()

def test_first_sample_must_start_actual_measurement_step_three():
    g,r=make();r[0]["physics_step"]=4
    with pytest.raises(ValueError,match="first"):g.observe_measurement(r[0])

def test_endpoint_drift_is_relative_to_reset_not_first_observed_sample():
    g,r=make()
    for row in r:row["positions"][:,2]+=.00002
    for row in r:p=g.observe_measurement(row)
    assert p["status"]=="failed" and "rod_reset_drift" in p["failure_reasons"]
