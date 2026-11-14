"""Quiet windows may describe a captured run, never grant a different grasp admission."""
import importlib.util
from copy import deepcopy
import numpy as np
import pytest

def module():
    name="scripts.audit_dh116_rod_quiet_window"
    assert importlib.util.find_spec(name) is not None,"same-clock quiet-window audit missing"
    return __import__(name,fromlist=["summarize_window"])

def rows():
    return [dict(run_uuid="r",env_id=0,episode_uid="e",reset_id=1,rod_uid="rod",
        physics_step=i+3,rod_tick=12*(i+1),timestamp_s=(i+3)/480,policy_step=i//8,substep=i%8,
        action24=np.zeros(24),positions=np.tile([.43,0,.305],(41,1)),
        velocities=np.tile([0,0,.0002],(41,1)),
        hands={s:dict(joint_names=["finger12","wrist_z"],q=np.array([.4,0.]),
                       qd=np.array([.003,.0001]),units={"finger12":"rad","wrist_z":"m"}) for s in ("left","right")},
        table_loaded_substeps=list(range(12)),hand_loaded_substeps=[],guard_failure=None)
       for i in range(16)]

def test_metrics_are_measured_per_unit_and_not_a_stable_or_native_certificate():
    r=module().summarize_window(rows(),16)
    assert r["samples"]==16 and r["max_node_speed_m_s"]==pytest.approx(.0002)
    assert r["table_loaded_solver_substeps"]==192
    assert r["hand_joint_speed"]["left"]["finger12"]["unit"]=="rad/s"
    assert r["hand_joint_speed"]["left"]["wrist_z"]["unit"]=="m/s"
    assert r["native_reset_qualified"] is r["teacher_admitted"] is False
    assert r["stable_grasp"] is None

@pytest.mark.parametrize("field,value",[("episode_uid","new"),("reset_id",2),("run_uuid","other"),
                                       ("rod_uid","newrod"),("env_id",1)])
def test_identity_changes_are_not_hidden_in_window_average(field,value):
    r=rows();r[8][field]=value
    with pytest.raises(ValueError,match="identity"):module().summarize_window(r,16)

def test_missing_tick_and_reversed_timestamp_rejected():
    for field,value in [("physics_step",99),("rod_tick",99),("timestamp_s",-.1),("substep",7)]:
        r=rows();r[7][field]=value
        if field=="substep":r[7][field]=6
        with pytest.raises(ValueError,match="clock"):module().summarize_window(r,16)

def test_nonzero_action_and_guard_stop_do_not_count_as_quiet():
    for key in ("action24","guard_failure"):
        r=rows()
        if key=="action24":r[2][key][7]=.001
        else:r[2][key]="mimic_error"
        with pytest.raises(ValueError,match="quiet"):module().summarize_window(r,16)

def test_no_missing_or_nonfinite_velocity_substitution():
    r=rows();r[5]["velocities"][3,1]=float("nan")
    with pytest.raises(ValueError,match="finite"):module().summarize_window(r,16)

def test_only_complete_60hz_windows_count():
    with pytest.raises(ValueError,match="whole"):module().summarize_window(rows(),12)
