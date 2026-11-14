"""Nominal substep interpolation must not masquerade as a measured clock."""
from copy import deepcopy
import pytest
from go2_pvcnn.control.dh116.tabletop_deformx_substep_pose import nominal_substep_clock

def test_preserves_actual_endpoint_time_separately_from_nominal():
    endpoint=dict(physics_step=234, timestamp_s=.4875, solver_time_s=.487501234)
    before=deepcopy(endpoint)
    out=nominal_substep_clock(endpoint,11)
    assert out["nominal_substep_time_s"]==pytest.approx(.4875)
    assert out["endpoint_solver_time_s"]==.487501234
    assert out["endpoint_timestamp_s"]==.4875
    assert out["measured_substep_time_s"] is None
    assert out["time_source"]=="nominal_480Hz_index_interpolation"
    assert endpoint==before

@pytest.mark.parametrize("step,sub",[(True,0),(0,0),(234,12),(234,False)])
def test_rejects_invalid_counters(step,sub):
    with pytest.raises(ValueError):
        nominal_substep_clock(dict(physics_step=step,timestamp_s=.2,solver_time_s=.2),sub)

@pytest.mark.parametrize("key",["timestamp_s","solver_time_s"])
def test_invalid_actual_endpoint_stays_invalid(key):
    endpoint=dict(physics_step=234,timestamp_s=.4875,solver_time_s=.487501234)
    endpoint[key]=float("nan")
    with pytest.raises(ValueError):nominal_substep_clock(endpoint,0)
