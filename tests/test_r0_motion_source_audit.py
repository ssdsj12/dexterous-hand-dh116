from copy import deepcopy
import numpy as np
import pytest
from scripts.replay_dh116_r0_hold_domain import motion_measurements

def sample():
    v=np.zeros((1,41,3));v[0,4]=[.003,.004,0]
    return dict(rod=dict(velocities=v,node_positions=np.zeros_like(v)),
        action_readback=dict(post=dict(identity=[dict(env_id=0,episode_uid="fixture",physics_step=3)])),
        guard_failure=None)

def test_actual_three_axis_node_speed_and_identity_are_retained():
    x=sample();r=motion_measurements([x])
    peak=r["maximum_speed_sample"]
    assert peak["node_index"]==4 and peak["max_node_speed_m_s"]==pytest.approx(.005)
    assert peak["identity"]==x["action_readback"]["post"]["identity"][0]
    assert peak["stable"] is None and not peak["stability_qualified"]
    assert r["adopted_stability_threshold"] is None and not r["teacher_admission"]

@pytest.mark.parametrize("bad", ["shape","nan","position"])
def test_invalid_raw_solver_measurement_is_not_clamped_or_filled(bad):
    x=sample()
    if bad=="shape":x["rod"]["velocities"]=np.zeros((41,3))
    elif bad=="nan":x["rod"]["velocities"][0,0,0]=float("nan")
    else:x["rod"]["node_positions"][0,0,0]=float("inf")
    with pytest.raises(ValueError):motion_measurements([x])

def test_rigid_threshold_comparison_is_not_a_flexible_stability_contract():
    x=sample();x["rod"]["velocities"][0,4]=[.031,0,0]
    r=motion_measurements([x])
    assert r["rigid_legacy_0_03_m_s_comparison_only"]==1
    assert r["stable_measurement"] is None
