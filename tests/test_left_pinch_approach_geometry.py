from copy import deepcopy
import importlib
import numpy as np
import pytest
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from scripts import prepare_dh116_left_pinch_approach as plan

def packet(ordinal=149):
    fixture=importlib.import_module("test_effective_right_pinch_geometry")
    with bindings((fixture,"plan",plan)):p,x=fixture.packet(ordinal)
    expected=plan.expected_boundary(p,ordinal)
    x["wrist_projection"]["palm_offsets_m"][0,0]=expected["left_offset_m"]
    return p,x

@pytest.mark.parametrize("ordinal",[1,48,49,149,208,232])
def test_both_target_coordinates_required(ordinal):
    m=importlib.import_module("go2_pvcnn.control.dh116.tabletop_left_pinch_approach_geometry")
    p,x=packet(ordinal);r=m.validate_projection(p,x)
    assert r["ordinal"]==ordinal
    np.testing.assert_allclose(r["left_offset_m"],plan.expected_boundary(p,ordinal)["left_offset_m"])
    assert r["native_qualified"] is False

@pytest.mark.parametrize("bad",["left_offset","left_root","right_offset","rotation","safe_z","reason","action","clock"])
def test_left_and_right_projection_damage_rejected(bad):
    m=importlib.import_module("go2_pvcnn.control.dh116.tabletop_left_pinch_approach_geometry")
    p,x=packet();x=deepcopy(x)
    if bad=="left_offset":x["wrist_projection"]["palm_offsets_m"][0,0,2]=0.
    if bad=="left_root":x["action_readback"]["submission"]["projected_root_poses"][0,0,2]+=.001
    if bad=="right_offset":x["wrist_projection"]["palm_offsets_m"][0,1,2]=0.
    if bad=="rotation":x["wrist_projection"]["projected_angles"][0,0,0]=.1
    if bad=="safe_z":x["wrist_projection"]["safe_z_clipped"][0,0]=True
    if bad=="reason":x["wrist_projection"]["projection_reasons"][0,0]=1
    if bad=="action":x["action_readback"]["request"]["action24"][0,14]=0.
    if bad=="clock":x["action_readback"]["post"]["identity"][0]["substep"]+=1
    with pytest.raises(ValueError):m.validate_projection(p,x)
