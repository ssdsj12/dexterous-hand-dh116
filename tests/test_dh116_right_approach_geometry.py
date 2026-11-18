from copy import deepcopy
import numpy as np
import pytest
from scripts.prepare_dh116_right_pinch_owned_contact import derive_pair,action_schedule,expected_boundary
from go2_pvcnn.control.dh116.tabletop_right_pinch_approach_geometry import validate_projection, retain_original_guards
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import SOURCE

def packet(ordinal=49):
    pair=derive_pair();expected=expected_boundary(pair,ordinal)
    offset=np.zeros((1,2,3));offset[0,1,2]=-.0005*expected["fraction"]
    roots=expected["root_poses_w"][None]
    action=action_schedule(pair)[(ordinal-1)//8].numpy()[None]
    identity=dict(physics_step=ordinal+2,policy_step=(ordinal-1)//8,substep=(ordinal-1)%8)
    projection=dict(projected_root_poses=roots,palm_offsets_m=offset,projected_angles=np.zeros((1,2,3)),
        safe_z_clipped=np.zeros((1,2),bool),projection_reasons=np.zeros((1,2),int),projection_source=SOURCE)
    exchange=dict(action_readback=dict(post=dict(identity=[identity]),request=dict(action24=action,
        requested_palm_offsets_m=offset,requested_angles_rad=np.zeros((1,2,3))),
        submission=dict(projected_root_poses=roots,projected_angles=np.zeros((1,2,3)),projection_source=SOURCE)),
        wrist_projection=projection)
    return pair,exchange

@pytest.mark.parametrize("ordinal",[1,48,49,208,232])
def test_known_nonzero_right_wrist_is_bound_to_exact_physical_step(ordinal):
    pair,x=packet(ordinal)
    r=validate_projection(pair,x)
    assert r["ordinal"]==ordinal and r["native_qualified"] is False

@pytest.mark.parametrize("damage",["left","right","rotation","clipped","clock","action","submission"])
def test_new_scope_rejects_unapproved_projection_or_action(damage):
    pair,x=packet();x=deepcopy(x)
    if damage=="left":x["wrist_projection"]["palm_offsets_m"][0,0,2]=.0001
    if damage=="right":x["wrist_projection"]["palm_offsets_m"][0,1,2]=0
    if damage=="rotation":x["wrist_projection"]["projected_angles"][0,1,0]=.01
    if damage=="clipped":x["wrist_projection"]["safe_z_clipped"][0,1]=True
    if damage=="clock":x["action_readback"]["post"]["identity"][0]["substep"]=1
    if damage=="action":x["action_readback"]["request"]["action24"][0,1]=.1
    if damage=="submission":x["action_readback"]["submission"]["projection_source"]="unknown"
    with pytest.raises(ValueError):validate_projection(pair,x)

def test_only_obsolete_zero_offset_assertion_is_removed():
    row=dict(failure_reasons=["unexpected_wrist_projection","original:right:joint_limit","right:table:source:finger22_link"],
             passed=False,terminal_reason="closure_geometry_guard")
    original=deepcopy(row)
    new=retain_original_guards(row)
    assert row==original
    assert new["failure_reasons"]==["original:right:joint_limit","right:table:source:finger22_link"]
    assert new["passed"] is False and new["terminal_reason"]=="closure_geometry_guard"
