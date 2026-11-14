from copy import deepcopy
import importlib
from pathlib import Path
import numpy as np
import pytest
from test_dh116_right_approach_geometry import packet as right_packet

def module():
    path=Path(__file__).parents[1]/"src/go2_pvcnn/control/dh116/tabletop_bilateral_owned_geometry.py"
    assert path.exists(),"bilateral geometry validator is not implemented"
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_bilateral_owned_geometry")

@pytest.mark.parametrize("ordinal",[1,48,49,208,232])
def test_full_combined_schedule_has_own_scope(ordinal):
    g=module();pair=g.plan.derive_pair();_,x=right_packet(ordinal)
    x["action_readback"]["request"]["action24"]=g.plan.action_schedule(pair)[(ordinal-1)//8].numpy()[None]
    r=g.validate_projection(pair,x)
    assert r["ordinal"]==ordinal and not r["native_qualified"]

@pytest.mark.parametrize("damage",["left","right","rotation","clock","submission"])
def test_unapproved_action_or_frame_fails(damage):
    g=module();pair=g.plan.derive_pair();_,x=right_packet(49);x=deepcopy(x)
    x["action_readback"]["request"]["action24"]=g.plan.action_schedule(pair)[6].numpy()[None]
    if damage=="left":x["action_readback"]["request"]["action24"][0,0]+=.01
    if damage=="right":x["action_readback"]["request"]["action24"][0,7]+=.01
    if damage=="rotation":x["wrist_projection"]["projected_angles"][0,1,0]=.01
    if damage=="clock":x["action_readback"]["post"]["identity"][0]["substep"]=1
    if damage=="submission":x["action_readback"]["submission"]["projected_root_poses"]=np.zeros((1,2,7))
    with pytest.raises(ValueError):g.validate_projection(pair,x)

def test_whole_hand_separation_is_conservative_and_never_inferred_from_centers():
    g=module()
    a=np.array([[0,0,0],[1,1,1]],float)
    assert g.separation_lower(a,a+[0,3,0])==pytest.approx(2.)
    assert g.separation_lower(a,a+[.5,0,0])<0
    with pytest.raises(ValueError):g.separation_lower(a,np.array([[float("nan"),1,1]]))

def test_old_collision_and_joint_guards_remain():
    g=module()
    original=dict(failure_reasons=["unexpected_wrist_projection","original:right:joint_limit","left:table:source:finger14_link"],passed=False,terminal_reason="closure_geometry_guard")
    r=g.retain_original_guards(original)
    assert r["failure_reasons"]==original["failure_reasons"][1:]
    assert not r["passed"] and r["terminal_reason"]=="closure_geometry_guard"
