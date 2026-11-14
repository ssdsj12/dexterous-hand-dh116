"""Moving Tripod scope validates every requested/projected/submitted wrist target."""
import importlib.util
from copy import deepcopy
import numpy as np
import pytest
from scripts import prepare_dh116_tripod_owned_approach as plan
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import SOURCE

def module():
    name="go2_pvcnn.control.dh116.tabletop_tripod_owned_geometry"
    assert importlib.util.find_spec(name) is not None,"Tripod measured geometry scope missing"
    return __import__(name,fromlist=["validate_projection"])

def packet(ordinal):
    p=plan.load_pair();expected=plan.expected_boundary(p,ordinal)
    x=dict(action_readback=dict(post=dict(identity=[dict(physics_step=ordinal+2,
           policy_step=(ordinal-1)//8,substep=(ordinal-1)%8)]),
           request=dict(action24=plan.action_schedule(p)[(ordinal-1)//8].numpy()[None],
                        requested_palm_offsets_m=expected["palm_offsets_m"][None],
                        requested_angles_rad=expected["intrinsic_angles_rad"][None]),
           submission=dict(projected_root_poses=expected["root_poses_w"][None],
                           projected_angles=expected["intrinsic_angles_rad"][None],projection_source=SOURCE)),
           wrist_projection=dict(projected_root_poses=expected["root_poses_w"][None],
               projected_angles=expected["intrinsic_angles_rad"][None],palm_offsets_m=expected["palm_offsets_m"][None],
               projection_source=SOURCE,safe_z_clipped=np.zeros((1,2),bool),
               projection_reasons=np.zeros((1,2),int)))
    return p,x

@pytest.mark.parametrize("ordinal",[1,8,9,10,107,208,209,232])
def test_original_requested_targets_match_all_three_frozen_phases(ordinal):
    p,x=packet(ordinal)
    result=module().validate_projection(p,x)
    assert result["ordinal"]==ordinal
    assert result["native_qualified"] is False

@pytest.mark.parametrize("mutation",["action","clock","world","angles","safe_z","source","right"])
def test_no_blanket_wrist_projection_exception(mutation):
    p,x=packet(18)
    if mutation=="action":x["action_readback"]["request"]["action24"][0,7]=.1
    elif mutation=="clock":x["action_readback"]["post"]["identity"][0]["substep"]=7
    elif mutation=="world":x["action_readback"]["submission"]["projected_root_poses"]=x["action_readback"]["submission"]["projected_root_poses"].copy();x["action_readback"]["submission"]["projected_root_poses"][0,0,2]+=.001
    elif mutation=="angles":x["action_readback"]["request"]["requested_angles_rad"]=np.zeros((1,2,3))
    elif mutation=="safe_z":x["wrist_projection"]["safe_z_clipped"][0,0]=True
    elif mutation=="source":x["wrist_projection"]["projection_source"]="unverified"
    else:x["action_readback"]["request"]["requested_palm_offsets_m"][0,1,2]+=.001
    with pytest.raises(ValueError):module().validate_projection(p,x)

def test_reclassification_is_only_the_new_left_middle_pad_after_validated_projection():
    m=module()
    packet=dict(failure_reasons=["unexpected_wrist_projection",
        "left:non_target_not_proven_clear:source:finger33_link",
        "left:non_target_not_proven_clear:cooked:finger33_link",
        "right:non_target_not_proven_clear:source:finger33_link",
        "left:table:source:finger33_link","original:left:mimic_error"],
        sides=dict(left=dict(non_target_surface_gap_lower_m={"source:finger33_link":-.1,"cooked:finger33_link":-.1,"source:finger13_link":.001})))
    r=m.reclassify_left_middle(packet)
    assert r["failure_reasons"]==["right:non_target_not_proven_clear:source:finger33_link",
        "left:table:source:finger33_link","original:left:mimic_error"]
    assert r["sides"]["left"]["minimum_non_target_gap_lower_m"]==.001
    assert len(packet["failure_reasons"])==6
