from copy import deepcopy
import numpy as np
import pytest
from scripts import prepare_dh116_effective_right_pinch as plan
from go2_pvcnn.control.dh116.tabletop_effective_right_pinch_geometry import validate_projection,retain_original_guards
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import SOURCE

def packet(ordinal=129):
    p=plan.derive_pair();target=plan.expected_boundary(p,ordinal)
    offset=np.zeros((1,2,3));offset[0,1]=target["right_offset_m"]
    roots=target["root_poses_w"][None];zeros=np.zeros((1,2,3))
    identity=dict(physics_step=ordinal+2,policy_step=(ordinal-1)//8,substep=(ordinal-1)%8)
    projection=dict(projected_root_poses=roots,palm_offsets_m=offset,projected_angles=zeros,
        safe_z_clipped=np.zeros((1,2),bool),projection_reasons=np.zeros((1,2),int),projection_source=SOURCE)
    x=dict(action_readback=dict(post=dict(identity=[identity]),
        request=dict(action24=plan.action_schedule(p)[(ordinal-1)//8].numpy()[None],
            requested_palm_offsets_m=offset,requested_angles_rad=zeros),
        submission=dict(projected_root_poses=roots,projected_angles=zeros,projection_source=SOURCE)),
        wrist_projection=projection)
    return p,x
@pytest.mark.parametrize("ordinal",[1,48,49,128,129,208,232])
def test_world_x_bow_and_original_z_path_bind_to_every_physics_boundary(ordinal):
    p,x=packet(ordinal);r=validate_projection(p,x)
    assert r["ordinal"]==ordinal and not r["native_qualified"]
@pytest.mark.parametrize("damage",["left","right","angles","clip","action","clock","source"])
def test_foreign_control_or_identity_never_bypasses_original_guard(damage):
    p,x=packet();x=deepcopy(x)
    if damage=="left":x["wrist_projection"]["palm_offsets_m"][0,0,0]=.0001
    if damage=="right":x["wrist_projection"]["palm_offsets_m"][0,1,0]=0
    if damage=="angles":x["wrist_projection"]["projected_angles"][0,1,0]=.01
    if damage=="clip":x["wrist_projection"]["safe_z_clipped"][0,1]=True
    if damage=="action":x["action_readback"]["request"]["action24"][0,6]=0
    if damage=="clock":x["action_readback"]["post"]["identity"][0]["substep"]=7
    if damage=="source":x["action_readback"]["submission"]["projection_source"]="unknown"
    with pytest.raises(ValueError):validate_projection(p,x)
def test_only_old_zero_wrist_assertion_replaced_after_exact_target_check():
    row=dict(failure_reasons=["unexpected_wrist_projection","right:table:source:finger13_link","original:right:joint_limit"],
        passed=False,terminal_reason="closure_geometry_guard")
    out=retain_original_guards(row)
    assert out["failure_reasons"]==row["failure_reasons"][1:]
    assert out["terminal_reason"]=="closure_geometry_guard" and not out["passed"]
