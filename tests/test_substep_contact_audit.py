"""Audit aggregation cannot promote reconstructed contacts to source qualification."""
from copy import deepcopy
import numpy as np
import pytest
from scripts.audit_dh116_substep_contact_frames import named_poses,summarize_candidates
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import PADS,HAND_NAMES

def hands():
    return {s:dict(body_names=list(PADS),link_paths=[[
        "/World/"+s+"Hand/"+p for p in PADS]],body_poses_xyzw=np.tile(
            [0.,0.,0.,0.,0.,0.,1.],(1,len(PADS),1))) for s in ("left","right")}

def test_all_own_link_identities():
    out=named_poses(hands(),"body_poses_xyzw")
    assert set(out)==set(HAND_NAMES) and all(p.shape==(7,) for p in out.values())

@pytest.mark.parametrize("fault",["side","duplicate","shape","nan"])
def test_no_inferred_or_replaced_pose(fault):
    h=hands()
    if fault=="side":h["right"]["link_paths"]=h["left"]["link_paths"]
    elif fault=="duplicate":h["left"]["body_names"][1]=h["left"]["body_names"][0]
    elif fault=="shape":h["left"]["body_poses_xyzw"]=h["left"]["body_poses_xyzw"][0]
    else:h["left"]["body_poses_xyzw"][0,0,0]=np.nan
    with pytest.raises(ValueError):named_poses(h,"body_poses_xyzw")

def test_changed_counts_preserve_failed_candidates():
    records=[dict(mesh_name="left.finger14_link",
        endpoint_region=dict(accepted_by_original_patch=a),
        substep_region=dict(accepted_by_original_patch=b,first_rejection=None if b else "wrong_normal"))
        for a,b in ((True,False),(False,True),(False,False))]
    result=summarize_candidates(records)
    assert result["endpoint_candidates"]=={"left.finger14_link":1}
    assert result["source_substep_candidates"]=={"left.finger14_link":1}
    assert result["changed_classifications"]=={"left.finger14_link":2}
    assert result["substep_first_rejections"]=={"wrong_normal":2,"cpu_candidate":1}
    assert not any(result[k] for k in ("positive_effective_contact_qualified",
        "negative_effective_contact_qualified","reset_mesh_pose_qualified","teacher_admission"))
