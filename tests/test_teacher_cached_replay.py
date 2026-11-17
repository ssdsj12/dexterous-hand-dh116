import pytest
from go2_pvcnn.control.dh116.tabletop_teacher_cached_replay import audited_link_poses

def data():
    records=[]
    for env in (0,1,2):
        for side in ("left","right"):
            for name in ("base_link","finger12_link"):
                records.append(dict(env_id=env,hand=side,link=name,
                    sdk_position_m=[env,0,0],sdk_quaternion_wxyz=[1,0,0,0]))
    return dict(display_audit=dict(records=records))

def test_identity_bound_poses_independent_of_tensor_index():
    rows=data();bindings,poses=audited_link_poses(rows)
    assert len(bindings)==6 and poses[(2,"right")].shape==(2,7)
    assert bindings[-1]["body_names"]==["base_link","finger12_link"]

def test_link_identity_change_rejected():
    rows=data();bindings,_=audited_link_poses(rows)
    rows["display_audit"]["records"][-1]["link"]="invented"
    with pytest.raises(ValueError,match="identity"):audited_link_poses(rows,bindings)

def test_duplicate_and_invalid_pose_rejected():
    rows=data();rows["display_audit"]["records"].append(rows["display_audit"]["records"][0])
    with pytest.raises(ValueError,match="duplicate"):audited_link_poses(rows)
    rows=data();rows["display_audit"]["records"][0]["sdk_quaternion_wxyz"]=[0,0,0,0]
    with pytest.raises(ValueError,match="pose"):audited_link_poses(rows)
