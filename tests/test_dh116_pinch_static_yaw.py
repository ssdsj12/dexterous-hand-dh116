"""Bounded static world-Z reset design; no native or contact qualification."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts import audit_dh116_pinch_static_yaw as m

@pytest.fixture
def parent():
    return dict(side="left",base_pose_wxyz=[.1,.2,.3,*Rotation.from_euler("xyz",[.4,-.3,.2]).as_quat()[[3,0,1,2]]],
        active_q_rad=[.3,.3,.3,.3,.3,.3],active_delta_rad=[0,.08,.09,0,0,0])

def test_yaw_uses_world_precomposition_and_keeps_joint_motion(parent):
    old=deepcopy(parent);v=m.rotate_candidate(parent,.03)
    r0=Rotation.from_quat(np.asarray(parent["base_pose_wxyz"])[[4,5,6,3]])
    r1=Rotation.from_quat(np.asarray(v["base_pose_wxyz"])[[4,5,6,3]])
    assert np.allclose((r1*r0.inv()).as_rotvec(),[0,0,.03],atol=1e-14)
    assert v["base_pose_wxyz"][:3]==parent["base_pose_wxyz"][:3]
    assert v["active_q_rad"]==parent["active_q_rad"] and v["active_delta_rad"]==parent["active_delta_rad"]
    assert parent==old and v["native_qualification"] is v["teacher_admission"] is False

@pytest.mark.parametrize("yaw",[-.03,-.0075,0,.0075,.03])
def test_world_z_rotation_preserves_every_point_height(parent,yaw):
    v=m.rotate_candidate(parent,yaw)
    a=Rotation.from_quat(np.asarray(parent["base_pose_wxyz"])[[4,5,6,3]])
    b=Rotation.from_quat(np.asarray(v["base_pose_wxyz"])[[4,5,6,3]])
    points=np.random.default_rng(31).normal(size=(100,3))
    assert np.allclose(a.apply(points)[:,2],b.apply(points)[:,2],atol=1e-14)

@pytest.mark.parametrize("yaw",[.031,-.031,np.nan,np.inf,True])
def test_no_unbounded_or_invalid_orientation(parent,yaw):
    with pytest.raises(ValueError):m.rotate_candidate(parent,yaw)

def test_hard_constraints_take_priority_over_promising_contact():
    assert m.rank([1,1],[-.1,-.1]) < m.rank([-.001,1],[-.1,-.1])
    assert m.rank([1,1],[-.1,-.1]) < m.rank([1,1],[.4,-.2])
    assert m.rank([1,1],[-.31,-.2]) > m.rank([1,1],[-.2,-.2])

def test_only_fixed_grid_and_102_complete_states():
    assert m.MAX_STATES==2*(2*len(m.ANGLES)+33)==102
    assert len(m.ANGLES)==9 and np.isclose(max(m.ANGLES),.03) and np.isclose(min(m.ANGLES),-.03)

@pytest.mark.parametrize("key,value",[("base_pose_wxyz",[0,0,0,0,0,0,0]),("side","unknown"),
 ("active_delta_rad",[.1,.08,.09,0,0,0]),("active_delta_rad",[0,.11,.09,0,0,0])])
def test_invalid_original_contract_rejected(parent,key,value):
    parent[key]=value
    with pytest.raises(ValueError):m.rotate_candidate(parent,0)

def test_complete_state_budget_cannot_expand_or_hide_errors():
    records=[];budget=m.trust.StateBudget(records,maximum=m.MAX_STATES)
    for i in range(102):assert budget.call(lambda x:x,i)==i
    with pytest.raises(m.trust.BudgetLimit):budget.call(lambda x:x,103)
    assert budget.count==len(records)==102
    records=[];budget=m.trust.StateBudget(records,maximum=m.MAX_STATES)
    def fail(x):raise ValueError("preserve this diagnostic")
    with pytest.raises(ValueError):budget.call(fail,0)
    assert budget.count==1 and records[0]["status"]=="error"
