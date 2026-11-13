"""Orientation-aware reset remains a CPU proposal with unchanged physical rules."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts import solve_dh116_pinch_orientation_reset as m

@pytest.fixture
def parent():
    q=Rotation.from_euler("xyz",[.4,-.3,.2]).as_quat()[[3,0,1,2]].tolist()
    return dict(side="left",base_pose_wxyz=[.5,-.2,.4,*q],
        active_q_rad=[.3,.4,.6,1.,1.,1.],active_delta_rad=[0,.025,.025,0,0,0])

def test_zero_tilt_preserves_exact_seven_coordinate_pose(parent):
    x=np.array([.1,.2,.3,.1,.2,.1,-.2,0.,0.])
    c=m.decode(parent,x);old=m.coupled.decode(parent,x[:7])
    for key in ("base_pose_wxyz","active_q_rad","active_delta_rad"):
        assert c[key]==old[key]
    assert not c["native_qualification"]

def test_tilt_is_world_precomposition_not_local_euler_or_rebased_position(parent):
    before=deepcopy(parent);x=np.array([.1,.2,.3,.1,.2,.1,-.2,.4,-.5])
    c=m.decode(parent,x);old=m.coupled.decode(parent,x[:7])
    r0=Rotation.from_quat(np.array(old["base_pose_wxyz"])[[4,5,6,3]])
    r1=Rotation.from_quat(np.array(c["base_pose_wxyz"])[[4,5,6,3]])
    np.testing.assert_allclose((r1*r0.inv()).as_rotvec(),[.012,-.015,0],atol=1e-14)
    assert c["base_pose_wxyz"][:3]==old["base_pose_wxyz"][:3] and parent==before
    assert c["active_q_rad"]==old["active_q_rad"] and c["active_delta_rad"]==old["active_delta_rad"]

@pytest.mark.parametrize("index,value",[(7,1.001),(8,-1.001),(8,np.nan)])
def test_unbounded_tilt_rejected(parent,index,value):
    x=np.zeros(9);x[index]=value
    with pytest.raises(ValueError):m.decode(parent,x)

def test_old_action_bounds_not_expanded_by_orientation(parent):
    low,high=m.bounds(parent);a,b=m.coupled.bounds(parent)
    np.testing.assert_array_equal(low[:7],a);np.testing.assert_array_equal(high[:7],b)
    x=np.zeros(9);x[3]=2.
    with pytest.raises(ValueError):m.decode(parent,x)

def test_historical_frozen_commit_bytes_are_verified_not_current_file(tmp_path,monkeypatch):
    monkeypatch.setattr(m.subprocess,"check_output",lambda *a,**kw:b"old immutable bytes")
    import hashlib
    assert m.verify_historical_source(tmp_path,"a"*40,"module.py",hashlib.sha256(b"old immutable bytes").hexdigest())
    with pytest.raises(ValueError):m.verify_historical_source(tmp_path,"a"*40,"module.py","0"*64)
