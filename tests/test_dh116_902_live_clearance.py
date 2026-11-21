import itertools
import numpy as np
import pytest
from scripts.audit_dh116_902_live_clearance import temporal_shape_bound

def box(lo,hi):
    return np.array(list(itertools.product(*zip(lo,hi))))

def test_bound_accounts_for_all_moving_hand_frames_and_rod_radius():
    first=box([.02,-.005,-.005],[.03,.005,.005])
    shapes=np.stack([first,first+[-.003,0,0]])
    nodes=np.zeros((2,41,3));nodes[:,:,2]=np.linspace(-.1,.1,41)
    supports=[(np.array([.1,.1,.1]),np.array([.11,.11,.11]))]
    row=temporal_shape_bound(shapes,nodes,supports,table_top=-.2)
    assert row["max_vertex_displacement_m"]==pytest.approx(.003)
    assert row["rod_gap_lower_m"]<=.012+1e-12
    assert row["rod_gap_lower_m"]>.011
    assert row["table_gap_m"]==pytest.approx(.195)

def test_one_late_collision_cannot_hide_behind_first_frame_clearance():
    first=box([.02,-.005,-.005],[.03,.005,.005])
    nodes=np.zeros((2,41,3));nodes[:,:,2]=np.linspace(-.1,.1,41)
    row=temporal_shape_bound(np.stack([first,first+[-.025,0,0]]),nodes,
        [(np.ones(3),np.ones(3)*2)],table_top=-1)
    assert row["rod_gap_lower_m"]<0

def test_partial_or_nonfinite_frame_data_rejected():
    first=box([.02,-.005,-.005],[.03,.005,.005])
    nodes=np.zeros((2,41,3));nodes[1,0,0]=np.nan
    with pytest.raises(ValueError):
        temporal_shape_bound(np.stack([first,first]),nodes,
            [(np.ones(3),np.ones(3)*2)],table_top=-1)


def test_sdk_xyzw_pose_rotates_about_z_with_scalar_last():
    from scripts.audit_dh116_902_live_clearance import sdk_matrix
    s=2**-.5
    t=sdk_matrix(([1.,2.,3.],[0.,0.,s,s]))
    np.testing.assert_allclose(t@np.array([1.,0.,0.,1.]),[1.,3.,3.,1.],atol=1e-12)
