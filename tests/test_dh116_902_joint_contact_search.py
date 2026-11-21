import numpy as np
import pytest
from scripts.search_dh116_902_joint_contact import decode_candidate, candidate_passed

def original():
    return dict(pose_wxyz=[.48,-.12,.45,1.,0.,0.,0.],active_q_rad=[.34,.008,.67,.79,.25,.25])

def test_bounded_pose_search_preserves_side_and_unused_fingers():
    for x in [np.zeros(9),np.array([1.,-1.,1.,1.,1.,-1.,1.,-1.,1.])]:
        pose,q=decode_candidate(x,original())
        assert np.linalg.norm(pose[:3]-original()["pose_wxyz"][:3])<=.0005+1e-12
        from scipy.spatial.transform import Rotation
        angle=Rotation.from_quat(pose[[4,5,6,3]]).magnitude()
        assert angle<=.015+1e-12 and pose[1]<0
        assert np.linalg.norm(pose[3:])==pytest.approx(1)
        np.testing.assert_array_equal(q[[1,4,5]],np.array(original()["active_q_rad"])[[1,4,5]])
        assert np.max(abs(q[[0,2,3]]-np.array(original()["active_q_rad"])[[0,2,3]]))<=.02+1e-12

def row():
    return dict(shape_count=22,noncontact_rod_clearance_m=.001,support_clearance_m=.002,
        table_clearance_m=.02,contact_rod_clearance_m=-.0001,all_envelopes_certified=True,
        pad_gaps_m=[-.00002]*3,node_ids=[2,3,4],opposition=[-.9,-.8])

def test_negative_sdf_does_not_override_source_geometry_or_uncontacted_finger():
    assert candidate_passed(row())
    bad=row();bad["contact_rod_clearance_m"]=-.00036
    assert not candidate_passed(bad)
    bad=row();bad["pad_gaps_m"][1]=.00002
    assert not candidate_passed(bad)

def test_nonfinite_or_unbounded_search_coordinate_rejected():
    with pytest.raises(ValueError):decode_candidate(np.ones(9)*2,original())
    with pytest.raises(ValueError):decode_candidate(np.full(9,np.nan),original())
