import numpy as np
import pytest
from scripts.solve_dh116_right_pinch_joint_root_constraints import root_candidate, constraint_residual

def test_root_delta_is_proper_and_original_not_mutated():
    root=np.eye(4);root[:3,3]=[.5,-.2,.4]
    original=root.copy();q=np.array([.4,.2,.6,1.15,1.15,1.15])
    moved,changed=root_candidate(root,q,[.1,.2,.3,.4,.5,.6,.7,.8,.9])
    np.testing.assert_array_equal(root,original)
    np.testing.assert_allclose(moved[:3,3],original[:3,3]+[.004,.005,.006])
    assert np.linalg.det(moved[:3,:3])==pytest.approx(1)
    np.testing.assert_allclose(changed[:3],q[:3]+[.02,.04,.06])
    np.testing.assert_array_equal(changed[3:],q[3:])

def test_collision_and_both_contact_residuals_survive_exact_other_target():
    a=dict(table_m={"source:base_link":.0023,"source:finger13_link":-.001},
        non_target_rod_gap_m={"source:finger13_link":-.0004},
        pads={"finger14_link":{"minimum_gap_m":-.00015},"finger23_link":{"minimum_gap_m":.002}},
        normal_dot=-.8)
    r=constraint_residual(a,np.zeros(9))
    assert r[0]==pytest.approx(0) and r[1]==pytest.approx(2.15)
    assert r[3]>3 and r[4]>0
    assert np.isfinite(r).all()

def test_absent_sdf_contact_is_rejected_not_filled_with_zero():
    a=dict(table_m={"a":.01},non_target_rod_gap_m={"b":.01},pads={
        "finger14_link":{"minimum_gap_m":None},"finger23_link":{"minimum_gap_m":0}},normal_dot=-.5)
    with pytest.raises(ValueError):constraint_residual(a,np.zeros(9))
