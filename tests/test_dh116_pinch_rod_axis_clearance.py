"""Analytic static-reset roll must preserve geometry and every safety gate."""
import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts import audit_dh116_pinch_rod_axis_clearance as m

def test_interval_matches_known_cosine_constraint():
    result=m.feasible_intervals([[1.,0.,.5]],[-math.pi,math.pi])
    assert np.allclose(result,[[-math.pi/3,math.pi/3]])

def test_periodic_constraint_retains_both_ends():
    result=m.feasible_intervals([[-1.,0.,.5]],[-math.pi,math.pi])
    assert np.allclose(result,[[-math.pi,-2*math.pi/3],[2*math.pi/3,math.pi]])

def test_simultaneous_opposite_constraints_fail_closed():
    assert m.feasible_intervals([[1,0,.9],[-1,0,.9]],[-.5,.5])==[]

@pytest.mark.parametrize("rhs,expected",[(0,[[-.2,.2]]),(.1,[]),(-.1,[[-.2,.2]])])
def test_zero_amplitude(rhs,expected):
    assert m.feasible_intervals([[0,0,rhs]],[-.2,.2])==expected

@pytest.mark.parametrize("rows,bounds",[
    ([[1,0,float("nan")]],[-.1,.1]),([],[-.1,.1]),([[1,0,0]],[1,-1]),
    ([[1,0,0]],[-4,4]),([[1,0]],[0,1])])
def test_bad_inputs_rejected(rows,bounds):
    with pytest.raises(ValueError):m.feasible_intervals(rows,bounds)

def test_interval_agrees_with_dense_evaluation():
    rng=np.random.default_rng(773)
    rows=np.column_stack((rng.uniform(.02,.15,30),rng.uniform(-.2,.2,30),
                          rng.uniform(-.02,.02,30)))
    intervals=m.feasible_intervals(rows,[-.7,.7])
    for x in np.linspace(-.7,.7,1001):
        in_set=any(a-1e-12<=x<=b+1e-12 for a,b in intervals)
        actual=bool(np.all(rows[:,0]*np.cos(x)+rows[:,1]*np.sin(x)>=rows[:,2]))
        assert in_set==actual

def test_rod_axis_rotation_is_not_base_rotation():
    pose=[.50,.2,.42,1,0,0,0];axis=[.43,0,.305]
    out=m.rotate_about_axis(pose,axis,.2)
    t=m.pose_matrix(out);old=m.pose_matrix(pose)
    assert np.linalg.norm(t[:3,3]-old[:3,3])>.01
    ry=Rotation.from_rotvec([0,.2,0]).as_matrix()
    assert np.allclose(t[:3,:3],ry)
    point=np.array([.43,.21,.305,1])
    assert np.allclose(np.linalg.inv(t)@point,np.linalg.inv(old)@point)
    assert np.isclose(np.linalg.norm(out[3:]),1)

def test_positive_y_rotation_has_correct_z_sign():
    out=m.rotate_about_axis([1,0,0,1,0,0,0],[0,0,0],.2)
    assert out[2]==pytest.approx(-math.sin(.2))

def test_zero_rotation_keeps_full_pose():
    pose=[.5,.1,.4,*np.roll(Rotation.from_rotvec([.2,-.1,.3]).as_quat(),1)]
    assert np.allclose(m.pose_matrix(m.rotate_about_axis(pose,[.43,0,.305],0)),m.pose_matrix(pose))

def test_selector_is_interior_and_does_not_choose_infeasible_gap():
    intervals=[[-.2,-.1],[.04,.08]]
    x=m.choose_angle(intervals)
    assert .04<x<.08
    assert m.choose_angle([[-.1,.1]])==0
    assert m.choose_angle([]) is None

def test_table_coefficients_include_unsampled_motion_bound():
    vertices=np.array([[.44,0,.31],[.42,0,.32]])
    coeff=m.table_coefficients(vertices,[.43,0,.305],.00064,161)
    for theta in (-.1,0,.1):
        pred=coeff[:,0]*np.cos(theta)+coeff[:,1]*np.sin(theta)-coeff[:,2]
        rotated=m.transform(vertices,m.axis_transform([.43,0,.305],theta))
        actual=rotated[:,2]-.3-m.CLEARANCE_M-.00064/320-m.NUMERICAL_MARGIN_M
        assert np.allclose(pred,actual)
    assert np.all(coeff[:,2]>.3+m.CLEARANCE_M-.305)

@pytest.mark.parametrize("bad",[-1,float("nan")])
def test_negative_or_unknown_path_bound_rejected(bad):
    with pytest.raises(ValueError):m.table_coefficients(np.zeros((2,3)),[0,0,0],bad,161)

def test_exact_joint_order_and_closure_caps():
    parent=dict(side="left",base_pose_wxyz=[0,0,0,1,0,0,0],
      active_joint_names=list(m.ACTIVE),active_q_rad=[.3,.4,.5,.8,.8,.8],
      active_delta_rad=[0,.1,.1,0,0,0])
    m.validate_parent(parent)
    for change in (dict(active_joint_names=list(reversed(m.ACTIVE))),
                   dict(active_delta_rad=[0,.10001,.1,0,0,0]),
                   dict(active_delta_rad=[0,.1,.1,.01,0,0]),
                   dict(side="both")):
        with pytest.raises(ValueError):m.validate_parent({**parent,**change})

def test_no_geometry_pass_can_be_promoted_to_native():
    flags=m.qualification(table=.003,other=.0001,initial=.0001,gaps=[-.00015,-.00015])
    assert flags["static_geometry_passed"] is True
    assert flags["native_ready"] is False and flags["teacher_admitted"] is False

@pytest.mark.parametrize("change",[
    dict(table=.00209),dict(other=-1e-9),dict(initial=-1e-9),
    dict(gaps=[-.000099,-.00015]),dict(gaps=[-.00031,-.00015]),
    dict(gaps=[None,-.00015])])
def test_independent_rejections_cannot_be_waived(change):
    flags=m.qualification(**{**dict(table=.003,other=.0001,initial=.0001,
                                   gaps=[-.00015,-.00015]),**change})
    assert flags["static_geometry_passed"] is False
