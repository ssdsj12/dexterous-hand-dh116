import numpy as np
import pytest
from scripts.solve_dh116_right_pinch_owned_approach import opening_proposal, grid_lipschitz

def test_absolute_differences_at_zero_and_bounded_original_joint_direction():
    a=np.array([[.04,.02,0],[0,0,.03]])
    calls=[]
    def f(q):calls.append(q.copy());return np.array([-.00015,-.00015])+a@q
    r=opening_proposal(f,np.zeros(3),np.array([-.1,-.002,-.1]),np.array([.1,.1,.1]),.00025)
    np.testing.assert_allclose(r["jacobians"][0],a,atol=1e-10)
    np.testing.assert_allclose(a@r["delta_rad"],[.0004,.0004],atol=1e-7)
    assert len(calls)==13 and all(np.max(abs(v))<=.0001+1e-12 for v in calls)

def test_opening_does_not_cross_original_bound_for_derivative():
    with pytest.raises(ValueError,match="neighborhood"):
        opening_proposal(lambda q:np.zeros(2),np.zeros(3),np.zeros(3),np.ones(3),.00025)

def test_nonlocal_derivative_rejected():
    def f(q):
        return np.array([q[0] if abs(q[0])>.000075 else 2*q[0],q[1]])
    with pytest.raises(ValueError,match="nonlocal"):opening_proposal(f,np.zeros(3),np.full(3,-.1),np.full(3,.1),.00025)

def test_sdf_lipschitz_bounds_trilinear_gradient_not_assumed_unit():
    x=np.linspace(0,.02,21)
    grid=dict(values=2*x[:,None,None]+3*x[None,:,None]+4*x[None,None,:],spacing=.001)
    assert grid_lipschitz(grid)==pytest.approx(np.sqrt(29))

def test_bounded_opening_chooses_smallest_joint_displacement():
    from scripts.solve_dh116_right_pinch_owned_approach import minimum_norm_bounded
    j=np.array([[1.,2.,0.],[0.,0.,1.]])
    d=minimum_norm_bounded(j,[.3,.2],[-1.,-.01,-1.],[1.,1.,1.])
    np.testing.assert_allclose(d,[.06,.12,.2],atol=1e-10)
    saturated=minimum_norm_bounded(j,[.3,.2],[-1.,-.01,-1.],[1.,.05,1.])
    np.testing.assert_allclose(saturated,[.2,.05,.2],atol=1e-10)
    with pytest.raises(ValueError,match="bounded"):
        minimum_norm_bounded(j,[10.,.2],[-1.,-.01,-1.],[1.,.05,1.])
