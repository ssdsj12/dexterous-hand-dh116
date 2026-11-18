"""Tight conservative convex-plane certificates for near-wire source links."""
import numpy as np
import pytest

from go2_pvcnn.control.dh116.clearance import hull_wire_lower_bound


def test_convex_envelope_refinement_removes_only_sampling_uncertainty():
    planes=np.c_[np.r_[np.eye(3),-np.eye(3)],-np.ones(6)]
    nodes=np.array([[1.03,.9882,0.],[.9882,1.03,0.]])
    coarse=hull_wire_lower_bound(planes,nodes,.008,.000496)
    fine=hull_wire_lower_bound(planes,nodes,.008,.000496,refine_below=.001)
    assert coarse<.001<fine
    assert fine<=.0011+1e-14
    assert fine>.00109999
    assert hull_wire_lower_bound(planes,[[-2.,0,0],[2.,0,0]],.008,.0005,refine_below=.001)<0


@pytest.mark.parametrize('seed',range(6))
def test_refined_bound_never_exceeds_exact_piecewise_linear_minimum(seed):
    rng=np.random.default_rng(seed)
    planes=np.c_[np.r_[np.eye(3),-np.eye(3)],-np.ones(6)]
    nodes=rng.uniform(-2.,2.,(2,3))
    intercept=planes[:,:3]@nodes[0]+planes[:,3]
    slope=planes[:,:3]@(nodes[1]-nodes[0])
    parameters=[0.,1.]
    for i in range(6):
        for j in range(i):
            if slope[i]!=slope[j]:
                t=(intercept[j]-intercept[i])/(slope[i]-slope[j])
                if 0<t<1:parameters.append(t)
    exact=min(float(np.max(intercept+slope*t)) for t in parameters)-.008
    bound=hull_wire_lower_bound(planes,nodes,.008,.01,refine_below=10.)
    assert bound<=exact+1e-12
    assert exact-bound<3e-7
