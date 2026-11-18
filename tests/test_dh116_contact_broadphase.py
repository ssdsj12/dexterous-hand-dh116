"""Conservative distance-to-box screening, never a replacement contact surface."""
import numpy as np
import pytest
from elastica.rigidbody import mesh_rigid_body as module


def test_corner_false_candidates_are_rejected_but_faces_and_inside_retained():
    # Each axis is within radius at the corner, but Euclidean separation is not.
    points=np.array([[1.08,1.08,0.], [1.1,0.,0.], [0.,0.,0.], [1.,1.,1.]])
    actual=module._contact_candidate_mask(points,np.full(4,.1),-np.ones(3),np.ones(3))
    np.testing.assert_array_equal(actual,[False,True,True,True])


@pytest.mark.parametrize('scale',[1e-3,1.,1e3])
def test_compiled_screen_matches_conservative_reference(scale):
    rng=np.random.default_rng(204)
    low=np.array([-1.,-2.,-3.])*scale
    high=-low
    points=rng.uniform(-4,4,size=(2000,3))*scale
    radii=rng.uniform(.001,.5,size=2000)*scale
    rounding_scale=np.maximum(1.,np.max(np.abs(points),axis=1))
    rounding_scale=np.maximum(rounding_scale,max(np.abs(low).max(),np.abs(high).max()))
    padded=radii+64*np.finfo(np.float32).eps*rounding_scale
    delta=np.maximum(np.maximum(low-points,points-high),0.)
    expected=np.sum(delta*delta,axis=1)<=padded*padded
    actual=module._contact_candidate_mask(points,radii,low,high)
    np.testing.assert_array_equal(actual,expected)
    # The test exercises compiled nopython code, not a Python-mode fallback.
    assert module._contact_candidate_mask.nopython_signatures
