import numpy as np
import pytest
from go2_pvcnn.control.dh116.grasp_geometry import cylinder_residual,align_grasp_to_wire


def contact():
    normals=np.array([[np.sqrt(.9),0,np.sqrt(.1)],[-np.sqrt(.9),0,np.sqrt(.1)]])
    centers=np.array([[0,-.02,0],[0,.02,0]])
    return centers-.008*normals,normals


def test_two_distinct_material_contacts_share_the_same_cylinder():
    p,n=contact()
    np.testing.assert_allclose(cylinder_residual(p,n,.008,-.8),0,atol=1e-14)
    assert np.linalg.norm(p[1]-p[0])>.016  # not required to be one cross section


@pytest.mark.parametrize('roll',[0.,.7,3.14])
def test_alignment_preserves_radius_normals_and_axial_separation(roll):
    p,n=contact();center=np.array([.43,.15,.308]);axis=np.array([0,1,0])
    T=align_grasp_to_wire(p,n,.008,center,axis,roll)
    R=T[:3,:3];shift=T[:3,3];a=p@R.T+shift;b=n@R.T;c=a+.008*b
    np.testing.assert_allclose(c.mean(0),center,atol=1e-14)
    np.testing.assert_allclose(np.cross(c[1]-c[0],axis),0,atol=1e-14)
    np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-14)
    np.testing.assert_allclose(b@axis,0,atol=1e-14)
    assert np.linalg.det(R)==pytest.approx(1.)


def test_rejects_invalid_normals_and_zero_wire_axis():
    p,n=contact()
    with pytest.raises(ValueError):cylinder_residual(p,n*2,.008,-.8)
    with pytest.raises(ValueError):align_grasp_to_wire(p,n,.008,[0,0,0],[0,0,0],0.)
