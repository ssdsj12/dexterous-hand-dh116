import numpy as np
import pytest
from go2_pvcnn.control.dh116.symmetric_geometry import mirror_functional_pose


def test_functional_mirror_is_proper_rotation_and_involution():
    from scipy.spatial.transform import Rotation
    p=np.array([.43,.2,.32]);R=Rotation.from_rotvec([.3,-.2,.4]).as_matrix()
    q,Q=mirror_functional_pose(p,R)
    S=np.diag([1.,-1.,1.])
    np.testing.assert_allclose(q,S@p)
    np.testing.assert_allclose(Q[:,0],S@R[:,0])
    np.testing.assert_allclose(Q[:,2],S@R[:,2])
    np.testing.assert_allclose(Q.T@Q,np.eye(3),atol=1e-12)
    assert np.linalg.det(Q)==pytest.approx(1.)
    pp,RR=mirror_functional_pose(q,Q)
    np.testing.assert_allclose(pp,p);np.testing.assert_allclose(RR,R)


@pytest.mark.parametrize('p,R',[(np.zeros(2),np.eye(3)),
    ([0,np.nan,0],np.eye(3)),(np.zeros(3),np.diag([1,-1,1])),
    (np.zeros(3),np.eye(3)*2),(np.zeros(3),np.full((3,3),np.nan))])
def test_rejects_invalid_functional_pose(p,R):
    with pytest.raises(ValueError):mirror_functional_pose(p,R)
