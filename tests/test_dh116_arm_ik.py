import numpy as np
import pytest
from go2_pvcnn.control.dh116.arm_ik import solve_arm_pose


class CartesianModel:
    bodies={'base_link':'palm'}
    def fk(self,q):
        from scipy.spatial.transform import Rotation
        t=np.eye(4);t[:3,3]=q[:3];t[:3,:3]=Rotation.from_rotvec(q[3:6]).as_matrix()
        return {'palm':t},{},{}


def test_solves_pose_and_preserves_hand_without_claiming_path_safe():
    m=CartesianModel();hand=np.arange(6)*.01
    target=m.fk(np.r_[[.2,.1,.4,.1,-.2,.3,0.],hand])[0]['palm']
    r=solve_arm_pose(m,target,hand,np.zeros(7),-np.ones(7),np.ones(7))
    assert r['converged'] and not r['collision_qualified']
    assert r['position_error_m']<1e-6 and r['rotation_error_rad']<1e-6
    np.testing.assert_array_equal(r['q'][7:],hand)


def test_unreachable_pose_is_not_returned_as_success():
    t=np.eye(4);t[0,3]=10
    r=solve_arm_pose(CartesianModel(),t,np.zeros(6),np.zeros(7),-np.ones(7),np.ones(7))
    assert not r['converged']


@pytest.mark.parametrize('bad',['reflection','nan','seed','bounds'])
def test_invalid_pose_or_joint_contract_rejected(bad):
    t=np.eye(4);seed=np.zeros(7);lower=-np.ones(7);upper=np.ones(7)
    if bad=='reflection':t[0,0]=-1
    if bad=='nan':t[0,3]=np.nan
    if bad=='seed':seed[0]=2
    if bad=='bounds':lower[0]=2
    with pytest.raises(ValueError):solve_arm_pose(CartesianModel(),t,np.zeros(6),seed,lower,upper)
