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


class SphereContactModel(CartesianModel):
    bodies={'base_link':'palm','thumb':'thumb','index':'index'}
    def fk(self,q):
        from scipy.spatial.transform import Rotation
        poses=super().fk(q)[0];palm=poses['palm']
        for name,x,joint in (('thumb',-.0126,7),('index',.0126,9)):
            local=np.eye(4);local[0,3]=x
            local[:3,:3]=Rotation.from_rotvec([0.,0.,q[joint]-.2]).as_matrix()
            poses[name]=palm@local
        return poses,{},{}


def sphere_contacts():
    def query(points):
        length=np.linalg.norm(points,axis=1);normal=points/length[:,None]
        return normal*.005,length-.005,normal
    angle=.05
    return [dict(body=name,reference_point=.005*np.array([sign*np.cos(angle),np.sin(angle),0.]),
                 reference_normal=np.array([sign*np.cos(angle),np.sin(angle),0.]),
                 mesh_offset=np.zeros(3),query=query)
            for name,sign in (('thumb',1.),('index',-1.))]


def test_contact_centre_alignment_preserves_measured_gaps_and_parks_other_fingers():
    from go2_pvcnn.control.dh116.arm_ik import align_contact_centres
    seed=np.r_[np.zeros(7),np.full(6,.2)]
    samples=np.c_[np.zeros(81),np.linspace(-.04,.04,81),np.zeros(81)]
    result=align_contact_centres(SphereContactModel(),seed,sphere_contacts(),samples,-np.ones(7),np.ones(7))
    assert result['converged']
    assert not result['collision_qualified'] and not result['physics_qualified']
    assert result['max_gap_change_m']<1e-5
    assert np.linalg.norm(result['palm_delta_world_m'])<=.002
    assert result['palm_rotation_delta_rad']<=np.deg2rad(3.)
    assert max(result['reference_offset_after_m'])<max(result['reference_offset_before_m'])*.7
    np.testing.assert_array_equal(result['q'][10:],seed[10:])


@pytest.mark.parametrize('kind',['panda_width','wire_nan','bad_normal'])
def test_contact_alignment_rejects_inconsistent_geometry(kind):
    from go2_pvcnn.control.dh116.arm_ik import align_contact_centres
    seed=np.r_[np.zeros(7),np.full(6,.2)];contacts=sphere_contacts()
    samples=np.c_[np.zeros(3),np.linspace(-.04,.04,3),np.zeros(3)]
    if kind=='panda_width':seed=np.zeros(26)
    if kind=='wire_nan':samples[0,0]=np.nan
    if kind=='bad_normal':contacts[0]['reference_normal']*=2
    with pytest.raises(ValueError):
        align_contact_centres(SphereContactModel(),seed,contacts,samples,-np.ones(7),np.ones(7))
