from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.hand_clearance import hand_wire_clearance_observations


def fixture():
    vertices=np.array(list(product((-.01,.01),repeat=3)))
    planes=np.c_[np.r_[np.eye(3),-np.eye(3)],-np.full(6,.01)]
    jac=np.zeros((6,6));jac[0,0]=1.;jac[4,1]=1.
    bodies=[dict(name=side+'_finger13',side=side,vertices=vertices,planes=planes,
                 position=np.array([x,0,0]),rotation=np.eye(3),jacobian=jac.copy(),motion_allowance_m=0.)
            for side,x in (('left',0.),('right',.0384))]
    nodes=np.array([[.0192,-.02,.002],[.0192,.02,.002]])
    return bodies,nodes


def test_clearance_observations_are_hand_only_and_match_rigid_pose_derivatives():
    bodies,nodes=fixture()
    result=hand_wire_clearance_observations(bodies,nodes,radius=.008)
    np.testing.assert_allclose(result['clearances'],.0012,atol=2e-8)
    jac=result['clearance_jacobians']
    assert jac.shape==(2,12)
    np.testing.assert_allclose(jac[0,:2],[-1.,-.002],atol=2e-6)
    np.testing.assert_allclose(jac[1,6:8],[1.,.002],atol=2e-6)
    np.testing.assert_array_equal(jac[0,6:],0.)
    np.testing.assert_array_equal(jac[1,:6],0.)


def test_clearance_observation_is_frame_invariant_and_debits_motion_allowance():
    bodies,nodes=fixture()
    bodies[0]['motion_allowance_m']=.00008
    before=hand_wire_clearance_observations(bodies,nodes,radius=.008)
    R=Rotation.from_rotvec([.2,-.3,.1]).as_matrix();p=np.array([.2,-.1,.3])
    transformed=deepcopy(bodies)
    for body in transformed:
        body['position']=R@body['position']+p;body['rotation']=R@body['rotation']
        body['jacobian'][:3]=R@body['jacobian'][:3]
        body['jacobian'][3:]=R@body['jacobian'][3:]
    after=hand_wire_clearance_observations(transformed,nodes@R.T+p,radius=.008)
    np.testing.assert_allclose(before['clearances'],[.00112,.0012],atol=2e-8)
    np.testing.assert_allclose(after['clearances'],before['clearances'],atol=2e-8)
    np.testing.assert_allclose(after['clearance_jacobians'],before['clearance_jacobians'],atol=2e-6)


def test_clearance_provider_skips_far_bodies_and_rejects_panda_columns():
    bodies,nodes=fixture()
    bodies[0]['position']+=np.array([1.,0,0])
    result=hand_wire_clearance_observations(bodies,nodes,radius=.008)
    assert result['bodies']==['right_finger13']
    bodies[1]['jacobian']=np.zeros((6,13))
    with pytest.raises(ValueError,match='Jacobian|jacobian'):
        hand_wire_clearance_observations(bodies,nodes,radius=.008)
