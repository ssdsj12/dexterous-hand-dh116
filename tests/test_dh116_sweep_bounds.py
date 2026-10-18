import numpy as np
import pytest
from go2_pvcnn.control.dh116.sweep_bounds import point_rotation_radii, relative_motion_bound


def trans(x):
    t=np.eye(4);t[0,3]=x;return t


def test_radius_bound_includes_intermediate_joint_offsets_and_body_extent():
    e=[('j1','root','a',trans(0),trans(1),[0,0,1]),
       ('fixed','a','b',trans(.5),trans(.5),None),
       ('j2','b','tip',trans(0),trans(1),[0,0,1])]
    np.testing.assert_allclose(point_rotation_radii(e,'tip',[[1,0,0]],['j1','j2']),[4,2])
    np.testing.assert_allclose(point_rotation_radii(e,'root',[[1,0,0]],['j1','j2']),0)


def test_relative_distance_cancels_common_rigid_ancestor_motion():
    a=np.array([4,0.]);b=np.array([5,2.]);delta=np.array([.3,.1])
    assert relative_motion_bound(a,b,delta,same_tree=True)==pytest.approx(.2)
    assert relative_motion_bound(a,b,delta,same_tree=False)==pytest.approx(2.9)


def test_missing_body_or_negative_delta_is_rejected():
    e=[('j1','root','a',trans(0),trans(1),[0,0,1])]
    with pytest.raises(ValueError):point_rotation_radii(e,'missing',[[1,0,0]],['j1'])
    with pytest.raises(ValueError):relative_motion_bound([1],[1],[-.1],same_tree=True)
