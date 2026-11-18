import numpy as np
import pytest
from go2_pvcnn.control.dh116.pad_front import check_pad_front


def check(points=None,normals=None,distances=None):
    return check_pad_front(
        np.array([[0.,0.,0.]]) if points is None else points,
        np.array([[0.,0.,1.]]) if normals is None else normals,
        np.array([.0079]) if distances is None else distances,
        reference_point=[0,0,0],reference_normal=[0,0,1],
        wire_radius=.008,patch_radius=.006,normal_cosine=.8660254)


def test_front_contact_allowed_not_whole_body_exempt():
    r=check()
    assert r['passed'] and r['front_samples']==1
    assert r['max_penetration_m']==pytest.approx(.0001)
    assert not check(points=[[.007,0,0]])['passed']
    assert not check(normals=[[0,0,-1]])['passed']


def test_non_front_requires_full_one_mm_clearance():
    assert not check(normals=[[0,0,-1]],distances=[.0089])['passed']
    assert check(normals=[[0,0,-1]],distances=[.0091])['passed']


def test_depth_and_inside_fail_closed():
    assert not check(distances=[.0069])['passed']
    assert not check(distances=[-.001])['passed']


@pytest.mark.parametrize('kwargs',[{'points':[[np.nan,0,0]]},{'normals':[[0,0,2]]},
                                  {'distances':[]},{'points':[]}])
def test_invalid_evidence_rejected(kwargs):
    with pytest.raises(ValueError):check(**kwargs)
