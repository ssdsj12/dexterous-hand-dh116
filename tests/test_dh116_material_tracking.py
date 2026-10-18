import numpy as np
import pytest
from go2_pvcnn.control.dh116.material_tracking import material_in_palm


def test_fixed_material_nodes_follow_measured_palm_not_nearest_point():
    nodes=np.c_[np.zeros(5),np.linspace(-.2,.2,5),np.ones(5)*.308]
    poses={'left':np.eye(4),'right':np.eye(4)}
    ids=(4,3,1,0)
    a=material_in_palm(nodes,poses,ids)
    shift=np.array([.01,.02,.03]);moved={k:v.copy() for k,v in poses.items()}
    for p in moved.values():p[:3,3]=shift
    np.testing.assert_allclose(material_in_palm(nodes+shift,moved,ids),a,atol=1e-12)
    np.testing.assert_allclose(material_in_palm(nodes,moved,ids)-a,np.tile(-shift,(4,1)))


def test_rotation_is_inverted_and_bad_frames_or_ids_rejected():
    nodes=np.array([[1.,0,0],[0,1,0],[0,0,1],[1,1,1]])
    R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    poses={s:np.eye(4) for s in ('left','right')}
    for p in poses.values():p[:3,:3]=R
    np.testing.assert_allclose(material_in_palm(nodes,poses,(0,1,2,3)),nodes@R)
    with pytest.raises(ValueError):material_in_palm(nodes,poses,(0,1,2,4))
    poses['left'][0,0]=2.
    with pytest.raises(ValueError):material_in_palm(nodes,poses,(0,1,2,3))
