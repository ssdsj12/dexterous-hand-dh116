import importlib
import numpy as np
import pytest

def module():
    n="scripts.audit_dh116_tripod_pose_path"
    assert importlib.util.find_spec(n) is not None, "complete pose-path audit missing"
    return importlib.import_module(n)

def test_pose_interpolation_preserves_endpoints_and_proper_rotation():
    from scipy.spatial.transform import Rotation
    a=np.array([1,2,3,1,0,0,0.])
    b=np.r_[[2,1,4],np.roll(Rotation.from_rotvec([.2,.4,.1]).as_quat(),1)]
    assert module().pose_at(a,b,0)==pytest.approx(a)
    assert module().pose_at(a,b,1)==pytest.approx(b)
    p=module().pose_at(a,b,.5)
    assert p[:3]==pytest.approx([1.5,1.5,3.5])
    assert np.linalg.norm(p[3:])==pytest.approx(1)

def test_complete_point_motion_bound_includes_root_and_joint_motion():
    r=module().combine_bounds({"pad":.01},{"pad":np.array([[2.,0,0],[0,2,0]])},
                              np.zeros(3),np.array([0,0,.03]),.2)
    assert r["pad"]==pytest.approx(.03+.2*2.01+.01)

def test_translation_does_not_disappear_when_joints_are_fixed():
    r=module().combine_bounds({"a":0},{"a":np.ones((4,3))},np.zeros(3),np.array([.003,.004,0]),0)
    assert r["a"]==pytest.approx(.005)

def test_missing_shape_or_invalid_angular_bound_rejected():
    with pytest.raises(ValueError):module().combine_bounds({"a":0},{},np.zeros(3),np.zeros(3),0)
    with pytest.raises(ValueError):module().combine_bounds({"a":0},{"a":np.ones((4,3))},np.zeros(3),np.zeros(3),-.1)

def test_fraction_outside_original_path_rejected():
    a=[0,0,0,1,0,0,0]
    with pytest.raises(ValueError):module().pose_at(a,a,1.1)
