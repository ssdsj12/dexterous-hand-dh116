import numpy as np
import pytest
from scipy.spatial.transform import Rotation

def m():
    from scripts import solve_dh116_right_pinch_relative_orientation as s
    return s

def test_fixed_relative_rotation_uses_world_rz_rx_order_and_keeps_pivot():
    s=m();pivot=np.array([.43,-.2,.305]);t=s.relative_rotation(pivot,[.6,-.4])
    expected=Rotation.from_rotvec([0,0,-.02]).as_matrix()@Rotation.from_rotvec([.03,0,0]).as_matrix()
    np.testing.assert_allclose(t[:3,:3],expected,atol=1e-15)
    np.testing.assert_allclose(t@np.r_[pivot,1],np.r_[pivot,1],atol=1e-15)
    assert np.linalg.det(t[:3,:3])==pytest.approx(1.)

@pytest.mark.parametrize("angles",[[1.001,0],[0,np.nan],[0]])
def test_relative_rotation_stays_in_new_frozen_reset_box(angles):
    with pytest.raises(ValueError):m().relative_rotation([.43,-.2,.305],angles)

def test_path_wrapper_retains_both_extra_angles_at_every_fraction():
    s=m();seen=[]
    class Parent:
        def poses(self,x,f):
            seen.append((x.copy(),f));return {"ok":np.eye(4)}
        other=31
    p=s.FrozenRelativeAngles(Parent(),np.array([.2,-.3]))
    for f in [0,.5,1]:p.poses(np.arange(7,dtype=float),f)
    assert p.other==31
    assert all(np.array_equal(x,np.r_[np.arange(7),.2,-.3]) for x,f in seen)

def test_nine_dimensional_direction_keeps_existing_bounds_and_nonlinear_gate():
    from scripts.solve_dh116_right_pinch_reset_direction import local_pass
    count=[]
    def f(x):count.append(x);return np.r_[x-.02,.5-x]
    r=local_pass(f,np.zeros(9),-np.ones(9),np.ones(9),trust_radius=.1)
    assert r["best_violation"]==0 and r["measurements"]==41
    assert max(np.max(np.abs(x)) for x in count)<=.1+1e-12
