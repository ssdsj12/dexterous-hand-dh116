import numpy as np
import pytest
from itertools import product
from go2_pvcnn.control.dh116.clearance import hull_wire_lower_bound, separation_lower_bound


BOX=np.array(list(product((-1.,1.),repeat=3)))
PLANES=np.c_[np.r_[np.eye(3),-np.eye(3)],-np.ones(6)]


def test_wire_certificate_includes_radius_and_between_sample_bound():
    p=np.array([[1.05,-.5,0],[1.05,.5,0]])
    r=hull_wire_lower_bound(PLANES,p,.01,.01)
    assert r==pytest.approx(.035)
    assert r<=.04  # exact cylinder/box gap; certificate must not overstate


def test_endpoints_outside_dont_miss_crossing_segment():
    assert hull_wire_lower_bound(PLANES,[[-2,0,0],[2,0,0]],.01,.01)<0


def test_coarse_sampling_cannot_inflate_clearance():
    p=[[1.01,-2,0],[1.01,2,0]]
    assert hull_wire_lower_bound(PLANES,p,.01,1.)<=0


def test_projection_certificate_uses_both_separation_directions():
    for direction in (-1,1):
        other=BOX+[direction*3,0,0]
        assert separation_lower_bound(BOX,other,np.eye(3))==pytest.approx(1.)
    assert separation_lower_bound(BOX,BOX+.5,np.eye(3))<0


@pytest.mark.parametrize('bad',['normal','nan','zero_radius','zero_spacing','empty'])
def test_invalid_contract_rejected(bad):
    planes=PLANES.copy();nodes=np.array([[2.,0,0],[2,1,0]]);r=.01;h=.01
    if bad=='normal':planes[0,:3]*=2
    if bad=='nan':nodes[0,0]=np.nan
    if bad=='zero_radius':r=0
    if bad=='zero_spacing':h=0
    if bad=='empty':planes=np.empty((0,4))
    with pytest.raises(ValueError):hull_wire_lower_bound(planes,nodes,r,h)
def test_separation_witness_is_recomputed_for_changed_geometry():
    from go2_pvcnn.control.dh116.clearance import separation_lower_bound,axis_lower_bound
    import numpy as np
    a=np.array([[0,0,0],[1,1,1]])
    b=a+[3,0,0]
    bound,axis=separation_lower_bound(a,b,np.eye(3),return_axis=True)
    assert bound==2.
    assert axis_lower_bound(a,b,axis)==bound
    assert axis_lower_bound(a,b-[2.5,0,0],axis)<0
