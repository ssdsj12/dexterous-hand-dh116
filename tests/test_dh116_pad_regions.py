import numpy as np
import pytest
from go2_pvcnn.control.dh116.pad_regions import classify_front_cap,sample_wire


def mesh():
    # Disconnected same-facing surface must not inherit a contact exemption.
    vertices=np.array([[-.001,-.001,0],[.001,-.001,0],[0,.001,0],
                       [.003,-.001,0],[.005,-.001,0],[.004,.001,0],
                       [-.001,-.001,-.002],[0,.001,-.002],[.001,-.001,-.002]])
    return vertices,np.array([[0,1,2],[3,4,5],[6,7,8]])


def test_only_connected_local_forward_cap_is_selected():
    v,f=mesh();mask=classify_front_cap(v,f,[0,0,0],[0,0,1])
    assert mask.tolist()==[True,False,False]


def test_reversed_reference_or_large_face_cannot_exempt_whole_body():
    v,f=mesh()
    with pytest.raises(ValueError):classify_front_cap(v,f,[0,0,1],[0,0,1])
    v[0]=[-.1,0,0]
    with pytest.raises(ValueError):classify_front_cap(v,f,[0,0,0],[0,0,1])


def test_invalid_region_geometry_rejected():
    v,f=mesh()
    with pytest.raises(ValueError):classify_front_cap(v,f,[0,0,0],[0,0,2])
    with pytest.raises(ValueError):classify_front_cap(v,[[0,0,1]],[0,0,0],[0,0,1])


def test_wire_sampling_has_proven_half_spacing_distance_bound():
    wire=np.array([[0,0,0],[.0011,0,0],[.0011,.0007,0]])
    points,allowance=sample_wire(wire,spacing=.0005)
    assert allowance<=.00025
    for a,b in zip(wire[:-1],wire[1:]):
        for u in np.linspace(0,1,301):
            assert np.linalg.norm(points-((1-u)*a+u*b),axis=1).min()<=allowance+1e-12
    with pytest.raises(ValueError):sample_wire([[0,0,0]])


def test_roundoff_at_integer_sample_count_is_frame_invariant():
    from scipy.spatial.transform import Rotation
    wire=np.array([[0,-.04,.008],[0,.04,.008]])
    R=Rotation.from_rotvec([.2,-.3,.4]).as_matrix()
    p,a=sample_wire(wire);q,b=sample_wire(wire@R.T+[1,2,3])
    assert len(p)==len(q)
    assert b==pytest.approx(a,abs=1e-15)
