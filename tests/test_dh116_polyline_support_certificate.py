import itertools
import numpy as np
import pytest

def module():
    from go2_pvcnn.control.dh116 import tabletop_polyline_support_certificate as m
    return m

def cube(center=(0,0,0),radius=.001):
    return np.array(list(itertools.product((-radius,radius),repeat=3)))+np.array(center)

def test_y_separation_retained_in_full_three_dimensional_certificate():
    m=module();v=cube();nodes=np.array([[0,.02,0],[0,.03,0]])
    p=m.certify(v,nodes)
    assert p["old_global_xz_lower_m"]<0
    assert p["lower_m"]==pytest.approx(.014-2e-6,abs=1e-12)
    assert m.validate(p,v,nodes) is True

def test_bent_polyline_does_not_fill_hidden_chord():
    m=module();v=cube()
    nodes=np.array([[-.02,0,0],[-.02,.02,0],[.02,.02,0],[.02,0,0]])
    p=m.certify(v,nodes)
    assert p["old_global_xz_lower_m"]<0 and p["lower_m"]>0
    assert len(p["segments"])==3

def test_crossing_or_touching_capsules_fail_clearance():
    m=module()
    for nodes in (np.array([[-.02,0,0],[.02,0,0]]),
                  np.array([[0,.006,0],[0,.02,0]])):
        assert m.certify(cube(),nodes)["lower_m"]<0

def test_certificate_recomputed_from_all_vertices_not_only_selected_hull():
    m=module();v=cube();nodes=np.array([[0,.02,0],[0,.03,0]])
    p=m.certify(v,nodes);v2=np.vstack([v,[0,.02,0]])
    with pytest.raises(ValueError):m.validate(p,v2,nodes)

def test_tampering_and_renormalization_rejected():
    import copy
    m=module();v=cube();nodes=np.array([[0,.02,0],[0,.03,0]])
    p=m.certify(v,nodes)
    for key,value in [("lower_m",1.),("rod_radius_m",.001),("numeric_guard_m",0.)]:
        q=copy.deepcopy(p);q[key]=value
        with pytest.raises(ValueError):m.validate(q,v,nodes)
    q=copy.deepcopy(p);q["segments"][0]["normal"]=[0,20,0]
    with pytest.raises(ValueError):m.validate(q,v,nodes)

def test_lower_bound_never_exceeds_known_point_segment_distance():
    m=module();rng=np.random.default_rng(713)
    for _ in range(20):
        v=rng.normal(size=(9,3))*.003
        nodes=rng.normal(size=(5,3))*.02
        p=m.certify(v,nodes)
        upper=float("inf")
        for a,b in zip(nodes[:-1],nodes[1:]):
            d=b-a;t=np.clip(((v-a)@d)/(d@d),0,1)
            upper=min(upper,np.linalg.norm(v-a-t[:,None]*d,axis=1).min())
        assert p["lower_m"]<=upper-.005-2e-6+1e-10
        assert p["lower_m"]>=p["old_global_xz_lower_m"]-1e-12
        assert m.validate(p,v,nodes)

def test_repeated_nodes_and_planar_mesh_are_supported_without_qhull_joggle():
    m=module();v=cube()[:4]
    nodes=np.array([[0,.02,0],[0,.02,0],[0,.03,0]])
    p=m.certify(v,nodes)
    assert p["lower_m"]>0 and m.validate(p,v,nodes)

@pytest.mark.parametrize("vertices,nodes",[
    (np.empty((0,3)),np.zeros((2,3))),
    (np.zeros((2,3)),np.zeros((1,3))),
    (np.array([[np.nan,0,0]]),np.zeros((2,3))),
    (np.zeros((2,2)),np.zeros((2,3))),
])
def test_unknown_or_malformed_geometry_fails_closed(vertices,nodes):
    with pytest.raises(ValueError):module().certify(vertices,nodes)

def test_original_global_bound_and_old_runtime_source_remain_unchanged():
    from pathlib import Path
    import hashlib
    root=Path(__file__).resolve().parents[1]
    p=root/"src/go2_pvcnn/control/dh116/tabletop_measured_closure_geometry.py"
    assert hashlib.sha256(p.read_bytes()).hexdigest()=="3e90dca3dfc2448ad2a1e82f545c29e7229fbda7510b90fba45bf88a5a7132de"
