from fixture_paths import ARCHIVE, ASSET, BACKEND, CONTACT, CONTACT_SHA, SURFACE_REFS
import numpy as np
import pytest
from go2_pvcnn.control.dh116.front_partition import partition_front_cap


def area(tri):
    t=np.asarray(tri,dtype=float).reshape(-1,3,3)
    return np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1).sum()/2


def source():
    return np.array([[-.02,-.01,0],[.02,-.01,0],[0,.04,0.]]),np.array([[0,1,2]])


def test_long_triangle_is_partitioned_not_wholly_exempted_or_rejected():
    v,f=source();r=partition_front_cap(v,f,[0,0,0],[0,0,1])
    front=np.asarray(r['front_triangles']);other=np.asarray(r['remainder_triangles'])
    assert len(front)>0 and len(other)>0
    assert np.linalg.norm(front,axis=2).max()<=.012+1e-12
    assert np.linalg.norm(other,axis=2).max()>.03
    assert area(front)+area(other)==pytest.approx(area(v[f]),abs=1e-12)
    assert set(r['front_source_ids'])=={0} and set(r['remainder_source_ids'])=={0}
    # A sizeable interior disk remains usable, not just a tiny sliver.
    assert area(front)>.00035


def test_disconnected_close_parallel_surface_remains_nonfront():
    v,f=source();v=np.r_[v,v+[0,0,.001]];f=np.r_[f,f+3]
    r=partition_front_cap(v,f,[0,0,0],[0,0,1])
    assert set(r['front_source_ids'])=={0}
    assert 1 in r['remainder_source_ids']
    assert area(r['front_triangles'])+area(r['remainder_triangles'])==pytest.approx(area(v[f]),abs=1e-12)


def test_connected_triangles_both_partition_and_shared_edge_is_not_a_gap():
    v=np.array([[-.02,-.02,0],[.02,-.02,0],[.02,.02,0],[-.02,.02,0]])
    f=np.array([[0,1,2],[0,2,3]])
    r=partition_front_cap(v,f,[0,0,0],[0,0,1])
    assert set(r['front_source_ids'])=={0,1}
    assert area(r['front_triangles'])+area(r['remainder_triangles'])==pytest.approx(.0016,abs=1e-12)


def test_partition_is_invariant_to_subdividing_the_same_source_surface():
    v,f=source();original=partition_front_cap(v,f,[0,0,0],[0,0,1])
    v=np.r_[v,[(v[0]+v[1])/2]];f=np.array([[0,3,2],[3,1,2]])
    refined=partition_front_cap(v,f,[0,0,0],[0,0,1])
    assert area(refined['front_triangles'])==pytest.approx(area(original['front_triangles']),abs=1e-12)
    assert area(refined['remainder_triangles'])==pytest.approx(area(original['remainder_triangles']),abs=1e-12)
    for name in ('front','remainder'):
        for fragment,parent in zip(refined[name+'_triangles'],refined[name+'_source_ids']):
            tri=v[f[parent]];A=np.column_stack((tri[1]-tri[0],tri[2]-tri[0]))
            uv=np.linalg.lstsq(A,(fragment-tri[0]).T,rcond=None)[0].T
            assert (uv>=-1e-10).all() and (uv.sum(1)<=1+1e-10).all()
            np.testing.assert_allclose(tri[0]+uv@A.T,fragment,atol=1e-12)


@pytest.mark.parametrize('change',['normal','nonfinite','index','floating_faces','degenerate','not_surface'])
def test_bad_inputs_fail_closed(change):
    v,f=source();p=[0,0,0];n=[0,0,1]
    if change=='normal':n=[0,0,2]
    if change=='nonfinite':v[0,0]=np.nan
    if change=='index':f[0,2]=10
    if change=='floating_faces':f=f.astype(float)
    if change=='degenerate':v[2]=v[0]
    if change=='not_surface':p=[0,0,.005]
    with pytest.raises(ValueError):partition_front_cap(v,f,p,n)


def test_back_facing_patch_cannot_become_front():
    v,f=source()
    with pytest.raises(ValueError):partition_front_cap(v,f,[0,0,0],[0,0,-1])


def test_actual_left_index_long_face_keeps_far_end_protected():
    import json
    from pathlib import Path
    o3d=pytest.importorskip('open3d')
    root=ARCHIVE
    if not (root/'zero_axial_point_references_v1.json').exists():pytest.skip('local candidate unavailable')
    ref=next(r for r in json.loads((root/'zero_axial_point_references_v1.json').read_text())['references'] if r['side']=='left' and r['digit']=='index')
    rec=json.loads((root/'contact_hulls_v2/manifest.json').read_text())['meshes']['left_index']
    mesh=o3d.io.read_triangle_mesh(str(root/'contact_hulls_v2'/rec['path']))
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);p=np.array(ref['point'])-rec['body_local_mesh_com_m']
    result=partition_front_cap(v,f,p,ref['normal'])
    assert 168 in result['front_source_ids'] and 168 in result['remainder_source_ids']
    assert area(result['front_triangles'])+area(result['remainder_triangles'])==pytest.approx(area(v[f]),abs=1e-12)
    assert np.linalg.norm(result['front_triangles']-p,axis=2).max()<=.012+1e-12
    tri=result['front_triangles'];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.linalg.norm(norm,axis=1)[:,None]
    assert (norm@ref['normal']).min()>=.5-1e-10
    assert result['inscribed_radial_error_bound_m']<.00006
