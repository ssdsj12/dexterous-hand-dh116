from fixture_paths import ARCHIVE, ASSET, BACKEND, CONTACT, CONTACT_SHA, SURFACE_REFS
import numpy as np
import pytest
o3d=pytest.importorskip('open3d')
from go2_pvcnn.control.dh116.pad_region_guard import PadRegionGuard


@pytest.fixture(scope='module')
def guard():
    mesh=o3d.geometry.TriangleMesh.create_sphere(radius=.01,resolution=30)
    mesh.translate([0,0,-.01])
    return PadRegionGuard(np.asarray(mesh.vertices),np.asarray(mesh.triangles),[0,0,0],[0,0,1])


def test_front_touch_not_back_surface_exemption(guard):
    wire=np.array([[0,-.04,.008],[0,.04,.008]])
    r=guard.check(wire,np.eye(3),np.zeros(3),.008)
    assert r['passed'],r
    assert r['nonfront_lower_bound_m']>=.001
    wire[:,2]=-.028
    r=guard.check(wire,np.eye(3),np.zeros(3),.008)
    assert not r['passed'] and r['nonfront_lower_bound_m']<.001


def test_frame_invariance_and_region_is_not_entire_body(guard):
    from scipy.spatial.transform import Rotation
    wire=np.array([[0,-.04,.008],[0,.04,.008]])
    R=Rotation.from_rotvec([.2,-.3,.4]).as_matrix();p=np.array([1.,2.,3.])
    a=guard.check(wire,np.eye(3),np.zeros(3),.008)
    b=guard.check(wire@R.T+p,R,p,.008)
    assert a['passed']==b['passed']
    assert b['nonfront_lower_bound_m']==pytest.approx(a['nonfront_lower_bound_m'],abs=1e-7)
    assert 0<len(guard.front_face_ids)<guard.total_faces


def test_invalid_pose_and_radius_rejected(guard):
    with pytest.raises(ValueError):guard.check([[0,0,0],[0,1,0]],2*np.eye(3),np.zeros(3),.008)
    with pytest.raises(ValueError):guard.check([[0,0,0],[0,1,0]],np.eye(3),np.zeros(3),0.)


def test_actual_pad_guard_does_not_turn_parity_ray_glitch_into_penetration():
    import json
    from pathlib import Path
    root=ARCHIVE
    if not (root/'grasp_point_references_v1.json').exists():pytest.skip('local grasp calibration unavailable')
    refs=json.loads((root/'grasp_point_references_v1.json').read_text())
    ref=next(r for r in refs['references'] if r['side']=='right' and r['digit']=='thumb')
    manifest=json.loads((root/'contact_hulls_v2/manifest.json').read_text())
    rec=manifest['meshes']['right_thumb'];mesh=o3d.io.read_triangle_mesh(str(root/'contact_hulls_v2'/rec['path']))
    region=PadRegionGuard(np.asarray(mesh.vertices),np.asarray(mesh.triangles),
        np.array(ref['point'])-rec['body_local_mesh_com_m'],ref['normal'])
    p=np.array([[-.0163660437423104,.0004152284341645757,-.013940692021180976]])
    _,d,_=region.surface_query(p)
    assert d[0]>.013
    assert region.signed_distance_method=='convex-halfspaces'


def test_actual_long_face_is_only_partly_front_and_old_pilot6_frame_is_rechecked():
    import json
    from pathlib import Path
    pytest.importorskip('pxr.Usd')
    from pxr import Usd
    from go2_pvcnn.control.dh116.usd_kinematics import SideKinematics
    root=ARCHIVE
    if not (root/'closure_pilot_trial6_same_section/summary.json').exists():pytest.skip('local failure evidence unavailable')
    ref=next(r for r in json.loads((root/'zero_axial_point_references_v1.json').read_text())['references'] if r['side']=='left' and r['digit']=='index')
    rec=json.loads((root/'contact_hulls_v2/manifest.json').read_text())['meshes']['left_index']
    m=o3d.io.read_triangle_mesh(str(root/'contact_hulls_v2'/rec['path']))
    offset=np.array(rec['body_local_mesh_com_m']);v=np.asarray(m.vertices);f=np.asarray(m.triangles)
    g=PadRegionGuard(v,f,np.array(ref['point'])-offset,ref['normal'])
    near=np.array([[-.005165629088878632,-.00686117447912693,.01064832229167223]])
    assert g.other.compute_distance(o3d.core.Tensor(near.astype(np.float32))).numpy()[0]>.001
    far=v[f[168,2]][None]
    assert g.other.compute_distance(o3d.core.Tensor(far.astype(np.float32))).numpy()[0]<1e-7
    d=json.loads((root/'closure_pilot_trial6_same_section/summary.json').read_text())
    model=SideKinematics(Usd.Stage.Open(str(ASSET)),'left')
    t=model.fk(d['measured_source_guard_active_q']['left'])[0][model.bodies['finger23_link']]
    old=d['measured_source_guard_failure']['pad_regions']['left_index']
    result=g.check(np.array(d['last_attempted_record']['rod_state']['rod_position']).T,
        t[:3,:3],t[:3,3]+t[:3,:3]@offset,old['wire_radius_m'])
    assert result['passed'],result
    assert result['nonfront_lower_bound_m']>.0011
    assert result['front_cap_method']=='connected-inscribed-icosphere3-clipped-cap'


@pytest.mark.parametrize('channel',['left_thumb','left_index','right_thumb','right_index'])
def test_actual_pinned_dh116_cap_retains_nonfront_margin(channel):
    import json
    from pathlib import Path
    pytest.importorskip('pxr.Usd')
    from pxr import Usd
    from go2_pvcnn.control.dh116.contact_runtime import load_mesh_manifest
    from go2_pvcnn.control.dh116.usd_kinematics import SideKinematics
    root=ARCHIVE
    if not (root/'grasp_point_references_v1.json').exists():pytest.skip('local grasp calibration unavailable')
    records=load_mesh_manifest(CONTACT,
        CONTACT_SHA)
    refs=json.loads((root/'grasp_point_references_v1.json').read_text())
    ref=next(r for r in refs['references'] if r['side']+'_'+r['digit']==channel)
    mesh=o3d.io.read_triangle_mesh(str(records[channel]['resolved_path']))
    offset=np.array(records[channel]['body_local_mesh_com_m'])
    region=PadRegionGuard(np.asarray(mesh.vertices),np.asarray(mesh.triangles),np.array(ref['point'])-offset,ref['normal'])
    stage=Usd.Stage.Open(str(ASSET))
    model=SideKinematics(stage,ref['side'])
    q=json.loads((root/'grasp_source_sweep_v1/summary.json').read_text())['input']['pair']['states'][ref['side']]
    t=model.fk(q)[0][model.bodies[ref['body_name']]]
    r=region.check([[.43,-.25,.308],[.43,.25,.308]],t[:3,:3],t[:3,3]+t[:3,:3]@offset,.008)
    assert r['passed'],r
    assert r['nonfront_lower_bound_m']>.0014
