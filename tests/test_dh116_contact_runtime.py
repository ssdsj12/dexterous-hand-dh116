from fixture_paths import ARCHIVE, ASSET, BACKEND, CONTACT, CONTACT_SHA, SURFACE_REFS
import hashlib
from pathlib import Path
import numpy as np
import pytest
pytest.MonkeyPatch().syspath_prepend(str(BACKEND))
from go2_pvcnn.control.fixed_dual_panda_o6.contact_bridge import BodyState
from go2_pvcnn.control.dh116.contact_runtime import BODY_NAMES,load_mesh_manifest,mesh_motions,probe_rod_start


def test_probe_places_element_midpoint_not_node_at_contact_and_accepts_readonly():
    direction=np.array([1.,0.,0.]);direction.setflags(write=False)
    start,tangent=probe_rod_start([0,0,0],[0,0,1],direction)
    np.testing.assert_allclose(start+tangent*(20.5*.5/40),[0,0,.0079],atol=1e-14)
    np.testing.assert_allclose(direction,[1,0,0])
    with pytest.raises(ValueError):probe_rod_start([0,0,0],[0,0,1],[0,0,1])
    start,tangent=probe_rod_start([0,0,0],[0,0,1],direction,overlap_m=1e-5)
    np.testing.assert_allclose(start+tangent*(20.5*.5/40),[0,0,.00799],atol=1e-14)
    for overlap in [0.,.002,np.nan]:
        with pytest.raises(ValueError):probe_rod_start([0,0,0],[0,0,1],direction,overlap_m=overlap)


def states(time=0.,token='a',name=None):
    return {c:BodyState(name or n,[1,2,3],np.eye(3),[1,2,2.9],[.1,.2,.3],[0,0,2],time,token)
            for c,n in BODY_NAMES.items()}


def test_dh116_motion_uses_mesh_com_and_physical_com_separately():
    offsets={c:[.02,0,0] for c in BODY_NAMES}
    result=mesh_motions(states(),offsets,None,.01)
    for m in result.values():
        np.testing.assert_allclose(m.position_w,[1.02,2,3])
        np.testing.assert_allclose(m.velocity_w,[.1,.24,.3])
        np.testing.assert_allclose(m.acceleration_w,0)
    next_result=mesh_motions(states(.01,'b'),offsets,states(),.01)
    for m in next_result.values():np.testing.assert_allclose(m.acceleration_w,0)


def test_dh116_rejects_old_body_names_stale_time_and_nonfinite_offsets():
    offsets={c:[0,0,0] for c in BODY_NAMES}
    with pytest.raises(ValueError):mesh_motions(states(name='left_index_distal'),offsets,None,.01)
    with pytest.raises(ValueError):mesh_motions(states(),offsets,states(),.01)
    offsets['left_thumb']=[np.nan,0,0]
    with pytest.raises(ValueError):mesh_motions(states(),offsets,None,.01)


def test_first_order_prediction_preserves_twist_without_differenced_acceleration():
    offsets={c:[0,0,0] for c in BODY_NAMES};before=states()
    after={c:BodyState(n,[1,2,3],np.eye(3),[1,2,2.9],[.2,.3,.4],[0,0,3],.01,'b')
           for c,n in BODY_NAMES.items()}
    first=mesh_motions(after,offsets,before,.01,prediction_order=1)
    second=mesh_motions(after,offsets,before,.01,prediction_order=2)
    for c in BODY_NAMES:
        np.testing.assert_allclose(first[c].velocity_w,second[c].velocity_w)
        np.testing.assert_allclose(first[c].omega_w,second[c].omega_w)
        np.testing.assert_allclose(first[c].acceleration_w,0)
        assert np.linalg.norm(second[c].acceleration_w)>0
    with pytest.raises(ValueError):mesh_motions(after,offsets,before,.01,prediction_order=0)


def test_pose_delta_motion_does_not_move_stationary_geometry_using_reported_solver_twist():
    offsets={c:[.02,0,0] for c in BODY_NAMES}
    motions=mesh_motions(states(.0001,'b'),offsets,states(),.0001,
                         prediction_order=1,velocity_source='pose_delta')
    for m in motions.values():
        np.testing.assert_array_equal(m.velocity_w,np.zeros(3))
        np.testing.assert_array_equal(m.omega_w,np.zeros(3))
        np.testing.assert_array_equal(m.acceleration_w,np.zeros(3))
    # No imported pose history on initialization: preserve the reported twist
    # for the first sample, never infer a stationary hand merely from restart.
    initial=mesh_motions(states(),offsets,None,.0001,prediction_order=1,velocity_source='pose_delta')
    for m in initial.values():np.testing.assert_allclose(m.velocity_w,[.1,.24,.3])


def test_pose_delta_motion_preserves_actual_translation_rotation_and_mesh_offset():
    from scipy.spatial.transform import Rotation
    dt=.0001;R=Rotation.from_rotvec([0,0,.2*dt]).as_matrix()
    offsets={c:[.02,0,0] for c in BODY_NAMES}
    old=states();new={c:BodyState(n,[1+.001*dt,2,3],R,[1,2,2.9],[99,99,99],
                                [99,99,99],dt,'b') for c,n in BODY_NAMES.items()}
    motions=mesh_motions(new,offsets,old,dt,prediction_order=1,velocity_source='pose_delta')
    expected=(np.array([.001*dt,0,0])+(R-np.eye(3))@np.array([.02,0,0]))/dt
    for m in motions.values():
        np.testing.assert_allclose(m.velocity_w,expected,atol=2e-12,rtol=0)
        np.testing.assert_allclose(m.omega_w,[0,0,.2],atol=1e-12,rtol=0)
    with pytest.raises(ValueError,match='first-order'):
        mesh_motions(new,offsets,old,dt,prediction_order=2,velocity_source='pose_delta')


def test_pose_window_rejects_quantization_spikes_without_hiding_real_fast_motion():
    from go2_pvcnn.control.dh116.contact_runtime import PoseDeltaMotionWindow
    dt=.0001;offsets={c:[0,0,0] for c in BODY_NAMES}
    for speed in (.0002,.004):
        window=PoseDeltaMotionWindow(dt=dt,steps=10)
        measured=[]
        for step in range(41):
            x=np.round(speed*step*dt/2e-7)*2e-7
            sample={c:BodyState(n,[x,0,0],np.eye(3),[x,0,0],[.01,0,0],[0,0,0],
                               step*dt,str(step)) for c,n in BODY_NAMES.items()}
            motions=window.update(sample,offsets)
            if step>=10:measured.append(motions['right_thumb'].velocity_w[0])
        np.testing.assert_allclose(measured,speed,atol=1e-15,rtol=0)
        assert window.interval_s==pytest.approx(.001)
        if speed>.001:assert min(measured)>.001
        else:assert max(measured)<.001
        with pytest.raises(ValueError,match='contiguous'):
            late={c:BodyState(n,[x,0,0],np.eye(3),[x,0,0],[.01,0,0],[0,0,0],
                             .0042,'late') for c,n in BODY_NAMES.items()}
            window.update(late,offsets)
        window.reset()
        initial=window.update(states(),offsets)
        assert window.interval_s==dt
        np.testing.assert_allclose(initial['right_thumb'].velocity_w,[.1,.2,.3])


def test_load_real_manifest_requires_external_pin():
    path=CONTACT
    if not path.exists():pytest.skip('local contact mesh manifest missing')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    m=load_mesh_manifest(path,digest)
    assert tuple(m)==tuple(BODY_NAMES)
    with pytest.raises(ValueError):load_mesh_manifest(path,'0'*64)


@pytest.mark.parametrize('extra,allowed',[(1.61e-15,True),(1e-9,False)])
def test_contact_duration_accepts_roundoff_but_rejects_extra_integration(extra,allowed):
    from co_sim.moving_contact_models import ContactImpulse
    from go2_pvcnn.control.fixed_dual_panda_o6.contact_bridge import reaction_wrenches
    dt=1/240
    impulses={c:ContactImpulse(np.zeros(3),np.zeros(3),1.,dt+extra) for c in BODY_NAMES}
    entries={c:dict(evaluations=417,linear_balance_w=np.zeros(3),mesh_linear_w=np.zeros(3),
        rod_linear_w=np.zeros(3),angular_origin_balance_w=np.zeros(3),
        mesh_angular_origin_w=np.zeros(3),rod_angular_origin_w=np.zeros(3)) for c in BODY_NAMES}
    diagnostic=dict(integrated_duration_s=dt,table_label='table',evaluation_count=417*4,channels=entries)
    positions={c:np.zeros(3) for c in BODY_NAMES}
    if allowed:assert len(reaction_wrenches(impulses,diagnostic,positions,dt))==4
    else:
        with pytest.raises(ValueError,match='duration'):reaction_wrenches(impulses,diagnostic,positions,dt)
