"""Source-derived mesh motion tests; no CUDA/Isaac or physical qualification."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from go2_pvcnn.control.dh116.tabletop_deformx_substep_pose import (
    pose_at_substep, target_pose, to_matrix, DeferredMeshPose,
    classification_status, verify_source_lock,
)

P=np.array([0.,0.,0.,0.,0.,0.,1.])

def test_translation_uses_substep_plus_one():
    b=P.copy(); b[0]=.012
    assert pose_at_substep(P,b,0)[0]==pytest.approx(.001)
    assert pose_at_substep(P,b,11)[0]==pytest.approx(.012)

def test_quaternion_shortest_arc_and_sign():
    a=P.copy();a[3:]=Rotation.from_euler('x',35,degrees=True).as_quat()
    b=a.copy();b[3:]=(Rotation.from_euler('z',120,degrees=True)*Rotation.from_quat(a[3:])).as_quat()
    expected=(Rotation.from_euler('z',60,degrees=True)*Rotation.from_quat(a[3:])).as_matrix()
    np.testing.assert_allclose(to_matrix(pose_at_substep(a,b,5))[:3,:3],expected,atol=3e-7)
    neg=b.copy();neg[3:]*=-1
    np.testing.assert_allclose(to_matrix(pose_at_substep(a,neg,5)),to_matrix(pose_at_substep(a,b,5)),atol=3e-7)

@pytest.mark.parametrize('sub',[-1,12,True,1.5])
def test_invalid_substep(sub):
    with pytest.raises(ValueError):pose_at_substep(P,P,sub)

@pytest.mark.parametrize('bad',[np.zeros(7),np.ones(6),[0,0,0,0,0,0,float('nan')], [0,0,0,0,0,0,2]])
def test_invalid_pose(bad):
    with pytest.raises(ValueError):target_pose(bad)

def test_teleport_is_pending_and_overwritten_by_false():
    state=DeferredMeshPose(P);reset=P.copy();reset[2]=.060
    state.set_target(reset,teleport=True)
    np.testing.assert_array_equal(state.current,target_pose(P))
    state.set_target(reset,teleport=False)
    poses=state.step_poses()
    assert poses[0][2]==pytest.approx(.005)
    np.testing.assert_allclose(state.current,target_pose(reset))

def test_teleport_not_overwritten_commits_at_begin_not_set():
    state=DeferredMeshPose(P);reset=P.copy();reset[2]=.060
    state.set_target(reset,teleport=True)
    np.testing.assert_allclose(state.current,P)
    for pose in state.step_poses():np.testing.assert_allclose(pose,target_pose(reset))

def test_substep_pose_changes_local_point_membership():
    end=P.copy();end[0]=.012
    point=np.array([.001,0,0])
    local=point-to_matrix(pose_at_substep(P,end,0))[:3,3]
    old=point-to_matrix(target_pose(end))[:3,3]
    assert abs(local[0])<1e-8 and abs(old[0])>.004

def test_cpu_reconstruction_never_grants_measured_contact():
    for accepted in [True,False]:
        status=classification_status(accepted)
        assert status['effective_contact_value'] is None
        assert not status['native_substep_pose_measured']
        assert not status['positive_contact_qualified']
        assert not status['negative_contact_qualified']

def test_source_lock_rejects_modified_bytes(tmp_path):
    import hashlib
    p=tmp_path/'csrc'/'sdf.cu';p.parent.mkdir();p.write_text('a')
    lock={'csrc/sdf.cu':hashlib.sha256(b'a').hexdigest()}
    assert verify_source_lock(tmp_path,lock)==lock
    p.write_text('b')
    with pytest.raises(ValueError,match='SHA'):verify_source_lock(tmp_path,lock)

def test_source_lock_rejects_escape(tmp_path):
    with pytest.raises(ValueError):verify_source_lock(tmp_path,{'../outside':'a'*64})
