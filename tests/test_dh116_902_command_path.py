import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation, Slerp
from scripts.screen_dh116_902_command_path import command_step


def inputs():
    seed=torch.tensor([[[.5,.15,.45,1.,0.,0.,0.],[.5,-.15,.45,1.,0.,0.,0.]]],dtype=torch.float64)
    return dict(action=np.zeros((1,24)),backend_targets=np.full((1,12),.4),
        measured_q=np.full((1,12),.35),seed=seed.numpy(),measured_roots=seed.numpy(),
        mount_anchor_poses=seed.numpy().copy(),
        clearance=np.full((1,2),.05),palms=np.zeros((1,2,3)),angles=np.zeros((1,2,3)))


def test_backend_target_and_first_substep_are_preserved():
    v=inputs();v["action"][0,8]=.2
    out=command_step(**v)
    assert out["finger_targets"][0,8]>.4
    assert out["finger_targets"][0,8]<.401
    np.testing.assert_allclose(out["finger_targets"][0,:6],.4)
    assert out["native_qualification"] is False


def test_command_output_retains_actual_float32_sdk_write():
    v=inputs();v["action"][0,8]=.234567
    out=command_step(**v)
    assert out["submitted_finger_targets"].dtype==np.float32
    np.testing.assert_array_equal(out["submitted_finger_targets"],out["finger_targets"].astype(np.float32))


def test_joint_lead_guard_depends_on_actual_q():
    v=inputs();v["backend_targets"][:]=.7;v["measured_q"][:]=.1
    out=command_step(**v)
    assert np.all(out["finger_targets"]<.7)


def test_wrist_path_is_seed_relative_intrinsic_euler_not_slerp():
    v=inputs();v["angles"][0,1]=[.1,-.2,.3];v["action"][0,21:24]=[.3,-.2,.4]
    out=command_step(**v)
    q=out["projection"]["projected_root_poses"][0,1,[4,5,6,3]]
    expected=Rotation.from_euler("XYZ",out["projection"]["projected_angles"][0,1])
    assert (Rotation.from_quat(q).inv()*expected).magnitude()<1e-12
    a=Rotation.from_euler("XYZ",[.1,-.2,.3]);b=Rotation.from_euler("XYZ",[.4,.3,-.2])
    euler_mid=Rotation.from_euler("XYZ",[.25,.05,.05])
    spherical=Slerp([0,1],Rotation.from_quat([a.as_quat(),b.as_quat()]))(.5)
    assert (euler_mid.inv()*spherical).magnitude()>.01


def test_table_projection_and_wrist_vector_limit_are_reused():
    v=inputs();v["clearance"][:]=.0021;v["action"][0,14]=-1.
    out=command_step(**v)
    assert out["projection"]["palm_offsets_m"][0,0,2]>=0.
    v=inputs();v["action"][0,12:15]=1.
    out=command_step(**v)
    assert np.linalg.norm(out["projection"]["palm_offsets_m"][0,0])<=.025/480+1e-12


@pytest.mark.parametrize("key",["action","backend_targets","measured_q","clearance"])
def test_reject_nonfinite_input(key):
    v=inputs();v[key].flat[0]=np.nan
    with pytest.raises(ValueError):command_step(**v)


def test_reject_carriage_outside_original_mount_limit():
    from go2_pvcnn.control.dh116.tabletop_palm_mount import CONTRACT
    v=inputs();v['mount_anchor_poses'][0,1,0]-=CONTRACT['limit_m']+.001
    with pytest.raises(ValueError,match='carriage'):
        command_step(**v)


def test_reject_invalid_mount_quaternion():
    v=inputs();v['mount_anchor_poses'][0,1,3:]=0.
    with pytest.raises(ValueError,match='quaternion'):
        command_step(**v)


def test_projected_state_mismatch_cannot_hide_behind_matching_requests():
    from scripts.screen_dh116_902_command_path import replay_errors
    v=inputs();out=command_step(**v)
    request=dict(requested_finger_targets_rad=out['finger_targets'],
                 requested_palm_offsets_m=out['requested_palms'],
                 requested_angles_rad=out['requested_angles'])
    projection={k:val.copy() if isinstance(val,np.ndarray) else val for k,val in out['projection'].items()}
    projection['projected_angles'][0,1,0]+=.001
    projection['palm_offsets_m'][0,1,0]+=.002
    errors=replay_errors(out,request,projection,out['submitted_finger_targets'],out['submitted_wrist_targets'])
    assert errors['projected_angles']==pytest.approx(.001)
    assert errors['projected_palms']==pytest.approx(.002)
    assert errors['projected_root']==0.


def test_wrist_submission_keeps_float32_write_distinct_from_integrator():
    v=inputs();v['action'][0,15:18]=[.12345,.23456,.34567]
    v['action'][0,21:24]=[.4321,.5432,.6543]
    out=command_step(**v)
    expected=np.concatenate([out['local_carriage_targets'],out['projection']['projected_angles']],axis=-1).astype(np.float32)
    np.testing.assert_array_equal(out['submitted_wrist_targets'],expected)
    assert out['submitted_wrist_targets'].dtype==np.float32
    assert out['projection']['projected_angles'].dtype==np.float64
