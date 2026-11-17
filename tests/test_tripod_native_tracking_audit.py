"""CPU-only replay contracts; never grants native/Teacher qualification."""
from copy import deepcopy
import numpy as np
import pytest
from scripts.audit_dh116_tripod_native_tracking import (
    Sequence, mount_pose, fk_joint_positions, joint_tracking, classify_clearance)

def identity(step):
    return dict(run_uuid="test",env_id=0,episode_uid="test:1",reset_id=1,rod_uid="rod0",
        physics_step=step,rod_tick=(step-2)*12,policy_step=(step-3)//8,
        substep=(step-3)%8,timestamp_s=step/480)
def test_terminal_prefix_preserved_without_full_budget_claim():
    s=Sequence(identity(2))
    for step in range(3,12):s.record(identity(step),terminal="guard" if step==11 else None)
    assert s.summary()==dict(physics_rows=9,last_sdk_step=11,terminal="guard",
        complete_budget=False,new_physics_steps=0,teacher_admission=False)
    with pytest.raises(ValueError,match="terminal"):s.record(identity(12))
@pytest.mark.parametrize("key,value",[("env_id",1),("episode_uid","another"),("reset_id",2),
    ("physics_step",4),("rod_tick",13),("policy_step",1),("substep",1),("timestamp_s",0.)])
def test_sequence_rejects_identity_and_clock_mismatch(key,value):
    s=Sequence(identity(2));row=identity(3);row[key]=value
    with pytest.raises(ValueError):s.record(row)
def test_mount_translation_is_anchor_local_rotation_intrinsic():
    # anchor +90deg Z; local +X maps world +Y, orientation Rx then Ry.
    q=np.sqrt(.5)
    p=mount_pose([1,2,3,q,0,0,q],[.1,0,0],[.2,.3,0])
    np.testing.assert_allclose(p[:3],[1,2.1,3],atol=1e-14)
    from scipy.spatial.transform import Rotation as R
    expected=R.from_euler("Z",np.pi/2)*R.from_euler("XYZ",[.2,.3,0])
    assert (R.from_quat(p[[4,5,6,3]]).inv()*expected).magnitude()<1e-14
def test_fk_uses_actual_passive_position_not_ideal_mimic():
    t0=np.eye(4);t0[0,3]=.1
    edges=[("p","root","child",t0,np.eye(4),np.array([0.,0.,1.]))]
    poses=fk_joint_positions("root",edges,[0,0,0,1,0,0,0],{"p":.6})
    np.testing.assert_allclose(poses["child"][:3,3],[.1,0,0])
    assert abs(poses["child"][0,0]-np.cos(.6))<1e-14
    with pytest.raises(ValueError):fk_joint_positions("root",edges,[0,0,0,1,0,0,0],{})
def test_passive_zero_sdk_targets_are_not_tracking_commands():
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand
    names=["wrist_x","wrist_y","wrist_z","wrist_roll","wrist_pitch","wrist_yaw",*JOINTS]
    q=np.r_[np.zeros(6),expand([.5,.3,.6,.7,.8,.9])]
    target=q.copy()
    for i,n in enumerate(names):
        if n not in ACTIVE and n in JOINTS:target[i]=0
    row=dict(joint_names=names,q=q[None],qd=np.zeros((1,len(q))),backend_targets=target[None],
        active_indices=[names.index(n) for n in ACTIVE],
        measured_source="physx_backend_get_dof_positions/get_dof_velocities",
        backend_source="physx_backend_get_dof_position_targets")
    result=joint_tracking(row)
    assert result["passive_coupling_error_rad"]==0
    assert result["maximum_active_error_rad"]==0
    assert result["maximum_mount_rotation_error_rad"]==0
    assert result["passive_sdk_zero_targets_are_not_commands"] is True
    row["q"][0,0]=np.nan
    with pytest.raises(ValueError):joint_tracking(row)
@pytest.mark.parametrize("value",[-1e-5,0,1e-5])
def test_negative_certificate_is_not_measured_contact(value):
    result=classify_clearance(value)
    assert result["geometric_clearance_proven"] is (value>=0)
    assert result["physical_contact_proven"] is False
    assert result["penetration_depth_m"] is None

def test_control_end_requires_exact_terminal_sample_count():
    from scripts.audit_dh116_tripod_native_tracking import validate_control_end
    s=Sequence(identity(2))
    for step in range(3,12):s.record(identity(step),terminal="guard" if step==11 else None)
    valid=dict(control_step=1,physical_samples=1,termination_reason="guard")
    validate_control_end(s,valid)
    for key,value in [("physical_samples",8),("control_step",0),("termination_reason",None)]:
        wrong=dict(valid);wrong[key]=value
        with pytest.raises(ValueError):validate_control_end(s,wrong)
