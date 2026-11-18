from types import SimpleNamespace as NS
import torch
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,MIMIC
from go2_pvcnn.control.dh116.tabletop_surface_failure_teacher import correction


def test_native_labeler_reads_actual_state_without_writing_controller_or_stepping():
    n=2;z=lambda *shape:torch.zeros(*shape,dtype=torch.float64)
    names=['wrist_roll','wrist_pitch','wrist_yaw']+list(ACTIVE)+[row[0] for row in MIMIC]
    class Physx:
        def get_jacobians(self):return z(n,5,6,len(names))
    hand=NS(joint_names=names,body_names=['base']+[str(i) for i in range(5)],
        root_physx_view=Physx(),data=NS(body_com_pos_w=z(n,6,3)))
    loads=z(n,2,5);loads[:,:,:2]=.1
    vectors=z(n,2,5,3);vectors[:,:,0,0]=.1;vectors[:,:,1,0]=-.1
    points=z(n,2,5,3);points[:,:,0,0]=-.008;points[:,:,1,0]=.008
    ids=torch.arange(1,6).expand(n,2,5).clone()
    pose=z(n,2,7);pose[...,3]=1
    m=dict(effective_pad_loads_n=loads,effective_pad_points_w=points,
        effective_pad_body_ids=ids,effective_pad_valid=loads>0,effective_pad_vectors_w=vectors,
        effective_pad_depth_m=z(n,2,5),rejected_contact_loads_n=z(n,2),q=z(n,12),wrist_pose=pose)
    w=NS(n=n,hands=[hand,hand],w=NS(mounts=[NS(angular_ids=[0,1,2])]*2),
        wp=z(n,3),wq=torch.tensor([[1.,0,0,0]]*n,dtype=torch.float64),
        wv=z(n,6),palms=z(n,2,3),clearances=torch.ones(n,2,dtype=torch.float64),
        wrist_angles=z(n,2,3),measure=lambda:m)
    t=NS(n=n,device=torch.device('cpu'),command=z(n,12),low=-torch.ones(12),
        high=torch.ones(12),failure=torch.zeros(n,dtype=torch.long))
    before={k:v.clone() for k,v in m.items()}
    result=correction(NS(world=w,task=t),z(n,24))
    assert result['actions'].shape==(n,24)
    assert torch.isfinite(result['actions']).all()
    assert result['native_correction_qualified'] is False
    assert not hasattr(t,'pinch_force_hold') and not hasattr(t,'teacher_force_controlled_digits')
    assert not t.command.any() and not w.palms.any()
    for key in m:assert torch.equal(before[key],m[key])
