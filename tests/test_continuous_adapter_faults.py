"""The real tensor adapter must return identifiable bad rows before recovery."""
from types import SimpleNamespace
import pytest
import torch
from test_dh116_tensor_task import template
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
from go2_pvcnn.control.dh116.tabletop_tensor_world import TensorWorld
from go2_pvcnn.control.dh116.tabletop_palm_mount import check_transport


def adapter():
    t=TensorTask(template(),2,'cpu')
    w=object.__new__(TensorWorld)
    w.n=2;w.device='cpu';w.dt=1/480;w.continuous_motion=True
    w.adapter_fault_bits=torch.zeros(2,dtype=torch.long)
    q=t.m['q'].reshape(2,2,6).clone()
    pose=torch.zeros((2,1,7),dtype=torch.float64);pose[:,:,2]=.4;pose[:,:,3]=1.
    w.hands=[SimpleNamespace(update=lambda dt:None,data=SimpleNamespace(
        joint_pos=q[:,h],joint_vel=torch.zeros((2,6)),body_link_pose_w=pose.clone(),
        root_pose_w=pose[:,0].clone())) for h in range(2)]
    w.wire=SimpleNamespace(update=lambda dt:None,data=SimpleNamespace(
        root_pos_w=torch.tensor([[.6,0.,.308]]*2,dtype=torch.float64),
        root_quat_w=torch.tensor([[1.,0.,0.,0.]]*2,dtype=torch.float64),
        root_vel_w=torch.zeros((2,6),dtype=torch.float64)))
    count=len(t.link_groups)
    sensors=[SimpleNamespace(update=lambda *a,**k:None,data=SimpleNamespace(
        contact_pos_w=torch.full((2,1,1,3),float('nan'),dtype=torch.float64),
        force_matrix_w=torch.zeros((2,1,1,3),dtype=torch.float64),
        friction_forces_w=torch.zeros((2,1,1,3),dtype=torch.float64))) for _ in range(2*count)]
    w.w=SimpleNamespace(sensors=sensors,pad_ids=[[0,0],[0,0]],ids=[list(range(6))]*2,mounts=None)
    w.limits=torch.stack((t.low,t.high),-1).reshape(1,2,6,2).repeat(2,1,1,1)
    w.shapes=[(h,0,torch.zeros((1,3),dtype=torch.float64)) for h in range(2)]
    w.shape_hands=[[0],[1]];w.mimic_ids=[[(0,0,1.,0.)]*5]*2
    w.pad_points=torch.zeros((4,3),dtype=torch.float64)
    w.pad_indices=[0,0,count,count]
    w.group_matrix=torch.nn.functional.one_hot(t.link_groups,6).double()
    w.palms=torch.zeros((2,2,3),dtype=torch.float64)
    w.gains=torch.full((2,2),10.,dtype=torch.float64)
    w.goal_roots=torch.zeros((2,2,3),dtype=torch.float64)
    w.spawn=w.wire.data.root_pos_w.clone()
    w.root_targets=pose[:,0,None].repeat(1,2,1)
    return w


def test_bad_quaternion_returns_original_nan_and_only_marks_affected_row():
    w=adapter();w.wire.data.root_quat_w[0,0]=float('nan')
    m=w.refresh()
    assert torch.isnan(m['object_quat'][0,0])
    assert w.adapter_fault_bits.tolist()==[1,0]
    assert not m['contact_valid'][0].any()
    assert torch.isfinite(m['object_quat'][1]).all()


@pytest.mark.parametrize('field,axis,bit',[('force_matrix_w',2,2),('friction_forces_w',0,4)])
def test_bad_sensor_vector_reaches_recovery_instead_of_global_reducer_error(field,axis,bit):
    w=adapter()
    raw=getattr(w.w.sensors[0].data,field)
    raw[0,0,0,axis]=float('nan')
    m=w.refresh()
    assert torch.isnan(raw[0,0,0,axis])
    assert w.adapter_fault_bits.tolist()==[bit,0]
    assert torch.isfinite(m['vertical_hand_force_n'][1]).all()


def test_nonfinite_friction_does_not_make_normal_only_measurement_throw():
    w=adapter();w.refresh()
    w.vertical_components[0,0,1]=float('nan')
    m=w.measure()
    assert m['vertical_hand_force_n'][0,0]==0.
    assert torch.isnan(w.vertical_components[0,0,1])


def test_transport_returns_per_row_validity_for_unified_recovery():
    pose=torch.zeros((2,1,7));pose[:,:,3]=1.
    hands=[SimpleNamespace(data=SimpleNamespace(
        body_link_pose_w=pose.clone(),body_lin_vel_w=torch.zeros((2,1,3)))) for _ in range(2)]
    mounts=[SimpleNamespace(base_id=0)]*2
    target=pose[:,0,None].repeat(1,2,1)
    target[0,0,0]=.004
    hands[0].data.body_lin_vel_w[1,0,0]=float('nan')
    mask=check_transport(hands,mounts,target,return_validity=True)
    assert mask.tolist()==[True,False]  # finite tracking lag is not invalid state
    with pytest.raises(RuntimeError,match='transport guard'):
        check_transport(hands,mounts,target)


@pytest.mark.parametrize('field', ['velocity', 'target'])
def test_post_reset_refresh_revalidates_transport_before_next_physics(field):
    w=adapter();w.refresh()
    w.w.mounts=[SimpleNamespace(base_id=0)]*2
    w.transport_warning_mask=torch.zeros(2,dtype=torch.bool)
    for h in w.hands:
        h.data.body_lin_vel_w=torch.zeros((2,1,3))
    if field=='velocity':
        w.hands[0].data.body_lin_vel_w[0,0,0]=float('nan')
    else:
        w.root_targets[0,0,0]=float('nan')
    # TensorWorld.reset ends here, before the next sim.step. Cleared failure
    # bits must not hide a still-invalid backend readback or reset target.
    w.adapter_fault_bits.zero_()
    result=w.refresh(sensors=False)
    assert result['adapter_fault_bits'].tolist()==[8,0]
    assert w.transport_warning_mask.tolist()==[True,False]


def test_post_reset_refresh_rejects_zero_object_quaternion():
    w=adapter();w.refresh()
    w.wire.data.root_quat_w[0]=0.
    w.adapter_fault_bits.zero_()
    result=w.refresh(sensors=False)
    assert result['adapter_fault_bits'].tolist()==[1,0]
    assert not result['object_quat'][0].any()
