import math
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_bimanual_tracking_guard import (
    BimanualTrackingGuard, TrackingGuardState, intrinsic_xyz_from_measured_pose,
    tracking_state_from_native,
)
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import compose_rotation


def state(n=1, dtype=torch.float64):
    pose=torch.zeros(n,2,7,dtype=dtype)
    pose[...,3]=1
    return TrackingGuardState(
        active_tracking_error_rad=torch.zeros(n,2,dtype=dtype),
        mimic_error_rad=torch.zeros(n,2,dtype=dtype),
        live=torch.ones(n,dtype=torch.bool), measured_palm_offset_m=torch.zeros(n,2,3,dtype=dtype),
        actual_base_pose_w=pose.clone(), wrist_seed_pose_w=pose.clone(),
        current_palm_target_m=torch.zeros(n,2,3,dtype=dtype),
        current_wrist_angles_rad=torch.zeros(n,2,3,dtype=dtype),
    )


def actions(n=1, dtype=torch.float64):
    return torch.linspace(-.9,.9,24,dtype=dtype).repeat(n,1)


def test_disabled_guard_is_bit_exact_and_needs_no_measurement():
    raw=actions(3)
    issued,info=BimanualTrackingGuard(3).project(raw)
    assert torch.equal(issued,raw)
    assert issued.data_ptr()!=raw.data_ptr()
    assert not info['enabled']
    assert not info['pause_hand'].any()


def test_only_unsafe_hand_wrist_is_paused_and_fingers_continue():
    s=state(2)
    s.active_tracking_error_rad[0,1]=.031
    s.mimic_error_rad[1,0]=.012
    s.measured_palm_offset_m[0,1]=torch.tensor([.001,-.002,.003])
    raw=actions(2)
    original=raw.clone()
    issued,info=BimanualTrackingGuard(2,enabled=True).project(raw,s)
    assert info['pause_hand'].tolist()==[[False,True],[True,False]]
    assert torch.equal(issued[:,:12],original[:,:12])
    assert torch.equal(issued[0,12:15],original[0,12:15])
    assert torch.equal(issued[0,18:21],original[0,18:21])
    assert issued[0,15:18].eq(0).all() and issued[0,21:24].eq(0).all()
    assert torch.equal(issued[1,15:18],original[1,15:18])
    assert torch.equal(info['hold_offset_m'][0,1],s.measured_palm_offset_m[0,1])
    assert info['cancel_translation'][0,1] and info['cancel_rotation'][0,1]
    assert torch.equal(raw,original)
    assert issued.abs().le(raw.abs()).all()


def test_coupled_pause_propagates_only_within_the_same_live_environment():
    s=state(4)
    s.active_tracking_error_rad[0,0]=.04
    s.mimic_error_rad[1,1]=.018
    s.active_tracking_error_rad[3,0]=.04
    s.live[3]=False
    s.measured_palm_offset_m[0,0]=torch.tensor([-.001,.002,.003])
    s.measured_palm_offset_m[0,1]=torch.tensor([.004,-.002,.006])
    s.measured_wrist_angles_rad=torch.zeros(4,2,3,dtype=torch.float64)
    s.measured_wrist_angles_rad[0]=torch.tensor([[.1,.2,.3],[-.2,.1,-.1]])
    s.actual_base_pose_w[...,3:]=compose_rotation(s.wrist_seed_pose_w[...,3:],s.measured_wrist_angles_rad)
    raw=actions(4)
    issued,info=BimanualTrackingGuard(4,enabled=True,coupled_pause=True).project(raw,s)
    assert info['local_pause_hand'].tolist()==[[True,False],[False,True],[False,False],[False,False]]
    assert info['pause_hand'].tolist()==[[True,True],[True,True],[False,False],[False,False]]
    assert info['propagated_partner_pause'].tolist()==[[False,True],[True,False],[False,False],[False,False]]
    assert torch.equal(issued[:,:12],raw[:,:12])
    assert issued[:2,12:].eq(0).all()
    assert torch.equal(issued[2:],raw[2:])
    assert info['cancel_translation'][:2].all() and info['cancel_rotation'][:2].all()
    assert torch.equal(info['hold_offset_m'][0],s.measured_palm_offset_m[0])
    assert not torch.equal(info['hold_offset_m'][0,0],info['hold_offset_m'][0,1])
    assert torch.equal(info['hold_intrinsic_xyz_rad'][0],s.measured_wrist_angles_rad[0])
    assert not torch.equal(info['hold_intrinsic_xyz_rad'][0,0],info['hold_intrinsic_xyz_rad'][0,1])
    assert info['coupled_pause']
    assert info['version']=='dh116-bimanual-tracking-wrist-projection-v2'


def test_coupled_pause_keeps_healthy_distinct_xyz_and_disabled_requests_exact():
    raw=actions(2)
    s=state(2)
    active,info=BimanualTrackingGuard(2,enabled=True,coupled_pause=True).project(raw,s)
    assert torch.equal(active,raw)
    assert not info['pause_hand'].any()
    assert not info['propagated_partner_pause'].any()
    assert not torch.equal(active[:,12:15],active[:,15:18])
    disabled,info=BimanualTrackingGuard(2,coupled_pause=True).project(raw)
    assert torch.equal(disabled,raw)
    assert not info['pause_hand'].any() and not info['local_pause_hand'].any()


def test_default_independent_profile_does_not_pause_the_healthy_partner():
    raw=actions()
    s=state()
    s.mimic_error_rad[0,0]=.018
    implicit,info=BimanualTrackingGuard(1,enabled=True).project(raw,s)
    explicit,other=BimanualTrackingGuard(1,enabled=True,coupled_pause=False).project(raw,s)
    assert torch.equal(implicit,explicit)
    assert torch.equal(implicit[:,15:18],raw[:,15:18])
    assert torch.equal(implicit[:,21:24],raw[:,21:24])
    assert torch.equal(info['local_pause_hand'],info['pause_hand'])
    assert not info['propagated_partner_pause'].any()
    assert not other['coupled_pause']


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_intrinsic_inverse_uses_authored_seed_and_quaternion_sign(dtype):
    torch.manual_seed(1165)
    seed_angle=(torch.rand(200,2,3,dtype=dtype)-.5)*2.8
    seed=compose_rotation(torch.tensor([1.,0,0,0],dtype=dtype).expand(200,2,4),seed_angle)
    actual_angle=(torch.rand(200,2,3,dtype=dtype)-.5)*2.9
    actual=compose_rotation(seed,actual_angle)
    actual[::2]*=-1
    recovered,valid,error=intrinsic_xyz_from_measured_pose(seed,actual,torch.zeros_like(actual_angle))
    assert valid.all()
    assert torch.allclose(recovered,actual_angle,atol=2e-5 if dtype==torch.float32 else 2e-12,rtol=0)
    assert error.max()<2e-5 if dtype==torch.float32 else error.max()<2e-12


@pytest.mark.parametrize('pitch',[-math.pi/2,math.pi/2])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_gimbal_configuration_is_recovered_without_internal_branch_jump(pitch,dtype):
    seed=torch.tensor([1.,0,0,0],dtype=dtype).reshape(1,1,4)
    measured=torch.tensor([[[.7,pitch,-.3]]],dtype=dtype)
    actual=compose_rotation(seed,measured)
    reference=torch.tensor([[[.72,pitch,-.28]]],dtype=dtype)
    recovered,valid,error=intrinsic_xyz_from_measured_pose(seed,actual,reference,measured)
    tolerance=1e-5 if dtype==torch.float32 else 1e-12
    assert valid.all() and error.max()<tolerance
    assert torch.equal(recovered,measured)
    mathematical,known,err=intrinsic_xyz_from_measured_pose(seed,actual,reference)
    assert known.all() and err.max()<tolerance
    assert (mathematical-reference).abs().max()<.03


def test_unrepresentable_measured_rotation_retains_target_and_marks_unavailable():
    s=state()
    s.mimic_error_rad[0,1]=.018
    measured=torch.zeros(1,2,3,dtype=torch.float64)
    measured[0,1,1]=math.pi/2+.02
    s.measured_wrist_angles_rad=measured
    s.actual_base_pose_w[...,3:]=compose_rotation(s.wrist_seed_pose_w[...,3:],measured)
    old=s.current_wrist_angles_rad.clone()
    issued,info=BimanualTrackingGuard(1,enabled=True).project(actions(),s)
    assert issued[0,21:24].eq(0).all()
    assert not info['cancel_rotation'][0,1]
    assert info['rotation_cancellation_unavailable'][0,1]
    assert torch.equal(info['hold_intrinsic_xyz_rad'][0,1],old[0,1])
    assert torch.equal(s.current_wrist_angles_rad,old)


def test_native_angles_must_reproduce_actual_seed_relative_pose():
    s=state()
    s.active_tracking_error_rad[0,1]=.04
    s.measured_wrist_angles_rad=torch.zeros(1,2,3,dtype=torch.float64)
    actual=torch.tensor([[[0.,0,0],[.1,0,0]]],dtype=torch.float64)
    s.actual_base_pose_w[...,3:]=compose_rotation(s.wrist_seed_pose_w[...,3:],actual)
    _,info=BimanualTrackingGuard(1,enabled=True).project(actions(),s)
    assert not info['cancel_rotation'][0,1]
    assert info['rotation_reproduction_error_rad'][0,1]>.09


def test_invalid_measurement_stops_wrist_without_fabricating_hold_pose():
    s=state()
    s.active_tracking_error_rad[0,1]=float('nan')
    s.actual_base_pose_w[0,1,0]=float('nan')
    s.actual_base_pose_w[0,1,3:]=0
    s.measured_palm_offset_m[0,1]=float('nan')
    issued,info=BimanualTrackingGuard(1,enabled=True).project(actions(),s)
    assert info['pause_hand'][0,1]
    assert issued[0,15:18].eq(0).all() and issued[0,21:24].eq(0).all()
    assert not info['cancel_rotation'][0,1] and not info['cancel_translation'][0,1]
    assert torch.equal(info['hold_offset_m'][0,1],s.current_palm_target_m[0,1])


def test_no_pause_state_leaks_into_next_episode_and_inactive_hand_is_preserved():
    guard=BimanualTrackingGuard(1,enabled=True)
    s=state()
    s.active_tracking_error_rad[:]=.05
    _,first=guard.project(actions(),s)
    assert first['pause_hand'].all()
    fresh=state()
    actual,second=guard.project(actions(),fresh)
    assert not second['pause_hand'].any() and torch.equal(actual,actions())
    fresh.live[:]=False
    fresh.active_tracking_error_rad[:]=.1
    actual,third=guard.project(actions(),fresh)
    assert not third['pause_hand'].any() and torch.equal(actual,actions())


def test_native_builder_reads_real_sdk_targets_and_current_offset_chart():
    s=state()
    q=torch.zeros(1,2,9,dtype=torch.float64)
    q[:,:,6:]=torch.tensor([.1,.2,-.3])
    seed=s.wrist_seed_pose_w.clone()
    seed[:,:,:3]=torch.tensor([1.,2.,3.])
    actual=seed.clone()
    actual[...,2]+=.06
    actual[...,3:]=compose_rotation(seed[...,3:],q[:,:,6:])
    hands=[]
    active=('finger11','finger12','finger21','finger31','finger41','finger51')
    names=list(active)+['wrist_roll','wrist_pitch','wrist_yaw']
    for hi in range(2):
        target=q[:,hi].clone();target[:,1]+=.04*hi
        hands.append(SimpleNamespace(joint_names=names,data=SimpleNamespace(joint_pos_target=target)))
    world=SimpleNamespace(n=1,roots=actual,root_seed=seed,palms=torch.ones(1,2,3)*.001,
        root_targets=actual.clone(),wrist_angles=torch.zeros(1,2,3),q=q,
        commands=torch.zeros(1,2,6),hands=hands,
        w=SimpleNamespace(ids=[list(range(6))]*2,mounts=[SimpleNamespace(angular_ids=[6,7,8])]*2))
    task=SimpleNamespace(done=torch.zeros(1,dtype=torch.bool),offset=torch.tensor([.06]),
        m={'q':torch.zeros(1,12,dtype=torch.float64),'mimic_errors':torch.zeros(1,10)})
    measured=tracking_state_from_native(world,task)
    assert measured.active_tracking_error_rad[0,1]==pytest.approx(.04)
    assert measured.measured_palm_offset_m.abs().max()<1e-8
    assert torch.equal(measured.measured_wrist_angles_rad,q[:,:,6:])
    assert measured.tracking_target_source=='native_articulation_joint_pos_target'
