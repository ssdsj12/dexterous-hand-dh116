"""Post-control / pre-transport runtime, using native-buffer-shaped CPU data."""
import importlib
from types import SimpleNamespace as NS

import pytest
import torch


def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_tracking_pause_runtime')


def scene():
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    pose=torch.zeros(1,2,7,dtype=torch.float64);pose[...,3]=1.
    actual=pose.clone();actual[0,:,:3]=torch.tensor([[.001,.002,.010],[-.001,-.002,.012]])
    backend=torch.zeros(1,6,dtype=torch.float32);backend[0,0]=.02
    hands=[NS(joint_names=ACTIVE,data=NS(joint_pos_target=backend.clone())) for _ in range(2)]
    for hand in hands:
        hand.root_physx_view=NS(targets=backend.clone())
        hand.root_physx_view.get_dof_position_targets=lambda view=hand.root_physx_view:view.targets
        hand.root_physx_view.get_dof_types=lambda:torch.zeros(1,6,dtype=torch.int32)
    world=NS(n=1,_physics_steps=5,dt=1/480,roots=actual.clone(),root_seed=pose.clone(),
        palms=torch.ones(1,2,3,dtype=torch.float64)*.008,wrist_angles=torch.zeros(1,2,3,dtype=torch.float64),
        commands=torch.ones(1,2,6,dtype=torch.float64)*.023456789012,root_targets=pose.clone(),hands=hands,
        w=NS(ids=[list(range(6))]*2,mounts=None,sim=NS(current_time_step_index=5,current_time=5/480)))
    task=NS(n=1,command=world.commands.reshape(1,12).clone(),done=torch.tensor([False]),
        offset=torch.tensor([.004],dtype=torch.float64),low=torch.full((12,),-1.,dtype=torch.float64),
        high=torch.ones(12,dtype=torch.float64),palm_targets=world.palms.clone(),rotation_targets=world.wrist_angles.clone(),
        m=dict(q=torch.zeros(1,12,dtype=torch.float64),qd=torch.zeros(1,12,dtype=torch.float64),mimic_errors=torch.zeros(1,10,dtype=torch.float64)))
    order=[]
    def original(action,live,dt):
        order.append('original_control');task.command+=.0001
        task.palm_targets.fill_(.009);task.rotation_targets.fill_(.001)
        task.offset.fill_(.006)  # cancellation must use the current lift chart
        return live.clone()
    task.control=original
    return world,task,order


def sdk_step(w):
    w._physics_steps+=1;w.w.sim.current_time_step_index+=1;w.w.sim.current_time+=w.dt


def submit(w):
    for h,hand in enumerate(w.hands):
        hand.data.joint_pos_target=w.commands[:,h].float().clone()
        hand.root_physx_view.targets=w.commands[:,h].float().clone()


def feedback(world,task,live):
    fingers=torch.zeros(1,12,dtype=torch.bool);fingers[0,1]=True
    return dict(mandatory_release_mask=fingers,mandatory_wrist_release_mask=torch.zeros(1,2,dtype=torch.bool),
        overload=torch.zeros(1,dtype=torch.bool))


def test_runtime_cancels_both_queued_wrists_after_original_control_and_preserves_original_release():
    w,t,order=scene();raw=torch.ones(1,24,dtype=torch.float64)*.1;live=torch.tensor([True]);before=t.command.clone();roots=w.roots.clone()
    r=module().CommonTrackingPauseRuntime(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),enabled=True,source='teacher_hybrid')
    allowed,info=r.control(w,t,raw,live,w.dt,original_guard_feedback=feedback)
    assert order==['original_control'] and allowed.all()
    assert info['pause_hand'].all()
    assert t.command[0,1]==before[0,1]+.0001
    assert torch.equal(t.command[0,[0,2,3,4,5,6,7,8,9,10,11]],before[0,[0,2,3,4,5,6,7,8,9,10,11]])
    actual_offset=w.roots[:,:,:3]-w.root_seed[:,:,:3];actual_offset[:,:,2]-=t.offset[:,None]
    assert torch.equal(w.palms,actual_offset) and torch.equal(t.palm_targets,actual_offset)
    assert not t.rotation_targets.any() and torch.equal(w.roots,roots)
    assert torch.equal(r.pending['raw_action24'],raw)
    assert not r.pending['native_submission_confirmed']
    assert r.pending['backend_target_readback_before_control_rad'].shape==(1,12)
    # Simulate only original advance's target writes, without advancing physics.
    w.commands=t.command.reshape(1,2,6).clone()
    submit(w);sdk_step(w)
    record=r.after_advance(w,t,live)
    assert record['native_submission_confirmed']
    assert record['logical_command_submitted_rad'].dtype==torch.float64
    assert record['sdk_expected_float32_rad'].dtype==torch.float32
    assert record['backend_target_readback_after_advance_rad'].dtype==torch.float32
    assert not torch.equal(record['logical_command_submitted_rad'],record['backend_target_readback_after_advance_rad'].double())


def test_installed_helper_is_explicit_hybrid_only_and_can_restore_original_control():
    w,t,order=scene();original=t.control
    with pytest.raises(ValueError,match='hybrid'):
        module().install_common_tracking_pause(w,t,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),source='pure_student',original_guard_feedback=feedback)
    r=module().install_common_tracking_pause(w,t,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),
        source='teacher_hybrid',original_guard_feedback=feedback)
    t.control(torch.zeros(1,24,dtype=torch.float64),torch.tensor([True]),w.dt)
    assert order==['original_control'] and t.control is not original
    r.uninstall()
    assert t.control is original


def test_disabled_runtime_calls_original_exactly_and_does_not_require_native_buffers():
    w,t,order=scene();original=t.control
    r=module().CommonTrackingPauseRuntime(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64))
    allowed,info=r.control(w,t,torch.zeros(1,24,dtype=torch.float64),torch.tensor([True]),w.dt)
    assert allowed.all() and order==['original_control'] and not info['enabled']
    assert t.palm_targets.eq(.009).all() and r.pending is None
    assert t.control is original


def test_enabled_runtime_requires_original_guard_feedback_and_forbids_mixing_tripod_reservations():
    w,t,_=scene();r=module().CommonTrackingPauseRuntime(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),enabled=True,source='teacher_hybrid')
    with pytest.raises(ValueError,match='guard feedback'):
        r.control(w,t,torch.zeros(1,24,dtype=torch.float64),torch.tensor([True]),w.dt)
    t.tripod_recovery_runtime=NS(pending='reservation')
    with pytest.raises(ValueError,match='independent'):
        r.control(w,t,torch.zeros(1,24,dtype=torch.float64),torch.tensor([True]),w.dt,original_guard_feedback=feedback)


def test_native_readback_mismatch_cannot_be_hidden_by_float32_recasting():
    w,t,_=scene();live=torch.tensor([True])
    r=module().CommonTrackingPauseRuntime(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),enabled=True,source='teacher_hybrid')
    r.control(w,t,torch.zeros(1,24,dtype=torch.float64),live,w.dt,original_guard_feedback=feedback)
    w.commands=t.command.reshape(1,2,6).clone()
    for h,hand in enumerate(w.hands):
        hand.data.joint_pos_target=w.commands[:,h].float().double()+1e-12
        hand.root_physx_view.targets=w.commands[:,h].float().double()+1e-12
    sdk_step(w)
    record=r.after_advance(w,t,live)
    assert not record['native_submission_confirmed']
    assert record['backend_target_readback_after_advance_rad'].dtype==torch.float64


def test_inactive_scope_does_not_cancel_queued_target_of_another_episode():
    w,t,_=scene();prior=w.palms.clone()
    r=module().CommonTrackingPauseRuntime(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),enabled=True,source='teacher_hybrid')
    r.control(w,t,torch.zeros(1,24,dtype=torch.float64),torch.tensor([False]),w.dt,original_guard_feedback=feedback)
    assert torch.equal(w.palms,prior) and t.palm_targets.eq(.009).all()


def test_installed_env_step_order_calls_original_control_then_original_advance_once():
    w,t,order=scene();original=t.control
    def original_advance(task,allowed,live):
        order.append('original_advance');w.commands=task.command.reshape(1,2,6).clone()
        submit(w);sdk_step(w)
        return {'actual_measured_q':task.m['q'].clone()}
    w.advance=original_advance
    r=module().install_common_tracking_pause(w,t,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),
        source='teacher_hybrid',original_guard_feedback=feedback)
    with pytest.raises(ValueError,match='already'):
        module().install_common_tracking_pause(w,t,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),
            source='teacher_hybrid',original_guard_feedback=feedback)
    live=torch.tensor([True]);allowed=t.control(torch.zeros(1,24,dtype=torch.float64),live,w.dt)
    m=w.advance(t,allowed,live)
    assert order==['original_control','original_advance'] and 'actual_measured_q' in m
    assert len(r.records)==1 and r.records[0]['native_submission_confirmed']
    r.uninstall()
    assert t.control is original and w.advance is original_advance


def test_actual_physx_target_drives_warning_while_command_cache_remains_separate():
    w,t,_=scene();live=torch.tensor([True])
    for hand in w.hands:hand.data.joint_pos_target.zero_()  # cache healthy; backend error .02
    r=module().CommonTrackingPauseRuntime(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),enabled=True,source='teacher_hybrid')
    _,info=r.control(w,t,torch.zeros(1,24,dtype=torch.float64),live,w.dt,original_guard_feedback=feedback)
    assert info['pause_hand'].all()
    assert info['tracking_target_source']=='physx_backend_get_dof_position_targets'
    assert r.pending['backend_target_readback_before_control_rad'][0,0]==pytest.approx(.02)
    assert not r.pending['isaaclab_command_cache_before_control_rad'].any()
    w.commands=t.command.reshape(1,2,6).clone()
    for h,hand in enumerate(w.hands):
        hand.root_physx_view.targets=w.commands[:,h].float().clone()
        hand.data.joint_pos_target.fill_(.9)
    sdk_step(w);record=r.after_advance(w,t,live)
    assert record['native_submission_confirmed']
    assert record['isaaclab_command_cache_after_advance_rad'].eq(.9).all()
    assert not torch.equal(record['isaaclab_command_cache_after_advance_rad'],record['backend_target_readback_after_advance_rad'])
    assert record['target_source']=='physx_backend_get_dof_position_targets'
    assert record['target_readback_clock_before_control']['clock_aligned']
    assert record['target_readback_clock_after_advance']['sdk_physics_step']==6


def test_missing_physx_getter_rejects_before_control_even_when_cache_exists():
    w,t,order=scene();del w.hands[0].root_physx_view.get_dof_position_targets
    r=module().CommonTrackingPauseRuntime(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),enabled=True,source='teacher_hybrid')
    with pytest.raises(ValueError,match='PhysX backend'):
        r.control(w,t,torch.zeros(1,24,dtype=torch.float64),torch.tensor([True]),w.dt,original_guard_feedback=feedback)
    assert not order and r.pending is None


def test_sdk_clock_change_during_target_readback_is_rejected():
    w,t,order=scene()
    def drifting_getter():
        w.w.sim.current_time_step_index+=1
        return w.hands[0].root_physx_view.targets
    w.hands[0].root_physx_view.get_dof_position_targets=drifting_getter
    r=module().CommonTrackingPauseRuntime(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),enabled=True,source='teacher_hybrid')
    with pytest.raises(ValueError,match='clock'):
        r.control(w,t,torch.zeros(1,24,dtype=torch.float64),torch.tensor([True]),w.dt,original_guard_feedback=feedback)
    assert not order


@pytest.mark.parametrize('actor_dtype',[torch.float32,torch.float64])
def test_real_original_control_and_transport_execute_with_pause_between_them(actor_dtype):
    from test_free_wrist_grasp import setup
    from test_dh116_tensor_world import palm_world
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    t,_=setup(n=3);w,_=palm_world();t.free_wrist_world=w
    t.surface_force_sharing=True;t.surface_policy_owned_targets=True;t.surface_redistribution=False
    w.free_wrist_grasp=True;w.continuous_motion=True;w.w.mounts=None
    w.wrist_angles=torch.zeros(3,2,3,dtype=torch.float64);w._physics_steps=0
    middle=(t.low+t.high)/2;t.command[:]=middle;t.m['q'][:]=middle;t.m['qd'].zero_();t.offset.zero_()
    w.commands=t.command.reshape(3,2,6).clone();w.palms.fill_(.008)
    w._root_pose_written=w.root_seed.float().clone()
    w.w.sim.current_time_step_index=0;original_step=w.w.sim.step;steps=[]
    def step(**kwargs):
        steps.append('original_sim_step');original_step(**kwargs);w.w.sim.current_time_step_index+=1
    w.w.sim.step=step
    writes=[]
    for h,hand in enumerate(w.hands):
        hand.joint_names=ACTIVE;hand.data=NS(joint_pos_target=w.commands[:,h].float().clone())
        hand.root_physx_view=NS(targets=hand.data.joint_pos_target.clone()+.02)
        hand.root_physx_view.get_dof_position_targets=lambda view=hand.root_physx_view:view.targets
        hand.root_physx_view.get_dof_types=lambda:torch.zeros(3,6,dtype=torch.int32)
        def set_target(value,*,joint_ids,active_hand=hand):active_hand.data.joint_pos_target=value.clone()
        def write(active_hand=hand,side=h):
            writes.append(side);active_hand.root_physx_view.targets=active_hand.data.joint_pos_target.clone()
        hand.set_joint_position_target=set_target;hand.write_data_to_sim=write
    def original_feedback(world,task,live):
        return dict(mandatory_release_mask=torch.zeros(3,12,dtype=torch.bool),
            mandatory_wrist_release_mask=torch.zeros(3,2,dtype=torch.bool),overload=torch.zeros(3,dtype=torch.bool))
    r=module().install_common_tracking_pause(w,t,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),
        source='teacher_hybrid',original_guard_feedback=original_feedback)
    live=torch.ones(3,dtype=torch.bool);before=t.command.clone();roots=w.roots.clone()
    try:
        raw=torch.full((3,24),.25,dtype=actor_dtype)
        allowed=t.control(raw,live,w.dt)
        assert torch.equal(r.pending['raw_action24'],raw)
        assert r.pending['original_controller_action24'].dtype==torch.float64
        assert torch.equal(r.pending['original_controller_action24'],raw.to(t.command))
        assert (r.pending['original_governed_target_rad']>before).all()
        assert torch.equal(t.command,before) and not t.palm_targets.any() and not t.rotation_targets.any()
        w.advance(t,allowed,live)
    finally:r.uninstall()
    assert steps==['original_sim_step'] and writes==[0,1]
    assert r.records[0]['native_submission_confirmed']
    assert torch.equal(w.root_targets,w.root_seed) and torch.equal(w.roots,roots)
    assert not w.hands[0].calls and not w.hands[1].calls
    assert 'control' not in vars(t) and 'advance' not in vars(w)


@pytest.mark.parametrize('bad',['quaternion','target_chart'])
def test_unrepresentable_ordinary_wrist_hold_refuses_before_original_advance(bad):
    w,t,order=scene();before=t.command.clone()
    if bad=='quaternion':w.roots[...,3:]=0.
    else:w.roots[...,:3]=1.
    actual=w.roots.clone()
    def advance(task,allowed,live):order.append('unsafe_original_advance')
    w.advance=advance
    r=module().install_common_tracking_pause(w,t,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),
        source='teacher_hybrid',original_guard_feedback=feedback)
    live=torch.tensor([True])
    try:
        with pytest.raises(RuntimeError,match='ordinary wrist hold'):
            t.control(torch.ones(1,24,dtype=torch.float64)*.1,live,w.dt)
        assert order==['original_control'] and not t.done.any()
        assert t.command[0,1]==before[0,1]+.0001  # original mandatory release remains untouched
        assert torch.equal(w.roots,actual) and w._physics_steps==5
        assert r.pending['candidate_motion_qualified'] is False and not r.pending['native_submission_confirmed']
        with pytest.raises(RuntimeError,match='not qualified'):
            w.advance(t,live,live)
        assert order==['original_control']
    finally:r.uninstall()


def test_original_mandatory_wrist_unloading_does_not_require_a_candidate_hold_certificate():
    w,t,_=scene();w.roots[...,3:]=0.
    def mandatory(world,task,live):
        result=feedback(world,task,live);result['mandatory_wrist_release_mask'][:]=True
        return result
    r=module().CommonTrackingPauseRuntime(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),enabled=True,source='teacher_hybrid')
    allowed,info=r.control(w,t,torch.ones(1,24,dtype=torch.float64)*.1,torch.tensor([True]),w.dt,
        original_guard_feedback=mandatory)
    assert allowed.all() and not info['cancel_translation'].any() and not info['cancel_rotation'].any()
    assert t.palm_targets.eq(.009).all() and t.rotation_targets.eq(.001).all()
    assert r.pending['candidate_motion_qualified'] is True


def test_float32_actor_request_is_logged_raw_while_guard_uses_original_command_dtype():
    w,t,_=scene();raw=torch.ones(1,24,dtype=torch.float32)*.1
    r=module().CommonTrackingPauseRuntime(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),enabled=True,source='teacher_hybrid')
    _,info=r.control(w,t,raw,torch.tensor([True]),w.dt,original_guard_feedback=feedback)
    assert info['pause_hand'].all() and r.pending['raw_action24'].dtype==torch.float32
    assert t.command.dtype==torch.float64 and r.pending['wrist_target_after_pause_offset_m'].dtype==torch.float64


def test_actual_vision_inference_float32_request_reaches_original_float64_controller_without_new_scaling():
    from go2_pvcnn.control.dh116.tabletop_vision_inference import VisionInference
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import SQUASHED_VERSION
    actor=torch.nn.Linear(3844,24)
    with torch.no_grad():actor.weight.zero_();actor.bias.copy_(torch.linspace(-.3,.3,24))
    inference=VisionInference(actor,SQUASHED_VERSION,'fixture-only',{}, {})
    raw=inference.act({'policy':torch.zeros(1,3844,dtype=torch.float64)})
    assert raw.dtype==torch.float32
    w,t,_=scene();original=t.control;received=[]
    def capture(action,live,dt):
        received.append(action.detach().clone())
        return original(action,live,dt)
    t.control=capture
    r=module().CommonTrackingPauseRuntime(1,teacher_error_p999_rad=torch.zeros(12,dtype=torch.float64),enabled=True,source='hybrid')
    r.control(w,t,raw,torch.tensor([True]),w.dt,original_guard_feedback=feedback)
    assert torch.equal(received[0],raw) and received[0].dtype==torch.float32
    assert torch.equal(r.pending['raw_action24'],raw)
    assert torch.equal(r.pending['original_controller_action24'],raw.to(t.command))
