"""Free pose targets are policy-owned, bounded and independent across slots."""
import importlib
import importlib.util
from types import SimpleNamespace
import pytest
import torch
from test_dh116_tensor_task import template
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
from go2_pvcnn.control.dh116.tabletop_policy_force_region import enable as enable_force
from go2_pvcnn.control.dh116.tabletop_free_grasp_lift import enable as enable_free


def module():
    name='go2_pvcnn.control.dh116.tabletop_free_wrist_grasp'
    assert importlib.util.find_spec(name) is not None, 'free wrist integration is missing'
    return importlib.import_module(name)


def setup(n=2):
    t=TensorTask(template(),n,'cpu',pressure_relief=True,contact_reward_window='close_start',
                 vertical_force_reward=True,vertical_force_reward_mode='continuous')
    t.enable_policy_control();enable_force(t);enable_free(t)
    pose=torch.zeros(n,2,7,dtype=torch.float64);pose[:,:,3]=1.
    t.m.update(wrist_pose=pose.clone(),wrist_velocity=torch.zeros(n,2,6,dtype=torch.float64),
               wrist_joint_angles=torch.zeros(n,2,3,dtype=torch.float64))
    w=SimpleNamespace(n=n,device='cpu',continuous_motion=True,free_wrist_grasp=True,
        palms=torch.zeros(n,2,3,dtype=torch.float64),wrist_angles=torch.zeros(n,2,3,dtype=torch.float64),
        root_seed=pose,w=SimpleNamespace(mounts=[SimpleNamespace(angular_ids=[3,4,5])]*2))
    w.measure=lambda:t.m
    module().enable(t,w)
    return t,w


def test_identity_and_intrinsic_rotations_are_real_quaternions():
    seed=torch.tensor([[[1.,0.,0.,0.],[1.,0.,0.,0.]]],dtype=torch.float64)
    angles=torch.zeros(1,2,3,dtype=torch.float64)
    torch.testing.assert_close(module().compose_rotation(seed,angles),seed)
    angles[0,0,0]=torch.pi/2
    q=module().compose_rotation(seed,angles)
    torch.testing.assert_close(q[0,0],torch.tensor([2**-.5,2**-.5,0.,0.],dtype=torch.float64))
    torch.testing.assert_close(q[0,1],seed[0,1])


def test_enable_exposes_full_joint_limits_and_26_actions():
    t,w=setup()
    assert t.action_size==26 and t.free_wrist_grasp
    assert torch.equal(t.lower,t.low) and torch.equal(t.upper,t.high)
    assert t.phase.tolist()==[3,3] and not hasattr(t,'policy_lift_hold')
    t.begin_step(torch.zeros(2,26))
    with pytest.raises(ValueError,match='26'):
        t.begin_step(torch.zeros(2,20))


def test_joint_target_integrates_current_pose_instead_of_home():
    t,w=setup();live=torch.ones(2,dtype=torch.bool)
    middle=(t.low+t.high)/2
    t.command[:]=middle;t.m['q'][:]=middle
    zero=torch.zeros(2,26);zero[:,-2:]=-.4
    module().control(t,zero,live,.01)
    torch.testing.assert_close(t.command,middle.expand(2,-1))
    action=zero.clone();action[0,0]=.5
    module().control(t,action,live,.01)
    assert t.command[0,0]>middle[0]
    torch.testing.assert_close(t.command[1],middle)


def test_pinch_hold_does_not_reverse_closing_to_chase_finite_follower_error():
    t,w=setup();t.pinch_force_hold=True;middle=(t.low+t.high)/2
    t.command[:]=middle;t.m['q'][:]=middle;t.m['mimic_errors'][:,6]=-.03
    t.m['hand_object_loads'][:]=2.
    action=torch.zeros(2,26);action[:,8]=.1
    module().control(t,action,torch.ones(2,dtype=torch.bool),.01)
    assert (t.command[:,8]>middle[8]).all()
    assert (t.command>=t.low).all() and (t.command<=t.high).all()


def test_pinch_hold_retains_a_stationary_grip_with_finite_mimic_error():
    t,w=setup();t.pinch_force_hold=True;middle=(t.low+t.high)/2
    t.command[:]=middle;t.m['q'][:]=middle;t.m['mimic_errors'][:,6]=-.03
    module().control(t,torch.zeros(2,26),torch.ones(2,dtype=torch.bool),.01)
    torch.testing.assert_close(t.command,middle.expand(2,-1))


def test_regular_policy_keeps_its_existing_follower_correction():
    t,w=setup();middle=(t.low+t.high)/2
    t.command[:]=middle;t.m['q'][:]=middle;t.m['mimic_errors'][:,6]=-.03
    action=torch.zeros(2,26);action[:,8]=.1
    module().control(t,action,torch.ones(2,dtype=torch.bool),.01)
    assert (t.command[:,8]<middle[8]).all()


def test_independent_wrist_translation_and_rotation_velocity_targets():
    t,w=setup();action=torch.zeros(2,26);action[:,-2:]=-.4
    action[0,12]=1.;action[0,18]=1.;action[1,17]=1.;action[1,23]=-1.
    module().control(t,action,torch.ones(2,dtype=torch.bool),.01)
    assert t.palm_targets[0,0,0]>0 and t.palm_targets[1,1,2]>0
    assert t.rotation_targets[0,0,0]>0 and t.rotation_targets[1,1,2]<0
    assert t.rotation_targets[0,1].abs().sum()==0
    assert t.rotation_targets.norm(dim=-1).max()<=.5*.01+1e-10


def test_rotation_and_translation_stop_at_configured_bounds():
    t,w=setup();w.palms[:]=torch.tensor([.03,.03,.05])
    w.wrist_angles[:]=torch.pi/2
    module().control(t,torch.ones(2,26),torch.ones(2,dtype=torch.bool),.01)
    assert (t.rotation_targets<=torch.pi/2).all()
    assert (t.palm_targets<=torch.tensor([.03,.03,.05])+1e-8).all()
    assert (t.command>=t.low).all() and (t.command<=t.high).all()


def test_observation_contains_current_targets_and_measured_orientation():
    t,w=setup();before=t.observe().clone()
    t.rotation_targets[0,0,0]=.2
    after=t.observe()
    assert before.shape==after.shape and before.shape[0]==2
    assert not torch.equal(before[0],after[0]) and torch.equal(before[1],after[1])
    assert torch.isfinite(after).all()


def test_selected_reset_clears_controller_state_only_for_selected_slot():
    t,w=setup();t.rotation_targets[:]=.3;t.actions[:]=.2;t.previous_actions[:]=.1
    t.reset(torch.tensor([0]),w.measure())
    assert not t.rotation_targets[0].any() and not t.actions[0].any()
    assert torch.all(t.rotation_targets[1]==.3) and torch.all(t.actions[1]==.2)


def test_inactive_slot_does_not_receive_policy_motion():
    t,w=setup();command=t.command.clone()
    module().control(t,torch.ones(2,26),torch.tensor([True,False]),.01)
    torch.testing.assert_close(t.command[1],command[1])
    assert not t.rotation_targets[1].any() and not t.palm_targets[1].any()


def test_26d_rsl_collection_storage_and_recovery_support():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import make_algorithm,observations
    obs=observations(torch.zeros(2,14))
    algorithm=make_algorithm(obs,3,action_size=26)
    assert algorithm.storage.actions.shape==(3,2,26)
    assert algorithm.act(obs).shape==(2,26)


def test_hand_dof_counts_are_separate_from_policy_command_size():
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,MIMIC
    c=module().CONTRACT
    assert len(ACTIVE)*2==c['finger_active_joints']==12
    assert len(MIMIC)*2==c['finger_passive_joints']==10
    assert len(JOINTS)*2==c['hand_body_dof_count']==22
    assert c['policy_command_size']==26 and c['normal_force_parameters']==2


def test_real_control_dispatch_has_no_mimic_only_wrist_veto():
    t,w=setup();t.m['mimic_errors'][0]=.04
    action=torch.zeros(2,26);action[0,18]=1.;action[:,-2:]=-.4
    bounded=t.begin_step(action)
    old=t.command.clone()
    allowed=t.control(bounded,torch.ones(2,dtype=torch.bool),.01)
    assert allowed.tolist()==[True,True] and t.rotation_targets[0,0,0]>0
    assert (t.command-old).abs().max()<=module().CONTRACT['joint_rate_rad_s']*.01+1e-10


def test_zero_object_quaternion_in_excluded_row_does_not_break_other_observations():
    t,w=setup();measurement={k:v.clone() for k,v in t.m.items()}
    before=t.m['object_quat'].clone();measurement['object_quat'][0]=0.
    # Env excluded this bad row from scoring and owns its isolated reset.
    t.consume(measurement,torch.tensor([False,True]),1/480,vertical_force_n=torch.zeros(2))
    torch.testing.assert_close(t.m['object_quat'][0],before[0])
    assert torch.isfinite(t.observe()).all()


def test_resolved_contract_describes_26_commands_and_no_pose_force_servo():
    t,w=setup()
    old=dict(policy_control={'automatic_force_correction':'all_episode_contact_feedback'},
        force_feedback={'enabled':True},free_grasp_lift={'actions':'old20'},
        palm_transport={'motion':'three_prismatic_drives'},initial_force_target_n=2.25)
    c=module().apply_contract(old,t)
    assert c['policy_control']['automatic_force_correction'] is False
    assert c['force_feedback']['enabled'] is False
    assert c['initial_force_target_n']==5. and c['initial_force_action_mean']==-.4
    assert c['palm_rate_m_s']==.01
    assert c['palm_exploration']['lower_m']==[-.03,-.03,-.004]
    assert c['palm_transport']['motion']=='three_prismatic_and_three_revolute_drives'
    assert '26' in c['free_grasp_lift']['actions']
    assert c['grasp_controller']['version']=='free_wrist_grasp_v1'
    assert c['force_region_control']['control'] is False
    assert c['force_region_control']['action_source']=='final_two_of_26_policy_commands'


def test_zero_wrist_quaternion_resets_only_affected_physics_slot():
    from test_continuous_motion_recovery import make_env
    t,w,env=make_env('none')
    original=w.advance
    def packet(*a,**kw):
        m=original(*a,**kw)
        pose=torch.zeros(2,2,7);pose[:,:,3]=1.
        if w.tick==3:pose[0,:,3:]=0.
        return dict(m,wrist_pose=pose)
    w.advance=packet
    obs,reward,done,extra=env.step(torch.zeros(2,20))
    assert w.resets==[[0]] and done.tolist()==[True,False]
    assert extra['recovered'].tolist()==[True,False]
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()


def test_reset_with_zero_wrist_quaternion_stops_before_another_physics_step():
    from test_continuous_motion_recovery import make_env
    t,w,env=make_env('velocity')
    original=w.reset
    def invalid_reset(ids):
        m=original(ids)
        pose=torch.zeros(2,2,7);pose[:,:,3]=1.;pose[0,:,3:]=0.
        return dict(m,wrist_pose=pose)
    w.reset=invalid_reset
    with pytest.raises(RuntimeError,match='immediate_reset'):
        env.step(torch.zeros(2,20))
    assert w.resets==[[0]] and w.tick==3


def test_saved_configuration_uses_resolved_schema_and_retains_physical_readback(tmp_path):
    import json
    from go2_pvcnn.control.dh116 import tabletop_tensor_training as training
    assert hasattr(training,'publish_resolved_tensor_configuration')
    p=tmp_path/'configuration.json'
    p.write_text(json.dumps(dict(contract={'action':'old20'},physical_parameter_readback={'real':True})))
    current={'action':'26D_free','observation_size':261}
    training.publish_resolved_tensor_configuration(tmp_path,current)
    saved=json.loads(p.read_text())
    assert saved['contract']==current and saved['tensor_setup_resolved'] is True
    assert saved['physical_parameter_readback']=={'real':True}
