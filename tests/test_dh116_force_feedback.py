"""Force feedback uses measured loads, not actuator commands or a pad-count gate."""
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_force_feedback import ReactionForceController, force_feedback_contract
from go2_pvcnn.control.dh116.tabletop_task import TabletopTask


def tick(c, forces=None, action=None, blocked=None, slip=0., phase='close', dt=1/480):
    c.update(action=np.ones(2) if action is None else action,
             forces=np.zeros(4) if forces is None else forces,
             blocked=np.zeros(2,dtype=bool) if blocked is None else blocked,
             slip_m=slip, phase=phase, dt=dt)


def test_insufficient_measured_load_closes_and_deadband_holds():
    c=ReactionForceController()
    for _ in range(30): tick(c,action=-np.ones(2))
    assert np.all(c.offset>0) and np.all(c.offset<=.03*30/480)
    for _ in range(300): tick(c,forces=[.1,0,0,.1],action=-np.ones(2))
    before=c.offset.copy()
    for _ in range(30): tick(c,forces=[.1,0,0,.1],action=-np.ones(2))
    np.testing.assert_allclose(c.offset,before)


def test_excess_measured_force_releases_even_with_protection_active():
    c=ReactionForceController()
    for _ in range(20): tick(c,action=-np.ones(2))
    before=c.offset.copy()
    tick(c,forces=np.full(4,.5),action=-np.ones(2),blocked=np.ones(2,dtype=bool))
    assert np.all(c.offset<before)


def test_protection_blocks_integrator_windup_per_hand():
    c=ReactionForceController()
    for _ in range(200): tick(c,blocked=[True,False])
    assert c.offset[0]==0 and c.offset[1]>0
    assert np.array_equal(c.blocked,[True,False])


def test_sustained_absent_contact_hits_bounded_offset_not_unlimited_force():
    c=ReactionForceController()
    for _ in range(3000): tick(c)
    assert c.offset==pytest.approx([.08,.08])
    assert c.target==pytest.approx([1.,1.])


def test_slip_boost_requires_increasing_slip_during_lift_and_contact():
    c=ReactionForceController()
    tick(c,forces=[.05,0,0,.05],action=-np.ones(2),slip=.004,phase='close')
    assert not c.boost.any()
    tick(c,forces=[.05,0,0,.05],action=-np.ones(2),slip=.005,phase='lift')
    assert np.all(c.boost>0)
    held=c.boost.copy()
    for _ in range(50): tick(c,forces=[.05,0,0,.05],action=-np.ones(2),slip=.005,phase='lift')
    np.testing.assert_array_equal(c.boost,held)
    tick(c,forces=np.zeros(4),action=-np.ones(2),slip=.01,phase='lift')
    np.testing.assert_array_equal(c.boost,held)
    assert np.all(c.requested<=1.)


def test_controller_state_is_observed_and_reset():
    c=ReactionForceController();tick(c)
    assert c.observation().shape==(11,) and np.isfinite(c.observation()).all()
    c.reset();assert not c.offset.any() and not c.boost.any() and not c.filtered.any()


@pytest.mark.parametrize('field,value', [('forces',[0,0,np.nan,0]),('forces',[-.1,0,0,0]),
    ('action',[2,0]),('blocked',[False]),('dt',0),('slip_m',np.nan)])
def test_bad_inputs_rejected_without_state_mutation(field,value):
    c=ReactionForceController();before=c.observation().copy()
    args=dict(action=np.zeros(2),forces=np.zeros(4),blocked=[False,False],slip_m=0.,phase='close',dt=1/480)
    args[field]=value
    with pytest.raises(ValueError):c.update(**args)
    np.testing.assert_array_equal(c.observation(),before)


def test_readiness_uses_measured_per_hand_load_and_not_four_contacts():
    c=ReactionForceController()
    tick(c,action=-np.ones(2))
    kwargs=dict(speed=.001,slip=.001)
    assert c.ready([.1,0,0,.1],**kwargs)
    assert not c.ready([.2,0,0,0],**kwargs)
    assert not c.ready([0,0,0,0],**kwargs)
    assert not c.ready([.1,0,0,.1],speed=.03,slip=.001)


def advance(task,ready,dt=.01,bottom=.3,forces=(.1,0,0,.1)):
    return task.update(dt,approach_error_m=.004,pad_forces_n=forces,clearance_m=.002,
        object_bottom_m=bottom,object_speed_m_s=0.,slip_m=0.,max_force_n=max(forces),grasp_ready=ready)


def test_force_mode_never_lifts_on_clock_alone_and_acquisition_times_out():
    task=TabletopTask(phase='close',grasp_mode='object_stability',force_feedback=True)
    for _ in range(310):advance(task,False)
    assert task.phase=='close' and not task.failure
    for _ in range(300):advance(task,False)
    assert task.failure=='grasp_not_established' and not task.success


def test_stability_dwell_is_continuous_and_loss_returns_to_close():
    task=TabletopTask(phase='close',grasp_mode='object_stability',force_feedback=True)
    for _ in range(21):advance(task,True)
    assert task.phase=='hold'
    advance(task,False)
    assert task.phase=='close' and task.contact_time==0
    for _ in range(52):advance(task,True)
    assert task.phase=='lift' and not task.success
    for _ in range(51):advance(task,True,bottom=.331)
    assert task.success


def test_hard_guards_still_preempt_force_readiness():
    task=TabletopTask(phase='close',grasp_mode='object_stability',force_feedback=True)
    advance(task,True,forces=(31,0,0,.1))
    assert task.failure=='excess_force' and not task.success


def test_legacy_timed_object_mode_unchanged_and_contract_changes():
    task=TabletopTask(phase='close',grasp_mode='object_stability')
    for _ in range(240):advance(task,False,forces=(0,0,0,0))
    assert task.phase=='lift'
    c=force_feedback_contract()
    assert c['observation_size']==136 and c['action']!='20D_joint_palm_stiffness_v1'
    assert c['lift_attempt_after_s'] is None


def force_env():
    from test_dh116_tabletop_adaptive import AdaptiveBackend,make_env
    b=AdaptiveBackend()
    direction=np.tile(np.array([0,1,1,0,0,0])/np.sqrt(2.),(2,1))
    return make_env(b,force_direction=direction)


def test_env_feedback_changes_real_joint_requests_and_exposes_state():
    env=force_env();obs,_=env.reset()
    assert obs.shape==env.observation_space.shape==(136,)
    assert env.action_space.shape==(20,)
    action=np.zeros(20);action[-2:]=-1
    for _ in range(60):env.step(action)
    assert np.all(env.command[[1,2,7,8]]>env.reference[[1,2,7,8]])
    assert env.task.phase=='close'
    assert all(np.array_equal(call[2]['grip_action'],[1,1]) for call in env.backend.calls)
    assert env.log[-1]['force_feedback']['target_force_n']==pytest.approx([.1,.1])
    np.testing.assert_allclose(env.observe()[-11:],env.force_controller.observation())
    env.reset();assert not env.force_controller.offset.any()


def test_env_force_gate_uses_post_physics_measurements_and_old_guards():
    env=force_env();env.reset()
    env.backend.forces[:]=[.1,0,0,.1]
    action=np.zeros(20);action[-2:]=-1
    for _ in range(32):env.step(action)
    assert env.task.phase=='lift' and not env.task.success
    env.backend.forces[:]=[31,0,0,.1]
    _,_,done,_,info=env.step(action)
    assert done and info['failure']=='excess_force'


def test_env_invalid_action_does_not_mutate_feedback_or_step_physics():
    env=force_env();initial,_=env.reset()
    _,reward,done,_,info=env.step(np.zeros(21))
    assert done and reward==-25 and info['failure']=='invalid_action'
    assert not env.backend.calls and not env.force_controller.offset.any()


def test_legacy_mode_rejects_force_controller_instead_of_ignoring_it():
    from test_dh116_tabletop_adaptive import make_env
    with pytest.raises(ValueError,match='adaptive'):
        make_env(mode='joint_target',force_direction=np.ones((2,6))/np.sqrt(6))


def test_contact_gated_admission_allows_learning_from_safe_acquisition_failure(tmp_path):
    from test_dh116_tabletop_vec_training import ToyEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    from go2_pvcnn.control.dh116.tabletop_vec_training import run_vector_training
    class AcquisitionFailure(ToyEnv):
        action_space=ToyEnv.action_space.__class__(-1.,1.,(20,),dtype=np.float32)
        def step(self, action):
            obs,reward,done,truncated,info=super().step(action)
            info.update(is_success=False,failure='grasp_not_established' if done else None)
            return obs,reward,done,truncated,info
    env=DummyVecEnv([lambda:AcquisitionFailure(reaches_lift=False) for _ in range(2)])
    report,model=run_vector_training(env,tmp_path,force_feedback_contract(),timesteps=32,
        eval_episodes=2,checkpoint_every=16,ppo_kwargs=dict(n_steps=8,batch_size=8,n_epochs=1))
    assert report['admission']['allowed'] and model.num_timesteps==32
    assert report['evaluation']['grasp_success_rate']==0 and not report['passed']


def test_contact_gated_admission_still_rejects_physical_failure(tmp_path):
    from test_dh116_tabletop_vec_training import vector
    from go2_pvcnn.control.dh116.tabletop_vec_training import run_vector_training
    report,model=run_vector_training(vector(dict(fail_mode='admission',reaches_lift=False)),
        tmp_path,force_feedback_contract(),timesteps=32)
    assert not report['admission']['allowed'] and model is None


def test_signed_closure_and_total_envelope_reach_actual_joint_requests():
    from test_dh116_tabletop_adaptive import AdaptiveBackend,make_env
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    direction=np.tile(np.array([0,-.2,np.sqrt(.96),0,0,0]),(2,1))
    env=make_env(AdaptiveBackend(),force_direction=direction)
    env.reset();env.force_controller.offset[:]=.08
    action=np.zeros(20);action[-2:]=-1
    for _ in range(120):env.step(action)
    assert np.all(env.command[[1,7]]<env.reference[[1,7]])
    assert np.all(env.command[[2,8]]>env.reference[[2,8]])
    action[:12]=1.
    for _ in range(200):env.step(action)
    lo,hi=active_limits()
    assert np.all(env.command>=np.tile(lo,2)) and np.all(env.command<=np.tile(hi,2))
    assert np.max(np.abs(env.command-env.reference))<=.15+1e-9
    assert env.command[[2,8]]==pytest.approx(env.reference[[2,8]]+.15)


def test_readiness_interruption_before_dwell_completion_restarts_acquisition():
    task=TabletopTask(phase='close',grasp_mode='object_stability',force_feedback=True)
    for _ in range(19):advance(task,True)
    advance(task,False)
    assert task.phase=='close' and task.contact_time==0
    for _ in range(20):advance(task,True)
    assert task.phase=='hold'
    for _ in range(29):advance(task,True)
    advance(task,False)
    assert task.phase=='close' and task.contact_time==0


def test_legacy_adaptive_checkpoint_rejected_before_any_policy_load(tmp_path):
    import json
    from go2_pvcnn.control.dh116.tabletop_training import load_checkpoint
    checkpoint=tmp_path/'legacy.zip'
    checkpoint.with_suffix('.json').write_text(json.dumps(dict(contract=dict(
        observation_version='dh116-adaptive-grasp-v1',observation_size=125,
        action='20D_joint_palm_stiffness_v1'),timesteps=26176,updates=186)))
    with pytest.raises(ValueError,match='contract'):
        load_checkpoint(checkpoint,None,force_feedback_contract())


def test_pressure_regulation_warning_does_not_erase_stable_grasp_but_still_blocks_palms():
    env=force_env();env.reset()
    env.backend.forces[:]=[.1,0,0,.1]
    env.backend.mimic[:]=.0041
    env._measure()
    action=np.zeros(20);action[-2:]=-1
    initial_offset=env.offset
    for _ in range(40):env.step(action)
    assert env.task.phase=='lift'
    assert env.offset==initial_offset
    assert all(not call[2]['palm_motion_allowed'] for call in env.backend.calls[1:])
    assert not env.force_controller.offset.any()


def test_lift_motion_waits_for_reaction_force_and_pauses_on_increasing_slip():
    env=force_env();env.reset()
    env.task.phase='lift'
    action=np.zeros(20);action[-2:]=-1
    env.command=env.reference.copy();env.backend.q=env.reference.copy();env._measure()
    initial=env.offset
    env.step(action)
    assert env.offset==initial
    assert np.all(env.force_controller.offset>0)
    env.backend.forces[:]=[.1,0,0,.1];env._measure()
    env.step(action)
    assert env.offset>initial
    env.force_controller.last_slip=.002
    assert not env.force_controller.lift_allowed([.1,0,0,.1],slip_m=.0021,dt=1/480)
    assert env.force_controller.lift_allowed([.1,0,0,.1],slip_m=.002,dt=1/480)


def test_live_info_exposes_actual_load_separately_from_force_setpoint():
    env=force_env();env.reset()
    env.backend.forces[:]=[.06,0,0,.04]
    _,_,_,_,info=env.step(np.zeros(20))
    feedback=info['force_feedback']
    assert feedback['measured_hand_load_n']==pytest.approx([.06,.04])
    assert feedback['requested_force_n']==pytest.approx([.55,.55])
    assert feedback['measured_hand_load_n']!=feedback['requested_force_n']


def test_new_policy_starts_near_reachable_pressure_without_changing_other_actions():
    import torch
    from stable_baselines3 import PPO
    from go2_pvcnn.control.dh116.tabletop_policy import BoundedMeanPolicy
    from go2_pvcnn.control.dh116.tabletop_force_feedback import initialize_force_policy
    model=PPO(BoundedMeanPolicy,force_env(),n_steps=8,batch_size=8,device='cpu',seed=42)
    layer=model.policy.action_net[0]
    bias=layer.bias.detach().clone();weight=layer.weight.detach().clone()
    initialize_force_policy(model.policy)
    torch.testing.assert_close(layer.bias[:-2],bias[:-2])
    torch.testing.assert_close(layer.weight,weight)
    with torch.no_grad():
        mean=model.policy.get_distribution(torch.zeros((1,136))).distribution.mean
    assert mean[0,-2:].cpu().numpy()==pytest.approx([-.95,-.95],abs=1e-6)
    assert np.isfinite(mean.cpu().numpy()).all()


def test_environment_increasing_slip_pauses_every_physics_substep_and_boosts_force():
    env=force_env();env.reset();env.task.phase='lift'
    env.command=env.reference.copy();env.backend.q=env.reference.copy()
    env.backend.forces[:]=[.1,0,0,.1]
    env.grip_relative=np.array([0.,0.,-.02])
    env.backend.override['object_relative']=np.array([.003,0.,-.02])
    env._measure();env.force_controller.last_slip=.0029
    original=env.backend.advance
    def advance(*args,**kwargs):
        original(*args,**kwargs)
        env.backend.override['object_relative'][0]+=.0001
    env.backend.advance=advance
    initial=env.offset
    action=np.zeros(20);action[-2:]=-1
    env.step(action)
    assert env.offset==initial
    assert all(call[0]==initial for call in env.backend.calls)
    assert np.all(env.force_controller.boost>0)
