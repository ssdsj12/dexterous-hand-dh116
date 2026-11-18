import math

import pytest
import torch


def runtime():
    from go2_pvcnn.control.dh116 import tabletop_surface_mpc_runtime
    return tabletop_surface_mpc_runtime


def test_intrinsic_rotation_velocity_matches_actual_rotation_composition():
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import compose_rotation
    seed=torch.tensor([[.9238795325,0.,0.,.3826834324]],dtype=torch.float64)
    angles=torch.tensor([[.25,-.3,.15]],dtype=torch.float64)
    rates=torch.tensor([[.1,-.2,.3]],dtype=torch.float64)
    point=torch.tensor([[.02,.01,-.03]],dtype=torch.float64)
    from go2_pvcnn.control.dh116.tabletop_effective_pad_runtime import _rotation
    def matrix(a):
        pose=torch.cat((torch.zeros(1,3,dtype=torch.float64),compose_rotation(seed,a)),-1)
        return _rotation(pose)
    step=1e-6
    derivative=((matrix(angles+step*rates)-matrix(angles-step*rates))/(2*step))@point[...,None]
    world_point=(matrix(angles)@point[...,None]).squeeze(-1)
    omega=runtime().intrinsic_world_velocity(seed,angles,rates)
    torch.testing.assert_close(torch.linalg.cross(omega,world_point),derivative.squeeze(-1),atol=1e-10,rtol=1e-7)


def test_nominal_horizon_integrates_from_applied_command_and_preserves_limits():
    command=torch.full((2,2,6),.2,dtype=torch.float64)
    action=torch.ones(2,24,dtype=torch.float64)
    lower=torch.zeros(2,6,dtype=torch.float64);upper=torch.full((2,6),.207,dtype=torch.float64)
    result=runtime().nominal_joint_trajectory(command,action,4,lower,upper)
    assert result.shape==(2,4,2,6)
    torch.testing.assert_close(result[:,0],command+.005)
    assert result.max()<=.207
    assert (result[:,1:]-result[:,:-1]).abs().max()<=.005000001


def test_corrected_joint_target_decodes_using_command_and_keeps_wrist_intent():
    action=torch.linspace(-.8,.8,48,dtype=torch.float64).reshape(2,24)
    command=torch.full((2,2,6),.2,dtype=torch.float64)
    target=command+.001
    result=runtime().action_from_joint_target(action,command,target)
    torch.testing.assert_close(result[:,:12],torch.full((2,12),.2,dtype=torch.float64))
    torch.testing.assert_close(result[:,12:],action[:,12:])


def native_env(n=1):
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,MIMIC
    dtype=torch.float64
    tensor=lambda value:torch.as_tensor(value,dtype=dtype)
    joints=list(ACTIVE)+[row[0] for row in MIMIC]
    names=['root','finger14_link','finger23_link','finger33_link','finger43_link','finger53_link']
    poses=torch.zeros(n,6,7,dtype=dtype);poses[...,3]=1
    jac=torch.zeros(n,5,6,len(joints),dtype=dtype)
    jac[:,:,0,0]=.01
    hand=SimpleNamespace(joint_names=joints,body_names=names,
        data=SimpleNamespace(body_link_pose_w=poses,body_com_pos_w=poses[...,:3]),
        root_physx_view=SimpleNamespace(get_jacobians=lambda:jac))
    roots=torch.zeros(n,2,7,dtype=dtype);roots[...,3]=1
    loads=torch.tensor([[[1.,.5,.5,0.,0.],[1.,.5,.5,0.,0.]]],dtype=dtype).repeat(n,1,1)
    vectors=torch.zeros(n,2,5,3,dtype=dtype)
    vectors[:,:,0,0]=1.;vectors[:,:,1:3,0]=-.5
    q=torch.full((n,12),.2,dtype=dtype)+torch.arange(n,dtype=dtype)[:,None]*.01
    m=dict(q=q,bottom=torch.full((n,),.39,dtype=dtype),
        object_relative=torch.zeros(n,3,dtype=dtype),wrist_pose=roots,
        object_quat=tensor([[1.,0.,0.,0.]]).repeat(n,1),effective_pad_valid=loads>0,
        effective_pad_loads_n=loads,effective_pad_vectors_w=vectors,
        effective_pad_points_w=torch.zeros(n,2,5,3,dtype=dtype))
    task=SimpleNamespace(m=m,command=m['q'].clone(),low=torch.zeros(12,dtype=dtype),
        high=torch.ones(12,dtype=dtype),surface_reference_bottom_m=tensor(.36))
    world=SimpleNamespace(n=n,device='cpu',t=tensor,hands=[hand,hand],roots=roots,
        root_seed=roots.clone(),wrist_angles=torch.zeros(n,2,3,dtype=dtype),
        palms=torch.zeros(n,2,3,dtype=dtype))
    return SimpleNamespace(task=task,world=world)


def record_plans(controller):
    calls=[]
    original=controller.planner.plan
    def record(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    controller.planner.plan=record
    return calls


def test_adapter_exercises_real_optimizer_with_five_digit_native_shaped_readback():
    controller=runtime().NativeSurfaceMPC(native_env())
    dtype=torch.float64
    action=torch.zeros(1,24,dtype=dtype)
    result=controller.refine(action)
    torch.testing.assert_close(result,action)
    assert controller.stats['active_env_steps']==1
    assert 'last_model_error' not in controller.stats
    assert controller.last_prediction.shape==(1,2,6)


@pytest.mark.parametrize('case',['translation_norm','rotation_norm','translation_limit','rotation_limit'])
def test_each_wrist_forecast_step_matches_native_command_integration(case):
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import _limited_step,compose_rotation,CONTRACT
    from go2_pvcnn.control.dh116.tabletop_effective_pad_runtime import _rotation
    env=native_env();w=env.world
    w.root_seed[:,:,3:]=w.t([.9238795325,0.,0.,.3826834324])
    w.wrist_angles[:]=w.t([.2,-.3,.4])
    w.palms[:]=w.t([.01,-.01,.02])
    # Measured roots differ from commanded palms; only command increments apply.
    w.roots[...,:3]=w.t([.1,.2,.3])
    env.task.m['effective_pad_points_w'][:]=w.t([.12,.21,.33])
    action=torch.zeros(1,24,dtype=torch.float32)
    if case.startswith('translation'):action[:,12:18]=1.
    else:action[:,18:24]=1.
    if case=='translation_limit':w.palms[:]=w.t(CONTRACT['palm_upper_m'])-.0001
    if case=='rotation_limit':w.wrist_angles[:]=math.pi/2-.002
    before={key:getattr(w,key).clone() for key in ('palms','wrist_angles','roots')}
    controller=runtime().NativeSurfaceMPC(env);calls=record_plans(controller)
    result=controller.refine(action)
    assert len(calls)==1 and controller.stats['active_env_steps']==1
    inputs=calls[0]
    assert inputs['q'].dtype==inputs['jacobian'].dtype==inputs['wrist_motion'].dtype==torch.float64
    palms=w.palms.clone();angles=w.wrist_angles.clone()
    def rotation(angles):
        quat=compose_rotation(w.root_seed[:,:,3:],angles)
        return _rotation(torch.cat((torch.zeros_like(palms),quat),-1))
    initial=rotation(angles);lever=inputs['points']-w.roots[:,:,None,:3]
    expected=[];clipped=action.to(palms).clamp(-1,1)
    for _ in range(4):
        palms=(palms+_limited_step(clipped[:,12:18].reshape(1,2,3),.01,1/60)).maximum(
            w.t(CONTRACT['palm_lower_m'])).minimum(w.t(CONTRACT['palm_upper_m']))
        angles=(angles+_limited_step(clipped[:,18:24].reshape(1,2,3),.5,1/60)).clamp(-math.pi/2,math.pi/2)
        relative=rotation(angles)@initial.transpose(-1,-2)
        expected.append((palms-w.palms)[:,:,None]+(relative[:,:,None]@lever[...,None]).squeeze(-1)-lever)
    torch.testing.assert_close(inputs['wrist_motion'],torch.stack(expected,1),atol=1e-13,rtol=0)
    torch.testing.assert_close(result[:,12:],action[:,12:],atol=0,rtol=0)
    for key,value in before.items():torch.testing.assert_close(getattr(w,key),value,atol=0,rtol=0)


def test_all_inactive_early_return_clears_lost_contact_anchors():
    env=native_env();controller=runtime().NativeSurfaceMPC(env)
    action=torch.zeros(1,24,dtype=torch.float64)
    controller.refine(action)
    assert controller.anchored.any()
    env.task.m['effective_pad_loads_n'].zero_()
    env.task.m['effective_pad_vectors_w'].zero_()
    env.task.m['effective_pad_valid'].zero_()
    torch.testing.assert_close(controller.refine(action),action,atol=0,rtol=0)
    assert not controller.anchored.any()
    assert not controller.pending.any()


def test_recontact_recaptures_material_points_instead_of_chasing_old_anchor():
    env=native_env();controller=runtime().NativeSurfaceMPC(env)
    action=torch.zeros(1,24,dtype=torch.float64)
    controller.refine(action)
    loads=env.task.m['effective_pad_loads_n'].clone()
    vectors=env.task.m['effective_pad_vectors_w'].clone()
    valid=env.task.m['effective_pad_valid'].clone()
    for key in ('effective_pad_loads_n','effective_pad_vectors_w','effective_pad_valid'):env.task.m[key].zero_()
    controller.refine(action)
    env.task.m['effective_pad_loads_n'].copy_(loads)
    env.task.m['effective_pad_vectors_w'].copy_(vectors)
    env.task.m['effective_pad_valid'].copy_(valid)
    env.world.hands[0].data.body_link_pose_w[:,1:,0]=.002
    env.task.m['effective_pad_points_w'][...,0]=.002
    actual=controller.refine(action)
    fresh=runtime().NativeSurfaceMPC(env).refine(action)
    torch.testing.assert_close(actual,fresh,atol=0,rtol=0)


@pytest.mark.parametrize('bad',['vectors','loads','bottom','reference'])
def test_invalid_activation_inputs_fallback_only_the_bad_environment(bad):
    env=native_env(3);controller=runtime().NativeSurfaceMPC(env)
    if bad=='vectors':env.task.m['effective_pad_vectors_w'][1,0,0,0]=float('nan')
    elif bad=='loads':env.task.m['effective_pad_loads_n'][1,0,0]=-1.
    elif bad=='bottom':env.task.m['bottom'][1]=float('nan')
    else:env.task.surface_reference_bottom_m=env.world.t([.36,float('nan'),.36])
    calls=record_plans(controller);action=torch.zeros(3,24,dtype=torch.float32)
    result=controller.refine(action)
    assert len(calls)==1
    torch.testing.assert_close(calls[0]['q'],env.task.m['q'][[0,2]].reshape(2,2,6))
    assert controller.pending.tolist()==[True,False,True]
    assert controller.stats['model_fallback_env_steps']==1
    torch.testing.assert_close(result[1],action[1],atol=0,rtol=0)


def test_inactive_nonfinite_state_never_enters_core_or_disables_healthy_rows():
    env=native_env(3);controller=runtime().NativeSurfaceMPC(env)
    for key in ('effective_pad_loads_n','effective_pad_vectors_w','effective_pad_valid'):env.task.m[key][1]=0
    env.task.m['q'][1,0]=float('nan')
    calls=record_plans(controller);action=torch.zeros(3,24,dtype=torch.float32)
    controller.refine(action)
    assert len(calls)==1 and torch.isfinite(calls[0]['q']).all()
    assert calls[0]['q'].shape==(2,2,6)
    assert controller.pending.tolist()==[True,False,True]
    assert controller.stats['model_fallback_env_steps']==0


@pytest.mark.parametrize('bad',['q','jacobian','command','angles','palms','object_quat'])
def test_only_finite_feasible_active_rows_are_sent_to_real_core(bad):
    env=native_env(3);controller=runtime().NativeSurfaceMPC(env)
    if bad=='q':env.task.m['q'][1,0]=float('nan')
    elif bad=='jacobian':env.world.hands[0].root_physx_view.get_jacobians()[1,0,0,0]=float('nan')
    elif bad=='command':env.task.command[1]=3.
    elif bad=='angles':env.world.wrist_angles[1,0,0]=float('nan')
    elif bad=='palms':env.world.palms[1,0,0]=float('nan')
    else:env.task.m['object_quat'][1]=0.
    calls=record_plans(controller);action=torch.zeros(3,24,dtype=torch.float32)
    result=controller.refine(action)
    assert len(calls)==1
    assert calls[0]['q'].shape==(2,2,6)
    assert controller.pending.tolist()==[True,False,True]
    assert controller.stats['model_fallback_env_steps']==1
    torch.testing.assert_close(result[1],action[1],atol=0,rtol=0)


def test_unexpected_planner_exception_is_isolated_and_does_not_erase_good_anchors():
    env=native_env(3);controller=runtime().NativeSurfaceMPC(env)
    action=torch.zeros(3,24,dtype=torch.float64)
    controller.refine(action)
    saved=controller.local_rod[[0,2]].clone()
    original=controller.planner.plan
    def fail_one(**kwargs):
        if (kwargs['q'][:,0,0]==env.task.m['q'][1,0]).any():raise RuntimeError('row 1 model failure')
        return original(**kwargs)
    controller.planner.plan=fail_one
    result=controller.refine(action)
    assert controller.pending.tolist()==[True,False,True]
    assert controller.stats['model_fallback_env_steps']==1
    torch.testing.assert_close(controller.local_rod[[0,2]],saved,atol=0,rtol=0)
    torch.testing.assert_close(result[1],action[1],atol=0,rtol=0)


def test_subset_0_2_predictions_scatter_and_reset_remains_local():
    env=native_env(3);controller=runtime().NativeSurfaceMPC(env)
    for key in ('effective_pad_loads_n','effective_pad_vectors_w','effective_pad_valid'):env.task.m[key][1]=0
    controller.refine(torch.zeros(3,24,dtype=torch.float64))
    assert controller.last_prediction.shape==(3,2,6)
    torch.testing.assert_close(controller.last_prediction[[0,2]],env.task.m['q'][[0,2]].reshape(2,2,6))
    env.task.m['q'][0]+= .006
    controller.observe_result(torch.tensor([False,True,False]))
    assert controller.stats['prediction_checks']==2
    assert controller.bad_predictions.tolist()==[1,0,0]
    assert controller.pending.tolist()==[True,False,True]
    saved=controller.last_prediction[2].clone()
    controller.reset(torch.tensor([0]))
    assert controller.pending.tolist()==[False,False,True]
    assert not controller.anchored[0].any() and controller.anchored[2].any()
    torch.testing.assert_close(controller.last_prediction[2],saved,atol=0,rtol=0)


@pytest.mark.parametrize('case',['below_air','no_opposition','prediction_disabled'])
def test_activation_only_selects_correction_and_otherwise_passes_full_nominal(case):
    env=native_env();controller=runtime().NativeSurfaceMPC(env)
    controller.anchored[:]=True
    if case=='below_air':env.task.m['bottom'][:]=.36299
    elif case=='no_opposition':env.task.m['effective_pad_vectors_w'][...,0]=1.
    else:controller.bad_predictions[:]=3
    calls=record_plans(controller)
    action=torch.linspace(-.8,.8,24,dtype=torch.float32).reshape(1,24)
    torch.testing.assert_close(controller.refine(action),action,atol=0,rtol=0)
    assert not calls and not controller.pending.any()
    assert not controller.anchored.any()
    assert controller.stats['model_fallback_env_steps']==0
    assert controller.stats['model']=='local_pad_kinematics_not_object_dynamics'
    assert controller.stats['force_prediction'] is False


def test_subset_commands_and_predictions_scatter_to_original_environment_ids():
    env=native_env(3);controller=runtime().NativeSurfaceMPC(env)
    env.task.m['bottom'][1]=.36
    action=torch.full((3,24),.25,dtype=torch.float32)
    action[2,:12]=-.4
    original=controller.planner.plan;solved=[]
    def record_result(**kwargs):
        target,diag=original(**kwargs)
        solved.append((target,diag))
        return target,diag
    controller.planner.plan=record_result
    result=controller.refine(action)
    assert len(solved)==1 and solved[0][0].shape==(2,2,6)
    decoded=env.task.command[[0,2]].reshape(2,2,6)+result[[0,2],:12].double().reshape(2,2,6)*(.3/60)
    torch.testing.assert_close(decoded,solved[0][0],atol=1e-10,rtol=0)
    torch.testing.assert_close(controller.last_prediction[[0,2]],solved[0][1]['predicted_joints'][:,0])
    torch.testing.assert_close(result[1],action[1],atol=0,rtol=0)
    torch.testing.assert_close(result[:,12:],action[:,12:],atol=0,rtol=0)
