import importlib
import numpy as np
import pytest
import torch


def module():
    name='go2_pvcnn.control.dh116.tabletop_control_probe'
    assert importlib.util.find_spec(name) is not None, 'control probe module missing'
    return importlib.import_module(name)


def test_joint_inverse_preserves_asymmetric_bounds_and_rejects_outside():
    p=module();low=torch.tensor([-2.,0.]);center=torch.tensor([0.,1.]);high=torch.tensor([1.,4.])
    target=torch.tensor([-1.,2.5])
    a=p.joint_actions(target,low,center,high)
    torch.testing.assert_close(a,torch.tensor([-.5,.5]))
    with pytest.raises(ValueError):p.joint_actions(torch.tensor([-3.,2.]),low,center,high)


def test_force_target_roundtrip_and_bounds():
    p=module()
    for f in (.1,4.,8.,12.):assert .1+(p.force_action(f,12.)+1)*.5*11.9==pytest.approx(f)
    with pytest.raises(ValueError):p.force_action(13.,12.)


def test_contract_only_environment_count_may_differ():
    p=module();saved=dict(num_envs=4096,observation_size=156,physics_hz=480,minibatches=8)
    p.check_probe_contract(saved,dict(saved,num_envs=8))
    for change in ({'physics_hz':240},{'observation_size':151},{'minibatches':4}):
        with pytest.raises(ValueError):p.check_probe_contract(saved,dict(saved,num_envs=8,**change))


def test_force_direction_probe_allows_only_controller_contract_change():
    p=module()
    saved=dict(num_envs=4096,observation_size=156,physics_hz=480,
               palm_reposition={'version':'release_reposition_reclose_v1'})
    current=dict(saved,num_envs=8,
                 palm_reposition={'version':'force_direction_release_reposition_v1'},
                 force_direction_reposition=True)
    p.check_probe_contract(saved,current,allow_controller_change=True)
    with pytest.raises(ValueError):
        p.check_probe_contract(saved,dict(current,observation_size=151),allow_controller_change=True)


def test_dwell_does_not_join_separate_contacts_and_requires_stable():
    p=module()
    assert p.longest_dwell(np.array([True]*30+[False]+[True]*30),1/480)==pytest.approx(30/480)
    assert p.longest_dwell(np.ones(48,dtype=bool),1/480)==pytest.approx(.1)
    assert p.longest_dwell(np.zeros(0,dtype=bool),1/480)==0


def test_observer_sees_terminal_force_before_reset_and_cannot_count_frozen_samples():
    from test_dh116_tensor_env import World,template,TensorTask,TensorEnv
    t=TensorTask(template(),2,'cpu');w=World(t);env=TensorEnv(w,t)
    seen=[]
    class Observer:
        def before_step(self):pass
        def after_step(self,live,previous_phase):
            if live[0]:seen.append((w.tick,t.m['hand_object_loads'][0,0].item(),env.episodes[0].item()))
    env.control_probe=Observer()
    env.step(torch.zeros((2,20)))
    assert seen[-1]==(3,31.,0)
    assert len(seen)==3 and env.episodes[0]==1


def test_contract_rejects_missing_guard_field():
    p=module()
    with pytest.raises(ValueError):p.check_probe_contract({'num_envs':4096,'guard':True},{'num_envs':8})


def test_lift_only_threshold_preserves_joint_recovery_and_palm_pause():
    from test_dh116_close_downward_penalty import task
    from test_dh116_tensor_contact_window import loads
    t=task(2,grasp_shaping=True);t.phase[:]=3;t.m=loads(t,4.)
    t.m['mimic_errors'][:,0]=.006
    t.command.copy_(t.m['q'])
    t.diagnostic_lift_mimic_threshold=torch.tensor([.004,.008])
    before=t.offset.clone();a=torch.zeros((2,20),dtype=torch.float64);a[:,18:]=module().force_action(4.,12.)
    allowed=t.control(a,~t.done,1/480)
    assert t.offset[0]==before[0] and t.offset[1]>before[1]
    assert t.paused.all() and not allowed.any()
    torch.testing.assert_close(t.command[0],t.command[1])
    torch.testing.assert_close(t.f_offset[0],t.f_offset[1])
    assert t.lift_diagnostic_masks[:,2].tolist()==[True,False]


def test_candidate_lift_still_respects_pressure_and_larger_mimic_error():
    from test_dh116_close_downward_penalty import task
    from test_dh116_tensor_contact_window import loads
    t=task(2,grasp_shaping=True);t.phase[:]=3;t.m=loads(t,12.)
    t.m['mimic_errors'][:,0]=torch.tensor([.006,.009]);t.command.copy_(t.m['q'])
    t.diagnostic_lift_mimic_threshold=torch.tensor([.008,.008]);before=t.offset.clone()
    a=torch.zeros((2,20),dtype=torch.float64);a[:,18:]=module().force_action(4.,12.)
    t.control(a,~t.done,1/480)
    torch.testing.assert_close(t.offset,before)
    assert t.pressure_limited.all()


def test_thumb_position_lead_candidate_keeps_other_joints_and_lift_gate():
    from test_dh116_close_downward_penalty import task
    from test_dh116_tensor_contact_window import loads
    t=task(2,grasp_shaping=True);t.m=loads(t,4.);t.phase[:]=3
    t.command.copy_(t.m['q']);t.command[:,[1,7]]+=.008
    before=t.command.clone();offset=t.offset.clone()
    t.diagnostic_thumb_position_lead=torch.tensor([0.,.008],dtype=torch.float64)
    a=torch.ones((2,20),dtype=torch.float64);a[:,18:]=module().force_action(4.,12.)
    t.control(a,~t.done,1/480)
    assert (t.command[0,[1,7]]>before[0,[1,7]]).all()
    torch.testing.assert_close(t.command[1,[1,7]],before[1,[1,7]])
    other=[i for i in range(12) if i not in (1,7)]
    torch.testing.assert_close(t.command[0,other],t.command[1,other])
    torch.testing.assert_close(t.offset-offset,(t.offset-offset)[0].expand(2))
