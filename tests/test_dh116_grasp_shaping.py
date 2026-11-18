import math
import pytest
import torch


def make(n=1,device='cpu'):
    from go2_pvcnn.control.dh116.tabletop_grasp_shaping import GraspShaping
    return GraspShaping(n,device)


def update(s,force=-1.,bottom=.3,dt=1/480,eligible=None,lifting=True,stable=True):
    n=len(s.force_ratio);device=s.force_ratio.device
    vec=lambda x:torch.full((n,),x,device=device,dtype=torch.float64)
    yes=torch.ones(n,device=device,dtype=torch.bool)
    return s.update(vec(force),vec(1.),vec(bottom),yes if lifting else ~yes,
                    yes if stable else ~yes,yes if eligible is None else eligible,dt)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_direction_progress_and_terminal_discounted_telescope(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    s=make(device=device);live=torch.ones(1,device=device,dtype=torch.bool);done=~live
    rewards=[]
    for force in [-2.,-1.,-.5,1.]:
        s.begin(live);update(s,force=force,dt=1/60);rewards.append(s.end(done).item())
    assert rewards[0]<0 and rewards[2]>0 and rewards[3]>0
    s.begin(live);terminal=s.end(live).item();rewards.append(terminal)
    assert sum(.995**i*r for i,r in enumerate(rewards))==pytest.approx(0.,abs=1e-12)
    assert not s.end(live).any()  # End is not a second shaping payment.
    # Splitting rollout collection must not reset the running potential.
    s=make(device=device);s.force_ratio[:]=-2
    s.begin(live);update(s,force=-1.,dt=1/60);assert s.end(done).item()>0
    s.begin(live);update(s,force=-3.,dt=1/60);assert s.end(done).item()<0


@pytest.mark.parametrize('hz',[60,480])
def test_filter_and_milestone_time_scaling(hz):
    s=make();last=None;total=torch.zeros(1,dtype=torch.float64)
    for i in range(hz//5):
        last=update(s,force=1.,bottom=.311,dt=1/hz);total+=last
    assert s.force_ratio.item()==pytest.approx(1-math.exp(-.2/.05))
    assert total.item()==5. and s.paid.all()
    for _ in range(hz):assert not update(s,force=1.,bottom=.311,dt=1/hz).any()


def test_milestone_stability_dwell_and_reset():
    s=make(2)
    for _ in range(47):assert not update(s,bottom=.304).any()
    update(s,bottom=.304,stable=False)
    assert not s.dwell.any()
    for _ in range(47):assert not update(s,bottom=.304).any()
    assert update(s,bottom=.304).tolist()==[2.,2.]
    for _ in range(47):assert not update(s,bottom=.311).any()
    assert update(s,bottom=.311).tolist()==[3.,3.]
    s.reset(torch.tensor([0]))
    assert s.paid.tolist()==[[False,False],[True,True]]
    assert s.observe().shape==(2,5)
    for _ in range(47):assert not update(s,bottom=.311).any()
    assert update(s,bottom=.311).tolist()==[5.,0.]


def test_inactive_invalid_samples_and_close_cannot_earn_milestone():
    s=make(2);mask=torch.tensor([True,False]);s.begin(mask)
    update(s,force=float('nan'),bottom=.311,eligible=torch.zeros(2,dtype=torch.bool))
    assert torch.isfinite(s.force_ratio).all() and not s.dwell.any()
    for _ in range(48):assert not update(s,bottom=.311,eligible=mask,lifting=False).any()
    assert not s.paid.any() and s.force_ratio[1]==0
    r=s.end(torch.tensor([True,False]));assert not r.any()


def tensor_task(n=2,device='cpu'):
    from test_dh116_tensor_task import template
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
    e=template();e.force_controller.parameters=LoadParameters(max_force_n=12.,force_rate_n_s=None)
    return TensorTask(e,n,device,pressure_relief=True,contact_reward_window='close_start',
        grasp_readiness='measured_load',vertical_force_reward=True,vertical_force_reward_mode='continuous',
        force_reward_start='downward_after_close',skip_hold=True,close_downward_penalty=True,grasp_shaping=True)


def test_task_control_boundary_terminal_and_selected_reset():
    from test_dh116_tensor_task import measurement
    from test_dh116_tensor_contact_window import loads
    t=tensor_task();m=loads(t,4.);t.begin_step(torch.zeros((2,20)))
    for _ in range(8):t.consume(m,~t.done,1/480,vertical_force_n=torch.tensor([-2.,2.]),object_weight_n=torch.ones(2))
    obs,r,d,extra=t.end_step()
    assert obs.shape==(2,156) and extra['terms']['force_progress'][0]<0 and extra['terms']['force_progress'][1]>0
    before=t.shaping.force_ratio.clone();t.begin_step(torch.zeros((2,20)))
    m['clearance'][0]=-.002
    for _ in range(8):t.consume(m,~t.done,1/480,vertical_force_n=torch.tensor([-2.,2.]),object_weight_n=torch.ones(2))
    _,r,d,extra=t.end_step()
    assert d.tolist()==[True,False]
    assert extra['terms']['force_progress'][0].item()==pytest.approx(-2*math.tanh(before[0]))
    assert torch.allclose(r,sum(extra['terms'].values()))
    t.reset([0],measurement(t));assert t.shaping.force_ratio[0]==0 and t.shaping.force_ratio[1]>0


def test_actual_lift_milestone_requires_phase_load_and_stability():
    from test_dh116_tensor_contact_window import loads
    t=tensor_task();t.phase[:]=3;t.begin_step(torch.zeros((2,20)));m=loads(t,4.)
    m['bottom'][:]=.311;m['object_velocity'][1,0]=.04
    for _ in range(48):t.consume(m,~t.done,1/480,vertical_force_n=torch.ones(2),object_weight_n=torch.ones(2))
    _,_,_,extra=t.end_step();assert extra['terms']['lift_milestone'].tolist()==[5.,0.]
    assert not t.success.any()


def test_exact_migration_and_zero_input_extension_preserve_policy_and_adam():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes,extend_checkpoint_observations,make_algorithm,observations
    from go2_pvcnn.control.dh116.tabletop_tensor_task import CLOSE_DOWNWARD_PENALTY_CONTRACT,measured_readiness_contract
    from go2_pvcnn.control.dh116.tabletop_vertical_force import vertical_reward_contract
    from go2_pvcnn.control.dh116.tabletop_grasp_shaping import CONTRACT
    v=vertical_reward_contract('continuous','downward_after_close',skip_hold=True)
    old=dict(reward=v['version'],phase_reward_version=v['version'],vertical_force_reward=v,
        grasp_readiness=measured_readiness_contract(True),close_downward_penalty=CLOSE_DOWNWARD_PENALTY_CONTRACT,
        observation_size=151,observation_version='dh116-after-close-force-v3',num_envs=4096)
    new=dict(old,grasp_shaping=CONTRACT,observation_size=156,observation_version='dh116-force-progress-milestones-v1')
    assert checkpoint_contract_changes(old,new)==['force_progress_lift_milestones_v1']
    for a,b in [(new,old),(old,dict(new,num_envs=16)),(old,dict(new,grasp_shaping=dict(CONTRACT,gamma=.99)))]:
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(a,b)
    before=make_algorithm(observations(torch.zeros(2,151)));after=make_algorithm(observations(torch.zeros(2,156)))
    loss=sum(p.square().sum() for p in before.policy.parameters());loss.backward();before.optimizer.step()
    saved={'policy':before.policy.state_dict(),'optimizer':before.optimizer.state_dict()}
    p,o=extend_checkpoint_observations(saved,after)
    for name,value in saved['policy'].items():
        if name in ('actor.0.weight','critic.0.weight'):
            assert torch.equal(p[name][:,:151],value) and not p[name][:,151:].any()
        else:assert torch.equal(p[name],value)
    for i,state in saved['optimizer']['state'].items():
        for key,value in state.items():
            actual=o['state'][i][key]
            if torch.is_tensor(value) and value.shape==(128,151):
                assert torch.equal(actual[:,:151],value) and not actual[:,151:].any()
            elif torch.is_tensor(value):assert torch.equal(actual,value)


def test_lift_gate_telemetry_matches_existing_control_blocks():
    from test_dh116_tensor_contact_window import loads
    from go2_pvcnn.control.dh116.tabletop_grasp_shaping import LIFT_GATE_NAMES,lift_gate_summary
    t=tensor_task(6);t.m=loads(t,4.);t.phase[:]=3;t.command.copy_(t.m['q'])
    t.m['mimic_errors'][1,0]=.005;t.command[2,0]+=.1
    for key in ('hand_object_loads','object_link_loads','grouped_object_loads'):t.m[key][3]=0.
    # At the 2 N minimum target, pressure relief requires more than 2.52 N/hand.
    for key in ('hand_object_loads','object_link_loads','grouped_object_loads'):t.m[key][5]*=1.5
    t.slip[4]=.003;t.f_last_slip[4]=0.
    a=torch.zeros((6,20),dtype=torch.float64);a[:,18:]=1.;a[5,18:]=-1.
    before=t.offset.clone();t.control(a,~t.done,1/480)
    assert (t.offset>before).tolist()==[True,False,False,False,False,False]
    masks=t.lift_diagnostic_masks
    for i,name in enumerate(('command_advanced','mimic_blocked','tracking_blocked','load_blocked','slip_blocked','pressure_blocked')):
        assert masks[i,LIFT_GATE_NAMES.index(name)]
    row=lift_gate_summary(masks.sum(0))
    assert row['lift_substeps']==6 and row['command_advanced_fraction']==pytest.approx(1/6)
    assert lift_gate_summary(torch.zeros(len(LIFT_GATE_NAMES)))['command_advanced_fraction'] is None
