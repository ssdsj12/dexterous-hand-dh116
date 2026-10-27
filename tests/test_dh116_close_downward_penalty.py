import pytest
import torch
from test_dh116_tensor_task import template, measurement
from test_dh116_tensor_contact_window import loads
from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask


def task(n=3,device='cpu',**overrides):
    e=template();e.force_controller.parameters=LoadParameters(max_force_n=12.,force_rate_n_s=None)
    options=dict(pressure_relief=True,contact_reward_window='close_start',grasp_readiness='measured_load',
        vertical_force_reward=True,vertical_force_reward_mode='continuous',force_reward_start='downward_after_close',
        skip_hold=True,close_downward_penalty=True)
    options.update(overrides)
    return TensorTask(e,n,device,**options)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('hz',[60,480])
def test_boundary_sign_fixed_penalty_once_plus_continuous_and_reset(device,hz):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=task(device=device);t.begin_step(torch.zeros((3,20),device=device));m=loads(t,4.)
    dt=1/hz;force=torch.tensor([-1.,0.,1.],device=device)
    for _ in range(round(2.*hz)+1):t.consume(m,~t.done,dt,vertical_force_n=force)
    assert t.phase.tolist()==[1,3,3]
    assert t.failure[0].item()!=0
    assert t.terms['downward_force'][0].item()<0.
    assert t.terms['close_downward'].tolist()==[0.,0.,0.]
    assert torch.allclose(t.reward,sum(t.terms.values()))
    t.reset([0],measurement(t));assert t.close_downward_impulse[0].item()==0.
    assert t.observe().shape==(3,151)


def test_failed_nonfinite_and_inactive_samples_never_pay_event():
    t=task(4);t.begin_step(torch.zeros((4,20)));t.contact_time[:]=.2-1/480
    m=loads(t,4.);m['clearance'][1]=-.002
    t.consume(m,torch.tensor([True,True,False,True]),1/480,
              vertical_force_n=torch.tensor([float('nan'),-2.,-2.,-2.]))
    assert t.terms['close_downward'].tolist()==[0.,0.,0.,0.]
    assert t.phase.tolist()==[1,1,1,1] and torch.isfinite(t.reward).all()


def test_close_downward_impulse_marks_episode_failed_before_lift():
    t=task();t.begin_step(torch.zeros((3,20)));m=loads(t,4.)
    dt=1/480
    downward=torch.full((3,),-1.)
    for _ in range(961):
        t.consume(m,~t.done,dt,vertical_force_n=downward)
    assert t.failure[0].item()!=0
    assert t.phase[0].item()==1


def test_close_timeout_marks_episode_failed_at_four_seconds():
    t=task();t.begin_step(torch.zeros((3,20)));m=loads(t,4.)
    t.elapsed[:]=4.01
    t.consume(m,~t.done,1/480,vertical_force_n=torch.zeros(3))
    assert t.failure.tolist()==[9,9,9]
    assert t.phase.tolist()==[1,1,1]


def test_downward_reposition_abort_marks_episode_failed_before_lift():
    t=task(2);t.begin_step(torch.zeros((2,20)));m=loads(t,4.)
    from go2_pvcnn.control.dh116.tabletop_force_direction_reposition import ForceDirectionReposition
    t.palm_reposition=ForceDirectionReposition(torch.ones(2,dtype=torch.bool))
    t.force_direction_reposition=True
    t.palm_reposition.aborted[:]=True
    t.consume(m,~t.done,1/480,vertical_force_n=torch.ones(2))
    assert t.failure.tolist()==[9,9]
    assert t.phase.tolist()==[1,1]


@pytest.mark.parametrize('bad',[{'skip_hold':False},{'force_reward_start':'after_close'},{'vertical_force_reward':False,'vertical_force_reward_mode':'window','force_reward_start':'episode_start'}])
def test_flag_rejects_unsupported_modes(bad):
    with pytest.raises(ValueError,match='close_downward_penalty'):task(**bad)


def test_exact_one_way_event_contract_migration():
    from go2_pvcnn.control.dh116.tabletop_tensor_task import CLOSE_DOWNWARD_PENALTY_CONTRACT,measured_readiness_contract
    from go2_pvcnn.control.dh116.tabletop_vertical_force import vertical_reward_contract
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    v=vertical_reward_contract('continuous','downward_after_close',skip_hold=True)
    assert CLOSE_DOWNWARD_PENALTY_CONTRACT['impulse_limit_n_s']==2.
    assert CLOSE_DOWNWARD_PENALTY_CONTRACT['close_timeout_s']==4.
    old=dict(reward=v['version'],phase_reward_version=v['version'],vertical_force_reward=v,
        grasp_readiness=measured_readiness_contract(True),observation_size=151,num_envs=4096,
        observation_version='dh116-after-close-force-v3')
    new=dict(old,close_downward_penalty=CLOSE_DOWNWARD_PENALTY_CONTRACT)
    assert checkpoint_contract_changes(old,new)==['close_downward_impulse_reset_v3']
    assert checkpoint_contract_changes(new,new)==[]
    for a,b in [(new,old),(old,dict(new,num_envs=16)),(old,dict(new,close_downward_penalty=dict(CLOSE_DOWNWARD_PENALTY_CONTRACT,points=-30.)))]:
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(a,b)
