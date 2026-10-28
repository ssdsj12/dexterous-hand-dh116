import pytest
import torch
from test_dh116_tensor_task import template,measurement
from test_dh116_tensor_contact_window import loads
from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask


def task(n=1,device='cpu'):
    e=template();e.force_controller.parameters=LoadParameters(max_force_n=12.,force_rate_n_s=None)
    return TensorTask(e,n,device,contact_reward_window='close_start',grasp_readiness='measured_load',
        vertical_force_reward=True,vertical_force_reward_mode='continuous',force_reward_start='downward_after_close')


def test_close_rewards_upward_but_exempts_downward_then_hold_scores_both():
    t=task(2);t.begin_step(torch.zeros((2,20)));m=loads(t,4.)
    force=torch.tensor([2.,-2.])
    for _ in range(95):t.consume(m,~t.done,1/480,vertical_force_n=force)
    assert t.phase.tolist()==[1,1] and not t.terms['downward_force'].any()
    # Complete close with a nonnegative current sample before testing hold costs.
    t.consume(m,~t.done,1/480,vertical_force_n=force.clamp_min(0.))
    assert t.phase.tolist()==[2,2]
    assert t.terms['contact'].tolist()==pytest.approx([.2,0.])
    assert t.terms['downward_force'].tolist()==[0.,0.]
    t.consume(m,~t.done,1/480,vertical_force_n=force)
    assert t.terms['contact'].tolist()==pytest.approx([.2+1/480,0.])
    assert t.terms['downward_force'].tolist()==pytest.approx([0.,-1/480])
    # Losing the grip cannot re-enable downward exemption.
    t.consume(loads(t,0.),~t.done,.01,vertical_force_n=force)
    assert t.phase.tolist()==[1,1] and t.force_scoring_started.all()
    assert t.terms['downward_force'][1].item()==pytest.approx(-1/480-.01)
    t.reset([1],measurement(t))
    assert t.force_scoring_started.tolist()==[True,False]
    assert t.observe().shape==(2,151)


@pytest.mark.parametrize('hz',[60,480])
def test_late_close_upward_reward_has_no_one_second_or_five_point_cap(hz):
    t=task();t.begin_step(torch.zeros((1,20)));m=loads(t,.4)
    for _ in range(2*hz):t.consume(m,~t.done,1/hz,vertical_force_n=torch.tensor([10.]))
    assert t.phase.item()==1 and not t.force_scoring_started.item()
    assert t.terms['contact'].item()==pytest.approx(10.)
    assert t.contact_reward_paid.item()==pytest.approx(10.)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_close_invalid_failed_and_inactive_samples_do_not_earn_upward_reward(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=task(4,device);t.begin_step(torch.zeros((4,20),device=device));m=loads(t,4.)
    m['clearance'][1]=-.002
    force=torch.tensor([float('nan'),10.,2.,10.],device=device)
    t.consume(m,torch.tensor([True,True,True,False],device=device),.01,vertical_force_n=force)
    assert torch.isfinite(t.reward).all()
    assert t.terms['contact'].tolist()==pytest.approx([0.,0.,.01,0.])
    assert t.terms['failure'].tolist()==[-25.,-25.,0.,0.]
    assert not t.terms['downward_force'].any()


def test_exact_one_way_reward_change_without_observation_resize():
    from go2_pvcnn.control.dh116.tabletop_vertical_force import vertical_reward_contract
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    def contract(start):
        v=vertical_reward_contract('continuous',start)
        return dict(reward=v['version'],phase_reward_version=v['version'],vertical_force_reward=v,
                    observation_size=151,observation_version='dh116-after-close-force-v3',num_envs=4096)
    old,new=contract('after_close'),contract('downward_after_close')
    assert checkpoint_contract_changes(old,new)==['close_upward_reward_v4']
    assert checkpoint_contract_changes(new,new)==[]
    for a,b in [(new,old),(old,dict(new,num_envs=16))]:
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(a,b)
