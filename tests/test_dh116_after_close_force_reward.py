import pytest
import torch
from test_dh116_tensor_task import template,measurement
from test_dh116_tensor_contact_window import loads
from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask


def task(n=1,device='cpu'):
    e=template();e.force_controller.parameters=LoadParameters(max_force_n=12.,force_rate_n_s=None)
    return TensorTask(e,n,device,contact_reward_window='close_start',grasp_readiness='measured_load',
        vertical_force_reward=True,vertical_force_reward_mode='continuous',force_reward_start='after_close')


@pytest.mark.parametrize('fz',[2.,-2.])
def test_only_after_close_completion_scores_force(fz):
    t=task();t.begin_step(torch.zeros((1,20)));m=loads(t,4.)
    # Close admission needs nonnegative normal force; signed scoring starts later.
    for _ in range(96):t.consume(m,~t.done,1/480,vertical_force_n=torch.full((1,),2.))
    assert t.phase.item()==2
    assert t.terms['contact'].item()==0 and t.terms['downward_force'].item()==0
    t.consume(m,~t.done,1/480,vertical_force_n=torch.tensor([fz]))
    assert t.terms['contact'].item()==pytest.approx(max(fz,0)*.5/480)
    assert t.terms['downward_force'].item()==pytest.approx(min(fz,0)*.5/480)
    assert t.observe().shape==(1,151) and t.observe()[0,-1]==1


def test_losing_grip_cannot_reopen_exemption_and_reset_is_selective():
    t=task(2);t.phase[0]=2;t.begin_step(torch.zeros((2,20)))
    t.consume(loads(t,0.),~t.done,.01,vertical_force_n=torch.tensor([-2.,-2.]))
    assert t.phase.tolist()==[1,1]
    assert t.terms['downward_force'].tolist()==pytest.approx([-.01,0.])
    t.consume(loads(t,0.),~t.done,.01,vertical_force_n=torch.tensor([-2.,-2.]))
    assert t.terms['downward_force'].tolist()==pytest.approx([-.02,0.])
    t.reset([0],measurement(t))
    assert not t.force_scoring_started.any()
    t.consume(loads(t,0.),~t.done,.01,vertical_force_n=torch.tensor([-2.,-2.]))
    assert t.terms['downward_force'].tolist()==[0.,0.]


def test_before_close_has_full_failure_protection_and_no_force_reward():
    t=task();t.begin_step(torch.zeros((1,20)));m=loads(t,4.);m['clearance'][:]=-.002
    t.consume(m,~t.done,.01,vertical_force_n=torch.tensor([-20.]))
    assert t.done.item() and t.terms['failure'].item()==-25
    assert t.terms['downward_force'].item()==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_after_close_terminal_downward_cost_once_and_gpu_observation():
    t=task(2,'cuda');t.phase[:]=2;t.begin_step(torch.zeros((2,20),device='cuda'))
    m=loads(t,4.);m['clearance'][0]=-.002
    force=torch.tensor([-4.,4.],device='cuda')
    t.consume(m,~t.done,.01,vertical_force_n=force)
    assert t.terms['downward_force'][0].item()==pytest.approx(-.02)
    t.consume(m,~t.done,.01,vertical_force_n=force)
    assert t.terms['downward_force'][0].item()==pytest.approx(-.02)
    assert t.observe().shape==(2,151) and t.observe().device.type=='cuda'


def test_reward_contract_migration_exact_and_one_way():
    from go2_pvcnn.control.dh116.tabletop_vertical_force import vertical_reward_contract
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    def contract(start):
        v=vertical_reward_contract('continuous',start)
        return dict(reward=v['version'],phase_reward_version=v['version'],vertical_force_reward=v,
          observation_version='dh116-free-grasp-contact-window-v2' if start=='episode_start' else 'dh116-after-close-force-v3',
          observation_size=150 if start=='episode_start' else 151,num_envs=4096)
    old,new=contract('episode_start'),contract('after_close')
    assert checkpoint_contract_changes(old,new)==['force_scoring_after_close_v3']
    for a,b in [(new,old),(old,dict(new,num_envs=16))]:
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(a,b)
