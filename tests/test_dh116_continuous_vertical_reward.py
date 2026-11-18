import pytest
import torch
from test_dh116_tensor_task import template,measurement
from test_dh116_tensor_contact_window import loads
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask


def task(n=1,device='cpu'):
    return TensorTask(template(),n,device,contact_reward_window='close_start',
                      vertical_force_reward=True,vertical_force_reward_mode='continuous')


@pytest.mark.parametrize('hz',[60,480])
@pytest.mark.parametrize('force,up,down',[(2.,2.,0.),(20.,10.,0.),(-4.,0.,-4.)])
def test_continuous_signed_reward_after_window_and_above_old_budget(hz,force,up,down):
    t=task();t.phase[:]=3;t.begin_step(torch.zeros((1,20)));m=loads(t,10.)
    for _ in range(2*hz):t.consume(m,~t.done,1/hz,vertical_force_n=torch.tensor([force]))
    assert t.terms['contact'].item()==pytest.approx(up)
    assert t.contact_reward_paid.item()==pytest.approx(up)
    assert t.terms['downward_force'].item()==pytest.approx(down)


def test_reward_starts_after_late_contact_without_restarting_or_needing_window():
    t=task();t.phase[:]=3;t.begin_step(torch.zeros((1,20)))
    for _ in range(120):t.consume(loads(t,0.),~t.done,1/60,vertical_force_n=torch.zeros(1))
    assert t.contact_reward_clock.item()==1.
    t.consume(loads(t,10.),~t.done,.01,vertical_force_n=torch.tensor([10.]))
    assert t.terms['contact'].item()==pytest.approx(.05)
    t.reset([0],measurement(t))
    assert t.contact_reward_paid.item()==0.


def test_continuous_rejects_nan_and_does_not_pay_failure_or_inactive_rows():
    t=task(4);t.phase[:]=3;t.begin_step(torch.zeros((4,20)));m=loads(t,10.)
    m['clearance'][1]=-.002
    t.consume(m,torch.tensor([True,True,True,False]),.01,
              vertical_force_n=torch.tensor([float('nan'),10.,10.,100.]))
    assert torch.isfinite(t.reward).all()
    assert t.terms['contact'].tolist()==pytest.approx([0.,0.,.05,0.])
    assert t.terms['failure'].tolist()==[-25.,-25.,0.,0.]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_continues_after_old_window_and_cap_with_selective_reset():
    t=task(2,'cuda');t.phase[:]=3;t.contact_reward_clock[:]=1.;t.contact_reward_paid[:]=5.
    t.begin_step(torch.zeros((2,20),device='cuda'))
    t.consume(loads(t,10.),~t.done,.01,vertical_force_n=torch.tensor([10.,-10.],device='cuda'))
    torch.testing.assert_close(t.terms['contact'],torch.tensor([.05,0.],device='cuda',dtype=torch.float64))
    assert t.contact_reward_paid[0]>5.
    t.reset([0],measurement(t))
    assert t.contact_reward_paid.tolist()==[0.,5.]
    assert t.terms['downward_force'].tolist()==pytest.approx([0.,-.05])


def test_modes_have_distinct_contract_and_cannot_silently_resume():
    from go2_pvcnn.control.dh116.tabletop_vertical_force import vertical_reward_contract
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    old=vertical_reward_contract('window');new=vertical_reward_contract('continuous')
    assert new['upward_window_s'] is None and new['upward_episode_max_points'] is None
    with pytest.raises(ValueError,match='contract mismatch'):
        checkpoint_contract_changes({'vertical_force_reward':old},{'vertical_force_reward':new})
