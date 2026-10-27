import pytest
import torch
from test_dh116_tensor_task import template, measurement
from test_dh116_tensor_contact_window import loads
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask


def task(n=1):
    return TensorTask(template(),n,'cpu',contact_reward_window='close_start',vertical_force_reward=True)


@pytest.mark.parametrize('force,up,down',[(0,0,0),(2,1,0),(10,5,0),(20,5,0),(-2,0,-1),(-10,0,-5),(-20,0,-10)])
def test_signed_force_replaces_unsigned_contact_and_downward_cost_is_linear(force,up,down):
    t=task();t.phase[:]=3;t.begin_step(torch.zeros((1,20)))
    m=loads(t,10)
    for _ in range(100):
        t.consume(m,~t.done,.01,vertical_force_n=torch.tensor([force]))
    assert t.terms['contact'].item()==pytest.approx(up)
    assert t.terms['downward_force'].item()==pytest.approx(down)
    # Positive credit expires; negative cost continues throughout hold/lift.
    for _ in range(100):
        t.consume(m,~t.done,.01,vertical_force_n=torch.tensor([force]))
    assert t.terms['contact'].item()==pytest.approx(up)
    assert t.terms['downward_force'].item()==pytest.approx(2*down)


def test_close_grasp_ready_requires_positive_normal_force_even_with_positive_lift_force():
    t=task();t.phase[:]=3;t.begin_step(torch.zeros((1,20)))
    t.consume(loads(t,4),~t.done,.01,
              vertical_force_n=torch.tensor([-1.]),
              lift_vertical_force_n=torch.tensor([1.]))
    assert t.terms['grasp_ready'].item()==0.
    assert not t.grasp_ready_awarded.item()


@pytest.mark.parametrize('hz',[60,480])
def test_substep_frequency_does_not_change_penalty_and_inactive_rows_do_not_pay(hz):
    t=task(2);t.phase[:]=3;t.begin_step(torch.zeros((2,20)))
    m=loads(t,10);live=torch.tensor([True,False])
    for _ in range(hz):t.consume(m,live,1/hz,vertical_force_n=torch.tensor([-4.,-20.]))
    assert t.terms['downward_force'].tolist()==pytest.approx([-2,0])
    assert not t.terms['contact'].any()


def test_nonfinite_vertical_force_fails_instead_of_corrupting_rewards():
    t=task(2);t.begin_step(torch.zeros((2,20)))
    t.consume(loads(t,10),~t.done,.01,vertical_force_n=torch.tensor([float('nan'),2.]))
    assert t.done.tolist()==[True,False]
    assert torch.isfinite(t.reward).all()
    assert t.terms['failure'].tolist()==[-25,0]
    assert t.terms['contact'].tolist()==pytest.approx([0,.01])


def test_missing_force_never_falls_back_to_unsigned_load():
    t=task();t.begin_step(torch.zeros((1,20)))
    with pytest.raises(ValueError,match='vertical force'):
        t.consume(loads(t,10),~t.done,.01)


def test_terminal_substep_keeps_downward_penalty_but_never_repeats_after_reset():
    from test_dh116_tensor_env import World
    from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv
    t=task(2);w=World(t)
    # Only signed normal components contribute: -2-4 = -6 N per sample.
    w.vertical_components=torch.tensor([[[-2.,1.],[-4.,1.]]]*2)
    original_reset=w.reset
    def reset(ids):
        m=original_reset(ids);w.vertical_components[ids]=-1000.
        return m
    w.reset=reset
    env=TensorEnv(w,t)
    _,_,done,extra=env.step(torch.zeros((2,20)))
    assert done.tolist()==[True,False]
    assert extra['terms']['downward_force'].tolist()==pytest.approx([-3*3/480,-3*8/480])
    assert not extra['terms']['contact'].any()


def test_old_unsigned_reward_checkpoint_cannot_silently_resume_as_signed_reward():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_vertical_force import REWARD_CONTRACT
    old=dict(reward='contact_window_v2_1s_10n_5max')
    new=dict(reward=REWARD_CONTRACT['version'],vertical_force_reward=REWARD_CONTRACT)
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,new)
    assert checkpoint_contract_changes(new,new)==[]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_signed_reward_runs_on_gpu_and_clips_final_positive_window_fraction():
    t=TensorTask(template(),2,'cuda',contact_reward_window='close_start',vertical_force_reward=True)
    t.phase[:]=3;t.contact_reward_clock[:]=.999
    t.begin_step(torch.zeros((2,20),device='cuda'))
    t.consume(loads(t,10),~t.done,.01,vertical_force_n=torch.tensor([20.,-20.],device='cuda'))
    torch.testing.assert_close(t.terms['contact'],torch.tensor([.005,0.],device='cuda',dtype=torch.float64))
    torch.testing.assert_close(t.terms['downward_force'],torch.tensor([0.,-.1],device='cuda',dtype=torch.float64))


def test_downward_grip_cannot_get_ready_bonus_but_later_upward_support_can():
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
    t=task();t.grasp_readiness='measured_load'
    t.parameters=LoadParameters(max_force_n=12.,force_rate_n_s=None)
    t.begin_step(torch.zeros((1,20)))
    m=loads(t,4.)
    for _ in range(240):
        t.consume(m,~t.done,1/480,vertical_force_n=torch.tensor([-2.25]))
    assert t.phase.item()==3  # Still allowed to try lifting; reward is separate.
    assert t.terms['grasp_ready'].item()==0
    assert not t.grasp_ready_awarded.item()
    t.consume(m,~t.done,1/480,vertical_force_n=torch.tensor([1.2]))
    assert t.terms['grasp_ready'].item()==3
    t.consume(m,~t.done,1/480,vertical_force_n=torch.tensor([1.2]))
    assert t.terms['grasp_ready'].item()==3
