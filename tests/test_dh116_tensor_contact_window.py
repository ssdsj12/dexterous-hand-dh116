import pytest
import torch
from test_dh116_tensor_task import template,measurement
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask


def loads(t,total):
    m=measurement(t)
    m['hand_object_loads'][:]=total/2
    m['object_link_loads'][:]=total/2
    m['grouped_object_loads'].zero_();m['grouped_object_loads'][:,:,5]=total/2
    return m


def task(n=1,start='first_contact'):
    return TensorTask(template(),n,'cpu',contact_reward_window=start)


@pytest.mark.parametrize('force,expected',[(0.,0.),(2.,1.),(5.,2.5),(10.,5.),(20.,5.)])
def test_only_first_second_pays_force_proportional_contact(force,expected):
    t=task();t.begin_step(torch.zeros((1,20)));m=loads(t,force)
    for _ in range(150):t.consume(m,~t.done,.01)
    assert t.terms['contact'].item()==pytest.approx(expected,abs=1e-10)
    assert 0<=t.contact_reward_paid.item()<=5
    before=t.terms['contact'].clone()
    for _ in range(50):t.consume(m,~t.done,.01)
    assert torch.equal(before,t.terms['contact'])


def test_first_contact_clock_never_pauses_or_restarts_on_loss_or_phase_switch():
    t=task();t.begin_step(torch.zeros((1,20)))
    for _ in range(100):t.consume(loads(t,0),~t.done,.01)
    assert t.contact_reward_clock.item()==-1
    t.consume(loads(t,10),~t.done,.25)
    assert t.terms['contact'].item()==pytest.approx(1.25)
    for _ in range(100):t.consume(loads(t,0),~t.done,.01)
    for phase in (2,1,3,1):
        t.phase[:]=phase;t.consume(loads(t,10),~t.done,.1)
    assert t.contact_reward_clock.item()==1
    assert t.terms['contact'].item()==pytest.approx(1.25)


def test_close_start_window_expires_even_without_contact():
    t=task(start='close_start');t.begin_step(torch.zeros((1,20)))
    for _ in range(110):t.consume(loads(t,0),~t.done,.01)
    t.consume(loads(t,10),~t.done,.1)
    assert t.terms['contact'].item()==0


@pytest.mark.parametrize('hz',[60,480])
def test_window_clips_last_substep_and_has_per_episode_five_point_cap(hz):
    t=task();t.begin_step(torch.zeros((1,20)))
    for _ in range(2*hz):t.consume(loads(t,10),~t.done,1/hz)
    assert t.contact_reward_paid.item()==pytest.approx(5,abs=1e-10)
    assert t.contact_reward_paid.item()<=5
    assert t.observe().shape==(1,150)
    assert t.observe()[0,-2].item()==1


def test_window_and_once_bonus_reset_only_for_selected_episode():
    t=task(2);t.begin_step(torch.zeros((2,20)))
    for _ in range(100):t.consume(loads(t,10),~t.done,.01)
    t.grasp_ready_awarded[:]=True
    t.reset([0],measurement(t))
    assert t.contact_reward_clock.tolist()==[-1,1]
    assert t.contact_reward_paid.tolist()==pytest.approx([0,5],abs=1e-12)
    assert t.grasp_ready_awarded.tolist()==[False,True]
    t.begin_step(torch.zeros((2,20)));t.consume(loads(t,10),~t.done,.1)
    assert t.terms['contact'].tolist()==pytest.approx([.5,0])


def test_grasp_ready_bonus_requires_full_dwell_and_cannot_repeat():
    t=task();t.f_requested[:]=.11;t.begin_step(torch.zeros((1,20)))
    for _ in range(239):t.consume(measurement(t),~t.done,1/480)
    assert t.terms['grasp_ready'].item()==0 and t.terms['contact_dwell'].item()==0
    t.consume(measurement(t),~t.done,1/480)
    assert t.phase.item()==3 and t.terms['grasp_ready'].item()==3
    t.phase[:]=1;t.contact_time.zero_()
    for _ in range(240):t.consume(measurement(t),~t.done,1/480)
    assert t.terms['grasp_ready'].item()==3


def test_failed_terminal_sample_pays_no_contact_or_ready_bonus():
    t=task();t.begin_step(torch.zeros((1,20)));m=loads(t,10);m['clearance'][:]=-.002
    t.consume(m,~t.done,.01)
    assert t.terms['contact'].item()==0 and t.terms['grasp_ready'].item()==0
    assert t.contact_reward_paid.item()==0
    assert t.terms['failure'].item()==-25


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_window_last_fraction_stays_on_cuda_and_clips_budget():
    t=TensorTask(template(),4,'cuda',contact_reward_window='first_contact')
    t.contact_reward_clock[:]=.998;t.contact_reward_paid[:]=4.99
    t.begin_step(torch.zeros((4,20),device='cuda'));t.consume(loads(t,10),~t.done,1/480)
    assert t.observe().shape==(4,150) and t.observe().device.type=='cuda'
    torch.testing.assert_close(t.contact_reward_clock,torch.ones_like(t.contact_reward_clock))
    assert bool((t.contact_reward_paid<=5.).all())
    torch.testing.assert_close(t.terms['contact'],torch.full_like(t.terms['contact'],.01))
