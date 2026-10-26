import numpy as np
import pytest
import torch
from test_dh116_batched_task import make_env
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask, FAILURE_CODES, phase_potential
from go2_pvcnn.control.dh116.tabletop_batched_task import BatchTask


def template():
    e=make_env(.055)
    e.full_hand_contact=True
    e.backend.object_contact_links=('palm_link',)
    e.measurement.update(object_link_loads=np.full((2,1),.11),grouped_object_loads=np.array([[0,0,0,0,0,.11]]*2),hand_object_loads=np.full(2,.11))
    return e


def measurement(t): return {k:v.clone() for k,v in t.m.items()}


def test_task_keeps_per_hand_vertical_force_for_direction_control():
    t = TensorTask(template(), 1, 'cpu')
    assert t.m['vertical_hand_force_n'].shape == (1, 2)
    assert torch.equal(t.m['vertical_hand_force_n'], torch.zeros((1, 2), dtype=torch.float64))


def test_observation_matches_scalar_and_control_matches_batch():
    e=template(); t=TensorTask(e,2,'cpu'); b=BatchTask([e,e])
    np.testing.assert_allclose(t.observe()[0].numpy(),e.observe(),atol=1e-7)
    rng=np.random.default_rng(7)
    for _ in range(20):
        a=rng.uniform(-1,1,(2,20)); live=np.ones(2,dtype=bool)
        t.control(torch.tensor(a),torch.tensor(live),1/480); b.control(a,live,1/480)
        for name in ('command','offset','f_offset','f_filtered','f_requested','f_target','f_boost','palm_actions'):
            np.testing.assert_allclose(getattr(t,name).numpy(),getattr(b,name),atol=1e-12)


def test_invalid_nonfinite_and_selective_reset():
    t=TensorTask(template(),2,'cpu'); a=torch.zeros((2,20));a[0,0]=2
    t.begin_step(a); assert t.failure[0]==FAILURE_CODES['invalid_action']
    m=measurement(t);m['q'][1,0]=float('nan');t.consume(m,~t.done,.01)
    obs,r,d,_=t.end_step(); assert d.all() and torch.isfinite(obs).all()
    t.reset([0],measurement(t));assert not t.done[0] and t.done[1]


def test_phase_progress_and_real_lift():
    t=TensorTask(template(),1,'cpu');t.f_requested[:]=.11
    t.begin_step(torch.zeros((1,20)))
    for _ in range(20):t.consume(measurement(t),~t.done,.01)
    assert t.phase.item()==2
    for _ in range(10):t.consume(measurement(t),~t.done,.01)
    gained=t.terms['contact_dwell'].item(); assert gained>1
    m=measurement(t);m['hand_object_loads'].zero_();m['object_link_loads'].zero_();m['grouped_object_loads'].zero_()
    t.consume(m,~t.done,.01); assert t.phase.item()==1
    assert t.terms['contact_dwell'].item()==pytest.approx(0,abs=1e-10)
    t.phase[:]=3;t.contact_time.zero_();m=measurement(t);m['bottom'][:]=.33
    t.consume(m,~t.done,.01);assert t.terms['lift'].item()==pytest.approx(5)
    for _ in range(49):t.consume(m,~t.done,.01)
    assert t.success.item();assert t.terms['lift_dwell'].item()==pytest.approx(2)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_residency():
    t=TensorTask(template(),8,'cuda');a=t.begin_step(torch.zeros((8,20),device='cuda'))
    t.control(a,~t.done,1/480);t.consume(measurement(t),~t.done,1/480)
    obs,r,d,extra=t.end_step()
    assert all(v.device.type=='cuda' for v in (obs,r,d,*t.m.values()))


def test_unsupported_phase_and_lift_dwell_regress():
    t=TensorTask(template(),1,'cpu');t.begin_step(torch.zeros((1,20)))
    t.f_requested[:]=4.
    for _ in range(30):t.consume(measurement(t),~t.done,.01)
    assert t.phase.item()==1
    assert t.terms['contact_dwell'].item()==0
    t.phase[:]=3;m=measurement(t);m['bottom'][:]=.33
    for _ in range(20):t.consume(m,~t.done,.01)
    assert t.terms['lift_dwell'].item()==pytest.approx(.8)
    m['bottom'][:]=.32;t.consume(m,~t.done,.01)
    assert t.terms['lift_dwell'].item()==pytest.approx(0,abs=1e-12)


def test_contact_scale_only_close_and_terminal_reward_once():
    t=TensorTask(template(),1,'cpu');t.begin_step(torch.zeros((1,20)))
    m=measurement(t);m['hand_object_loads'][:]=5.;m['object_link_loads'][:]=5.;m['grouped_object_loads'][:,:,5]=5.
    t.consume(m,~t.done,.01)
    assert t.terms['contact'].item()==pytest.approx(5-.11)
    t.phase[:]=2;m['hand_object_loads'][:]=4.;m['object_link_loads'][:]=4.;m['grouped_object_loads'][:,:,5]=4.
    old=t.terms['contact'].clone();t.consume(m,~t.done,.01)
    assert t.terms['contact'].item()==pytest.approx(old.item()-1.)
    m['clearance'][:]=-.002;t.consume(m,~t.done,.01)
    assert t.terms['failure'].item()==-25
    t.consume(m,~t.done,.01);assert t.terms['failure'].item()==-25


def test_480hz_boundaries_and_repeated_readiness_cycles():
    t=TensorTask(template(),1,'cpu');t.f_requested[:]=.11
    t.begin_step(torch.zeros((1,20)));dt=1/480
    for _ in range(95):t.consume(measurement(t),~t.done,dt)
    assert t.phase.item()==1
    t.consume(measurement(t),~t.done,dt);assert t.phase.item()==2
    for _ in range(143):t.consume(measurement(t),~t.done,dt)
    assert t.phase.item()==2
    t.consume(measurement(t),~t.done,dt);assert t.phase.item()==3
    assert t.terms['contact_dwell'].item()==pytest.approx(3)
    t.reset([0],measurement(t));t.f_requested[:]=.11;t.begin_step(torch.zeros((1,20)))
    for _ in range(3):
        for _ in range(120):t.consume(measurement(t),~t.done,dt)
        assert t.phase.item()==2
        m=measurement(t);m['object_velocity'][:,0]=.021
        t.consume(m,~t.done,dt)
        assert t.phase.item()==1
        m['object_velocity'].zero_();t.m=m
    assert t.terms['contact_dwell'].item()==pytest.approx(0,abs=1e-10)


def test_hold_contact_loss_repays_close_reacquisition():
    t=TensorTask(template(),1,'cpu');t.f_requested[:]=1.
    t.begin_step(torch.zeros((1,20)))
    high=measurement(t);high['hand_object_loads'][:]=1.;high['object_link_loads'][:]=1.;high['grouped_object_loads'][:,:,5]=1.
    t.consume(high,~t.done,.01)
    first=t.terms['contact'].item()
    for _ in range(19):t.consume(high,~t.done,.01)
    assert t.phase.item()==2
    for _ in range(3):
        low=measurement(t);low['hand_object_loads'][:]=.1;low['object_link_loads'][:]=.1;low['grouped_object_loads'][:,:,5]=.1
        before=t.terms['contact'].item();t.consume(low,~t.done,.01)
        assert t.phase.item()==1
        assert t.terms['contact'].item()==pytest.approx(before-.9)
        for _ in range(20):t.consume(high,~t.done,.01)
        assert t.phase.item()==2
        assert t.terms['contact'].item()==pytest.approx(first)
    # Increasing support during hold provides no positive contact shaping.
    higher={k:v.clone() for k,v in high.items()};higher['hand_object_loads'][:]=1.1;higher['object_link_loads'][:]=1.1;higher['grouped_object_loads'][:,:,5]=1.1
    before=t.terms['contact'].clone();t.consume(higher,~t.done,.01)
    assert torch.equal(before,t.terms['contact'])
