"""A smaller physics step must preserve control and stability durations."""
from types import SimpleNamespace
import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_multifinger_runtime import MultifingerLiftControl


def controller(hz, n=1):
    world=SimpleNamespace(n=n,device='cpu',dt=1/hz,hands=[],shapes=[],
        t=lambda value:torch.as_tensor(value,dtype=torch.float64))
    return MultifingerLiftControl(world)


@pytest.mark.parametrize('hz',[480,960])
def test_physical_sampling_preserves_feedback_and_stability_time(hz):
    control=controller(hz)
    assert control.history.shape[0]*control.dt==pytest.approx(1/60)
    assert control.monitor.window_steps*control.dt==pytest.approx(.3)


@pytest.mark.parametrize('hz',[480,960])
def test_feedback_keeps_the_last_control_interval_of_raw_samples(hz):
    control=controller(hz)
    samples=round(hz/60)
    for value in range(1,2*samples+1):
        control.record_loads(torch.full((1,2,5),float(value),dtype=torch.float64))
    expected=torch.arange(samples+1,2*samples+1,dtype=torch.float64).mean()
    actual=control.history.sum(0)/control.count[:,None,None]
    torch.testing.assert_close(actual,torch.full_like(actual,expected))
    assert control.count.tolist()==[samples]


def test_partial_reset_clears_only_selected_feedback_at_refined_rate():
    control=controller(960,n=2)
    for _ in range(16):control.record_loads(torch.ones(2,2,5,dtype=torch.float64))
    control.reset(torch.tensor([0]))
    assert control.count.tolist()==[0,16]
    assert not control.history[:,0].any()
    assert control.history[:,1].eq(1).all()
