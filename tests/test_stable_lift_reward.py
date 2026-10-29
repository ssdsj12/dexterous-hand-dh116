import pytest
import torch


def module():
    from go2_pvcnn.control.dh116 import tabletop_stable_lift_reward
    return tabletop_stable_lift_reward


def state(n=2):
    return dict(bottom=torch.full((n,),.315,dtype=torch.float64),
                lifting=torch.ones(n,dtype=torch.bool),
                load_ready=torch.ones(n,dtype=torch.bool),
                speed=torch.zeros(n,dtype=torch.float64),
                slip=torch.zeros(n,dtype=torch.float64),
                vertical_force=torch.ones(n,dtype=torch.float64),
                eligible=torch.ones(n,dtype=torch.bool))


def test_real_object_height_and_stationary_object():
    r=module().StableLiftReward(2,'cpu');s=state()
    s['bottom'][1]=.30
    progress,hold=r.update(**s,dt=.01)
    torch.testing.assert_close(progress,torch.tensor([2.5,0.],dtype=torch.float64))
    torch.testing.assert_close(hold,torch.tensor([.02,0.],dtype=torch.float64))
    progress,_=r.update(**s,dt=.01)
    assert not progress.any()


@pytest.mark.parametrize('key,value',[
    ('lifting',False),('load_ready',False),('eligible',False),
    ('speed',.03),('slip',.0021),('vertical_force',-.001),
    ('bottom',float('nan')),('slip',float('nan')),('speed',float('inf')),
    ('vertical_force',float('nan'))])
def test_bad_grip_or_invalid_measurement_does_not_pay_or_consume_budget(key,value):
    r=module().StableLiftReward(2,'cpu');s=state();s[key][:]=value
    progress,hold=r.update(**s,dt=.01)
    assert torch.isfinite(progress).all() and torch.isfinite(hold).all()
    assert not progress.any() and not hold.any()
    assert not r.observe().any()


def test_repeated_drop_and_raise_cannot_farm_height():
    r=module().StableLiftReward(2,'cpu');s=state();total=torch.zeros(2)
    for height in [.31,.305,.31,.32,.30,.32,.33,.4,.30,.33]:
        s['bottom'][:]=height
        progress,_=r.update(**s,dt=.01);total+=progress.float()
    torch.testing.assert_close(total,torch.full((2,),5.))


@pytest.mark.parametrize('dt',[1/480,1/60,.01])
def test_holding_is_time_scaled_and_capped_per_episode(dt):
    r=module().StableLiftReward(2,'cpu');s=state();s['bottom'][:]=.33
    total=0.
    for _ in range(round(1/dt)):
        _,hold=r.update(**s,dt=dt);total+=float(hold[0])
    assert total==pytest.approx(2.)
    assert r.observe().shape==(2,2)


def test_hold_requires_real_clearance_and_no_budget_reset_after_loss():
    r=module().StableLiftReward(2,'cpu');s=state();s['bottom'][:]=.302
    assert not r.update(**s,dt=.1)[1].any()
    s['bottom'][:]=.33
    assert r.update(**s,dt=.25)[1][0]==pytest.approx(1.)
    s['load_ready'][:]=False
    assert not r.update(**s,dt=.25)[1].any()
    s['load_ready'][:]=True
    assert r.update(**s,dt=.5)[1][0]==pytest.approx(1.)
    assert not r.update(**s,dt=.1)[1].any()
    r.reset(torch.tensor([0]))
    assert not r.observe()[0].any() and r.observe()[1].all()
    p,h=r.update(**s,dt=.1)
    assert p[0]==pytest.approx(5.) and p[1]==0
    assert h[0]==pytest.approx(.4) and h[1]==0


@pytest.mark.parametrize('dt',[0,-.1,float('nan'),float('inf')])
def test_invalid_dt_rejected(dt):
    r=module().StableLiftReward(2,'cpu')
    with pytest.raises(ValueError):r.update(**state(),dt=dt)
