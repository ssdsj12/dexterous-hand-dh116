import torch
from test_phase_force_integration import rewarded


def light():
    t=rewarded();t.manual_control=True;t.manual_lightweight=True
    t.manual_lift_qualified=torch.zeros(t.n,dtype=torch.bool)
    t.last_observation=t.observe().clone()
    return t


def tick(t):
    t.begin_step(torch.zeros((2,20)))
    t.consume(t.m,~t.done,.01,vertical_force_n=torch.ones(2),
        lift_vertical_force_n=torch.ones(2),hand_lift_force_n=torch.ones((2,2)),object_weight_n=torch.ones(2))
    return t.end_step()


def test_light_manual_has_diagnostic_observation_without_training_rewards():
    t=light();obs,r,done,extra=tick(t)
    assert torch.isfinite(obs).all() and not done.any() and not r.any()
    assert not any(v.any() for v in extra['terms'].values())


def test_light_manual_preserves_failure_penalty_and_nonfinite_detection():
    t=light();t.m['joint_failure'][0]=1
    obs,r,done,extra=tick(t)
    assert done.tolist()==[True,False] and r[0]==-25.
    t=light();t.m['object_relative'][0,0]=1e100
    _,r,done,_=tick(t)
    assert done[0] and t.nonfinite[0] and r[0]==-25.
