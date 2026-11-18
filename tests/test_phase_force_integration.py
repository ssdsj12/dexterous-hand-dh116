import pytest
import torch
from test_force_region_integration import constrained


def rewarded():
    t=constrained();t.phase_force_reward=True
    t.terms['grasp_upward_force']=torch.zeros_like(t.reward)
    t.terms['lift_upward_force']=torch.zeros_like(t.reward)
    t.m['vertical_hand_force_n'][:]=.4
    return t


@pytest.mark.parametrize('phase',[1,3])
def test_phase_force_reward_reaches_actual_task_reward(phase):
    t=rewarded();t.phase[:]=phase
    t.consume(t.m,~t.done,.01,vertical_force_n=torch.full((2,),.8),
        lift_vertical_force_n=torch.ones(2),hand_lift_force_n=torch.full((2,2),.5),
        object_weight_n=torch.ones(2))
    assert (t.terms['grasp_upward_force']>0).all()==(phase==1)
    assert (t.terms['lift_upward_force']>0).all()==(phase==3)
    _,r,_,extra=t.end_step()
    torch.testing.assert_close(r,sum(extra['terms'].values()))


def test_friction_only_support_rewards_lift_not_close():
    t=rewarded();t.phase[:]=3;t.m['vertical_hand_force_n'].zero_()
    t.consume(t.m,~t.done,.01,vertical_force_n=torch.zeros(2),lift_vertical_force_n=torch.ones(2),
        hand_lift_force_n=torch.full((2,2),.5),object_weight_n=torch.ones(2))
    assert not t.terms['grasp_upward_force'].any()
    assert (t.terms['lift_upward_force']>0).all()


def test_phase_force_reward_requires_real_per_hand_lift_measurements():
    t=rewarded()
    with pytest.raises(ValueError,match='per.hand'):
        t.consume(t.m,~t.done,.01,vertical_force_n=torch.zeros(2),object_weight_n=torch.ones(2))


def test_nonfinite_raw_hand_force_preserves_task_failure_without_reward_error():
    t=rewarded()
    hand=torch.ones((2,2));hand[0,0]=float('nan')
    t.consume(t.m,~t.done,.01,vertical_force_n=torch.ones(2),
        lift_vertical_force_n=torch.ones(2),hand_lift_force_n=hand,object_weight_n=torch.ones(2))
    assert t.failure[0]!=0 and t.nonfinite[0]
    assert t.terms['grasp_upward_force'][0]==0.
    assert torch.isfinite(t.reward).all()
