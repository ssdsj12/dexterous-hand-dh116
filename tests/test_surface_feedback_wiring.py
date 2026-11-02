import torch

from test_surface_exploration import surface_task
from go2_pvcnn.control.dh116 import tabletop_surface_exploration as mode


def test_feedback_observation_keeps_exact_legacy_prefix_and_does_not_change_actions_or_reward():
    from go2_pvcnn.control.dh116.tabletop_surface_feedback import SurfaceFeedback
    t,w=surface_task(); before=t.observe().clone(); rates=mode.reward_rates(t)
    t.surface_feedback=SurfaceFeedback(t.n,t.device,dtype=t.command.dtype)
    after=t.observe()
    assert after.shape[-1]==before.shape[-1]+20
    assert torch.equal(after[:,:before.shape[-1]],before)
    assert not after[:,-20:].any()
    assert t.action_size==24 and t.surface_redistribution is False
    for key,value in rates.items():assert torch.equal(mode.reward_rates(t)[key],value)


def test_371_input_actor_is_explicitly_supported():
    from go2_pvcnn.control.dh116.tabletop_surface_imitation import make_actor
    actor=make_actor(371)
    assert actor(torch.zeros(2,371)).shape==(2,24)
