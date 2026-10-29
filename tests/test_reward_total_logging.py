from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import pytest


def test_ppo_total_is_logged_with_components_at_same_update(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_tensor_training import write_reward_scalars
    with SummaryWriter(str(tmp_path)) as writer:
        write_reward_scalars(writer,{'lift':2.,'stable_lift_hold':.5,'failure':-1.},1.5,37)
        write_reward_scalars(writer,{'lift':0.,'stable_lift_hold':0.,'failure':0.},0.,38)
    events=EventAccumulator(str(tmp_path)).Reload()
    total=events.Scalars('reward/total')
    assert [(v.step,v.value) for v in total]==[(37,1.5),(38,0.)]
    for term in ('lift','stable_lift_hold','failure'):
        assert [v.step for v in events.Scalars('reward/'+term)]==[37,38]
    assert sum(events.Scalars('reward/'+term)[0].value for term in (
        'lift','stable_lift_hold','failure'))==pytest.approx(total[0].value)
