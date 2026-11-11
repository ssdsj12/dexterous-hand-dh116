"""Three independent action input modes; no critic state in Actor inputs."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

torch.set_num_threads(4)


def history():
    name='_actor_modes_fixture'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('test_dh116_stereo_estimator.py'))
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name].history(1)


def task_config():
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import state_contract
    row=state_contract()['dimensions'][116]
    return dict(allowlist=[116],source_qualification={key:True for key in row['required_validity_keys']})


@pytest.mark.parametrize('mode',['latent','estimated_task_state','hybrid'])
def test_mode_explicit_factory_and_bound_canonical24(mode):
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor,normalize_network_config
    cfg=dict(embed_dim=32,actor_input_mode=mode,point_coordinate_mode='palms',task_state_config=task_config())
    actor=make_stereo_actor(cfg).eval()
    assert actor.input_mode==mode
    assert normalize_network_config(cfg)['actor_input_mode']==mode
    action=actor.deterministic(history())
    assert action.shape==(1,24) and torch.isfinite(action).all() and (action.abs()<=1).all()


@pytest.mark.parametrize('mode',['estimated_task_state','hybrid'])
def test_action_loss_alone_reaches_selected371_prediction_head(mode):
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor
    actor=make_stereo_actor(dict(embed_dim=32,actor_input_mode=mode,task_state_config=task_config())).train()
    action=actor(history())
    action.square().sum().backward()
    gradient=actor.estimator.heads['privileged_state'].weight.grad
    assert gradient is not None and gradient[116].abs().sum()>0
    assert gradient[180:200].abs().sum()==0,'unobservable state must not enter Actor via the adapter'


def test_latent_baseline_keeps_unused371_head_and_critic_cannot_change_action():
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import StereoAsymmetricPolicy
    model=StereoAsymmetricPolicy(dict(embed_dim=32)).eval();h=history()
    first=model.deterministic(h)
    with torch.no_grad():
        for p in model.critic.parameters():p.add_(10.)
    second=model.deterministic(h)
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    first.sum().backward()
    assert model.actor.estimator.heads['privileged_state'].weight.grad is None


def test_unknown_actor_mode_rejected_instead_of_silently_latent():
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import normalize_network_config
    with pytest.raises(ValueError,match='mode'):
        normalize_network_config({'actor_input_mode':'all_371_ground_truth'})
