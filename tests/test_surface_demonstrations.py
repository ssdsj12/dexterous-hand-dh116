from pathlib import Path
from types import SimpleNamespace
import torch
import pytest


def module():
    from go2_pvcnn.control.dh116 import tabletop_surface_demonstrations
    return tabletop_surface_demonstrations


def test_source_requires_verified_success_and_bounded_actions(tmp_path):
    path=tmp_path/'teacher.pt'
    torch.save(dict(metadata=dict(qualified_lift=False,action_size=24),
                    control=dict(action=torch.zeros(3,1,24))),path)
    with pytest.raises(ValueError,match='qualified'):
        module().load_teacher_actions(path)
    torch.save(dict(metadata=dict(qualified_lift=True,action_size=24),
                    control=dict(action=torch.full((3,1,24),1.1))),path)
    with pytest.raises(ValueError,match='bounded'):
        module().load_teacher_actions(path)


class ReplayEnv:
    def __init__(self):
        self.n=2;self.device='cpu';self.tick=0
        self.task=SimpleNamespace(surface_force_sharing=True)
        self.world=SimpleNamespace(surface_trace=[])
    def step(self,action):
        self.tick+=1
        obs=torch.full((2,351),float(self.tick))
        done=torch.tensor([self.tick==3,self.tick==1])
        success=done&torch.tensor([True,False])
        return obs,torch.ones(2),done,dict(success=success,failure=(done&~success).long(),
                    recovered=torch.zeros(2,dtype=torch.bool),terminal_observation=obs.clone())


def test_replay_pairs_pre_action_obs_and_rejects_failed_environment(tmp_path):
    obs=torch.zeros(2,351)
    actions=torch.zeros(3,24);actions[:,0]=torch.tensor([.1,.2,.3])
    env=ReplayEnv()
    result=module().collect_replay(env,obs,actions,dict(observation_size=351),tmp_path,
                                 source_identity=dict(sha256='test'))
    assert result['successful_replays']==1
    data=torch.load(tmp_path/'surface_demonstrations.pt',weights_only=False)
    assert data['version']=='dh116-surface-demonstrations-v1'
    assert data['observations'].shape==(3,351)
    assert torch.equal(data['observations'][:,0],torch.tensor([0.,1.,2.]))
    assert torch.equal(data['next_observations'][:,0],torch.tensor([1.,2.,3.]))
    assert torch.equal(data['actions'][:,0],actions[:,0])
    assert data['qualification']['qualified'] is True
    assert data['qualification']['successful_replays']==1
    assert data['episode_ids'].unique().numel()==1


def test_failed_replay_keeps_diagnostic_but_exports_no_teacher(tmp_path):
    env=ReplayEnv()
    result=module().collect_replay(env,torch.zeros(2,351),torch.zeros(2,24),dict(observation_size=351),tmp_path,
                                 source_identity=dict(sha256='test'))
    assert result['successful_replays']==0
    assert not (tmp_path/'surface_demonstrations.pt').exists()
    assert (tmp_path/'surface_replay.json').is_file()


def test_new_surface_training_options_cannot_silently_enter_legacy_mode():
    from go2_pvcnn.control.dh116.tabletop_surface_exploration import validate_options
    for name in ('surface_teacher_replay','surface_student_actor','surface_demonstrations'):
        args=SimpleNamespace(surface_force_sharing=False,**{name:Path('new.pt')})
        with pytest.raises(ValueError,match='surface'):
            validate_options(args)


def test_teacher_collection_cannot_be_mixed_with_student_optimization():
    from go2_pvcnn.control.dh116.tabletop_surface_exploration import validate_options
    args=SimpleNamespace(surface_force_sharing=True,free_wrist_grasp=True,
        surface_teacher_replay=Path('source.pt'),surface_demonstrations=Path('dataset.pt'))
    with pytest.raises(ValueError,match='distinct'):
        validate_options(args)


@pytest.mark.parametrize('settings',[
    dict(surface_student_eval_steps=-1),
    dict(surface_student_eval_steps=10),
    dict(surface_mpc=True),
    dict(surface_teacher_replay=Path('teacher.pt'),surface_student_eval_steps=10,
         surface_student_actor=Path('actor.pt')),
])
def test_evaluation_requires_a_real_student_and_mpc_is_inference_only(settings):
    from go2_pvcnn.control.dh116.tabletop_surface_exploration import validate_options
    args=SimpleNamespace(surface_force_sharing=True,free_wrist_grasp=True,**settings)
    with pytest.raises(ValueError):validate_options(args)


def test_grasp_style_requires_both_hands_and_excludes_third_bearing_digit():
    loads=torch.zeros(300,2,5,dtype=torch.float64)
    loads[...,:2]=1.
    assert module().contact_style(loads)['style']=='pinch_thumb_index'
    loads[10,1,2]=.1
    assert module().contact_style(loads)['style']!='pinch_thumb_index'
    loads[...,2]=.5
    assert module().contact_style(loads)['style']=='tripod_thumb_index_middle'
    loads[-1,0,0]=0.
    assert module().contact_style(loads)['style']!='tripod_thumb_index_middle'
