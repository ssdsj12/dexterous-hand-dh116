"""Recorded bounded source tails are CPU schema fixtures, never native trials."""
from types import SimpleNamespace

import pytest
import torch

from test_vision_teacher_collection import options


@pytest.mark.parametrize('tail', [-1, True, 1801])
def test_tail_requires_a_bounded_integer(tail):
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import validate_options
    with pytest.raises(ValueError, match='tail'):
        validate_options(options(vision_teacher_mpc=True,vision_teacher_mpc_profile='absolute_reference',
            vision_teacher_hold_tail_control_steps=tail))


@pytest.mark.parametrize('changes', [dict(vision_teacher_mpc=False),
    dict(vision_teacher_source=None),dict(vision_teacher_mpc_profile='local_pad')])
def test_tail_requires_an_explicit_source_and_experimental_mpc(changes):
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import validate_options
    args=options(vision_teacher_mpc=True,vision_teacher_mpc_profile='absolute_reference',
        vision_teacher_hold_tail_control_steps=3)
    for k,v in changes.items():setattr(args,k,v)
    with pytest.raises(ValueError):validate_options(args)


@pytest.mark.parametrize('tail,early', [(0,False),(3,False),(3,True)])
def test_recorded_tail_owns_actual_actions_and_preserves_first_terminal(tmp_path,monkeypatch,tail,early):
    from test_vision_training import fixture,Camera
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_collection as collection
    from go2_pvcnn.control.dh116 import tabletop_vision_supervision_dataset as dataset
    from go2_pvcnn.control.dh116 import tabletop_vision_training as training
    dw,task,env,args=fixture(tmp_path)
    for k,v in vars(options(num_envs=2,vision_teacher_mpc=True,
            vision_teacher_mpc_profile='absolute_reference',vision_teacher_max_control_steps=9,
            vision_teacher_hold_tail_control_steps=tail)).items():setattr(args,k,v)
    styles=['pinch_thumb_index','tripod_thumb_index_middle']
    refs=[dict(source_episode_id=i,grasp_style=s,source_trajectory_id=s,
        actions=torch.full((i+1,24),.1*(i+1))) for i,s in enumerate(styles)]
    monkeypatch.setattr(dataset,'load_teacher_reference_dataset',lambda *a:
        dict(source_sha256='a'*64,episodes=refs,references={}))
    monkeypatch.setattr(training,'make_squashed_vision_algorithm',lambda *a,**kw:
        pytest.fail('tail collection must not create PPO'))
    sim=SimpleNamespace(current_time=0.,current_time_step_index=0)
    captured=[];contexts=[]
    class Optimizer:
        def refine(self,reference,*,context,alive):
            captured.append(reference.clone());contexts.append(context)
            issued=reference.clone();issued[(reference==0).all(-1),0]=.15
            return issued,dict(per_env=[dict(env=i,active=True,changed=True) for i in range(2)])
        def observe_result(self,done):pass
    monkeypatch.setattr(collection,'_teacher_optimizer',lambda *a:Optimizer())
    class Capture(Camera):
        def set_raw_frame_sink(self,sink):self.sink=sink
        def sample(self):
            frame=sim.current_time_step_index+1
            self.sink(dict(rgb=torch.zeros(2,2,2,3,dtype=torch.uint8),depth=torch.ones(2,2,2,1),
                frame_id=frame,simulation_time=sim.current_time,rendered_simulation_time=sim.current_time,
                physics_step=sim.current_time_step_index))
            return dict(super().sample(),frame_id=frame,simulation_time=sim.current_time)
    actual=[]
    def step(action):
        actual.append(action.clone());env.tick+=1;sim.current_time+=1/60;sim.current_time_step_index+=8
        done=torch.tensor([env.tick==(1 if early else 5),env.tick==5])
        return torch.ones(2,371)*env.tick,torch.zeros(2),done,dict(success=done.clone(),
            failure=torch.zeros(2,dtype=torch.long),recovered=torch.zeros(2,dtype=torch.bool))
    env.step=step
    report=collection.run_teacher_collection(SimpleNamespace(sim=sim),dw,task,env,args,
        {'observation_size':371},camera=Capture(2))
    assert report['control_steps']==(2 if not tail else 5)
    assert report['native_runtime_verified'] is False and report['successful_episodes']==0
    assert report['native_successful_episodes']==0 and report['ppo_updates']==0
    assert not (args.output/'successful_visual_supervision.pt').exists()
    data=torch.load(args.output/'all_executed_supervision.pt',weights_only=False)
    assert all(not ep['native_physics'] for ep in data['episodes'].values())
    for row,action,reference in zip(data['sample_metadata'],data['actions'],data['reference_actions']):
        i,t=row['env_id'],row['tick']
        assert torch.equal(action,actual[t][i])
        if t>=len(refs[i]['actions']):
            assert tail and not reference.any()
            assert row['reference_phase']=='recorded_zero_hold_tail'
            assert row['reference_source_tick'] is None
            assert action[0]==pytest.approx(.15)
        elif tail:
            assert row['reference_phase']=='source' and row['reference_source_tick']==t
    if tail:
        assert report['recorded_reference_execution']['extra_steps_after_longest_source']==3
        assert report['recorded_reference_execution']['source_lengths']==[1,2]
        assert data['episodes']['1']['terminal']['recorded_execution_complete'] is True
        assert data['episodes']['1']['terminal']['terminal_reference_phase']=='recorded_zero_hold_tail'
        assert all(c['reference_phase'][1]=='recorded_zero_hold_tail' for c in contexts[2:])
    if early:
        assert all(not a[0].any() for a in actual[1:])
        assert sum(row['env_id']==0 for row in data['sample_metadata'])==1
    if not tail:
        assert len(data['sample_metadata'])==3
        assert all(ep['terminal'] is None for ep in data['episodes'].values())
