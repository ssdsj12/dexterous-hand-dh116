"""CPU alignment/qualification fixtures; these are never native evidence."""
import json
from types import SimpleNamespace

import pytest
import torch


def options(**kw):
    value=dict(vision_teacher_source='teacher.pt',vision_teacher_mpc=False,
        vision_teacher_max_control_steps=1300,vision_teacher_raw_chunk_frames=16,
        vision_student=True,vision_camera_mode='fixed',vision_action_distribution='tanh_gaussian',
        num_envs=4,resume=None,vision_evaluation_checkpoint=None,grasp_contact_probe=False,
        surface_mpc=False,vision_warm_start_optimizer='preserve',vision_lift_reward_experiment='R0')
    return SimpleNamespace(**dict(value,**kw))


@pytest.mark.parametrize('changes',[{'resume':'old.pt'},{'num_envs':18},
    {'vision_camera_mode':'bounded'},{'vision_action_distribution':'gaussian_clip'},
    {'vision_teacher_max_control_steps':10001},{'vision_teacher_raw_chunk_frames':1000},
    {'vision_evaluation_checkpoint':'eval.pt'},{'vision_lift_reward_experiment':'R1'}])
def test_collection_excludes_resume_ppo_and_unbounded_budgets(changes):
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import validate_options
    with pytest.raises(ValueError):validate_options(options(**changes))


def test_collection_default_off_and_mpc_requires_teacher():
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import validate_options
    assert validate_options(SimpleNamespace()) is None
    with pytest.raises(ValueError,match='source'):
        validate_options(options(vision_teacher_source=None,vision_teacher_mpc=True))
    validate_options(options())


@pytest.mark.parametrize('changes',[
    {'vision_teacher_mpc_profile':'unknown','vision_teacher_mpc':True},
    {'vision_teacher_mpc_profile':'absolute_reference','vision_teacher_mpc':False},
    {'vision_teacher_mpc_profile':'absolute_reference','vision_teacher_source':None},
])
def test_experimental_mpc_profile_requires_explicit_collection_opt_in(changes):
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import validate_options
    with pytest.raises(ValueError,match='profile|MPC'):
        validate_options(options(**changes))


def test_absolute_reference_factory_preserves_full_future_and_zero_pads_short_source(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_collection as collection
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_absolute_mpc as module
    captured={}
    def construct(env,source_sha,reference_actions):
        captured.update(env=env,source_sha=source_sha,sequence=reference_actions.clone())
        return captured
    monkeypatch.setattr(module,'AbsoluteTeacherReferenceMPC',construct)
    first=torch.arange(48,dtype=torch.float32).reshape(2,24)/100
    second=torch.full((3,24),.3)
    args=options(vision_teacher_mpc=True,vision_teacher_mpc_profile='absolute_reference')
    args.device='cpu'
    env=object()
    result=collection._teacher_optimizer(env,args,{'source_sha256':'a'*64},
        [{'actions':first},{'actions':second}])
    assert result is captured and captured['env'] is env
    assert captured['source_sha']=='a'*64 and captured['sequence'].shape==(2,3,24)
    assert torch.equal(captured['sequence'][0,:2],first)
    assert torch.equal(captured['sequence'][1],second)
    assert torch.equal(captured['sequence'][0,2],torch.zeros(24))
    captured['sequence'].zero_()
    assert first.sum()>0 and second.sum()>0


def test_pressure_profile_is_explicitly_supported_as_a_separate_experiment():
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import validate_options
    validate_options(options(vision_teacher_mpc=True,vision_teacher_mpc_profile='pressure_reference'))


def test_pressure_factory_passes_actual_grasp_identity(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_collection as collection
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_pressure_mpc as module
    captured={}
    def construct(env,source_sha,reference_actions,*,grasp_styles):
        captured.update(styles=grasp_styles,sequence=reference_actions.clone())
        return captured
    monkeypatch.setattr(module,'PressureTeacherReferenceMPC',construct)
    args=options(vision_teacher_mpc=True,vision_teacher_mpc_profile='pressure_reference')
    args.device='cpu'
    styles=['pinch_thumb_index','tripod_thumb_index_middle']
    references=[dict(actions=torch.zeros(2,24),grasp_style=s) for s in styles]
    result=collection._teacher_optimizer(object(),args,{'source_sha256':'a'*64},references)
    assert result is captured and captured['styles']==styles
    assert captured['sequence'].shape==(2,2,24)


def test_geometric_acquisition_requires_explicit_radius_and_collection_opt_in():
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import validate_options
    args=options(vision_teacher_mpc=True,vision_teacher_mpc_profile='pressure_geometry_acquire',
        solid_cylinder_radius_m=.008,object_kind='solid_cylinder')
    assert validate_options(args) is None
    for changes in ({'vision_teacher_mpc':False},{'vision_teacher_source':None},
            {'solid_cylinder_radius_m':None},{'solid_cylinder_radius_m':float('nan')},
            {'solid_cylinder_radius_m':0.},{'object_kind':'hollow_straw'},{'object_kind':None}):
        with pytest.raises(ValueError):validate_options(options(**dict(vars(args),**changes)))


def test_geometry_factory_binds_explicit_object_geometry_and_source_without_changing_actions(monkeypatch):
    import sys
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_collection as collection
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_absolute_mpc as absolute
    captured={}
    def construct(env,source_sha,reference_actions,*,grasp_styles,cylinder_radius_m,cylinder_length_m):
        captured.update(env=env,source_sha=source_sha,sequence=reference_actions.clone(),
            styles=grasp_styles,radius=cylinder_radius_m,length=cylinder_length_m)
        return captured
    module_name='go2_pvcnn.control.dh116.tabletop_vision_teacher_acquisition_mpc'
    monkeypatch.setitem(sys.modules,module_name,SimpleNamespace(AcquisitionTeacherReferenceMPC=construct))
    monkeypatch.setattr(absolute,'AbsoluteTeacherReferenceMPC',lambda *args,**kwargs: {'wrong_route':True})
    args=options(vision_teacher_mpc=True,vision_teacher_mpc_profile='pressure_geometry_acquire',
        solid_cylinder_radius_m=.008,object_kind='solid_cylinder')
    args.device='cpu';env=object()
    styles=['pinch_thumb_index','tripod_thumb_index_middle']
    references=[dict(actions=torch.full((2+i,24),.1*(i+1)),grasp_style=s)
        for i,s in enumerate(styles)]
    result=collection._teacher_optimizer(env,args,{'source_sha256':'a'*64},references)
    assert result is captured
    assert captured['env'] is env and captured['source_sha']=='a'*64
    assert captured['styles']==styles and captured['radius']==.008 and captured['length']==.5
    assert torch.equal(captured['sequence'][0,:2],references[0]['actions'])
    assert torch.equal(captured['sequence'][0,2],torch.zeros(24))
    assert torch.equal(captured['sequence'][1],references[1]['actions'])


def test_constrained_geometry_profile_has_the_same_explicit_geometry_admission():
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import validate_options
    args=options(vision_teacher_mpc=True,vision_teacher_mpc_profile='pressure_geometry_constrained',
        solid_cylinder_radius_m=.008,object_kind='solid_cylinder')
    assert validate_options(args) is None
    for changes in ({'vision_teacher_mpc':False},{'vision_teacher_source':None},
            {'solid_cylinder_radius_m':0.},{'object_kind':'hollow_straw'}):
        with pytest.raises(ValueError):validate_options(options(**dict(vars(args),**changes)))


def test_constrained_factory_selects_new_controller_and_preserves_source_sequence(monkeypatch):
    import sys
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_collection as collection
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_absolute_mpc as absolute
    captured={}
    def construct(env,source_sha,reference_actions,*,grasp_styles,cylinder_radius_m,cylinder_length_m):
        captured.update(env=env,sha=source_sha,actions=reference_actions.clone(),styles=grasp_styles,
            radius=cylinder_radius_m,length=cylinder_length_m)
        return captured
    monkeypatch.setitem(sys.modules,'go2_pvcnn.control.dh116.tabletop_vision_teacher_constrained_mpc',
        SimpleNamespace(ConstrainedAcquisitionTeacherReferenceMPC=construct))
    monkeypatch.setattr(absolute,'AbsoluteTeacherReferenceMPC',lambda *args,**kwargs: {'wrong_route':True})
    args=options(vision_teacher_mpc=True,vision_teacher_mpc_profile='pressure_geometry_constrained',
        solid_cylinder_radius_m=.008,object_kind='solid_cylinder');args.device='cpu'
    refs=[dict(actions=torch.full((2+i,24),.1*(i+1)),grasp_style=style)
        for i,style in enumerate(['pinch_thumb_index','tripod_thumb_index_middle'])]
    env=object();result=collection._teacher_optimizer(env,args,{'source_sha256':'a'*64},refs)
    assert result is captured and captured['env'] is env and captured['sha']=='a'*64
    assert captured['radius']==.008 and captured['length']==.5
    assert captured['styles']==[item['grasp_style'] for item in refs]
    assert torch.equal(captured['actions'][0,:2],refs[0]['actions'])
    assert torch.equal(captured['actions'][0,2],torch.zeros(24))
    assert torch.equal(captured['actions'][1],refs[1]['actions'])


def test_raw_chunks_are_bounded_exclusive_and_bind_frame_identity(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import RawRGBDChunks
    sink=RawRGBDChunks(tmp_path/'raw',chunk_frames=2)
    frame=dict(rgb=torch.zeros(2,3,4,3,dtype=torch.uint8),depth=torch.ones(2,3,4,1),
        K=torch.eye(3).expand(2,3,3).clone(),T_world_camera=torch.eye(4).expand(2,4,4).clone(),
        hand_poses=torch.zeros(2,2,7),frame_id=3,simulation_time=.1,
        rendered_simulation_time=.1,physics_step=48)
    sink(frame);frame['rgb'].fill_(99)
    sink(dict(frame,frame_id=4,simulation_time=.2,rendered_simulation_time=.2,physics_step=96))
    assert len(sink.pending)==0
    sink.close()
    manifest=json.loads((tmp_path/'raw/manifest.json').read_text())
    assert manifest['frames']==2 and manifest['chunks'][0]['frame_ids']==[3,4]
    saved=torch.load(tmp_path/'raw/chunk_00000.pt',weights_only=False)
    assert saved[0]['rgb'].sum()==0 and saved[1]['rgb'].max()==99
    with pytest.raises(FileExistsError):RawRGBDChunks(tmp_path/'raw',chunk_frames=2)


def test_raw_chunks_reject_nonmonotonic_and_misaligned_time(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import RawRGBDChunks
    sink=RawRGBDChunks(tmp_path/'raw',chunk_frames=2)
    with pytest.raises(ValueError,match='time'):
        sink(dict(frame_id=1,simulation_time=.1,rendered_simulation_time=.2,physics_step=1))


@pytest.mark.parametrize('early_success,short_reference,preflight_trace',[
    (False,False,False),(True,False,False),(False,True,False),(False,False,True)])
def test_collection_owns_preaction_state_and_issued_labels_without_constructing_ppo(tmp_path,monkeypatch,early_success,short_reference,preflight_trace):
    from test_vision_training import fixture,Camera
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_collection as collection
    from go2_pvcnn.control.dh116 import tabletop_vision_supervision_dataset as dataset
    from go2_pvcnn.control.dh116 import tabletop_vision_training as training
    dw,task,env,args=fixture(tmp_path)
    if preflight_trace:
        dw.surface_trace=[dict(preflight_marker=torch.tensor(1))]
        dw.surface_trace_limit=1;dw.decimation=8
    for key,value in vars(options(num_envs=2)).items():setattr(args,key,value)
    styles=['pinch_thumb_index','tripod_thumb_index_middle']
    refs=[dict(source_episode_id=i*2,grasp_style=style,source_trajectory_id=style,
        actions=torch.full((3,24),.1*(i+1))) for i,style in enumerate(styles)]
    if short_reference:refs[0]['actions']=refs[0]['actions'][:1]
    monkeypatch.setattr(dataset,'load_teacher_reference_dataset',lambda *a:
        dict(source_sha256='a'*64,episodes=refs,references={'x':refs[0]}))
    monkeypatch.setattr(training,'make_squashed_vision_algorithm',lambda *a,**kw:
        pytest.fail('supervised collection must not construct PPO'))
    sim=SimpleNamespace(current_time=0.,current_time_step_index=0)
    class CapturingCamera(Camera):
        def set_raw_frame_sink(self,sink):self.sink=sink
        def sample(self):
            frame=int(sim.current_time_step_index)+1
            self.sink(dict(rgb=torch.zeros(2,2,2,3,dtype=torch.uint8),depth=torch.ones(2,2,2,1),
                frame_id=frame,simulation_time=sim.current_time,rendered_simulation_time=sim.current_time,
                physics_step=sim.current_time_step_index))
            return dict(super().sample(),frame_id=frame,simulation_time=sim.current_time)
    received=[]
    def step(action):
        received.append(action.clone());action.fill_(.99)
        env.tick+=1;sim.current_time+=1./60.;sim.current_time_step_index+=8
        done=torch.tensor([env.tick==1 if early_success else env.tick==3,env.tick==3])
        recovered=torch.tensor([early_success and env.tick==2,False])
        return torch.ones(2,371)*env.tick,torch.zeros(2),done,dict(
            success=done.clone(),failure=torch.zeros(2,dtype=torch.long),recovered=recovered)
    env.step=step
    report=collection.run_teacher_collection(SimpleNamespace(sim=sim),dw,task,env,args,
        {'observation_size':371},camera=CapturingCamera(2))
    assert report['native_runtime_verified'] is False
    assert report['successful_episodes']==0 and report['ppo_updates']==0
    assert report['native_successful_episodes']==0 and report['observed_terminal_successes']==2
    assert not (args.output/'successful_visual_supervision.pt').exists()
    data=torch.load(args.output/'all_executed_supervision.pt',weights_only=False)
    assert all(not ep['native_physics'] for ep in data['episodes'].values())
    assert data['observations'].shape==(4 if early_success or short_reference else 6,3844)
    for row,action in zip(data['sample_metadata'],data['actions']):
        assert torch.equal(action,received[row['tick']][row['env_id']])
        assert row['physics_step']==row['tick']*8
    assert data['sample_metadata'][0]['previous_action']==[0.]*24
    assert next(row for row in data['sample_metadata'] if row['env_id']==1 and row['tick']==1)[
        'previous_action']==pytest.approx([.2]*24)
    assert data['episodes']['0']['recovered'] is False
    assert data['episodes']['0']['terminal']['success'] is True
    assert data['episodes']['0']['terminal']['reference_sequence_complete'] is not short_reference
    if preflight_trace:
        assert len(dw.surface_trace)==1 and int(dw.surface_trace[0]['preflight_marker'])==1
        assert dw.surface_trace_limit>=1+args.vision_teacher_max_control_steps*8
    assert not list(args.output.glob('*ppo.pt'))
