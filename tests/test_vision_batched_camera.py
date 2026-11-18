import importlib
import json
from types import SimpleNamespace
from contextlib import nullcontext

import pytest
import numpy as np
import torch


@pytest.fixture
def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_batched_camera')


class Sim:
    current_time=1.
    current_time_step_index=480
    def __init__(self): self.renders=0
    def render(self): self.renders+=1


class State:
    device='cpu'
    def __init__(self,n):
        self.n=n
        self.origins=torch.zeros(n,3);self.origins[:,0]=torch.arange(n)
        self.roots=torch.zeros(n,2,7);self.roots[:,:,:3]=self.origins[:,None]
        self.roots[:,:,3]=1.
    @property
    def wp(self): raise AssertionError('object pose forbidden')
    @property
    def wq(self): raise AssertionError('object orientation forbidden')
    @property
    def wire(self): raise AssertionError('object access forbidden')


class Reader:
    def __init__(self,sim,origins,width,height):
        self.sim=sim;self.n=len(origins);self.width=width;self.height=height
        self.prim_paths=sorted(f'/World/DH116VisionCameras/camera_{i}' for i in range(self.n))
        self.slots=torch.tensor([int(p.rsplit('_',1)[1]) for p in self.prim_paths])
        self.K=torch.eye(3).repeat(self.n,1,1)
        self.moves=[];self.stale=False;self.old_time=False
        self.move_to_origins(origins)
    def move_to_origins(self,origins):
        self.moves.append(torch.as_tensor(origins).clone())
        self.current=torch.as_tensor(origins)[self.slots]
        self.T_world_camera=torch.eye(4).repeat(self.n,1,1)
        self.T_world_camera[:,:3,3]=self.current
        self.moved_at=self.sim.renders
    def read(self):
        rgb=torch.tensor([160,90,30],dtype=torch.uint8).expand(self.n,self.height,self.width,3).clone()
        z=1.+self.current[:,0]*.0001
        depth=z[:,None,None,None].expand(self.n,self.height,self.width,1).clone()
        return dict(rgb=rgb,depth=depth,frame_id=0 if self.stale else self.sim.renders,
                    simulation_time=self.sim.current_time-(.01 if self.old_time else 0.),
                    time_mapping='test_reader')


def make(module,monkeypatch,tmp_path,n=128,batch_size=64,hide=False,output_device=None,**options):
    state=State(n);sim=Sim();made=[]
    def factory(**kwargs):
        reader=Reader(kwargs['sim'],kwargs['origins'],kwargs['width'],kwargs['height'])
        made.append(reader);return reader
    monkeypatch.setattr(module,'_MovableTiledReader',factory)
    config=dict(position_world_m=[.1,0,1.05],target_world_m=[.48,0,.43],world_up=[0,0,1],
                horizontal_fov_degrees=75,clipping_range_m=[.1,2.],depth_convention='optical_axis_z_m')
    path=tmp_path/'camera.json';path.write_text(json.dumps(config))
    world=SimpleNamespace(sim=sim)
    cam=module.BoundedVisionCamera(world,state,path,width=4,height=4,num_points=3,
        batch_size=batch_size,hide_inactive=hide,output_device=output_device,**options)
    return cam,made,state,sim


@pytest.mark.parametrize('batch_size',[64,256])
def test_4096_envs_use_one_bounded_camera_reader_and_correct_slot_scatter(module,monkeypatch,tmp_path,batch_size):
    cam,made,state,sim=make(module,monkeypatch,tmp_path,n=4096,batch_size=batch_size)
    result=cam.sample()
    assert len(made)==1 and made[0].n==batch_size
    assert result['points'].shape==(4096,2,3,3)
    assert result['camera_count']==batch_size and result['render_batches']==4096//batch_size
    assert torch.allclose(result['points'][:,0,0,2],1.+torch.arange(4096)*.0001,atol=1e-6)
    assert result['point_valid'].all() and result['object_valid'].all()
    assert len(result['batch_frame_ids'])==4096//batch_size
    assert len(set(result['batch_frame_ids']))==4096//batch_size
    assert sim.current_time==1. and sim.current_time_step_index==480
    assert all(len(move)==batch_size for move in made[0].moves)


@pytest.mark.parametrize('n,batch_size',[(70,64),(270,256)])
def test_partial_last_batch_never_overwrites_previous_envs(module,monkeypatch,tmp_path,n,batch_size):
    cam,made,state,sim=make(module,monkeypatch,tmp_path,n=n,batch_size=batch_size)
    result=cam.sample()
    assert result['points'].shape==(n,2,3,3)
    assert result['render_batches']==2
    assert torch.allclose(result['points'][:,0,0,2],1.+torch.arange(n)*.0001,atol=1e-6)
    assert result['batch_environment_ranges']==[[0,batch_size],[batch_size,n]]


def test_256_visibility_pool_supported_and_larger_rejected(module,monkeypatch,tmp_path):
    selected=[]
    selector=SimpleNamespace(select=lambda ids:selected.append(list(ids)))
    world=SimpleNamespace(paths=list(range(4096)),_vision_bounded_visibility=selector)
    assert module.prepare_bounded_visibility(world,batch_size=256) is selector
    assert selected==[list(range(256))]
    with pytest.raises(ValueError,match='batch_size'):
        module.prepare_bounded_visibility(world,batch_size=257)
    with pytest.raises(ValueError,match='batch_size'):
        make(module,monkeypatch,tmp_path,batch_size=257)


def test_second_sample_moves_back_and_uses_current_fk(module,monkeypatch,tmp_path):
    cam,made,state,sim=make(module,monkeypatch,tmp_path,n=128)
    first=cam.sample()
    state.roots[:,:,1]+=2.
    second=cam.sample()
    assert torch.allclose(second['points'][...,1],first['points'][...,1]-2.,atol=1e-6)
    assert second['batch_frame_ids'][0]>first['batch_frame_ids'][-1]
    assert second['rendered_simulation_time']==1.


def test_every_move_discards_at_least_one_render_before_read(module,monkeypatch,tmp_path):
    cam,made,state,sim=make(module,monkeypatch,tmp_path,n=128)
    original=made[0].read
    def checked_read():
        assert sim.renders-made[0].moved_at>=2
        return original()
    made[0].read=checked_read
    cam.sample()


def test_metadata_preserves_actual_render_time_within_alignment_tolerance(module,monkeypatch,tmp_path):
    cam,made,state,sim=make(module,monkeypatch,tmp_path,n=128)
    original=made[0].read
    def rounded_time():
        result=original();result['simulation_time']+=.5e-6;return result
    made[0].read=rounded_time
    result=cam.sample()
    assert result['simulation_time']==1.
    assert result['rendered_simulation_time']==1.+.5e-6
    assert result['batch_rendered_simulation_times']==[1.+.5e-6]*2
    assert result['camera_frame_first']==result['batch_frame_ids'][0]
    assert result['camera_frame_last']==result['batch_frame_ids'][-1]


def test_stale_group_cannot_reuse_previous_groups_frame(module,monkeypatch,tmp_path):
    cam,made,state,sim=make(module,monkeypatch,tmp_path,n=128)
    original=made[0].read
    def stale_second():
        data=original()
        if made[0].current[0,0]>=64:data['frame_id']=0
        return data
    made[0].read=stale_second
    with pytest.raises(RuntimeError,match='fresh'):
        cam.sample()


def test_group_with_old_physics_time_is_rejected(module,monkeypatch,tmp_path):
    cam,made,state,sim=make(module,monkeypatch,tmp_path)
    made[0].old_time=True
    with pytest.raises(RuntimeError,match='time-aligned'):
        cam.sample()


def test_physics_cannot_advance_between_groups(module,monkeypatch,tmp_path):
    cam,made,state,sim=make(module,monkeypatch,tmp_path)
    normal_render=sim.render
    def advancing(): normal_render();sim.current_time+=.01
    sim.render=advancing
    with pytest.raises(RuntimeError,match='physics'):
        cam.sample()


def test_missing_object_returns_zero_mask_without_gt_fallback(module,monkeypatch,tmp_path):
    cam,made,state,sim=make(module,monkeypatch,tmp_path,n=70,output_device='cpu')
    original=made[0].read
    def no_color():
        result=original();result['rgb'].fill_(255);return result
    made[0].read=no_color
    result=cam.sample()
    assert result['image_valid'].all()
    assert not result['object_valid'].any()
    assert not result['point_valid'].any()
    assert torch.equal(result['points'],torch.zeros_like(result['points']))
    assert result['points'].device.type=='cpu'


def test_visibility_sets_local_values_for_all_clones_and_only_toggles_delta(module):
    calls=[]
    class Attr:
        def __init__(self,i):self.i=i
        def Set(self,value):calls.append((self.i,value))
    selector=module.EnvironmentVisibility.__new__(module.EnvironmentVisibility)
    selector._configure([Attr(i) for i in range(8)],change_block=nullcontext,
        visible_token='inherited',hidden_token='invisible')
    assert calls==[(0,'inherited')]+[(i,'invisible') for i in range(1,8)]
    selector.select([0,1]);selector.select([2,3])
    assert calls[-3:]==[(1,'invisible'),(2,'inherited'),(3,'inherited')]
    assert [call for call in calls if call[0]==0]==[(0,'inherited')]
    assert selector._visible=={0,2,3}
    before=len(calls);selector.select([2,3]);assert len(calls)==before
    with pytest.raises(ValueError):selector.select([8])


def test_source_visibility_stays_pinned_through_every_group_and_sample(module,monkeypatch,tmp_path):
    calls=[];values={}
    class Attr:
        def __init__(self,i):self.i=i
        def Set(self,value):calls.append((self.i,value));values[self.i]=value
    selector=module.EnvironmentVisibility.__new__(module.EnvironmentVisibility)
    selector._configure([Attr(i) for i in range(128)],change_block=nullcontext,
        visible_token='inherited',hidden_token='invisible')
    monkeypatch.setattr(module,'prepare_bounded_visibility',lambda world,batch_size:selector)
    cam,_,_,_=make(module,monkeypatch,tmp_path,n=128,hide=True)
    for _ in range(2):
        result=cam.sample()
        assert result['pinned_source_environment']==0
        assert [call for call in calls if call[0]==0]==[(0,'inherited')]
        assert {i for i,value in values.items() if value=='inherited'}=={0,*range(64,128)}


def test_debug_saves_only_requested_first_sample_sensor_data(module,monkeypatch,tmp_path):
    folder=tmp_path/'debug'
    cam,made,state,sim=make(module,monkeypatch,tmp_path,n=70,
        debug_output_dir=folder,debug_env_ids=(3,65))
    result=cam.sample()
    paths=sorted(folder.glob('*.npz'))
    assert [p.name for p in paths]==['env_00003_rgbd.npz','env_00065_rgbd.npz']
    original_bytes=[p.read_bytes() for p in paths]
    for path,env_id in zip(paths,(3,65)):
        with np.load(path,allow_pickle=False) as saved:
            assert set(saved.files)=={'env_id','rgb','depth','K','T_world_camera','hand_poses',
                'hand_pose_source','selected_pixels_uv','points_in_hands','point_valid',
                'image_valid','object_valid','frame_id','simulation_time'}
            assert saved['rgb'].shape==(4,4,3) and saved['rgb'].dtype==np.uint8
            assert np.allclose(saved['depth'],1.+env_id*.0001)
            assert np.array_equal(saved['hand_poses'],state.roots[env_id].numpy())
            assert np.array_equal(saved['points_in_hands'],result['points'][env_id].numpy())
            assert np.array_equal(saved['selected_pixels_uv'],result['selected_pixels_uv'][env_id].numpy())
            assert saved['hand_pose_source'].item()=='current_palm_encoder_FK_simulation_standin'
    state.roots[:,:,1]+=1.
    cam.sample()
    assert [p.read_bytes() for p in paths]==original_bytes
    assert len(result['debug_saved_files'])==2


def test_debug_is_disabled_by_default_and_bounded_to_two_valid_envs(module,monkeypatch,tmp_path):
    cam,_,_,_=make(module,monkeypatch,tmp_path,n=70)
    result=cam.sample()
    assert result['debug_saved_files']==[]
    assert not list(tmp_path.rglob('*.npz'))
    for ids in ((0,1,2),(70,),(-1,),(1,1)):
        with pytest.raises(ValueError,match='debug'):
            make(module,monkeypatch,tmp_path,n=70,debug_output_dir=tmp_path/'debug',debug_env_ids=ids)


def test_batch_wall_seconds_records_each_complete_batch(module,monkeypatch,tmp_path):
    cam,_,_,_=make(module,monkeypatch,tmp_path,n=70)
    times=iter((10.,10.25,11.,11.75))
    monkeypatch.setattr(module,'perf_counter',lambda:next(times))
    result=cam.sample()
    assert result['batch_wall_seconds']==[.25,.75]
