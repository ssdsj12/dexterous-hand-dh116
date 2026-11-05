import importlib
import json
from types import SimpleNamespace

import pytest
import numpy as np
import torch


@pytest.fixture
def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_fixed_camera')


class Sim:
    current_time=.125
    current_time_step_index=60
    def __init__(self):self.renders=0
    def render(self):self.renders+=1


class State:
    device='cpu'
    def __init__(self,n):
        self.n=n;self.origins=torch.zeros(n,3);self.origins[:,0]=torch.arange(n)
        self.roots=torch.zeros(n,2,7);self.roots[:,:,:3]=self.origins[:,None];self.roots[...,3]=1.
    @property
    def wp(self):raise AssertionError('object GT must never be read')
    @property
    def wq(self):raise AssertionError('object GT must never be read')


class Reader:
    def __init__(self,sim,origins,width,height):
        self.sim=sim;self.n=len(origins);self.width=width;self.height=height
        self.prim_paths=sorted(f'/World/DH116VisionCameras/camera_{i}' for i in range(self.n))
        self.slots=torch.tensor([int(p.rsplit('_',1)[1]) for p in self.prim_paths])
        self.K=torch.eye(3).repeat(self.n,1,1)
        self.T_world_camera=torch.eye(4).repeat(self.n,1,1)
        self.T_world_camera[:,:3,3]=origins[self.slots]
        self._calibration_checked=False;self.calibration_reads=0;self.read_calls=0
        self.frame_override=None;self.time_offset=0.
    def move_to_origins(self,*args):raise AssertionError('fixed cameras must never move')
    def read(self):
        self.read_calls+=1
        if not self._calibration_checked:
            self.calibration_reads+=1;self._calibration_checked=True
        rgb=torch.tensor([160,90,30],dtype=torch.uint8).expand(self.n,self.height,self.width,3).clone()
        depth=(.8+self.slots*.0001)[:,None,None,None].expand(self.n,self.height,self.width,1).clone()
        return dict(rgb=rgb,depth=depth,frame_id=self.sim.renders if self.frame_override is None else self.frame_override,
            simulation_time=self.sim.current_time+self.time_offset,time_mapping='same_render_product_IsaacReadTimes')


def make(module,monkeypatch,tmp_path,n=2,**options):
    state=State(n);sim=Sim();created=[]
    def factory(**kw):
        reader=Reader(kw['sim'],kw['origins'],kw['width'],kw['height']);created.append(reader);return reader
    monkeypatch.setattr(module,'_NativeTiledReader',factory)
    config=dict(position_world_m=[.1,0,1.05],target_world_m=[.48,0,.43],world_up=[0,0,1],
        horizontal_fov_degrees=75,clipping_range_m=[.1,2.],depth_convention='optical_axis_z_m')
    path=tmp_path/'camera.json';path.write_text(json.dumps(config))
    # No paths or stage: the fixed provider must not touch environment visibility.
    cam=module.FixedVisionCamera(SimpleNamespace(sim=sim),state,path,width=9,height=7,num_points=8,**options)
    return cam,created,state,sim


@pytest.mark.parametrize('n',[2,1024,2048])
def test_one_fixed_reader_captures_all_envs_without_movement_or_group_loop(module,monkeypatch,tmp_path,n):
    cam,created,state,sim=make(module,monkeypatch,tmp_path,n=n)
    packet=cam.sample()
    assert len(created)==1 and created[0].n==n
    assert sim.renders==8 and created[0].read_calls==1
    assert packet['points'].shape==(n,2,8,3)
    assert torch.allclose(packet['points'][:,0,0,2],.8+torch.arange(n)*.0001,atol=1e-6)
    assert packet['point_valid'].all() and packet['image_valid'].all()
    assert packet['camera_count']==n and packet['render_batches']==1
    assert packet['camera_mode']=='fixed_per_env'
    assert packet['batch_environment_ranges']==[[0,n]]
    assert packet['batch_frame_ids']==[8] and packet['physics_frozen']
    assert packet['pinned_source_environment'] is None
    assert not packet['inactive_environment_visibility']


def test_steady_observation_uses_one_fresh_render_and_cached_calibration(module,monkeypatch,tmp_path):
    cam,created,state,sim=make(module,monkeypatch,tmp_path,n=12)
    first=cam.sample();sim.current_time+=1./60.;sim.current_time_step_index+=8
    state.roots[:,:,1]+=.025
    second=cam.sample()
    assert sim.renders==9 and second['render_passes']==1
    assert second['camera_frame_first']==second['camera_frame_last']==9
    assert second['rendered_simulation_time']==sim.current_time
    assert created[0].calibration_reads==1
    assert torch.allclose(second['points'][...,1],first['points'][...,1]-.025,atol=1e-6)


@pytest.mark.parametrize('failure',['stale_frame','old_time','nan_time','wrong_shape','metadata_missing'])
def test_unusable_frames_are_rejected_after_at_most_eight_steady_renders(module,monkeypatch,tmp_path,failure):
    cam,created,state,sim=make(module,monkeypatch,tmp_path)
    first=cam.sample();reader=created[0];original=reader.read
    def bad():
        result=original()
        if failure=='stale_frame':result['frame_id']=first['frame_id']
        elif failure=='old_time':result['simulation_time']-=.02
        elif failure=='nan_time':result['simulation_time']=float('nan')
        elif failure=='wrong_shape':result['rgb']=result['rgb'][:1]
        else:raise ValueError('same render product time metadata mismatch')
        return result
    reader.read=bad
    with pytest.raises(RuntimeError,match='fresh.*time-aligned'):
        cam.sample()
    assert sim.renders==16


def test_temporarily_old_metadata_retries_then_records_actual_time(module,monkeypatch,tmp_path):
    cam,created,state,sim=make(module,monkeypatch,tmp_path)
    cam.sample();reader=created[0];original=reader.read
    def delayed():
        result=original();result['simulation_time']+=(-.02 if sim.renders==9 else .5e-6);return result
    reader.read=delayed
    packet=cam.sample()
    assert packet['render_passes']==2 and packet['frame_id']==10
    assert packet['simulation_time']==sim.current_time
    assert packet['rendered_simulation_time']==sim.current_time+.5e-6


@pytest.mark.parametrize('change',['time','step','FK'])
def test_render_cannot_advance_physics_or_change_fk(module,monkeypatch,tmp_path,change):
    cam,created,state,sim=make(module,monkeypatch,tmp_path)
    original=sim.render
    def changed():
        original()
        if change=='time':sim.current_time+=.01
        elif change=='step':sim.current_time_step_index+=1
        else:state.roots[:,:,0]+=.01
    sim.render=changed
    with pytest.raises(RuntimeError,match='physics|FK'):
        cam.sample()


@pytest.mark.parametrize('n',[0,1,3,2049,4096])
def test_only_even_environment_counts_two_through_2048_are_allowed(module,monkeypatch,tmp_path,n):
    with pytest.raises(ValueError,match='even.*2048'):
        make(module,monkeypatch,tmp_path,n=n)


def test_debug_captures_only_sensor_data_once_for_two_requested_envs(module,monkeypatch,tmp_path):
    folder=tmp_path/'debug'
    cam,_,state,sim=make(module,monkeypatch,tmp_path,n=12,debug_output_dir=folder,debug_env_ids=(0,6))
    packet=cam.sample()
    files=sorted(folder.glob('*.npz'))
    assert [f.name for f in files]==['env_00000_rgbd.npz','env_00006_rgbd.npz']
    original=[f.read_bytes() for f in files]
    for path,env_id in zip(files,(0,6)):
        with np.load(path,allow_pickle=False) as data:
            assert set(data.files)=={'env_id','rgb','depth','K','T_world_camera','hand_poses',
                'hand_pose_source','selected_pixels_uv','points_in_hands','point_valid',
                'image_valid','object_valid','frame_id','simulation_time'}
            assert data['rgb'].shape==(7,9,3) and data['rgb'].dtype==np.uint8
            assert np.allclose(data['depth'],.8+env_id*.0001)
            assert np.array_equal(data['hand_poses'],state.roots[env_id].numpy())
            assert np.array_equal(data['points_in_hands'],packet['points'][env_id].numpy())
            assert np.array_equal(data['selected_pixels_uv'],packet['selected_pixels_uv'][env_id].numpy())
            assert data['hand_pose_source'].item()=='current_palm_encoder_FK_simulation_standin'
    assert packet['debug_saved_files']==[str(f) for f in files]
    state.roots[:,:,1]+=.025;sim.current_time+=1./60.;sim.current_time_step_index+=8
    cam.sample()
    assert [f.read_bytes() for f in files]==original


def test_debug_defaults_to_no_writes_and_rejects_invalid_ids(module,monkeypatch,tmp_path):
    cam,_,_,_=make(module,monkeypatch,tmp_path,n=12)
    assert cam.sample()['debug_saved_files']==[]
    assert not list(tmp_path.rglob('*.npz'))
    for ids in ((0,1,2),(12,),(-1,),(1,1)):
        with pytest.raises(ValueError,match='debug'):
            make(module,monkeypatch,tmp_path,n=12,debug_output_dir=tmp_path/'debug',debug_env_ids=ids)
