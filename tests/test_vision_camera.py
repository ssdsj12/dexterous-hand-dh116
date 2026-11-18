import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_camera')


def test_numeric_slot_mapping_is_verified_and_reordered(module):
    paths = [f'/World/Vision/camera_{i}' for i in (0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9)]
    assert module.numeric_slot_order(paths, 12).tolist() == [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3]
    with pytest.raises(ValueError):
        module.numeric_slot_order(paths[:-1]+[paths[0]], 12)


def test_rectangular_tiled_frames_keep_every_pixel(module):
    expected = torch.arange(4*2*3*3).reshape(4, 2, 3, 3)
    tiled = expected.reshape(2, 2, 2, 3, 3).permute(0, 2, 1, 3, 4).reshape(4, 6, 3)
    assert torch.equal(module.detile(tiled, count=4, width=3, height=2, columns=2, channels=3), expected)
    # The installed CameraView depth helper reshapes to (tile_width,tile_height).
    depth = tiled[..., :1].contiguous().reshape(6, 4, 1)
    assert torch.equal(module.detile(depth, count=4, width=3, height=2, columns=2, channels=1), expected[..., :1])


def test_gpu_brown_mask_matches_existing_visible_rgb_rule(module):
    from go2_pvcnn.control.dh116.tabletop_depth_camera_probe import brown_rod_mask
    rng = np.random.default_rng(123)
    rgb = rng.integers(0, 256, (2, 7, 9, 3), dtype=np.uint8)
    result = module.brown_mask(torch.tensor(rgb))
    assert np.array_equal(result.numpy(), np.stack([brown_rod_mask(x) for x in rgb]))


def test_camera_axes_override_catches_parent_constructor_and_preserves_device_dtype(module):
    recorded={}
    q=torch.tensor([[2**-.5,0.,0.,2**-.5]],dtype=torch.float64)
    transform=np.array([[0.,0.,1.,0.],[0.,1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,0.,1.]])
    def matrices_to_quats(matrix,device=None):
        recorded['matrix']=matrix;recorded['device']=device
        # Installed backend returns float32; the adapter must restore input dtype.
        return torch.tensor([[1.,0.,0.,0.]],device=device,dtype=torch.float32)
    class Parent:
        def __init__(self):
            self._backend_utils=SimpleNamespace(quats_to_rot_matrices=module._quaternion_matrices,
                rot_matrices_to_quats=matrices_to_quats)
            self.result=self._convert_camera_axes(q,transform)
        def _convert_camera_axes(self,*args):
            raise AssertionError('unsafe parent conversion called')
    class Safe(module._CameraAxesDeviceMixin,Parent):
        pass
    view=Safe()
    assert view.result.dtype==q.dtype and view.result.device==q.device
    assert recorded['device']==q.device
    expected=torch.as_tensor(transform[:3,:3],dtype=q.dtype)@module._quaternion_matrices(q)
    assert torch.allclose(recorded['matrix'],expected)


def test_camera_axes_numpy_uses_original_backend(module):
    sentinel=object()
    class Parent:
        def _convert_camera_axes(self,orientations,transform): return sentinel
    class Safe(module._CameraAxesDeviceMixin,Parent):
        pass
    assert Safe()._convert_camera_axes(np.array([[1.,0.,0.,0.]]),np.eye(4)) is sentinel


def test_same_render_product_time_uses_aov_not_legacy_mapping(module):
    reference=dict(referenceTimeNumerator=50,referenceTimeDenominator=30)
    times=dict(swhFrameNumber=1,frameNumber=50,simulationTime=.03750000195577741,
               simulationTimeMonotonic=.03750000195577741,rationalTimeOfSimNumerator=50,rationalTimeOfSimDenominator=30)
    result=module.paired_render_timestamp(reference,times,legacy_time=0.)
    assert result['simulation_time']==pytest.approx(.0375)
    assert result['legacy_simulation_time']==0.
    assert result['frame_id']==50 and result['render_frame_number']==50
    assert result['swh_frame_number']==1
    assert result['time_mapping']=='same_render_product_IsaacReadTimes'


def test_different_render_frame_time_cannot_be_paired(module):
    with pytest.raises(ValueError,match='frame'):
        module.paired_render_timestamp(dict(referenceTimeNumerator=50,referenceTimeDenominator=30),
            dict(swhFrameNumber=1,frameNumber=49,simulationTime=.0375,
                 rationalTimeOfSimNumerator=49,rationalTimeOfSimDenominator=30),legacy_time=.0375)


def test_render_reference_denominator_must_match_exactly(module):
    with pytest.raises(ValueError,match='frame'):
        module.paired_render_timestamp(dict(referenceTimeNumerator=50,referenceTimeDenominator=30),
            dict(swhFrameNumber=1,frameNumber=50,simulationTime=.0375,
                 rationalTimeOfSimNumerator=100,rationalTimeOfSimDenominator=60),legacy_time=.0375)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.])
def test_invalid_aov_time_is_rejected_even_when_legacy_matches(module,value):
    with pytest.raises(ValueError,match='time'):
        module.paired_render_timestamp(dict(referenceTimeNumerator=50,referenceTimeDenominator=1),
            dict(swhFrameNumber=1,frameNumber=50,simulationTime=value,
                 rationalTimeOfSimNumerator=50,rationalTimeOfSimDenominator=1),legacy_time=.0375)


def fixture_frames(count=2, height=2, width=4):
    rgb = torch.tensor([160, 90, 30], dtype=torch.uint8).expand(count, height, width, 3).clone()
    depth = torch.ones((count, height, width, 1))
    K = torch.eye(3).repeat(count, 1, 1)
    T = torch.eye(4).repeat(count, 1, 1)
    hands = torch.tensor([[0.,0.,0.,1.,0.,0.,0.], [1.,2.,3.,1.,0.,0.,0.]]).repeat(count,1,1)
    return rgb, depth, K, T, hands


def test_visible_points_sample_uniform_pixels_and_apply_both_hand_frames(module):
    args = fixture_frames()
    result = module.frames_to_points(*args, num_points=3, min_depth_m=.1, max_depth_m=2.,depth_edge_threshold_m=None)
    assert result['points'].shape == (2, 2, 3, 3)
    assert result['point_valid'].all()
    assert torch.equal(result['selected_pixels_uv'][0], torch.tensor([[0,0],[3,0],[3,1]]))
    assert torch.allclose(result['points'][0,0], torch.tensor([[.5,.5,1.],[3.5,.5,1.],[3.5,1.5,1.]]))
    assert torch.allclose(result['points'][0,1], result['points'][0,0]-torch.tensor([1.,2.,3.]))


def test_missing_points_zero_pad_and_invalid_images_never_use_gt(module):
    rgb, depth, K, T, hands = fixture_frames()
    rgb[0] = 255
    rgb[0,0,2] = torch.tensor([160,90,30], dtype=torch.uint8)
    depth[1] = float('inf')
    result = module.frames_to_points(rgb,depth,K,T,hands,num_points=3,min_depth_m=.1,max_depth_m=2.,depth_edge_threshold_m=None)
    assert result['point_valid'].tolist() == [[True,False,False],[False,False,False]]
    assert result['image_valid'].tolist() == [True,False]
    assert result['object_valid'].tolist() == [True,False]
    assert torch.equal(result['points'][0,:,1:], torch.zeros(2,2,3))
    assert torch.equal(result['points'][1], torch.zeros(2,3,3))
    assert torch.isfinite(result['points']).all()


def test_depth_clipping_and_negative_values_are_rejected(module):
    rgb, depth, K, T, hands = fixture_frames(count=1)
    depth.flatten()[:] = torch.tensor([0., -.5, .09, .1, 2., 2.1, float('nan'), float('inf')])
    result = module.frames_to_points(rgb,depth,K,T,hands,num_points=4,min_depth_m=.1,max_depth_m=2.,depth_edge_threshold_m=None)
    assert result['visible_point_count'].tolist() == [2]
    assert result['depth_valid_fraction'].item() == pytest.approx(.25)


def test_64_environments_keep_individual_depth_and_optical_rotation(module):
    rgb,depth,K,T,hands=fixture_frames(count=64)
    depth[:]=torch.linspace(.2,1.9,64)[:,None,None,None]
    # Camera +Z points toward world +X, preserving a proper rotation.
    T[:,:3,:3]=torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    T[:,:3,3]=torch.tensor([4.,5.,6.])
    result=module.frames_to_points(rgb,depth,K,T,hands,num_points=3,min_depth_m=.1,max_depth_m=2.,depth_edge_threshold_m=None)
    assert result['points'].shape==(64,2,3,3)
    assert torch.allclose(result['points'][:,0,0,0],4.+torch.linspace(.2,1.9,64))
    z=torch.linspace(.2,1.9,64)
    assert torch.allclose(result['points'][:,0,0,1:],torch.stack((5.+.5*z,6.-.5*z),dim=1))


def test_palm_rotation_uses_inverse_orientation(module):
    rgb,depth,K,T,hands=fixture_frames(count=1)
    hands[0,0,3:]=torch.tensor([2**-.5,0.,0.,2**-.5])
    result=module.frames_to_points(rgb,depth,K,T,hands,num_points=3,min_depth_m=.1,max_depth_m=2.,depth_edge_threshold_m=None)
    assert torch.allclose(result['points'][0,0,1],torch.tensor([.5,-3.5,1.]),atol=1e-6)


class Sim:
    current_time = 1.
    def __init__(self): self.renders = 0
    def render(self): self.renders += 1


class WorldState:
    n = 2
    device = 'cpu'
    origins = torch.zeros(2,3)
    roots = fixture_frames()[-1]
    @property
    def wp(self): raise AssertionError('forbidden object GT access')
    @property
    def wq(self): raise AssertionError('forbidden object GT access')
    @property
    def wire(self): raise AssertionError('forbidden object GT access')


class Reader:
    def __init__(self, sim):
        self.sim = sim
        self.prim_paths = ['/World/Vision/camera_1', '/World/Vision/camera_0']
        rgb, depth, self.K, self.T_world_camera, _ = fixture_frames(height=4)
        depth[0] *= 1.5
        self.rgb, self.depth = rgb, depth
        self.stale = False
    def read(self):
        return dict(rgb=self.rgb, depth=self.depth, frame_id=0 if self.stale else self.sim.renders,
                    simulation_time=self.sim.current_time)


def provider(module, monkeypatch, tmp_path):
    sim = Sim(); reader = Reader(sim)
    config = dict(position_world_m=[.1,0,1.05], target_world_m=[.48,0,.43], world_up=[0,0,1],
                  horizontal_fov_degrees=75, clipping_range_m=[.1,2.], depth_convention='optical_axis_z_m')
    path = tmp_path/'camera.json'; path.write_text(json.dumps(config))
    monkeypatch.setattr(module, '_NativeTiledReader', lambda **kwargs: reader)
    cam = module.VisionCamera(SimpleNamespace(sim=sim), WorldState(), path, width=4,height=4,num_points=3)
    return cam, reader, sim


def test_sample_reads_actual_slot_order_and_current_fk_only(module, monkeypatch, tmp_path):
    cam, reader, sim = provider(module,monkeypatch,tmp_path)
    first = cam.sample()
    assert first['points'][0,0,0,2].item() == pytest.approx(1.)
    assert first['points'][1,0,0,2].item() == pytest.approx(1.5)
    moved = WorldState.roots.clone(); moved[...,0] += 2.
    second = cam.sample(hand_poses=moved)
    assert torch.allclose(second['points'][...,:1], first['points'][...,:1]-2.)
    assert sim.current_time == 1.
    assert sim.renders >= 2


def test_sample_rejects_stale_frame_instead_of_reusing_previous_image(module, monkeypatch, tmp_path):
    cam, reader, sim = provider(module,monkeypatch,tmp_path)
    cam.sample()
    reader.stale = True
    with pytest.raises(RuntimeError, match='fresh|stale'):
        cam.sample()


def test_sample_rejects_render_that_advances_physics(module, monkeypatch, tmp_path):
    cam, reader, sim = provider(module,monkeypatch,tmp_path)
    def bad_render(): sim.current_time += .01; sim.renders += 1
    sim.render = bad_render
    with pytest.raises(RuntimeError, match='physics|time'):
        cam.sample()


def test_sample_rejects_new_render_frame_with_old_physics_timestamp(module, monkeypatch, tmp_path):
    cam,reader,sim=provider(module,monkeypatch,tmp_path)
    original=reader.read
    def old_time():
        frame=original();frame['simulation_time']-=1./60.
        return frame
    reader.read=old_time
    with pytest.raises(RuntimeError,match='time-aligned'):
        cam.sample()


def test_same_frames_and_fk_are_invariant_to_unrelated_gt(module, monkeypatch, tmp_path):
    cam, reader, sim = provider(module,monkeypatch,tmp_path)
    before = cam.sample()
    cam._state.unrelated_object_state = torch.randn(2,100)*1000.
    after = cam.sample()
    for key in ('points','point_valid','image_valid','object_valid'):
        assert torch.equal(before[key], after[key])


def test_provider_accepts_320_by_240_and_retains_160_default(module,monkeypatch,tmp_path):
    import inspect
    sim=Sim();reader=Reader(sim)
    config=dict(position_world_m=[.1,0,1.05],target_world_m=[.48,0,.43],world_up=[0,0,1],
                horizontal_fov_degrees=75,clipping_range_m=[.1,2.],depth_convention='optical_axis_z_m')
    path=tmp_path/'camera.json';path.write_text(json.dumps(config))
    passed=[]
    def factory(**kwargs): passed.append(kwargs);return reader
    monkeypatch.setattr(module,'_NativeTiledReader',factory)
    cam=module.VisionCamera(SimpleNamespace(sim=sim),WorldState(),path,width=320,height=240)
    assert (cam.width,cam.height)==(320,240)
    assert (passed[0]['width'],passed[0]['height'])==(320,240)
    signature=inspect.signature(module.VisionCamera)
    assert signature.parameters['width'].default==160
    assert signature.parameters['height'].default==120
    with pytest.raises(ValueError,match='320x240'):
        module.VisionCamera(SimpleNamespace(sim=sim),WorldState(),path,width=321,height=240)
    with pytest.raises(ValueError,match='320x240'):
        module.VisionCamera(SimpleNamespace(sim=sim),WorldState(),path,width=320,height=241)


def test_depth_edges_remove_brown_table_mix_and_keep_rod_interior(module):
    rgb,depth,K,T,hands=fixture_frames(count=1,height=9,width=9)
    rgb[:]=255;depth[:]=.9
    rgb[:,3:7,2:7]=torch.tensor([160,90,30],dtype=torch.uint8)
    depth[:,3:7,2:7]=.7
    # This brown RGB sample is actually table depth at the silhouette.
    rgb[:,3,1]=torch.tensor([160,90,30],dtype=torch.uint8)
    result=module.frames_to_points(rgb,depth,K,T,hands,num_points=16,min_depth_m=.1,max_depth_m=2.)
    assert result['raw_brown_point_count'].tolist()==[21]
    assert result['visible_point_count'].tolist()==[6]
    assert result['edge_rejected_point_count'].tolist()==[15]
    pixels=result['selected_pixels_uv'][0,result['point_valid'][0]]
    assert all(3<=int(u)<=5 and 4<=int(v)<=5 for u,v in pixels)


def test_invalid_neighbor_depth_rejects_center_pixel(module):
    rgb,depth,K,T,hands=fixture_frames(count=1,height=3,width=3)
    depth[0,0,0]=float('nan')
    result=module.frames_to_points(rgb,depth,K,T,hands,num_points=1,min_depth_m=.1,max_depth_m=2.)
    assert not result['point_valid'].any()
    assert not result['object_valid'].any()


def test_rtx_pixel_centers_match_optical_axis_at_principal_point(module):
    rgb,depth,K,T,hands=fixture_frames(count=1,height=3,width=3)
    K[0]=torch.tensor([[100.,0.,1.5],[0.,100.,1.5],[0.,0.,1.]])
    result=module.frames_to_points(rgb,depth,K,T,hands,num_points=1,min_depth_m=.1,max_depth_m=2.)
    assert result['selected_pixels_uv'][0,0].tolist()==[1,1]
    assert torch.allclose(result['points'][0,0,0],torch.tensor([0.,0.,1.]))
