"""Independent synthetic CPU stereo checks; never native/hardware evidence."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth
from go2_pvcnn.control.dh116.tabletop_stereo_camera import StereoCamera
from go2_pvcnn.control.dh116.tabletop_stereo_geometry import backproject, transform_points, world_to_palms, order_polyline


def calibration(width=64, height=32, focal=40., baseline=.08):
    k=torch.tensor([[focal,0.,width/2],[0.,focal,height/2],[0.,0.,1.]],dtype=torch.float64)
    left=torch.eye(4,dtype=torch.float64)
    right=left.clone();right[0,3]=baseline
    return StereoCalibration(k,k.clone(),left,right,width,height,baseline)


def pair(disparity=4):
    g=torch.Generator().manual_seed(54)
    left=torch.rand(1,3,32,64,generator=g)
    right=torch.zeros_like(left)
    right[...,:-disparity]=left[...,disparity:]
    return left,right


def identity():
    return dict(run_uuid='cpu_synthetic',env_id=0,episode_uid='episode0',reset_id=0,
                policy_tick=0,physics_tick_start=0,physics_tick_end=0,simulation_timestamp=.1)


def reader_packet(c, left=None, right=None):
    if left is None:left,right=pair()
    return dict(left_rgb=left,right_rgb=right,left_frame_id=7,right_frame_id=7,
                left_simulation_time=.1,right_simulation_time=.1,
                K_left=c.K_left,K_right=c.K_right,T_world_left=c.T_world_left,
                T_world_right=c.T_world_right,native_evidence=False)


class Reader:
    def __init__(self, packet):self.packet=packet
    def read(self):return self.packet


def test_calibration_digest_and_exact_readback():
    c=calibration()
    assert len(c.sha256)==64
    c.validate_readback(c.K_left,c.K_right,c.T_world_left,c.T_world_right)
    wrong=c.K_left.clone();wrong[0,0]+=1
    with pytest.raises(ValueError,match='calibration|intrinsic'):
        c.validate_readback(wrong,c.K_right,c.T_world_left,c.T_world_right)
    assert calibration(focal=41.).sha256!=c.sha256


@pytest.mark.parametrize('mutation',['baseline','rotation','intrinsic','vertical','size'])
def test_invalid_calibration_rejected(mutation):
    c=calibration();kw=c.to_dict()
    if mutation=='baseline':kw['baseline_m']=.1
    if mutation=='rotation':kw['T_world_left'][0][0]=2.
    if mutation=='intrinsic':kw['K_left'][0][0]=0.
    if mutation=='vertical':kw['T_world_right'][1][3]=.01
    if mutation=='size':kw['width']=0
    with pytest.raises(ValueError):StereoCalibration.from_dict(kw)


def test_default_config_geometry_explains_disparity_budget():
    root=Path(__file__).resolve().parents[1]
    path=root/'configs/dh116_training/stereo_camera_v1.json'
    c=StereoCalibration.from_json(path)
    assert (c.width,c.height)==(224,96)
    assert math.isclose(c.baseline_m,.08)
    fx=float(c.K_left[0,0])
    assert 14.<fx*c.baseline_m/.8<16.
    assert 23.<fx*c.baseline_m/.5<24.
    assert c.sha256==StereoCalibration.from_dict(c.to_dict()).sha256


def test_real_blockmatch_known_disparity_without_depth_input():
    c=calibration();left,right=pair()
    result=StereoDepth(max_disparity=8,block_size=3,min_depth_m=.2,max_depth_m=2.).compute(left,right,c)
    inner=result['valid'][:,4:-4,12:-12]
    assert inner.float().mean()>.85
    disparity=result['disparity_px'][:,4:-4,12:-12][inner]
    torch.testing.assert_close(disparity,torch.full_like(disparity,4.))
    depth=result['depth_m'][:,4:-4,12:-12][inner]
    torch.testing.assert_close(depth,torch.full_like(depth,.8))
    assert all(v.shape==(1,32,64) for v in result.values())
    assert result['valid'].dtype==torch.bool
    assert torch.isnan(result['depth_m'][~result['valid']]).all()


def test_no_texture_and_far_depth_are_masked_without_gt_fallback():
    c=calibration();blank=torch.ones(1,3,32,64)*.5
    result=StereoDepth(max_disparity=8).compute(blank,blank,c)
    assert not result['valid'].any()
    left,right=pair()
    result=StereoDepth(max_disparity=8,max_depth_m=.5).compute(left,right,c)
    assert not result['valid'][:,4:-4,12:-12].any()


def test_occluded_and_image_boundary_pixels_excluded():
    c=calibration();left,right=pair()
    right[...,22:29]=0.
    result=StereoDepth(max_disparity=8,block_size=3).compute(left,right,c)
    assert not result['valid'][...,:3].any()
    assert result['occlusion'][...,26:32].float().mean()>.25
    assert result['valid'][...,26:32].float().mean()<.5


@pytest.mark.parametrize('failure',['shape','range','nan','unrectified'])
def test_depth_schema_and_unrectified_images_fail_closed(failure):
    c=calibration();left,right=pair()
    if failure=='shape':right=right[:,:,:-1]
    if failure=='range':left[0,0,0,0]=255.
    if failure=='nan':left[0,0,0,0]=torch.nan
    if failure=='unrectified':
        d=c.to_dict();d['rectified']=False;c=StereoCalibration.from_dict(d)
    with pytest.raises((ValueError,RuntimeError)):
        StereoDepth(max_disparity=8).compute(left,right,c)


def test_optional_sgbm_missing_dependency_reports_it_explicitly():
    if importlib.util.find_spec('cv2') is not None:pytest.skip('cv2 available; covered by optional native/tool environment')
    with pytest.raises(RuntimeError,match='OpenCV|cv2'):
        StereoDepth(method='sgbm').compute(*pair(),calibration())


def test_backproject_world_and_actual_palm_round_trip_with_nan_mask():
    c=calibration(width=2,height=2,focal=2.)
    depth=torch.ones(1,2,2)
    valid=torch.ones_like(depth,dtype=torch.bool);valid[0,0,0]=False
    points=backproject(depth,c.K_left,valid)
    assert torch.isnan(points[0,0,0]).all()
    torch.testing.assert_close(points[0,1,1],torch.tensor([.25,.25,1.],dtype=points.dtype))
    T=torch.eye(4);T[0,3]=1.
    world=transform_points(points,T)
    pose=torch.tensor([[[1.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,math.sqrt(.5),0.,0.,math.sqrt(.5)]]])
    hands=world_to_palms(world,pose)
    torch.testing.assert_close(hands[:,0,1,1],points[:,1,1])
    torch.testing.assert_close(hands[:,1,1,1],torch.tensor([[.25,-1.25,1.]],dtype=hands.dtype),atol=1e-6,rtol=0.)


def test_folded_polyline_follows_explicit_sensor_topology_and_keeps_mask():
    p=torch.tensor([[[0.,0.,1.],[1.,1.,1.],[1.,0.,1.],[0.,1.,1.],[float('nan')]*3]])
    valid=torch.tensor([[True,True,True,True,False]])
    adjacency=torch.zeros(1,5,5,dtype=torch.bool)
    for a,b in [(0,2),(2,1),(1,3)]:adjacency[0,a,b]=adjacency[0,b,a]=True
    r=order_polyline(p,valid,adjacency)
    assert r['order'][0,:4].tolist()==[0,2,1,3]
    torch.testing.assert_close(r['arc_length_m'][0,:4],torch.arange(4,dtype=r['arc_length_m'].dtype))
    assert not r['valid'][0,4] and torch.isnan(r['points'][0,4]).all()
    adjacency[0,0,1]=adjacency[0,1,0]=True
    with pytest.raises(ValueError,match='topology|branch|path'):
        order_polyline(p,valid,adjacency)


def test_cpu_camera_identity_readback_and_stereo_depth_are_explicit():
    c=calibration();packet=reader_packet(c)
    camera=StereoCamera(c,Reader(packet),depth_estimator=StereoDepth(max_disparity=8))
    result=camera.sample(identity=identity())
    assert result['frame_id']==7 and result['simulation_time']==.1
    assert {key:result['identity'][key] for key in identity()}==identity()
    assert result['identity']['frame_id']==7
    assert result['identity']['calibration_sha256']==c.sha256
    assert result['native_evidence'] is False and result['calibration_sha256']==c.sha256
    assert result['left_rgb'].shape==(1,3,32,64)
    assert 'depth_m' in result and 'depth' not in result
    with pytest.raises(RuntimeError,match='fresh|stale'):
        camera.sample(identity=identity())


@pytest.mark.parametrize('fault',['timestamp','frame','readback','caller_time','identity','rendered_depth'])
def test_stereo_camera_rejects_mismatch_without_physics_or_truth(fault):
    c=calibration();packet=reader_packet(c);ident=identity()
    if fault=='timestamp':packet['right_simulation_time']=.11
    if fault=='frame':packet['right_frame_id']=8
    if fault=='readback':packet['K_right']=c.K_right.clone();packet['K_right'][0,0]+=1
    if fault=='caller_time':ident['simulation_timestamp']=.2
    if fault=='identity':del ident['reset_id']
    if fault=='rendered_depth':packet['depth']=torch.ones(1,32,64)
    with pytest.raises((ValueError,RuntimeError)):
        StereoCamera(c,Reader(packet)).sample(identity=ident)


def test_import_does_not_load_native_modules_and_camera_never_steps_physics():
    import sys
    assert not any(k.startswith(('omni.','isaaclab.','isaacsim.')) for k in sys.modules)
    import go2_pvcnn.control.dh116.tabletop_stereo_camera as m
    source=Path(m.__file__).read_text(encoding='utf-8')
    assert '.step(' not in source
    assert "output_annotators=['rgb']" in source


def test_list_identity_is_preserved_per_environment_and_camera_stamped():
    c=calibration();packet=reader_packet(c)
    packet['left_rgb']=packet['left_rgb'].repeat(2,1,1,1)
    packet['right_rgb']=packet['right_rgb'].repeat(2,1,1,1)
    first=identity();first.update(env_id=0,episode_uid='a',reset_id=2)
    second=identity();second.update(env_id=1,episode_uid='b',reset_id=5)
    source=[first,second]
    result=StereoCamera(c,Reader(packet),depth_estimator=StereoDepth(max_disparity=8)).sample(identity=source)
    assert [x['episode_uid'] for x in result['identity']]==['a','b']
    assert all(x['frame_id']==7 and x['calibration_sha256']==c.sha256 for x in result['identity'])
    assert all('frame_id' not in x for x in source)


def test_batch_calibration_offsets_use_world_origins_not_object_truth():
    c=calibration();origin=torch.tensor([[0.,0.,0.],[2.,3.,0.]])
    k=c.K_left[None].repeat(2,1,1)
    left=c.T_world_left[None].repeat(2,1,1);left[:,:3,3]+=origin
    right=c.T_world_right[None].repeat(2,1,1);right[:,:3,3]+=origin
    c.validate_readback(k,k,left,right,origins=origin)
    with pytest.raises(ValueError):c.validate_readback(k,k,left,right)


def test_invalid_geometry_transform_is_rejected():
    p=torch.ones(1,2,3);t=torch.eye(4);t[0,0]=2.
    with pytest.raises(ValueError,match='rigid|rotation'):
        transform_points(p,t)


def test_unrectified_depth_requires_explicit_returned_rectified_calibration():
    c=calibration();data=c.to_dict();data['rectified']=False
    c=StereoCalibration.from_dict(data)
    with pytest.raises(ValueError,match='rectif'):
        StereoDepth(max_disparity=8,rectify=True).compute(*pair(),c)


def test_camera_render_never_advances_simulation():
    class Clock:
        current_time=.1
        def render(self):self.current_time+=.01
    c=calibration()
    with pytest.raises(RuntimeError,match='advanced physics'):
        StereoCamera(c,Reader(reader_packet(c)),sim=Clock()).sample(identity=identity())


def test_declared_calibration_digest_is_not_silently_ignored():
    c=calibration();data=c.to_dict();data['calibration_sha256']='0'*64
    with pytest.raises(ValueError,match='SHA|digest'):
        StereoCalibration.from_dict(data)


def test_rectification_validity_masks_propagate_to_depth_and_cpu_reader_cannot_claim_native():
    c=calibration();left,right=pair()
    mask=torch.ones(1,32,64,dtype=torch.bool);mask[:,:,15:25]=False
    result=StereoDepth(max_disparity=8,block_size=3).compute(left,right,c,left_valid=mask,right_valid=torch.ones_like(mask))
    assert not result['valid'][:,:,15:25].any()
    packet=reader_packet(c);packet['native_evidence']=True
    assert StereoCamera(c,Reader(packet),depth_estimator=StereoDepth(max_disparity=8)).sample(identity=identity())['native_evidence'] is False


@pytest.mark.parametrize('key,value',[('env_id',-1),('reset_id',None),('policy_tick',.5),('episode_uid','')])
def test_camera_identity_rejects_invalid_environment_and_episode_fields(key,value):
    c=calibration();row=identity();row[key]=value
    with pytest.raises(ValueError,match='identity'):
        StereoCamera(c,Reader(reader_packet(c))).sample(identity=row)
