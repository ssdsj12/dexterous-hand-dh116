"""Resolution/subpixel software checks; synthetic accuracy is never native evidence."""
import copy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth

ROOT=Path(__file__).resolve().parents[1]


def resolution_module():
    name='go2_pvcnn.control.dh116.tabletop_stereo_resolution'
    assert importlib.util.find_spec(name),'independent resolution-v2 helpers are missing'
    return importlib.import_module(name)


def baseline():
    return StereoCalibration.from_json(ROOT/'configs/dh116_training/stereo_camera_v1.json')


def shifted_pair(disparity=4.35,batch=1,height=32,width=96):
    left=torch.rand(batch,3,height,width,generator=torch.Generator().manual_seed(116))
    yy,xx=torch.meshgrid(torch.arange(height),torch.arange(width),indexing='ij')
    grid=torch.stack((2*(xx+disparity)/(width-1)-1,2*yy/(height-1)-1),-1).float()
    right=F.grid_sample(left,grid[None].expand(batch,-1,-1,-1),align_corners=True,padding_mode='zeros')
    k=[[60.,0.,width/2],[0.,60.,height/2],[0.,0.,1.]]
    t=torch.eye(4,dtype=torch.float64);tr=t.clone();tr[0,3]=.08
    calibration=StereoCalibration(k,k,t,tr,width,height,.08)
    return left,right,calibration


@pytest.mark.parametrize('size',[(224,96),(320,240),(640,360),(640,480),(800,600)])
def test_rerender_calibration_reauthors_square_pixels_and_preserves_extrinsics(size):
    m=resolution_module();old=baseline();new=m.rerender_calibration(old,*size)
    fx=size[0]/(2*math.tan(math.radians(75)/2))
    assert (new.width,new.height)==size
    assert float(new.K_left[0,0])==pytest.approx(fx)
    assert float(new.K_left[1,1])==pytest.approx(fx)
    assert new.K_left[0,2]==size[0]/2 and new.K_left[1,2]==size[1]/2
    assert torch.equal(new.T_world_left,old.T_world_left) and torch.equal(new.T_world_right,old.T_world_right)
    assert new.baseline_m==old.baseline_m


def test_image_resize_calibration_scales_K_while_rerender_uses_FOV():
    m=resolution_module();old=baseline()
    resized=m.resize_calibration(old,640,480);rendered=m.rerender_calibration(old,640,480)
    scale=torch.diag(torch.tensor([640/224,480/96,1.],dtype=torch.float64))
    torch.testing.assert_close(resized.K_left,scale@old.K_left)
    assert not torch.equal(resized.K_left,rendered.K_left)
    restored=m.resize_calibration(resized,224,96)
    torch.testing.assert_close(restored.K_left,old.K_left)
    assert resized.sha256!=rendered.sha256


def test_resolution_resource_guard_is_overridable_without_hidden_resize():
    m=resolution_module()
    assert m.validate_resolution(640,480)==(640,480)
    with pytest.raises(ValueError,match='resource|limit'):m.validate_resolution(4096,2048)
    assert m.validate_resolution(4096,2048,allow_resource_override=True)==(4096,2048)
    for invalid in ((0,480),(640,-1),(640.0,480),(True,480)):
        with pytest.raises(ValueError):m.validate_resolution(*invalid)


def test_new_profiles_keep_old224_config_and_cover_fx_baseline_near_range():
    m=resolution_module()
    legacy=ROOT/'configs/dh116_training/stereo_camera_v1.json'
    assert hashlib.sha256(legacy.read_bytes()).hexdigest()==LEGACY_224_SHA
    for width,height in ((320,240),(640,360),(640,480)):
        path=ROOT/f'configs/dh116_training/stereo_camera_{width}x{height}_v2.json'
        cfg=json.loads(path.read_text());c=StereoCalibration.from_dict(cfg)
        options=cfg['depth_estimator']
        assert options['subpixel'] is True and options['image_processing_batch_size']==1
        assert options['max_disparity']>=math.ceil(float(c.K_left[0,0])*c.baseline_m/options['min_depth_m'])
        assert torch.equal(c.T_world_left,baseline().T_world_left)
    cfg=json.loads((ROOT/'configs/dh116_training/stereo_resolution_experiment_v1.json').read_text())
    assert cfg['default_accuracy_profile'].endswith('stereo_camera_640x480_v2.json')
    assert cfg['selected_resolution'] is None


def test_subpixel_fractional_shift_improves_disparity_and_depth_over_integer_ablation():
    left,right,c=shifted_pair()
    kwargs=dict(min_disparity=1,max_disparity=12,block_size=5,min_depth_m=.2,max_depth_m=2.,
                uniqueness_threshold=.01,lr_tolerance_px=.75)
    integer=StereoDepth(**kwargs,subpixel=False).compute(left,right,c)
    subpixel=StereoDepth(**kwargs,subpixel=True).compute(left,right,c)
    mask=integer['valid']&subpixel['valid'];mask[:,:,:16]=False;mask[:,:,-16:]=False
    assert mask.sum()>500
    e0=(integer['disparity_px'][mask]-4.35).abs().mean()
    e1=(subpixel['disparity_px'][mask]-4.35).abs().mean()
    assert e1<e0*.8
    expected=60*.08/4.35
    assert (subpixel['depth_m'][mask]-expected).abs().mean()<(integer['depth_m'][mask]-expected).abs().mean()
    assert ((subpixel['disparity_px'][mask]-subpixel['disparity_px'][mask].round()).abs()>1e-3).any()


def test_streamed_disparities_and_image_minibatches_preserve_results_and_masks():
    left,right,c=shifted_pair(batch=3)
    mask=torch.ones(3,32,96,dtype=torch.bool);mask[:,:,30:35]=False
    kwargs=dict(max_disparity=12,block_size=3,max_depth_m=2.,subpixel=True)
    estimator=StereoDepth(**kwargs,image_processing_batch_size=1)
    one=estimator.compute(left,right,c,left_valid=mask)
    all_images=StereoDepth(**kwargs,image_processing_batch_size=3).compute(left,right,c,left_valid=mask)
    for key in ('depth_m','disparity_px','valid','occlusion','confidence'):
        torch.testing.assert_close(one[key],all_images[key],atol=1e-6,rtol=0,equal_nan=True)
    assert not one['valid'][:,:,30:35].any()
    assert estimator.last_resource_metadata['cost_volume_allocated'] is False
    assert estimator.last_resource_metadata['disparity_planes_in_memory']<=3
    assert estimator.last_resource_metadata['image_processing_batch_size']==1


def test_real_image_resize_scales_disparity_and_intrinsics_preserving_depth():
    left,right,c=shifted_pair(disparity=4.)
    resized=resolution_module().resize_calibration(c,192,64)
    a=StereoDepth(max_disparity=12,max_depth_m=2.,block_size=3,subpixel=False).compute(left,right,c)
    left2=F.interpolate(left,size=(64,192),mode='bilinear',align_corners=False)
    right2=F.interpolate(right,size=(64,192),mode='bilinear',align_corners=False)
    b=StereoDepth(max_disparity=24,max_depth_m=2.,block_size=3,subpixel=False).compute(left2,right2,resized)
    assert int(a['valid'].sum())>500 and int(b['valid'].sum())>1000
    assert a['disparity_px'][a['valid']].median()==4.
    assert b['disparity_px'][b['valid']].median()==8.
    assert a['depth_m'][a['valid']].median()==pytest.approx(1.2)
    assert b['depth_m'][b['valid']].median()==pytest.approx(1.2)


def test_fractional_lr_requires_both_interpolation_neighbours_valid():
    m=resolution_module();m  # explicit helper module is part of the release
    value=torch.tensor([[[1.,2.,4.,8.]]]);valid=torch.tensor([[[True,True,False,True]]])
    sampled,accepted=StereoDepth.sample_horizontal(value,torch.tensor([[[.5,1.5,3.,4.]]]),valid)
    assert sampled[0,0,0]==1.5 and accepted[0,0,0]
    assert not accepted[0,0,1] and accepted[0,0,2] and not accepted[0,0,3]


def test_invalid_remapped_pixels_cannot_change_subpixel_curve_or_valid_outputs():
    left,right,c=shifted_pair()
    valid=torch.ones(1,32,96,dtype=torch.bool);valid[:,:,40:44]=False
    estimator=StereoDepth(max_disparity=12,max_depth_m=2.,block_size=5,subpixel=True)
    a=estimator.compute(left,right,c,left_valid=valid,right_valid=valid)
    changed_left=left.clone();changed_right=right.clone()
    changed_left[:,:,:,40:44]=1.-changed_left[:,:,:,40:44]
    changed_right[:,:,:,40:44]=1.-changed_right[:,:,:,40:44]
    b=estimator.compute(changed_left,changed_right,c,left_valid=valid,right_valid=valid)
    torch.testing.assert_close(a['valid'],b['valid'],atol=0,rtol=0)
    common=a['valid']&b['valid']
    torch.testing.assert_close(a['disparity_px'][common],b['disparity_px'][common],atol=0,rtol=0)


def test_cpu_fixture_metrics_are_na_and_cannot_select_even_with_zero_gpu_memory():
    m=resolution_module()
    report=dict(width=640,height=480,batch_size=1,sample_count=100,cohort_sha256='a'*64,
        source_scope='cpu_fixture',schema_only=True,metrics=dict(depth_mae_m=0.,depth_rmse_m=0.,
        depth_p95_m=0.,wire_point_cloud_completeness=1.,gpu_peak_allocated_bytes=0,
        gpu_peak_reserved_bytes=0,gpu_free_bytes=0,capture_ms=1.,depth_ms=1.,actor_ms=1.))
    result=m.compare_resolution_results([report])
    assert result['selected_resolution'] is None
    assert result['resolutions'][0]['metrics']['depth_rmse_m'] is None
    assert result['resolutions'][0]['metrics']['gpu_peak_allocated_bytes'] is None
    assert result['resolutions'][0]['gpu_device_used_bytes'] is None
    assert result['resolutions'][0]['stereo_pairs_per_second'] is None


def test_resolution_compare_requires_verified_same_cohort_and_explicit_thresholds():
    m=resolution_module()
    result=m.compare_resolution_results([])
    assert result['selected_resolution'] is None and result['native_reports']==0
    report=dict(width=640,height=480,source_scope='native',schema_only=False,
        cohort_sha256='a'*64,sample_count=100,batch_size=1,metrics={'depth_mae_m':0.})
    result=m.compare_resolution_results([report])
    assert result['selected_resolution'] is None and result['resolutions'][0]['qualified'] is False


def test_encoder_image_chunks_preserve_GN_features_and_parameter_gradients():
    from go2_pvcnn.control.dh116.tabletop_stereo_encoder import StereoEncoder
    full=StereoEncoder(embed_dim=32,image_processing_batch_size=None)
    chunked=StereoEncoder(embed_dim=32,image_processing_batch_size=1)
    chunked.load_state_dict(full.state_dict())
    left=torch.rand(2,3,32,32);right=torch.rand_like(left);valid=torch.ones(2,2,dtype=torch.bool)
    a,_=full(left,right,valid);b,_=chunked(left,right,valid)
    torch.testing.assert_close(a,b,atol=2e-5,rtol=1e-5)
    a.square().mean().backward();b.square().mean().backward()
    torch.testing.assert_close(full.backbone.stem[0].weight.grad,chunked.backbone.stem[0].weight.grad,atol=2e-5,rtol=1e-4)
    assert chunked.image_processing_batch_size==1


def test_native_factory_reads_matching_depth_profile_without_native_import(monkeypatch):
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_stereo_camera')
    calls=[]
    class Reader:
        def __init__(self,sim,origins,calibration,device,**kwargs):calls.append((calibration,kwargs))
    monkeypatch.setattr(m,'_NativeRGBPairReader',Reader)
    camera=m.StereoCamera.from_native(None,torch.zeros(2,3),
        ROOT/'configs/dh116_training/stereo_camera_640x480_v2.json',device='cpu')
    assert camera.depth_estimator.max_disparity==192 and camera.depth_estimator.subpixel
    assert (calls[0][0].width,calls[0][0].height)==(640,480)


def test_native_readback_resource_rejection_precedes_isaac_imports():
    import sys
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_stereo_camera')
    c=resolution_module().rerender_calibration(baseline(),640,480)
    with pytest.raises(ValueError,match='resource limit'):
        m._NativeRGBPairReader(None,torch.zeros(8,3),c,'cpu',max_readback_bytes=1)
    assert not any(name.startswith(('isaacsim.','isaaclab.','omni.')) for name in sys.modules)


def test_matching_never_uses_disparity_as_pooling_batch_dimension(monkeypatch):
    left,right,c=shifted_pair(batch=3)
    original=F.avg_pool2d;shapes=[]
    def observed(value,*args,**kwargs):
        shapes.append(tuple(value.shape));return original(value,*args,**kwargs)
    monkeypatch.setattr(F,'avg_pool2d',observed)
    StereoDepth(max_disparity=12,max_depth_m=2.,subpixel=True,image_processing_batch_size=1).compute(left,right,c)
    assert shapes and all(shape[0]==1 and shape[1]==1 for shape in shapes)
    assert all(shape[-2:]==(32,96) for shape in shapes)


def test_resolution_selection_uses_verified_measured_accuracy_and_same_cohort_only(monkeypatch):
    m=resolution_module()
    # Isolate pure comparison arithmetic by mocking the common proof verifier;
    # this creates no native artifact or actual qualification.
    def verified(key):
        width=320 if key=='small' else 640
        return dict(qualified=True,measurements_verified=True,source_scope='native',width=width,height=240 if width==320 else 480,
            cohort_sha256='a'*64,sample_count=100,metrics=dict(mae_m=.002,rmse_m=.004 if width==320 else .003,
                p95_m=.006,wire_point_cloud_completeness=.95),performance=dict(source_scope='native',
                measurement_device='cuda',batch_size=2,gpu_peak_allocated_bytes=1024,gpu_peak_reserved_bytes=2048,
                gpu_free_bytes=4096,gpu_device_used_bytes=8192,capture_ms=10.,depth_ms=20.,actor_ms=5.))
    monkeypatch.setattr(m,'_verified_stereo',verified)
    reports=[dict(width=w,height=h,source_scope='native',schema_only=False,stereo_qualification=key)
             for w,h,key in [(320,240,'small'),(640,480,'large')]]
    thresholds=dict(max_depth_mae_m=.005,max_depth_rmse_m=.005,max_depth_p95_m=.01,
        min_wire_point_cloud_completeness=.9,min_sample_count=100,min_stereo_pairs_per_second=30.)
    result=m.compare_resolution_results(reports,thresholds=thresholds)
    assert result['selected_resolution']==[640,480]
    assert result['resolutions'][0]['stereo_pairs_per_second']==pytest.approx(2000/35)
    assert result['resolutions'][0]['gpu_device_used_bytes']==8192
    def different_cohort(key):
        value=verified(key)
        if key=='large':value['cohort_sha256']='b'*64
        return value
    monkeypatch.setattr(m,'_verified_stereo',different_cohort)
    assert m.compare_resolution_results(reports,thresholds=thresholds)['selected_resolution'] is None
    def failed_accuracy(key):
        value=verified(key);value['qualified']=False;value['metrics']['rmse_m']=.4;return value
    monkeypatch.setattr(m,'_verified_stereo',failed_accuracy)
    bad=m.compare_resolution_results(reports,thresholds=thresholds)
    assert bad['selected_resolution'] is None
    assert bad['resolutions'][0]['metrics']['depth_rmse_m']==.4
    assert bad['native_reports']==2 and bad['qualified_native_reports']==0


def test_vga_float32_sdk_lens_roundtrip_is_numeric_K_error_not_extrinsic_slack():
    c=resolution_module().rerender_calibration(baseline(),640,480)
    focal_mm=float(torch.tensor(2.4,dtype=torch.float32))*10.
    horizontal=float(torch.tensor(c.width*24./float(c.K_left[0,0]),dtype=torch.float32))
    vertical=float(torch.tensor(c.height*24./float(c.K_left[1,1]),dtype=torch.float32))
    actual=c.K_left.clone();actual[0,0]=c.width*focal_mm/horizontal;actual[1,1]=c.height*focal_mm/vertical
    diagnostics=c.validate_readback(actual,actual,c.T_world_left,c.T_world_right)
    assert diagnostics['max_intrinsic_residual_px']>1e-5
    assert diagnostics['max_translation_residual_m']==0.
    bad=actual.clone();bad[0,0]+=.01
    with pytest.raises(ValueError,match='readback mismatch'):
        c.validate_readback(bad,actual,c.T_world_left,c.T_world_right)
    bad_t=c.T_world_right.clone();bad_t[0,3]+=2e-5
    with pytest.raises(ValueError,match='readback mismatch'):
        c.validate_readback(actual,actual,c.T_world_left,bad_t)


LEGACY_224_SHA='f5c53f5c9d3042ee0ce97453a3420dda3ecb7417469d7362be20604b0171d494'
