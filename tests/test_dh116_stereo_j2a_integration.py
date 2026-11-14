"""J2-A integration/contract software fixtures, not native success tests."""
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth


def pair(batch=2):
    torch.manual_seed(41)
    left=torch.rand(batch,3,17,48);right=torch.zeros_like(left);right[:,:,:,:-4]=left[:,:,:,4:]
    k=[[30.,0,24.],[0,30.,8.5],[0,0,1.]];a=torch.eye(4);b=a.clone();b[0,3]=.08
    return left,right,StereoCalibration(k,k,a,b,48,17,.08)


def test_nested_diagnostic_masks_concatenate_in_image_batches():
    left,right,c=pair()
    full=StereoDepth(min_disparity=1,max_disparity=8,block_size=3,subpixel=True,image_processing_batch_size=None).compute(left,right,c)
    chunk=StereoDepth(min_disparity=1,max_disparity=8,block_size=3,subpixel=True,image_processing_batch_size=1).compute(left,right,c)
    assert isinstance(full.get('diagnostics'),dict),'diagnostic contract is required'
    for name,value in full['diagnostics']['reasons'].items():
        if value is None:assert chunk['diagnostics']['reasons'][name] is None
        else:
            assert value.shape==(2,17,48)
            assert torch.equal(value,chunk['diagnostics']['reasons'][name])
    assert full['valid_mask'] is full['valid']
    assert full['occlusion_mask'] is full['occlusion']


def test_candidate_policy_is_opt_in_and_bound_without_changing_legacy_config():
    legacy=StereoDepth().to_config()
    assert 'reliability' not in legacy
    candidate=StereoDepth(method='reliable_sad',subpixel=True).to_config()
    assert isinstance(candidate['reliability'],dict)
    assert candidate['method']=='reliable_sad'
    assert StereoDepth.from_config(candidate).to_config()==candidate


def test_reliable_candidate_has_no_vga_cost_volume_and_bounded_output():
    left,right,c=pair(1)
    matcher=StereoDepth(method='reliable_sad',min_disparity=1,max_disparity=8,block_size=3,subpixel=True)
    out=matcher.compute(left,right,c)
    assert out['valid'].any(),'all-invalid is not observable-fixture accuracy'
    assert torch.isnan(out['depth_m'][~out['valid']]).all()
    assert out['disparity_px'].is_floating_point()
    assert out['diagnostics']['confidence_is_probability'] is False
    assert out['diagnostics']['true_occlusion_available'] is False
    assert matcher.last_resource_metadata['cost_volume_allocated'] is False


def test_processing_contract_binds_reliable_candidate_source():
    from go2_pvcnn.control.dh116.tabletop_stereo_preprocessing import processing_contract
    _,_,c=pair(1);cfg=c.to_dict();cfg['depth_estimator']=dict(method='reliable_sad',subpixel=True)
    contract=processing_contract(cfg)
    assert 'tabletop_stereo_matcher_candidates.py' in contract['source_sha256']


def test_offline_replay_uses_same_candidate_and_separated_reasons():
    from go2_pvcnn.control.dh116.tabletop_stereo_error_diagnostics import replay_depth_conditions
    left,right,c=pair(1)
    cfg=dict(method='reliable_sad',min_disparity=1,max_disparity=8,block_size=3,subpixel=True)
    expected=StereoDepth(**cfg).compute(left,right,c)
    replay=replay_depth_conditions(left,right,c,cfg)
    assert torch.equal(replay['valid'],expected['valid'])
    assert replay['reasons']['true_occlusion'] is None
    assert replay['reasons']['search_boundary'].shape==expected['valid'].shape


def test_photometric_support_uses_raw_pixel_masks_once(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_stereo_matcher_candidates as candidates
    left,right,c=pair(1);left_mask=torch.ones_like(left[:,0],dtype=torch.bool)
    right_mask=left_mask.clone();right_mask[:,:,23:26]=False
    calls=[];original=candidates.photometric_refine_winner
    def observe(a,b,d,sign,size,ma,mb,**kwargs):
        calls.append((ma.clone(),mb.clone()))
        return original(a,b,d,sign,size,ma,mb,**kwargs)
    monkeypatch.setattr(candidates,'photometric_refine_winner',observe)
    StereoDepth(method='reliable_sad',min_disparity=1,max_disparity=8,block_size=3,subpixel=True).compute(
        left,right,c,left_valid=left_mask,right_valid=right_mask)
    assert len(calls)==2
    assert torch.equal(calls[0][0],left_mask) and torch.equal(calls[0][1],right_mask)
    assert torch.equal(calls[1][0],right_mask) and torch.equal(calls[1][1],left_mask)
