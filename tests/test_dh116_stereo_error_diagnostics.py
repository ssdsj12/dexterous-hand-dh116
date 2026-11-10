"""Analytic software tests only; none are native qualification evidence."""
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_error_diagnostics import (
    region_metrics, reference_visibility, wire_bands, rejection_summary, sampled_cost_curves,
)


def test_region_keeps_reference_denominator_and_empty_quantiles():
    ref=torch.ones(1,3,4);pred=ref+.004
    valid=torch.zeros_like(ref,dtype=torch.bool);valid[0,1,1]=True
    roi=torch.ones_like(valid);k=torch.tensor([[100.,0,2.],[0,100.,1.5],[0,0,1.]])
    m=region_metrics(pred,ref,valid,roi,roi,k)
    assert m['reference_valid_count']==12 and m['intersection_count']==1
    assert m['prediction_coverage']==pytest.approx(1/12)
    assert m['correct_3d_point_completeness_5mm']==pytest.approx(1/12)
    assert m['mae_m']==pytest.approx(.004,abs=1e-7)
    empty=region_metrics(pred,ref,valid,roi,torch.zeros_like(roi),k)
    assert empty['mae_m'] is None and empty['p95_m'] is None
    assert empty['prediction_coverage'] is None


def test_correct_3d_uses_ray_length_not_z_only():
    ref=torch.ones(1,1,2);pred=ref+.004
    mask=torch.ones_like(ref,dtype=torch.bool)
    k=torch.tensor([[1.,0,0.],[0,1.,.5],[0,0,1.]])
    m=region_metrics(pred,ref,mask,mask,mask,k)
    assert m['correct_z_completeness_5mm']==1.
    assert m['correct_3d_point_completeness_5mm']==.5


def test_visibility_retains_unknown_and_detects_actual_right_occluder():
    z=torch.ones(1,3,8);right=z.clone();right[0,1,3]=.5
    v=torch.ones_like(z,dtype=torch.bool)
    k=torch.tensor([[10.,0,4.],[0,10.,1.5],[0,0,1.]])
    out=reference_visibility(z,right,v,k,.1,tolerance_m=.001)
    assert out['left_only_occluded'][0,1,4]
    assert out['common_visible'][0,1,5]
    assert out['outside_right_fov'][0,1,0]
    assert not (out['left_only_occluded']&out['common_visible']).any()
    missing=reference_visibility(z,None,v,k,.1)
    assert missing['common_visible'] is None and missing['left_only_occluded'] is None


def test_bands_and_overlapping_rejections_do_not_double_count_as_exclusive():
    mask=torch.zeros(1,9,13,dtype=torch.bool);mask[:,3:6,2:11]=True
    bands=wire_bands(mask)
    assert (bands['wire_edge']&bands['wire_interior']).sum()==0
    assert torch.equal(bands['wire_edge']|bands['wire_interior'],mask)
    reasons={'low_texture':mask.clone(),'left_right_inconsistent':mask.clone()}
    s=rejection_summary(reasons,mask)
    assert s['overlapping_counts']['low_texture']==27
    assert s['overlapping_counts']['left_right_inconsistent']==27
    assert sum(s['first_failure_counts'].values())==27


def test_cost_probe_identifies_flat_ambiguous_cost_and_patch_mixing():
    rgb=torch.full((1,3,9,20),.5)
    mask=torch.zeros(1,9,20,dtype=torch.bool);mask[:,4,2:18]=True
    curves=sampled_cost_curves(rgb,rgb,torch.tensor([[0,4,12]]),[1,2,3,4],3,wire_mask=mask)
    assert curves[0]['sad_cost']==pytest.approx([0.,0.,0.,0.])
    assert curves[0]['census_hamming']==[0,0,0,0]
    assert curves[0]['non_adjacent_minimum_count']==2
    assert curves[0]['left_patch_wire_fraction']==pytest.approx(1/3)


def test_diagnostics_reject_bad_shapes_and_negative_visibility_tolerance():
    z=torch.ones(1,3,8);v=torch.ones_like(z,dtype=torch.bool);k=torch.eye(3)
    with pytest.raises(ValueError):region_metrics(z,z,v,v,torch.ones(3,8,dtype=torch.bool),k)
    with pytest.raises(ValueError):reference_visibility(z,z,v,k,.1,tolerance_m=-.1)


def test_replay_rejections_exposes_unimplemented_filter_as_na():
    from go2_pvcnn.control.dh116.tabletop_stereo_error_diagnostics import replay_depth_conditions
    from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
    from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth
    torch.manual_seed(5)
    left=torch.rand(1,3,15,40);right=torch.zeros_like(left);right[:,:,:,:-4]=left[:,:,:,4:]
    k=[[30.,0,20.],[0,30.,7.5],[0,0,1.]];a=torch.eye(4);b=a.clone();b[0,3]=.08
    c=StereoCalibration(k,k,a,b,40,15,.08)
    cfg=dict(min_disparity=1,max_disparity=8,block_size=3,subpixel=True)
    actual=StereoDepth(**cfg).compute(left,right,c)
    replay=replay_depth_conditions(left,right,c,cfg)
    assert torch.equal(replay['valid'],actual['valid'])
    assert replay['reasons']['depth_edge_rejection'] is None
    assert replay['scope'].startswith('offline_replay')


def test_bound_artifact_digest_checked_before_deserialization(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import checked_artifact
    p=tmp_path/'not_a_pickle.pt';p.write_bytes(b'not a trusted tensor')
    with pytest.raises(ValueError,match='SHA'):
        checked_artifact(dict(path=str(p),sha256='0'*64))


def test_unqualified_wire_is_na_and_actual_reference_denominator_retained():
    from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import analyze_frame
    from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
    shape=(1,9,20);k=[[20.,0,10.],[0,20.,4.5],[0,0,1.]]
    a=torch.eye(4);b=a.clone();b[0,3]=.08;c=StereoCalibration(k,k,a,b,20,9,.08)
    pred=torch.ones(shape)*.5;v=torch.ones(shape,dtype=torch.bool)
    frame=dict(left_rgb=torch.rand(1,3,9,20),right_rgb=torch.rand(1,3,9,20),depth_m=pred,
        valid=v,disparity_px=torch.ones(shape)*3.2,identity=[dict(env_id=0)],frame_id=1,simulation_time=.1)
    ref=dict(depth_m=pred.clone(),valid=v,wire_reference_mask=v)
    replay_valid=v.clone();replay_valid[0,0,0]=False
    replay=dict(valid=replay_valid,reasons={'depth_edge_rejection':None},scope='offline_replay_fixture')
    out,regions=analyze_frame(frame,ref,c,dict(block_size=3,min_depth_m=.2,max_depth_m=1.2,
        min_disparity=1,max_disparity=6,texture_threshold=.001),replay=replay)
    assert not out['per_env'][0]['regions']['wire']['available']
    assert out['per_env'][0]['regions']['global_image']['reference_valid_count']==180
    assert out['per_env'][0]['regions']['global_image']['saved_replay_validity']['saved_valid_replay_invalid']==1
    assert out['right_wire_instance_mask'] is None


def test_cli_output_directory_is_exclusive(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import main
    with pytest.raises(FileExistsError):main(['--manifest',str(tmp_path/'missing.json'),'--output',str(tmp_path)])


def test_visibility_tolerance_is_explicit_and_farther_surface_is_not_occlusion():
    left=torch.ones(1,3,8);right=left+.002
    v=torch.ones_like(left,dtype=torch.bool);k=torch.tensor([[10.,0,4.],[0,10.,1.5],[0,0,1.]])
    narrow=reference_visibility(left,right,v,k,.1,tolerance_m=.001)
    wide=reference_visibility(left,right,v,k,.1,tolerance_m=.003)
    assert not narrow['left_only_occluded'].any()
    assert narrow['reference_disagreement'].sum()>0
    assert wide['common_visible'].sum()>narrow['common_visible'].sum()


def test_current_implementation_identity_binds_replay_sources():
    from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import implementation_identity
    result=implementation_identity()
    assert result['sources']['tabletop_stereo_depth']['sha256']
    assert len(result['sources']['tabletop_stereo_error_diagnostics']['sha256'])==64


@pytest.mark.parametrize('info',[None,{},dict(idToLabels={})])
def test_missing_instance_map_is_na_not_measured_empty_roi(info):
    from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import _label_masks
    result=_label_masks(dict(raw_instance_ids=torch.zeros(1,3,4,dtype=torch.long),raw_instance_info=info),
        dict(identity=[dict(env_id=0)]))
    assert all(value is None for value in result.values())


def test_parent_benchmark_cannot_substitute_other_proof(tmp_path):
    import hashlib,json
    from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import resolve_proof
    cap=tmp_path/'capture.json';cap.write_text(json.dumps(dict(width=640,height=480,cohort_sha256='x',stereo_qualification='actual.json')))
    parent=tmp_path/'parent.json';parent.write_text(json.dumps(dict(resolutions=[dict(width=640,height=480,
        capture_report_path=str(cap),capture_report_sha256=hashlib.sha256(cap.read_bytes()).hexdigest(),
        cohort_sha256='x',stereo_qualification='substituted.json')])))
    row=dict(width=640,height=480,cohort_sha256='x',report_artifact=dict(path=str(parent),sha256=hashlib.sha256(parent.read_bytes()).hexdigest()))
    with pytest.raises(ValueError,match='proof identity'):resolve_proof(row)


def test_sgbm_internal_uniqueness_is_na_not_binary_range_confidence(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_stereo_error_diagnostics import replay_depth_conditions
    from go2_pvcnn.control.dh116.tabletop_stereo_depth import StereoDepth
    from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
    # External OpenCV result boundary is injected; no fake native evidence.
    monkeypatch.setattr(StereoDepth,'_sgbm',lambda self,a,b:(torch.full_like(a,4.),torch.full_like(b,4.)))
    rgb=torch.rand(1,3,9,20);k=[[20.,0,10.],[0,20.,4.5],[0,0,1.]]
    a=torch.eye(4);b=a.clone();b[0,3]=.08;c=StereoCalibration(k,k,a,b,20,9,.08)
    result=replay_depth_conditions(rgb,rgb,c,dict(method='sgbm',min_disparity=1,max_disparity=8,block_size=3))
    assert result['reasons']['uniqueness_rejection'] is None
    assert result['uniqueness_scope']=='OpenCV_internal_rejection_not_separately_exposed'


def test_visual_overlay_does_not_index_missing_sgbm_rejection():
    from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import rejection_overlay
    m=torch.zeros(1,3,4,dtype=torch.bool);m[0,1,1]=True
    code,missing=rejection_overlay(dict(low_texture_left=m,left_right_inconsistent=~m,uniqueness_rejection=None),0)
    assert code[1,1]==1 and code[0,0]==2
    assert missing==['uniqueness_rejection']
