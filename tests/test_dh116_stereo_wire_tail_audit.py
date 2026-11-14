import pytest
import torch


def scene():
    rgb=torch.zeros(1,3,12,24)
    rgb[:,0,4:9,4:20]=.8;rgb[:,1,4:9,4:20]=.5;rgb[:,2,4:9,4:20]=.2
    depth=torch.full((1,12,24),float('nan'));depth[:,4,4:20]=1.
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tracking import TrackedWirePointExtractor
    tracker=TrackedWirePointExtractor(point_sampling='observed_ribbon',ribbon_radius_px=3)
    points=tracker.extract(rgb,depth,torch.isfinite(depth),torch.eye(3),torch.eye(4))
    return rgb,depth,points


def test_sampling_audit_checks_actual_unique_same_component_pixels():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tail_audit import validate_surface_samples
    rgb,depth,points=scene()
    result=validate_surface_samples(rgb,depth,points,torch.eye(3),torch.eye(4),radius=3)
    assert result[0]['unique_pixels'] == int(points['point_valid'].sum())
    assert result[0]['all_source_pixels_measured'] and result[0]['all_same_visible_component']
    points['point_pixels'][0,-1]=points['point_pixels'][0,96]
    points['point_valid'][0,-1]=True;points['point_provenance'][0,-1]=2
    with pytest.raises(ValueError,match='duplicate'):
        validate_surface_samples(rgb,depth,points,torch.eye(3),torch.eye(4),radius=3)


def test_sampling_audit_rejects_hidden_pixel_or_modified_xyz():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tail_audit import validate_surface_samples
    rgb,depth,points=scene()
    first=points['point_valid'][0].nonzero()[0,0]
    points['xyz'][0,first,0]+=1
    with pytest.raises(ValueError,match='XYZ'):
        validate_surface_samples(rgb,depth,points,torch.eye(3),torch.eye(4),radius=3)


def test_tail_metrics_use_euclidean_threshold_and_true_fractional_right_pixel():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tail_audit import tail_geometry
    K=torch.tensor([[10.,0,0],[0,10.,0],[0,0,1.]])
    # Z error4mm, optical ray at x20 makes point error>5mm.
    values=tail_geometry(1.,1.004,K,y=0,x=20,disparity=2.25,baseline=.1)
    assert values['is_tail_5mm'] is True
    assert values['right_match_x_px']==17.75
    assert values['z_absolute_error_m']==pytest.approx(.004)
    assert values['point_error_3d_m']>.005


def test_missing_rejection_channel_stays_unknown():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tail_audit import rejection_at_pixel
    assert rejection_at_pixel({'absent':None,'present':torch.ones(1,2,2,dtype=torch.bool)},0,0,0)=={
        'absent':None,'present':True}


def test_correct_completeness_uses_fixed_native_reference_denominator():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tail_audit import completeness_counts
    mask=torch.ones(1,2009,dtype=torch.bool)
    assert completeness_counts(mask,mask,91,96)==dict(valid_sample_count=96,
        correct_3d_points_5mm=91,reference_denominator=2009,correct_3d_completeness=91/2009)
    other=torch.ones(1,2054,dtype=torch.bool)
    assert completeness_counts(other,other,89,96)['correct_3d_completeness']==89/2054


def test_separate_fragments_have_distinct_components_and_hidden_gap_is_rejected():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tracking import TrackedWirePointExtractor
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tail_audit import validate_surface_samples
    rgb=torch.zeros(1,3,16,32)
    for x0,x1 in ((2,12),(20,30)):
        rgb[:,0,5:10,x0:x1]=.8;rgb[:,1,5:10,x0:x1]=.5;rgb[:,2,5:10,x0:x1]=.2
    depth=torch.ones(1,16,32)
    tracker=TrackedWirePointExtractor(point_sampling='observed_ribbon',ribbon_radius_px=3)
    points=tracker.extract(rgb,depth,torch.isfinite(depth),torch.eye(3),torch.eye(4))
    result=validate_surface_samples(rgb,depth,points,torch.eye(3),torch.eye(4))
    assert {p['component_id'] for p in result[0]['samples'] if p['source_code']==2}=={0,1}
    assert all(p['x']<12 or p['x']>=20 for p in result[0]['samples'])
    slot=points['point_valid'][0].nonzero()[0,0]
    points['point_pixels'][0,slot]=torch.tensor([7,16])
    with pytest.raises(ValueError,match='actual visible'):
        validate_surface_samples(rgb,depth,points,torch.eye(3),torch.eye(4))


def test_wrong_stored_component_rejected_but_absent_component_is_not_recorded():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tail_audit import validate_surface_samples
    rgb,depth,points=scene()
    first=points['point_valid'][0].nonzero()[0,0]
    points['component_id'][0,first]=999
    with pytest.raises(ValueError,match='stored component'):
        validate_surface_samples(rgb,depth,points,torch.eye(3),torch.eye(4))
    del points['component_id']
    result=validate_surface_samples(rgb,depth,points,torch.eye(3),torch.eye(4))
    assert result[0]['stored_component_id_status']=='not_recorded'


def test_track_alias_cannot_join_two_visible_components():
    from go2_pvcnn.control.dh116.tabletop_stereo_wire_tail_audit import validate_track_components
    with pytest.raises(ValueError,match='multiple visible components'):
        validate_track_components(torch.tensor([7,7]),[0,1],'image_track_id')
    assert validate_track_components(torch.tensor([-1,-1]),[0,1],'image_track_id')=='all_unknown'
