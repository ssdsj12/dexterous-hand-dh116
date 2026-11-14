"""Schema-aligned runtime state; CPU tensors do not qualify physical sources."""
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from test_dh116_stereo_estimator import history, module


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT/'docs/contracts/stereo_privileged_state_contract.json').read_text(encoding='utf-8'))


def adapter(**kwargs):
    name = 'go2_pvcnn.control.dh116.tabletop_stereo_task_state_adapter'
    assert importlib.util.find_spec(name), 'TaskStateAdapter must expose selected estimated371 fields to Actor'
    return importlib.import_module(name).TaskStateAdapter(**kwargs)


def qualifications(**overrides):
    result = {key: True for row in SCHEMA['dimensions'] for key in row['required_validity_keys']
              if key != 'supervision_forbidden'}
    result['rotation_symmetry_valid'] = True
    result.update(overrides)
    return result


def estimates(batch=2):
    prediction = (torch.arange(371).float()[None].expand(batch,-1)/371).clone().requires_grad_()
    return {'privileged_state': prediction}


def normalization_config(**kwargs):
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    low,high=active_limits()
    return {'joint_lower':low.tolist(),'joint_upper':high.tolist(),**kwargs}


def test_declared_order_selects_schema_fields_instead_of_concatenating_371():
    state = adapter(source_qualification=qualifications(),known_config=normalization_config())
    h = history(); output = state(estimates(), h)
    indices = state.field_indices
    assert indices == tuple(sorted(indices)) and len(set(indices)) == len(indices)
    assert 0 in indices and 61 in indices and 68 in indices and 321 in indices
    assert len(indices) < 371
    assert state.field_order == tuple(SCHEMA['dimensions'][i]['semantic_name'] for i in indices)
    assert state.input_size == 3*len(indices)
    assert output['actor_features'].shape == (2,state.input_size)
    assert output['valid'].dtype == torch.bool
    torch.testing.assert_close(output['actor_features'], torch.cat((torch.where(output['valid'],output['values'],0.),
        output['valid'].float(), output['uncertainty']), -1))
    for index in indices:
        row = SCHEMA['dimensions'][index]
        assert row['deployment_allowed'] and row['supervision_status'] != 'NA'
        assert row['student_source'] != 'unobservable'


def test_predictions_align_exactly_with_training_schema_but_proprio_is_measured():
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    state = adapter(source_qualification=qualifications(),known_config=normalization_config())
    h = history(); h.proprio[:,-1,:12] = .4; h.proprio[:,-1,12:24] = .06
    h.proprio[:,-1,38:62] = .25
    pred = estimates(); result = state(pred,h)
    low, high = (torch.tensor(x,dtype=torch.float32).repeat(2) for x in active_limits())
    truth = pred['privileged_state'].detach().clone()
    truth[:,:12] = 2*(.4-low)/(high-low)-1
    truth[:,12:24] = .06/.3
    truth[:,215:239] = .25
    assert result['valid'][:,:24].all()
    for col,index in enumerate(state.field_indices):
        if not result['valid'][:,col].all(): continue
        expected = truth[:,index]
        if SCHEMA['dimensions'][index]['loss_type'] == 'classification': expected = expected.sigmoid()
        torch.testing.assert_close(result['values'][:,col],expected)
    # Mutating the re-predicted joint/head columns cannot replace encoder input.
    changed = {'privileged_state': pred['privileged_state'].detach().clone()}
    changed['privileged_state'][:,:24] = 1000.
    torch.testing.assert_close(state(changed,h)['values'][:,:24], result['values'][:,:24])


def test_action_features_backpropagate_allowed_visual_fields_and_never_na_or_proprio():
    state = adapter(source_qualification=qualifications())
    pred = estimates(); result = state(pred,history())
    torch.nn.Linear(state.input_size,24,bias=False)(result['actor_features']).square().sum().backward()
    grad = pred['privileged_state'].grad
    assert grad[:,61:64].abs().sum() > 0 and grad[:,321:351].abs().sum() > 0
    assert grad[:,:24].eq(0).all() and grad[:,215:239].eq(0).all()
    forbidden = [row['index'] for row in SCHEMA['dimensions'] if row['supervision_status']=='NA']
    assert grad[:,forbidden].eq(0).all()
    changed = {'privileged_state': pred['privileged_state'].detach().clone()}
    changed['privileged_state'][:,forbidden] = float('nan')
    torch.testing.assert_close(state(changed,history())['actor_features'],result['actor_features'])


def test_masks_are_fail_closed_per_sample_and_uncertainty_never_claims_zero_error():
    state = adapter()
    result = state(estimates(),history())
    assert not result['valid'].any()
    assert result['values'].isnan().all() and result['uncertainty'].eq(1).all()
    state = adapter(source_qualification=qualifications())
    pred = estimates()
    pred['state_valid'] = torch.ones(2,371,dtype=torch.bool)
    pred['state_valid'][0,61] = False
    pred['state_uncertainty'] = torch.full((2,371),.3,requires_grad=True)
    result = state(pred,history()); col = state.field_indices.index(61)
    assert result['valid'][:,col].tolist() == [False,True]
    assert result['values'][0,col].isnan() and result['uncertainty'][0,col] == 1
    assert result['uncertainty'][1,col] == .3
    result['actor_features'].sum().backward()
    assert pred['privileged_state'].grad[0,61] == 0
    assert pred['state_uncertainty'].grad[0,61] == 0
    assert pred['state_uncertainty'].grad[1,61] == 1


def test_contact_classification_is_probability_with_validity_separate_from_contact():
    state = adapter(allowlist=['contact_geometry_valid','contact_all_below_centerline',
        'interval_paired'],source_qualification=qualifications(contact_valid=False))
    pred = estimates(); pred['privileged_state'].data[:,149:153] = 0
    h=history()
    for value in h.as_dict().values():value[:,-2]=value[:,-1]
    result = state(pred,h)
    for index in (149,150):
        col = state.field_indices.index(index)
        assert result['valid'][:,col].all() and result['values'][:,col].eq(.5).all()
    for index in (151,152):
        col = state.field_indices.index(index)
        assert not result['valid'][:,col].any() and result['values'][:,col].isnan().all()
    for index in (355,356):
        col = state.field_indices.index(index)
        torch.testing.assert_close(result['values'][:,col],pred['privileged_state'][:,index])


def test_missing_visibility_correspondence_or_current_sensor_data_masks_visual_prediction():
    state = adapter(source_qualification=qualifications(correspondence_valid=False))
    result = state(estimates(),history()); col = state.field_indices.index(61)
    assert not result['valid'][:,col].any()
    state = adapter(source_qualification=qualifications())
    h = history(); h.image_valid[0,-1] = False
    result = state(estimates(),h)
    assert result['valid'][:,col].tolist() == [False,True]
    h.history_valid.zero_()
    assert not state(estimates(),h)['valid'].any()


def test_known_config_explicitly_supplied_in_raw_units_and_unknown_sources_excluded():
    state = adapter(allowlist=['reset_joint_reference','grip_drive_stiffness'],
        known_config=normalization_config(reset_joint_reference=[.4]*12,grip_drive_stiffness=[15.,20.]),
        source_qualification=qualifications())
    result = state(estimates(),history())
    assert result['valid'].all()
    torch.testing.assert_close(result['values'][:,-2:],torch.tensor([[.5,1.],[.5,1.]]))
    missing = adapter(allowlist=['reset_joint_reference'],source_qualification=qualifications())
    assert not missing(estimates(),history())['valid'].any()
    with pytest.raises(ValueError,match='NA|forbidden|deploy'):
        adapter(allowlist=['force_target'])
    with pytest.raises(ValueError,match='unknown|unsupported'):
        adapter(known_config={'mass':.2})


def test_no_implicit_simulator_limits_or_substitute_fk_measurements():
    state=adapter(allowlist=['joint_position','joint_velocity','adaptive_palm_offset',
        'wrist_velocity','wrist_mount_joint_angles','interval_wrist_z_increment'],
        source_qualification=qualifications())
    result=state(estimates(),history())
    for col,index in enumerate(state.field_indices):
        assert bool(result['valid'][:,col].all()) == (12<=index<24)
    assert result['metadata']['blocked_reasons']


def test_per_field_source_eligibility_and_reversed_allowlist_follow_schema_order():
    eligible=qualifications()
    eligible['source_valid']=torch.ones(2,371,dtype=torch.bool)
    eligible['source_valid'][0,116]=False
    state=adapter(allowlist=[120,'legacy_pad_relative.left.thumb.x',116],source_qualification=eligible)
    assert state.field_indices == (109,116,120)
    result=state(estimates(),history())
    assert result['valid'][:,1].tolist()==[False,True]
    assert result['valid'][:,0].all() and result['valid'][:,2].all()
    pred=estimates(); pred['privileged_state'].data[1,120]=float('nan')
    result=state(pred,history())
    assert not result['valid'][1,2] and torch.isfinite(result['actor_features']).all()
    result['actor_features'].sum().backward()
    assert pred['privileged_state'].grad[1,120]==0


def test_declared_reset_xyz_does_not_qualify_palm_to_wrist_root_transform():
    state=adapter(allowlist=['wrist_pose_from_reset'],known_config={'wrist_reset_xyz':[[1.,2.,3.],[4.,5.,6.]]},
        source_qualification=qualifications())
    h=history()
    h.proprio[:,-1,24:27]=torch.tensor([1.05,2.1,3.15])
    h.proprio[:,-1,31:34]=torch.tensor([4.2,5.25,6.3])
    pred=estimates(); result=state(pred,h)
    assert not result['valid'].any() and result['values'].isnan().all()
    assert not result['values'].requires_grad


def test_object_quaternion_requires_nonzero_prediction_and_declared_symmetry_eligibility():
    state=adapter(allowlist=['object_quaternion'],source_qualification=qualifications(rotation_symmetry_valid=False))
    assert not state(estimates(),history())['valid'].any()
    state=adapter(allowlist=['object_quaternion'],source_qualification=qualifications())
    pred=estimates(); pred['privileged_state'].data[0,64:68]=0
    result=state(pred,history())
    assert not result['valid'][0].any() and result['valid'][1].all()


def test_metadata_has_exact_schema_units_frames_sources_masks_uncertainty_and_na_coverage():
    state = adapter(source_qualification=qualifications())
    result = state(estimates(),history())
    metadata = result['metadata']
    assert len(metadata['schema_coverage']) == 371
    for col,row in enumerate(metadata['fields']):
        source = SCHEMA['dimensions'][state.field_indices[col]]
        assert row['semantic_name'] == source['semantic_name']
        assert row['unit'] == source['unit'] and row['coordinate_frame'] == source['coordinate_frame']
        assert row['normalization'] == source['normalization']
        assert row['source'] and row['validity'] and row['uncertainty']
    assert all(row['status']=='NA' for row in metadata['schema_coverage']
        if SCHEMA['dimensions'][row['index']]['supervision_status']=='NA')
    assert metadata['evidence_scope'] == 'runtime_configuration_not_native_qualification'


def test_optional_visual_heads_remain_differentiable_and_do_not_replace_estimated371():
    shapes={'geometry':(128,3),'velocity':(128,3),'deformation':(128,),
        'contact_logits':(2,3),'slip_logits':(2,),'slip_velocity':(2,)}
    state = adapter(extra_heads=tuple(shapes),source_qualification=qualifications(),
        extra_head_metadata={name:dict(unit='fixture_declared_units',coordinate_frame='fixture_local_frame') for name in shapes})
    pred=estimates()
    pred.update({key:torch.full((2,*shape),.2,requires_grad=True) for key,shape in shapes.items()})
    h=history()
    for value in h.as_dict().values():value[:,-2]=value[:,-1]
    result=state(pred,h)
    result['actor_features'].sum().backward()
    assert pred['privileged_state'].grad[:,61:64].abs().sum()>0
    for key in shapes:
        assert pred[key].grad is not None and pred[key].grad.abs().sum()>0


def test_force_proxy_needs_actual_raw_verified_source_not_qualified_metadata():
    state = adapter(extra_heads=['force_proxy'],source_qualification=qualifications())
    pred=estimates(); pred['force_proxy']=torch.ones(2,2,3,requires_grad=True)
    result=state(pred,history())
    assert not result['valid'][:,-6:].any()
    assert result['values'][:,-6:].isnan().all()
    assert 'force' in result['metadata']['blocked_reasons']['force_proxy']
    declaration=qualifications()
    declaration['force_proxy']={'evidence_scope':'native_qualified','source_sha256':'a'*64}
    spoofed=adapter(extra_heads=['force_proxy'],source_qualification=declaration)(pred,history())
    assert not spoofed['valid'][:,-6:].any()
    spoofed['actor_features'].sum().backward()
    assert pred['force_proxy'].grad.eq(0).all()


def test_force_complete_metadata_still_reopens_raw_proof_before_any_gradient(monkeypatch):
    loss_module=module('losses')
    proof={key:torch.ones(2,2,3,dtype=torch.bool) for key in ('source_valid','contact_valid','attributable')}
    proof.update(source_sha256='a'*64,units='N',coordinate_frame='accepted_pad_normal',
        partner_attribution='verified_object_partner',per_digit_source=[['native_effective_pad']*3]*2,
        sign_convention='compressive_positive_on_object',evidence_scope='native_qualified',
        proof_path='absent-independent-native-proof.json',proof_sha256='b'*64)
    state=adapter(extra_heads=['force_proxy'],source_qualification={**qualifications(),'force_proxy':proof})
    pred=estimates();pred['force_proxy']=torch.ones(2,2,3,requires_grad=True)
    result=state(pred,history())
    assert not result['valid'][:,-6:].any()
    assert result['metadata']['blocked_reasons']['force_proxy']=='force_native_proof_not_verified'
    # This isolated arithmetic mock produces no native proof and grants no
    # qualification. It checks that a future verified raw source can preserve
    # per-channel masking and differentiability through the same gate.
    monkeypatch.setattr(loss_module,'_force_proof_verified',lambda binding:True)
    proof['source_valid'][0]=False
    result=state(pred,history())
    assert not result['valid'][0,-6:].any() and result['valid'][1,-6:].all()
    result['actor_features'].sum().backward()
    assert pred['force_proxy'].grad[0].eq(0).all() and pred['force_proxy'].grad[1].eq(1).all()


def test_rejects_privileged_mappings_bad_prediction_sidebands_and_undeclared_fields():
    state=adapter(); h=history()
    with pytest.raises((TypeError,ValueError),match='history|privileged|StereoHistoryBatch'):
        state(estimates(),{'history':h,'critic':torch.ones(2,371)})
    with pytest.raises((TypeError,ValueError),match='unknown|privileged|forbidden'):
        state({**estimates(),'critic':torch.ones(2,371)},h)
    with pytest.raises((TypeError,ValueError),match='371|shape'):
        state({'privileged_state':torch.ones(2,370)},h)
    with pytest.raises((TypeError,ValueError),match='mask|bool|shape'):
        state({**estimates(),'state_valid':torch.ones(2,371)},h)
    with pytest.raises(ValueError,match='uncertainty'):
        state({**estimates(),'state_uncertainty':torch.full((2,371),-.1)},h)
