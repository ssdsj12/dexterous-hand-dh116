"""New stereo-only estimator contracts; CPU tests are not physics evidence."""
from dataclasses import asdict
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'go2_pvcnn.control.dh116.tabletop_stereo_'


def module(name):
    full = PREFIX + name
    assert importlib.util.find_spec(full), f'Phase3 stereo module missing: {full}'
    return importlib.import_module(full)


def sensor(batch=2, size=32):
    m = module('observation')
    proprio = torch.zeros(batch, 62)
    proprio[:, 27] = 1.; proprio[:, 34] = 1.
    return m.SensorBatch(left_rgb=torch.rand(batch, 3, size, size),
        right_rgb=torch.rand(batch, 3, size, size), depth_m=torch.ones(batch, size, size),
        depth_valid=torch.ones(batch, size, size, dtype=torch.bool),
        depth_confidence=torch.ones(batch, size, size),
        points_world=torch.randn(batch, 128, 3)*.01,
        point_valid=torch.ones(batch, 128, dtype=torch.bool), proprio=proprio,
        dt=torch.full((batch,), 1/60), image_age=torch.zeros(batch),
        image_valid=torch.ones(batch, 2, dtype=torch.bool))


def history(batch=2):
    m = module('observation'); store = m.StereoHistory(batch, history=4, device='cpu')
    return store.update(sensor(batch))


def test_phase3_modules_start_as_explicit_red():
    for name in ('encoder', 'temporal', 'state_estimator', 'losses', 'observation'):
        module(name)


def test_sensor_rejects_unrecognized_privileged_keys_and_supports_device_clone():
    m = module('observation'); batch = sensor()
    batch.validate()
    poisoned = asdict(batch); poisoned['privileged_state'] = torch.randn(2, 371)
    with pytest.raises((ValueError, TypeError), match='privileged|unknown|unexpected'):
        m.SensorBatch.from_mapping(poisoned)
    for key in ('phase', 'success', 'teacher_action', 'wire_nodes'):
        with pytest.raises((ValueError, TypeError)):
            m.SensorBatch(**{**asdict(batch), key: torch.zeros(2)})
    copied = batch.clone().detach().to('cpu')
    assert torch.equal(copied.proprio, batch.proprio)
    assert copied.left_rgb.data_ptr() != batch.left_rgb.data_ptr()
    assert copied.index_select(torch.tensor([1])).left_rgb.shape[0] == 1


@pytest.mark.parametrize('field,replacement', [
    ('proprio', torch.zeros(2, 63)), ('dt', torch.zeros(2)),
    ('image_age', torch.full((2,), -1.)), ('image_valid', torch.ones(2, 2)),
    ('left_rgb', torch.full((2, 3, 32, 32), float('nan'))),
    ('right_rgb', torch.ones(2, 3, 32, 32, dtype=torch.float64)),
    ('point_valid', torch.ones(2, 127, dtype=torch.bool)),
])
def test_sensor_rejects_shape_dtype_and_finite_violations(field, replacement):
    m = module('observation'); values = asdict(sensor()); values[field] = replacement
    with pytest.raises((ValueError, TypeError)):
        m.SensorBatch.from_mapping(values)


def test_masked_depth_points_accept_nan_but_valid_nan_is_rejected():
    m = module('observation'); values = asdict(sensor())
    values['depth_m'][0, 0, 0] = float('nan'); values['depth_valid'][0, 0, 0] = False
    values['points_world'][0, 0] = float('nan'); values['point_valid'][0, 0] = False
    m.SensorBatch.from_mapping(values).validate()
    values['point_valid'][0, 0] = True
    with pytest.raises(ValueError, match='finite'):
        m.SensorBatch.from_mapping(values)


def test_history_reset_clears_prior_masks_appends_current_and_owns_values():
    m = module('observation'); store = m.StereoHistory(2, history=4, device='cpu')
    frame = sensor(); first = store.update(frame)
    assert first.history_valid.tolist() == [[False, False, False, True]]*2
    frame.proprio[:, 0] = .7
    second = store.update(frame, reset_mask=torch.tensor([True, False]))
    assert second.history_valid.tolist() == [[False, False, False, True], [False, False, True, True]]
    assert first.proprio[:, -1, 0].eq(0.).all()
    frame.proprio.zero_()
    assert second.proprio[:, -1, 0].eq(.7).all()
    selected = second.index_select(torch.tensor([1])).detach().clone().to('cpu')
    assert selected.proprio.shape == (1, 4, 62)
    joined = m.StereoHistoryBatch.cat([selected, selected])
    assert joined.proprio.shape == (2, 4, 62)


def test_shared_resnet18_topology_and_four_head_two_layer_temporal_baseline():
    enc = module('encoder'); temporal = module('temporal')
    network = enc.StereoEncoder(embed_dim=32)
    assert sum(isinstance(layer, enc.BasicBlock) for layer in network.backbone.modules()) == 8
    assert [len(getattr(network.backbone, f'layer{i}')) for i in range(1, 5)] == [2, 2, 2, 2]
    assert len([name for name, _ in network.named_children() if 'backbone' in name]) == 1
    sequence = temporal.StereoTemporal(embed_dim=32, history=4, num_heads=4, num_layers=2)
    assert len(sequence.transformer.layers) == 2
    assert sequence.transformer.layers[0].self_attn.num_heads == 4


def test_shared_backbone_processes_both_views_and_missing_view_outputs_zero_tokens():
    enc=module('encoder'); network=enc.StereoEncoder(embed_dim=32)
    batches=[]
    hook=network.backbone.register_forward_pre_hook(lambda model,args:batches.append(args[0].shape[0]))
    left=torch.rand(1,3,32,32); right=torch.rand_like(left)
    tokens,valid=network(left,right,torch.tensor([[True,False]]))
    hook.remove()
    assert batches==[2]
    assert torch.isfinite(tokens).all() and tokens[~valid].eq(0.).all()
    assert sum(isinstance(layer,torch.nn.Dropout) and layer.p>0 for layer in network.modules())==0


def test_temporal_order_time_and_padding_are_explicit_and_train_mode_deterministic():
    temporal=module('temporal'); network=temporal.StereoTemporal(embed_dim=32)
    frames=torch.randn(1,4,32); valid=torch.ones(1,4,dtype=torch.bool)
    a=network(frames,valid); b=network(frames.flip(1),valid)
    assert not torch.equal(a,b)
    torch.testing.assert_close(network(frames,valid),a,atol=0,rtol=0)
    timed=network(frames,valid,dt=torch.full((1,4),.2),image_age=torch.full((1,4),.1))
    assert not torch.equal(a,timed)
    valid[:,0]=False; changed=frames.clone();changed[:,0]=100.
    torch.testing.assert_close(network(frames,valid),network(changed,valid),atol=0,rtol=0)
    with pytest.raises(ValueError,match='finite'):
        network(frames,valid,dt=torch.full((1,4),float('nan')))
    with pytest.raises(ValueError,match='finite'):
        module('encoder').StereoEncoder(embed_dim=32)(torch.full((1,3,32,32),float('nan')),
            torch.zeros(1,3,32,32),torch.ones(1,2,dtype=torch.bool))


def test_estimator_head_shapes_finite_and_differentiable_without_truth_input():
    m = module('state_estimator'); network = m.StereoStateEstimator(embed_dim=32).eval()
    output = network(history())
    expected = dict(latent=(2,32), privileged_state=(2,371), geometry=(2,128,3),
        velocity=(2,128,3), deformation=(2,128), contact_logits=(2,2,3),
        force_proxy=(2,2,3), slip_logits=(2,2), slip_velocity=(2,2))
    for key, shape in expected.items():
        assert output[key].shape == shape and output[key].dtype == torch.float32
        assert output[key].device.type == 'cpu' and torch.isfinite(output[key]).all()
    sum(value.square().mean() for value in output.values()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    with pytest.raises((ValueError, TypeError)):
        network({'history': history(), 'privileged_state': torch.rand(2, 371)})


def test_invalid_points_and_padding_never_influence_estimator_output():
    m = module('state_estimator'); network = m.StereoStateEstimator(embed_dim=32).eval()
    a = history(); a.point_valid[:, -1, :8] = False
    b = a.clone(); b.points_world[:, -1, :8] = float('nan')
    b.left_rgb[:, :3] = .9; b.proprio[:, :3] = 9.
    with torch.no_grad():
        x, y = network(a), network(b)
    for key in x:
        torch.testing.assert_close(x[key], y[key], atol=0, rtol=0)


def test_effective_views_change_features_and_all_invalid_input_remains_finite():
    m = module('state_estimator'); network = m.StereoStateEstimator(embed_dim=32).eval()
    a = history(); b = a.clone(); b.right_rgb[:, -1] = 1.-b.right_rgb[:, -1]
    with torch.no_grad():
        x, y = network(a), network(b)
    assert not torch.equal(x['latent'], y['latent'])
    c = a.clone(); c.image_valid.zero_(); c.point_valid.zero_(); c.depth_valid.zero_()
    c.points_world.fill_(float('nan')); c.depth_m.fill_(float('nan'))
    with torch.no_grad(): out = network(c)
    assert all(torch.isfinite(value).all() for value in out.values())
    c.history_valid.zero_()
    with torch.no_grad(): out = network(c)
    assert out['latent'].eq(0.).all()


def test_student_actor_mean_deterministic_and_canonical_sha_enforcement():
    m = module('state_estimator')
    from go2_pvcnn.control.dh116.tabletop_vision_command_shadow import action_contract_sha256
    actor = m.StereoStudentActor(embed_dim=32, action_contract_sha256=action_contract_sha256()).eval()
    with torch.no_grad():
        mean = actor.latent_mean(history()); direct = actor(history())
    assert mean.shape == direct.shape == (2,24) and actor.log_std.shape == (24,)
    fixed = history()
    torch.testing.assert_close(actor.deterministic(fixed), actor.latent_mean(fixed).tanh())
    with pytest.raises(ValueError, match='SHA|sha|contract'):
        m.StereoStudentActor(embed_dim=32, action_contract_sha256='0'*64)
    with pytest.raises((ValueError, TypeError)):
        actor({'policy': fixed, 'critic': torch.randn(2,371)})


def test_batch_one_training_uses_no_batch_statistics_or_dropout():
    m=module('state_estimator'); actor=m.StereoStudentActor(embed_dim=32).train()
    batch=history(1)
    first=actor(batch);second=actor(batch)
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    assert not any(isinstance(layer,torch.nn.BatchNorm2d) for layer in actor.modules())
    first.sum().backward()
    assert actor.estimator.encoder.backbone.stem[0].weight.grad.abs().sum()>0
    from go2_pvcnn.control.dh116.tabletop_squashed_gaussian import squashed_log_prob
    std=actor.log_std.exp().expand_as(first)
    assert torch.isfinite(squashed_log_prob(first+.1,first,std)).all()


def test_geodesic_quaternion_sign_and_nonzero_error():
    m = module('losses'); q = torch.tensor([[1.,0.,0.,0.]], requires_grad=True)
    mask = torch.tensor([True])
    same = m.quaternion_geodesic_loss(q, -q.detach(), mask)
    assert float(same.detach()) == pytest.approx(0., abs=1e-6)
    half_turn = m.quaternion_geodesic_loss(q, torch.tensor([[0.,1.,0.,0.]]), mask)
    assert half_turn > 3.
    half_turn.backward(); assert torch.isfinite(q.grad).all()


def test_masked_reduction_ignores_nan_invalid_and_all_invalid_has_zero_gradient():
    m = module('losses'); prediction = torch.tensor([[.2,.3]], requires_grad=True)
    target = torch.tensor([[float('nan'),float('nan')]])
    loss = m.masked_smooth_l1(prediction, target, torch.zeros(1,2,dtype=torch.bool))
    assert torch.isfinite(loss) and loss.item() == 0.
    loss.backward(); assert prediction.grad.eq(0.).all()


def test_geometry_classification_and_slip_losses_propagate_only_valid_gradients():
    m=module('losses'); loss_fn=m.StereoEstimationLoss()
    shapes=dict(geometry=(2,128,3),velocity=(2,128,3),deformation=(2,128),
        contact_logits=(2,2,3),slip_logits=(2,2),slip_velocity=(2,2))
    predictions={name:torch.full(shape,.2,requires_grad=True) for name,shape in shapes.items()}
    targets={name:torch.zeros(shape) for name,shape in shapes.items()}
    valid={name:torch.ones(shape,dtype=torch.bool) for name,shape in shapes.items()}
    for name in shapes:
        valid[name][0]=False;targets[name][0]=float('nan')
    result=loss_fn(predictions,targets,valid);result['total'].backward()
    for name,prediction in predictions.items():
        assert torch.isfinite(prediction.grad).all()
        assert prediction.grad[0].eq(0.).all() and prediction.grad[1].abs().sum()>0
        assert result['coverage'][name]==.5


def test_privileged_loss_contract_excludes_internal_unavailable_and_checks_width():
    m = module('losses'); loss_fn = m.StereoEstimationLoss()
    contract = json.loads((ROOT/'docs/contracts/stereo_privileged_state_contract.json').read_text(encoding='utf-8'))
    forbidden = [r['index'] for r in contract['dimensions'] if not r['deployment_allowed']
        or r['loss_type']=='none' or r['supervision_status']=='NA' or r['student_source']=='unobservable']
    pred = torch.randn(2,371,requires_grad=True); target = torch.zeros_like(pred)
    target[:,156] = 1.; target[:,163] = 1.; target[:,64] = 1.
    source_keys={key for row in contract['dimensions'] for key in row['required_validity_keys']}
    sources={key:torch.ones(2,dtype=torch.bool) for key in source_keys}
    sources['rotation_symmetry_valid']=torch.ones(2,dtype=torch.bool)
    result = loss_fn({'privileged_state':pred}, {'privileged_state':target},
        {'privileged_state':torch.ones(2,371,dtype=torch.bool), 'state_sources':sources})
    result['total'].backward()
    assert pred.grad[:, forbidden].eq(0.).all()
    assert pred.grad[:,0].abs().sum()>0
    blocked=loss_fn({'privileged_state':pred}, {'privileged_state':target},
        {'privileged_state':torch.ones(2,371,dtype=torch.bool)})
    assert blocked['total'].item()==0.
    with pytest.raises(ValueError, match='371|shape|dimension'):
        loss_fn({'privileged_state':pred[:, :370]}, {'privileged_state':target},
            {'privileged_state':torch.ones(2,371,dtype=torch.bool)})


def test_force_requires_per_label_native_source_contact_and_attribution_masks(monkeypatch):
    m = module('losses'); loss_fn = m.StereoEstimationLoss()
    prediction = torch.ones(2,2,3,requires_grad=True); targets={'force_proxy':torch.zeros(2,2,3)}
    validity={'force_proxy':torch.ones(2,2,3,dtype=torch.bool)}
    no_source=loss_fn({'force_proxy':prediction},targets,validity)
    assert no_source['total'].item()==0.
    qualified={key:torch.ones(2,2,3,dtype=torch.bool) for key in ('source_valid','contact_valid','attributable')}
    qualified['source_valid'][0]=False
    qualified.update(source_sha256='a'*64, units='N', coordinate_frame='accepted_pad_normal',
        partner_attribution='verified_object_partner',
        per_digit_source=[['native_effective_pad']*3]*2,
        sign_convention='compressive_positive_on_object', evidence_scope='cpu_fixture')
    fixture_result=loss_fn({'force_proxy':prediction},targets,validity,force_qualification=qualified)
    assert fixture_result['total'].item()==0.
    assert fixture_result['coverage']['force_proxy']==0.
    # Metadata without a verified proof must remain blocked, even if a caller
    # writes native_qualified into its dictionary.
    qualified['evidence_scope']='native_qualified'
    metadata_only=loss_fn({'force_proxy':prediction},targets,validity,force_qualification=qualified)
    assert metadata_only['total'].item()==0. and metadata_only['coverage']['force_proxy']==0.
    # Only replace the proof verifier to isolate CPU mask/loss arithmetic.
    # This mock grants no actual native qualification or persisted artifact.
    monkeypatch.setattr(m,'_force_proof_verified',lambda binding:True,raising=False)
    result=loss_fn({'force_proxy':prediction},targets,validity,force_qualification=qualified)
    result['total'].backward()
    assert prediction.grad[0].eq(0.).all() and prediction.grad[1].abs().sum()>0
    missing=dict(qualified); missing.pop('source_sha256')
    assert loss_fn({'force_proxy':prediction},targets,validity,
        force_qualification=missing)['total'].item()==0.
    with pytest.raises((ValueError, TypeError), match='shape|mask|tensor|qualification'):
        loss_fn({'force_proxy':prediction},targets,validity,force_qualification={'source_valid':True})


def test_privileged_quaternion_groups_never_receive_raw_component_huber():
    m=module('losses'); loss_fn=m.StereoEstimationLoss()
    target=torch.zeros(1,371); prediction=target.clone()
    for start in (64,156,163): target[:,start]=1.;prediction[:,start]=-1.
    prediction.requires_grad_()
    sources={key:torch.ones(1,dtype=torch.bool) for key in loss_fn.required}
    sources['rotation_symmetry_valid']=torch.ones(1,dtype=torch.bool)
    result=loss_fn({'privileged_state':prediction},{'privileged_state':target},
        {'privileged_state':torch.ones(1,371,dtype=torch.bool),'state_sources':sources})
    assert float(result['total'].detach())==pytest.approx(0.,abs=1e-6)
    result['total'].backward();assert torch.isfinite(prediction.grad).all()


def test_metrics_report_na_for_invalid_fields_instead_of_zero_error():
    m=module('losses')
    metrics=m.per_dimension_metrics(torch.tensor([[1.,2.],[3.,4.]]),
        torch.tensor([[0.,float('nan')],[0.,float('nan')]]),
        torch.tensor([[True,False],[True,False]]))
    assert metrics['count'].tolist()==[2,0]
    assert metrics['mae'][0]==2. and torch.isnan(metrics['mae'][1])
    assert metrics['p95'][0]==pytest.approx(2.9)


def test_versioned_configs_bind_canonical_action_and_exact_state_contract():
    import hashlib
    m=module('losses')
    from go2_pvcnn.control.dh116.tabletop_vision_command_shadow import action_contract_sha256
    network=json.loads((ROOT/'configs/dh116_training/stereo_network_v1.json').read_text())
    state=json.loads((ROOT/'configs/dh116_training/stereo_state_estimator_v1.json').read_text())
    assert network['action_contract_sha256']==action_contract_sha256()
    assert network['backbone']['stages']==[2,2,2,2]
    assert network['backbone']['pretrained'] is False and network['dropout']==0.
    assert state['force']['default_valid'] is False
    assert state['force']['coordinate_frame']=='accepted_pad_normal'
    path=ROOT/state['state_contract_path']
    actual=module('contract').verify_estimator_state_contract_file(
        path, state['state_contract_file_sha256'])
    assert actual==hashlib.sha256(path.read_bytes()).hexdigest()
    assert m.StereoEstimationLoss(expected_contract_sha256=state['state_contract_semantic_sha256'])
    with pytest.raises(ValueError,match='SHA'):
        m.StereoEstimationLoss(expected_contract_sha256='0'*64)


@pytest.mark.parametrize('variant', ('lf', 'crlf', 'mixed', 'extra_space', 'mutated_field'))
def test_estimator_state_contract_only_accepts_audited_checkout_bytes(tmp_path, variant):
    import hashlib
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import verify_estimator_state_contract_file

    state=json.loads((ROOT/'configs/dh116_training/stereo_state_estimator_v1.json').read_text())
    expected=state['state_contract_file_sha256']
    lf=(ROOT/state['state_contract_path']).read_bytes().replace(b'\r\n', b'\n')
    variants={
        'lf': lf,
        'crlf': lf.replace(b'\n', b'\r\n'),
        'mixed': lf.replace(b'\n', b'\r\n', 1),
        'extra_space': lf + b' ',
        'mutated_field': lf.replace(b'371', b'372', 1),
    }
    path=tmp_path/'state.json'
    raw=variants[variant]
    path.write_bytes(raw)
    if variant in ('lf', 'crlf'):
        assert verify_estimator_state_contract_file(path, expected)==hashlib.sha256(raw).hexdigest()
    else:
        with pytest.raises(ValueError, match='state contract file SHA256 mismatch'):
            verify_estimator_state_contract_file(path, expected)


def test_estimator_state_contract_does_not_accept_unknown_pin(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import verify_estimator_state_contract_file

    path=tmp_path/'state.json'
    path.write_bytes((ROOT/'docs/contracts/stereo_privileged_state_contract.json').read_bytes())
    with pytest.raises(ValueError, match='state contract file SHA256 mismatch'):
        verify_estimator_state_contract_file(path, 'a'*64)
