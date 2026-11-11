"""CPU regressions for the estimator-only path; fixtures carry no native proof."""
import pytest
import torch

from test_dh116_stereo_estimator import history, module


def identity(initial, group):
    return {'identity': {'physical_initial_condition_sha256': initial*64,
                         'source_episode_group': group}}


def test_group_split_is_three_way_and_transitive():
    rows = [identity('a', 'shared'), identity('b', 'shared'),
            identity('b', 'another-run'), identity('c', 'val'), identity('d', 'test')]
    split = module('state_supervision').split_estimator_rows(rows, seed=116)
    partitions = [split[key+'_indices'] for key in ('train', 'validation', 'test')]
    assert all(partitions)
    assert sorted(sum(partitions, [])) == list(range(len(rows)))
    assert any({0, 1, 2}.issubset(partition) for partition in map(set, partitions))


def test_adjacent_frames_cannot_create_a_third_independent_split():
    rows = [identity('a', 'train')]*20 + [identity('b', 'second')]*20
    split = module('state_supervision').split_estimator_rows(rows, seed=116)
    assert split['independent_groups'] == 2
    assert split['status'] == 'blocked_insufficient_independent_groups'
    assert not any(split[k+'_indices'] for k in ('train', 'validation', 'test'))


def test_exact_field_audit_distinguishes_pass_through_schema_only_and_na():
    audit = module('state_supervision').audit_candidate_fields()
    assert audit['candidate_count'] == 143
    assert len(audit['fields']) == len({x['index'] for x in audit['fields']}) == 143
    assert all(x['actor_eligible'] and x['validity_keys'] for x in audit['fields'])
    assert sum(x['supervision_role'] == 'direct_sensor_or_command' for x in audit['fields']) == 48
    assert sum(x['supervision_role'] == 'vision_prediction' for x in audit['fields']) == 95
    assert not any(x['supervision_samples'] for x in audit['fields'])
    assert not any(x['uncertainty_calibrated'] for x in audit['fields'])
    assert next(x for x in audit['fields'] if x['index'] == 68)['minimum_history_frames'] == 2


def test_invalid_reported_task_value_is_na_while_actor_input_remains_finite():
    adapter = module('task_state_adapter').TaskStateAdapter(allowlist=[61])
    output = adapter({'privileged_state': torch.ones(1,371)},history(1))
    assert torch.isnan(output['values']).all()
    assert not output['valid'].any()
    assert output['actor_features'].tolist() == [[0.,0.,1.]]


def test_untrained_estimator_does_not_emit_numeric_uncertainty_as_a_prediction():
    model = module('state_estimator').StereoStateEstimator(embed_dim=32, task_state_metadata=True)
    with torch.no_grad():
        estimates = model(history(1))
    assert 'state_valid' in estimates
    assert 'state_uncertainty' not in estimates


def test_rgb_depth_proxy_and_finite_privileged_state_do_not_qualify_visual_labels():
    supervision = module('state_supervision')
    rows = []
    for i, initial in enumerate('abc'):
        row = identity(initial, str(i))
        row.update(history=history(1).as_dict(), privileged=torch.ones(371),
                   privileged_valid=torch.ones(371, dtype=torch.bool),
                   state_sources={key: torch.ones(371, dtype=torch.bool) for key in supervision.SOURCE_KEYS},
                   extra_labels={'training_label_source_scope':'simulation_SDK_encoder_readback_only',
                                 'geometry':torch.ones(128,3), 'force_proxy':torch.ones(2,3)},
                   label_validity={'geometry':torch.ones(128,3,dtype=torch.bool),
                                   'force_proxy':torch.ones(2,3,dtype=torch.bool)}, force_qualification=None)
        rows.append(row)
    split = supervision.split_estimator_rows(rows)
    assert all(not report['enabled'] for report in supervision.inspect_heads(rows, split).values())


def test_encoder_only_candidate_can_admit_exact_geometry_after_independent_reference():
    supervision = module('state_supervision')
    row = dict(history=history(1).as_dict(), privileged=torch.zeros(371),
        privileged_valid=torch.arange(371) < 24, state_sources={},
        extra_labels={'training_label_source_scope':'simulation_SDK_encoder_readback_only'})
    exact = torch.zeros(371,dtype=torch.bool)
    exact[61:64] = True
    exact[68:74] = True
    supervision._admit_rigid_labels(row,exact,wire_observed=True,temporal_observed=False)
    criterion = module('losses').StereoEstimationLoss()
    _, mask = supervision._targets(row,'geometry',criterion)['privileged_state']
    assert mask.nonzero().flatten().tolist() == [61,62,63]
    assert not supervision._targets(row,'velocity',criterion)
    assert row['privileged_valid'][:24].all()
    assert not row['privileged_valid'][68:74].any()


def test_observed_wire_requires_valid_sampled_pixel_and_reference_agreement():
    supervision = module('state_supervision')
    point_pixels = torch.tensor([[1,1], [1,2], [-1,-1]])
    point_valid = torch.tensor([True,True,False])
    rgb_depth = torch.ones(4,4)
    optical_z = rgb_depth.clone()
    optical_z[1,2] = 2
    mask = torch.zeros(4,4,dtype=torch.bool)
    mask[1,2] = True
    args = (point_pixels,point_valid,rgb_depth,rgb_depth.bool(),optical_z)
    assert not supervision._observed_wire_points(*args,mask).any()
    mask[1,1] = True
    assert supervision._observed_wire_points(*args,mask).tolist() == [True,False,False]


def test_group_calibration_has_na_when_only_two_independent_groups():
    result = module('state_supervision').calibrate_residuals({'a':[.1,.2], 'b':[.3]})
    assert result['calibrated'] is False and result['radius'] is None
    result = module('state_supervision').calibrate_residuals({str(i):[i/100] for i in range(20)})
    assert result['radius'] == pytest.approx(.19)


def test_legacy_teacher_action_artifact_cannot_supply_missing_stereo_labels(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    path = tmp_path/'privileged_candidate_trajectories.pt'
    raw = tmp_path/'native_physics.jsonl'
    torch.save(dict(version='dh116-privileged-candidate-collection-v1', samples=[]), path)
    raw.write_text('{}\n', encoding='utf-8')
    with pytest.raises(ValueError,match='RGB history and label masks'):
        module('state_supervision').read_candidate(path, raw=raw, sha256=file_sha256(path), raw_sha256=file_sha256(raw))


def test_raw_candidate_must_include_authoritative_camera_frame_and_calibration(tmp_path):
    import json
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import physical_initial_sha256
    initial = dict(coordinate_frame='environment_local',object_pose=[0,0,0,1,0,0,0],
        palm_pose=[[0,0,0,1,0,0,0]]*2,joint_position=[0]*12,joint_velocity=[0]*12,object_velocity=[0]*6)
    keys = dict(run_uuid='CPU-fixture',env_id=0,episode_uid='episode',reset_id=0)
    ident = dict(**keys, policy_tick=0, physics_tick_end=0, simulation_timestamp=0., frame_id=3,
                 calibration_sha256='c'*64,physical_initial_condition_sha256=physical_initial_sha256(initial),
                 source_episode_group='fixture-group')
    sample = dict(**keys, kind='sample', owner='teacher', action_owner='teacher', checkpoint_sha256='a'*64,
                  actor_issued_action=[0.]*24, policy_tick=0, physics_step=1, timestamp=1/480)
    header = dict(**keys, kind='episode_start', initial_measurement_valid=True, source_episode_group='fixture-group',
                  physical_initial_condition_source='actual_sdk_readback', physical_initial_condition=initial)
    raw = tmp_path/'raw.jsonl'
    raw.write_text(json.dumps(header)+'\n'+json.dumps(sample)+'\n', encoding='utf-8')
    rows = [dict(identity=ident, issued_actions=torch.zeros(24), done=False,
                 execution=dict(physics_tick_end=1,simulation_timestamp_end=1/480))]
    with pytest.raises(ValueError,match='identity'):
        module('state_supervision')._raw_bindings(raw,rows,'a'*64)


def test_candidate_raw_identity_uses_existing_camera_packet_and_rejects_stale_frame():
    from types import SimpleNamespace
    bridge = object.__new__(module('native').NativeTensorBridge)
    identity = dict(run_uuid='cpu-fixture',env_id=0,episode_uid='e',reset_id=1,policy_tick=4,
        physics_tick_start=32,physics_tick_end=32,simulation_timestamp=32/480,calibration_sha256='c'*64)
    row = dict(identity,frame_id=10)
    bridge.n = 1
    bridge._in_step = False
    bridge.execution_owner = 'teacher'
    bridge.checkpoint_sha256 = 'd'*64
    bridge.camera = SimpleNamespace(calibration=SimpleNamespace(sha256='c'*64),last_frame=10)
    bridge.pre_action_identities = lambda sha: [identity]
    packet = dict(identity=[row],frame_id=10,calibration_sha256='c'*64,simulation_time=32/480)
    bridge.bind_stereo_candidate_frame(packet,[row])
    assert bridge._stereo_candidate_pre_action == [row]
    bridge.camera.last_frame = 11
    with pytest.raises(ValueError,match='camera'):
        bridge.bind_stereo_candidate_frame(packet,[row])


def test_estimator_training_uses_only_supervised_head_and_grouped_heldouts():
    supervision = module('state_supervision')
    rows = []
    for i, initial in enumerate('abc'):
        row = identity(initial, str(i))
        valid = torch.zeros(371, dtype=torch.bool)
        valid[61] = True
        target = torch.zeros(371)
        target[61] = 2.
        row.update(history=history(1).as_dict(), privileged=target, privileged_valid=valid,
                   state_sources={key: torch.ones(371, dtype=torch.bool) for key in supervision.SOURCE_KEYS},
                   extra_labels={'training_label_source_scope':'arithmetic_fixture'}, label_validity={})
        rows.append(row)
    torch.manual_seed(116)
    model = module('state_estimator').StereoStateEstimator(embed_dim=32,num_layers=1)
    head_before = model.heads['privileged_state'].weight.detach().clone()
    deform_before = model.heads['deformation'].weight.detach().clone()
    split = supervision.split_estimator_rows(rows)
    result = supervision.fit_state_heads(model, rows, split, epochs=1, batch_size=1)
    assert [key for key, report in result.items() if report['enabled']] == ['geometry']
    assert not torch.equal(head_before[61],model.heads['privileged_state'].weight[61])
    torch.testing.assert_close(head_before[62:], model.heads['privileged_state'].weight[62:])
    torch.testing.assert_close(deform_before, model.heads['deformation'].weight)
    stats = result['geometry']
    assert stats['optimization'][0]['gradient_norm'] > 0
    assert stats['splits']['test']['outputs']['privileged_state']['valid_count'] == 1
    assert stats['uncertainty_calibration']['privileged_state']['radius'] is None
