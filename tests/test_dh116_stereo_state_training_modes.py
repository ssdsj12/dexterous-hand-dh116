"""Mode and gradient arithmetic fixtures are not native qualification."""
import copy
import pytest
import torch

from test_dh116_stereo_estimator import history, module
from test_dh116_stereo_state_supervision import identity


def center_rows():
    rows = []
    for group in 'ab':
        for tick in range(2):
            row = identity(group, group)
            mask = torch.zeros(371, dtype=torch.bool)
            mask[61:64] = True
            target = torch.full((371,), float('nan'))
            target[61:64] = torch.tensor([.1, -.2, .3])
            row.update(history=history(1).as_dict(), privileged=target, privileged_valid=mask,
                state_sources={key: mask.clone() for key in module('state_supervision').SOURCE_KEYS},
                verified_canonical_native=mask.clone(),
                extra_labels={'training_label_source_scope':'independently_matched_rigid_state_readback'},
                label_validity={})
            rows.append(row)
    return rows


def estimator():
    torch.manual_seed(116)
    return module('state_estimator').StereoStateEstimator(embed_dim=32, num_layers=1)


def test_schema_test_does_not_train_or_create_qualification():
    model = estimator()
    before = copy.deepcopy(model.state_dict())
    report = module('state_supervision').run_training_mode(model, center_rows(), mode='schema_test')
    assert report['status'] == 'schema_validated'
    assert report['state_training_started'] is False
    assert report['generalization_qualified'] is False
    assert report['successful_demonstration_pool'] is False
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_diagnostic_fit_two_groups_has_real_gradient_only_on_center_xyz():
    rows = center_rows()
    model = estimator()
    before = copy.deepcopy(model.state_dict())
    report = module('state_supervision').run_training_mode(model, rows, mode='diagnostic_fit', epochs=1, batch_size=2)
    assert report['status'] == 'diagnostic_fit_completed'
    assert report['split']['independent_groups'] == 2
    assert report['generalization_qualified'] is False
    assert report['uncertainty_calibrated'] is False
    geometry = report['heads']['geometry']
    assert geometry['supervised_indices'] == [61, 62, 63]
    assert geometry['training_rows'] == 4 and geometry['training_independent_groups'] == 2
    assert geometry['optimization'][0]['gradient_norm'] > 0
    assert geometry['optimization'][0]['encoder_gradient_norm'] > 0
    assert geometry['optimization'][0]['unsupervised_output_gradient_max'] == 0
    assert geometry['metrics_scope'] == 'in_sample_diagnostic_only'
    assert geometry['splits']['validation'] is None and geometry['splits']['test'] is None
    metric = geometry['splits']['train']['outputs']['371:61']
    assert metric['physical_mae_m'] == pytest.approx(metric['mae']*.05)
    assert metric['valid_count'] == 4
    assert geometry['label_audit'][0]['target_m'] == pytest.approx([.005,-.01,.015])
    assert not torch.equal(before['heads.privileged_state.weight'][61:64],model.heads['privileged_state'].weight[61:64])
    torch.testing.assert_close(before['heads.privileged_state.weight'][:61],model.heads['privileged_state'].weight[:61],atol=0,rtol=0)
    torch.testing.assert_close(before['heads.privileged_state.weight'][64:],model.heads['privileged_state'].weight[64:],atol=0,rtol=0)
    assert all(not report['heads'][name]['enabled'] for name in ('velocity','deformation','contact','slip','force_proxy'))


def test_formal_training_cannot_reuse_two_groups_as_video_heldouts():
    model = estimator()
    before = copy.deepcopy(model.state_dict())
    report = module('state_supervision').run_training_mode(model, center_rows(),mode='formal_training')
    assert report['status'] == 'blocked_insufficient_independent_groups'
    assert report['state_training_started'] is False
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())


def test_diagnostic_fit_requires_independently_verified_label_masks():
    rows = center_rows()
    for row in rows:
        row.pop('verified_canonical_native')
    report = module('state_supervision').run_training_mode(estimator(),rows,mode='diagnostic_fit')
    assert report['status'] == 'blocked_no_verified_center_labels'
    assert report['state_training_started'] is False


def test_diagnostic_fit_leaves_missing_labels_na_and_does_not_admit_velocity():
    rows = center_rows()
    for row in rows:
        row['verified_canonical_native'][62] = False
        row['privileged'][62] = float('nan')
        row['verified_canonical_native'][68:71] = True
        row['privileged'][68:71] = 0
    report = module('state_supervision').run_training_mode(estimator(),rows,mode='diagnostic_fit',epochs=1)
    geometry = report['heads']['geometry']
    assert geometry['supervised_indices'] == [61,63]
    assert geometry['label_audit'][0]['target_normalized'][1] is None
    assert geometry['splits']['train']['outputs']['371:62']['mae'] is None
    assert not report['heads']['velocity']['enabled']


def test_unknown_mode_is_rejected_before_training():
    with pytest.raises(ValueError,match='mode'):
        module('state_supervision').run_training_mode(estimator(),center_rows(),mode='fit')


def test_separate_mode_configs_cannot_override_qualification(tmp_path):
    import json
    from pathlib import Path
    supervision = module('state_supervision')
    root = Path(__file__).resolve().parents[1]
    for mode in (*supervision.TRAINING_MODES, 'formal_training'):
        config = supervision.read_training_config(root/'configs/dh116_training'/('stereo_state_'+mode+'_v1.json'))
        assert config['mode'] == mode
    wrong = tmp_path/'wrong.json'
    wrong.write_text(json.dumps(dict(version='dh116-stereo-state-training-mode-v1',settings={
        'mode':'diagnostic_fit','generalization_qualified':True})))
    with pytest.raises(ValueError,match='setting'):
        supervision.read_training_config(wrong)


def test_admitting_geometry_preserves_existing_direct_encoder_audit():
    supervision = module('state_supervision')
    row = center_rows()[0]
    row['extra_labels']['training_label_source_scope'] = 'simulation_SDK_encoder_readback_only'
    row['privileged_valid'][:24] = True
    supervision._admit_rigid_labels(row,row['verified_canonical_native'],wire_observed=True,temporal_observed=False)
    audit = supervision.audit_candidate_fields([row])
    assert all(x['direct_readback_samples'] == 1 for x in audit['fields'] if x['index'] < 24)
    assert all(x['supervision_samples'] == 1 for x in audit['fields'] if x['index'] in (61,62,63))


def test_eight_cold_initials_cannot_enter_formal_fit_even_with_valid_center_labels():
    rows = center_rows()
    cohort = []
    for index in range(8):
        row = copy.deepcopy(rows[index % len(rows)])
        row['identity'].update(physical_initial_condition_sha256=hex(index+1)[2:]*64,
                               source_episode_group=f'cold-{index}')
        row['dataset_usage'] = 'diagnostic_only_cold_cohort'
        cohort.append(row)
    model = estimator()
    before = copy.deepcopy(model.state_dict())
    report = module('state_supervision').run_training_mode(model, cohort, mode='formal_training')
    assert report['status'] == 'blocked_diagnostic_only_cold_cohort'
    assert report['split']['independent_groups'] == 8
    assert not report['state_training_started']
    assert report['stages']['center_xyz']['eligible_independent_groups'] == 8
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_velocity_needs_contiguous_native_clocks_and_the_same_centerline_track():
    supervision = module('state_supervision')
    prior = dict(identity=dict(run_uuid='run', env_id=0, episode_uid='epoch', reset_id=3,
        policy_tick=10, frame_id=200, physics_tick_end=80, simulation_timestamp=80/480,
        calibration_sha256='c'*64, observation_preprocessor_sha256='b'*64),
        execution=dict(physics_tick_end=88, simulation_timestamp_end=88/480), done=False)
    current = dict(prior['identity'], policy_tick=11, frame_id=203, physics_tick_end=88,
                   simulation_timestamp=88/480)
    assert supervision._temporal_support(prior, current, {5}, {5})
    assert not supervision._temporal_support(prior, current, {4}, {5})
    assert not supervision._temporal_support(prior, dict(current, physics_tick_end=89), {5}, {5})
    assert not supervision._temporal_support(prior, dict(current, reset_id=4), {5}, {5})
    assert not supervision._temporal_support(prior, dict(current, frame_id=200), {5}, {5})
    assert not supervision._temporal_support(dict(prior, done=True), current, {5}, {5})


def test_state_reference_binding_is_same_clock_and_label_only():
    reference = module('reference')
    identity = dict(frame_id=12, reset_id=2, physics_tick_start=16, physics_tick_end=16,
                    simulation_timestamp=16/480, calibration_sha256='a'*64)
    proof = dict(frame_id=12, simulation_time=16/480, calibration_sha256='a'*64,
                 raw_instance_ids=torch.zeros((1, 3, 4), dtype=torch.long),
                 right_raw_instance_ids=None, source_scope='native', validation_only=True,
                 reference_used_as_actor_input=False, source_units='optical_axis_z_m')
    binding = reference.state_supervision_binding(identity, proof, 'd'*64)
    assert binding['depth_input_source'] == 'rgb_stereo'
    assert binding['reference_role'] == 'labels_and_audit_only'
    assert binding['left_native_ids_available'] and not binding['right_native_ids_available']
    with pytest.raises(ValueError, match='same-frame'):
        reference.state_supervision_binding(identity, dict(proof, frame_id=13), 'd'*64)
