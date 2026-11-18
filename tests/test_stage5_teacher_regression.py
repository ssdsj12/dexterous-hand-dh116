import importlib.util
import copy
from pathlib import Path

import numpy as np
import pytest
import torch


path = Path(__file__).resolve().parents[1] / 'tools' / 'audit_stage5_teacher_regression.py'
spec = importlib.util.spec_from_file_location('teacher_regression_audit', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_initial_tensor_identity_is_bitwise_including_signed_zero_and_dtype():
    original = {'q': torch.tensor([0.]), 'seed': 19, 'source': 'synthetic_fixture'}
    assert module.compare_initial_states(original, original)['bitwise_equal']
    changed = dict(original, q=torch.tensor([-0.]))
    assert not module.compare_initial_states(original, changed)['bitwise_equal']
    changed = dict(original, q=original['q'].double())
    assert not module.compare_initial_states(original, changed)['bitwise_equal']


def action_data(length):
    return dict(actions=torch.zeros((length, 24)), sample_metadata=[
        dict(env_id=0, episode_id=0, tick=i) for i in range(length)])


def test_issued_comparison_uses_common_live_prefix_and_detects_exact_channel_change():
    reference, adapter = action_data(3), action_data(2)
    result = module.compare_action_tapes(reference, adapter)
    assert result['bitwise_equal'] and result['common_samples'] == 2
    adapter['actions'][1, 17] = .01
    result = module.compare_action_tapes(reference, adapter)
    assert not result['bitwise_equal']
    assert result['first_mismatch']['key'] == [0, 0, 1]
    assert result['first_mismatch']['differing_dimensions'] == [17]


def test_supervision_tick_duplicates_and_gaps_are_rejected():
    reference, adapter = action_data(3), action_data(3)
    adapter['sample_metadata'][1]['tick'] = 0
    with pytest.raises(ValueError, match='tick'):
        module.compare_action_tapes(reference, adapter)


def physical_fixture():
    fields = ['env_id', 'episode_id', 'physics_step', 'timestamp_s', 'teacher_control_step',
        'live', 'newly_terminal', 'native_success', 'failure_code', 'object_net_lift_m',
        'native_hold_s', 'native_anchor_drift_m', 'native_anchor_drift_valid',
        'left_palm_dz_m', 'right_palm_dz_m', 'left_load_n', 'right_load_n',
        'left_contact_valid', 'right_contact_valid', 'left_contact_measurement_valid',
        'right_contact_measurement_valid', 'right_sdk_target_native_readback',
        'sdk_target_right_finger12_rad', 'actual_q_right_finger12_rad']
    c = {name: i for i, name in enumerate(fields)}
    rows = np.zeros((4, len(fields)))
    rows[:, c['physics_step']] = [1, 2, 3, 4]
    rows[:, c['timestamp_s']] = np.arange(1, 5) / 480
    rows[:, c['live']] = [1, 1, 1, 0]
    rows[:, c['newly_terminal']] = [0, 0, 1, 0]
    rows[:, c['native_success']] = [0, 0, 1, 0]
    rows[:, c['object_net_lift_m']] = [.01, .03, .04, 99.]
    rows[:, c['native_hold_s']] = [0, 1, 5, 0]
    rows[:, c['native_anchor_drift_m']] = np.nan
    rows[:, c['right_palm_dz_m']] = [.001, .002, .003, 99.]
    for hand in ('left', 'right'):
        rows[:, c[hand+'_contact_valid']] = 1
        rows[:, c[hand+'_contact_measurement_valid']] = 1
    return rows, c


def test_first_terminal_retained_and_missing_anchor_slip_is_unknown():
    rows, c = physical_fixture()
    record = module.first_episode_metrics(rows, c, {'env_id': 0, 'episode_id': 0,
        'supervision_episode_id': 0, 'episode_uid': 'fixture:0:0'})
    assert record['physical_samples'] == 3
    assert record['lift_peak_mm'] == 40.
    assert record['hold_max_s'] == 5.
    assert record['slip_max_mm'] is None and record['slip_valid_samples'] == 0
    assert record['right_palm_peak']['physics_step'] == 3
    assert record['right_palm_peak']['value_mm'] == 3.
    assert record['right_finger12_tracking']['native_readback_samples'] == 0


def test_legacy_v1_native_flag_is_explicit_cache_provenance_and_not_backend_evidence():
    rows, c = physical_fixture()
    rows[:, c['right_sdk_target_native_readback']] = 1
    record = module.first_episode_metrics(rows, c, {'env_id': 0, 'episode_id': 0})
    assert record['right_finger12_tracking']['native_readback_samples'] == 0
    assert record['right_finger12_tracking']['command_cache_readback_samples'] == 3
    assert record['target_provenance']['sdk_target_columns_source'] == 'legacy_IsaacLab_command_cache_or_reconstruction;not_PhysX_backend'


def v2_target_rows():
    rows, columns = physical_fixture()
    added = ['right_sdk_target_finite', 'right_sdk_command_cache_readback',
             'right_sdk_command_cache_finite', 'command_cache_target_right_finger12_rad',
             'reconstructed_target_right_finger12_rad']
    for name in added:
        columns[name] = len(columns)
    rows = np.pad(rows, ((0, 0), (0, len(added))))
    rows[:, columns['right_sdk_target_native_readback']] = 1
    rows[:, columns['right_sdk_target_finite']] = 1
    rows[:, columns['right_sdk_command_cache_readback']] = 1
    rows[:, columns['right_sdk_command_cache_finite']] = 1
    rows[:, columns['sdk_target_right_finger12_rad']] = .3
    rows[:, columns['command_cache_target_right_finger12_rad']] = .9
    rows[:, columns['reconstructed_target_right_finger12_rad']] = .21
    return rows, columns


def test_v2_backend_tracking_is_separate_from_cache_and_reconstructed_tracking():
    rows, columns = v2_target_rows()
    record = module.first_episode_metrics(rows, columns, {'env_id': 0, 'episode_id': 0})
    assert record['right_finger12_tracking']['native_readback_samples'] == 3
    assert record['right_finger12_tracking']['max_abs_rad'] == .3
    assert record['right_finger12_tracking']['command_cache_readback_samples'] == 3
    assert record['right_finger12_command_cache_tracking']['max_abs_rad'] == .9
    assert record['right_finger12_reconstructed_tracking']['max_abs_rad'] == .21
    assert record['target_provenance']['sdk_target_columns_source'] == 'physx_backend_get_dof_position_targets'


def test_unknown_teacher_target_version_cannot_claim_a_known_target_source():
    rows, columns = physical_fixture()
    with pytest.raises(ValueError, match='target audit version'):
        module.first_episode_metrics(rows, columns, {'env_id': 0, 'episode_id': 0}, audit_version='unknown')


def actual_v2_writer_archive(tmp_path):
    fixture_path = Path(__file__).with_name('test_bimanual_teacher_audit.py')
    fixture_spec = importlib.util.spec_from_file_location('teacher_writer_fixture', fixture_path)
    fixtures = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixtures)
    env, world, measurement, _, _ = fixtures.full_target_fixture(tmp_path)
    audit = fixtures.install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    audit.submit(torch.zeros(2, 24), teacher_control_step=0)
    fixtures.fresh(audit)
    audit.observe(env, measurement, torch.ones(2, dtype=torch.bool), torch.tensor([True, False]))
    summary = audit.finish()
    with np.load(tmp_path/'teacher_physics_audit.npz', allow_pickle=False) as raw:
        data = {key: raw[key] for key in raw.files}
    return data, summary


def test_v2_consumer_checks_actual_writer_full_named_target_archives(tmp_path):
    assert hasattr(module, 'validate_teacher_target_archive'), 'V2 full target archive reader required'
    data, summary = actual_v2_writer_archive(tmp_path)
    result = module.validate_teacher_target_archive(data, summary)
    assert result['full_target_dof_count_by_hand'] == {'left': 17, 'right': 17}
    assert result['backend_readback_samples_by_hand'] == {'left': 2, 'right': 2}
    del data['backend_full_target_left']
    with pytest.raises(ValueError, match='full target'):
        module.validate_teacher_target_archive(data, summary)


def test_v2_consumer_rejects_full_backend_active_column_disagreement(tmp_path):
    assert hasattr(module, 'validate_teacher_target_archive'), 'V2 full target archive reader required'
    data, summary = actual_v2_writer_archive(tmp_path)
    index = data['joint_names_left'].tolist().index('finger12')
    data['backend_full_target_left'][0, 0, index] += .01
    with pytest.raises(ValueError, match='active target'):
        module.validate_teacher_target_archive(data, summary)


def test_v2_consumer_checks_full_target_terminal_freeze_not_only_old_scalar_row(tmp_path):
    assert hasattr(module, 'validate_teacher_target_archive'), 'V2 full target archive reader required'
    data, summary = actual_v2_writer_archive(tmp_path)
    data['terminal_command_cache_full_target_left'][0, 0] += .01
    with pytest.raises(ValueError, match='terminal target'):
        module.validate_teacher_target_archive(data, summary)


def test_physical_identity_switch_before_terminal_is_rejected():
    rows, c = physical_fixture()
    rows[1, c['episode_id']] = 1
    with pytest.raises(ValueError, match='episode'):
        module.first_episode_metrics(rows, c, {'env_id': 0, 'episode_id': 0,
            'supervision_episode_id': 0, 'episode_uid': 'fixture:0:0'})


def test_missing_physical_first_terminal_is_explicitly_incomplete():
    rows, c = physical_fixture()
    rows[:, c['newly_terminal']] = 0
    rows[:, c['live']] = 1
    record = module.first_episode_metrics(rows, c, {'env_id': 0, 'episode_id': 0,
        'supervision_episode_id': 0, 'episode_uid': 'fixture:0:0'})
    assert not record['terminal_recorded']
    assert not record['native_terminal_success']


def test_real_writer_summary_identity_including_terminal_flag_is_supported():
    rows, c = physical_fixture()
    identity = dict(env_id=0, episode_id=0, supervision_episode_id=0,
                    episode_uid='native_run:0:0', terminal_recorded=True)
    record = module.first_episode_metrics(rows, c, identity)
    assert record['terminal_recorded']
    assert record['episode_uid'] == identity['episode_uid']
    assert record['lift_peak']['physics_step'] == 3
    assert identity['terminal_recorded'] is True


def test_fresh_contact_slip_stays_separate_from_native_success_hold_anchor():
    rows, columns = physical_fixture()
    names = ('fresh_contact_anchor_slip_m', 'fresh_contact_anchor_slip_valid',
             'fresh_contact_anchor_slip_applicable')
    for name in names:
        columns[name] = len(columns)
    rows = np.pad(rows, ((0, 0), (0, 3)))
    rows[:, columns[names[0]]] = [.0001, .0007, .0002, 1.]
    rows[:, columns[names[1]]] = 1
    rows[:, columns[names[2]]] = 1
    record = module.first_episode_metrics(rows, columns, {'env_id': 0, 'episode_id': 0,
        'supervision_episode_id': 0, 'episode_uid': 'fixture:0:0'})
    assert record['fresh_contact_slip_max_mm'] == pytest.approx(.7)
    assert record['fresh_contact_slip_valid_samples'] == 3
    assert record['fresh_contact_slip_applicable_samples'] == 3
    assert record['slip_max_mm'] is None
    assert record['hold_max_s'] == 5.


def proof_fixture(tmp_path):
    # A schema fixture tests gates only and contributes zero native trials.
    proof = dict(version='dh116-stage5-teacher-regression-v1', counts_as_student_success=False,
        native_runtime_attested_by_collector={'reference': True, 'adapter': True},
        initial_state={'bitwise_equal': True, 'seed': 19}, teacher_source_sha256='a'*64,
        paired_source_constraints={key: True for key in ('same_initial_state', 'same_seed',
            'same_teacher_source', 'same_visual_contract', 'same_controller_profile')},
        by_style={arm: {'pinch_thumb_index': {'native_success_count': 4,
            'qualified_supervision_success_count': 4, 'completed_first_episodes': 4,
            'first_episodes': 4}} for arm in ('reference', 'adapter')},
        adapter_internal_identity_execution={'verified_zero_execution_difference': True,
            'max_executed_action_difference': 0., 'contract': {'physical_action_adapter': {
                'actor_size': 3844, 'critic_size': 371, 'action': {'size': 24}, 'sha256': 'b'*64}}},
        pinch_teacher_four_of_four_regression_passed=True,
        success_contract={'net_lift_m': .03, 'hold_s': 5., 'fixed_anchor_slip_m': .0005})
    proof['input_directories'], proof['input_hashes'] = {}, {}
    for arm in ('reference', 'adapter'):
        folder = tmp_path / arm
        folder.mkdir()
        names = ['all_executed_supervision.pt', 'teacher_collection_report.json',
                 'teacher_physics_audit.npz', 'teacher_physics_audit_summary.json',
                 'native_physics_substeps.pt', 'initial_native_state.pt']
        if arm == 'adapter':
            names.append('physical_action_adapter_contract.json')
        for name in names:
            (folder/name).write_bytes(b'synthetic CPU contract fixture; no native trials')
        proof['input_directories'][arm] = str(folder.resolve())
        proof['input_hashes'][arm] = {name: module.digest(folder/name) for name in names}
    return proof


def test_t5_proof_gate_requires_actual_four_success_and_identity_execution(tmp_path):
    proof = proof_fixture(tmp_path)
    assert module.validate_teacher_regression_proof(proof)['verified']
    failed = copy.deepcopy(proof)
    failed['by_style']['adapter']['pinch_thumb_index']['native_success_count'] = 3
    with pytest.raises(ValueError, match='four'):
        module.validate_teacher_regression_proof(failed)
    failed = copy.deepcopy(proof)
    failed['adapter_internal_identity_execution']['max_executed_action_difference'] = 1e-8
    with pytest.raises(ValueError, match='identity'):
        module.validate_teacher_regression_proof(failed)


def test_t5_proof_does_not_impose_extra_exact_closed_loop_replay_gate(tmp_path):
    proof = proof_fixture(tmp_path)
    proof['actions'] = {'bitwise_equal': False}
    proof['physics'] = {'all_common_fields_bitwise_equal': False}
    assert module.validate_teacher_regression_proof(proof)['verified']
    proof['paired_source_constraints']['same_seed'] = False
    with pytest.raises(ValueError, match='paired'):
        module.validate_teacher_regression_proof(proof)


def test_t5_proof_rechecks_input_file_hashes_instead_of_trusting_approval_boolean(tmp_path):
    proof = proof_fixture(tmp_path)
    (Path(proof['input_directories']['adapter'])/'teacher_collection_report.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        module.validate_teacher_regression_proof(proof)
