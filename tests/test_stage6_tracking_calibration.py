"""CPU evidence/statistic fixtures do not perform a native calibration trial."""
import copy
import hashlib
import importlib
import json

import numpy as np
import pytest
import torch

from test_vision_sequence_dataset import fixture_context, fixture_data
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import BACKEND_SOURCE, CACHE_SOURCE


def api():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_teacher_tracking_calibration')


def evidence():
    data = fixture_data((3, 3)); context = fixture_context(data)
    # Pre-action feedback, command caches and reconstructed targets are
    # deliberately unrelated to the only admitted post-physics evidence.
    context['physics_context']['measured_q_rad'].fill_(500.)
    context['teacher_targets']['finger'].fill_(-700.)
    context['command_cache'] = 'not_eligible_for_calibration'
    context['native_backend_targets'] = []; context['post_physics_feedback'] = []
    expected = []
    for index, row in enumerate(data['sample_metadata']):
        key = tuple(row[name] for name in ('run_uuid', 'env_id', 'episode_uid', 'tick'))
        back, post = [], []
        for substep in range(8):
            step = row['first_physics_step'] + substep + 1
            when = row['policy_timestamp_s'] + (substep + 1) / 480
            error = torch.arange(1, 13, dtype=torch.float64) * .001 + (index * 8 + substep) * .00001
            target = torch.arange(12, dtype=torch.float64) * .01
            target = target.float().double()
            q = target - error
            hands = []
            for hand in range(2):
                names = ['wrist_x', *reversed(ACTIVE), 'wrist_y']
                values = np.zeros((1, len(names)), np.float32)
                for j, name in enumerate(ACTIVE): values[0, names.index(name)] = target[6 * hand + j].item()
                hands.append(dict(joint_names=names, backend=dict(available=True,
                    source=BACKEND_SOURCE, values=values, dtype='float32', shape=[8, len(names)],
                    finite_env=np.ones(8, dtype=bool), error=None)))
            back.append(dict(sdk_physics_step=step, sdk_timestamp_s=when, clock_aligned=True, hands=hands))
            post.append(dict(measured_q_rad=q[None], measured_qd_rad_s=torch.zeros(1, 12, dtype=torch.float64),
                measured_root_pose=torch.zeros(1, 2, 7, dtype=torch.float64),
                sdk_physics_step=step, sdk_timestamp_s=when,
                backend_readback_sdk_physics_step=step, backend_readback_sdk_timestamp_s=when,
                clock_aligned=True, source_valid=torch.tensor([True]), native_runtime_verified=True,
                source='original_advance_returned_refresh_before_TensorEnv_terminal_reset'))
            expected.append(error)
        context['native_backend_targets'].append(dict(sample_key=key, physics_substeps=back))
        context['post_physics_feedback'].append(dict(sample_key=key, physics_substeps=post))
    return data, context, torch.stack(expected)


def calibrate(data, context):
    return api().calibrate_teacher_tracking(data, context, dataset_sha256='d' * 64, context_sha256='e' * 64)


def test_only_named_physx_targets_and_same_clock_post_q_define_joint_quantiles():
    data, context, expected = evidence()
    before = copy.deepcopy(context['post_physics_feedback'])
    # The two evidence tables are independently reordered, never row-zipped.
    context['native_backend_targets'].reverse()
    context['post_physics_feedback'] = context['post_physics_feedback'][2:] + context['post_physics_feedback'][:2]
    report = calibrate(data, context)
    assert report['qualified'] is True
    wanted = torch.quantile(expected, .999, dim=0)
    torch.testing.assert_close(torch.tensor(report['teacher_error_p999_rad']), wanted.float(), rtol=0, atol=1e-8)
    torch.testing.assert_close(torch.tensor(report['trigger_rad']), torch.maximum(wanted * 1.5, wanted.new_full((12,), .01)).float(), rtol=0, atol=1e-8)
    assert report['samples_per_hand'] == {'left': 48, 'right': 48}
    assert report['per_joint_sample_count'] == [48] * 12
    assert report['dataset_sha256'] == 'd' * 64 and report['context_sha256'] == 'e' * 64
    assert report['controller_installed'] is False and report['new_native_trial_performed'] is False
    assert report['clock_mapping']['sdk_minus_metadata_physics_step'] == 0
    assert report['clock_mapping']['world_physics_steps_available'] is False
    assert torch.equal(before[0]['physics_substeps'][0]['measured_q_rad'],
        next(row for row in context['post_physics_feedback'] if row['sample_key'] == before[0]['sample_key'])['physics_substeps'][0]['measured_q_rad'])


@pytest.mark.parametrize('field', ['native_backend_targets', 'post_physics_feedback'])
def test_missing_or_duplicate_sample_identity_refuses_qualification(field):
    data, context, _ = evidence(); context[field].pop(1)
    result = calibrate(data, context)
    assert not result['qualified'] and result['reasons']
    assert 'trigger_rad' not in result and 'teacher_error_p999_rad' not in result
    data, context, _ = evidence(); context[field].append(copy.deepcopy(context[field][0]))
    result = calibrate(data, context)
    assert not result['qualified'] and 'duplicate' in json.dumps(result['reasons']).lower()


@pytest.mark.parametrize('bad', ['source', 'unavailable', 'nan', 'finite_env', 'names', 'shape'])
def test_cache_reconstruction_or_missing_backend_can_never_substitute(bad):
    data, context, _ = evidence()
    hand = context['native_backend_targets'][1]['physics_substeps'][2]['hands'][0]
    if bad == 'source': hand['backend']['source'] = CACHE_SOURCE
    elif bad == 'unavailable': hand['backend']['available'] = False
    elif bad == 'nan': hand['backend']['values'][0, 1] = np.nan
    elif bad == 'finite_env': hand['backend']['finite_env'][0] = False
    elif bad == 'names': hand['joint_names'][1] = hand['joint_names'][2]
    elif bad == 'shape': hand['backend']['values'] = np.zeros((8, len(hand['joint_names'])), np.float32)
    result = calibrate(data, context)
    assert not result['qualified']
    assert 'trigger_rad' not in result


@pytest.mark.parametrize('bad', ['pre_feedback', 'native', 'missing_qd', 'nan_q', 'nan_qd', 'source_valid'])
def test_postphysics_feedback_is_required_with_explicit_provenance_and_validity(bad):
    data, context, _ = evidence(); packet = context['post_physics_feedback'][1]['physics_substeps'][2]
    if bad == 'pre_feedback': packet['source'] = 'pre_action_measured_context'
    elif bad == 'native': packet['native_runtime_verified'] = False
    elif bad == 'missing_qd': packet.pop('measured_qd_rad_s')
    elif bad == 'nan_q': packet['measured_q_rad'][0, 0] = float('nan')
    elif bad == 'nan_qd': packet['measured_qd_rad_s'][0, 0] = float('inf')
    elif bad == 'source_valid': packet['source_valid'][0] = False
    result = calibrate(data, context)
    assert not result['qualified'] and 'teacher_error_p999_rad' not in result


@pytest.mark.parametrize('field', ['sdk_physics_step', 'sdk_timestamp_s', 'backend_readback_sdk_physics_step',
                                  'backend_readback_sdk_timestamp_s', 'clock_aligned'])
def test_getter_and_post_feedback_clocks_must_match_exactly(field):
    data, context, _ = evidence(); packet = context['post_physics_feedback'][0]['physics_substeps'][2]
    packet[field] = False if field == 'clock_aligned' else packet[field] + (1e-12 if 'timestamp' in field else 1)
    result = calibrate(data, context)
    assert not result['qualified'] and 'trigger_rad' not in result


def test_sdk_mapping_is_fixed_and_does_not_assume_world_counter_equals_sdk_index():
    data, context, _ = evidence()
    for row in data['sample_metadata']:
        row['first_physics_step'] += 53; row['physics_step'] += 53
    context['samples']['first_physics_step'] = [row['first_physics_step'] for row in data['sample_metadata']]
    result = calibrate(data, context)
    assert result['qualified'] and result['clock_mapping']['sdk_minus_metadata_physics_step'] == -53
    context['post_physics_feedback'][1]['physics_substeps'][2]['sdk_physics_step'] += 1
    context['post_physics_feedback'][1]['physics_substeps'][2]['backend_readback_sdk_physics_step'] += 1
    context['native_backend_targets'][1]['physics_substeps'][2]['sdk_physics_step'] += 1
    assert not calibrate(data, context)['qualified']


def test_bulk_finite_env_metadata_is_selected_by_real_env_identity():
    data, context, _ = evidence()
    record = context['native_backend_targets'][3]
    assert record['sample_key'][1] == 1
    for packet in record['physics_substeps']:
        packet['hands'][0]['backend']['finite_env'][0] = False
    assert calibrate(data, context)['qualified']
    record['physics_substeps'][0]['hands'][0]['backend']['finite_env'][1] = False
    assert not calibrate(data, context)['qualified']


def test_terminal_inactive_suffix_is_excluded_but_missing_live_feedback_disqualifies():
    data, context, _ = evidence()
    context['physics_context']['live'][2, 3:] = False
    context['physics_context']['source_valid'][2, 3:] = False
    for field in ('native_backend_targets', 'post_physics_feedback'):
        context[field][2]['physics_substeps'][3:] = [None] * 5
    report = calibrate(data, context)
    assert report['qualified'] and report['samples_per_hand'] == {'left': 43, 'right': 43}
    context['post_physics_feedback'][2]['physics_substeps'][2] = None
    assert not calibrate(data, context)['qualified']


def test_failed_recovered_or_unfinished_episode_is_not_a_calibration_source():
    for field, value in (('success', False), ('native_physics', False), ('recovered', True)):
        data, context, _ = evidence(); data['episodes']['0'][field] = value
        assert not calibrate(data, context)['qualified']
    data, context, _ = evidence(); data['episodes']['0'].pop('terminal')
    assert not calibrate(data, context)['qualified']


def test_file_entry_binds_actual_bytes_and_saves_success_or_ineligible_evidence_exclusively(tmp_path):
    data, context, _ = evidence()
    dataset, sidecar, output = tmp_path / 'data.pt', tmp_path / 'context.pt', tmp_path / 'calibration.json'
    torch.save(data, dataset)
    context['dataset_sha256'] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    torch.save(context, sidecar)
    result = api().save_teacher_tracking_calibration(dataset, sidecar, output)
    saved = json.loads(output.read_text())
    assert result == saved and saved['qualified']
    assert saved['context_sha256'] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError): api().save_teacher_tracking_calibration(dataset, sidecar, output)
    with pytest.raises(ValueError): api().save_teacher_tracking_calibration(dataset, sidecar, dataset)
    context.pop('post_physics_feedback'); torch.save(context, sidecar)
    rejected_path = tmp_path / 'rejected.json'
    rejected = api().save_teacher_tracking_calibration(dataset, sidecar, rejected_path)
    assert not rejected['qualified'] and rejected_path.exists()
    assert 'trigger_rad' not in json.loads(rejected_path.read_text())


def reported_sdk_clock(data, context, cadence):
    """A single global reported clock; nominal controller dt stays float64."""
    for row in data['sample_metadata']:
        row['policy_timestamp_s'] = row['first_physics_step'] * cadence
        row['simulation_time_s'] = row['policy_timestamp_s']
    context['samples']['policy_timestamp_s'] = [row['policy_timestamp_s'] for row in data['sample_metadata']]
    for backend, post in zip(context['native_backend_targets'], context['post_physics_feedback']):
        for b, p in zip(backend['physics_substeps'], post['physics_substeps']):
            when = p['sdk_physics_step'] * cadence
            b['sdk_timestamp_s'] = p['sdk_timestamp_s'] = p['backend_readback_sdk_timestamp_s'] = when


def test_one_global_reported_float32_cadence_is_separate_from_nominal_controller_dt():
    data, context, expected = evidence()
    cadence = float(np.float32(1 / 480))
    reported_sdk_clock(data, context, cadence)
    nominal_before = context['physics_context']['dt_s'].clone()
    report = calibrate(data, context)
    assert report['qualified'] is True
    clock = report['clock_mapping']
    assert clock['reported_sdk_cadence_s'] == cadence
    assert clock['reported_sdk_time_origin_s'] == 0.
    assert clock['distinct_sdk_steps'] == 24
    assert clock['maximum_timestamp_roundoff_s'] == 0.
    assert clock['reported_minus_nominal_dt_s_min'] == cadence - 1 / 480
    assert clock['reported_minus_nominal_dt_s_max'] == cadence - 1 / 480
    assert clock['maximum_policy_interval_nominal_dt_drift_s'] > 8e-10
    assert torch.equal(context['physics_context']['dt_s'], nominal_before)
    torch.testing.assert_close(torch.tensor(report['teacher_error_p999_rad']),
        torch.quantile(expected, .999, dim=0).float(), rtol=0, atol=1e-8)


@pytest.mark.parametrize('bad', ['one_row_origin', 'one_row_cadence', 'metadata_origin'])
def test_clock_mapping_cannot_reset_origin_or_cadence_per_policy_row(bad):
    data, context, _ = evidence()
    cadence = float(np.float32(1 / 480))
    reported_sdk_clock(data, context, cadence)
    # Keep each backend/post getter pair exact, so only the global mapping
    # invariant can reject these cross-row/cadence changes.
    for k, (b, p) in enumerate(zip(context['native_backend_targets'][1]['physics_substeps'],
                                  context['post_physics_feedback'][1]['physics_substeps'])):
        delta = 1e-12 if bad == 'one_row_origin' else (k + 1) * (1 / 480 - cadence)
        if bad == 'metadata_origin': delta = 0.
        when = p['sdk_timestamp_s'] + delta
        b['sdk_timestamp_s'] = p['sdk_timestamp_s'] = p['backend_readback_sdk_timestamp_s'] = when
    if bad == 'metadata_origin':
        data['sample_metadata'][1]['policy_timestamp_s'] += 1e-12
        context['samples']['policy_timestamp_s'][1] += 1e-12
    result = calibrate(data, context)
    assert not result['qualified'] and 'trigger_rad' not in result


@pytest.mark.parametrize('switch_cadence', [False, True])
def test_all_environments_cannot_share_a_piecewise_reported_clock(switch_cadence):
    data, context, _ = evidence()
    cadence = float(np.float32(1 / 480))
    reported_sdk_clock(data, context, cadence)
    def change(step):
        return max(0, step - 8) * 1e-12 if switch_cadence else (1e-12 if step > 8 else 0.)
    # Every environment still reports the exact same time for a shared index.
    # Backend and feedback getter clocks also remain exact. A per-policy
    # mapping would accept this, but the global constant-cadence audit rejects it.
    for back, post in zip(context['native_backend_targets'], context['post_physics_feedback']):
        for b, p in zip(back['physics_substeps'], post['physics_substeps']):
            when = p['sdk_timestamp_s'] + change(p['sdk_physics_step'])
            b['sdk_timestamp_s'] = p['sdk_timestamp_s'] = p['backend_readback_sdk_timestamp_s'] = when
    for row in data['sample_metadata']:
        row['policy_timestamp_s'] += change(row['first_physics_step'])
    context['samples']['policy_timestamp_s'] = [row['policy_timestamp_s'] for row in data['sample_metadata']]
    result = calibrate(data, context)
    assert not result['qualified'] and 'trigger_rad' not in result
    assert result['reason_counts'].get('reported_clock_mapping', 0) > 0
