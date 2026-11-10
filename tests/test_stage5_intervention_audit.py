import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('audit_stage5_interventions', TOOLS / 'audit_stage5_interventions.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def trace():
    columns = {name: i for i, name in enumerate(('live', 'done', 'measurement_valid', 'lift_m',
        'qualified_hold_s', 'slip_m', 'slip_valid', 'bilateral_opposed_contact', 'contact_valid',
        'right_palm_dz_m', 'left_palm_dz_m', 'physics_step', 'timestamp_s', 'episode_id',
        'right_finger12_tracking_error_rad', 'left_load_n', 'right_load_n'))}
    values = np.zeros((4, len(columns)))
    values[:, columns['live']] = [1, 1, 1, 0]
    values[:, columns['done']] = [0, 0, 1, 1]
    values[:, columns['measurement_valid']] = 1
    values[:, columns['lift_m']] = [.001, .002, .001, 10.]
    values[:, columns['left_palm_dz_m']] = [.003, .004, .002, 10.]
    values[:, columns['right_palm_dz_m']] = [.001, -.001, .001, 10.]
    values[:, columns['physics_step']] = [10, 11, 12, 13]
    values[:, columns['timestamp_s']] = np.arange(4) / 480
    values[:, columns['episode_id']] = [2, 2, 2, 3]
    values[:, columns['contact_valid']] = 1
    values[:, columns['right_finger12_tracking_error_rad']] = [.01, -.02, -.01, 100.]
    return values, columns


def test_first_terminal_and_physical_peak_identity_cannot_be_overwritten_by_noop():
    values, columns = trace()
    row = module.trial_metrics(values, columns, {'env_id': 7, 'trial_id': 'case7'}, 'z2')
    assert row['physical_samples'] == 3
    assert row['lift_peak_mm'] == 2.
    assert row['left_palm_peak']['value_mm'] == 4.
    assert row['left_palm_peak']['env_id'] == 7
    assert row['left_palm_peak']['episode_id'] == 2
    assert row['left_palm_peak']['physics_step'] == 11
    assert row['sync_peak_abs_dz_mm'] == 5.
    assert row['right_finger12_tracking']['max_abs_rad'] == .02
    assert row['slip_peak_mm'] is None
    assert row['slip_valid_samples'] == 0


def test_missing_finger12_field_and_one_hand_contact_are_unknown():
    values, columns = trace()
    columns.pop('right_finger12_tracking_error_rad')
    columns.pop('episode_id')
    row = module.trial_metrics(values, columns, {'env_id': 7, 'trial_id': 'case7'}, 'm0')
    assert row['right_finger12_tracking']['valid_samples'] == 0
    assert row['right_finger12_tracking']['max_abs_rad'] is None
    assert row['right_palm_peak']['episode_id'] is None
    assert row['left_effective_contact_fraction'] is None


def test_slip_mask_and_load_are_reduced_on_same_episode_only():
    values, columns = trace()
    values[:, columns['slip_m']] = [np.nan, .0003, np.nan, 1.]
    values[:, columns['slip_valid']] = [0, 1, 0, 1]
    values[:, columns['left_load_n']] = [1., 2., 3., 100.]
    row = module.trial_metrics(values, columns, {'env_id': 7, 'trial_id': 'case7'}, 'z3')
    assert row['slip_peak_mm'] == pytest.approx(.3)
    assert row['slip_valid_samples'] == 1
    assert row['left_load_mean_n'] == 2.


def test_episode_switch_before_terminal_is_rejected():
    values, columns = trace()
    values[1, columns['episode_id']] = 3
    with pytest.raises(ValueError, match='episode'):
        module.trial_metrics(values, columns, {'env_id': 7, 'trial_id': 'case7'}, 'z3')


def test_nonfinite_lift_with_wrong_validity_flag_still_fails_closed():
    values, columns = trace()
    values[1, columns['lift_m']] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        module.trial_metrics(values, columns, {'env_id': 7, 'trial_id': 'case7'}, 'z3')


@pytest.mark.parametrize('corrupt', [False, True])
def test_z1_non17_channel_corruption_fails_independent_of_argument_order(monkeypatch, corrupt):
    values, columns = trace()

    class FakeArm:
        def __init__(self, name):
            self.name = name
            self.folder = Path(name)
            self.identity = {'checkpoint': 'same', 'seed': 1}
            self.inputs = {}
            self.c = columns
            self.actions = np.zeros((1, 200, 24))
            self.actions[:, :, 17] = .25
            if name == 'z3':
                self.actions[:, :, 14] = .75
            if name == 'right_z_zero':
                self.actions[:, :, 17] = 0
                if corrupt:
                    self.actions[:, :, 14] = .1  # unintended left-hand intervention
            self.result = {'trials': [dict(env_id=i, trial_id=f'trial{i}',
                grasp_style='pinch_thumb_index' if i < 100 else 'tripod_thumb_index_middle',
                first_terminal_success=False, failure_reason='timeout') for i in range(200)]}

        def first(self, index):
            return values[:3]

    monkeypatch.setattr(module, 'Arm', FakeArm)
    if corrupt:
        with pytest.raises(ValueError, match='different executed action'):
            module.audit(['z3', 'right_z_zero', 'm0', 'z2'])
    else:
        result = module.audit(['z3', 'right_z_zero', 'm0', 'z2'])
        assert result['arms']['right_z_zero']['fixed_23_channels_independently_verified']


def extended_trace():
    values, columns = trace()
    additions = ('objective_valid', 'objective_lift', 'coordination_ready',
        'cancel_upward_left', 'wrist_guard_paused_right', 'right_finger12_sdk_target_readback_valid')
    additions += tuple(f'executed_action_{i}' for i in range(24))
    for name in additions:
        columns[name] = len(columns)
    values = np.pad(values, ((0, 0), (0, len(additions))))
    return values, columns


def test_objective_and_guard_validity_masks_ignore_terminal_noop_and_unknowns():
    values, columns = extended_trace()
    values[:, columns['objective_valid']] = [1, 0, 1, 1]
    values[:, columns['objective_lift']] = [2, 999, 4, 999]
    values[:, columns['coordination_ready']] = [0, 1, 1, 1]
    values[:, columns['cancel_upward_left']] = [1, 0, 1, 1]
    values[:, columns['wrist_guard_paused_right']] = [0, np.nan, 1, 1]
    values[:, columns['right_finger12_sdk_target_readback_valid']] = [1, 0, 1, 1]
    row = module.trial_metrics(values, columns, {'env_id': 7, 'trial_id': 'case7'}, 'z3')
    assert row['coordination_objectives']['terms']['lift']['valid_samples'] == 2
    assert row['coordination_objectives']['terms']['lift']['mean'] == 3.
    assert row['coordination_objectives']['terms']['contact']['mean'] is None
    assert row['coordination_flags']['coordination_ready']['fraction'] == pytest.approx(2/3)
    assert row['coordination_flags']['wrist_guard_paused_right']['valid_samples'] == 2
    assert row['coordination_flags']['wrist_guard_paused_right']['fraction'] == .5
    assert row['right_finger12_sdk_native_readback']['fraction'] == pytest.approx(2/3)


def test_source_policy_request_and_physics_execution_have_distinct_rates_and_masks():
    values, columns = extended_trace()
    values[:, columns['executed_action_17']] = [0, .5, np.nan, 1]
    source = np.zeros((2, 24))
    source[0, 17] = 1
    source[1, 17] = -1  # a subsequent control step, after first terminal
    row = module.trial_metrics(values, columns, {'env_id': 7, 'trial_id': 'case7'}, 'z3',
        source_requests24=source)
    request = row['action_statistics']['source_request24']
    execution = row['action_statistics']['executed_physical24']
    assert request['valid_samples'] == 1 and request['mean24'][17] == 1.
    assert execution['valid_samples'] == 2 and execution['mean24'][17] == .25
    assert request['sample_hz'] == 60 and execution['sample_hz'] == 480
    assert row['coordination_objectives']['terms']['lift']['valid_samples'] == 0


def test_missing_sdk_readback_flag_and_execution_keep_unknown_coverage():
    values, columns = trace()
    row = module.trial_metrics(values, columns, {'env_id': 7, 'trial_id': 'case7'}, 'm0')
    assert row['right_finger12_sdk_native_readback']['recorded'] is False
    assert row['right_finger12_sdk_native_readback']['fraction'] is None
    assert row['action_statistics']['executed_physical24']['mean24'] is None
    assert row['coordination_flags']['coordination_ready']['fraction'] is None


def test_style_aggregation_weights_valid_substeps_instead_of_trial_means():
    values, columns = extended_trace()
    values[:, columns['objective_valid']] = 1
    values[:, columns['objective_lift']] = [0, 1, 2, 999]
    values[:, columns['coordination_ready']] = [0, 1, 1, 1]
    first = module.trial_metrics(values, columns, {'env_id': 7, 'trial_id': 'case7'}, 'z3')
    second_trace = values.copy()
    second_trace[1, columns['done']] = 1
    second_trace[:, columns['objective_lift']] = 10
    second_trace[:, columns['coordination_ready']] = 0
    second_trace[:, columns['executed_action_17']] = 1
    second = module.trial_metrics(second_trace, columns, {'env_id': 8, 'trial_id': 'case8'}, 'z3')
    for row in (first, second):
        row.update(native_success=False, failure_reason='timeout')
    result = module.summarize_style([first, second])
    assert result['coordination_objectives']['terms']['lift']['valid_samples'] == 5
    assert result['coordination_objectives']['terms']['lift']['mean'] == pytest.approx(4.6)
    assert result['coordination_flags']['coordination_ready']['fraction'] == .4
    assert result['action_statistics']['executed_physical24']['mean24'][17] == .4
