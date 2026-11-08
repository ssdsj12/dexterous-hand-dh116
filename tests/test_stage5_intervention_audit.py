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
