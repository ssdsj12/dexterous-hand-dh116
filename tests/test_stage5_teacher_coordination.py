import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('teacher_coordination_audit', TOOLS/'audit_stage5_teacher_coordination.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_tripod_is_rejected_from_rows_or_episode_records_in_bc_pool():
    pool = dict(actions=torch.zeros((1, 24)), sample_metadata=[
        dict(env_id=0, episode_id=0, grasp_style='pinch_thumb_index')],
        episodes={'0': dict(env_id=0, grasp_style='pinch_thumb_index')})
    assert module.verify_tripod_exclusion(pool, {4, 5, 6, 7})['tripod_rows'] == 0
    pool['episodes']['4'] = dict(env_id=4, grasp_style='tripod_thumb_index_middle')
    with pytest.raises(ValueError, match='tripod'):
        module.verify_tripod_exclusion(pool, {4, 5, 6, 7})


def test_hidden_tripod_env_cannot_be_admitted_under_pinch_label():
    pool = dict(actions=torch.zeros((1, 24)), sample_metadata=[
        dict(env_id=4, episode_id=4, grasp_style='pinch_thumb_index')],
        episodes={'4': dict(env_id=4, grasp_style='pinch_thumb_index')})
    with pytest.raises(ValueError, match='tripod'):
        module.verify_tripod_exclusion(pool, {4})


def execution_fixture():
    names = ['env_id', 'episode_id', 'policy_step', 'physics_step', 'timestamp_s',
        'live', 'newly_terminal', 'resume_ready', 'guard_pause_left', 'J_lift']
    c = {name: i for i, name in enumerate(names)}
    rows = np.zeros((4, len(c)))
    rows[:, c['physics_step']] = [1, 2, 3, 4]
    rows[:, c['timestamp_s']] = np.arange(1, 5)/480
    rows[:, c['live']] = [1, 1, 1, 0]
    rows[:, c['newly_terminal']] = [0, 0, 1, 0]
    rows[:, c['resume_ready']] = [0, 1, 0, 1]
    rows[:, c['guard_pause_left']] = [0, 0, 1, 1]
    rows[:, c['J_lift']] = [1, np.nan, 3, 999]
    source, executed = np.zeros((4, 24)), np.zeros((4, 24))
    executed[1, 17] = .2
    executed[3, :] = 999  # closed no-op never included
    return rows, c, source, executed


def test_execution_difference_retains_identity_time_and_does_not_include_noop():
    rows, c, source, executed = execution_fixture()
    result = module.execution_metrics(rows, c, source, executed,
        {'env_id': 0, 'episode_id': 0, 'episode_uid': 'fixture:0:0'}, 3)
    assert result['first_source_execution_difference']['physics_step'] == 2
    assert result['first_source_execution_difference']['changed_dimensions'] == [17]
    assert result['source_execution_max_abs_difference'] == .2
    assert result['flags']['resume_ready']['fraction'] == pytest.approx(1/3)
    assert result['objective_terms']['lift']['valid_samples'] == 2
    assert result['objective_terms']['lift']['mean'] == 2.


def test_pinch_preservation_checks_actual_execution_and_sdk_targets():
    source = np.zeros((3, 24))
    execute = source.copy()
    reference_targets = np.zeros((3, 12))
    current_targets = reference_targets.copy()
    assert module.compare_pinch_execution(source, source, execute, reference_targets, current_targets)['verified']
    current_targets[1, 7] = .001
    assert not module.compare_pinch_execution(source, source, execute, reference_targets, current_targets)['verified']
    execute[1, 17] = .001
    result = module.compare_pinch_execution(source, source, execute, reference_targets, reference_targets)
    assert not result['verified']


def test_physics_float64_request_record_matches_exact_promoted_float32_candidate_request():
    source = np.ones((3, 24), dtype=np.float32) * .3
    reference = source.astype(np.float64)  # TeacherPhysicsAudit stores double diagnostics
    targets = np.zeros((3, 12), dtype=np.float64)
    result = module.compare_pinch_execution(reference, source, source.copy(), targets, targets)
    assert result['verified']
    assert result['comparison_representation'] == 'exact_float64_diagnostic_promotion'


def test_first_episode_clock_or_live_mismatch_is_rejected():
    rows, c, source, executed = execution_fixture()
    rows[1, c['live']] = 0
    with pytest.raises(ValueError, match='first'):
        module.execution_metrics(rows, c, source, executed,
            {'env_id': 0, 'episode_id': 0, 'episode_uid': 'fixture:0:0'}, 3)


def test_synthetic_candidate_success_never_counts_as_native_or_bc_success():
    record = dict(native_terminal_success=True, lift_peak_mm=31., hold_max_s=5.,
        slip_max_mm=.1, grasp_style='tripod_thumb_index_middle', terminal_recorded=True,
        right_finger12_tracking={'max_abs_rad': .01},
        left_contact={'valid_samples': 3, 'paired_samples': 3},
        right_contact={'valid_samples': 3, 'paired_samples': 3},
        bilateral_contact={'valid_samples': 3, 'paired_samples': 3},
        left_load_mean_n=1., right_load_mean_n=1.)
    result = module.style_summary([record]*4, genuine=False)
    assert result['native_success_count'] == 0
    assert result['supervision_qualified_count'] == 0
    assert result['counts_as_student_success'] is False
    assert result['fresh_contact_slip_applicable_samples'] is None
    assert result['fresh_contact_slip_mm']['maximum'] is None
