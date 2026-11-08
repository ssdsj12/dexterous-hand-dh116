"""Audit a completed tripod Teacher control candidate separately from BC data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_stage5_teacher_regression import (
    compare_initial_states, digest, first_episode_metrics, stats,
    validate_teacher_regression_proof,
)
from audit_stage5_interventions import action_statistics, flag_statistics, numeric_statistics


ACTIVE = ('finger11', 'finger12', 'finger21', 'finger31', 'finger41', 'finger51')
FLAGS = ('resume_ready', 'cancel_upward_requested_left', 'cancel_upward_requested_right',
    'cancel_upward_applied_left', 'cancel_upward_applied_right', 'guard_pause_left', 'guard_pause_right',
    'guard_cancel_translation_left', 'guard_cancel_translation_right',
    'guard_cancel_rotation_left', 'guard_cancel_rotation_right')


def verify_tripod_exclusion(pool, tripod_envs):
    metadata, episodes = pool['sample_metadata'], pool['episodes']
    if pool['actions'].shape != (len(metadata), 24):
        raise ValueError('BC actions and metadata length disagree')
    for row in [*metadata, *episodes.values()]:
        if row['env_id'] in tripod_envs or row['grasp_style'] != 'pinch_thumb_index':
            raise ValueError('modified tripod rows or episodes leaked into a BC supervision pool')
    return dict(tripod_rows=0, admitted_rows=len(metadata), admitted_episode_ids=sorted(episodes),
                all_rows_are_unchanged_pinch=True)


def execution_metrics(series, columns, source, executed, identity, count):
    first = series[:count]
    if (len(first) != count or not np.all(first[:, columns['live']] == 1)
            or not np.all(first[:, columns['env_id']] == identity['env_id'])
            or not np.all(first[:, columns['episode_id']] == identity['episode_id'])):
        raise ValueError('candidate first-episode live identity mismatch')
    if (np.any(np.diff(first[:, columns['physics_step']]) != 1)
            or np.any(np.diff(first[:, columns['timestamp_s']]) <= 0)):
        raise ValueError('candidate first-episode physical clock mismatch')
    source, executed = source[:count], executed[:count]
    if source.shape != (count, 24) or executed.shape != (count, 24):
        raise ValueError('complete 480 Hz source and execution 24D requests required')
    if not np.isfinite(source).all() or not np.isfinite(executed).all():
        raise ValueError('nonfinite candidate source or execution request')
    difference = np.abs(source-executed)
    changed = np.flatnonzero(np.any(source != executed, axis=1))
    event = None
    if len(changed):
        index = int(changed[0])
        event = dict(**identity, policy_step=int(first[index, columns['policy_step']]),
            physics_step=int(first[index, columns['physics_step']]),
            timestamp_s=float(first[index, columns['timestamp_s']]),
            changed_dimensions=np.flatnonzero(source[index] != executed[index]).tolist(),
            source_request24=source[index].tolist(), executed_request24=executed[index].tolist())
    objectives = {}
    for name in ('lift', 'contact', 'slip', 'joint', 'sync', 'total'):
        data = first[:, columns['J_'+name]] if 'J_'+name in columns else None
        objectives[name] = numeric_statistics(data[np.isfinite(data)] if data is not None else [],
                                               recorded=data is not None)
    flags = {name: flag_statistics(first[:, columns[name]] if name in columns else None) for name in FLAGS}
    return dict(physical_samples=count, first_source_execution_difference=event,
        source_execution_changed_samples=len(changed),
        source_execution_max_abs_difference=float(difference.max()),
        source_requests=action_statistics(source, sample_hz=480, source='Teacher/MPC request before candidate'),
        executed_requests=action_statistics(executed, sample_hz=480,
            source='candidate projection output before unchanged native controller limits'),
        flags=flags, objective_terms=objectives,
        objective_coverage='finite pre-control diagnostic costs; no explicit J_valid mask recorded')


def compare_pinch_execution(reference, candidate_source, candidate_executed, reference_targets, candidate_targets):
    count = min(len(reference), len(candidate_source), len(candidate_executed),
                len(reference_targets), len(candidate_targets))
    if count < 1:
        raise ValueError('pinch common native window is empty')
    arrays = [np.ascontiguousarray(value[:count], dtype=np.float64) for value in
              (reference, candidate_source, candidate_executed, reference_targets, candidate_targets)]
    reference, candidate_source, candidate_executed, reference_targets, candidate_targets = arrays
    if reference.shape != candidate_source.shape or reference.shape != candidate_executed.shape:
        raise ValueError('pinch source/execution action shape mismatch')
    source_equal = reference.dtype == candidate_source.dtype and reference.tobytes() == candidate_source.tobytes()
    execution_equal = reference.dtype == candidate_executed.dtype and reference.tobytes() == candidate_executed.tobytes()
    targets_equal = reference_targets.dtype == candidate_targets.dtype and reference_targets.tobytes() == candidate_targets.tobytes()
    return dict(verified=source_equal and execution_equal and targets_equal, common_physical_substeps=count,
        comparison_representation='exact_float64_diagnostic_promotion',
        source_requests_bitwise_equal=source_equal, actual_execution_requests_bitwise_equal=execution_equal,
        sdk_targets_bitwise_equal=targets_equal,
        source_max_abs_difference=float(np.max(np.abs(reference-candidate_source))),
        executed_request_max_abs_difference=float(np.max(np.abs(reference-candidate_executed))),
        sdk_target_max_abs_difference_rad=float(np.max(np.abs(reference_targets-candidate_targets))))


def style_summary(rows, *, genuine):
    modified = rows[0]['grasp_style'] == 'tripod_thumb_index_middle'
    result = dict(first_episode_attempts=len(rows), completed_first_episodes=sum(row['terminal_recorded'] for row in rows),
        native_success_count=sum(row['native_terminal_success'] for row in rows) if genuine else 0,
        native_success_rate=sum(row['native_terminal_success'] for row in rows)/len(rows) if genuine else None,
        supervision_qualified_count=0 if modified or not genuine else sum(row.get('qualified_supervision_success', False) for row in rows),
        counts_as_student_success=False, lift_mm=stats(row['lift_peak_mm'] for row in rows),
        hold_s=stats(row['hold_max_s'] for row in rows), slip_mm=stats(row['slip_max_mm'] for row in rows),
        fresh_contact_slip_mm=stats(row.get('fresh_contact_slip_max_mm') for row in rows),
        fresh_contact_slip_valid_samples=sum(row.get('fresh_contact_slip_valid_samples', 0) for row in rows),
        fresh_contact_slip_recorded_first_episodes=sum(row.get('fresh_contact_slip_recorded', False) for row in rows),
        fresh_contact_slip_applicable_samples=(sum(row.get('fresh_contact_slip_applicable_samples') or 0 for row in rows)
            if any(row.get('fresh_contact_slip_applicable_samples') is not None for row in rows) else None),
        right_finger12_max_abs_error_rad=stats(row['right_finger12_tracking']['max_abs_rad'] for row in rows))
    for hand in ('left', 'right', 'bilateral'):
        valid = sum(row[hand+'_contact']['valid_samples'] for row in rows)
        paired = sum(row[hand+'_contact']['paired_samples'] for row in rows)
        result[hand+'_opposed_contact_fraction'] = paired/valid if valid else None
        result[hand+'_opposed_contact_valid_samples'] = valid
    for hand in ('left', 'right'):
        result[hand+'_mean_load_n'] = stats(row[hand+'_load_mean_n'] for row in rows)
    return result


def load_physical(folder):
    folder = Path(folder)
    report = json.loads((folder/'teacher_collection_report.json').read_text())
    summary = json.loads((folder/'teacher_physics_audit_summary.json').read_text())
    if report['status'] != 'completed' or summary['status'] != 'recorded' or summary.get('error'):
        raise ValueError('completed native collection and closed physical audit required')
    with np.load(folder/'teacher_physics_audit.npz', allow_pickle=False) as data:
        columns = {name: i for i, name in enumerate(data['fields'].tolist())}
        samples, terminals, valid = data['samples'], data['terminal_samples'], data['terminal_valid']
    records = []
    for identity in summary['episodes']:
        env = identity['env_id']
        row = first_episode_metrics(samples[:, env], columns, identity)
        episode = report['episodes'][str(identity['supervision_episode_id'])]
        row.update(grasp_style=episode['grasp_style'], qualified_supervision_success=episode['success'])
        if bool(valid[env]) != row['terminal_recorded']:
            raise ValueError('frozen candidate terminal coverage changed')
        if row['terminal_recorded'] and samples[row['physical_samples']-1, env].tobytes() != terminals[env].tobytes():
            raise ValueError('candidate terminal physical sample overwritten')
        records.append(row)
    return dict(folder=folder, report=report, summary=summary, samples=samples, columns=columns, records=records)


def audit(candidate_folder, reference_folder):
    from go2_pvcnn.control.dh116.tabletop_vision_supervision_dataset import validate_dataset, _canonical_visual_contract
    candidate, reference = load_physical(candidate_folder), load_physical(reference_folder)
    folder = candidate['folder']
    contract_path = folder/'tripod_teacher_coordination_contract.json'
    contract = json.loads(contract_path.read_text())
    if contract.get('error') or contract.get('tripod_supervision_qualified') is not False:
        raise ValueError('completed independent unqualified tripod candidate contract required')
    proof_path = Path(contract['proof']['path'])
    if digest(proof_path) != contract['proof']['sha256']:
        raise ValueError('candidate T5 proof SHA changed')
    proof = json.loads(proof_path.read_text())
    proof_validation = validate_teacher_regression_proof(proof)
    if reference['folder'].resolve() != Path(proof['input_directories']['reference']).resolve():
        raise ValueError('pinch reference must be the actual T5 original run')
    initial = compare_initial_states(
        torch.load(reference['folder']/'initial_native_state.pt', map_location='cpu', weights_only=True),
        torch.load(folder/'initial_native_state.pt', map_location='cpu', weights_only=True))
    if not initial['bitwise_equal'] or candidate['report']['source_sha256'] != reference['report']['source_sha256']:
        raise ValueError('candidate and T5 native initial state/source/seed differ')
    with np.load(folder/'tripod_teacher_coordination.npz', allow_pickle=False) as data:
        c = {name: i for i, name in enumerate(data['fields'].tolist())}
        samples, source, executed, mask = data['samples'], data['source_requests'], data['executed_requests'], data['tripod_mask']
    expected_mask = np.array([row['grasp_style'] == 'tripod_thumb_index_middle' for row in candidate['records']])
    if not np.array_equal(mask, expected_mask):
        raise ValueError('candidate actual grasp styles disagree with modified environment mask')
    tripod_envs = set(np.flatnonzero(mask).tolist())
    pools, hashes = {}, {}
    for name in ('all_executed_supervision.pt', 'successful_visual_supervision.pt'):
        if not (folder/name).exists():
            continue
        pool = torch.load(folder/name, map_location='cpu', weights_only=True)
        validate_dataset(pool, require_successful=name.startswith('successful'))
        pools[name] = verify_tripod_exclusion(pool, tripod_envs)
        hashes[name] = digest(folder/name)
    if 'all_executed_supervision.pt' not in pools:
        raise ValueError('candidate BC audit pool missing')
    pool = torch.load(folder/'all_executed_supervision.pt', map_location='cpu', weights_only=True)
    reference_pool = torch.load(reference['folder']/'all_executed_supervision.pt', map_location='cpu', weights_only=True)
    if _canonical_visual_contract(pool['visual_contract']) != _canonical_visual_contract(reference_pool['visual_contract']):
        raise ValueError('candidate physical/camera/action contract differs from T5')
    bc = candidate['columns']
    rc = reference['columns']
    requests = [rc[f'teacher_issued_action_{i}'] for i in range(24)]
    targets = [f'sdk_target_{side}_{name}_rad' for side in ('left', 'right') for name in ACTIVE]
    preservation = []
    for record in candidate['records']:
        env, count = record['env_id'], record['physical_samples']
        physical = candidate['samples'][:count, env]
        diagnostic = samples[:count, env]
        for key in ('env_id', 'episode_id', 'physics_step', 'timestamp_s'):
            if not np.array_equal(physical[:, bc[key]], diagnostic[:, c[key]]):
                raise ValueError('candidate source/execution and actual physical clocks disagree')
        record['control_candidate'] = execution_metrics(diagnostic, c, source[:, env], executed[:, env], record, count)
        right = (physical[:, bc['sdk_target_right_finger12_rad']]-physical[:, bc['actual_q_right_finger12_rad']])
        record['right_finger12_sdk_error_mean_abs_rad'] = float(np.abs(right).mean())
        if env not in tripod_envs:
            ref_record = next(row for row in reference['records'] if row['env_id'] == env)
            prefix = min(count, ref_record['physical_samples'])
            baseline = reference['samples'][:prefix, env]
            result = compare_pinch_execution(baseline[:, requests], source[:prefix, env], executed[:prefix, env],
                baseline[:, [rc[name] for name in targets]], physical[:prefix, [bc[name] for name in targets]])
            preservation.append(dict(env_id=env, **result))
    genuine = (candidate['report'].get('native_runtime_verified') is True
               and candidate['summary'].get('native_runtime_verified') is True)
    groups = {style: style_summary([row for row in candidate['records'] if row['grasp_style'] == style], genuine=genuine)
              for style in sorted({row['grasp_style'] for row in candidate['records']})}
    filenames = ('teacher_collection_report.json', 'teacher_physics_audit_summary.json', 'teacher_physics_audit.npz',
        'tripod_teacher_coordination.npz', 'tripod_teacher_coordination_contract.json', 'initial_native_state.pt')
    hashes.update({name: digest(folder/name) for name in filenames})
    return dict(version='dh116-stage5-teacher-coordination-audit-v1', source='Teacher/MPC candidate;not Student',
        native_runtime_attested_by_collector=genuine, by_style=groups, physical_first_episodes=candidate['records'],
        initial_state_matches_T5=initial, T5_proof=proof_validation, candidate_contract=contract,
        BC_tripod_exclusion=pools, unchanged_pinch_actual_execution_comparison=preservation,
        unchanged_pinch_execution_and_SDK_targets_verified=all(row['verified'] for row in preservation),
        tripod_supervision_qualified=False, counts_as_student_success=False,
        input_directories=dict(candidate=str(folder.resolve()), reference=str(reference['folder'].resolve())),
        input_hashes=hashes, T5_proof_sha256=digest(proof_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.candidate, args.reference)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write('\n')
