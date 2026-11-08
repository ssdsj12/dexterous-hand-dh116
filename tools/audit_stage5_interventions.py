"""Compare Z0..Z3 native first episodes with explicit optional-field coverage."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit_vision_paired_bias import Arm, check_replay, distribution


OBJECTIVE_TERMS = ('lift', 'contact', 'slip', 'joint', 'sync', 'total')
COORDINATION_FLAGS = ('coordination_ready', 'cancel_upward_left', 'cancel_upward_right',
    'wrist_guard_enabled', 'wrist_guard_paused_left', 'wrist_guard_paused_right',
    'wrist_rotation_cancel_left', 'wrist_rotation_cancel_right')


def numeric_statistics(values, *, recorded):
    values = np.asarray(values, dtype=float)
    return dict(recorded=recorded, valid_samples=len(values),
        mean=float(values.mean()) if len(values) else None,
        std=float(values.std()) if len(values) else None,
        minimum=float(values.min()) if len(values) else None,
        maximum=float(values.max()) if len(values) else None)


def flag_statistics(values):
    if values is None:
        return dict(recorded=False, valid_samples=0, true_samples=None, fraction=None)
    values = np.asarray(values)
    valid = values[np.isfinite(values) & np.isin(values, (0, 1))]
    count = int(np.sum(valid == 1))
    return dict(recorded=True, valid_samples=len(valid), true_samples=count,
        fraction=count/len(valid) if len(valid) else None)


def action_statistics(values, *, sample_hz, source):
    recorded = values is not None
    matrix = np.asarray(values) if recorded else np.empty((0, 24))
    if matrix.ndim != 2 or matrix.shape[1] != 24:
        raise ValueError('24 action columns required')
    valid = matrix[np.isfinite(matrix).all(1)]
    return dict(recorded=recorded, sample_hz=sample_hz, source=source,
        available_samples=len(matrix), valid_samples=len(valid),
        mean24=valid.mean(0).tolist() if len(valid) else None,
        std24=valid.std(0).tolist() if len(valid) else None,
        minimum24=valid.min(0).tolist() if len(valid) else None,
        maximum24=valid.max(0).tolist() if len(valid) else None,
        saturation_fraction24=(np.abs(valid) >= 1-1e-6).mean(0).tolist() if len(valid) else None)


def pool_numeric_statistics(rows):
    known = [row for row in rows if row['valid_samples']]
    total = sum(row['valid_samples'] for row in known)
    mean = sum(row['valid_samples']*row['mean'] for row in known)/total if total else None
    variance = (sum(row['valid_samples']*(row['std']**2+row['mean']**2) for row in known)/total
                -mean**2) if total else None
    return dict(recorded=any(row['recorded'] for row in rows), valid_samples=total, mean=mean,
        std=float(np.sqrt(max(0., variance))) if total else None,
        minimum=min((row['minimum'] for row in known), default=None),
        maximum=max((row['maximum'] for row in known), default=None))


def pool_flag_statistics(rows):
    total = sum(row['valid_samples'] for row in rows)
    true = sum(row['true_samples'] or 0 for row in rows)
    recorded = any(row['recorded'] for row in rows)
    return dict(recorded=recorded, valid_samples=total, true_samples=true if recorded else None,
                fraction=true/total if total else None)


def pool_action_statistics(rows):
    known = [row for row in rows if row['valid_samples']]
    result = action_statistics(None, sample_hz=rows[0]['sample_hz'], source=rows[0]['source'])
    result.update(recorded=any(row['recorded'] for row in rows),
        available_samples=sum(row['available_samples'] for row in rows),
        valid_samples=sum(row['valid_samples'] for row in rows))
    if known:
        weights = np.array([row['valid_samples'] for row in known])
        means = np.array([row['mean24'] for row in known])
        mean = np.average(means, axis=0, weights=weights)
        variance = np.average(np.array([row['std24'] for row in known])**2 + means**2,
                              axis=0, weights=weights)-mean**2
        result.update(mean24=mean.tolist(), std24=np.sqrt(np.maximum(variance, 0)).tolist(),
            minimum24=np.min([row['minimum24'] for row in known], axis=0).tolist(),
            maximum24=np.max([row['maximum24'] for row in known], axis=0).tolist(),
            saturation_fraction24=np.average([row['saturation_fraction24'] for row in known],
                                            axis=0, weights=weights).tolist())
    return result


def trial_metrics(trace, columns, label, arm, *, source_requests24=None):
    """Reduce one episode and retain actual peak identity and physical clock."""
    terminal = np.flatnonzero(trace[:, columns['done']])
    if not len(terminal):
        raise ValueError('missing first terminal')
    trace = trace[:int(terminal[0])+1]
    if not trace[:, columns['live']].all() or not trace[:, columns['measurement_valid']].all():
        raise ValueError('invalid first-episode live readback')

    def values(name):
        return trace[:, columns[name]] if name in columns else None

    for name in ('lift_m', 'qualified_hold_s', 'right_palm_dz_m', 'physics_step', 'timestamp_s'):
        if not np.isfinite(values(name)).all():
            raise ValueError('nonfinite required native metric: '+name)

    episodes = values('episode_id')
    if episodes is not None and (not np.isfinite(episodes).all() or len(set(episodes)) != 1):
        raise ValueError('episode changed before first terminal')

    def peak(name):
        data = values(name)
        if data is None:
            return dict(value_mm=None, env_id=label['env_id'], trial_id=label['trial_id'],
                        episode_id=None, physics_step=None, timestamp_s=None)
        if not np.isfinite(data).all():
            raise ValueError('nonfinite measured palm displacement')
        index = int(np.argmax(data))
        return dict(value_mm=float(1000*data[index]), env_id=label['env_id'],
            trial_id=label['trial_id'], episode_id=int(episodes[index]) if episodes is not None else None,
            physics_step=int(values('physics_step')[index]), timestamp_s=float(values('timestamp_s')[index]))

    def mean(name):
        data = values(name)
        return float(data.mean()) if data is not None and np.isfinite(data).all() else None

    def final_mm(name):
        data = values(name)
        return float(1000*data[-1]) if data is not None and np.isfinite(data[-1]) else None

    contact = values('contact_valid').astype(bool)
    slip = values('slip_valid').astype(bool)
    slips = values('slip_m')[slip]
    if not np.isfinite(slips).all():
        raise ValueError('nonfinite valid fixed-anchor slip')
    left, right = values('left_palm_dz_m'), values('right_palm_dz_m')
    sync = np.abs(left-right) if left is not None and right is not None else None
    error = values('right_finger12_tracking_error_rad')
    finite_error = error[np.isfinite(error)] if error is not None else np.array([])
    objective_valid = values('objective_valid')
    objective_mask = (np.isfinite(objective_valid) & (objective_valid == 1)
                      if objective_valid is not None else np.zeros(len(trace), dtype=bool))
    objectives = {}
    for term in OBJECTIVE_TERMS:
        data = values('objective_'+term)
        selected = data[objective_mask & np.isfinite(data)] if data is not None else np.array([])
        objectives[term] = numeric_statistics(selected, recorded=data is not None)
    execution_fields = [values(f'executed_action_{i}') for i in range(24)]
    execution = np.stack(execution_fields, axis=1) if all(value is not None for value in execution_fields) else None
    policy_samples = (len(trace)+7)//8
    if source_requests24 is not None:
        if len(source_requests24) < policy_samples:
            raise ValueError('short source-request action tape')
        source_requests24 = np.asarray(source_requests24)[:policy_samples]
    result = dict(env_id=label['env_id'], trial_id=label['trial_id'], arm=arm,
        episode_id=int(episodes[0]) if episodes is not None else None,
        physical_samples=len(trace), lift_peak_mm=float(1000*values('lift_m').max()),
        hold_max_s=float(values('qualified_hold_s').max()),
        slip_peak_mm=float(1000*slips.max()) if len(slips) else None, slip_valid_samples=len(slips),
        opposed_samples=int((contact & values('bilateral_opposed_contact').astype(bool)).sum()),
        contact_valid_samples=int(contact.sum()),
        left_palm_peak=peak('left_palm_dz_m'), right_palm_peak=peak('right_palm_dz_m'),
        left_final_dz_mm=final_mm('left_palm_dz_m'), right_final_dz_mm=final_mm('right_palm_dz_m'),
        sync_peak_abs_dz_mm=float(1000*sync.max()) if sync is not None else None,
        sync_mean_abs_dz_mm=float(1000*sync.mean()) if sync is not None else None,
        right_finger12_tracking=dict(valid_samples=len(finite_error),
            mean_signed_rad=float(finite_error.mean()) if len(finite_error) else None,
            mean_abs_rad=float(np.abs(finite_error).mean()) if len(finite_error) else None,
            max_abs_rad=float(np.abs(finite_error).max()) if len(finite_error) else None),
        left_load_mean_n=mean('left_load_n'), right_load_mean_n=mean('right_load_n'),
        coordination_objectives=dict(validity=flag_statistics(objective_valid), terms=objectives),
        coordination_flags={key: flag_statistics(values(key)) for key in COORDINATION_FLAGS},
        right_finger12_sdk_native_readback=flag_statistics(values('right_finger12_sdk_target_readback_valid')),
        action_statistics=dict(
            source_request24=action_statistics(source_requests24, sample_hz=60,
                source='issued_actions.npy physical policy request or fixed reference intervention; raw Actor latent unavailable'),
            executed_physical24=action_statistics(execution, sample_hz=480,
                source='native_samples executed_action_0..23 after control projection')))
    for hand in ('left', 'right'):
        opposed = values(hand+'_opposed')
        valid = values(hand+'_contact_valid')
        if valid is None:
            valid = values('hand_contact_valid')
        valid = valid.astype(bool) & contact if valid is not None else contact
        result[hand+'_effective_contact_valid_samples'] = int(valid.sum()) if opposed is not None else 0
        result[hand+'_effective_contact_samples'] = int((valid & opposed.astype(bool)).sum()) if opposed is not None else None
        result[hand+'_effective_contact_fraction'] = (result[hand+'_effective_contact_samples'] / int(valid.sum())
            if opposed is not None and valid.any() else None)
    return result


def summarize_style(rows):
    contact_count = sum(row['contact_valid_samples'] for row in rows)
    result = dict(trials=len(rows), native_success_count=sum(row['native_success'] for row in rows),
        native_success_rate=sum(row['native_success'] for row in rows)/len(rows),
        failure_counts=dict(Counter(row['failure_reason'] for row in rows)),
        lift_mm=distribution([row['lift_peak_mm'] for row in rows]),
        hold_s=distribution([row['hold_max_s'] for row in rows]),
        slip_mm=distribution([row['slip_peak_mm'] for row in rows]),
        slip_valid_samples=sum(row['slip_valid_samples'] for row in rows),
        opposed_contact_fraction=sum(row['opposed_samples'] for row in rows)/contact_count if contact_count else None,
        opposed_samples=sum(row['opposed_samples'] for row in rows), contact_valid_samples=contact_count,
        sync_peak_abs_dz_mm=distribution([row['sync_peak_abs_dz_mm'] for row in rows]),
        sync_mean_abs_dz_mm=distribution([row['sync_mean_abs_dz_mm'] for row in rows]),
        right_finger12_tracking=dict(valid_samples=sum(row['right_finger12_tracking']['valid_samples'] for row in rows),
            trial_mean_signed_rad=distribution([row['right_finger12_tracking']['mean_signed_rad'] for row in rows]),
            trial_max_abs_rad=distribution([row['right_finger12_tracking']['max_abs_rad'] for row in rows])))
    result.update(physical_samples=sum(row['physical_samples'] for row in rows),
        coordination_objectives=dict(
            validity=pool_flag_statistics([row['coordination_objectives']['validity'] for row in rows]),
            terms={term: pool_numeric_statistics([row['coordination_objectives']['terms'][term] for row in rows])
                   for term in OBJECTIVE_TERMS}),
        coordination_flags={key: pool_flag_statistics([row['coordination_flags'][key] for row in rows])
                            for key in COORDINATION_FLAGS},
        right_finger12_sdk_native_readback=pool_flag_statistics([row['right_finger12_sdk_native_readback'] for row in rows]),
        action_statistics={key: pool_action_statistics([row['action_statistics'][key] for row in rows])
                           for key in ('source_request24', 'executed_physical24')})
    for hand in ('left', 'right'):
        peaks = [row[hand+'_palm_peak'] for row in rows if row[hand+'_palm_peak']['value_mm'] is not None]
        result[hand+'_palm_peak_mm'] = distribution([row['value_mm'] for row in peaks])
        result[hand+'_highest_palm_sample'] = max(peaks, key=lambda row: row['value_mm'], default=None)
        result[hand+'_final_dz_mm'] = distribution([row[hand+'_final_dz_mm'] for row in rows])
        result[hand+'_trial_mean_load_n'] = distribution([row[hand+'_load_mean_n'] for row in rows])
        valid_count = sum(row[hand+'_effective_contact_valid_samples'] for row in rows)
        accepted = sum(row[hand+'_effective_contact_samples'] or 0 for row in rows)
        result[hand+'_effective_contact_valid_samples'] = valid_count
        result[hand+'_effective_contact_fraction'] = accepted/valid_count if valid_count else None
    return result


def audit(folders):
    arms = [Arm(folder) for folder in folders]
    expected_names = {'m0', 'right_z_zero', 'z2', 'z3'}
    if len(arms) != 4 or {arm.name for arm in arms} != expected_names:
        raise ValueError('exactly m0, right_z_zero, z2 and z3 native arms required')
    if any(arm.identity != arms[0].identity for arm in arms[1:]):
        raise ValueError('checkpoint, seed or physical reset mismatch')
    m0 = next(arm for arm in arms if arm.name == 'm0')
    reports = {}
    baseline_labels = [(row['trial_id'], row['grasp_style']) for row in arms[0].result['trials']]
    for arm in arms:
        if [(row['trial_id'], row['grasp_style']) for row in arm.result['trials']] != baseline_labels:
            raise ValueError('per-environment trial or grasp style mismatch')
        trials = []
        for i, label in enumerate(arm.result['trials']):
            physical = arm.first(i)
            if arm.name == 'right_z_zero':
                check_replay(m0.actions[:, i], arm.actions[:, i], len(physical), zero_z=True)
            row = trial_metrics(physical, arm.c, label, arm.name, source_requests24=arm.actions[:, i])
            row.update(grasp_style=label['grasp_style'], native_success=label['first_terminal_success'],
                       failure_reason=label['failure_reason'])
            trials.append(row)
        styles = {row['grasp_style'] for row in trials}
        if styles != {'pinch_thumb_index', 'tripod_thumb_index_middle'}:
            raise ValueError('both native grasp styles required')
        groups = {}
        for style in sorted(styles):
            group = [row for row in trials if row['grasp_style'] == style]
            if len(group) < 100:
                raise ValueError('at least 100 native initial states per style required')
            groups[style] = summarize_style(group)
        reports[arm.name] = dict(directory=str(arm.folder), hashes=arm.inputs,
            recorded_physics_fields=sorted(arm.c), by_style=groups, trials=trials,
            fixed_23_channels_independently_verified=True if arm.name == 'right_z_zero' else None)
    return dict(schema='dh116-stage5-z-intervention-audit-v2', identity=arms[0].identity,
        success_contract=dict(lift_m=.03, hold_s=5., fixed_anchor_slip_m=.0005), arms=reports,
        interpretation='first episodes; equal initial physical state; Z2/Z3 are closed-loop control interventions',
        missing_fields='null with zero coverage; no inferred contact, torque or slip',
        action_statistics_contract='source requests sampled at 60Hz; separate projected execution sampled at 480Hz; raw Actor latent unavailable',
        objective_statistics_contract='native coordination diagnostic costs under objective_valid; independent of PPO reward',
        palm_sync_definition='absolute difference of measured reset-relative left and right Z displacement')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', nargs=4, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.arms)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write('\n')
