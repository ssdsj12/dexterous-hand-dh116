"""Summarize native T1 readback without starting Isaac or inferring causation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def records(path):
    with path.open('rb') as stream:
        for index, line in enumerate(stream, 1):
            if not line.endswith(b'\n'):
                raise ValueError(f'incomplete JSONL line {index}')
            yield json.loads(line)


def scalar_stats(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(valid_count=0, p50=None, p90=None, p95=None, min=None, max=None)
    return dict(valid_count=len(values), p50=float(np.percentile(values, 50)),
        p90=float(np.percentile(values, 90)), p95=float(np.percentile(values, 95)),
        min=float(values.min()), max=float(values.max()))


def reconstruct_normal_contact_moment(row, model):
    """Approximate accepted-normal moment about right finger12's actual axis.

    FK uses measured full joint angles and the native base pose. Each measured
    pad centroid carries its summed normal force. Varying contact normals can
    leave an unresolved intrapatch couple, so this is not the exact sum of
    manifold moments, a total contact wrench, or an actuator/solver torque.
    """
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS
    result = dict(valid=False, normal_pad_moment_nm=None, contributing_pads=[],
        identity=row['identity'], source_asset_sha256=model.source_sha256,
        formula='axis_world dot ((native_contact_centroid_world - joint_origin_world) cross accepted_normal_force_on_hand_world)',
        frame='world;positive_moment_follows_authored_right_finger12_positive_axis',
        source='measured_base_and_full_q_USD_FK_with_native_accepted_normal_centroid_and_force',
        exact_manifold_moment=False, unresolved_intrapatch_couple_nm=None,
        friction_moment_nm=None, rejected_contact_moment_nm=None,
        passive_coupling_reaction_nm=None, solver_or_motor_torque_nm=None)
    roots = row.get('actual_base_link_pose_w')
    if roots is None or not row.get('actual_base_link_pose_valid', [False, False])[1]:
        return dict(result, reason='measured_base_pose_unavailable')
    contact = row['contacts']
    if not contact.get('effective_pad_measurement_valid'):
        return dict(result, reason='native_effective_contact_measurement_invalid')
    names = row['full_joint_names'][1]
    full = np.asarray(row['right_full_position_rad'], dtype=float)
    if len(names) != len(full) or any(names.count(name) != 1 for name in JOINTS):
        return dict(result, reason='measured_joint_name_mapping_invalid')
    q11 = np.array([full[names.index(name)] for name in JOINTS])
    try:
        _, axes, ancestors = model._fk(q11, np.asarray(roots[1], dtype=float))
        origin, axis = axes['finger12']
    except (ValueError, KeyError, TypeError):
        return dict(result, reason='measured_state_fk_invalid')
    if model.side != 'right' or not np.isfinite(np.r_[origin, axis]).all():
        return dict(result, reason='right_joint_axis_invalid')
    force = np.asarray(contact['effective_pad_force_on_hand_w_n'], dtype=float)
    loads = np.asarray(contact['effective_pad_load_n'], dtype=float)
    if force.shape != (2, 5, 3) or loads.shape != (2, 5):
        return dict(result, reason='accepted_pad_force_shape_invalid')
    points = contact.get('effective_pad_points_w_m')
    point_valid = contact.get('effective_pad_point_valid')
    digits = ('finger14_link', 'finger23_link', 'finger33_link', 'finger43_link', 'finger53_link')
    contributions = []
    for digit_index, body_name in enumerate(digits):
        body = next((key for key in ancestors if key.rsplit('/', 1)[-1] == body_name), None)
        if body is None:
            return dict(result, reason='source_distal_body_missing')
        if 'finger12' not in ancestors[body]:
            continue
        vector = force[1, digit_index]
        if not np.isfinite(vector).all() or not np.isfinite(loads[1, digit_index]):
            return dict(result, reason='accepted_normal_force_invalid')
        if loads[1, digit_index] == 0 and not vector.any():
            continue
        if (points is None or point_valid is None or not point_valid[1][digit_index]
                or not np.isfinite(points[1][digit_index]).all()):
            return dict(result, reason='relevant_positive_contact_point_invalid')
        point = np.asarray(points[1][digit_index], dtype=float)
        moment = float(np.dot(axis, np.cross(point-origin, vector)))
        contributions.append(dict(body_name=body_name, digit_index=digit_index,
            point_w_m=point.tolist(), force_on_hand_w_n=vector.tolist(),
            normal_moment_about_finger12_nm=moment))
    moment = sum(item['normal_moment_about_finger12_nm'] for item in contributions)
    applied = row.get('torque', {}).get('applied_estimate_nm')
    drive = applied[1][1] if applied is not None and applied[1] is not None else None
    return dict(result, valid=True, reason=None, normal_pad_moment_nm=moment,
        contributing_pads=contributions, joint_origin_w_m=np.asarray(origin).tolist(),
        joint_axis_w=np.asarray(axis).tolist(),
        opposite_sign_to_implicit_drive_estimate=bool(moment*drive < 0) if drive is not None else None,
        approximation='sum of per-pad centroid-resultant moments; excludes friction, rejected contacts, intrapatch couples and passive reactions')


def compact(row):
    finger = row['right_finger12']
    torque = row['torque']['applied_estimate_nm'][1]
    mimic = next((entry for entry in row['passive_coupling'][1]
                  if entry['follower'] == 'finger13'), None)
    contacts = row['contacts']
    return dict(**row['identity'], terminal=row['terminal'], failure_code=row['failure_code'],
        q_rad=finger['position_rad'], qd_rad_s=finger['velocity_rad_s'],
        target_rad=finger['sdk_target_rad'], error_rad=finger['target_minus_actual_rad'],
        physical_command=finger['physical_normalized_velocity_command'],
        estimated_torque_nm=torque[1] if torque is not None else None,
        thumb_follower_residual_rad=mimic['position_residual_rad'] if mimic else None,
        right_pad_load_n=contacts['effective_pad_load_n'][1]
            if contacts['effective_pad_measurement_valid'] else None,
        right_pad_normal_impulse_ns=contacts['accepted_normal_impulse_ns'][1]
            if contacts['effective_pad_measurement_valid'] else None,
        right_wrist_world_translation_action=row['wrist_world_translation_action'][1],
        right_wrist_intrinsic_xyz_rotation_action=row['wrist_intrinsic_xyz_rotation_action'][1],
        hand_object_load_n=row.get('hand_object_load_n'), wrist_tracking_guard=row.get('wrist_tracking_guard'),
        reconstructed_normal_contact_moment=row.get('reconstructed_normal_contact_moment'))


def passive_lower_stop_timeline(rows, replay):
    """Order actual thumb follower, master and contact events in one episode."""
    first = rows[0]
    names = first['full_joint_names'][1]
    finger13 = names.index('finger13')
    identity = first['identity']
    bounds = replay[identity['episode_uid'], identity['physics_step']]['limits'][1][finger13]
    lower = bounds[0]
    samples = []
    for row in rows:
        follower = next(entry for entry in row['passive_coupling'][1] if entry['follower'] == 'finger13')
        samples.append(dict(**row['identity'], q12_rad=row['right_finger12']['position_rad'],
            q13_rad=follower['measured_position_rad'], affine_q13_rad=follower['expected_position_rad'],
            coupling_residual_rad=follower['position_residual_rad'],
            thumb_load_n=row['contacts']['effective_pad_load_n'][1][0]
                if row['contacts']['effective_pad_measurement_valid'] else None))
    near = [sample for sample in samples if abs(sample['q13_rad']-lower) <= 1e-6]
    first_near = near[0] if near else None
    after = [sample for sample in samples if first_near is not None
             and sample['physics_step'] >= first_near['physics_step']]
    nominal = first['right_finger12']['nominal_lower_rad']
    guard = first['right_finger12']['lower_guard_rad']
    return dict(passive_joint_name='finger13', actual_sdk_joint_index=finger13,
        nominal_bounds_rad=bounds, numerical_near_lower_band_rad=1e-6,
        first_positive_thumb_contact=next((sample for sample in samples
            if sample['thumb_load_n'] is not None and sample['thumb_load_n'] > 0), None),
        first_thumb_contact_above_005n=next((sample for sample in samples
            if sample['thumb_load_n'] is not None and sample['thumb_load_n'] > .05), None),
        first_passive_near_lower=first_near,
        subsequent_samples=len(after),
        max_abs_passive_distance_from_lower_after_first_near_rad=max(
            (abs(sample['q13_rad']-lower) for sample in after), default=None),
        first_master_below_nominal_lower=next((sample for sample in samples if sample['q12_rad'] < nominal), None),
        first_master_below_guard=next((sample for sample in samples if sample['q12_rad'] < guard), None),
        terminal_sample=samples[-1],
        interpretation='observed stop proximity and event order; passive constraint reaction and causality unmeasured')


def compare_tracking_guard(baseline_path, observed_path, moment_model=None):
    """Verify fixed finger requests and retain each hand's first pause context."""
    def load(path):
        groups = {}
        for row in records(path):
            if moment_model is not None:
                row['reconstructed_normal_contact_moment'] = reconstruct_normal_contact_moment(row, moment_model)
            groups.setdefault(row['identity']['episode_uid'], []).append(row)
        return groups

    baseline, observed = load(baseline_path), load(observed_path)
    reports = []
    for uid, original in baseline.items():
        alternatives = {row['identity']['physics_step']: row for row in observed[uid]}
        matched = [(row, alternatives[row['identity']['physics_step']]) for row in original
                   if row['identity']['physics_step'] in alternatives]
        if not matched or matched[0][0]['identity']['physics_step'] != original[0]['identity']['physics_step']:
            raise ValueError('missing common initial physical sample')
        common_steps = [row['identity']['physics_step'] for row, _ in matched]
        if np.any(np.diff(common_steps) != 1):
            raise ValueError('gap in common first-episode physical samples')
        if len(matched) < len(original) and not observed[uid][-1]['terminal']:
            raise ValueError('alternative truncated without a first terminal')
        finger_error = max(np.max(np.abs(np.array(a.get('executed_physical_action24', a['action_after_inverse24'])[:12])
            -np.array(b.get('executed_physical_action24', b['action_after_inverse24'])[:12]))) for a, b in matched)
        actor_error = max(np.max(np.abs(np.array(a['actor_raw24'])-np.array(b['actor_raw24'])))
            for a, b in matched)
        sdk_error = max(abs(float(np.float32(a['right_finger12']['sdk_target_rad']))
            -b['right_finger12']['sdk_target_rad']) for a, b in matched)
        all_sdk_error = np.max(np.stack([np.abs(np.asarray(a['sdk_active_target_rad'], dtype=np.float32).astype(float)
            -np.asarray(b['sdk_active_target_rad'], dtype=np.float32).astype(float)) for a, b in matched]), axis=(0, 2))
        full_q_error = max(np.max(np.abs(np.asarray(a[hand+'_full_position_rad'])
            -np.asarray(b[hand+'_full_position_rad']))) for a, b in matched for hand in ('left', 'right'))
        pauses = []
        for hand_index, side in enumerate(('left', 'right')):
            rows = observed[uid]
            paused = [row for row in rows if row.get('wrist_tracking_guard', {}).get('pause_hand', [False, False])[hand_index]]
            first = paused[0] if paused else None
            previous = alternatives.get(first['identity']['physics_step']-1) if first else None
            before = None
            if previous is not None:
                names = previous['full_joint_names'][hand_index]
                q = previous[side+'_full_position_rad']
                actual = [q[names.index(name)] for name in previous['active_joint_names'][hand_index]]
                target = previous['sdk_active_target_rad'][hand_index]
                before = dict(identity=previous['identity'],
                    max_active_tracking_error_rad=float(np.max(np.abs(np.array(actual)-np.array(target)))),
                    max_mimic_residual_rad=max(abs(entry['position_residual_rad'])
                        for entry in previous['passive_coupling'][hand_index]))
            pauses.append(dict(hand=side, paused_sample_count=len(paused),
                first_pause_identity=first['identity'] if first else None,
                first_pause_previous_measured_sample=before))
        first_pause_step = min((item['first_pause_identity']['physics_step'] for item in pauses
            if item['first_pause_identity'] is not None), default=float('inf'))
        prefix = [(a, b) for a, b in matched if b['identity']['physics_step'] < first_pause_step]
        prefix_evidence = dict(common_samples=len(prefix),
            measured_full_q_max_abs_difference_rad=max((np.max(np.abs(np.asarray(a[side+'_full_position_rad'])
                -np.asarray(b[side+'_full_position_rad']))) for a, b in prefix for side in ('left', 'right')), default=None))
        for key in ('actual_base_link_pose_w', 'hand_object_load_n'):
            available = prefix and all(a.get(key) is not None and b.get(key) is not None for a, b in prefix)
            prefix_evidence[key+'_max_abs_difference'] = float(max(np.max(np.abs(np.asarray(a[key])
                -np.asarray(b[key]))) for a, b in prefix)) if available else None
        reports.append(dict(episode_uid=uid, common_physics_samples=len(matched),
            baseline_physics_samples=len(original), observed_physics_samples=len(observed[uid]),
            actor_raw_max_abs_difference=float(actor_error), finger12d_request_max_abs_difference=float(finger_error),
            right_finger12_float32_submission_max_abs_difference_rad=float(sdk_error),
            per_hand_active6_float32_submission_max_abs_difference_rad=all_sdk_error.tolist(),
            measured_full_q_max_abs_difference_rad=float(full_q_error),
            common_prefix_before_first_observed_pause=prefix_evidence,
            per_hand_pauses=pauses, original_terminal=bool(original[-1]['terminal']),
            observed_terminal=bool(observed[uid][-1]['terminal']),
            common_end_baseline=compact(matched[-1][0]), common_end_observed=compact(matched[-1][1]),
            original_last=compact(original[-1]), observed_last=compact(observed[uid][-1])))
    summary_files = [path.parent/'joint_replay_summary.json' for path in (baseline_path, observed_path)]
    metadata = [json.loads(path.read_text()) for path in summary_files]
    if metadata[0]['initial_state_sha256'] != metadata[1]['initial_state_sha256']:
        raise ValueError('guard and baseline physical initial states differ')
    if metadata[0]['source_policy_sha256'] != metadata[1]['source_policy_sha256']:
        raise ValueError('guard and baseline original Actor tapes differ')
    return dict(physical_initial_state_sha256=metadata[0]['initial_state_sha256'],
        source_policy_sha256=metadata[0]['source_policy_sha256'], episodes=reports,
        baseline_terminal_events=metadata[0]['terminal_events'], observed_terminal_events=metadata[1]['terminal_events'])


def summarize(path, replay_path=None, moment_model=None, selected_steps=()):
    groups = {}
    for row in records(path):
        if moment_model is not None:
            row['reconstructed_normal_contact_moment'] = reconstruct_normal_contact_moment(row, moment_model)
        identity = row['identity']
        uid = identity['episode_uid']
        group = groups.setdefault(uid, [])
        if group:
            previous = group[-1]
            if (identity['env_id'] != previous['identity']['env_id']
                    or identity['physics_step'] <= previous['identity']['physics_step']
                    or identity['timestamp_s'] <= previous['identity']['timestamp_s']
                    or previous['terminal']):
                raise ValueError('invalid identity, clock or sample after terminal')
        group.append(row)
    episodes = []
    for uid, rows in groups.items():
        terminal = [row for row in rows if row['terminal']]
        if len(terminal) > 1:
            raise ValueError('duplicate terminal')
        trace = [compact(row) for row in rows]
        finger_rows = [row['right_finger12'] for row in rows]
        errors = [abs(row['error_rad']) if row['error_rad'] is not None else None for row in trace]
        peak = max((row for row in trace if row['error_rad'] is not None),
                   key=lambda row: abs(row['error_rad']), default=None)
        episodes.append(dict(episode_uid=uid, env_id=rows[0]['identity']['env_id'],
            row_count=len(rows), first_identity=rows[0]['identity'], last_identity=rows[-1]['identity'],
            terminal_count=len(terminal), all_joint_mappings_valid=all(row['joint_mapping_valid'] for row in rows),
            all_sdk_targets_within_nominal_limits=all(row['target_within_nominal_limits'] for row in finger_rows),
            q_min_rad=min(row['position_rad'] for row in finger_rows),
            qd_abs_rad_s=scalar_stats(abs(row['qd_rad_s']) if row['qd_rad_s'] is not None else None for row in trace),
            abs_tracking_error_rad=scalar_stats(errors), peak_tracking_error_sample=peak,
            abs_thumb_coupling_residual_rad=scalar_stats(abs(row['thumb_follower_residual_rad'])
                if row['thumb_follower_residual_rad'] is not None else None for row in trace),
            estimated_torque_nm=scalar_stats(row['estimated_torque_nm'] for row in trace),
            reconstructed_normal_contact_moment_nm=scalar_stats(
                row['reconstructed_normal_contact_moment']['normal_pad_moment_nm']
                if row.get('reconstructed_normal_contact_moment', {}).get('valid') else None for row in rows),
            terminal_sample=terminal[0] if terminal else None, last_eight_samples=trace[-8:],
            selected_physics_samples=[row for row in trace if row['physics_step'] in selected_steps]))
    target_crosscheck = None
    if replay_path is not None:
        replay = {(row['episode_uid'], row['physics_step']): row for row in records(replay_path)}
        differences = []
        submission_matches = []
        for rows in groups.values():
            for row in rows:
                identity = row['identity']
                reference = replay[identity['episode_uid'], identity['physics_step']]
                target = row['right_finger12']['sdk_target_rad']
                sdk_float = reference['sdk_active_target_rad'][1][1]
                differences.append(abs(target-sdk_float))
                submission_matches.append(float(np.float32(target)) == sdk_float)
        target_crosscheck = dict(replay_source=str(replay_path), checked_rows=len(differences),
            all_float32_reconstructed_targets_match=all(submission_matches),
            max_integrated_double_vs_submitted_float32_rad=max(differences))
        for episode in episodes:
            episode['passive_lower_stop_timeline'] = passive_lower_stop_timeline(groups[episode['episode_uid']], replay)
    return dict(schema='dh116-stage5-joint-tracking-analysis-v1', source=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), episodes=episodes,
        source_schemas=sorted({row['schema'] for rows in groups.values() for row in rows}),
        target_crosscheck=target_crosscheck,
        legacy_v1_target_caveat='v1 sdk_target_rad contains integrated pre-float32 world.commands; use float32 replay crosscheck',
        legacy_v2_guard_boundary_caveat='guard schema v2 action_after_inverse24 contains post-guard execution; schema v3 records pure inverse and executed request separately',
        root_cause_proven=False,
        estimate_limitations=['implicit actuator torque is not hardware torque or actual solver drive torque',
            'pad normal impulse is derived from the current PhysX force report times physics dt',
            'observational q/qd, torque, mimic and contact coincidence cannot establish a unique causal contribution'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--asset', type=Path, help='actual frozen dual-hand USD for normal-moment reconstruction')
    parser.add_argument('--asset-sha256', help='required exact native source USD SHA-256 when --asset is supplied')
    parser.add_argument('--baseline', type=Path, help='original native T1 audit for exact finger-request/first-pause comparison')
    parser.add_argument('--at-step', type=int, action='append', default=[], help='retain an additional keyed physics sample per episode')
    args = parser.parse_args()
    model = None
    if args.asset is not None:
        if hashlib.sha256(args.asset.read_bytes()).hexdigest() != args.asset_sha256:
            raise ValueError('native source USD SHA-256 mismatch or missing expected SHA')
        from go2_pvcnn.control.dh116.tabletop_bimanual_physical_action_adapter import HandPhysicalKinematics
        model = HandPhysicalKinematics.from_usd(args.asset, 'right')
    elif args.asset_sha256 is not None:
        raise ValueError('--asset required with --asset-sha256')
    result = summarize(args.input, args.replay, model, args.at_step)
    if args.baseline is not None:
        result['guard_comparison'] = compare_tracking_guard(args.baseline, args.input, model)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write('\n')
