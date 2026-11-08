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
        reconstructed_normal_contact_moment=row.get('reconstructed_normal_contact_moment'))


def summarize(path, replay_path=None, moment_model=None):
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
            terminal_sample=terminal[0] if terminal else None, last_eight_samples=trace[-8:]))
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
    return dict(schema='dh116-stage5-joint-tracking-analysis-v1', source=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), episodes=episodes,
        source_schemas=sorted({row['schema'] for rows in groups.values() for row in rows}),
        target_crosscheck=target_crosscheck,
        legacy_v1_target_caveat='v1 sdk_target_rad contains integrated pre-float32 world.commands; use float32 replay crosscheck',
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
    args = parser.parse_args()
    model = None
    if args.asset is not None:
        if hashlib.sha256(args.asset.read_bytes()).hexdigest() != args.asset_sha256:
            raise ValueError('native source USD SHA-256 mismatch or missing expected SHA')
        from go2_pvcnn.control.dh116.tabletop_bimanual_physical_action_adapter import HandPhysicalKinematics
        model = HandPhysicalKinematics.from_usd(args.asset, 'right')
    elif args.asset_sha256 is not None:
        raise ValueError('--asset required with --asset-sha256')
    result = summarize(args.input, args.replay, model)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write('\n')
