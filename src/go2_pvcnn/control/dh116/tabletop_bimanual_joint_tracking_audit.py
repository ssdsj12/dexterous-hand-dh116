"""Read-only joint tracking evidence captured after guards, before reset.

The caller supplies all three action boundaries from the same policy step.
``capture_joint_tracking`` performs no simulator update and writes no commands.
Its flags describe observed signatures, not causal attribution. Implicit drive
torque and force-integrated impulse are explicitly simulation estimates.

Call the writer once per live environment at each physical substep, including
the final terminal substep. An explicit reset row admits a new episode UID.
Subsequent no-op rows of closed episodes are counted and never overwrite the
saved terminal sample. Use a new output path for each native experiment.
"""

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .linear_coupling import ACTIVE
from .tabletop_surface_contract import ACTION_CONTRACT


SCHEMA = 'dh116-bimanual-joint-tracking-substep-v2'
TORQUE_SOURCE = 'implicit_actuator_estimate_not_hardware_measurement'


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _safe(value, path='', nonfinite=None):
    if hasattr(value, 'detach') or isinstance(value, np.ndarray):
        value = _array(value).tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {key: _safe(item, f'{path}.{key}' if path else key, nonfinite)
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item, f'{path}.{i}', nonfinite) for i, item in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        if nonfinite is not None:
            nonfinite.append(path)
        return None
    return value


def _native_field(hand, name, env_id, joint_ids):
    try:
        data = _array(getattr(hand.data, name))[env_id, joint_ids]
    except (AttributeError, IndexError, TypeError, RuntimeError):
        return None
    return data.copy() if data.shape == (6,) else None


def _optional_env(values, name, env_id):
    value = values.get(name)
    return None if value is None else _array(value)[env_id].copy()


def _contacts(world, env_id):
    pads = getattr(world, 'pad_measurements', {})
    loads = _optional_env(pads, 'effective_pad_loads_n', env_id)
    vectors = _optional_env(pads, 'effective_pad_vectors_w', env_id)
    validity = _optional_env(pads, 'effective_pad_valid', env_id)
    points = _optional_env(pads, 'effective_pad_points_w', env_id)
    bad_env = pads.get('effective_pad_bad_env')
    known = (loads is not None and loads.shape == (2, 5) and np.isfinite(loads).all()
             and vectors is not None and vectors.shape == (2, 5, 3) and np.isfinite(vectors).all()
             and validity is not None and validity.shape == (2, 5)
             and bad_env is not None and not bool(_array(bad_env)[env_id]))
    point_valid = (np.isfinite(points).all(-1) & validity.astype(bool) & known
        if points is not None and points.shape == (2, 5, 3) and validity is not None
           and validity.shape == (2, 5) else np.zeros((2, 5), dtype=bool))
    return dict(effective_pad_load_n=loads, effective_pad_force_on_hand_w_n=vectors,
        effective_pad_points_w_m=points, effective_pad_point_valid=point_valid,
        effective_pad_point_source='native_accepted_normal_force_weighted_contact_centroid',
        effective_pad_contact_valid=validity, effective_pad_measurement_valid=bool(known),
        force_source='current_native_effective_pad_PhysX_force_on_hand',
        force_integrated_impulse_on_hand_w_ns=vectors * float(world.dt) if known else None,
        accepted_normal_impulse_ns=loads * float(world.dt) if known else None,
        accepted_normal_impulse_formula='effective_pad_loads_n * physics_dt_s',
        accepted_normal_impulse_source='derived_from_native_PhysX_get_contact_data_impulse_divided_by_same_dt',
        impulse_source='force_times_physics_dt_estimate_not_raw_solver_impulse',
        raw_solver_impulse_ns=None, raw_solver_impulse_valid=False,
        effective_pad_depth_m=_optional_env(pads, 'effective_pad_depth_m', env_id))


def _mimic_rows(world, env_id, hand_index, positions, velocities):
    rows = []
    indices = getattr(world, 'mimic_ids', None)
    if indices is None:
        return rows
    names = world.hands[hand_index].joint_names
    for follower, master, ratio, offset in indices[hand_index]:
        expected_position = float(ratio * positions[master] + offset)
        expected_velocity = float(ratio * velocities[master])
        rows.append(dict(follower=names[follower], master=names[master],
            follower_sdk_index=int(follower), master_sdk_index=int(master),
            runtime_ratio=float(ratio), runtime_offset_rad=float(offset),
            measured_position_rad=float(positions[follower]),
            expected_position_rad=expected_position,
            position_residual_rad=float(positions[follower] - expected_position),
            measured_velocity_rad_s=float(velocities[follower]),
            velocity_residual_rad_s=float(velocities[follower] - expected_velocity)))
    return rows


def capture_joint_tracking(world, measurement, *, env_ids, episode_uids, policy_step,
                           actor_raw24, action_before_inverse24, action_after_inverse24,
                           live, terminal, failure_codes, guard_tolerance_rad=.02,
                           event_kind='physics'):
    """Return JSON-safe rows for explicitly selected global environment IDs.

    episode_uids, action arrays and flags are full-world batches. ``world.q``
    and ``world.qd`` must be the current native readback before physical reset.
    Native ``hand.data.joint_pos_target`` supplies the submitted SDK target.
    When unavailable, the exact float32 submission expression is reconstructed
    from the integrated double ``world.commands`` and marked as reconstructed.
    A mismatch between active order and the named contract records an invalid
    mapping and withholds the target comparison.
    """
    if event_kind not in ('physics', 'reset'):
        raise ValueError('event_kind must be physics or reset')
    if type(policy_step) is not int or policy_step < 0:
        raise ValueError('nonnegative policy_step required')
    if not math.isfinite(guard_tolerance_rad) or guard_tolerance_rad < 0:
        raise ValueError('finite nonnegative original guard tolerance required')
    actions = [_array(value) for value in
               (actor_raw24, action_before_inverse24, action_after_inverse24)]
    if any(value.shape != (world.n, 24) or not np.isfinite(value).all() for value in actions):
        raise ValueError('three finite full-world N x 24 action boundaries required')
    selected = list(env_ids)
    if len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < world.n for i in selected):
        raise ValueError('distinct valid global env_ids required')
    if len(episode_uids) != world.n or any(not isinstance(uid, str) or not uid for uid in episode_uids):
        raise ValueError('one nonempty episode_uid per environment required')
    flags = [_array(value) for value in (live, terminal, failure_codes)]
    if any(value.shape != (world.n,) for value in flags):
        raise ValueError('full-world live, terminal and failure arrays required')
    positions, velocities, bounds = [_array(value) for value in (world.q, world.qd, world.limits)]
    commands = _array(world.commands)
    if commands.shape != (world.n, 2, 6):
        raise ValueError('SDK target shape must be N x 2 x 6')
    rows = []
    for env_id in selected:
        mappings = [[hand.joint_names[index] for index in world.w.ids[h]]
                    for h, hand in enumerate(world.hands)]
        valid_mapping = all(tuple(names) == ACTIVE for names in mappings)
        sdk_targets = []
        sdk_target_sources = []
        sdk_target_native_valid = []
        for h, hand in enumerate(world.hands):
            native_target = _native_field(hand, 'joint_pos_target', env_id, world.w.ids[h])
            native_valid = native_target is not None and np.isfinite(native_target).all()
            sdk_targets.append(native_target if native_valid else commands[env_id, h].astype(np.float32).astype(np.float64))
            sdk_target_sources.append('native_articulation_data_joint_pos_target' if native_valid
                else 'float32_submission_reconstructed_from_integrated_command')
            sdk_target_native_valid.append(bool(native_valid))
        actual_right_index = world.hands[1].joint_names.index('finger12')
        q = float(positions[env_id, 1, actual_right_index])
        qd = float(velocities[env_id, 1, actual_right_index])
        lower, upper = map(float, bounds[env_id, 1, actual_right_index])
        measured_joint_valid = bool(np.isfinite([q, qd, lower, upper]).all())
        target = float(sdk_targets[1][1]) if valid_mapping else None
        below = q < lower - guard_tolerance_rad
        above = q > upper + guard_tolerance_rad
        target_valid = target is not None and math.isfinite(target) and lower <= target <= upper
        mimic = [_mimic_rows(world, env_id, h, positions[env_id, h], velocities[env_id, h])
                 for h in range(2)]
        applied = [_native_field(hand, 'applied_torque', env_id, world.w.ids[h])
                   for h, hand in enumerate(world.hands)]
        computed = [_native_field(hand, 'computed_torque', env_id, world.w.ids[h])
                    for h, hand in enumerate(world.hands)]
        rate = ACTION_CONTRACT['joint_rate_rad_s']
        right_command = float(actions[2][env_id, 7])
        command_nonzero = dict(fingers=bool(np.any(actions[2][env_id, :12] != 0)),
            wrist_translation=bool(np.any(actions[2][env_id, 12:18] != 0)),
            wrist_rotation=bool(np.any(actions[2][env_id, 18:24] != 0)))
        roots = getattr(world, 'roots', None)
        base_poses = _array(roots)[env_id].copy() if roots is not None else None
        base_valid = [bool(base_poses is not None and base_poses.shape == (2, 7)
            and np.isfinite(base_poses[h]).all()
            and abs(np.linalg.norm(base_poses[h, 3:])-1.) <= 1e-5) for h in range(2)]
        row = dict(schema=SCHEMA, event_kind=event_kind,
            identity=dict(env_id=env_id, episode_uid=episode_uids[env_id],
                policy_step=policy_step, physics_step=int(world._physics_steps),
                timestamp_s=float(world.w.sim.current_time)),
            physics_dt_s=float(world.dt), live=bool(flags[0][env_id]),
            terminal=bool(flags[1][env_id]), failure_code=int(flags[2][env_id]),
            actor_raw24=actions[0][env_id], action_before_inverse24=actions[1][env_id],
            action_after_inverse24=actions[2][env_id],
            action_contract=ACTION_CONTRACT['version'],
            active_joint_names=mappings, joint_mapping_valid=valid_mapping,
            integrated_active_command_before_float32_rad=commands[env_id],
            sdk_active_target_rad=sdk_targets, sdk_target_sources=sdk_target_sources,
            sdk_target_native_readback_valid=sdk_target_native_valid,
            left_full_position_rad=positions[env_id, 0], right_full_position_rad=positions[env_id, 1],
            left_full_velocity_rad_s=velocities[env_id, 0], right_full_velocity_rad_s=velocities[env_id, 1],
            full_joint_names=[list(hand.joint_names) for hand in world.hands],
            actual_base_link_pose_w=base_poses, actual_base_link_pose_valid=base_valid,
            actual_base_link_pose_source='TensorWorld.current_native_base_link_readback_wxyz',
            right_finger12=dict(sdk_joint_index=actual_right_index, physical_action_index=7,
                physical_normalized_velocity_command=right_command,
                requested_velocity_rad_s=right_command * rate,
                requested_velocity_source='normalized_action_times_original_contract_rate_before_guards',
                sdk_target_rad=target, position_rad=q, velocity_rad_s=qd,
                sdk_target_source=sdk_target_sources[1],
                integrated_command_before_float32_rad=float(commands[env_id, 1, 1]) if valid_mapping else None,
                target_minus_actual_rad=target-q if target is not None else None,
                nominal_lower_rad=lower, nominal_upper_rad=upper,
                measured_joint_valid=measured_joint_valid,
                guard_tolerance_rad=guard_tolerance_rad,
                lower_guard_rad=lower-guard_tolerance_rad, upper_guard_rad=upper+guard_tolerance_rad,
                target_within_nominal_limits=bool(target_valid), actual_below_guard=bool(below),
                actual_above_guard=bool(above),
                joint_limit_observation=('right_finger12_measurement_invalid' if not measured_joint_valid else
                    'right_finger12_below_lower_guard' if below else
                    'right_finger12_above_upper_guard' if above else 'right_finger12_inside_guard')),
            passive_coupling=mimic,
            torque=dict(source=TORQUE_SOURCE, applied_estimate_nm=applied,
                computed_estimate_nm=computed,
                applied_valid=[bool(valid_mapping and v is not None and np.isfinite(v).all()) for v in applied],
                computed_valid=[bool(valid_mapping and v is not None and np.isfinite(v).all()) for v in computed],
                hardware_measured_torque_nm=None, actual_solver_drive_torque_nm=None),
            contacts=_contacts(world, env_id),
            hand_object_load_n=_optional_env(measurement, 'hand_object_loads', env_id),
            wrist_world_translation_action=actions[2][env_id, 12:18].reshape(2, 3),
            wrist_intrinsic_xyz_rotation_action=actions[2][env_id, 18:24].reshape(2, 3),
            observed_signatures=dict(mapping_contract_mismatch=not valid_mapping,
                legal_target_actual_guard_violation=bool(target_valid and (below or above)),
                passive_coupling_residual_present=any(abs(v['position_residual_rad']) > .02
                    for hand in mimic for v in hand),
                simultaneous_command_channels=command_nonzero),
            root_cause_proven=False,
            unresolved_contributions=['contact_impulse', 'passive_coupling', 'wrist_finger_interaction'])
        nonfinite = []
        row = _safe(row, nonfinite=nonfinite)
        row['nonfinite_fields'] = nonfinite
        rows.append(row)
    return rows


class JointTrackingAuditWriter:
    """Exclusive JSONL output with per-episode clock and terminal protection."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open('x', encoding='utf-8')
        self.terminal_rows = {}
        self.skipped_closed_episode_rows = 0
        self.rows_written = 0
        self._previous = {}
        self._env_episode = {}
        self._uid_env = {}
        self._env_clock = {}

    def append(self, row):
        if row.get('schema') != SCHEMA:
            raise ValueError('joint tracking schema required')
        identity = row['identity']
        env_id, uid = identity['env_id'], identity['episode_uid']
        step, timestamp = identity['physics_step'], identity['timestamp_s']
        if (type(env_id) is not int or env_id < 0 or not isinstance(uid, str) or not uid
                or type(step) is not int or step < 0 or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)):
            raise ValueError('valid episode identity and physical clock required')
        previous_uid = self._env_episode.get(env_id)
        reset = row.get('event_kind') == 'reset'
        if uid in self._uid_env and self._uid_env[uid] != env_id:
            raise ValueError('episode_uid reused by a different environment')
        if previous_uid != uid and uid in self._previous:
            raise ValueError('episode_uid reused after a different episode')
        if previous_uid is not None and previous_uid != uid and not reset:
            raise ValueError('new episode requires explicit physical reset row')
        if uid in self.terminal_rows and not reset:
            self.skipped_closed_episode_rows += 1
            return False
        if reset and uid in self._previous:
            raise ValueError('reset reused episode_uid')
        env_clock = self._env_clock.get(env_id)
        if env_clock is not None and (step < env_clock[0] or timestamp < env_clock[1]):
            raise ValueError('physical clock regressed across episode reset')
        previous = self._previous.get(uid)
        if previous is not None and (step <= previous[0] or timestamp <= previous[1]
                                     or identity['policy_step'] < previous[2]):
            raise ValueError('nonmonotonic or duplicate per-episode physical clock')
        if row.get('terminal') and not row.get('live'):
            raise ValueError('terminal must be the final live physics sample')
        encoded = json.dumps(row, allow_nan=False, separators=(',', ':'))
        self.stream.write(encoded + '\n')
        self._previous[uid] = (step, timestamp, identity['policy_step'])
        self._env_episode[env_id] = uid
        self._uid_env[uid] = env_id
        self._env_clock[env_id] = (step, timestamp)
        self.rows_written += 1
        if row.get('terminal'):
            self.terminal_rows[uid] = deepcopy(row)
            self.stream.flush()
        return True

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
