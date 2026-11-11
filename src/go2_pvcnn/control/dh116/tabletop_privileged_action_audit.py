"""Read-only Teacher position guard and action evidence from live pre-reset samples."""

import math


VIOLATION_SOURCES = (
    'joint_position_limit', 'joint_velocity_limit', 'action_target_out_of_range',
    'controller_projection', 'passive_contact_limit', 'kinematic_singularity',
    'unknown_joint_limit',
)


def _finite(value):
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _margin(value, lower, upper):
    return None if None in (value, lower, upper) else min(value-lower, upper-value)


def joint_evidence(*, hand, name, position, velocity, lower, upper, hard_tolerance,
                   training_tolerance, velocity_limit, sdk_target, controller_target,
                   action_group, action_index, unit):
    """Joint position, target and velocity comparisons in the joint's own units."""
    position, velocity, lower, upper = map(_finite, (position, velocity, lower, upper))
    sdk_target, controller_target, velocity_limit = map(_finite,
        (sdk_target, controller_target, velocity_limit))
    if lower is not None and upper is not None and lower > upper:
        raise ValueError('inverted runtime joint position limits')
    nominal_margin = _margin(position, lower, upper)
    guard_margin = None if nominal_margin is None else nominal_margin+hard_tolerance
    training_margin = (None if nominal_margin is None or training_tolerance is None else
        nominal_margin+training_tolerance)
    velocity_margin = (None if velocity is None or velocity_limit is None else
        velocity_limit-abs(velocity))
    return dict(hand=hand, joint=name, position_unit=unit, velocity_unit=f'{unit}/s',
        position=position, velocity=velocity, lower_limit=lower, upper_limit=upper,
        hard_guard_tolerance=hard_tolerance, training_guard_tolerance=training_tolerance,
        nominal_margin=nominal_margin, guard_margin=guard_margin,
        training_guard_margin=training_margin, velocity_limit=velocity_limit,
        velocity_margin=velocity_margin, sdk_target=sdk_target, controller_target=controller_target,
        target_margin=_margin(sdk_target, lower, upper),
        controller_target_margin=_margin(controller_target, lower, upper),
        action_group=action_group, action_index=action_index)


def _negative(value, *, tolerance=0.):
    return value is not None and value < -tolerance


def classify_joint_evidence(joints, *, failure_code, projection=None):
    """Separate established guard crossings from co-occurring observations."""
    from .tabletop_tensor_task import FAILURE_CODES

    positions, velocities, targets = [], [], []
    for joint in joints:
        identity = {k: joint[k] for k in ('hand', 'joint', 'action_group', 'action_index')}
        if _negative(joint['guard_margin']):
            positions.append(dict(identity, guard='full_articulation_position'))
        elif _negative(joint['training_guard_margin']):
            positions.append(dict(identity, guard='early_active_training_position'))
        if _negative(joint['velocity_margin'], tolerance=1e-5):
            velocities.append(identity)
        if (_negative(joint['target_margin'], tolerance=1e-6)
                or _negative(joint['controller_target_margin'], tolerance=1e-6)):
            targets.append(identity)
    observed = []
    if velocities:
        observed.append('joint_velocity_limit')
    if targets:
        observed.append('action_target_out_of_range')
    if projection is True:
        observed.append('controller_projection')
    source = None
    if failure_code == FAILURE_CODES['joint_limit']:
        source = 'joint_position_limit' if positions else 'unknown_joint_limit'
    return dict(violation_source=source, concurrent_evidence=observed,
        position_crossings=positions, velocity_exceedances=velocities,
        out_of_range_targets=targets, observed_palm_projection=projection,
        passive_contact_limit=None, kinematic_singularity=None)


class TeacherJointActionAudit:
    """Bind runtime actuator limits once, then observe each live substep."""

    def __init__(self, world):
        self.velocity_limits = []
        for hand in world.hands:
            read = getattr(getattr(hand, 'root_physx_view', None), 'get_dof_max_velocities', None)
            self.velocity_limits.append(None if not callable(read) else read().detach().cpu().clone())

    def capture(self, world, task, env, i, row, policy):
        from .tabletop_palm_mount import ANGULAR_JOINT_NAMES, JOINT_NAMES
        from .tabletop_tensor_env import JOINT_LIMIT_RESET_CONTRACT, TRAINING_JOINT_LIMIT_CONTRACT

        hard = JOINT_LIMIT_RESET_CONTRACT['joint_margin']
        early = getattr(env, 'training_joint_limit', False) and not env.continuous_motion
        command = task.command[i].detach().cpu().tolist()
        position, velocity, limits = (v[i].detach().cpu().tolist()
            for v in (world.q, world.qd, world.limits))
        active_limits = ((task.low.detach().cpu().tolist(), task.high.detach().cpu().tolist())
            if early else None)
        mounts = getattr(world.w, 'mounts', None)
        wrist_command = getattr(world, 'wrist_angles', None)
        world_target = getattr(world, 'root_targets', None)
        joints = []
        for hi, hand in enumerate(world.hands):
            side = ('left', 'right')[hi]
            active = {int(joint): hi*6+index for index, joint in enumerate(world.w.ids[hi])}
            mount = None if mounts is None else mounts[hi]
            rotation_indices = ({joint: axis for axis, joint in enumerate(mount.angular_ids)}
                if mount is not None and getattr(task, 'free_wrist_grasp', False) else {})
            sdk_tensor = getattr(hand.data, 'joint_pos_target', None)
            sdk = None if sdk_tensor is None else sdk_tensor[i].detach().cpu().tolist()
            max_velocity = self.velocity_limits[hi]
            for index, name in enumerate(hand.joint_names):
                finger_action = active.get(index)
                linear = name in JOINT_NAMES
                angular = name in ANGULAR_JOINT_NAMES
                group = (f'{side}_fingers' if finger_action is not None else
                    f'{side}_translation' if linear else f'{side}_rotation' if angular else None)
                rotation_axis = rotation_indices.get(index)
                action_index = 18+hi*3+rotation_axis if rotation_axis is not None else finger_action
                controller_target = None if finger_action is None else command[finger_action]
                if rotation_axis is not None and wrist_command is not None:
                    controller_target = wrist_command[i, hi, rotation_axis]
                is_controlled = (finger_action is not None or mount is not None
                    and index in (*mount.ids, *mount.angular_ids))
                item = joint_evidence(hand=side, name=name, position=position[hi][index],
                    velocity=velocity[hi][index], lower=limits[hi][index][0],
                    upper=limits[hi][index][1], hard_tolerance=hard,
                    training_tolerance=None,
                    velocity_limit=None if max_velocity is None else max_velocity[i, index],
                    sdk_target=sdk[index] if sdk is not None and is_controlled else None,
                    controller_target=controller_target, action_group=group,
                    action_index=action_index, unit='m' if linear else 'rad')
                if early and finger_action is not None:
                    lo, hi_limit = (x[finger_action] for x in active_limits)
                    margin = _margin(item['position'], lo, hi_limit)
                    tolerance = TRAINING_JOINT_LIMIT_CONTRACT['outside_nominal_limit_rad']
                    item.update(training_guard_tolerance=tolerance,
                        training_guard_margin=None if margin is None else margin+tolerance,
                        training_lower_limit=lo, training_upper_limit=hi_limit)
                joints.append(item)
        projection = row.get('palm_safety_projection')
        projected = None if projection is None else bool((
            (projection['pre_projection_palm_target_m']-projection['projected_palm_target_m']).abs()
            > 1e-9).any())
        return dict(version='dh116-privileged-action-audit-v1', joints=joints,
            sample_source='actual_pre_reset_physics_substep',
            velocity_limit_source=[None if v is None else
                'PhysX.get_dof_max_velocities_cached_at_first_audited_substep' for v in self.velocity_limits],
            controller_wrist_world_target_m=None if world_target is None else world_target[i, :, :3],
            actor_mean=row['teacher_mean_action'], latent_mean=policy['latent_mean'][i],
            latent_std=policy['latent_std'][i], sampled_latent=policy['latent_actions'][i],
            sampled_action=row['teacher_sampled_action'], issued_action=row['actually_issued_action'],
            **classify_joint_evidence(joints, failure_code=row['failure_code'], projection=projected))


def summarize_joint_terminals(rows):
    """Use monotone per-episode rows to retain the first observable crossing."""
    from .tabletop_tensor_task import FAILURE_CODES

    first_crossings, previous, terminals = {}, {}, []
    for row in rows:
        if row.get('kind') != 'sample':
            continue
        key = (row['run_uuid'], row['env_id'], row['episode_uid'])
        if key in previous:
            prior = previous[key]
            if prior['terminal']:
                raise ValueError('live sample after terminal')
            if (row['physics_step'] <= prior['physics_step']
                    or row['timestamp'] <= prior['timestamp']):
                raise ValueError('nonmonotone physical evidence')
        previous[key] = row
        audit = row.get('action_audit') or {}
        identity = {k: row[k] for k in ('run_uuid', 'env_id', 'episode_uid', 'policy_tick',
            'physics_step', 'timestamp')}
        if audit.get('position_crossings'):
            first_crossings.setdefault(key, dict(identity, joints=audit['position_crossings']))
        if row['terminal'] and row['failure_code'] == FAILURE_CODES['joint_limit']:
            crossing_names = {(j['hand'], j['joint']) for j in audit.get('position_crossings', [])}
            terminals.append(dict(identity,
                violation_source=audit.get('violation_source') or 'unknown_joint_limit',
                terminal_position_crossings=audit.get('position_crossings', []),
                crossing_joint_values=[j for j in audit.get('joints', [])
                    if (j['hand'], j['joint']) in crossing_names],
                concurrent_evidence=audit.get('concurrent_evidence', []),
                first_observed_position_crossing=first_crossings.get(key)))
    return sorted(terminals, key=lambda row: (row['physics_step'], row['env_id']))


def legacy_teacher128_rows(rows, scene, rollout):
    """Join preserved historical evidence, retaining the scene exemplar scope."""
    import torch

    from .linear_coupling import ACTIVE
    from .tabletop_stereo_contract import require_sha

    if (rollout['layout'] != 'step_major_then_environment'
            or scene['contract']['joint_limit_reset']['early_active_joint_margin'] is not None):
        raise ValueError('legacy audit requires a step-major rollout with only the hard position guard')
    hard = scene['contract']['joint_limit_reset']['joint_margin']
    batch, count, steps = rollout['batch'], rollout['num_envs'], rollout['rollout_steps']
    fields = ('old_mu', 'old_std', 'latent_actions', 'actions')
    if any(batch[name].shape != (count*steps, 24) for name in fields):
        raise ValueError('legacy diagnostic requires complete 24D rollout tensors')
    if (any(not bool(torch.isfinite(batch[name]).all()) for name in fields)
            or not bool((batch['old_std'] > 0).all())
            or not torch.allclose(batch['latent_actions'].tanh(), batch['actions'])):
        raise ValueError('legacy diagnostic policy distribution is malformed')
    checkpoint = require_sha(rollout['initial_checkpoint_sha256'], 'collection checkpoint SHA')
    hands = scene['physical_parameter_readback']['hands']
    if len(hands) != 2 or checkpoint == '0'*64:
        raise ValueError('legacy diagnostic requires bilateral SDK readback and checkpoint identity')
    bounds = []
    for hand in hands:
        names = hand['joint_names']
        for name in ACTIVE:
            index = names.index(name)
            lower, upper = hand['joint_limits_rad'][0][index]
            bounds.append((lower, upper, hand['velocity_limits_rad_s'][0][index]))
    for row in rows:
        if row.get('kind') != 'sample':
            yield row
            continue
        env_id, tick = row['env_id'], row['policy_tick']
        if (not 0 <= env_id < count or not 0 <= tick < steps
                or row['checkpoint_sha256'] != checkpoint):
            raise ValueError('legacy diagnostic checkpoint or env/tick identity mismatch')
        index = tick*count+env_id
        action = batch['actions'][index]
        if any(row[key] != action.tolist() for key in ('actually_issued_action', 'teacher_sampled_action')):
            raise ValueError('legacy native issued/sample action differs from the policy ledger')
        if row['terminal'] and not bool(batch['done'][index]):
            raise ValueError('legacy terminal differs from the policy ledger')
        if not torch.allclose(batch['old_mu'][index].tanh(),
                action.new_tensor(row['teacher_mean_action'])):
            raise ValueError('legacy native mean action differs from the policy ledger')
        position, velocity = row['active_joint_position'], row['active_joint_velocity']
        sdk = row['applied_targets']['actual_sdk_joint_target_rad']
        command = row['applied_targets']['controller_joint_command_rad']
        if any(len(value) != 12 for value in (position, velocity, sdk, command)):
            raise ValueError('legacy audit requires complete bilateral active joint vectors')
        joints = []
        for dim, (lower, upper, max_vel) in enumerate(bounds):
            side = 'left' if dim < 6 else 'right'
            joints.append(joint_evidence(hand=side, name=ACTIVE[dim % 6],
                position=position[dim], velocity=velocity[dim], lower=lower, upper=upper,
                hard_tolerance=hard, training_tolerance=None, velocity_limit=max_vel,
                sdk_target=sdk[dim], controller_target=command[dim],
                action_group=f'{side}_fingers', action_index=dim, unit='rad'))
        projection = row.get('palm_safety_projection')
        projected = None if projection is None else any(
            abs(a-b)>1e-9 for before, after in zip(projection['pre_projection_palm_target_m'],
                projection['projected_palm_target_m']) for a, b in zip(before, after))
        yield dict(row, action_audit=dict(
            sample_source='historical_active_joints_compared_to_same_scene_sdk_exemplar',
            joints=joints, actor_mean=row['teacher_mean_action'],
            latent_mean=batch['old_mu'][index].tolist(), latent_std=batch['old_std'][index].tolist(),
            sampled_latent=batch['latent_actions'][index].tolist(),
            sampled_action=action.tolist(), issued_action=row['actually_issued_action'],
            **classify_joint_evidence(joints, failure_code=row['failure_code'], projection=projected)))


def main(argv=None):
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--scene-summary', type=Path)
    parser.add_argument('--rollout', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if bool(args.scene_summary) != bool(args.rollout):
        parser.error('legacy --scene-summary and --rollout must be provided together')
    raw = args.raw.read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError('raw evidence lacks a complete final JSONL record')
    rows = [json.loads(line) for line in raw.splitlines()]
    source = 'recorded_live_joint_audit'
    evidence = {'native_physics_sha256': hashlib.sha256(raw).hexdigest()}
    if args.rollout:
        import torch

        scene = args.scene_summary.read_bytes()
        rollout_bytes = args.rollout.read_bytes()
        from io import BytesIO
        rows = list(legacy_teacher128_rows(rows, json.loads(scene),
            torch.load(BytesIO(rollout_bytes), map_location='cpu', weights_only=True)))
        source = 'historical_active_joints_compared_to_same_scene_sdk_exemplar'
        evidence.update(scene_summary_sha256=hashlib.sha256(scene).hexdigest(),
            rollout_sha256=hashlib.sha256(rollout_bytes).hexdigest())
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(dict(source=source, evidence=evidence,
            samples=sum(r.get('kind') == 'sample' for r in rows),
            joint_limit_terminals=summarize_joint_terminals(rows)), stream,
            indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
