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
                   action_group, action_index, unit, joint_role=None):
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
    guard_side = (None if guard_margin is None or guard_margin >= 0 else
                  'lower' if position < lower else 'upper')
    target_margin = _margin(sdk_target, lower, upper)
    tracking_departure = (None if None in (guard_margin, target_margin) else
                          guard_margin < 0 and target_margin >= -1e-6)
    return dict(hand=hand, joint=name, position_unit=unit, velocity_unit=f'{unit}/s',
        joint_role=joint_role,
        position=position, velocity=velocity, lower_limit=lower, upper_limit=upper,
        hard_guard_tolerance=hard_tolerance, training_guard_tolerance=training_tolerance,
        nominal_margin=nominal_margin, guard_margin=guard_margin,
        training_guard_margin=training_margin, velocity_limit=velocity_limit,
        velocity_margin=velocity_margin, sdk_target=sdk_target, controller_target=controller_target,
        target_margin=target_margin,
        controller_target_margin=_margin(controller_target, lower, upper),
        position_guard_side=guard_side, legal_target_tracking_departure=tracking_departure,
        actual_minus_sdk_target=None if None in (position, sdk_target) else position-sdk_target,
        sdk_minus_controller_target=None if None in (sdk_target, controller_target) else sdk_target-controller_target,
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
    active_crossings = [j for j in positions if j['action_index'] is not None and j['action_index'] < 12]
    passive_names = {(j['hand'], j['joint']) for j in joints if j.get('joint_role') == 'passive_finger'}
    passive_crossings = [j for j in positions if (j['hand'], j['joint']) in passive_names]
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
        active_finger_position_crossings=active_crossings, passive_finger_position_crossings=passive_crossings,
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
        from .linear_coupling import MIMIC
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
        passive_names = {name for name, _, _, _ in MIMIC}
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
                    action_index=action_index, unit='m' if linear else 'rad',
                    joint_role='active_finger' if finger_action is not None else
                        'passive_finger' if name in passive_names else 'wrist' if linear or angular else None)
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
    first_by_joint, last_by_joint = {}, {}
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
        joints = {(j['hand'], j['joint']): j for j in audit.get('joints', [])}
        if audit.get('position_crossings'):
            first_crossings.setdefault(key, dict(identity, joints=audit['position_crossings']))
            for crossing in audit['position_crossings']:
                name = (crossing['hand'], crossing['joint'])
                first_by_joint.setdefault((key, name), dict(identity, crossing=crossing, joint=joints.get(name)))
        if row['terminal'] and row['failure_code'] == FAILURE_CODES['joint_limit']:
            crossing_names = {(j['hand'], j['joint']) for j in audit.get('position_crossings', [])}
            terminals.append(dict(identity,
                violation_source=audit.get('violation_source') or 'unknown_joint_limit',
                terminal_position_crossings=audit.get('position_crossings', []),
                crossing_joint_values=[j for j in audit.get('joints', [])
                    if (j['hand'], j['joint']) in crossing_names],
                concurrent_evidence=audit.get('concurrent_evidence', []),
                terminal_joint_history=[dict(hand=hand, joint=name,
                    first_guard_crossing=first_by_joint[(key, (hand, name))],
                    previous_measurement=last_by_joint.get((key, (hand, name))),
                    terminal_measurement=dict(identity, joint=joints.get((hand, name))))
                    for hand, name in sorted(crossing_names)],
                first_observed_position_crossing=first_crossings.get(key)))
        for name, joint in joints.items():
            last_by_joint[(key, name)] = dict(identity, joint=joint)
    return sorted(terminals, key=lambda row: (row['physics_step'], row['env_id']))


def finger12_event_windows(rows):
    """Report first observed right-finger lower-limit events, with same-episode samples."""
    stages = ('near_nominal', 'inside_hard_tolerance', 'hard_guard_crossing')
    radius, proximity_rad = 4, .005
    episodes, previous = {}, {}
    for row in rows:
        if row.get('kind') != 'sample':
            continue
        key = (row['run_uuid'], row['env_id'], row['episode_uid'])
        step, timestamp = row['physics_step'], row['timestamp']
        prior = previous.get(key)
        if (type(step) is not int or step < 0 or _finite(timestamp) is None
                or (prior is not None and (prior['terminal'] or step <= prior['physics_step']
                                           or timestamp <= prior['timestamp']))):
            raise ValueError('invalid physical chronology in finger12 event windows')
        previous[key] = row
        audit = row.get('action_audit') or {}
        selected = [joint for joint in audit.get('joints', [])
                    if (joint['hand'], joint['joint']) == ('right', 'finger12')]
        if not selected:
            continue
        if len(selected) != 1 or selected[0]['action_index'] != 7 or selected[0]['position_unit'] != 'rad':
            raise ValueError('right finger12 must match canonical radian action index 7')
        joint, = selected
        episodes.setdefault(key, {})[step] = dict(physics_step=step, timestamp=timestamp,
            policy_tick=row['policy_tick'], terminal=row['terminal'], failure_code=row['failure_code'],
            issued_action=audit.get('issued_action', row.get('actually_issued_action')), joint=joint)

    output = []
    for (run, env, episode), samples in episodes.items():
        events = dict.fromkeys(stages)
        for step, sample in samples.items():
            joint = sample['joint']
            position, lower, tolerance = map(_finite,
                (joint['position'], joint['lower_limit'], joint['hard_guard_tolerance']))
            if None in (position, lower, tolerance) or tolerance < 0:
                continue
            margin = position - lower
            if 0 <= margin <= proximity_rad:
                stage = 'near_nominal'
            elif -tolerance <= margin < 0:
                stage = 'inside_hard_tolerance'
            elif margin < -tolerance:
                stage = 'hard_guard_crossing'
            else:
                continue
            if events[stage] is not None:
                continue
            before, after = range(max(0, step-radius), step), range(step+1, step+radius+1)
            events[stage] = dict(physics_step=step, lower_margin_rad=margin,
                samples=[samples[index] for index in range(before.start, after.stop) if index in samples],
                missing_before_physics_steps=[index for index in before if index not in samples],
                missing_after_physics_steps=[index for index in after if index not in samples])
        output.append(dict(run_uuid=run, env_id=env, episode_uid=episode, action_index=7,
            window_radius_physics_steps=radius, nominal_proximity_rad=proximity_rad, events=events))
    return output


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
                action_group=f'{side}_fingers', action_index=dim, unit='rad', joint_role='active_finger'))
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


def _paired_policy_rows(rows, rollout):
    """Index every archived 371D policy row with its recorded pre-action identity."""
    import torch

    from .tabletop_stereo_contract import require_sha, validate_identity

    batch = rollout['batch']
    n, steps = rollout['num_envs'], rollout['rollout_steps']
    if rollout['layout'] != 'step_major_then_environment' or n < 1 or steps < 1:
        raise ValueError('step-major diagnostic policy ledger required')
    expected = dict(actor_obs=(n*steps, 371), old_mu=(n*steps, 24), old_std=(n*steps, 24),
                    latent_actions=(n*steps, 24), actions=(n*steps, 24))
    for key, shape in expected.items():
        value = batch.get(key)
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError(f'policy ledger requires finite {key} with shape {shape}')
    if (not bool((batch['old_std'] > 0).all())
            or not torch.allclose(batch['latent_actions'].tanh(), batch['actions'], atol=2e-6, rtol=1e-5)):
        raise ValueError('archived sample does not agree with its latent action')
    if batch['done'].shape != (n*steps,) or batch['done'].dtype != torch.bool:
        raise ValueError('complete archived terminal flags required')
    checkpoint = require_sha(rollout['initial_checkpoint_sha256'], 'Teacher initial checkpoint SHA')
    if checkpoint == '0'*64:
        raise ValueError('nonzero Teacher initial checkpoint identity required')
    by_index, terminals = {}, {}
    shared = ('run_uuid', 'env_id', 'episode_uid', 'reset_id', 'policy_tick', 'frame_id',
              'calibration_sha256', 'task_identity_sha256', 'checkpoint_sha256', 'policy_version')
    for row in rows:
        if row.get('kind') != 'sample':
            continue
        pre = validate_identity(row.get('pre_action_identity'))
        env_id, tick = pre['env_id'], pre['policy_tick']
        if (any(k not in pre or pre[k] != row.get(k) for k in shared)
                or pre['physics_tick_start'] != pre['physics_tick_end']
                or pre['physics_tick_end'] >= row['physics_step']
                or pre['simulation_timestamp'] >= row['timestamp']
                or not 0 <= env_id < n or not 0 <= tick < steps
                or pre['checkpoint_sha256'] != checkpoint
                or pre['policy_version'] != rollout['policy_version']):
            raise ValueError('policy/substep pre-action identity differs')
        if any(require_sha(pre.get(k), k) == '0'*64 for k in
               ('calibration_sha256', 'task_identity_sha256', 'physical_initial_condition_sha256')):
            raise ValueError('nonzero pre-action identity hashes required')
        index = tick*n+env_id
        if by_index.setdefault(index, pre) != pre:
            raise ValueError('substeps disagree on their common pre-action identity')
        action = batch['actions'][index]
        if any(row[key] != action.tolist() for key in ('actually_issued_action', 'teacher_sampled_action')):
            raise ValueError('recorded issued/sample action differs from the policy ledger')
        if not torch.allclose(action.new_tensor(row['teacher_mean_action']),
                              batch['old_mu'][index].tanh(), atol=2e-6, rtol=1e-5):
            raise ValueError('recorded deterministic mean differs from the policy ledger')
        if row['terminal']:
            if index in terminals:
                raise ValueError('multiple terminal substeps for one policy state')
            terminals[index] = row
    if set(by_index) != set(range(n*steps)):
        raise ValueError('every archived policy state requires an observed 480Hz substep')
    if set(terminals) != set(batch['done'].nonzero().flatten().tolist()):
        raise ValueError('archive terminal flags disagree with the physics ledger')
    return by_index, terminals


def _delta_metrics(first, second):
    delta = (first - second).abs().double()
    return dict(sample_count=len(delta), mae=float(delta.mean()), p95=float(delta.flatten().quantile(.95)),
                max=float(delta.max()), right_finger12_mae=float(delta[:, 7].mean()))


def compare_teacher_archive(rows, rollout, teacher):
    """Recompute deterministic output on the exact state used to sample each archived action."""
    import torch

    from .tabletop_tensor_task import FAILURE_CODES

    identities, terminals = _paired_policy_rows(rows, rollout)
    batch = rollout['batch']
    with torch.no_grad():
        latent_mean = teacher.latent_mean(batch['actor_obs'])
        std = teacher.log_std.exp().expand_as(latent_mean)
    if not torch.allclose(latent_mean, batch['old_mu'], atol=2e-6, rtol=1e-5):
        raise ValueError('checkpoint latent mean disagrees with the archived 371D states')
    if not torch.allclose(std, batch['old_std'], atol=1e-7, rtol=1e-6):
        raise ValueError('checkpoint std disagrees with the archived distribution')
    mean, actions = latent_mean.tanh(), batch['actions']
    limit_indices = [i for i, row in terminals.items() if row['failure_code'] == FAILURE_CODES['joint_limit']]
    details = []
    for index in sorted(limit_indices):
        row = terminals[index]
        right = 7
        details.append(dict(identity=identities[index], physics_step=row['physics_step'],
            sampled_action24=actions[index].tolist(), deterministic_mean24=mean[index].tolist(),
            issued_action24=row['actually_issued_action'], latent_std24=std[index].tolist(),
            normalized_applied_action24=row.get('applied_action'),
            applied_action_source=row.get('applied_action_source'),
            right_finger12=dict(action_index=right, mean_action=float(mean[index, right]),
                sample_action=float(actions[index, right]), latent_std=float(std[index, right]),
                sample_latent_z_score=float((batch['latent_actions'][index, right]-latent_mean[index, right])
                                           /std[index, right])),
            counterfactual_physics_outcome=None))
    return dict(evidence_scope='paired_actions_only_no_counterfactual_physics',
        paired_policy_states=len(identities), joint_limit_policy_states=len(limit_indices), terminals=details,
        physical_initial_groups=len({v['physical_initial_condition_sha256'] for v in identities.values()}),
        mean_replay_max_abs_error=float((latent_mean-batch['old_mu']).abs().max()),
        std_replay_max_abs_error=float((std-batch['old_std']).abs().max()),
        sampled_vs_mean_all=_delta_metrics(actions, mean),
        sampled_vs_mean_joint_limit=_delta_metrics(actions[limit_indices], mean[limit_indices]) if limit_indices else None)


def compare_mpc_archive(payload, rows, rollout):
    """Keep legacy visual trajectories separate; compare only explicitly joined 371D states."""
    import torch

    from .tabletop_privileged_checkpoint import canonical_sha256
    from .tabletop_privileged_dataset import audit_expert_compatibility
    from .tabletop_stereo_contract import digest, state_contract
    from .tabletop_surface_contract import ACTION_CONTRACT

    identity = dict(action_contract_sha256=canonical_sha256(ACTION_CONTRACT),
                    state_contract_sha256=digest(state_contract()))
    report = audit_expert_compatibility(payload, expected_identity=identity)

    def bounded(value):
        return (isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[-1] == 24 and len(value) > 0
                and value.is_floating_point() and bool(torch.isfinite(value).all()) and bool((value.abs() <= 1).all()))

    actions, reference = (payload.get(k) for k in ('actions', 'reference_actions'))
    report.update(paired_policy_states=0, same_state_action_metrics=None, counterfactual_physics_outcome=None,
                  own_trajectory_mpc_vs_reference=None, compared_policy_scope='no_exact_state_pairing')
    if report['action_contract_matches'] and bounded(actions) and bounded(reference) and actions.shape == reference.shape:
        report['own_trajectory_mpc_vs_reference'] = _delta_metrics(actions, reference)
    if (payload.get('version') != 'dh116-mpc-pre-action-comparison-v1'
            or not report['action_contract_matches'] or not report['state_contract_matches'] or not bounded(actions)):
        return report
    observations, validity, candidate_ids = (payload.get(k) for k in (
        'observations', 'privileged_valid', 'pre_action_identities'))
    shape = (len(actions), 371)
    if (not isinstance(observations, torch.Tensor) or observations.shape != shape
            or not observations.is_floating_point() or not bool(torch.isfinite(observations).all())
            or not isinstance(validity, torch.Tensor) or validity.shape != shape or validity.dtype != torch.bool
            or not isinstance(candidate_ids, list) or len(candidate_ids) != len(actions)):
        raise ValueError('MPC pairing requires complete explicit 371D state, validity and pre-action identities')
    ids, _ = _paired_policy_rows(rows, rollout)
    expected_indices = {digest(identity): index for index, identity in ids.items()}
    pairs, seen = [], set()
    batch = rollout['batch']
    for i, candidate in enumerate(candidate_ids):
        identity_key = digest(candidate)
        if identity_key in seen:
            raise ValueError('duplicate MPC pre-action identity')
        seen.add(identity_key)
        idx = expected_indices.get(identity_key)
        if idx is not None and bool(validity[i].all()) and torch.equal(observations[i], batch['actor_obs'][idx]):
            pairs.append((i, idx))
    if pairs:
        mpc_indices, teacher_indices = zip(*pairs)
        reference = actions[list(mpc_indices)]
        report.update(paired_policy_states=len(pairs), compared_policy_scope='paired_actions_only_no_counterfactual_physics',
            same_state_action_metrics=dict(mpc_vs_teacher_mean=_delta_metrics(reference, batch['old_mu'][list(teacher_indices)].tanh()),
                                           mpc_vs_teacher_sample=_delta_metrics(reference, batch['actions'][list(teacher_indices)])))
    return report


def _read_verified_teacher(path, rollout):
    """Load weights on CPU without restoring optimizer, RNG or environment state."""
    import hashlib
    from io import BytesIO
    from pathlib import Path

    import numpy as np
    import torch
    from numpy.core.multiarray import _reconstruct

    from .tabletop_privileged_checkpoint import VERSION as CHECKPOINT_VERSION, canonical_sha256
    from .tabletop_privileged_teacher import VERSION as POLICY_VERSION, PrivilegedTeacher
    from .tabletop_stereo_contract import digest, file_sha256, state_contract
    from .tabletop_surface_contract import ACTION_CONTRACT

    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != rollout['initial_checkpoint_sha256']:
        raise ValueError('checkpoint bytes disagree with the archived initial checkpoint SHA')
    # Historic checkpoints include NumPy RNG metadata, which is parsed but never restored.
    with torch.serialization.safe_globals([(_reconstruct, 'numpy.core.multiarray._reconstruct'),
                                          np.ndarray, np.dtype, type(np.dtype('uint32'))]):
        payload = torch.load(BytesIO(raw), map_location='cpu', weights_only=True)
    identity = payload['identity']
    source = 'src/go2_pvcnn/control/dh116/tabletop_privileged_teacher.py'
    current_teacher_source = Path(__file__).with_name('tabletop_privileged_teacher.py')
    if (payload['version'] != CHECKPOINT_VERSION or payload['policy_version'] != rollout['policy_version']
            or identity['policy_version'] != POLICY_VERSION or identity['observation_size'] != 371
            or identity['action_size'] != 24 or identity['distribution'] != 'tanh_latent_gaussian'
            or identity['state_contract_sha256'] != digest(state_contract())
            or identity['action_contract_sha256'] != canonical_sha256(ACTION_CONTRACT)
            or identity['source_sha256'].get(source) != file_sha256(current_teacher_source)):
        raise ValueError('Teacher checkpoint policy source or action/state contract differs')
    with torch.random.fork_rng(devices=[]):
        teacher = PrivilegedTeacher()
    teacher.load_state_dict(payload['policy'], strict=True)
    return teacher.eval()

def main(argv=None):
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--scene-summary', type=Path)
    parser.add_argument('--rollout', type=Path)
    parser.add_argument('--teacher-checkpoint', type=Path, help='exact collection checkpoint for CPU mean replay')
    parser.add_argument('--mpc', type=Path, action='append', default=[], help='read-only MPC action artifact audit')
    parser.add_argument('--finger12-event-windows', action='store_true',
                        help='include observed right finger12 lower-limit event windows')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if bool(args.scene_summary) != bool(args.rollout):
        parser.error('legacy --scene-summary and --rollout must be provided together')
    if (args.teacher_checkpoint or args.mpc) and not args.rollout:
        parser.error('action comparison requires the archived --rollout and --scene-summary')
    raw = args.raw.read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError('raw evidence lacks a complete final JSONL record')
    rows = [json.loads(line) for line in raw.splitlines()]
    source = 'recorded_live_joint_audit'
    evidence = {'native_physics_sha256': hashlib.sha256(raw).hexdigest()}
    comparisons = {}
    if args.rollout:
        import torch

        scene = args.scene_summary.read_bytes()
        rollout_bytes = args.rollout.read_bytes()
        from io import BytesIO
        rollout = torch.load(BytesIO(rollout_bytes), map_location='cpu', weights_only=True)
        rows = list(legacy_teacher128_rows(rows, json.loads(scene), rollout))
        source = 'historical_active_joints_compared_to_same_scene_sdk_exemplar'
        evidence.update(scene_summary_sha256=hashlib.sha256(scene).hexdigest(),
            rollout_sha256=hashlib.sha256(rollout_bytes).hexdigest())
        if args.teacher_checkpoint:
            teacher = _read_verified_teacher(args.teacher_checkpoint, rollout)
            comparisons['teacher'] = compare_teacher_archive(rows, rollout, teacher)
            evidence['teacher_checkpoint_sha256'] = rollout['initial_checkpoint_sha256']
        if args.mpc:
            comparisons['mpc'] = []
            for path in args.mpc:
                content = path.read_bytes()
                payload = torch.load(BytesIO(content), map_location='cpu', weights_only=True)
                comparisons['mpc'].append(dict(file=path.name, sha256=hashlib.sha256(content).hexdigest(),
                    **compare_mpc_archive(payload, rows, rollout)))
    result = dict(source=source, evidence=evidence,
        samples=sum(r.get('kind') == 'sample' for r in rows),
        joint_limit_terminals=summarize_joint_terminals(rows), action_comparisons=comparisons)
    if args.finger12_event_windows:
        result['finger12_position_event_windows'] = finger12_event_windows(rows)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
