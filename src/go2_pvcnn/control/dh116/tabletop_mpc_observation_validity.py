"""Audit actual 371D observation values against synchronized native sidecars."""

import math

import torch

from .tabletop_mpc_privileged_capture import ACTION_CONTRACT, ACTION_OWNER, VERSION
from .tabletop_dexhands_feedback import geometry_features


PROOF_VERSION = 'dh116-mpc-native-observation-value-proof-v1'
PROOF_SCOPE = 'simulator_observation_value_and_source_parity_only'
GROUPS = {
    (0, 12): 'sdk_active_position',
    (12, 24): 'sdk_active_velocity',
    (24, 36): 'prior_joint_command',
    (55, 56): 'task_offset_cache',
    (60, 61): 'measured_hand_table_clearance',
    (61, 64): 'sdk_relative_object_center',
    (64, 68): 'sdk_object_root_quaternion_wxyz',
    (68, 74): 'sdk_object_root_velocity',
    (78, 88): 'measured_passive_mimic_residual',
    (100, 106): 'sdk_palm_offset_from_submitted_target',
    (106, 108): 'drive_stiffness_command_cache',
    (108, 109): 'geometry_cylinder_bottom',
    (109, 121): 'sdk_fk_legacy_pad_points_relative_to_object',
    (153, 167): 'sdk_measured_wrist_pose_and_reset_root_seed',
    (167, 179): 'sdk_measured_wrist_velocity',
    (179, 185): 'sdk_measured_wrist_mount_angles',
    (191, 197): 'submitted_wrist_angles',
    (197, 203): 'submitted_palm_offset',
    (203, 215): 'joint_command_state',
    (215, 239): 'issued_prior_action_or_original_reset_zero',
    (239, 251): 'sdk_fk_legacy_pad_cylinder_geometry',
    (321, 351): 'sdk_fk_all_digit_reference_points_relative_to_object',
}


def _validate_sources(archive, sidecar):
    if (archive['version'] != VERSION or archive['action_owner'] != ACTION_OWNER
            or archive['action_contract'] != ACTION_CONTRACT
            or archive['counts_as_Student_success'] is not False
            or archive['qualified_for_bc'] is not False
            or sidecar['version'] != 'dh116-native-sequence-capture-unqualified-v1'
            or sidecar['source'] != 'actual_precontrol_and_native_submission_snapshots'
            or sidecar['controller_contract']['action_contract'] != ACTION_CONTRACT
            or sidecar['error'] is not None or sidecar['incomplete_policy_context'] is not None):
        raise ValueError('completed original native MPC sidecar and unqualified archive required')
    observations = archive['observations']
    actions = archive['actions']
    rows = archive['sample_metadata']
    if (not rows or observations.shape != (len(rows), 371) or observations.dtype != torch.float32
            or actions.shape != (len(rows), 24) or not actions.is_floating_point()
            or not torch.isfinite(observations).all() or not torch.isfinite(actions).all()
            or not (actions.abs() <= 1).all()
            or archive['privileged_valid'].shape != observations.shape
            or archive['privileged_valid'].dtype != torch.bool):
        raise ValueError('canonical sidecar rows and finite 371D/24D tensors required')


def _bind(record, row, action, run_uuid):
    sample = record['samples']
    tick, i = row['tick'], row['env_id']
    if not 0 <= i < len(sample['env_id']):
        raise ValueError('sidecar environment index outside capture')
    post = record['post_physics_feedback'][0]
    context = record['physics_context']
    now = row['simulation_time_s']
    step = row['physics_step']
    clock = row['policy_clock_before']
    if (row['run_uuid'] != run_uuid or sample['run_uuid'][i] != run_uuid
            or sample['env_id'][i] != i or sample['episode_uid'][i] != row['camera_episode_uid']
            or sample['tick'][i] != tick or sample['first_physics_step'][i] != step
            or sample['policy_timestamp_s'][i] != now or clock['sdk_step'] != step
            or clock['sdk_time_s'] != now or clock['world_step'] != tick * 8
            or not torch.equal(sample['actions'][i], action)
            or bool(sample['terminal'][i]) != row['terminal_in_interval']
            or not sample['valid'][i] or not context['source_valid'][i, 0] or not context['live'][i, 0]
            or post['source'] != 'original_advance_returned_refresh_before_TensorEnv_terminal_reset'
            or post['native_runtime_verified'] is not True or post['clock_aligned'] is not True
            or not post['source_valid'][i] or post['sdk_physics_step'] != step + 1
            or post['backend_readback_sdk_physics_step'] != post['sdk_physics_step']
            or post['backend_readback_sdk_timestamp_s'] != post['sdk_timestamp_s']
            or not math.isclose(post['sdk_timestamp_s'] - now, 1 / 480, abs_tol=1e-7, rel_tol=0)):
        raise ValueError('sidecar native identity, clock, feedback, terminal or issued action differs')
    return sample


def _expected_groups(record, previous):
    context = record['physics_context']
    command = record['controller_start']
    m = record['raw_measured_packets'][0]
    low, high = context['joint_lower'][:, 0], context['joint_upper'][:, 0]
    if not (high > low).all():
        raise ValueError('sidecar native joint normalization bounds are invalid')

    def joints(value):
        return 2 * (value - low) / (high - low) - 1

    pose = m['wrist_pose'].clone()
    pose[..., :3] -= context['root_seed_pose'][:, 0, :, :3]
    pose[..., :3] /= .05
    center = m['object_relative'] + m['wrist_pose'][..., :3].mean(1)
    references = (m['pad_reference_points_w'] - center[:, None, None]) / .05
    return {
        (0, 12): joints(m['q']),
        (12, 24): m['qd'] / .3,
        (24, 36): joints(command['joint_command_rad']),
        (55, 56): context['task_offset_m'][:, :1] / .05,
        (60, 61): m['clearance'][:, None] / .05,
        (61, 64): m['object_relative'] / .05,
        (64, 68): m['object_quat'],
        (68, 74): m['object_velocity'] / m['q'].new_tensor([.1, .1, .1, 1, 1, 1]),
        (78, 88): m['mimic_errors'] / .02,
        (100, 106): (m['adaptive_palm_offset'] / m['q'].new_tensor([.003, .003, .004])).flatten(1),
        (106, 108): (m['grip_stiffness'] - 10.) / 10.,
        (108, 109): (m['bottom'][:, None] - .30) / .03,
        (109, 121): m['pad_relative'].flatten(1) / .05,
        (153, 167): pose.flatten(1),
        (167, 179): (m['wrist_velocity'] / m['q'].new_tensor([.05, .05, .05, 1, 1, 1])).flatten(1),
        (179, 185): m['wrist_joint_angles'].flatten(1) / (math.pi / 2),
        (191, 197): command['wrist_command_angle_rad'].flatten(1) / (math.pi / 2),
        (197, 203): command['palm_command_offset_m'].flatten(1) / .05,
        (203, 215): joints(command['joint_command_rad']),
        (215, 239): previous,
        (239, 251): geometry_features(m)[:, :12],
        (321, 351): references.flatten(1),
    }


def audit_observation_mask(archive, sidecar):
    """Prove only the exact, same-episode geometry and command groups listed above."""
    _validate_sources(archive, sidecar)
    rows = archive['sample_metadata']
    actions = archive['actions']
    records = sidecar['policy_records']
    observations = archive['observations']
    mask = torch.zeros_like(observations, dtype=torch.bool)
    previous_by_env = {}
    for index, (row, action) in enumerate(zip(rows, actions)):
        tick, env = row['tick'], row['env_id']
        if type(tick) is not int or not 0 <= tick < len(records) or type(env) is not int:
            raise ValueError('sidecar policy tick or environment is invalid')
        record = records[tick]
        sample = _bind(record, row, action, archive['run_uuid'])
        previous = action.new_zeros(24)
        if tick == 0:
            if env in previous_by_env or row['reset_generation'] != 0:
                raise ValueError('invalid first-episode prior action reset')
        else:
            prior = previous_by_env.get(env)
            if (prior is None or prior[0]['tick'] != tick - 1
                    or prior[0]['episode_uid'] != row['episode_uid']
                    or prior[0]['reset_generation'] != row['reset_generation']
                    or prior[0]['terminal_in_interval']):
                raise ValueError('missing or mismatched same-episode prior issued action')
            previous = prior[1]
        if not torch.equal(torch.as_tensor(sample['previous_action'][env]).to(previous), previous):
            raise ValueError('sidecar prior action does not match the recorded issued action')
        previous_by_env[env] = row, action
        obs = observations[index]
        expected_groups = _expected_groups(record, previous)
        for (start, stop), values in expected_groups.items():
            candidate = values if start == 215 else values[env]
            candidate = candidate.to(obs)
            if candidate.shape != obs[start:stop].shape:
                raise ValueError('sidecar field shape disagrees with the canonical 371D block')
            mask[index, start:stop] = torch.isfinite(candidate) & candidate.eq(obs[start:stop])
    return mask
