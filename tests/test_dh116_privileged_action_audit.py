import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_privileged_action_audit import (
    classify_joint_evidence, joint_evidence, legacy_teacher128_rows, summarize_joint_terminals,
)


def finger(position, *, velocity=0., target=.01, hand='right'):
    return joint_evidence(hand=hand, name='finger12', position=position,
        velocity=velocity, lower=.007912623, upper=1.03, hard_tolerance=.02,
        training_tolerance=None, velocity_limit=.3, sdk_target=target,
        controller_target=target, action_group=f'{hand}_fingers',
        action_index=7 if hand == 'right' else 1, unit='rad')


def test_position_guard_and_sdk_velocity_limit_are_separate():
    sample = finger(-.012467, velocity=-.6)
    assert sample['guard_margin'] == pytest.approx(-.000379623)
    assert sample['nominal_margin'] == pytest.approx(-.020379623)
    assert sample['target_margin'] > 0.
    audit = classify_joint_evidence([sample], failure_code=3)
    assert audit['violation_source'] == 'joint_position_limit'
    assert audit['concurrent_evidence'] == ['joint_velocity_limit']
    assert audit['position_crossings'][0]['action_index'] == 7
    assert audit['position_crossings'][0]['hand'] == 'right'


def test_passive_guard_crossing_alone_does_not_qualify_contact_cause():
    passive = dict(finger(-.014), joint='finger13', joint_role='passive_finger', action_index=None)
    audit = classify_joint_evidence([passive], failure_code=3)
    assert [value['joint'] for value in audit['passive_finger_position_crossings']] == ['finger13']
    assert audit['active_finger_position_crossings'] == []
    assert audit['passive_contact_limit'] is None


def test_projection_and_target_events_cannot_substitute_for_unobserved_terminal_cause():
    audit = classify_joint_evidence([finger(.008, velocity=-.6, target=.002)],
        failure_code=3, projection=True)
    assert audit['violation_source'] == 'unknown_joint_limit'
    assert set(audit['concurrent_evidence']) == {
        'joint_velocity_limit', 'action_target_out_of_range', 'controller_projection'}


def test_first_observed_crossing_is_episode_local_and_keeps_terminal_tick():
    def row(episode, step, pos, terminal=False):
        return dict(kind='sample', run_uuid='fixture', env_id=0, episode_uid=episode,
            policy_tick=step // 8, physics_step=step, timestamp=step / 480,
            failure_code=3 if terminal else 0, terminal=terminal,
            action_audit=classify_joint_evidence([finger(pos)], failure_code=3 if terminal else 0))

    rows = [row('a', 1, .01), row('a', 2, -.014), row('a', 3, -.014, True),
        row('b', 8, .01), row('b', 9, -.014, True)]
    summary = summarize_joint_terminals(rows)
    assert [v['physics_step'] for v in summary] == [3, 9]
    assert [v['first_observed_position_crossing']['physics_step'] for v in summary] == [2, 9]
    with pytest.raises(ValueError, match='after terminal'):
        summarize_joint_terminals(rows + [row('b', 10, .01)])


def test_legacy_join_uses_checkpoint_and_issued_action_identity():
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE

    latent = torch.full((1, 24), -.2)
    action = latent.tanh()
    readback = dict(joint_names=list(reversed(ACTIVE)),
        joint_limits_rad=[[[.007912623, 1.03]]*6], velocity_limits_rad_s=[[.3]*6])
    scene = dict(physical_parameter_readback={'hands': [readback, readback]},
        contract={'joint_limit_reset': {'joint_margin': .02, 'early_active_joint_margin': None}})
    rollout = dict(initial_checkpoint_sha256='a'*64, num_envs=1, rollout_steps=1,
        layout='step_major_then_environment',
        batch=dict(old_mu=latent, old_std=torch.ones_like(latent),
            latent_actions=latent, actions=action, done=torch.tensor([True])))
    row = dict(kind='sample', env_id=0, policy_tick=0, checkpoint_sha256='a'*64,
        terminal=True, failure_code=3, actually_issued_action=action[0].tolist(),
        teacher_mean_action=action[0].tolist(), teacher_sampled_action=action[0].tolist(),
        active_joint_position=[.2]*7+[-.014]+[.2]*4, active_joint_velocity=[.1]*12,
        applied_targets=dict(actual_sdk_joint_target_rad=[.2]*12,
            controller_joint_command_rad=[.2]*12))
    actual, = legacy_teacher128_rows([row], scene, rollout)
    assert actual['action_audit']['position_crossings'][0]['joint'] == 'finger12'
    assert actual['action_audit']['sampled_latent'] == latent[0].tolist()
    with pytest.raises(ValueError, match='identity'):
        list(legacy_teacher128_rows([dict(row, checkpoint_sha256='b'*64)], scene, rollout))
    with pytest.raises(ValueError, match='issued'):
        list(legacy_teacher128_rows([dict(row, actually_issued_action=[0.]*24)], scene, rollout))


def test_terminal_retains_the_previous_measurement_of_each_crossing_joint():
    def row(step, right, left, terminal=False):
        return dict(kind='sample', run_uuid='fixture', env_id=0, episode_uid='first',
            policy_tick=0, physics_step=step, timestamp=step/480, failure_code=3 if terminal else 0,
            terminal=terminal, action_audit=dict(joints=[right, left], **classify_joint_evidence(
                [right, left], failure_code=3 if terminal else 0)))
    summary, = summarize_joint_terminals([
        row(1, finger(.01), finger(-.014, hand='left')),
        row(2, finger(-.014, velocity=-.5), finger(.01, hand='left'), True),
    ])
    right, = summary['terminal_joint_history']
    assert right['first_guard_crossing']['physics_step'] == 2
    assert right['previous_measurement']['joint']['position'] == .01
    assert right['terminal_measurement']['joint']['position_guard_side'] == 'lower'
    assert right['terminal_measurement']['joint']['actual_minus_sdk_target'] == pytest.approx(-.024)
    assert right['terminal_measurement']['joint']['legal_target_tracking_departure'] is True


def replay_fixture():
    from go2_pvcnn.control.dh116.tabletop_privileged_teacher import PrivilegedTeacher

    teacher = PrivilegedTeacher()
    observations = torch.zeros(2, 371)
    observations[1, 0] = 1.
    with torch.no_grad():
        mean = teacher.latent_mean(observations)
    std = teacher.log_std.detach().exp().expand_as(mean)
    latent = mean.clone()
    latent[0, 7] += 2 * std[0, 7]
    action = latent.tanh()
    batch = dict(actor_obs=observations, old_mu=mean, old_std=std,
                 actions=action, latent_actions=latent, done=torch.tensor([True, False]))
    rollout = dict(layout='step_major_then_environment', num_envs=2, rollout_steps=1,
                   policy_version=0, initial_checkpoint_sha256='a'*64, batch=batch)
    rows = []
    for i in range(2):
        identity = dict(run_uuid='test', env_id=i, episode_uid=f'test:{i}:1', reset_id=1,
            policy_tick=0, physics_tick_start=0, physics_tick_end=0, simulation_timestamp=0.,
            frame_id=0, calibration_sha256='b'*64, physical_initial_condition_sha256='c'*64,
            task_identity_sha256='d'*64, checkpoint_sha256='a'*64, policy_version=0)
        rows.append(dict(kind='sample', **{k: identity[k] for k in (
            'run_uuid','env_id','episode_uid','reset_id','policy_tick','frame_id','calibration_sha256',
            'task_identity_sha256','checkpoint_sha256','policy_version')},
            pre_action_identity=identity, physics_step=1, timestamp=1/480,
            teacher_mean_action=mean[i].tanh().tolist(), teacher_sampled_action=action[i].tolist(),
            actually_issued_action=action[i].tolist(), terminal=i == 0, failure_code=3 if i == 0 else 0))
    return teacher, rollout, rows


def test_replay_recomputes_same_371d_mean_without_resampling():
    from go2_pvcnn.control.dh116.tabletop_privileged_action_audit import compare_teacher_archive

    teacher, rollout, rows = replay_fixture()
    before = torch.get_rng_state().clone()
    comparison = compare_teacher_archive(rows, rollout, teacher)
    assert torch.equal(before, torch.get_rng_state())
    assert comparison['paired_policy_states'] == 2
    assert comparison['joint_limit_policy_states'] == 1
    assert comparison['terminals'][0]['right_finger12']['sample_latent_z_score'] == pytest.approx(2.)
    assert comparison['terminals'][0]['counterfactual_physics_outcome'] is None
    assert comparison['terminals'][0]['normalized_applied_action24'] is None
    assert comparison['mean_replay_max_abs_error'] < 1e-6


def test_replay_rejects_misaligned_identity_and_modified_state():
    from go2_pvcnn.control.dh116.tabletop_privileged_action_audit import compare_teacher_archive

    teacher, rollout, rows = replay_fixture()
    with pytest.raises(ValueError, match='identity'):
        compare_teacher_archive([dict(rows[0], reset_id=2), rows[1]], rollout, teacher)
    rollout['batch']['actor_obs'][0] += 1.
    with pytest.raises(ValueError, match='mean'):
        compare_teacher_archive(rows, rollout, teacher)


def test_replay_checks_checkpoint_bytes_before_deserializing(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_privileged_action_audit import _read_verified_teacher

    archive = tmp_path / 'different_checkpoint.pt'
    archive.write_bytes(b'not the saved collection checkpoint')
    with pytest.raises(ValueError, match='checkpoint bytes'):
        _read_verified_teacher(archive, {'initial_checkpoint_sha256': 'a'*64})


def test_legacy_mpc_action_semantics_do_not_enable_same_state_pairing():
    from go2_pvcnn.control.dh116.tabletop_privileged_action_audit import compare_mpc_archive
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT

    _, rollout, rows = replay_fixture()
    legacy = dict(version='dh116-vision-supervised-action-dataset-v1',
        observations=torch.zeros(2, 3844), actions=torch.zeros(2, 24),
        reference_actions=torch.ones(2, 24)*.05, visual_contract={'action_contract': ACTION_CONTRACT})
    comparison = compare_mpc_archive(legacy, rows, rollout)
    assert comparison['action_contract_matches'] is True
    assert comparison['paired_policy_states'] == 0
    assert comparison['same_state_action_metrics'] is None
    assert comparison['own_trajectory_mpc_vs_reference']['sample_count'] == 2
    assert not comparison['bc_admitted']


def test_mpc_pairing_requires_exact_full_state_and_measurement_mask():
    from go2_pvcnn.control.dh116.tabletop_privileged_action_audit import compare_mpc_archive
    from go2_pvcnn.control.dh116.tabletop_privileged_checkpoint import canonical_sha256
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import digest, state_contract
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT

    _, rollout, rows = replay_fixture()
    mpc = dict(version='dh116-mpc-pre-action-comparison-v1',
        identity=dict(action_contract_sha256=canonical_sha256(ACTION_CONTRACT),
                      state_contract_sha256=digest(state_contract())),
        observations=rollout['batch']['actor_obs'].clone(),
        privileged_valid=torch.ones(2, 371, dtype=torch.bool), actions=torch.zeros(2, 24),
        pre_action_identities=[r['pre_action_identity'].copy() for r in rows])
    assert compare_mpc_archive(mpc, rows, rollout)['paired_policy_states'] == 2
    mpc['privileged_valid'][0, 17] = False
    mpc['observations'][1, 17] = .01
    assert compare_mpc_archive(mpc, rows, rollout)['paired_policy_states'] == 0


def test_right_finger12_event_windows_keep_observed_boundaries_and_gaps():
    from go2_pvcnn.control.dh116.tabletop_privileged_action_audit import finger12_event_windows

    def row(step, position, *, episode='a', terminal=False):
        action = [0.] * 24
        action[7] = -.2
        return dict(kind='sample', run_uuid='fixture', env_id=0, episode_uid=episode, policy_tick=step//8,
                    physics_step=step, timestamp=step/480, terminal=terminal, failure_code=3 if terminal else 0,
                    action_audit=dict(joints=[finger(position)], issued_action=action))
    rows = [row(1, .03), row(2, .01), row(3, 0.), row(5, -.014, terminal=True), row(6, .001, episode='b')]
    first, second = finger12_event_windows(rows)
    assert first['episode_uid'] == 'a'
    assert second['episode_uid'] == 'b'
    assert second['events']['hard_guard_crossing'] is None
    assert second['events']['near_nominal'] is None
    events = first['events']
    assert events['near_nominal']['physics_step'] == 2
    assert events['inside_hard_tolerance']['physics_step'] == 3
    crossing = events['hard_guard_crossing']
    assert crossing['physics_step'] == 5
    assert crossing['missing_before_physics_steps'] == [4]
    assert crossing['missing_after_physics_steps'] == [6, 7, 8, 9]
    assert [sample['physics_step'] for sample in crossing['samples']] == [1, 2, 3, 5]
    assert crossing['samples'][-1]['issued_action'][7] == -.2
    assert crossing['samples'][-1]['joint']['legal_target_tracking_departure'] is True


def test_right_finger12_event_window_rejects_mismapped_joint():
    from go2_pvcnn.control.dh116.tabletop_privileged_action_audit import finger12_event_windows

    row = dict(kind='sample', run_uuid='fixture', env_id=0, episode_uid='a', policy_tick=1,
               physics_step=8, timestamp=8/480, terminal=True, failure_code=3,
               action_audit=dict(joints=[dict(finger(-.02), action_index=1)]))
    with pytest.raises(ValueError, match='canonical'):
        finger12_event_windows([row])
