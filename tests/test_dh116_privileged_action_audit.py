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
