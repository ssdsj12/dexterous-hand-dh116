import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_mpc_privileged_capture import ACTION_CONTRACT


def _fixture():
    first_clock = 18
    n = 2
    q = torch.full((n, 12), .25, dtype=torch.float64)
    pose = torch.zeros(n, 2, 7, dtype=torch.float64)
    pose[..., 3] = 1
    pose[..., 0] = .01
    seed = pose.clone()
    seed[..., 0] = 0
    rows = [dict(run_uuid='run', env_id=0, episode_id=0, episode_uid='run:0:0',
                 camera_episode_uid='run:0:first', reset_generation=0, tick=tick,
                 physics_step=first_clock+8*tick, simulation_time_s=(first_clock+8*tick)/480,
                 policy_clock_before=dict(sdk_step=first_clock+8*tick, world_step=8*tick,
                                          sdk_time_s=(first_clock+8*tick)/480),
                 terminal_in_interval=False) for tick in range(n)]
    actions = torch.zeros(n, 24)
    actions[0, 7] = .125
    obs = torch.zeros(n, 371)
    obs[:, :12] = .25
    obs[:, 24:36] = .25
    obs[:, 55] = .4
    obs[:, 60] = .6
    obs[:, 61] = .8
    obs[:, 64] = 1
    obs[:, 68] = .5
    obs[:, 78:88] = .1
    obs[:, 100] = 1
    obs[:, 106:108] = .4
    obs[:, 108] = 1
    obs[:, 153] = obs[:, 160] = .2
    obs[:, 156] = obs[:, 163] = 1
    obs[:, 191] = .5
    obs[:, 197] = .4
    obs[:, 239:243] = 1
    obs[:, 203:215] = .25
    obs[1, 215:239] = actions[0]
    archive = dict(version='dh116-native-mpc-privileged-capture-v1', run_uuid='run',
                   action_owner='legacy_teacher_mpc', action_contract=ACTION_CONTRACT,
                   counts_as_Student_success=False, qualified_for_bc=False,
                   observations=obs, actions=actions, sample_metadata=rows,
                   privileged_valid=torch.zeros_like(obs, dtype=torch.bool))
    archive['privileged_valid'][:, :24] = True
    m = dict(q=q, qd=q*0, clearance=torch.full((n,),.03,dtype=q.dtype),
             object_relative=torch.tensor([[.04, 0, 0]]*n, dtype=q.dtype),
             object_quat=torch.tensor([[1., 0, 0, 0]]*n, dtype=q.dtype),
             object_velocity=torch.tensor([[.05, 0, 0, 0, 0, 0]]*n, dtype=q.dtype),
             grouped_object_loads=torch.zeros(n,2,6,dtype=q.dtype),
             hand_object_loads=torch.zeros(n,2,dtype=q.dtype),
             mimic_errors=torch.full((n, 10), .002, dtype=q.dtype),
             adaptive_palm_offset=torch.tensor([[[.003,0,0],[0,0,0]]]*n, dtype=q.dtype),
             grip_stiffness=torch.full((n, 2), 14., dtype=q.dtype),
             bottom=torch.full((n,), .33, dtype=q.dtype), pad_relative=q.reshape(n, 4, 3)*0,
             wrist_pose=pose, wrist_velocity=pose[:, :, :6]*0,
             wrist_joint_angles=pose[:, :, :3]*0,
             pad_reference_points_w=torch.tensor([[[[.05,0,0]]*5]*2]*n, dtype=q.dtype))
    rs = []
    for tick, row in enumerate(rows):
        batch_actions = torch.zeros(n, 24)
        batch_actions[0] = actions[tick]
        prior = [0.] * 24 if tick == 0 else actions[tick-1].tolist()
        rs.append(dict(samples=dict(run_uuid=['run', None], env_id=[0, None],
            episode_uid=['run:0:first', None], tick=[tick, None], actions=batch_actions,
            previous_action=[prior, None], first_physics_step=[row['physics_step'],None],
            policy_timestamp_s=[row['simulation_time_s'],None],
            terminal=torch.zeros(n,dtype=torch.bool), valid=torch.tensor([True,False])),
            post_physics_feedback=[dict(sdk_physics_step=row['physics_step']+1,
                backend_readback_sdk_physics_step=row['physics_step']+1,
                sdk_timestamp_s=row['simulation_time_s']+1/480,
                backend_readback_sdk_timestamp_s=row['simulation_time_s']+1/480,
                native_runtime_verified=True, clock_aligned=True, source_valid=torch.tensor([True,False]),
                source='original_advance_returned_refresh_before_TensorEnv_terminal_reset')],
            physics_context=dict(joint_lower=-torch.ones(n,1,12,dtype=q.dtype),
                joint_upper=torch.ones(n,1,12,dtype=q.dtype), task_offset_m=torch.full((n,1),.02,dtype=q.dtype),
                root_seed_pose=seed[:,None], source_valid=torch.ones(n,1,dtype=torch.bool),
                live=torch.ones(n,1,dtype=torch.bool)),
            controller_start=dict(joint_command_rad=q, palm_command_offset_m=m['adaptive_palm_offset'].clone(),
                wrist_command_angle_rad=m['wrist_joint_angles'].clone()),
            raw_measured_packets=[m]))
    for r in rs:
        r['controller_start']['wrist_command_angle_rad'][:,0,0] = torch.pi/4
        r['controller_start']['palm_command_offset_m'][:,0,0] = .02
    return archive, dict(version='dh116-native-sequence-capture-unqualified-v1',
                         source='actual_precontrol_and_native_submission_snapshots',
                         controller_contract=dict(action_contract=ACTION_CONTRACT),
                         error=None, incomplete_policy_context=None, policy_records=rs)


def test_source_bound_mask_is_exact_and_keeps_contact_and_force_unqualified():
    from go2_pvcnn.control.dh116.tabletop_mpc_observation_validity import audit_observation_mask

    archive, sequence = _fixture()
    archive['observations'][0, 61] += .001
    original = archive['privileged_valid'].clone()
    mask = audit_observation_mask(archive, sequence)
    assert mask.dtype == torch.bool and mask.shape == archive['observations'].shape
    assert mask[:, 0:36].all()
    assert mask[:, 153:185].all()
    assert mask[:, 191:251].all()
    assert mask[:, 321:351].all()
    assert mask[1, 61] and not mask[0, 61]
    assert not mask[:, 185:191].any()
    assert not mask[:, 251:321].any()
    assert not mask[:, 351:371].any()
    assert archive['privileged_valid'].equal(original)
    assert archive['qualified_for_bc'] is False


@pytest.mark.parametrize('fault', ['episode', 'clock', 'action', 'readback', 'prior'])
def test_refuses_incompatible_sidecar_binding(fault):
    from go2_pvcnn.control.dh116.tabletop_mpc_observation_validity import audit_observation_mask

    archive, sequence = _fixture()
    first, second = sequence['policy_records']
    if fault == 'episode':
        first['samples']['episode_uid'][0] = 'another:first'
    if fault == 'clock':
        first['samples']['first_physics_step'][0] += 1
    if fault == 'action':
        first['samples']['actions'][0,0] = .25
    if fault == 'readback':
        first['post_physics_feedback'][0]['native_runtime_verified'] = False
    if fault == 'prior':
        second['samples']['previous_action'][0][0] = .25
    with pytest.raises(ValueError, match='sidecar|prior'):
        audit_observation_mask(archive, sequence)


def test_hash_bound_file_audit_retains_original_and_exclusive_output(tmp_path):
    from scripts.audit_dh116_mpc_observation_validity import audit_files
    from go2_pvcnn.control.dh116.tabletop_mpc_privileged_capture import file_identity

    archive, sequence = _fixture()
    a, s, output = [tmp_path / name for name in ('archive.pt', 'sequence.pt', 'proof.pt')]
    torch.save(archive, a)
    torch.save(sequence, s)
    a_id, s_id = file_identity(a), file_identity(s)
    with pytest.raises(ValueError, match='SHA'):
        audit_files(a, '0' * 64, s, s_id['sha256'], output)
    assert not output.exists()
    summary = audit_files(a, a_id['sha256'], s, s_id['sha256'], output)
    proof = torch.load(output, weights_only=True)
    assert summary['checked_rows'] == len(archive['sample_metadata'])
    assert proof['archive'] == a_id and proof['sequence_sidecar'] == s_id
    assert not proof['qualified_for_bc'] and not proof['force_contact_qualification']
    assert proof['privileged_valid'][:, 153:185].all()
    assert file_identity(a) == a_id and file_identity(s) == s_id
    with pytest.raises(ValueError, match='new'):
        audit_files(a, a_id['sha256'], s, s_id['sha256'], output)


@pytest.mark.parametrize('fault', [None, 'missing', 'bad_clock', 'inactive', 'terminal', 'reset_boundary'])
def test_wrist_interval_requires_previous_issued_action_and_eight_native_samples(fault):
    from go2_pvcnn.control.dh116.tabletop_mpc_observation_validity import audit_observation_mask

    archive, sequence = _fixture()
    first, second = sequence['policy_records']
    a = archive['sample_metadata'][0]
    archive['actions'][0, 14] = .25
    first['samples']['actions'][0, 14] = .25
    second['samples']['previous_action'][0][14] = .25
    archive['observations'][1, 215+14] = .25
    archive['observations'][1, 367:371] = torch.tensor([.25, 0., 1., -2.])
    a['physical_substeps'] = 8
    a['last_physics_step'] = a['physics_step'] + 8
    endpoint = first['raw_measured_packets'][0]['wrist_pose'].clone()
    endpoint[0, :, 2] = torch.tensor([.001, -.002], dtype=endpoint.dtype)
    second['raw_measured_packets'] = [dict(second['raw_measured_packets'][0], wrist_pose=endpoint)]
    first['post_physics_feedback'] = [
        dict(first['post_physics_feedback'][0], sdk_physics_step=a['physics_step']+k,
             backend_readback_sdk_physics_step=a['physics_step']+k,
             sdk_timestamp_s=a['simulation_time_s']+k/480,
             backend_readback_sdk_timestamp_s=a['simulation_time_s']+k/480,
             measured_root_pose=endpoint) for k in range(1, 9)]
    first['physics_context']['live'] = torch.ones(2, 8, dtype=torch.bool)
    first['physics_context']['source_valid'] = torch.ones(2, 8, dtype=torch.bool)
    first['physics_context']['dt_s'] = torch.full((2, 8), 1/480, dtype=torch.float64)
    if fault == 'missing':
        first['post_physics_feedback'].pop(4)
    if fault == 'bad_clock':
        first['post_physics_feedback'][3]['sdk_physics_step'] += 1
    if fault == 'inactive':
        first['physics_context']['live'][0, 3] = False
    if fault == 'terminal':
        a['terminal_in_interval'] = True
        first['samples']['terminal'][0] = True
    if fault == 'reset_boundary':
        archive['sample_metadata'][1]['reset_generation'] = 1
    if fault in ('terminal','reset_boundary'):
        with pytest.raises(ValueError, match='prior'):
            audit_observation_mask(archive, sequence)
    else:
        mask = audit_observation_mask(archive, sequence)
        assert not mask[0, 367:371].any()
        if fault:
            assert not mask[1, 367:371].any()
        else:
            assert mask[1, 367:371].all()
        assert not mask[:, 351:367].any()
