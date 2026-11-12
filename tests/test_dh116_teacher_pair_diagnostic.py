"""CPU contract tests only. Fixtures never qualify a native grasp."""
from copy import deepcopy

import pytest
import torch

from test_dh116_stereo_trials import manifest
from go2_pvcnn.control.dh116.tabletop_evaluation_trials import state_sha256


def test_plan_selects_exact_pinch_tripod_and_rejects_changed_contract():
    from go2_pvcnn.control.dh116.tabletop_teacher_pair_diagnostic import diagnostic_plan
    value = manifest(4)
    identifiers = [next(t['trial_id'] for t in value['trials'] if t['grasp_style'] == style)
                   for style in ('pinch_thumb_index', 'tripod_thumb_index_middle')]
    meta = value['metadata']
    plan = diagnostic_plan(value, identifiers, checkpoint_sha256=meta['source_sha256'],
        physics_sha256=meta['physics_sha256'], calibration_sha256=meta['camera_sha256'],
        seed=meta['seed'], steps=4)
    assert [r['trial_id'] for r in plan['trials']] == identifiers
    for changed in ('checkpoint_sha256', 'physics_sha256', 'calibration_sha256', 'seed'):
        arguments = dict(checkpoint_sha256=meta['source_sha256'], physics_sha256=meta['physics_sha256'],
            calibration_sha256=meta['camera_sha256'], seed=meta['seed'], steps=4)
        arguments[changed] = 999 if changed == 'seed' else 'f'*64
        with pytest.raises(ValueError, match='manifest'):
            diagnostic_plan(value, identifiers, **arguments)
    with pytest.raises(ValueError, match='control step'):
        diagnostic_plan(value, identifiers, checkpoint_sha256=meta['source_sha256'],
            physics_sha256=meta['physics_sha256'], calibration_sha256=meta['camera_sha256'],
            seed=meta['seed'], steps=5)


def test_frozen_sampled_vs_mean_returns_true_provenance_without_drawing_for_mean():
    from go2_pvcnn.control.dh116.tabletop_privileged_teacher import PrivilegedTeacher
    from go2_pvcnn.control.dh116.tabletop_teacher_pair_diagnostic import frozen_teacher_command
    model = PrivilegedTeacher()
    state = torch.zeros(2, 371)
    rng = torch.get_rng_state()
    mean = frozen_teacher_command(model, state, mode='teacher_mean')
    assert torch.equal(torch.get_rng_state(), rng)
    assert mean['sampled_action'] is None and mean['policy_audit']['latent_actions'] is None
    assert torch.equal(mean['issued_action'], model.deterministic(state))
    sampled = frozen_teacher_command(model, state, mode='teacher_sampled')
    assert torch.equal(sampled['issued_action'], sampled['sampled_action'])
    assert torch.equal(sampled['policy_audit']['latent_actions'].tanh(), sampled['issued_action'])


def test_pairing_checks_actually_read_full_joint_and_root_hashes_and_control_contract():
    from go2_pvcnn.control.dh116.tabletop_teacher_pair_diagnostic import compare_start_bindings
    starts = []
    trials = manifest(1)['trials']
    for trial in trials:
        state = trial['physical_state']
        starts.append(dict(trial_id=trial['trial_id'], physical_state=state,
            readback_state_sha256=state_sha256(state)))
    binding = dict(physics_sha256='1'*64, calibration_sha256='2'*64,
        checkpoint_sha256='3'*64, controller_sha256='4'*64,
        pre_action_371_sha256='5'*64, runtime_seed_sha256='6'*64, clock=[2, 2, 0.01],
        policy_steps_budget=4, seed=116, reset_readback={'actual_states':starts})
    a = dict(binding, mode='teacher_sampled', episode_uids=['sample:0','sample:1'])
    b = deepcopy(dict(binding, mode='teacher_mean', episode_uids=['mean:0','mean:1']))
    assert compare_start_bindings(a,b)['matching_full_physical_readbacks'] == 2
    b['reset_readback']['actual_states'][0]['physical_state']['joint_velocities'][0][2] += 0.01
    with pytest.raises(ValueError, match='readback'):
        compare_start_bindings(a,b)
    b = deepcopy(dict(binding, mode='teacher_mean', episode_uids=['mean:0','mean:1']))
    c = deepcopy(dict(binding, mode='legacy_mpc', episode_uids=['mpc:0','mpc:1']))
    c['controller_sha256'] = '9'*64
    with pytest.raises(ValueError, match='controller_sha256'):
        compare_start_bindings(a,b,c)


def test_mpc_preflight_refuses_reference_incompatible_with_observed_native_contract(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_vision_supervision_dataset as references
    from go2_pvcnn.control.dh116.tabletop_teacher_pair_diagnostic import check_mpc_reference
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    path = tmp_path / 'legacy.pt'
    path.write_bytes(b'private dataset')
    seen = []
    def reject(name, contract):
        seen.append((name, contract))
        raise ValueError('saved and current contracts must both be declared reset family members')
    monkeypatch.setattr(references, 'load_teacher_reference_dataset', reject)
    with pytest.raises(ValueError, match='reset family'):
        check_mpc_reference(path, file_sha256(path), {'actual_controller': True})
    assert seen == [(path, {'actual_controller': True})]


def test_cpu_preflight_preserves_identity_and_does_not_require_cuda_rng_device(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_privileged_checkpoint import VERSION
    from go2_pvcnn.control.dh116.tabletop_privileged_cli import training_identity
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig
    from go2_pvcnn.control.dh116.tabletop_privileged_teacher import PrivilegedTeacher
    from go2_pvcnn.control.dh116.tabletop_teacher_pair_diagnostic import check_frozen_teacher
    ppo = PPOConfig()
    model = PrivilegedTeacher()
    task = {'task': 'authored'}
    identity = training_identity(ppo, {'native': True}, task)
    file = tmp_path / 'model.pt'
    payload = dict(version=VERSION, identity=identity, num_envs=2, history_reset='clear',
        policy=model.state_dict(), rng={'cuda': [torch.zeros(17, dtype=torch.uint8)]})
    torch.save(payload, file)
    result = check_frozen_teacher(file, model, ppo, {'native': True}, task)
    assert result == identity
    with pytest.raises(ValueError, match='identity'):
        check_frozen_teacher(file, model, ppo, {'native': False}, task)


def test_mean_start_requires_completed_sampled_native_artifacts_and_exact_hashes(tmp_path):
    import json
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
    from go2_pvcnn.control.dh116.tabletop_teacher_pair_diagnostic import load_completed_sampled_start
    start = tmp_path/'start_binding.json'
    episodes = ['sampled:0:1', 'sampled:1:1']
    starts = []
    for trial in manifest(1)['trials']:
        state = trial['physical_state']
        starts.append(dict(trial_id=trial['trial_id'], physical_state=state,
            readback_state_sha256=state_sha256(state)))
    start.write_text(json.dumps(dict(mode='teacher_sampled', episode_uids=episodes,
        checkpoint_sha256='3'*64, calibration_sha256='2'*64,
        policy_steps_budget=4, reset_readback={'actual_states':starts})))
    with pytest.raises(FileNotFoundError):
        load_completed_sampled_start(start)
    native = tmp_path/'native'
    native.mkdir()
    raw = native/'native_physics.jsonl'
    records = [dict(kind='episode_start', episode_uid=uid, owner='teacher',
        env_id=index, checkpoint_sha256='3'*64, calibration_sha256='2'*64,
        first_episode=True, trial_id=starts[index]['trial_id'],
        initial_measurement_valid=True) for index,uid in enumerate(episodes)]
    records += [dict(kind='sample', episode_uid=uid, owner='teacher',
        action_owner='teacher', action_audit={'execution_mode':'teacher_sampled'}) for uid in episodes]
    raw.write_text(''.join(json.dumps(row)+'\n' for row in records))
    state = tmp_path/'teacher_pre_action.pt'
    state.write_bytes(b'archived pre-action data')
    completion = tmp_path/'completion.json'
    report = dict(status='bounded_frozen_teacher_diagnostic', mode='teacher_sampled',
        native_execution_performed=True, policy_steps=1, ppo_updates=0,
        start_binding_sha256=file_sha256(start), raw_480hz_sha256=file_sha256(raw),
        pre_action_sha256=file_sha256(state))
    completion.write_text(json.dumps(report))
    assert load_completed_sampled_start(start)['episode_uids'] == episodes
    raw.write_text(raw.read_text()+'\n')
    with pytest.raises(ValueError, match='SHA256'):
        load_completed_sampled_start(start)
    records[-1]['action_owner'] = 'mpc'
    raw.write_text(''.join(json.dumps(row)+'\n' for row in records))
    report['raw_480hz_sha256'] = file_sha256(raw)
    completion.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='owner'):
        load_completed_sampled_start(start)
