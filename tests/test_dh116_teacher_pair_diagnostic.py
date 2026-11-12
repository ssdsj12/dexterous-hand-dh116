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
