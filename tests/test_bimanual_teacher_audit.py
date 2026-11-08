"""CPU evidence-writer tests; fixture samples are not native successes."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS
from go2_pvcnn.control.dh116.tabletop_bimanual_teacher_audit import install_teacher_physics_audit, FIELDS


def fixture(tmp_path):
    n = 2
    roots = torch.zeros(n, 2, 7, dtype=torch.float64)
    roots[..., 3] = 1.
    roots[:, :, 2] = .4
    ids = [[JOINTS.index(name) for name in ACTIVE] for _ in range(2)]
    q = torch.zeros(n, 2, 11, dtype=torch.float64)
    q[:, :, ids[0]] = .2
    hands = [SimpleNamespace(joint_names=list(JOINTS),
               data=SimpleNamespace(joint_pos_target=torch.full((n, 11), .21))) for _ in range(2)]
    world = SimpleNamespace(n=n, device=torch.device('cpu'), roots=roots, q=q, qd=q*.1,
        commands=torch.full((n, 2, 6), .21, dtype=torch.float64), hands=hands,
        _physics_steps=10, w=SimpleNamespace(ids=ids, sim=SimpleNamespace(current_time=.2)),
        pad_measurements={'effective_pad_bad_env': torch.zeros(n, dtype=torch.bool)})
    loads = torch.zeros(n, 2, 5, dtype=torch.float64)
    loads[:, :, :2] = 1.
    vectors = torch.zeros(n, 2, 5, 3, dtype=torch.float64)
    vectors[:, :, 0, 0] = 1.
    vectors[:, :, 1, 0] = -1.
    measurement = dict(bottom=torch.full((n,), .365, dtype=torch.float64),
        effective_pad_loads_n=loads, effective_pad_vectors_w=vectors,
        effective_pad_valid=loads > 0,
        hand_object_loads=torch.ones(n, 2, dtype=torch.float64)*2)
    task = SimpleNamespace(m=measurement, surface_reference_bottom_m=torch.full((n,), .36),
        success=torch.zeros(n, dtype=torch.bool), failure=torch.zeros(n, dtype=torch.long),
        surface_lift_success=SimpleNamespace(duration=torch.zeros(n, dtype=torch.float64),
            drift_m=torch.full((n,), .0001, dtype=torch.float64),
            _latched=torch.tensor([True, False]), _physical_window_valid=torch.tensor([True, False])))
    env = SimpleNamespace(task=task, diagnostic_episode_ids=torch.tensor([7, 9]), _diagnostic_policy_step=41)
    return env, world, measurement


def fresh(audit, *, slip=None, valid=None, applicable=None, reason=None):
    n = audit.n
    audit.capture_packet(dict(
        slip_m=torch.full((n,), float('nan'), dtype=torch.float64) if slip is None else slip,
        slip_valid=torch.zeros(n, dtype=torch.bool) if valid is None else valid,
        slip_applicable=torch.zeros(n, dtype=torch.bool) if applicable is None else applicable,
        slip_reason_code=torch.ones(n, dtype=torch.long) if reason is None else reason))


def test_default_does_not_read_state_or_install_callback(tmp_path):
    class Unreadable:
        def __getattr__(self, key): raise AssertionError('default audit touched ' + key)
    assert install_teacher_physics_audit(Unreadable(), Unreadable(), tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_terminal_is_frozen_before_noop_and_episode_identity_changes(tmp_path):
    env, world, m = fixture(tmp_path)
    called = []
    prior = lambda *args: called.append('prior')
    env.joint_replay_observer = prior
    audit = install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    action = torch.linspace(-.5, .5, 48).reshape(2, 24)
    audit.submit(action, teacher_control_step=3)
    m['bottom'] = torch.tensor([.4, .39], dtype=torch.float64)
    world.roots[:, :, 2] += .001
    env.task.success[0] = True
    fresh(audit)
    env.joint_replay_observer(env, m, torch.tensor([True, True]), torch.tensor([True, False]))
    first = audit.terminal_samples[0].copy()
    env.diagnostic_episode_ids[0] += 1
    env._diagnostic_policy_step += 1
    world._physics_steps += 1
    world.w.sim.current_time += 1/480
    world.q[0] = 99.
    m['bottom'][0] = 9.
    audit.submit(action*0, teacher_control_step=4)
    fresh(audit)
    env.joint_replay_observer(env, m, torch.tensor([True, True]), torch.tensor([True, True]))
    summary = audit.finish()
    assert env.joint_replay_observer is prior and called == ['prior', 'prior']
    assert summary['first_terminal_count'] == 2 and summary['counts_as_student_success'] is False
    assert summary['native_runtime_verified'] is False
    assert summary['episodes'][0]['episode_uid'].endswith(':0:7')
    assert summary['episodes'][0]['supervision_episode_id'] == 0
    result = np.load(tmp_path/'teacher_physics_audit.npz', allow_pickle=False)
    fields = result['fields'].tolist()
    samples = result['samples']
    np.testing.assert_array_equal(result['terminal_samples'][0], first)
    assert samples[0, 0, fields.index('episode_id')] == 7
    assert samples[1, 0, fields.index('episode_id')] == 8
    assert samples[1, 0, fields.index('live')] == 0
    assert samples[1, 0, fields.index('newly_terminal')] == 0
    assert samples[0, 0, fields.index('policy_step')] == 41
    assert samples[0, 0, fields.index('teacher_control_step')] == 3
    assert samples[1, 0, fields.index('teacher_control_step')] == 4
    assert samples[0, 0, fields.index('physics_step')] == 10
    assert samples[0, 0, fields.index('timestamp_s')] == .2
    assert samples[0, 0, fields.index('object_net_lift_m')] == pytest.approx(.04, abs=2e-8)
    assert samples[0, 0, fields.index('object_reset_relative_lift_m')] == pytest.approx(.035)
    assert samples[0, 0, fields.index('left_palm_dz_m')] == pytest.approx(.001)
    assert samples[0, 0, fields.index('left_contact_valid')] == 1
    assert samples[0, 0, fields.index('native_anchor_drift_m')] == .0001
    assert np.isnan(samples[0, 1, fields.index('native_anchor_drift_m')])
    np.testing.assert_array_equal(samples[0, :, fields.index('teacher_issued_action_0'):fields.index('teacher_issued_action_0')+24], action.double().numpy())
    assert not any('actor' in field.lower() or 'latent' in field.lower() for field in FIELDS)


def test_sdk_command_cache_does_not_claim_backend_readback_or_fill_backend_columns(tmp_path):
    env, world, m = fixture(tmp_path)
    del world.hands[0].data.joint_pos_target
    world.commands[:, 0] = .123456789
    audit = install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    audit.submit(torch.zeros(2, 24), teacher_control_step=0)
    fresh(audit)
    audit.observe(env, m, torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    audit.finish()
    data = np.load(tmp_path/'teacher_physics_audit.npz')
    fields = data['fields'].tolist()
    row = data['samples'][0, 0]
    assert row[fields.index('left_sdk_target_native_readback')] == 0
    assert row[fields.index('right_sdk_target_native_readback')] == 0
    assert row[fields.index('left_sdk_command_cache_readback')] == 0
    assert row[fields.index('right_sdk_command_cache_readback')] == 1
    assert np.isnan(row[fields.index('sdk_target_left_finger12_rad')])
    assert np.isnan(row[fields.index('sdk_target_right_finger12_rad')])
    assert row[fields.index('reconstructed_target_left_finger12_rad')] == float(np.float32(.123456789))
    assert row[fields.index('command_cache_target_right_finger12_rad')] == float(np.float32(.21))


def test_invalid_contacts_are_not_manufactured_and_observer_restores_on_error(tmp_path):
    env, world, m = fixture(tmp_path)
    audit = install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    audit.submit(torch.zeros(2, 24), teacher_control_step=0)
    world.pad_measurements['effective_pad_bad_env'][0] = True
    fresh(audit)
    audit.observe(env, m, torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    audit.finish(error='fixture failure')
    assert env.joint_replay_observer is None
    assert json.loads((tmp_path/'teacher_physics_audit_summary.json').read_text())['status'] == 'error'
    data = np.load(tmp_path/'teacher_physics_audit.npz')
    fields = data['fields'].tolist()
    assert data['samples'][0, 0, fields.index('left_contact_valid')] == 0
    assert data['samples'][0, 0, fields.index('left_contact_measurement_valid')] == 0


def test_nonfinite_command_cache_is_preserved_without_claiming_backend_readback(tmp_path):
    env, world, m = fixture(tmp_path)
    world.hands[1].data.joint_pos_target[0, JOINTS.index('finger12')] = float('nan')
    audit = install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    audit.submit(torch.zeros(2, 24), teacher_control_step=0)
    fresh(audit)
    audit.observe(env, m, torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    audit.finish()
    data = np.load(tmp_path/'teacher_physics_audit.npz')
    fields = data['fields'].tolist()
    row = data['samples'][0, 0]
    assert row[fields.index('right_sdk_target_native_readback')] == 0
    assert row[fields.index('right_sdk_command_cache_readback')] == 1
    assert row[fields.index('right_sdk_command_cache_finite')] == 0
    assert np.isnan(row[fields.index('command_cache_target_right_finger12_rad')])


def full_target_fixture(tmp_path):
    env, world, measurement = fixture(tmp_path)
    names = list(JOINTS) + ['wrist_x', 'wrist_y', 'wrist_z', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']
    per_hand = [names, list(reversed(names))]
    backend, cache = [], []
    hands = []
    for hand_names in per_hand:
        targets = np.full((2, 17), .3, dtype=np.float32)
        targets[:, hand_names.index('wrist_z')] = .007
        commands = np.full((2, 17), .9, dtype=np.float32)
        types = np.array([[1 if name in ('wrist_x', 'wrist_y', 'wrist_z') else 0
                           for name in hand_names]]*2, dtype=np.uint8)
        view = SimpleNamespace(get_dof_position_targets=lambda targets=targets: targets,
                               get_dof_types=lambda types=types: types)
        hands.append(SimpleNamespace(joint_names=hand_names, root_physx_view=view,
                                     data=SimpleNamespace(joint_pos_target=torch.from_numpy(commands))))
        backend.append(targets)
        cache.append(commands)
    world.hands = hands
    world.w.ids = [[hand_names.index(name) for name in ACTIVE] for hand_names in per_hand]
    world.q = torch.full((2, 2, 17), .2, dtype=torch.float64)
    world.qd = world.q*.1
    return env, world, measurement, backend, cache


def test_v2_separates_full_backend_cache_and_reconstructed_targets_in_actual_named_order(tmp_path):
    env, world, m, backend, cache = full_target_fixture(tmp_path)
    before = [value.clone() for value in (world.q, world.qd, world.roots, world.commands)]
    audit = install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    audit.submit(torch.zeros(2, 24), teacher_control_step=0)
    fresh(audit)
    audit.observe(env, m, torch.ones(2, dtype=torch.bool), torch.tensor([True, False]))
    summary = audit.finish()
    assert summary['version'] == 'dh116-teacher-480hz-first-episode-audit-v2'
    assert summary['target_sources']['backend'] == 'physx_backend_get_dof_position_targets'
    assert summary['target_sources']['command_cache'] == 'isaaclab_user_command_cache_joint_pos_target'
    assert summary['target_sources']['reconstructed'] == 'float32_submission_reconstructed_from_integrated_world.commands;active6_only'
    assert summary['legacy_v1_target_scope'] == 'IsaacLab_command_cache_or_reconstruction;not_PhysX_backend_readback'
    for actual, expected in zip((world.q, world.qd, world.roots, world.commands), before):
        assert torch.equal(actual, expected)
    data = np.load(tmp_path/'teacher_physics_audit.npz', allow_pickle=False)
    fields = data['fields'].tolist()
    row = data['samples'][0, 0]
    assert row[fields.index('left_sdk_target_native_readback')] == 1
    assert row[fields.index('left_sdk_target_finite')] == 1
    assert row[fields.index('sdk_target_left_finger12_rad')] == float(np.float32(.3))
    assert row[fields.index('command_cache_target_left_finger12_rad')] == float(np.float32(.9))
    assert row[fields.index('reconstructed_target_left_finger12_rad')] == float(np.float32(.21))
    for hi, side in enumerate(('left', 'right')):
        assert data[f'joint_names_{side}'].tolist() == world.hands[hi].joint_names
        np.testing.assert_array_equal(data[f'backend_full_target_{side}'][0], backend[hi])
        np.testing.assert_array_equal(data[f'command_cache_full_target_{side}'][0], cache[hi])
        assert data[f'backend_full_target_{side}'].shape == (1, 2, 17)
        index = world.hands[hi].joint_names.index('wrist_z')
        assert data[f'dof_types_{side}'][0, 0, index] == 1
        assert summary['target_readback_statuses'][hi]['units_by_name']['wrist_z'] == 'm'
        assert summary['target_readback_statuses'][hi]['units_by_name']['wrist_roll'] == 'rad'


def test_backend_getter_failure_never_fills_backend_with_available_cache(tmp_path):
    env, world, m, _, _ = full_target_fixture(tmp_path)
    def fail():
        raise RuntimeError('backend read failed')
    world.hands[0].root_physx_view.get_dof_position_targets = fail
    audit = install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    audit.submit(torch.zeros(2, 24), teacher_control_step=0)
    fresh(audit)
    audit.observe(env, m, torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    summary = audit.finish()
    data = np.load(tmp_path/'teacher_physics_audit.npz')
    fields = data['fields'].tolist()
    assert data['samples'][0, :, fields.index('left_sdk_target_native_readback')].tolist() == [0, 0]
    assert data['samples'][0, :, fields.index('left_sdk_command_cache_readback')].tolist() == [1, 1]
    assert np.isnan(data['backend_full_target_left']).all()
    assert 'backend read failed' in summary['target_readback_statuses'][0]['backend']['error']


def test_nonfinite_backend_keeps_raw_evidence_and_distinguishes_availability_from_finiteness(tmp_path):
    env, world, m, backend, _ = full_target_fixture(tmp_path)
    backend[1][0, world.hands[1].joint_names.index('finger12')] = np.nan
    audit = install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    audit.submit(torch.zeros(2, 24), teacher_control_step=0)
    fresh(audit)
    audit.observe(env, m, torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    audit.finish()
    data = np.load(tmp_path/'teacher_physics_audit.npz')
    fields = data['fields'].tolist()
    row = data['samples'][0, 0]
    assert row[fields.index('right_sdk_target_native_readback')] == 1
    assert row[fields.index('right_sdk_target_finite')] == 0
    assert row[fields.index('right_sdk_command_cache_finite')] == 1
    assert np.isnan(row[fields.index('sdk_target_right_finger12_rad')])
    assert row[fields.index('command_cache_target_right_finger12_rad')] == float(np.float32(.9))


def test_full_target_terminal_snapshots_survive_noop_and_real_reset(tmp_path):
    env, world, m, backend, cache = full_target_fixture(tmp_path)
    expected_backend = backend[0][0].copy()
    expected_cache = cache[0][0].copy()
    audit = install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    audit.submit(torch.zeros(2, 24), teacher_control_step=0)
    fresh(audit)
    audit.observe(env, m, torch.ones(2, dtype=torch.bool), torch.tensor([True, False]))
    backend[0][:] = 9.
    cache[0][:] = 8.
    env.diagnostic_episode_ids[0] += 1
    world._physics_steps += 1
    world.w.sim.current_time += 1/480
    fresh(audit)
    audit.observe(env, m, torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    audit.finish()
    data = np.load(tmp_path/'teacher_physics_audit.npz', allow_pickle=False)
    np.testing.assert_array_equal(data['terminal_backend_full_target_left'][0], expected_backend)
    np.testing.assert_array_equal(data['terminal_command_cache_full_target_left'][0], expected_cache)
    np.testing.assert_array_equal(data['backend_full_target_left'][1, 0], np.full(17, 9.))


def test_immutable_output_collision_leaves_prior_observer_untouched(tmp_path):
    env, world, _ = fixture(tmp_path)
    prior = lambda *args: None
    env.joint_replay_observer = prior
    (tmp_path/'teacher_physics_audit.npz').write_bytes(b'prior evidence')
    with pytest.raises(FileExistsError): install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    assert env.joint_replay_observer is prior
    assert (tmp_path/'teacher_physics_audit.npz').read_bytes() == b'prior evidence'


def test_fresh_contact_slip_survives_after_consume_reanchor_and_missing_anchor_is_na(tmp_path):
    env, world, m = fixture(tmp_path)
    audit = install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    audit.submit(torch.zeros(2, 24), teacher_control_step=0)
    # Consume has already reanchored the completed native Hold tracker.
    env.task.surface_lift_success.drift_m.zero_()
    fresh(audit, slip=torch.tensor([.0006, float('nan')], dtype=torch.float64),
        valid=torch.tensor([True, False]), applicable=torch.tensor([True, False]),
        reason=torch.tensor([0, 1]))
    audit.observe(env, m, torch.ones(2, dtype=torch.bool), torch.tensor([True, False]))
    audit.finish()
    data = np.load(tmp_path/'teacher_physics_audit.npz')
    fields = data['fields'].tolist()
    assert data['samples'][0, 0, fields.index('native_anchor_drift_m')] == 0
    assert data['samples'][0, 0, fields.index('fresh_contact_anchor_slip_m')] == .0006
    assert data['samples'][0, 0, fields.index('fresh_contact_anchor_slip_valid')] == 1
    assert np.isnan(data['samples'][0, 1, fields.index('fresh_contact_anchor_slip_m')])
    assert data['samples'][0, 1, fields.index('fresh_contact_anchor_slip_applicable')] == 0


@pytest.mark.parametrize('mismatch', ['episode', 'physics', 'timestamp'])
def test_fresh_contact_packet_rejects_wrong_physical_identity(tmp_path, mismatch):
    env, world, m = fixture(tmp_path)
    audit = install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    audit.submit(torch.zeros(2, 24), teacher_control_step=0)
    fresh(audit)
    if mismatch == 'episode': env.diagnostic_episode_ids[0] += 1
    elif mismatch == 'physics': world._physics_steps += 1
    else: world.w.sim.current_time += 1/480
    with pytest.raises(RuntimeError, match='episode/physics/time mismatch'):
        audit.observe(env, m, torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    audit.finish(error='fixture mismatch')
    assert not hasattr(env, 'evaluation_telemetry')


def test_existing_evaluation_callback_and_reference_are_restored(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_evaluation_telemetry import EvaluationTelemetry
    env, world, _ = fixture(tmp_path)
    calls = []
    callback = lambda packet: calls.append(True)
    prior = EvaluationTelemetry(2, world.device, callback)
    prior.reference_bottom_m.fill_(.75)
    env.evaluation_telemetry = prior
    audit = install_teacher_physics_audit(env, world, tmp_path, enabled=True)
    fresh(audit)
    audit.finish()
    assert calls == [True]
    assert env.evaluation_telemetry is prior and prior.callback is callback
    assert prior.reference_bottom_m.eq(.75).all()


def test_launcher_rejects_teacher_audit_in_non_teacher_mode(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root/'tools/run_dh116_stage5_native.py'),
        '--root', str(tmp_path), '--name', 'blocked', '--mode', 'z0', '--bimanual-teacher-physics-audit'],
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and 'requires teacher or teacher_adapter' in result.stderr
    assert not list(tmp_path.iterdir())
