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


def test_sdk_target_readback_and_explicit_reconstruction_source_flag(tmp_path):
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
    assert row[fields.index('right_sdk_target_native_readback')] == 1
    assert row[fields.index('sdk_target_left_finger12_rad')] == float(np.float32(.123456789))


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


def test_nonfinite_native_target_is_preserved_with_truthful_source_flag(tmp_path):
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
    assert row[fields.index('right_sdk_target_native_readback')] == 1
    assert np.isnan(row[fields.index('sdk_target_right_finger12_rad')])


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
