"""CPU tests execute the actual bridge/Teacher loop; no physical qualification."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_privileged_cli as cli
from go2_pvcnn.control.dh116.tabletop_stereo_contract import EpisodeIdentities, require_sha
from go2_pvcnn.control.dh116.tabletop_stereo_native import NativeTensorBridge


class ClockCamera:
    def __init__(self, bridge):
        self.bridge = bridge
        self.calibration = SimpleNamespace(sha256='b'*64)
        self.last_frame = -1
        self.corrupt = None

    def sample(self, *, identity):
        self.last_frame += 1
        rows = [dict(row, frame_id=self.last_frame) for row in identity]
        if self.corrupt:
            name, value = self.corrupt
            rows[0][name] = value
        return dict(identity=rows, frame_id=self.last_frame,
            calibration_sha256=self.calibration.sha256,
            simulation_time=self.bridge.world.sim.current_time,
            native_evidence=False)  # A boolean must not substitute for actual identity checks.


@pytest.fixture
def bridge(tmp_path):
    value = NativeTensorBridge.__new__(NativeTensorBridge)
    value.n, value.device, value.policy_tick = 1, torch.device('cpu'), 0
    value.identities = EpisodeIdentities(1, run_uuid='CPU-teacher-identity')
    value.critic = torch.zeros(1, 371)
    value.previous_issued = torch.zeros(1, 24)
    value._terminal_rows, value._last_live_rows = {}, {}
    value._headers_written = set()
    value._initial_conditions = {value.identities.uid(0): {'source': 'CPU fixture'}}
    value._initial_sha = {value.identities.uid(0): 'c'*64}
    value._initial_clock = {value.identities.uid(0): (0, 0.)}
    value._initial_policy_tick = {value.identities.uid(0): 0}
    value._in_step = False
    value._cold_trials, value._trial_reset_counts = None, [1]
    value.grasp_styles = ['pinch_thumb_index']
    value.runtime_descriptor = None
    value.execution_owner, value.action_owners = 'teacher', ['teacher']
    value.checkpoint_sha256 = None
    value.output = tmp_path
    value._raw = (tmp_path / 'native_physics.jsonl').open('x', encoding='utf-8')
    value.world = SimpleNamespace(sim=SimpleNamespace(current_time=0.))
    data = SimpleNamespace(joint_pos_target=torch.full((1, 6), .72))
    value.device_world = SimpleNamespace(_physics_steps=0, hands=[SimpleNamespace(data=data)]*2,
        w=SimpleNamespace(ids=[list(range(6))]*2))
    value.task = SimpleNamespace(failure=torch.tensor([0]),
        surface_lift_success=SimpleNamespace(drift_m=torch.tensor([.0001]),
            duration=torch.tensor([0.]), window_good=torch.tensor([False])),
        surface_reference_bottom_m=torch.tensor(.36),
        m={'bottom': torch.tensor([.361]), 'q': torch.zeros(1, 12), 'qd': torch.zeros(1, 12)},
        command=torch.full((1, 12), .61))
    value.palm_poses = lambda: torch.zeros(1, 2, 7)
    value.physical_calls = 0

    def step(action):
        value.physical_calls += 1
        for _ in range(8):
            value.device_world._physics_steps += 1
            value.world.sim.current_time = value.device_world._physics_steps / 480
            value._capture(dict(live=torch.tensor([True]), done=torch.tensor([False]),
                native_success=torch.tensor([False]), action=action.clone(), dt_s=1/480,
                measurement_valid=torch.tensor([True])))
        return value.critic.clone(), torch.zeros(1), torch.tensor([False]), dict(
            success=torch.tensor([False]), failure=torch.tensor([0]), terms={'fixture': torch.zeros(1)})
    value.env = SimpleNamespace(step=step, successes=torch.zeros(1), episodes=torch.zeros(1))
    value.camera = ClockCamera(value)
    yield value
    value.close()


def checkpoint(tmp_path):
    path = tmp_path / 'actual_teacher.pt'
    path.write_bytes(b'CPU actual checkpoint bytes')
    return path


class DistinctTeacher:
    def sample(self, obs):
        return dict(actions=torch.full((len(obs), 24), -.3),
            latent_mean=torch.full((len(obs), 24), .2), latent_std=torch.full((len(obs), 24), .1))

    def deterministic(self, obs):
        return torch.full((len(obs), 24), .2)


def run_collection(bridge, tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    monkeypatch.setattr(cli, '_policy_for_evaluation', lambda *args: (DistinctTeacher(), {'cpu_fixture': True}))
    cli.collect_or_evaluate(bridge, {}, None, checkpoint=path, output=tmp_path, max_steps=2, collect=True)
    return path, torch.load(tmp_path / 'privileged_candidate_trajectories.pt', weights_only=False)


def test_actual_teacher_collection_binds_checkpoint_and_real_camera_before_first_physics(bridge, tmp_path, monkeypatch):
    path, saved = run_collection(bridge, tmp_path, monkeypatch)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in (tmp_path / 'native_physics.jsonl').read_text().splitlines()]
    assert rows[0]['kind'] == 'episode_start'
    for row in rows:
        assert row['owner'] == 'teacher' and row['checkpoint_sha256'] == sha
        assert row['calibration_sha256'] == bridge.camera.calibration.sha256
        assert row['frame_id'] >= 0
        source = saved['samples'][row['policy_tick']]['identity']
        assert row['pre_action_identity'] == source
        assert row['episode_uid'] == source['episode_uid'] and row['reset_id'] == source['reset_id']
        assert source['checkpoint_sha256'] == sha
    a, b = saved['samples']
    assert a['identity']['frame_id'] < b['identity']['frame_id']
    assert a['identity']['physics_tick_start'] == 0 and b['identity']['physics_tick_start'] == 8
    assert all(row['calibration_sha256'] != '0'*64 for row in rows)


def test_diagnostic_sample_mean_issued_and_applied_targets_stay_separate(bridge, tmp_path, monkeypatch):
    _, saved = run_collection(bridge, tmp_path, monkeypatch)
    sample = saved['samples'][0]
    torch.testing.assert_close(sample['teacher_sampled_action'], torch.full((24,), -.3))
    torch.testing.assert_close(sample['teacher_mean_action'], torch.full((24,), .2))
    torch.testing.assert_close(sample['actually_issued_action'], sample['teacher_mean_action'])
    assert sample['applied_action'] is None
    rows = [json.loads(line) for line in (tmp_path / 'native_physics.jsonl').read_text().splitlines()]
    native = next(row for row in rows if row['kind'] == 'sample')
    assert native['teacher_sampled_action'] != native['actually_issued_action']
    assert native['teacher_mean_action'] == native['actually_issued_action'] == native['actor_issued_action']
    assert native['applied_action'] is None
    assert native['applied_targets']['controller_joint_command_rad'] == pytest.approx([.61]*12)
    assert native['applied_targets']['actual_sdk_joint_target_rad'] == pytest.approx([.72]*12)


@pytest.mark.parametrize('field,value', [('calibration_sha256', '0'*64), ('env_id', 7),
    ('episode_uid', 'other-episode'), ('reset_id', 9), ('policy_tick', 4),
    ('physics_tick_start', 8), ('physics_tick_end', 8), ('simulation_timestamp', 1.),
    ('frame_id', -1)])
def test_real_camera_identity_corruption_is_rejected_before_action(bridge, tmp_path, monkeypatch, field, value):
    bridge.camera.corrupt = (field, value)
    with pytest.raises(ValueError, match='identity|calibration|frame|aligned|physics tick'):
        run_collection(bridge, tmp_path, monkeypatch)
    assert bridge.physical_calls == 0


def test_zero_sha_is_rejected_only_by_strict_native_identity_binding(bridge, tmp_path):
    assert require_sha('0'*64) == '0'*64  # Historical schema sentinel remains compatible.
    bridge.camera.calibration.sha256 = '0'*64
    with pytest.raises(ValueError, match='nonzero|calibration'):
        bridge.bind_teacher_checkpoint(checkpoint(tmp_path))
    assert bridge.physical_calls == 0


def test_checkpoint_changes_after_binding_are_rejected_before_action(bridge, tmp_path):
    path = checkpoint(tmp_path)
    bridge.bind_teacher_checkpoint(path)
    path.write_bytes(b'changed after binding')
    with pytest.raises(ValueError, match='checkpoint'):
        bridge.teacher_pre_action_identities()
    assert bridge.physical_calls == 0


def test_teacher_strict_action_requires_same_pre_state_and_recorded_diagnostics(bridge, tmp_path):
    bridge.bind_teacher_checkpoint(checkpoint(tmp_path))
    with pytest.raises(ValueError, match='pre.action|identity'):
        bridge.step(torch.zeros(1, 24))
    bridge.teacher_pre_action_identities()
    with pytest.raises(ValueError, match='diagnostic|action'):
        bridge.step(torch.zeros(1, 24))
    assert bridge.physical_calls == 0


def test_training_saves_and_binds_actual_initial_checkpoint_before_first_rollout(bridge, tmp_path):
    from go2_pvcnn.control.dh116.tabletop_surface_contract import bind_contract, ACTION_NAME, LEGACY_OBSERVATION_VERSION
    bridge.contract = bind_contract(dict(action_size=24, action=ACTION_NAME,
        observation_size=351, observation_version=LEGACY_OBSERVATION_VERSION))
    config, _ = cli.load_config(Path(__file__).resolve().parents[1] / 'configs/dh116_training/privileged_teacher_ppo_v1.json')
    config['rollout_steps'] = 1
    cli.train_teacher(bridge, config, cli.PPOConfig(epochs=1, minibatches=1),
        output=tmp_path, updates=2, checkpoint_interval=10)
    rows = [json.loads(line) for line in (tmp_path / 'native_physics.jsonl').read_text().splitlines()]
    sample0 = next(row for row in rows if row['kind'] == 'sample' and row['policy_tick'] == 0)
    sample1 = next(row for row in rows if row['kind'] == 'sample' and row['policy_tick'] == 1)
    assert sample0['checkpoint_sha256'] == hashlib.sha256((tmp_path / 'teacher_update_000000.pt').read_bytes()).hexdigest()
    assert sample1['checkpoint_sha256'] == hashlib.sha256((tmp_path / 'teacher_update_000001.pt').read_bytes()).hexdigest()
    assert sample0['checkpoint_sha256'] != sample1['checkpoint_sha256']
    assert sample0['teacher_sampled_action'] == sample0['actually_issued_action']



def test_stale_physics_clock_is_rejected_before_teacher_action(bridge, tmp_path):
    bridge.bind_teacher_checkpoint(checkpoint(tmp_path))
    bridge.teacher_pre_action_identities()
    action = torch.full((1, 24), .2)
    bridge.set_teacher_action_diagnostics(sampled_action=-action, mean_action=action, issued_action=action)
    bridge.device_world._physics_steps += 1
    with pytest.raises(ValueError, match='stale'):
        bridge.step(action)
    assert bridge.physical_calls == 0


def test_diagnostic_issued_action_cannot_override_submitted_action(bridge, tmp_path):
    bridge.bind_teacher_checkpoint(checkpoint(tmp_path))
    bridge.teacher_pre_action_identities()
    action = torch.full((1, 24), .2)
    bridge.set_teacher_action_diagnostics(sampled_action=-action, mean_action=action, issued_action=action)
    with pytest.raises(ValueError, match='diagnostics'):
        bridge.step(-action)
    assert bridge.physical_calls == 0


def test_native_packet_action_mismatch_cannot_be_written_as_teacher_evidence(bridge, tmp_path):
    bridge.bind_teacher_checkpoint(checkpoint(tmp_path))
    bridge.teacher_pre_action_identities()
    action = torch.full((1, 24), .2)
    bridge.set_teacher_action_diagnostics(sampled_action=-action, mean_action=action, issued_action=action)
    step = bridge.env.step
    bridge.env.step = lambda value: step(-value)
    with pytest.raises(ValueError, match='raw sample identity/action'):
        bridge.step(action)
    assert (tmp_path / 'native_physics.jsonl').read_text() == ''


def test_missing_raw_telemetry_cannot_complete_teacher_execution(bridge, tmp_path):
    bridge.bind_teacher_checkpoint(checkpoint(tmp_path))
    bridge.teacher_pre_action_identities()
    action = torch.full((1, 24), .2)
    bridge.set_teacher_action_diagnostics(sampled_action=-action, mean_action=action, issued_action=action)
    bridge.env.step = lambda _: (bridge.critic, torch.zeros(1), torch.tensor([False]), {})
    with pytest.raises(RuntimeError, match='actual aligned native raw identity'):
        bridge.step(action)


def test_first_teacher_checkpoint_cannot_be_bound_after_episode_physics_evidence(bridge, tmp_path):
    bridge.step(torch.zeros(1, 24))
    assert bridge.physical_calls == 1
    with pytest.raises(ValueError, match='before.*physical|episode.*evidence'):
        bridge.bind_teacher_checkpoint(checkpoint(tmp_path))


def test_checkpoint_bytes_cannot_change_between_model_load_and_identity_binding(bridge, tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    def loaded_policy(*args):
        path.write_bytes(b'different bytes after policy was loaded')
        return DistinctTeacher(), {'cpu_fixture': True}
    monkeypatch.setattr(cli, '_policy_for_evaluation', loaded_policy)
    with pytest.raises(ValueError, match='checkpoint'):
        cli.collect_or_evaluate(bridge, {}, None, checkpoint=path, output=tmp_path,
            max_steps=1, collect=True)
    assert bridge.physical_calls == 0
