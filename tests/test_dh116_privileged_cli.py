"""CPU integration uses an explicitly synthetic task; it is not native proof."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_privileged_cli as cli
from go2_pvcnn.control.dh116.tabletop_surface_contract import bind_contract, ACTION_NAME, LEGACY_OBSERVATION_VERSION


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/dh116_training/privileged_teacher_ppo_v1.json'


class CPUFixtureBridge:
    """Synthetic reward/transition plumbing only; cannot qualify demonstrations."""
    def __init__(self):
        self.n, self.device = 2, torch.device('cpu')
        self.contract = bind_contract(dict(action_size=24, action=ACTION_NAME,
            observation_size=351, observation_version=LEGACY_OBSERVATION_VERSION))
        self.critic = torch.zeros(2, 371)
        self.env = SimpleNamespace(successes=torch.zeros(2, dtype=torch.long),
            episodes=torch.zeros(2, dtype=torch.long))

    def step(self, actions):
        assert actions.shape == (2, 24)
        self.critic = self.critic + .001
        reward = -actions.square().mean(-1)
        return self.critic.clone(), reward, torch.zeros(2, dtype=torch.bool), {'terms': {'fixture': reward.clone()}}


def test_native_teacher_loop_updates_saves_and_recovers_without_touching_source(tmp_path):
    config, _ = cli.load_config(CONFIG)
    config['rollout_steps'] = 2
    ppo = cli.PPOConfig(epochs=1, minibatches=2)
    first = tmp_path / 'first'; first.mkdir()
    bridge = CPUFixtureBridge()
    result = cli.train_teacher(bridge, config, ppo, output=first, updates=2,
        checkpoint_interval=1)
    assert result['updates'] == 2 and result['native_grasp_qualification'] is False
    paths = sorted(first.glob('teacher_update_*.pt'))
    assert len(paths) == 3
    before = [p.read_bytes() for p in paths]
    second = tmp_path / 'resume'; second.mkdir()
    result = cli.train_teacher(CPUFixtureBridge(), config, ppo, output=second,
        updates=1, checkpoint=paths[-1], checkpoint_interval=10)
    assert result['updates'] == 3
    assert (second / 'teacher_update_000003.pt').exists()
    assert [p.read_bytes() for p in paths] == before
    assert len((first / 'iterations.jsonl').read_text().splitlines()) == 2


def test_bounded_teacher_diagnostic_records_frozen_inputs_and_terminal_bootstrap(tmp_path):
    config, _ = cli.load_config(CONFIG)
    bridge = CPUFixtureBridge()
    real_step = bridge.step
    def with_terminal(action):
        obs, reward, done, extra = real_step(action)
        if float(obs[0, 0]) < .0015:
            done[0] = True
        return obs, reward, done, extra
    bridge.step = with_terminal
    result = cli.train_teacher(bridge, config, cli.PPOConfig(epochs=1, minibatches=8, target_kl=.02),
        output=tmp_path, updates=1, record_diagnostic_rollout=True)
    assert result['updates'] == 1
    saved = torch.load(tmp_path / 'diagnostic_rollout.pt', weights_only=False)
    batch = saved['batch']
    assert batch['actor_obs'].shape == (64, 371)
    assert batch['old_mu'].shape == batch['latent_actions'].shape == (64, 24)
    assert batch['old_log_prob'].shape == (64,)
    # Terminal return is (reward - value) + value in float32 GAE.
    # Account for one cancellation ULP without admitting terminal bootstrap.
    rounding = torch.finfo(batch['rewards'].dtype).eps * (
        batch['values'][0].abs() + batch['rewards'][0].abs())
    torch.testing.assert_close(saved['returns'][0], batch['rewards'][0], rtol=0., atol=float(rounding))
    assert batch['done'][0] and not batch['bootstrap_mask'][0]
    assert saved['initial_checkpoint_sha256'] == cli.file_sha256(tmp_path / 'teacher_update_000000.pt')
    assert saved['rng']['torch'].dtype == torch.uint8
    row = json.loads((tmp_path / 'iterations.jsonl').read_text())
    assert row['action_diagnostics']['normalized_action_clipping_fraction'] == 0.
    assert row['action_diagnostics']['joint_safety_projection'] is None
    assert row['gae_diagnostics']['terminal_bootstrap_count'] == 0
    assert row['gae_diagnostics']['terminal_count'] == 1


def test_diagnostic_capture_rejects_larger_budget_before_writes(tmp_path):
    config, ppo = cli.load_config(CONFIG)
    with pytest.raises(ValueError, match='diagnostic rollout.*2 environments'):
        cli.train_teacher(CPUFixtureBridge(), config, ppo, output=tmp_path, updates=2,
            record_diagnostic_rollout=True)
    assert not list(tmp_path.iterdir())


def test_teacher_no_accepted_optimizer_step_preserves_initial_checkpoint_and_reports_rejection(tmp_path):
    config, _ = cli.load_config(CONFIG)
    config['rollout_steps'] = 2
    ppo = cli.PPOConfig(epochs=1, minibatches=2, lr=.01, target_kl=1e-12, kl_backtrack_steps=1)
    result = cli.train_teacher(CPUFixtureBridge(), config, ppo, output=tmp_path, updates=1)
    assert result['status'] == 'teacher_optimizer_update_rejected'
    assert result['updates'] == 0
    assert [path.name for path in tmp_path.glob('teacher_update_*.pt')] == ['teacher_update_000000.pt']
    row = json.loads((tmp_path / 'iterations.jsonl').read_text())
    assert row['update'] == 0 and not row['optimizer_update_accepted']


def test_cli_config_inspection_does_not_start_isaac_or_create_run(tmp_path, capsys):
    target = tmp_path / 'forbidden'
    assert cli.main('train', ['--config', str(CONFIG), '--output', str(target), '--check-config']) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['validated'] and not value['native_started'] and not target.exists()
    import sys
    assert not any(k.startswith(('isaaclab', 'omni.', 'isaacsim')) for k in sys.modules)
    with pytest.raises(ValueError, match='disabled'):
        cli.main('train', ['--config', str(CONFIG), '--output', str(target)])
    assert not target.exists()


def test_native_entry_fails_before_io_without_opt_in(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_native import launch_native
    with pytest.raises(ValueError, match='disabled'):
        launch_native(None, native_arguments='missing.json', output=tmp_path / 'never')
    assert not (tmp_path / 'never').exists()


def test_pre_reset_terminal_observer_preserves_last_live_sample_across_noops(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_native import NativeTensorBridge
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import EpisodeIdentities
    bridge = NativeTensorBridge.__new__(NativeTensorBridge)
    bridge.n, bridge.policy_tick = 1, 4
    bridge.identities = EpisodeIdentities(1, run_uuid='cpu-fixture')
    bridge._terminal_rows, bridge._last_live_rows = {}, {}
    bridge._raw = (tmp_path / 'fixture.jsonl').open('x', encoding='utf-8')
    bridge.world = SimpleNamespace(sim=SimpleNamespace(current_time=8/480))
    bridge.device_world = SimpleNamespace(_physics_steps=8)
    bridge.task = SimpleNamespace(failure=torch.tensor([7]),
        surface_lift_success=SimpleNamespace(drift_m=torch.tensor([.0001]),
            duration=torch.tensor([0.]), window_good=torch.tensor([False])),
        surface_reference_bottom_m=torch.tensor(.36),
        m={'bottom': torch.tensor([.361]), 'q': torch.zeros(1, 12), 'qd': torch.zeros(1, 12)},
        command=torch.zeros(1, 12))
    bridge.palm_poses = lambda: torch.zeros(1, 2, 7)
    packet = dict(live=torch.tensor([True]), done=torch.tensor([True]),
        native_success=torch.tensor([False]), action=torch.zeros(1, 24),
        dt_s=1/480, measurement_valid=torch.tensor([True]))
    bridge._capture(packet)
    before = bridge._terminal_rows['cpu-fixture:0:0'].copy()
    bridge.task.m['bottom'].fill_(99)
    bridge.device_world._physics_steps = 9
    bridge.world.sim.current_time = 9/480
    bridge._capture(dict(packet, live=torch.tensor([False]), done=torch.tensor([False])))
    assert bridge._terminal_rows['cpu-fixture:0:0'] == before
    bridge.close()
    rows = (tmp_path / 'fixture.jsonl').read_text().splitlines()
    assert len(rows) == 1
    record = json.loads(rows[0])
    assert record['physics_step'] == 8 and record['terminal']
    assert record['support_qualification'] is None
    assert record['evidence_scope'] == 'native_runtime_capture_unqualified'


def test_cold_trial_metadata_never_moves_physics_and_episode_clocks_are_unique():
    from go2_pvcnn.control.dh116.tabletop_stereo_native import NativeTensorBridge
    from go2_pvcnn.control.dh116.tabletop_stereo_contract import EpisodeIdentities
    bridge = NativeTensorBridge.__new__(NativeTensorBridge)
    bridge.n = 2; bridge.policy_tick = 0; bridge._in_step = False
    bridge.identities = EpisodeIdentities(2, run_uuid='cpu-cold-fixture')
    bridge._seen_trial_ids = set(); bridge._cold_trials = None
    bridge._trial_reset_counts = [0,0]
    bridge._initial_conditions = {}; bridge._initial_sha = {}; bridge._initial_clock = {}
    bridge._initial_policy_tick = {}
    bridge.world = SimpleNamespace(sim=SimpleNamespace(current_time=0.))
    wp = torch.tensor([[0.,0.,.36],[1.,0.,.36]])
    bridge.device_world = SimpleNamespace(origins=torch.tensor([[0.,0.,0.],[1.,0.,0.]]),
        wp=wp.clone(), wq=torch.tensor([[1.,0.,0.,0.]]).expand(2,-1), wv=torch.zeros(2,6), _physics_steps=0)
    palms = torch.zeros(2,2,7); palms[:,:,3] = 1.
    palms[1,:,:3] += torch.tensor([1.,0.,0.])
    bridge.palm_poses = lambda: palms.clone()
    styles = ['pinch_thumb_index','tripod_thumb_index_middle']
    trials = [dict(trial_id=f'fixture-{i}', grasp_style=style) for i,style in enumerate(styles)]
    bridge.bind_cold_trials(trials)
    assert torch.equal(bridge.device_world.wp,wp)  # metadata hook is not a reset writer
    measurement = dict(q=torch.zeros(2,12),qd=torch.zeros(2,12))
    bridge._physical_reset(torch.tensor([0,1]), measurement)
    first = bridge.pre_action_identities('a'*64)
    assert [row['episode_policy_tick'] for row in first] == [0,0]
    assert [row['trial_episode_index'] for row in first] == [1,1]
    assert first[0]['physical_initial_condition_sha256'] == first[1]['physical_initial_condition_sha256']
    bridge.policy_tick = 5; bridge._in_step = True
    bridge.world.sim.current_time = 48/480; bridge.device_world._physics_steps = 48
    bridge._physical_reset(torch.tensor([0]), measurement)
    bridge._in_step = False; bridge.policy_tick = 6
    after = bridge.pre_action_identities('a'*64)
    assert after[0]['episode_uid'] != first[0]['episode_uid']
    assert after[0]['episode_policy_tick'] == 0 and after[0]['trial_episode_index'] == 2
    assert after[0]['source_episode_group'] == 'fixture-0'
    with pytest.raises(ValueError,match='unvisited'):
        bridge.bind_cold_trials(trials)
