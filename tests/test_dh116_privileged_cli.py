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
        surface_reference_bottom_m=torch.tensor([.36]),
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
