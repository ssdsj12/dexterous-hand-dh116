"""Synthetic camera/physics protocol integration; never native evidence."""
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_contract import EpisodeIdentities
from go2_pvcnn.control.dh116.tabletop_stereo_runtime import NativeStereoEnvironment


class FixtureBridge:
    evidence_scope = 'cpu_fixture'
    def __init__(self):
        self.n, self.device = 1, torch.device('cpu')
        self.identities = EpisodeIdentities(1, run_uuid='cpu-fixture')
        self.previous_issued = torch.zeros(1, 24)
        self.policy_tick = 0
        self.critic = torch.zeros(1, 371)
        self.world = SimpleNamespace(sim=SimpleNamespace(current_time=0.))
        data = SimpleNamespace(joint_pos_target=torch.zeros(1, 6))
        self.device_world = SimpleNamespace(_physics_steps=0, origins=torch.zeros(1, 3),
            hands=[SimpleNamespace(data=data)]*2, w=SimpleNamespace(ids=[list(range(6))]*2))
        self.task = SimpleNamespace(command=torch.zeros(1, 12))
        self.output = __import__('pathlib').Path('.')

    def set_execution_context(self, **kwargs):
        self.context = kwargs

    def set_action_owners(self, owners):
        self.owners = owners

    def pre_action_identities(self, calibration):
        tick = self.device_world._physics_steps
        return [dict(run_uuid='cpu-fixture', env_id=0, episode_uid=self.identities.uid(0),
            reset_id=self.identities.reset_ids[0], policy_tick=self.policy_tick,
            physics_tick_start=tick, physics_tick_end=tick, simulation_timestamp=self.world.sim.current_time,
            calibration_sha256=calibration, physical_initial_condition_sha256='a'*64,
            source_episode_group=self.identities.uid(0))]

    def proprioception(self):
        p = torch.zeros(1, 62); p[:, 27] = 1; p[:, 34] = 1
        p[:, 38:] = self.previous_issued
        return p

    def palm_poses(self):
        p = torch.zeros(1, 2, 7); p[:, :, 3] = 1
        return p

    def step(self, actions):
        self.policy_tick += 1
        self.device_world._physics_steps += 8
        self.world.sim.current_time += 1/60
        done = torch.tensor([self.policy_tick == 2])
        self.previous_issued = torch.where(done[:, None], 0., actions)
        if bool(done[0]): self.identities.reset([0])
        reward = -actions.square().mean(-1)
        return self.critic.clone(), reward, done, dict(success=torch.zeros(1, dtype=torch.bool),
            failure=done.long(), terms={'fixture': reward}, recovered=torch.zeros(1, dtype=torch.bool))


class FixtureCamera:
    def __init__(self, bridge):
        self.bridge = bridge; self.frame = 0
        self.calibration = SimpleNamespace(sha256='b'*64)

    def sample(self, *, identity):
        self.frame += 1
        rows = [dict(row, frame_id=self.frame) for row in identity]
        return dict(left_rgb=torch.zeros(1, 3, 32, 32), right_rgb=torch.zeros(1, 3, 32, 32),
            depth_m=torch.full((1, 32, 32), float('nan')),
            valid=torch.zeros(1, 32, 32, dtype=torch.bool), confidence=torch.zeros(1, 32, 32),
            disparity_px=torch.full((1, 32, 32), float('nan')), K_left=torch.eye(3),
            T_world_left=torch.eye(4), identity=rows, native_evidence=False)


def test_protocol_caches_exact_pre_state_and_never_leaks_truth_to_sensor(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_stereo_runtime as runtime
    bridge = FixtureBridge(); camera = FixtureCamera(bridge)
    monkeypatch.setattr(runtime.StereoCamera, 'from_native', lambda *a, **k: camera)
    from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration, rerender_calibration
    from pathlib import Path
    import json
    calibration = rerender_calibration(StereoCalibration.from_json(Path(__file__).resolve().parents[1]
        / 'configs/dh116_training/stereo_camera_v1.json'), 32, 32)
    config = tmp_path / 'camera.json'; config.write_text(json.dumps(calibration.to_dict()))
    camera.calibration = calibration
    env = NativeStereoEnvironment(bridge, config)
    before = env.snapshot(); again = env.snapshot()
    assert camera.frame == 1
    torch.testing.assert_close(before.sensor_history.proprio, again.sensor_history.proprio)
    assert not before.privileged_valid.any()
    assert env.evidence_scope == 'cpu_fixture'
    bridge.critic[:] = 9.
    assert not before.privileged.any()  # owned prior observation
    assert set(before.sensor_history.as_dict()).isdisjoint({'privileged', 'success', 'phase'})
    action = torch.full((1, 24), .2)
    result = env.step(action)
    assert camera.frame == 2 and not result.done.any()
    assert torch.equal(result.next_state.sensor_history.proprio[:, -1, 38:], action)
    result = env.step(action)
    assert result.done.all() and int(result.next_state.sensor_history.history_valid.sum()) == 1
    assert not result.next_state.sensor_history.proprio[:, -1, 38:].any()
    assert result.next_state.identities[0]['episode_uid'] != before.identities[0]['episode_uid']
    assert result.info['execution'][0]['physics_tick_end'] == 16
    assert result.info['evidence_scope'] == 'cpu_fixture'
    assert not result.applied_targets['post_step_readback_valid'].any()
    assert torch.isnan(result.applied_targets['actual_measured_palm_pose_wxyz']).all()
    assert torch.isnan(result.applied_targets['actual_sdk_joint_target_rad']).all()
