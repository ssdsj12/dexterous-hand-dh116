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
        p = torch.zeros(self.n, 2, 7); p[:, :, 3] = 1
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
        n=self.bridge.n
        rows = [dict(row, frame_id=self.frame) for row in identity]
        return dict(left_rgb=torch.zeros(n, 3, 32, 32), right_rgb=torch.zeros(n, 3, 32, 32),
            depth_m=torch.full((n, 32, 32), float('nan')),
            valid=torch.zeros(n, 32, 32, dtype=torch.bool), confidence=torch.zeros(n, 32, 32),
            disparity_px=torch.full((n, 32, 32), float('nan')), K_left=torch.eye(3),
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


class EncoderFixtureBridge(FixtureBridge):
    """Known CPU SDK-shaped readback against the unchanged actual observe kernel."""
    def __init__(self):
        super().__init__()
        from phase0_state_contract_fixture import cpu_task
        self.task=cpu_task();self.n=self.task.n
        self.task.m['q'][1]+=.05*(self.task.high-self.task.low)
        self.critic=self.task.observe()
        self.identities=EpisodeIdentities(self.n,run_uuid='CPU-encoder-fixture')
        self.previous_issued=torch.zeros(self.n,24)
        hands=[];indices=[5,2,4,1,3,0]
        for side in range(2):
            q=torch.zeros(self.n,7);qd=torch.zeros_like(q)
            q[:,indices]=self.task.m['q'][:,side*6:(side+1)*6].float()
            qd[:,indices]=self.task.m['qd'][:,side*6:(side+1)*6].float()
            hands.append(SimpleNamespace(data=SimpleNamespace(joint_pos=q,joint_vel=qd)))
        self.device_world.hands=hands
        self.device_world.w.ids=[indices.copy(),indices.copy()]
        self.device_world.origins=torch.zeros(self.n,3)

    def proprioception(self):
        from go2_pvcnn.control.dh116.tabletop_stereo_native import NativeTensorBridge
        return NativeTensorBridge.proprioception(self)

    def pre_action_identities(self,calibration):
        tick=self.device_world._physics_steps
        return [dict(run_uuid=self.identities.run_uuid,env_id=i,episode_uid=self.identities.uid(i),
            reset_id=self.identities.reset_ids[i],policy_tick=self.policy_tick,
            physics_tick_start=tick,physics_tick_end=tick,simulation_timestamp=self.world.sim.current_time,
            calibration_sha256=calibration,physical_initial_condition_sha256='a'*64,
            source_episode_group=self.identities.uid(i)) for i in range(self.n)]


@pytest.fixture
def encoder_runtime(tmp_path,monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_stereo_runtime as runtime
    from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration,rerender_calibration
    from pathlib import Path
    import json
    bridge=EncoderFixtureBridge();camera=FixtureCamera(bridge)
    calibration=rerender_calibration(StereoCalibration.from_json(Path(__file__).resolve().parents[1]
        /'configs/dh116_training/stereo_camera_v1.json'),32,32)
    config=tmp_path/'encoder_camera.json';config.write_text(json.dumps(calibration.to_dict()))
    camera.calibration=calibration
    monkeypatch.setattr(runtime.StereoCamera,'from_native',lambda *a,**k:camera)
    return NativeStereoEnvironment(bridge,config),bridge,camera


def test_verified_simulator_encoder_labels_have_24_dimension_loss_coverage_only(encoder_runtime):
    from go2_pvcnn.control.dh116.tabletop_stereo_losses import StereoEstimationLoss
    env,bridge,_=encoder_runtime;state=env.snapshot()
    assert state.privileged_valid[:,:24].all() and not state.privileged_valid[:,24:].any()
    assert state.state_sources['proprio_source_confirmed'].shape==(2,371)
    assert torch.equal(state.state_sources['proprio_source_confirmed'],state.privileged_valid)
    for key in ('visibility_valid','correspondence_valid','contact_valid','feedback.valid'):
        assert not state.state_sources[key].any()
    assert state.extra_labels['training_label_source_scope']=='simulation_SDK_encoder_readback_only'
    assert state.extra_labels['hardware_feedback_qualified'] is False
    assert env.evidence_scope=='cpu_fixture' and state.force_qualification is None
    assert set(state.sensor_history.as_dict()).isdisjoint({'privileged','training_label_source_scope','hardware_feedback_qualified'})
    prediction=torch.zeros_like(state.privileged,requires_grad=True)
    loss=StereoEstimationLoss()({'privileged_state':prediction},{'privileged_state':state.privileged},
        {'privileged_state':state.privileged_valid,'state_sources':state.state_sources})
    assert loss['coverage']['privileged_state']==pytest.approx(24/371)
    assert float(loss['total'].detach())>0
    loss['total'].backward()
    assert prediction.grad[:,:24].abs().sum()>0 and not prediction.grad[:,24:].any()


@pytest.mark.parametrize('index',[7,19])
def test_one_mismatched_encoder_label_masks_only_that_environment_dimension(encoder_runtime,index):
    env,bridge,_=encoder_runtime;bridge.critic[0,index]+=.01
    state=env.snapshot()
    assert int(state.privileged_valid.sum())==47
    assert not state.privileged_valid[0,index] and state.privileged_valid[1,:24].all()


@pytest.mark.parametrize('field,value',[('env_id',7),('episode_uid','stale-CPU-episode'),('physics_tick_end',8)])
def test_encoder_supervision_does_not_borrow_another_environment_or_tick(encoder_runtime,field,value):
    env,_,camera=encoder_runtime;sample=camera.sample
    def shifted(**kwargs):
        packet=sample(**kwargs);packet['identity'][0][field]=value;return packet
    camera.sample=shifted
    state=env.snapshot()
    assert not state.privileged_valid[0].any() and state.privileged_valid[1,:24].all()


@pytest.mark.parametrize('mode',['equal','reversed','nonfinite'])
def test_invalid_task_joint_normalization_bounds_never_enable_labels(encoder_runtime,mode):
    env,bridge,_=encoder_runtime
    bridge.task.low[0]=bridge.task.high[0]+(1. if mode=='reversed' else 0.)
    if mode=='nonfinite':bridge.task.low[0]=float('nan')
    state=env.snapshot()
    assert not state.privileged_valid[:,[0,12]].any()
    assert int(state.privileged_valid.sum())==44


def test_missing_task_limits_keeps_simulator_encoder_supervision_unavailable(encoder_runtime):
    env,bridge,_=encoder_runtime;del bridge.task.low
    assert not env.snapshot().privileged_valid.any()


def test_runtime_converts_visible_rgb_points_to_same_sample_measured_palms_without_filling_occlusion(encoder_runtime):
    import math
    env,bridge,camera=encoder_runtime
    poses=torch.tensor([[[1.,2.,3.,1.,0.,0.,0.],[-2.,1.,4.,math.sqrt(.5),0.,0.,math.sqrt(.5)]]]).repeat(2,1,1)
    bridge.palm_poses=lambda:poses.clone()
    sample=camera.sample
    def visible(**kwargs):
        packet=sample(**kwargs)
        packet['left_rgb'][:,0,16,3:13]=.8
        packet['left_rgb'][:,1,16,3:13]=.4
        packet['left_rgb'][:,2,16,3:13]=.1
        packet['depth_m'][0,16,3:13]=1.
        packet['valid'][0,16,3:13]=True
        return packet
    camera.sample=visible
    state=env.snapshot();labels=state.extra_labels
    points=state.sensor_history.points_world[:,-1]
    mask=labels['visible_wire_points_palms_valid'];local=labels['visible_wire_points_palms_m']
    assert local.shape==(2,2,128,3) and mask.shape==(2,2,128)
    assert mask[0].sum()==20 and not mask[1].any()
    assert torch.isnan(local[~mask]).all()
    visible_mask=state.sensor_history.point_valid[0,-1]
    delta_left=points[0,visible_mask]-poses[0,0,:3]
    torch.testing.assert_close(local[0,0,visible_mask],delta_left)
    delta_right=points[0,visible_mask]-poses[0,1,:3]
    expected_right=torch.stack((delta_right[:,1],-delta_right[:,0],delta_right[:,2]),-1)
    torch.testing.assert_close(local[0,1,visible_mask],expected_right,atol=5e-6,rtol=1e-6)
    # The Actor contract retains the original WORLD points and exact FK input.
    torch.testing.assert_close(points,env.last_camera_packet['visible_wire_points_world_m'],equal_nan=True)
    torch.testing.assert_close(state.sensor_history.proprio[:,-1,24:38].reshape(2,2,7),poses)
    assert not state.state_sources['visibility_valid'].any()
    assert labels['visible_wire_geometry_is_ground_truth'] is False
    assert 'visible_wire_points_palms_m' not in state.label_validity
    torch.testing.assert_close(env.last_visible_geometry['visible_wire_points_palms_m'],local,equal_nan=True)
