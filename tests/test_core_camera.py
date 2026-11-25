import copy
from types import SimpleNamespace
import pytest, torch
from dh116.physics.vision import StereoCalibration as StereoCalibration
from dh116.env.cameras import TiledStereoCapture

def harness(*, reset_epoch=False):
    k = [[40.0, 0.0, 16.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]]
    left = torch.eye(4, dtype=torch.float64)
    right = left.clone()
    right[0, 3] = 0.1
    cal = StereoCalibration(k, k, left, right, 32, 24, 0.1)
    state = dict(run_uuid='run', env_id=0, episode_uid='ep', reset_id=1, policy_tick=3, physics_tick_start=4, physics_tick_end=4, rod_step=4, sdk_step=4, sdk_timestamp_s=1.0, rod_tick=4, solver_time_s=0.4, rod_uid='rod', material_reference_uid='material', state_sha256='a' * 64, simulation_timestamp=1.0, calibration_sha256=cal.sha256)

    reset_time = 2 / 480 if reset_epoch else 0.0
    if reset_epoch:
        state.update(sdk_step=2, sdk_timestamp_s=reset_time, rod_step=2, rod_tick=0, policy_tick=0,
                     physics_tick_start=2, physics_tick_end=2, solver_time_s=0.0, simulation_timestamp=0.0)

    class Sim:
        current_time = state['sdk_timestamp_s']
        current_time_step_index = state['sdk_step']
        epoch = 10

        def render(self):
            self.epoch += 1
    sim = Sim()

    class Reader:
        closes = 0
        raw_time_drift = 0.0

        def read_clock(self):
            return dict(frame_id=sim.epoch, simulation_time=sim.current_time - reset_time + self.raw_time_drift,
                        render_frame_number=sim.epoch)

        def read_pair(self):
            return {side: dict(rgb=torch.full((1, 3, 24, 32), value), K=getattr(cal, 'K_' + side)[None].clone(), T_world=getattr(cal, 'T_world_' + side)[None].clone(), raw_clock=self.read_clock(), native_evidence=False) for (side, value) in (('left', 0.2), ('right', 0.7))}

        def close(self):
            self.closes += 1

    class Setting:
        value = False

        def get(self):
            return self.value

        def set(self, value):
            self.value = value
    reader = Reader()
    setting = Setting()
    options = {'initial_reset_clock': True} if reset_epoch else {}
    capture = TiledStereoCapture(sim, reader, cal, lambda: copy.deepcopy(state),
                                 render_setting=setting, max_render_attempts=2, **options)
    return capture, sim, reader, state, setting

def test_both_fixed_cameras_share_one_render_keep_raw_clock_and_own_images():
    (cap, sim, reader, state, setting) = harness()
    packet = cap.capture()
    assert sim.epoch == 11 and packet['render_passes'] == 1
    assert packet['render_timing']['raw_left_clock'] == packet['render_timing']['raw_right_clock']
    assert packet['render_timing']['physical_timestamp_s'] == 1.0
    assert packet['render_timing']['native_synchronization_qualified'] is False
    assert packet['source_scope'] == 'cpu_fixture' and packet['identity'] == state
    assert (packet['left_rgb'] == 0.2).all() and (packet['right_rgb'] == 0.7).all()
    assert setting.get() is False
    cap.close()
    cap.close()
    assert reader.closes == 1 and cap.sim is None and (cap.snapshot is None)

@pytest.mark.parametrize('fault', ['clock', 'right_pose', 'physical', 'depth'])
def test_fixed_capture_rejects_bad_pair_without_committing_identity(fault):
    (cap, sim, reader, state, setting) = harness()
    original = reader.read_pair

    def corrupt():
        pair = original()
        if fault == 'clock':
            pair['right']['raw_clock']['render_frame_number'] -= 1
        elif fault == 'right_pose':
            pair['right']['T_world'][:, 0, 3] += 0.01
        elif fault == 'physical':
            state['reset_id'] += 1
        else:
            pair['left']['depth'] = torch.ones(1)
        return pair
    reader.read_pair = corrupt
    with pytest.raises((RuntimeError, ValueError)):
        cap.capture()
    assert cap.last_capture_id == 0 and setting.get() is False

def test_reset_changes_identity_without_reusing_previous_epoch():
    (cap, sim, reader, state, setting) = harness()
    assert cap.capture()['render_timing']['capture_id'] == 1
    state['reset_id'] += 1
    state['episode_uid'] = 'new'
    sim.epoch = 0
    assert cap.capture()['render_timing']['capture_id'] == 2

def test_stereo_rejects_pair_from_a_shared_render_with_wrong_physical_time():
    cap, _, reader, _, setting = harness()
    reader.raw_time_drift = 0.000002
    with pytest.raises(ValueError, match='physical timestamp'):
        cap.capture()
    assert cap.last_capture_id == 0 and setting.get() is False


def test_stereo_anchors_raw_zero_at_the_initial_sdk_reset_and_keeps_origin_across_lane_reset():
    cap, sim, reader, state, _ = harness(reset_epoch=True)
    reset_time = sim.current_time
    reader.raw_time_drift = 0.000002
    with pytest.raises(ValueError, match='physical timestamp'):
        cap.capture()
    reader.raw_time_drift = 0.0
    first = cap.capture()['render_timing']
    assert first['physical_timestamp_s'] == first['raw_left_clock']['simulation_time'] == 0.0
    assert first['absolute_sdk_timestamp_s'] == first['reset_origin_sdk_timestamp_s'] == reset_time
    assert first['reset_origin_sdk_step'] == 2

    sim.current_time_step_index = 10
    sim.current_time = reset_time + 8 / 480
    state.update(reset_id=2, episode_uid='selected_lane_reset', policy_tick=1, sdk_step=10, rod_step=10,
                 physics_tick_start=10, physics_tick_end=10, sdk_timestamp_s=sim.current_time,
                 simulation_timestamp=sim.current_time - reset_time, solver_time_s=8 / 480)
    second = cap.capture()['render_timing']
    assert second['physical_timestamp_s'] == pytest.approx(8 / 480)
    assert second['raw_right_clock']['simulation_time'] == second['physical_timestamp_s']
    assert second['absolute_sdk_timestamp_s'] == sim.current_time
    assert second['reset_origin_sdk_step'] == 2 and second['reset_origin_sdk_timestamp_s'] == reset_time


def test_student_fixed_rig_centers_on_supported_rod_in_native_world():
    import json

    from dh116.env.contracts import PROJECT_ROOT
    from dh116.physics.vision import stere_calib__rerender_calibration

    config = json.loads((PROJECT_ROOT / 'configs/cameras/student_stereo.json').read_text())
    cal = StereoCalibration.from_dict(config)
    fov = config['horizontal_fov_degrees']
    assert fov == 86.0
    assert cal.sha256 == stere_calib__rerender_calibration(
        cal, cal.width, cal.height, horizontal_fov_degrees=fov,
    ).sha256
    left, right = cal.T_world_left, cal.T_world_right
    center = torch.tensor([0.43, 0.0, 0.365], dtype=torch.float64)
    eye_mid = center + center.new_tensor([0.32, 0.0, 0.42])
    target = center + center.new_tensor([0.0, 0.0, 0.02])
    torch.testing.assert_close(left[:3, 3], eye_mid + center.new_tensor([0.0, -0.04, 0.0]))
    torch.testing.assert_close(right[:3, 3], eye_mid + center.new_tensor([0.0, 0.04, 0.0]))
    torch.testing.assert_close(left[:3, 0], center.new_tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(left[:3, 2], (target - eye_mid) / (target - eye_mid).norm())
    assert cal.rectified and (cal.width, cal.height) == (640, 480) and cal.baseline_m == 0.08


def test_proof_producer_binds_current_api_and_rejects_arbitrary_source():
    from dh116.rl import qualification as module
    from dh116.env.contracts import PROJECT_ROOT
    runner=PROJECT_ROOT/"src/dh116/rl/training.py"
    source=PROJECT_ROOT/"src/dh116/env/cameras.py"
    command=["python",str(PROJECT_ROOT/"scripts/train_teacher.py"),"--worker"]
    assert module.current_producer_api(runner,source,command,stereo=True)=="dh116.env.cameras.TiledStereoCapture.capture"
    with pytest.raises(ValueError):
        module.current_producer_api(source,source,command,stereo=True)
    with pytest.raises(ValueError):
        module.current_producer_api(runner,runner,command,stereo=True)


def test_preview_mesh_uses_all_measured_node_centres_and_the_contact_radius():
    import numpy as np
    from dh116.env import cameras

    y = np.linspace(-0.15, 0.15, 41)
    nodes = np.stack((0.1 * y**2, y, 0.36 + 0.02 * np.sin(10.0 * y)), axis=1)
    points, faces = cameras.rod_preview_mesh(nodes, 0.005)
    rings = points.reshape(41, -1, 3)
    np.testing.assert_allclose(rings.mean(axis=1), nodes, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(rings - nodes[:, None], axis=-1), 0.005, atol=1e-7)
    assert len(faces) == 40 * rings.shape[1] + 2
    assert {vertex for face in faces for vertex in face} == set(range(len(points)))
    nodes[20] = nodes[19]
    with pytest.raises(ValueError, match='degenerate'):
        cameras.rod_preview_mesh(nodes, 0.005)


def test_student_replay_carrier_retains_native_camera_clock_and_sensor_fields():
    from dh116.env import cameras
    from dh116.rl.training import RAW_FIELDS

    cap, _, _, _, _ = harness()
    packet = cap.capture()
    proprio = torch.zeros(1, 62)
    proprio[0, 27] = proprio[0, 34] = 1.0
    proprio[0, 38] = 0.5

    packet['source_scope'] = 'cpu_fixture'
    with pytest.raises(ValueError, match='native'):
        cameras.student_rgb_frame(packet, proprio)
    packet['source_scope'] = 'native_tiled_reader'
    timings = packet['render_timing']
    clocks = [timings[name] for name in ('raw_left_clock', 'raw_right_clock')]
    for clock in clocks:
        clock['simulation_time'] = 1.0 - 2.0 / 480
    with pytest.raises(ValueError, match='physical timestamp'):
        cameras.student_rgb_frame(packet, proprio)
    render_time = 1.0 + 0.5e-6
    for clock in clocks:
        clock['simulation_time'] = render_time
    raw = cameras.student_rgb_frame(packet, proprio)
    assert raw.keys() == RAW_FIELDS
    assert raw['frame_id'] == packet['render_timing']['raw_left_clock']['render_frame_number']
    assert raw['left_timestamp_s'] == raw['right_timestamp_s'] == raw['now_timestamp_s'] == render_time
    assert torch.equal(raw['proprio'], proprio)
    packet['left_rgb'].zero_()
    proprio.zero_()
    assert (raw['left_rgb'] == 0.2).all() and raw['proprio'][0, 38] == 0.5
    packet['render_timing']['raw_right_clock']['render_frame_number'] -= 1
    with pytest.raises(ValueError, match='shared render'):
        cameras.student_rgb_frame(packet, raw['proprio'])


@pytest.mark.parametrize('checkpoint_lanes', [2, 512, 1024, 2048])
def test_replay_rebinds_only_actor_batch_and_rejects_another_action_contract(checkpoint_lanes):
    from dh116.env.contracts import action_sha256, runti_io__digest
    from scripts import train_teacher

    def identity(lanes, action):
        fields = dict(parent_num_envs=lanes, action_contract_sha256=action, grasp='pinch')
        return dict(fields, model_identity_sha256=runti_io__digest(fields))

    actor_names = ('actor', 'actor_rod', 'actor_context', 'actor_head')
    modules = {name: torch.nn.Linear(1, 1) for name in (*actor_names, 'critic')}
    model = torch.nn.ModuleDict(modules)
    model.model_identity = identity(2, action_sha256())
    checkpoint_model = identity(checkpoint_lanes, action_sha256())
    policy = {
        f'{name}.{field}': torch.ones_like(value)
        for name in actor_names
        for field, value in model[name].state_dict().items()
    }
    policy['_extra_state'] = checkpoint_model
    critic_before = model['critic'].weight.detach().clone()
    train_teacher.freeze_replay_actor(model, policy, checkpoint_model)
    assert all((model[name].weight == 1).all() for name in actor_names)
    assert torch.equal(model['critic'].weight, critic_before)
    assert not model.training and not any(p.requires_grad for p in model.parameters())

    invalid = identity(checkpoint_lanes, 'f' * 64)
    with pytest.raises(ValueError, match='action'):
        train_teacher.freeze_replay_actor(model, dict(policy, _extra_state=invalid), invalid)


def test_preview_peak_lift_excludes_extra_link_contact_and_clamped_reaction():
    from dh116.rl.reward import TensorTrainingReward
    from scripts import train_teacher

    reward = TensorTrainingReward.__new__(TensorTrainingReward)
    yes, no = torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)
    values = dict(coverage_left=yes.float(), coverage_right=yes.float(),
                  opposed_left=yes, opposed_right=yes, bilateral_support=yes, off_table=yes,
                  no_external_support=yes, prohibited_support=no, constraint_failure=no, numerical_health=yes,
                  soft_non_target_contact=torch.tensor([0.0, 0.25, 0.0]),
                  soft_reaction_force=torch.tensor([0.99, 0.2, 1.0]))
    raw_lift = torch.tensor([0.02, 0.03, 0.04])
    qualified = train_teacher.replay_detached_lift(reward, values, raw_lift)
    torch.testing.assert_close(qualified, torch.tensor([0.02, 0.0, 0.0]))
    torch.testing.assert_close(raw_lift, torch.tensor([0.02, 0.03, 0.04]))
