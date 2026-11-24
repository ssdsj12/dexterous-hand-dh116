import copy
from types import SimpleNamespace
import pytest, torch
from dh116.physics.vision import StereoCalibration as StereoCalibration
from dh116.env.cameras import TiledStereoCapture

def harness():
    k = [[40.0, 0.0, 16.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]]
    left = torch.eye(4, dtype=torch.float64)
    right = left.clone()
    right[0, 3] = 0.1
    cal = StereoCalibration(k, k, left, right, 32, 24, 0.1)
    state = dict(run_uuid='run', env_id=0, episode_uid='ep', reset_id=1, policy_tick=3, physics_tick_start=4, physics_tick_end=4, rod_step=4, sdk_step=4, sdk_timestamp_s=1.0, rod_tick=4, solver_time_s=0.4, rod_uid='rod', material_reference_uid='material', state_sha256='a' * 64, simulation_timestamp=1.0, calibration_sha256=cal.sha256)

    class Sim:
        current_time = 1.0
        current_time_step_index = 4
        epoch = 10

        def render(self):
            self.epoch += 1
    sim = Sim()

    class Reader:
        closes = 0

        def read_clock(self):
            return dict(frame_id=sim.epoch, simulation_time=sim.epoch / 30.0, render_frame_number=sim.epoch)

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
    return (TiledStereoCapture(sim, reader, cal, lambda : copy.deepcopy(state), render_setting=setting, max_render_attempts=2), sim, reader, state, setting)

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
