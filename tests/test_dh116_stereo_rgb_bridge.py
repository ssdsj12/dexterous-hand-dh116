import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_rgb_bridge import RGBPairBridge


def calibration():
    k = [[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]
    left = torch.eye(4, dtype=torch.float64)
    right = left.clone()
    right[0, 3] = .1
    return StereoCalibration(k, k, left, right, 640, 480, .1)


def state(c):
    return dict(run_uuid="run", env_id=0, episode_uid="ep", reset_id=1,
                policy_tick=3, physics_tick_start=4, physics_tick_end=4,
                rod_step=4, sdk_step=4, sdk_timestamp_s=1., rod_tick=4, solver_time_s=.4,
                rod_uid="rod", material_reference_uid="material", state_sha256="a"*64, simulation_timestamp=1.,
                calibration_sha256=c.sha256)


class Sim:
    current_time = 1.
    current_time_step_index = 4
    def render(self):
        pass


class RenderSetting:
    def __init__(self, value=False):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class Reader:
    def __init__(self, c):
        self.c = c
    def read(self):
        c = self.c
        image = torch.zeros((1, 3, c.height, c.width))
        return dict(left_rgb=image, right_rgb=image.clone(), left_frame_id=1, right_frame_id=1,
                    left_simulation_time=1., right_simulation_time=1.,
                    K_left=c.K_left[None], K_right=c.K_right[None],
                    T_world_left=c.T_world_left[None], T_world_right=c.T_world_right[None],
                    render_timing=dict(frame_id=1, simulation_time=1., render_frame_number=9))


def test_capture_binds_only_rgb_and_validated_identity():
    c = calibration()
    bridge = RGBPairBridge(Sim(), Reader(c), c, lambda: state(c), render_setting=RenderSetting())
    packet = bridge.capture()
    assert packet["identity"]["rod_step"] == 4
    assert packet["calibration_sha256"] == c.sha256
    assert set(packet) == {"left_rgb", "right_rgb", "identity", "frame_id", "simulation_time", "calibration_sha256", "K_left", "K_right", "render_timing", "render_passes"}
    with pytest.raises((ValueError, RuntimeError)):
        bridge.capture()


@pytest.mark.parametrize("mutation", ["physics", "rod", "timestamp", "sha", "depth", "frame", "intrinsics"])
def test_rejects_changed_or_invalid_capture(mutation):
    c = calibration()
    sim = Sim()
    reader = Reader(c)
    original = reader.read
    snapshot = state(c)
    if mutation == "sha":
        snapshot["calibration_sha256"] = "b"*64
    if mutation in ("physics", "rod"):
        def render():
            snapshot["physics_tick_end" if mutation == "physics" else "rod_step"] += 1
        sim.render = render
    def read():
        p = original()
        if mutation == "timestamp": p["right_simulation_time"] = 1.1
        if mutation == "depth": p["depth"] = torch.ones(1)
        if mutation == "frame": p["render_timing"]["frame_id"] = 2
        if mutation == "intrinsics": p["K_left"] = p["K_left"] + 1
        return p
    reader.read = read
    with pytest.raises((ValueError, RuntimeError)):
        RGBPairBridge(sim, reader, c, lambda: snapshot, render_setting=RenderSetting()).capture()



def test_native_factory_authors_default_lens_and_forwards_limits(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_stereo_rgb_owned_native as native
    seen = {}
    def create(sim, origins, c, device, **kwargs):
        seen.update(calibration=c, device=device, **kwargs)
        return Reader(c)
    monkeypatch.setattr(native, "OwnedNativeRGBPairReader", create)
    c = calibration()
    bridge = RGBPairBridge.from_native(Sim(), [[0., 0., 0.]], c, lambda: state(c), render_setting=RenderSetting())
    authored = seen["calibration"]
    assert (authored.width, authored.height) == (640, 480)
    assert authored.K_left[0, 0].item() == pytest.approx(320 / __import__("math").tan(__import__("math").radians(37.5)))
    assert authored.K_left[0, 2] == 320 and authored.K_left[1, 2] == 240
    assert bridge.calibration.sha256 == authored.sha256
    assert seen["max_readback_bytes"] == 1024**3


def test_rejects_state_change_during_read():
    c = calibration()
    row = state(c)
    reader = Reader(c)
    read = reader.read
    def changed():
        p = read()
        row["reset_id"] += 1
        return p
    reader.read = changed
    with pytest.raises(RuntimeError):
        RGBPairBridge(Sim(), reader, c, lambda: row, render_setting=RenderSetting()).capture()


def test_render_metadata_is_whitelisted():
    c = calibration()
    reader = Reader(c)
    read = reader.read
    def extra():
        p = read()
        p["render_timing"]["ground_truth"] = "secret"
        return p
    reader.read = extra
    packet = RGBPairBridge(Sim(), reader, c, lambda: state(c), render_setting=RenderSetting()).capture()
    assert set(packet["render_timing"]) == {"frame_id", "simulation_time", "render_frame_number"}


@pytest.mark.parametrize("key", ["sdk_step", "sdk_timestamp_s", "rod_tick", "solver_time_s",
                                  "material_reference_uid", "rod_uid"])
def test_missing_raw_clock_or_material_identity_rejected(key):
    c = calibration()
    row = state(c)
    del row[key]
    with pytest.raises(ValueError):
        RGBPairBridge(Sim(), Reader(c), c, lambda: row, render_setting=RenderSetting()).capture()


@pytest.mark.parametrize("key", ["sdk_step", "sdk_timestamp_s", "rod_tick", "solver_time_s",
                                  "material_reference_uid", "rod_uid"])
@pytest.mark.parametrize("when", ["render", "read"])
def test_raw_identity_change_is_rejected(key, when):
    c = calibration()
    row = state(c)
    sim = Sim()
    reader = Reader(c)
    def change():
        row[key] = row[key] + (1 if isinstance(row[key], int) else .01 if isinstance(row[key], float) else "2")
    if when == "render":
        sim.render = change
    else:
        original = reader.read
        def read():
            result = original()
            change()
            return result
        reader.read = read
    with pytest.raises(RuntimeError):
        RGBPairBridge(sim, reader, c, lambda: row, render_setting=RenderSetting()).capture()


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_restores_render_setting_on_success_and_exception(initial, fail):
    c = calibration()
    setting = RenderSetting(initial)
    sim = Sim()
    def render():
        setting.value = not initial
        if fail:
            raise RuntimeError("render failed")
    sim.render = render
    bridge = RGBPairBridge(sim, Reader(c), c, lambda: state(c), render_setting=setting)
    if fail:
        with pytest.raises(RuntimeError):
            bridge.capture()
    else:
        bridge.capture()
    assert setting.value is initial


def test_raw_clock_type_change_during_render_is_rejected():
    c = calibration()
    row = state(c)
    sim = Sim()
    def render():
        row["sdk_step"] = float(row["sdk_step"])
    sim.render = render
    with pytest.raises(RuntimeError):
        RGBPairBridge(sim, Reader(c), c, lambda: row, render_setting=RenderSetting()).capture()


def test_setting_restoration_failure_is_not_silent():
    c = calibration()
    setting = RenderSetting(False)
    setting.set = lambda value: None
    sim = Sim()
    sim.render = lambda: setattr(setting, "value", True)
    with pytest.raises(RuntimeError, match="restoration"):
        RGBPairBridge(sim, Reader(c), c, lambda: state(c), render_setting=setting).capture()



def test_sdk_step_advancing_independently_of_snapshot_is_rejected():
    c = calibration()
    sim = Sim()
    sim.render = lambda: setattr(sim, "current_time_step_index", 5)
    with pytest.raises(RuntimeError):
        RGBPairBridge(sim, Reader(c), c, lambda: state(c), render_setting=RenderSetting()).capture()


def test_sdk_step_must_match_snapshot():
    c = calibration()
    sim = Sim()
    sim.current_time_step_index = 5
    with pytest.raises(ValueError):
        RGBPairBridge(sim, Reader(c), c, lambda: state(c), render_setting=RenderSetting()).capture()


def test_fresh_render_epoch_accepts_same_physics_frame_and_new_episode():
    c = calibration()
    row = state(c)
    reader = Reader(c)
    read = reader.read
    epoch = [9]
    def read_epoch():
        p = read()
        p["render_timing"]["render_frame_number"] = epoch[0]
        return p
    reader.read = read_epoch
    bridge = RGBPairBridge(Sim(), reader, c, lambda: row, render_setting=RenderSetting())
    assert bridge.capture()["render_passes"] == 1
    epoch[0] = 10
    bridge.capture()
    with pytest.raises(RuntimeError):
        bridge.capture()
    row["reset_id"] += 1
    row["episode_uid"] = "new"
    epoch[0] = 1
    bridge.capture()


def test_bounded_readiness_attempts_report_passes_and_limit():
    c = calibration()
    sim = Sim()
    calls = []
    sim.render = lambda: calls.append(1)
    reader = Reader(c)
    original = reader.read
    reader.read = lambda: original() if len(calls) == 3 else None
    bridge = RGBPairBridge(sim, reader, c, lambda: state(c), render_setting=RenderSetting(), max_render_attempts=3)
    assert bridge.capture()["render_passes"] == 3
    assert len(calls) == 3
    calls.clear()
    reader.read = lambda: None
    with pytest.raises(RuntimeError, match="unavailable"):
        bridge.capture()
    assert len(calls) == 3


def test_no_retry_after_physics_change():
    c = calibration()
    sim = Sim()
    calls = []
    def render():
        calls.append(1)
        sim.current_time_step_index += 1
    sim.render = render
    bridge = RGBPairBridge(sim, Reader(c), c, lambda: state(c), render_setting=RenderSetting(), max_render_attempts=3)
    with pytest.raises(RuntimeError):
        bridge.capture()
    assert len(calls) == 1



def test_bridge_closes_owned_reader_once_and_refuses_capture_after_close(monkeypatch):
    import go2_pvcnn.control.dh116.tabletop_stereo_rgb_owned_native as native
    c = calibration()
    reader = Reader(c)
    calls = []
    reader.close = lambda: calls.append("closed")
    monkeypatch.setattr(native, "OwnedNativeRGBPairReader", lambda *args, **kwargs: reader)
    bridge = RGBPairBridge.from_native(Sim(), [[0., 0., 0.]], c, lambda: state(c), render_setting=RenderSetting())
    bridge.close()
    bridge.close()
    assert calls == ["closed"]
    with pytest.raises(RuntimeError, match="closed"):
        bridge.capture()



def test_readback_must_advance_from_pre_render_clock():
    c = calibration()
    reader = Reader(c)
    times = iter([dict(frame_id=1, simulation_time=1., render_frame_number=9),
                  dict(frame_id=1, simulation_time=1., render_frame_number=9)])
    reader.read_clock = lambda: next(times)
    with pytest.raises(RuntimeError, match="unavailable"):
        RGBPairBridge(Sim(), reader, c, lambda: state(c), render_setting=RenderSetting()).capture()


def test_readback_clock_after_rgb_must_match_packet():
    c = calibration()
    reader = Reader(c)
    times = iter([dict(frame_id=1, simulation_time=1., render_frame_number=8),
                  dict(frame_id=1, simulation_time=1., render_frame_number=9),
                  dict(frame_id=1, simulation_time=1., render_frame_number=10)])
    reader.read_clock = lambda: next(times)
    with pytest.raises(ValueError, match="clock"):
        RGBPairBridge(Sim(), reader, c, lambda: state(c), render_setting=RenderSetting()).capture()


def test_startup_empty_isaac_times_returns_not_ready():
    import types
    from go2_pvcnn.control.dh116.tabletop_stereo_rgb_owned_native import OwnedNativeRGBPairReader
    reader = object.__new__(OwnedNativeRGBPairReader)
    reader.reference = types.SimpleNamespace(get_data=lambda: {"referenceTimeNumerator": 2, "referenceTimeDenominator": 30})
    reader.times = types.SimpleNamespace(get_data=lambda: {})
    reader.core = types.SimpleNamespace(get_sim_time_at_swh_frame=lambda _: 1.)
    assert reader.read_clock() is None


@pytest.mark.parametrize("times", [
    {"frameNumber": 3},
    {"swhFrameNumber": 1, "frameNumber": 2, "simulationTime": 1.,
     "rationalTimeOfSimNumerator": 3, "rationalTimeOfSimDenominator": 30},
    "corrupt",
])
def test_invalid_or_mismatched_isaac_time_is_not_startup_retry(times):
    import types
    from go2_pvcnn.control.dh116.tabletop_stereo_rgb_owned_native import OwnedNativeRGBPairReader
    reader = object.__new__(OwnedNativeRGBPairReader)
    reader.reference = types.SimpleNamespace(get_data=lambda: {"referenceTimeNumerator": 2, "referenceTimeDenominator": 30})
    reader.times = types.SimpleNamespace(get_data=lambda: times)
    reader.core = types.SimpleNamespace(get_sim_time_at_swh_frame=lambda _: 1.)
    with pytest.raises(ValueError):
        reader.read_clock()


def test_unready_pre_render_clock_uses_budget_without_reading_rgb():
    c = calibration()
    reader = Reader(c)
    clocks = iter([None, dict(frame_id=1, simulation_time=1., render_frame_number=8),
                   dict(frame_id=1, simulation_time=1., render_frame_number=9),
                   dict(frame_id=1, simulation_time=1., render_frame_number=9)])
    reader.read_clock = lambda: next(clocks)
    reads = []
    read = reader.read
    reader.read = lambda: (reads.append(1), read())[1]
    sim = Sim()
    renders = []
    sim.render = lambda: renders.append(1)
    bridge = RGBPairBridge(sim, reader, c, lambda: state(c), render_setting=RenderSetting(), max_render_attempts=2)
    assert bridge.capture()["render_passes"] == 2
    assert len(renders) == 2 and len(reads) == 1



@pytest.mark.parametrize("reference", [
    {"referenceTimeNumerator": -1, "referenceTimeDenominator": 30},
    {"referenceTimeNumerator": 2, "referenceTimeDenominator": 0},
    {"referenceTimeNumerator": 2},
])
def test_invalid_reference_is_not_hidden_by_empty_isaac_times(reference):
    import types
    from go2_pvcnn.control.dh116.tabletop_stereo_rgb_owned_native import OwnedNativeRGBPairReader
    reader = object.__new__(OwnedNativeRGBPairReader)
    reader.reference = types.SimpleNamespace(get_data=lambda: reference)
    reader.times = types.SimpleNamespace(get_data=lambda: {})
    reader.core = types.SimpleNamespace(get_sim_time_at_swh_frame=lambda _: 1.)
    with pytest.raises(ValueError):
        reader.read_clock()


def test_unready_post_render_is_retried_within_budget():
    c = calibration()
    reader = Reader(c)
    good = dict(frame_id=1, simulation_time=1., render_frame_number=9)
    clocks = iter([dict(frame_id=1, simulation_time=1., render_frame_number=7), None,
                   dict(frame_id=1, simulation_time=1., render_frame_number=8), good, good])
    reader.read_clock = lambda: next(clocks)
    sim = Sim()
    renders = []
    sim.render = lambda: renders.append(1)
    bridge = RGBPairBridge(sim, reader, c, lambda: state(c), render_setting=RenderSetting(), max_render_attempts=2)
    assert bridge.capture()["render_passes"] == 2
    assert len(renders) == 2



def test_corrupt_isaac_times_not_hidden_when_reference_unready():
    import types
    from go2_pvcnn.control.dh116.tabletop_stereo_rgb_owned_native import OwnedNativeRGBPairReader
    reader = object.__new__(OwnedNativeRGBPairReader)
    reader.reference = types.SimpleNamespace(get_data=lambda: {})
    reader.times = types.SimpleNamespace(get_data=lambda: {"frameNumber": 3})
    reader.core = types.SimpleNamespace(get_sim_time_at_swh_frame=lambda _: 1.)
    with pytest.raises(ValueError):
        reader.read_clock()
