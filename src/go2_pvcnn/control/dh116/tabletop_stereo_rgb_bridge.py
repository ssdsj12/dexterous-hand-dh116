"""Opt-in synchronized RGB capture tied to caller-owned physical identity.

The snapshot callback must return one plain identity dictionary per environment,
including a trusted state digest and physics/rod clocks. It must be sampled at
the actual control boundary. This adapter does not establish provenance of that
callback and does not own or release native renderer resources.
"""
import copy
import math
import importlib
import re

import torch

from .tabletop_stereo_calibration import StereoCalibration, rerender_calibration
from .tabletop_stereo_validation import validate_identity, validate_images, validate_pair_timing

_IDENTITY_KEYS = frozenset(("run_uuid", "env_id", "episode_uid", "reset_id", "policy_tick",
                            "physics_tick_start", "physics_tick_end", "rod_step",
                            "state_sha256", "simulation_timestamp", "calibration_sha256"))
_VALIDATION_KEYS = frozenset(("run_uuid", "env_id", "episode_uid", "reset_id", "policy_tick",
                              "physics_tick_start", "physics_tick_end", "simulation_timestamp"))
_RAW_KEYS = frozenset(("sdk_step", "sdk_timestamp_s", "rod_tick", "solver_time_s",
                       "material_reference_uid", "rod_uid"))
_PACKET_KEYS = frozenset(("left_rgb", "right_rgb", "left_frame_id", "right_frame_id",
                          "left_simulation_time", "right_simulation_time", "K_left", "K_right",
                          "T_world_left", "T_world_right", "native_evidence", "render_timing"))


def _same_snapshot(a, b):
    # All admitted identity fields are scalars; do not equate e.g. 1 and True.
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(_same_snapshot(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_snapshot(a[k], b[k]) for k in a)
    return a == b


class _NativeRenderSetting:
    """Carb setting access only; must be constructed inside an active Kit process."""
    path = "/app/player/playSimulations"

    def __init__(self):
        self.settings = importlib.import_module("carb.settings").get_settings()

    def get(self):
        return self.settings.get(self.path)

    def set(self, value):
        self.settings.set_bool(self.path, value)


class RGBPairBridge:
    def __init__(self, sim, reader, calibration, snapshot, *, render_setting, origins=None, max_render_attempts=1):
        if not isinstance(calibration, StereoCalibration) or not callable(snapshot):
            raise ValueError("calibration and physical snapshot callback required")
        self.sim, self.reader, self.calibration = sim, reader, calibration
        if not callable(getattr(render_setting, "get", None)) or not callable(getattr(render_setting, "set", None)):
            raise ValueError("render setting get/set required")
        self.snapshot, self.origins, self.last_frame = snapshot, origins, -1
        self.render_setting = render_setting
        if type(max_render_attempts) is not int or not 1 <= max_render_attempts <= 32:
            raise ValueError("max_render_attempts must be an integer from 1 to 32")
        self.max_render_attempts = max_render_attempts
        self.last_epoch = None
        self.last_episode = None
        self._owns_reader = False
        self._closed = False
        self._close_error = None

    @classmethod
    def from_native(cls, sim, origins, calibration, snapshot, *, width=640, height=480,
                    horizontal_fov_degrees=75., device="cuda", max_readback_bytes=1024**3,
                    allow_resource_override=False, render_setting=None, max_render_attempts=1):
        # Author a new lens at the requested resolution; never resize captured pixels.
        from .tabletop_stereo_rgb_owned_native import OwnedNativeRGBPairReader
        if not isinstance(calibration, StereoCalibration) or not callable(snapshot):
            raise ValueError("calibration and physical snapshot callback required")
        if type(max_render_attempts) is not int or not 1 <= max_render_attempts <= 32:
            raise ValueError("max_render_attempts must be an integer from 1 to 32")
        c = rerender_calibration(calibration, width, height,
                                 horizontal_fov_degrees=horizontal_fov_degrees,
                                 allow_resource_override=allow_resource_override)
        setting = _NativeRenderSetting() if render_setting is None else render_setting
        if not callable(getattr(setting, "get", None)) or not callable(getattr(setting, "set", None)):
            raise ValueError("render setting get/set required")
        if type(setting.get()) is not bool:
            raise ValueError("render setting must be a readable boolean")
        reader = OwnedNativeRGBPairReader(sim, origins, c, device,
                                     max_readback_bytes=max_readback_bytes,
                                     allow_resource_override=allow_resource_override)
        try:
            bridge = cls(sim, reader, c, snapshot, origins=origins,
                         render_setting=setting, max_render_attempts=max_render_attempts)
        except BaseException:
            reader.close()
            raise
        bridge._owns_reader = True
        return bridge

    def close(self):
        if self._closed:
            if self._close_error is not None: raise RuntimeError(self._close_error)
            return
        self._closed = True
        try:
            if self._owns_reader:
                self.reader.close()
        except BaseException as exc:
            self._close_error = str(exc)
            raise
        finally:
            self.sim = self.reader = self.snapshot = self.origins = self.render_setting = None

    def capture(self):
        if self._closed:
            raise RuntimeError("RGB bridge is closed")
        original = self.render_setting.get()
        if type(original) is not bool:
            raise ValueError("render setting must be a readable boolean")
        try:
            return self._capture()
        finally:
            self.render_setting.set(original)
            if self.render_setting.get() is not original:
                raise RuntimeError("render setting restoration failed")

    def _capture(self):
        before = copy.deepcopy(self.snapshot())
        rows = before if isinstance(before, list) else [before]
        if not rows or any(not isinstance(r, dict) or set(r) != _IDENTITY_KEYS | _RAW_KEYS for r in rows):
            raise ValueError("physical identity fields are missing or unexpected")
        for row in rows:
            if (type(row["rod_step"]) is not int or row["rod_step"] < 0
                    or not isinstance(row["state_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", row["state_sha256"])
                    or row["calibration_sha256"] != self.calibration.sha256):
                raise ValueError("invalid rod clock, state digest or calibration SHA")
            if (any(type(row[k]) is not int or row[k] < 0 for k in ("sdk_step", "rod_tick"))
                    or any(isinstance(row[k], bool) or not isinstance(row[k], (int, float))
                           or not math.isfinite(row[k]) or row[k] < 0
                           for k in ("sdk_timestamp_s", "solver_time_s"))
                    or any(not isinstance(row[k], str) or not row[k]
                           for k in ("rod_uid", "material_reference_uid"))):
                raise ValueError("invalid raw SDK/rod clock or material identity")
        sim_time = float(self.sim.current_time)
        if not math.isfinite(sim_time) or sim_time < 0:
            raise ValueError("invalid physical simulation time")
        sdk_step = self.sim.current_time_step_index
        if (type(sdk_step) is not int or sdk_step < 0
                or any(row["sdk_step"] != sdk_step or abs(row["sdk_timestamp_s"] - sim_time) > 1e-6
                       for row in rows)):
            raise ValueError("raw SDK clock and simulation clock mismatch")
        identity_rows = [{key: row[key] for key in _VALIDATION_KEYS} for row in rows]
        validate_identity(identity_rows, len(rows), sim_time)
        episode = tuple((row["run_uuid"], row["env_id"], row["episode_uid"], row["reset_id"]) for row in rows)
        previous_epoch = self.last_epoch if episode == self.last_episode else None

        def check_state():
            if (float(self.sim.current_time) != sim_time or type(self.sim.current_time_step_index) is not int
                    or self.sim.current_time_step_index != sdk_step or not _same_snapshot(self.snapshot(), before)):
                raise RuntimeError("physical state advanced during render or RGB read")

        packet = None
        read_clock = getattr(self.reader, "read_clock", None)
        for attempts in range(1, self.max_render_attempts + 1):
            check_state()
            prior_clock = read_clock() if callable(read_clock) else None
            check_state()
            self.sim.render()
            check_state()
            if callable(read_clock):
                if prior_clock is None:
                    continue
                if (not isinstance(prior_clock, dict) or type(prior_clock.get("render_frame_number")) is not int
                        or prior_clock["render_frame_number"] < 0):
                    raise ValueError("invalid pre-render clock")
                rendered_clock = read_clock()
                check_state()
                if rendered_clock is None:
                    continue
                if (not isinstance(rendered_clock, dict)
                        or type(rendered_clock.get("render_frame_number")) is not int
                        or rendered_clock["render_frame_number"] < 0):
                    raise ValueError("invalid post-render clock")
                if rendered_clock["render_frame_number"] <= prior_clock["render_frame_number"]:
                    continue
            candidate = self.reader.read()
            current_clock = read_clock() if callable(read_clock) else None
            check_state()
            if candidate is None:
                continue
            if not isinstance(candidate, dict) or not set(candidate) <= _PACKET_KEYS:
                raise ValueError("RGB reader returned forbidden fields")
            frame, timestamp = validate_pair_timing(candidate, -1)
            timing = candidate.get("render_timing")
            if (not isinstance(timing, dict) or timing.get("frame_id") != frame
                    or timing.get("simulation_time") != timestamp
                    or type(timing.get("render_frame_number")) is not int
                    or timing["render_frame_number"] < 0 or abs(timestamp - sim_time) > 1e-6):
                raise ValueError("render frame and physical timestamp mismatch")
            epoch = timing["render_frame_number"]
            if callable(read_clock):
                expected = {key: timing[key] for key in ("frame_id", "simulation_time", "render_frame_number")}
                if rendered_clock != expected or current_clock != expected:
                    raise ValueError("render clock changed during RGB read")
            if previous_epoch is not None and epoch <= previous_epoch:
                continue
            packet = candidate
            break
        if packet is None:
            raise RuntimeError("fresh synchronized RGB unavailable within render attempt limit")
        left, right = packet["left_rgb"], packet["right_rgb"]
        validate_images(left, right, self.calibration)
        if len(rows) != left.shape[0]:
            raise ValueError("identity and camera batch mismatch")
        self.calibration.validate_readback(packet["K_left"], packet["K_right"],
            packet["T_world_left"], packet["T_world_right"], origins=self.origins)
        if float(self.sim.current_time) != sim_time or type(self.sim.current_time_step_index) is not int or self.sim.current_time_step_index != sdk_step or not _same_snapshot(self.snapshot(), before):
            raise RuntimeError("physical state advanced during RGB read")
        self.last_frame = frame
        self.last_episode, self.last_epoch = episode, epoch
        return dict(left_rgb=left.clone(), right_rgb=right.clone(), identity=before,
                    frame_id=frame, simulation_time=timestamp, calibration_sha256=self.calibration.sha256,
                    render_passes=attempts,
                    K_left=torch.as_tensor(packet["K_left"]).clone(),
                    K_right=torch.as_tensor(packet["K_right"]).clone(),
                    render_timing={key: timing[key] for key in ("frame_id", "simulation_time", "render_frame_number")})
