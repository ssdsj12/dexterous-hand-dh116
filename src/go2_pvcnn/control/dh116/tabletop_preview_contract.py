"""CPU-only frozen identity for the independent, zero-PPO native preview."""
import hashlib
from pathlib import Path
from .tabletop_preview_debugger import POLICY_VERSION
from .tabletop_preview_launcher import ADAPTER_VERSION, preview_launcher_type

VERSION = "dh116-webrtc-physical-preview-v3"
BUDGET = dict(control_intervals=16, setup_sdk_steps=2, measured_sdk_steps=128,
              total_sdk_steps=130, solver_substeps_per_sdk_step=12,
              total_solver_substeps=1536, max_camera_renders_per_frame=8,
              max_render_calls=162, soft_deadline_s=180, ppo_updates=0)
CAMERA_CONFIG = "configs/dh116_training/stereo_camera_front_640x480_v3.json"


def _hash(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("source must be a regular non-symlink file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root):
    root = Path(root).resolve()
    result = {}
    for part, extensions in (("src", (".py",)), ("scripts", (".py",)),
                             ("configs", (".json",)), ("docs/contracts", (".json",))):
        base = root / part
        if not base.is_dir(): raise ValueError("source inventory directory missing: " + part)
        for path in sorted(base.rglob("*")):
            if path.suffix in extensions:
                if not path.resolve().is_relative_to(root):
                    raise ValueError("source escapes repository")
                result[path.relative_to(root).as_posix()] = _hash(path)
    return result


def _capture_resolution(value):
    from .tabletop_stereo_resolution import validate_resolution
    if not isinstance(value, dict) or set(value) != {"width", "height"}:
        raise ValueError("one shared left/right capture resolution required")
    validate_resolution(value["width"], value["height"])
    return dict(value)


def create_preview_camera(root, sim, contract, snapshot, *, render_setting=None):
    """Author both native lenses at the frozen size; never resize readback."""
    from .tabletop_stereo_calibration import StereoCalibration
    from .tabletop_stereo_rgb_bridge import RGBPairBridge
    resolution = _capture_resolution(contract.get("capture_resolution"))
    if contract.get("camera_config") != CAMERA_CONFIG:
        raise ValueError("preview camera rig differs")
    calibration = StereoCalibration.from_json(Path(root) / CAMERA_CONFIG)
    backend = contract.get("camera_backend", "legacy_view")
    if backend == "tiled_step":
        from .tabletop_tiled_stereo_capture import TiledStereoCapture
        return TiledStereoCapture.from_native(sim, [[0., 0., 0.]], calibration, snapshot,
            **resolution, horizontal_fov_degrees=75., device="cuda:0",
            max_render_attempts=BUDGET["max_camera_renders_per_frame"] // 2,
            render_setting=render_setting)
    if backend != "legacy_view":
        raise ValueError("unsupported preview camera backend")
    return RGBPairBridge.from_native(sim, [[0., 0., 0.]], calibration, snapshot,
        **resolution, horizontal_fov_degrees=75., device="cuda:0",
        max_render_attempts=BUDGET["max_camera_renders_per_frame"],
        render_setting=render_setting)


def _camera_api_sources(root, backend):
    if backend == "legacy_view":
        return None
    if backend == "tiled_step":
        from .tabletop_tiled_stereo_resources import verify_tiled_sources
        return verify_tiled_sources(root)
    raise ValueError("unsupported preview camera backend")


def freeze(root, experience, *, physical_gpu_index=3, width=640, height=480,
           camera_backend="legacy_view"):
    if type(physical_gpu_index) is not int or physical_gpu_index<0:
        raise ValueError("nonnegative physical GPU index required")
    resolution = _capture_resolution(dict(width=width, height=height))
    sources = _camera_api_sources(root, camera_backend)
    experience = Path(experience).resolve(strict=True)
    return dict(version=VERSION, launcher_adapter=ADAPTER_VERSION, debugger_policy=POLICY_VERSION, source_sha256=_inventory(root), physical_gpu_index=physical_gpu_index,
                experience=dict(path=str(experience), sha256=_hash(experience)),
                budget=dict(BUDGET), camera_config=CAMERA_CONFIG, capture_resolution=resolution,
                camera_backend=camera_backend, camera_api_sources=sources,
                launcher=dict(headless=True, livestream=2, enable_cameras=True, device="cuda:0", kit_args="--enable omni.kit.widget.toolbar"),
                action="zero_24d", permission="independent_preview_only")


def verify(root, contract):
    if (not isinstance(contract,dict) or set(contract) !=
        {"version","source_sha256","experience","budget","camera_config","launcher","action","permission","physical_gpu_index","launcher_adapter","debugger_policy","capture_resolution","camera_backend","camera_api_sources"}
        or contract["version"] != VERSION or contract["budget"] != BUDGET
        or contract["launcher_adapter"] != ADAPTER_VERSION
        or contract["debugger_policy"] != POLICY_VERSION
        or type(contract["physical_gpu_index"]) is not int or contract["physical_gpu_index"]<0
        or contract["camera_config"] != CAMERA_CONFIG
        or contract["launcher"] != dict(headless=True,livestream=2,enable_cameras=True,device="cuda:0",kit_args="--enable omni.kit.widget.toolbar")
        or contract["action"] != "zero_24d" or contract["permission"] != "independent_preview_only"):
        raise ValueError("independent preview settings differ")
    _capture_resolution(contract["capture_resolution"])
    if contract["camera_api_sources"] != _camera_api_sources(root, contract["camera_backend"]):
        raise ValueError("preview camera API source binding differs")
    if contract["source_sha256"] != _inventory(root):
        raise ValueError("preview source inventory differs")
    exp = contract["experience"]
    if (not isinstance(exp,dict) or set(exp)!={"path","sha256"}
            or not Path(exp["path"]).is_absolute() or _hash(exp["path"]) != exp["sha256"]):
        raise ValueError("preview experience identity differs")
    return contract


class RenderBudget:
    """Count render attempts, including failed calls; forwards all other access."""
    def __init__(self, sim, limit):
        if type(limit) is not int or limit < 1: raise ValueError("positive render budget required")
        self._sim, self.limit, self.count = sim, limit, 0
        self._render = sim.render
        self._installed = False
        self._closed = False
        self._had_override = "render" in vars(sim)
        self._prior_override = vars(sim).get("render")

    def __getattr__(self, name):
        if self._closed: raise RuntimeError("render budget is closed")
        return getattr(self._sim, name)

    def install(self):
        if self._closed: raise RuntimeError("render budget is closed")
        if self._installed: raise RuntimeError("render budget already installed")
        self._sim.render = self.render
        self._installed = True

    def close(self):
        if self._closed:return
        try:
            if self._installed:
                if self._had_override: self._sim.render = self._prior_override
                else: del self._sim.render
        finally:
            self._installed = False
            self._closed = True
            self._sim = self._render = self._prior_override = None

    def render(self):
        if self._closed: raise RuntimeError("render budget is closed")
        if self.count >= self.limit: raise RuntimeError("preview render budget exhausted")
        self.count += 1
        return self._render()


def launch_preview_app(launcher_factory, contract):
    """Pass the frozen startup dependencies before IsaacLab's toolbar hook."""
    kwargs = dict(contract["launcher"])
    kwargs.update(experience=contract["experience"]["path"], fast_shutdown=False)
    return preview_launcher_type(launcher_factory)(**kwargs).app
