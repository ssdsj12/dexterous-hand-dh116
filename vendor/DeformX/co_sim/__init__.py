"""Public API for the simplified co-simulation package."""

try:  # CPU/PyElastica remains optional for the explicit torch CUDA backend.
    from .engine import CoSimEngine, default_frame_initial_state, default_rod_initial_state
    from .isaac_process import circular_yz_frame_state, sine_frame_state
except ModuleNotFoundError as exc:  # pragma: no cover - depends on installation
    if exc.name != "elastica":
        raise
    CoSimEngine = None
    default_frame_initial_state = default_rod_initial_state = None
    circular_yz_frame_state = sine_frame_state = None
from .models import CoSimConfig, FrameState, ImpulseResult, RodInitialState, SceneSnapshot
from .gpu_cosserat import GpuRodConfig, TorchCudaCosseratRod

__all__ = [
    "CoSimConfig",
    "CoSimEngine",
    "FrameState",
    "ImpulseResult",
    "RodInitialState",
    "SceneSnapshot",
    "default_frame_initial_state",
    "default_rod_initial_state",
    "circular_yz_frame_state",
    "sine_frame_state",
    "GpuRodConfig",
    "TorchCudaCosseratRod",
]
