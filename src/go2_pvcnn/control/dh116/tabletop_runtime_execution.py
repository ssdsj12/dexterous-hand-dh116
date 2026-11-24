"""Eager-only contract shared by native Teacher and stereo Student execution."""
from collections.abc import Mapping
import os

VERSION = "dh116-shared-eager-execution-v1"

def validate_runtime_execution(*, config=None, model=None):
    """Reject compiler configuration and already compiled modules before binding.

    No compiler is imported, invoked, monkey-patched or globally disabled. Adam
    may import Torch's ordinary internal helpers independently; that does not
    authorize compiling policy or native-runtime graphs.
    """
    geometry = os.environ.get("DH116_TENSOR_GEOMETRY_BACKEND", "eager")
    if geometry not in ("eager", "cuda_cabi"):
        raise ValueError("torch.compile/compiled runtime is disabled; choose eager or cuda_cabi")
    def inspect_config(value):
        if isinstance(value, Mapping):
            for name, item in value.items():
                if name in ("compile", "torch_compile", "compile_backend") and item not in (None, False, ""):
                    raise ValueError("torch.compile configuration is disabled")
                inspect_config(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                inspect_config(item)
    inspect_config(config)
    if model is not None:
        for child in model.modules():
            if (getattr(child, "_compiled_call_impl", None) is not None
                    or hasattr(child, "_orig_mod") or hasattr(child, "_torchdynamo_orig_callable")):
                raise ValueError("already compiled policy module is forbidden by shared runtime")
    return dict(version=VERSION, torch_compile=False, geometry_backend=geometry)

class EagerRuntimePolicy:
    """Prevent in-place Module.compile on both Teacher and Student policies."""
    def compile(self, *args, **kwargs):
        raise RuntimeError("torch.compile is disabled for the shared physical runtime")
