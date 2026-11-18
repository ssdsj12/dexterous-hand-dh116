"""Scoped Isaac4.5/Warp non-owning pointer ABI adapter for RGB readback.

SDK/Warp files and global array classes stay unchanged. No force/control/state
inputs are accessed. GPU lifetimes remain subject to native qualification.
"""
from contextlib import contextmanager
import hashlib
import importlib
import inspect
from pathlib import Path
import threading
import types

ANNOTATOR_SHA256="1898abce52559be7befbc7b41bdd6752139df113ac69e62b5b2a97a9c5743a95"
BUNDLED_WARP_SOURCES={
    "warp/__init__.py":"0a969e4fe2e5b10cd8770c2608834d42f49e20d3cd91c32206c69314b0bb8f00",
    "warp/types.py":"6114a3fc38c971f1e4a341a260aa088f78543981210edc6bdc9428f3465ade1f",
}
POLICY="dh116-isaac45-warp112-rgb-nonowning-v1"
_LOCK=threading.RLock()

def nonowning_constructor(constructor):
    def create(*args,**kwargs):
        if "owner" in kwargs:
            if kwargs.pop("owner") is not False:
                raise ValueError("only verified owner=False external buffers are supported")
            if kwargs.get("ptr") is None or kwargs.get("deleter") is not None:
                raise ValueError("non-owning external pointer without deleter required")
            kwargs["deleter"]=None
        return constructor(*args,**kwargs)
    return create

class _Proxy:
    def __init__(self,target,**overrides):
        self._target,self._overrides=target,overrides
    def __getattr__(self,name):
        return self._overrides[name] if name in self._overrides else getattr(self._target,name)

def adapt_pointer_function(raw,warp):
    raw=inspect.unwrap(raw)
    if not isinstance(raw,types.FunctionType):raise ValueError("Python annotator pointer function required")
    proxy=_Proxy(warp,types=_Proxy(warp.types,array=nonowning_constructor(warp.array)))
    namespace=dict(raw.__globals__,wp=proxy)
    adapted=types.FunctionType(raw.__code__,namespace,raw.__name__,raw.__defaults__,raw.__closure__)
    adapted.__kwdefaults__=raw.__kwdefaults__
    return adapted

@contextmanager
def binding_scope(module,replacement):
    with _LOCK:
        previous=module._reshape_output_ptr
        module._reshape_output_ptr=replacement
        try:yield
        finally:
            if module._reshape_output_ptr is not replacement:
                raise RuntimeError("annotator pointer binding changed concurrently")
            module._reshape_output_ptr=previous

def _file_identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _pointer_abi(module, warp):
    annotator = _file_identity(module.__file__)
    warp_init = _file_identity(warp.__file__)
    warp_array = _file_identity(inspect.getfile(warp.array.__init__))
    parameters = inspect.signature(warp.array.__init__).parameters
    raw = inspect.unwrap(module._reshape_output_ptr)
    same_warp = isinstance(raw, types.FunctionType) and raw.__globals__.get("wp") is warp
    receipt = dict(warp_version=warp.__version__, warp_module=warp_init, warp_array=warp_array,
                   annotator_module=annotator, annotator_uses_same_warp=same_warp,
                   owner_argument="owner" in parameters, deleter_argument="deleter" in parameters,
                   warp_extension_config=None, policy=None, error=None)
    if annotator["sha256"] != ANNOTATOR_SHA256 or not same_warp:
        receipt["error"] = "unverified Isaac Replicator RGB pointer source"
    elif warp.__version__ == "1.5.0" and "owner" in parameters and parameters["owner"].default is False:
        root = Path(warp_init["path"]).parents[1]
        fingerprints = {str(Path(row["path"]).relative_to(root)):row["sha256"]
                        for row in (warp_init, warp_array) if Path(row["path"]).is_relative_to(root)}
        if fingerprints == BUNDLED_WARP_SOURCES:
            receipt["policy"] = "native_owner_false"
            config = root / "config/extension.toml"
            if config.is_file():
                receipt["warp_extension_config"] = _file_identity(config)
        else:
            receipt["error"] = "unverified bundled Warp RGB pointer source"
    elif warp.__version__ == "1.12.1" and "owner" not in parameters and "deleter" in parameters:
        receipt["policy"] = "scoped_owner_to_deleter"
    else:
        receipt["error"] = "unverified Isaac/Warp RGB pointer ABI"
    return receipt


def inspect_rgb_readback_abi():
    """Read the active Isaac process source and constructor contract without reading RGB."""
    with _LOCK:
        module = importlib.import_module("omni.replicator.core.scripts.utils.annotator_utils")
        return _pointer_abi(module, importlib.import_module("warp"))


@contextmanager
def isaac45_rgb_readback_scope():
    """Only use the exact audited installed combination, inside an active App."""
    with _LOCK:
        module = importlib.import_module("omni.replicator.core.scripts.utils.annotator_utils")
        warp = importlib.import_module("warp")
        receipt = _pointer_abi(module, warp)
        if receipt["policy"] == "native_owner_false":
            yield
        elif receipt["policy"] == "scoped_owner_to_deleter":
            replacement = adapt_pointer_function(module._reshape_output_ptr, warp)
            with binding_scope(module, replacement):
                yield
        else:
            raise ValueError(receipt["error"])
