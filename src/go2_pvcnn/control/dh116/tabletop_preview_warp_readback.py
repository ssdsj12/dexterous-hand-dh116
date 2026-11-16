"""Scoped Isaac4.5/Warp1.12 non-owning pointer ABI adapter for RGB readback.

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

@contextmanager
def isaac45_rgb_readback_scope():
    """Only use the exact audited installed combination, inside an active App."""
    module=importlib.import_module("omni.replicator.core.scripts.utils.annotator_utils")
    warp=importlib.import_module("warp")
    path=Path(module.__file__).resolve()
    if (hashlib.sha256(path.read_bytes()).hexdigest()!=ANNOTATOR_SHA256
            or warp.__version__!="1.12.1"
            or "owner" in inspect.signature(warp.array.__init__).parameters
            or "deleter" not in inspect.signature(warp.array.__init__).parameters):
        raise ValueError("unverified Isaac/Warp RGB pointer ABI")
    replacement=adapt_pointer_function(module._reshape_output_ptr,warp)
    with binding_scope(module,replacement):
        yield
