"""Installed-source, mocked-native verification of external cleanup suggestions.

Opt in explicitly on Ubuntu. No Isaac imports, CUDA initialization or native API
calls; these checks do not qualify SDK destruction, GPU synchronization or grasp.
"""
import ast
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(
    os.environ.get("DH116_INSTALLED_SDK_AUDIT")!="1",
    reason="explicit Ubuntu installed-source audit only; not native qualification")


@pytest.fixture(autouse=True)
def no_native_execution():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="", "CPU-only process required"
    import torch
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()
    assert not any(name=="isaaclab" or name.startswith("isaaclab.")
        or name=="isaacsim" or name.startswith("isaacsim.") for name in sys.modules)


def installed_wrapper():
    contract=json.loads((ROOT/"docs/contracts/dh116_teacher_tensor_owner_probe_v1.json").read_text())
    pin=contract["installed_sources"]["tensor_api"]
    path=Path(pin["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin["sha256"]
    node=next(n for n in ast.parse(path.read_text()).body
        if isinstance(n,ast.ClassDef) and n.name=="SimulationView")
    methods={n.name for n in node.body if isinstance(n,ast.FunctionDef)}
    # Execute only these pure Python methods with a fake backend, without
    # importing omni, carb, Kit, PhysX bindings or constructing a native view.
    selected=copy.deepcopy(node)
    selected.body=[n for n in selected.body
        if isinstance(n,ast.FunctionDef) and n.name in ("__init__","invalidate")]
    namespace={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selected],type_ignores=[])),
        str(path)+"[CPU-methods-only]","exec"),namespace)
    return namespace["SimulationView"],methods


class Backend:
    def __init__(self):
        self.valid=True
        self.invalidations=0
        self.releases=0
    def invalidate(self):
        self.valid=False
        self.invalidations+=1
    def release(self):
        # This is a test spy, not evidence that this method exists in pybind.
        self.releases+=1


def test_installed_wrapper_uses_private_backend_and_has_no_public_release():
    wrapper,methods=installed_wrapper()
    backend=Backend();view=wrapper(backend,object())
    assert view._backend is backend and not hasattr(view,"backend")
    assert "invalidate" in methods and "release" not in methods and "close" not in methods


def test_actual_invalidate_keeps_wrapper_backend_reference():
    wrapper,_=installed_wrapper()
    backend=Backend();view=wrapper(backend,object())
    view.invalidate()
    assert backend.invalidations==1 and backend.valid is False
    assert view._backend is backend and backend.releases==0


def test_pasted_public_backend_branch_skips_actual_wrapper():
    wrapper,_=installed_wrapper()
    backend=Backend();view=wrapper(backend,object())
    # The screenshot's public-name branch misses the actual installed field.
    if hasattr(view,"backend") and view.backend is not None:
        view.backend.release()
        view.backend=None
    assert view._backend is backend
    assert backend.invalidations==0 and backend.releases==0


def test_actual_empty_cache_does_not_release_python_scene_owners(monkeypatch):
    import torch
    wrapper,_=installed_wrapper()
    backend=Backend();view=wrapper(backend,object());calls=[]
    fn=torch.cuda.empty_cache
    monkeypatch.setitem(fn.__globals__,"is_initialized",lambda:True)
    monkeypatch.setattr(torch._C,"_cuda_emptyCache",lambda:calls.append("allocator"))
    fn()
    assert calls==["allocator"]
    assert view._backend is backend and backend.valid and backend.releases==0
    # C++ allocator semantics and native backend destruction are not exercised.


def test_actual_synchronize_enters_lazy_init_before_native_sync(monkeypatch):
    import torch
    events=[]
    monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:events.append("lazy_init"))
    monkeypatch.setattr(torch.cuda,"device",lambda device:nullcontext())
    monkeypatch.setattr(torch._C,"_cuda_synchronize",lambda:events.append("native_sync_spy"))
    torch.cuda.synchronize()
    assert events==["lazy_init","native_sync_spy"]


def test_screenshot_unindexed_cuda_device_rejects_before_native_call(monkeypatch):
    import torch
    calls=[]
    monkeypatch.setattr(torch._C,"_cuda_setDevice",lambda index:calls.append(index))
    with pytest.raises(ValueError,match="specified index"):
        torch.cuda.set_device(torch.device("cuda"))
    assert calls==[]


def test_indexed_logical_device_routes_to_index_zero_without_native_execution(monkeypatch):
    import torch
    calls=[]
    monkeypatch.setattr(torch._C,"_cuda_setDevice",lambda index:calls.append(index))
    torch.cuda.set_device(torch.device("cuda:0"))
    assert calls==[0]
    # Logical0 maps to physicalGPU2 only in a correctly bound native process.


def test_actual_lazy_queue_records_formatted_strings_not_live_tracebacks(monkeypatch):
    import torch
    queued=[];calls=[]
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:False)
    monkeypatch.setattr(torch.cuda,"_queued_calls",queued)
    callback=lambda:calls.append("executed")
    torch.cuda._lazy_call(callback)
    assert calls==[] and len(queued)==1 and queued[0][0] is callback
    stack=queued[0][1]
    assert isinstance(stack,list) and stack and all(isinstance(line,str) for line in stack)
    assert not any(isinstance(value,type(sys._getframe())) for value in stack)
