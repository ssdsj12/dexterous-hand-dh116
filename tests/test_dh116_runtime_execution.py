from test_native_teacher_context import backend_profile
from test_dh116_device_ppo import teacher_data
"""Shared eager-only policy contracts for Teacher and Student; CPU only."""
import importlib
import json
from pathlib import Path

import pytest
import torch

def api():
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_runtime_execution")

@pytest.mark.parametrize("config", [
    {"torch_compile":True}, {"compile":True}, {"compile_backend":"inductor"},
    {"torch_compile":{"enabled":True}}, {"network":{"torch_compile":True}},
])
def test_compile_configuration_is_rejected(config):
    with pytest.raises(ValueError, match="compile"):
        api().validate_runtime_execution(config=config)

def test_eager_and_independent_cuda_modes_are_allowed(monkeypatch):
    for backend in ("eager", "cuda_cabi"):
        monkeypatch.setenv("DH116_TENSOR_GEOMETRY_BACKEND", backend)
        info = api().validate_runtime_execution(config={"torch_compile":False})
        assert info["torch_compile"] is False and info["geometry_backend"] == backend

def test_compiled_environment_and_wrapped_models_are_rejected(monkeypatch):
    monkeypatch.setenv("DH116_TENSOR_GEOMETRY_BACKEND", "compiled")
    with pytest.raises(ValueError, match="compile"):
        api().validate_runtime_execution()
    monkeypatch.setenv("DH116_TENSOR_GEOMETRY_BACKEND", "eager")
    model = torch.nn.Sequential(torch.nn.Linear(2,2))
    model[0]._compiled_call_impl = object()
    with pytest.raises(ValueError, match="compiled"):
        api().validate_runtime_execution(model=model)
    model[0]._compiled_call_impl = None
    model[0]._orig_mod = torch.nn.Identity()
    with pytest.raises(ValueError, match="compiled"):
        api().validate_runtime_execution(model=model)

def test_student_factory_rejects_compile_and_module_compile_method(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor
    monkeypatch.setattr(torch, "compile", lambda *a,**k:pytest.fail("compiler must not be invoked"))
    with pytest.raises(ValueError, match="compile"):
        make_stereo_actor({"torch_compile":True})
    actor=make_stereo_actor(dict(embed_dim=16,num_heads=2,num_layers=1,history=1))
    with pytest.raises(RuntimeError, match="disabled"):
        actor.compile()

def test_teacher_policy_explicit_compile_is_disabled(teacher_data,monkeypatch):
    model,_=teacher_data
    monkeypatch.setattr(torch, "compile", lambda *a,**k:pytest.fail("compiler must not be invoked"))
    # Tensor Teacher inherits the same pure native network protocol but adds the
    # shared execution contract. This test exercises its actual compile method.
    from go2_pvcnn.control.dh116.tabletop_tensor_native_teacher import TensorNativeTeacher
    with pytest.raises(RuntimeError, match="disabled"):
        TensorNativeTeacher.compile(model)
