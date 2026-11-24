"""Reusable geometry corpus and a strict ban on runtime graph compilation."""
from dataclasses import fields
import os

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_tensor_geometry import TensorGeometry


def inventory():
    rng = np.random.default_rng(914)
    shapes = [(s, i, "source:link"+str(i), rng.normal(size=(129+17*i, 3))*.001)
              for s in (0, 1) for i in range(3)]
    x, y, z = np.indices((9, 9, 9))
    grid = dict(values=.03+.001*x+.002*y+.003*z+.0001*x*y,
                origin=np.full(3, -.04), spacing=np.array(.01))
    shapes += [(s, 3, "cooked:pad", rng.normal(size=(34, 3))*.001) for s in (0, 1)]
    return shapes, [(s, 3, "pad", grid) for s in (0, 1)]


def inputs(device, n=2):
    p = torch.zeros((n, 2, 23, 7), dtype=torch.float64, device=device)
    p[..., 6], p[..., 2] = 1, .6
    p[:, 0, :3, 0], p[:, 1, :3, 0] = -1, 1
    rotations = Rotation.from_euler("xyz", [[.21, -.14, .09], [-.13, .18, -.23]])
    q = rotations.as_quat()[np.arange(n) % 2]
    p[:, :, 3, 3:] = torch.tensor(q, device=device)[:, None]
    local = np.random.default_rng(927).uniform(-.025, .025, (n, 41, 3))
    nworld = np.einsum("nvi,nji->nvj", local, rotations.as_matrix()[np.arange(n) % 2])+[0, 0, .6]
    return p, torch.tensor(nworld, device=device)


def assert_parity(actual, expected):
    for field in fields(actual):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if not isinstance(a, torch.Tensor):
            assert a == b, field.name
        elif a.is_floating_point():
            torch.testing.assert_close(a, b, rtol=0, atol=1e-10, equal_nan=True, msg=field.name)
        else:
            assert torch.equal(a, b), field.name


def test_default_eager_and_explicit_backend_validation(monkeypatch):
    shapes, pads = inventory()
    monkeypatch.delenv("DH116_TENSOR_GEOMETRY_BACKEND", raising=False)
    assert TensorGeometry(shapes, pads, device="cpu").backend == "eager"
    monkeypatch.setenv("DH116_TENSOR_GEOMETRY_BACKEND", "bad")
    with pytest.raises(ValueError, match="backend"):
        TensorGeometry(shapes, pads, device="cpu")
    assert TensorGeometry(shapes, pads, device="cpu", backend="eager").backend == "eager"


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("via_env", (False, True))
def test_compiled_backend_rejected_before_device_or_compiler(monkeypatch, device, via_env):
    shapes, pads = inventory()
    called = []
    def forbidden(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("runtime graph compiler or device allocation was called")
    monkeypatch.setattr(torch, "compile", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setenv("DH116_TENSOR_GEOMETRY_BACKEND", "compiled")
    with pytest.raises(ValueError, match="compiled.*disabled"):
        TensorGeometry(shapes, pads, device=device, backend=None if via_env else "compiled")
    assert not called


def test_eager_capture_does_not_call_runtime_compiler(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("runtime graph compiler was called")
    monkeypatch.setattr(torch, "compile", forbidden)
    shapes, pads = inventory()
    geometry = TensorGeometry(shapes, pads, device="cpu", backend="eager")
    p, n = inputs("cpu")
    assert bool(geometry.capture(p, n, frame="no-compile").source_complete.all())


