import numpy as np
import pytest

from vendor.DeformX.co_sim.gpu_cosserat import GpuRodConfig, TorchCudaCosseratRod


def test_gpu_backend_is_explicit_and_fails_without_cuda():
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None or not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            TorchCudaCosseratRod(device="cuda")
        return
    rod = TorchCudaCosseratRod(GpuRodConfig(n_elem=4), device="cuda")
    snap = rod.step()
    assert snap["rod_position"].is_cuda
    assert snap["rod_position"].shape == (3, 5)
    assert np.isfinite(snap["rod_position"].detach().cpu().numpy()).all()


def test_gpu_backend_never_accepts_cpu_device():
    with pytest.raises(RuntimeError):
        TorchCudaCosseratRod(device="cpu")
