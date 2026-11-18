import pytest

from go2_pvcnn.control.dh116.hand_only_rod_runtime import HandOnlyRodRuntime


def test_cpu_runtime_is_explicitly_non_physical_probe():
    runtime = HandOnlyRodRuntime("cpu", device="cpu")
    report = runtime.step()
    assert report["backend"] == "cpu"
    assert report["physics_valid"] is False
    assert report["contact_coupled"] is False


def test_cuda_runtime_advances_cuda_resident_rod_without_cpu_fallback():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            HandOnlyRodRuntime("torch_cuda", device="cuda:0")
        return
    runtime = HandOnlyRodRuntime("torch_cuda", device="cuda:0")
    report = runtime.step()
    assert report["backend"] == "torch_cuda"
    assert report["state_device"] == "cuda:0"
    assert report["steps"] == 1
    assert report["physics_valid"] is False
    assert report["contact_coupled"] is False
