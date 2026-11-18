from pathlib import Path
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_fixed_dual_panda_dh116.py"


def test_auto_rod_backend_selects_cuda_for_table_wire_without_cpu_fallback():
    from co_sim.gpu_cosserat import resolve_rod_backend

    assert resolve_rod_backend("auto", table_wire=True, cuda_available=True) == "torch_cuda"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_rod_backend("auto", table_wire=True, cuda_available=False)


def test_cpu_rod_backend_requires_explicit_selection_and_non_wire_diagnostic_stays_cpu():
    from co_sim.gpu_cosserat import resolve_rod_backend

    assert resolve_rod_backend("cpu", table_wire=True, cuda_available=True) == "cpu"
    assert resolve_rod_backend("auto", table_wire=False, cuda_available=True) == "cpu"


def test_integrated_runner_exposes_cuda_rod_backend_without_replacing_usd_scene():
    source = SCRIPT.read_text()
    assert "--rod-backend" in source
    assert "choices=('auto','cpu','torch_cuda')" in source
    assert "default='auto'" in source
    assert "TorchCudaCosseratRod" in source
    assert "register_contact_mesh" in source
    assert "fixed_dual_panda_dh116.usd" in source
    assert "spawn.func('/World/Rig',spawn)" in source


def test_cuda_runner_selects_gpu_collision_engine_and_keeps_cpu_as_explicit_option():
    source = SCRIPT.read_text()
    assert "args.rod_backend=='torch_cuda'" in source
    assert "engine.advance_with_meshes(motions,dt)" in source
    assert "contact_geometry(c)" in source
    assert "gpu_contact_backend" in source
