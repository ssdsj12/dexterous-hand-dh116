from pathlib import Path
import subprocess

import pytest

from scripts import build_dh116_deformx_cuda as build


def test_configure_uses_explicit_sdk_compiler_and_sm89(tmp_path):
    source = tmp_path / "solver"
    output = tmp_path / "build"
    sdk = tmp_path / "cuda-12.8.0"
    python = Path("/mnt/mydisk/xk/envs/dh116/bin/python")
    ninja = tmp_path / "ninja"
    pybind = tmp_path / "pybind11"
    cmake = tmp_path / "cmake"
    command = build.configure_command(
        source, output, sdk, python=python, cmake=cmake, ninja=ninja, pybind=pybind,
    )
    assert command[:7] == [str(cmake), "-S", str(source), "-B", str(output), "-G", "Ninja"]
    assert f"-DCMAKE_CUDA_COMPILER={sdk / 'bin' / 'nvcc'}" in command
    assert f"-DCUDAToolkit_ROOT={sdk}" in command
    assert "-DCMAKE_CUDA_ARCHITECTURES=89" in command
    assert f"-DPython_EXECUTABLE={python}" in command
    assert f"-Dpybind11_DIR={pybind}" in command
    assert f"-DCMAKE_MAKE_PROGRAM={ninja}" in command


def test_validate_sdk_rejects_existing_old_nvcc_before_configure(tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents=True)
    compiler = sdk / "bin" / "nvcc"
    compiler.write_text("#!/bin/sh\nprintf 'Cuda compilation tools, release 12.4, V12.4.99\\n'\n")
    compiler.chmod(0o700)
    with pytest.raises(ValueError, match="12.8.61"):
        build.validate_toolkit(sdk)


def test_validate_sdk_does_not_accept_executable_from_other_toolkit(tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents=True)
    external = tmp_path / "nvcc"
    external.write_text("#!/bin/sh\nexit 0\n")
    external.chmod(0o700)
    (sdk / "bin" / "nvcc").symlink_to(external)
    with pytest.raises(ValueError, match="outside"):
        build.validate_toolkit(sdk)


def test_checked_command_preserves_failure_log(tmp_path):
    log = tmp_path / "cmake.log"
    with pytest.raises(subprocess.CalledProcessError):
        build.checked(["sh", "-c", "printf configure-failed; exit 7"], log, {})
    assert log.read_text() == "configure-failed"


def test_validate_sdk_checks_cicc_in_nvcc_profile_target(tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents=True)
    compiler = sdk / "bin" / "nvcc"
    compiler.write_text("#!/bin/sh\nprintf 'Cuda compilation tools, release 12.8, V12.8.61\\n'\n")
    compiler.chmod(0o700)
    for name in ("include/cuda_runtime.h", "nvvm/libdevice/libdevice.10.bc"):
        path = sdk / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("payload")
    for name in ("ptxas", "nvlink"):
        (sdk / "bin" / name).symlink_to(compiler)
    with pytest.raises(ValueError, match="cicc"):
        build.validate_toolkit(sdk)
