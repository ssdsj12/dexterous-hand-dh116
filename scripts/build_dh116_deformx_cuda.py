"""Build an audited DeformX checkout into an isolated cp310/SM89 package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

if __package__:
    from .verify_dh116_deformx_cuda_source import inspect_source
else:
    from verify_dh116_deformx_cuda_source import inspect_source


def configure_command(source, output, sdk, *, python, cmake, ninja, pybind):
    return [
        str(cmake), "-S", str(source), "-B", str(output), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_MAKE_PROGRAM={ninja}",
        f"-DCMAKE_CUDA_COMPILER={sdk / 'bin' / 'nvcc'}",
        f"-DCUDAToolkit_ROOT={sdk}", "-DCMAKE_CUDA_ARCHITECTURES=89",
        "-DPYBIND11_FINDPYTHON=ON", f"-DPython_EXECUTABLE={python}",
        f"-Dpybind11_DIR={pybind}",
    ]


def validate_toolkit(sdk):
    sdk = sdk.resolve(strict=True)
    nvcc = (sdk / "bin" / "nvcc").resolve(strict=True)
    if not nvcc.is_relative_to(sdk):
        raise ValueError("nvcc resolves outside the isolated SDK")
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "V12.8.61" not in version:
        raise ValueError("The pinned CUDA 12.8.0 SDK requires nvcc 12.8.61")
    for name in ("include/cuda_runtime.h", "nvvm/libdevice/libdevice.10.bc"):
        path = (sdk / name).resolve(strict=True)
        if not path.is_relative_to(sdk) or not path.is_file():
            raise ValueError(f"CUDA component outside SDK or missing: {name}")
    for name in ("bin/ptxas", "bin/nvlink", "targets/x86_64-linux/nvvm/bin/cicc"):
        executable = (sdk / name).resolve()
        if not executable.is_relative_to(sdk) or not os.access(executable, os.X_OK):
            raise ValueError(f"Missing or external CUDA tool: {name}")
    return version.strip()


def checked(command, log, env, *, timeout=900):
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       env=env, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "toolkit", "output", "cmake", "ninja", "pybind"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit must be a full SHA-1")
    source = args.source.resolve(strict=True)
    sdk = args.toolkit.resolve(strict=True)
    out = args.output.resolve()
    if out.is_relative_to(source) or out.is_relative_to(sdk):
        parser.error("--output must be separate from source and toolkit")
    out.mkdir(mode=0o700)
    source_audit = inspect_source(source, expected_commit=args.source_commit)
    (out / "source-audit.json").write_text(json.dumps(source_audit, indent=2) + "\n")
    if not source_audit["source_inventory_available"]:
        raise ValueError(f"DeformX checkout failed source audit: {source_audit['blockers']}")

    import torch

    if sys.version_info[:3] != (3, 10, 12):
        raise ValueError("Build must use the current DH116 Python 3.10.12")
    if (torch.__version__, torch.version.cuda) != ("2.7.0+cu128", "12.8"):
        raise ValueError("Build must use the current DH116 PyTorch 2.7.0+cu128")
    version = validate_toolkit(sdk)
    env = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "", "CUDAARCHS": "89",
        "CUDA_HOME": str(sdk), "CUDA_PATH": str(sdk), "CUDAToolkit_ROOT": str(sdk),
        "CUDACXX": str(sdk / "bin" / "nvcc"), "PYTHONDONTWRITEBYTECODE": "1",
    }
    cmake = args.cmake.resolve(strict=True)
    build = out / "cmake"
    staged = out / "python"
    command = configure_command(
        source, build, sdk, python=Path(sys.executable), cmake=cmake,
        ninja=args.ninja.resolve(strict=True), pybind=args.pybind.resolve(strict=True),
    )
    (out / "configure-command.json").write_text(json.dumps(command, indent=2) + "\n")
    checked(command, out / "configure.log", env)
    checked([str(cmake), "--build", str(build), "--parallel", "2"], out / "build.log", env)
    checked([str(cmake), "--install", str(build), "--prefix", str(staged)],
            out / "install.log", env)
    package = staged / "ropesim"
    shutil.copytree(source / "ropesim", package, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    extension, = package.glob("_ropesim.cpython-310-*.so")
    checked([str(sdk / "bin" / "cuobjdump"), "--list-elf", str(extension)],
            out / "device-code.log", env)
    if "sm_89" not in (out / "device-code.log").read_text():
        raise ValueError("Expected SM89 machine code is missing from extension")
    result = {
        "source_commit": args.source_commit, "python": sys.executable,
        "python_version": sys.version.split()[0], "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "sdk": str(sdk), "nvcc": version,
        "cuda_architecture": "89", "extension": str(extension),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "build_verified": True, "solver_executed": False, "isaac_coupling_verified": False,
    }
    (out / "build-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
