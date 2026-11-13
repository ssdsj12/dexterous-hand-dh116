"""Compile opt-in native parameter getters, using the existing locked toolchain only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts import build_dh116_deformx_cuda as base
from scripts import prepare_dh116_deformx_parameter_getters as patch

def validate_output(source, sdk, output):
    source, sdk, output = (Path(x).resolve() for x in (source, sdk, output))
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(sdk):
        raise ValueError("build output must be new and separate from source and SDK")
    return output


def audit_worktree(source):
    """Retain the original payload audit; scope mount ownership trust to one child."""
    env = dict(os.environ)
    index = int(env.get("GIT_CONFIG_COUNT", "0"))
    env.update({f"GIT_CONFIG_KEY_{index}": "safe.directory",
                f"GIT_CONFIG_VALUE_{index}": str(Path(source).resolve()),
                "GIT_CONFIG_COUNT": str(index + 1)})
    result = subprocess.run([sys.executable, "-m", "scripts.verify_dh116_deformx_cuda_source",
                             str(source)], env=env, text=True, capture_output=True, timeout=120)
    if result.returncode not in (0, 2):
        raise ValueError(f"source audit child failed: {result.stderr}")
    return json.loads(result.stdout)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "toolkit", "output", "cmake", "ninja", "pybind"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    source, sdk = a.source.resolve(strict=True), a.toolkit.resolve(strict=True)
    output = validate_output(source, sdk, a.output)
    source_audit = audit_worktree(source)
    if not source_audit["source_inventory_available"]:
        raise ValueError(f"source audit failed: {source_audit['blockers']}")
    # Source edits happen only after immutable source/environment checks pass.
    import torch
    if sys.version_info[:3] != (3, 10, 12) or (torch.__version__, torch.version.cuda) != ("2.7.0+cu128", "12.8"):
        raise ValueError("requires current Python 3.10.12 / torch 2.7.0+cu128")
    version = base.validate_toolkit(sdk)
    cmake, ninja, pybind = (x.resolve(strict=True) for x in (a.cmake, a.ninja, a.pybind))
    patch.planned_changes(source)
    output.mkdir(mode=0o700)
    def save(name, value):
        with (output / name).open("x") as f:
            json.dump(value, f, indent=2); f.write("\n")
    save("source-audit.json", source_audit)
    save("patched-source-sha256.json", patch.apply(source))
    delta = subprocess.check_output(["git", "-c", f"safe.directory={source}", "-C",
                                     str(source), "diff", "--binary"])
    (output / "locked-source-delta.patch").write_bytes(delta)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "CUDAARCHS": "89",
           "CUDA_HOME": str(sdk), "CUDA_PATH": str(sdk), "CUDAToolkit_ROOT": str(sdk),
           "CUDACXX": str(sdk / "bin/nvcc"), "PYTHONDONTWRITEBYTECODE": "1"}
    command = base.configure_command(source, output / "cmake", sdk,
        python=Path(sys.executable), cmake=cmake, ninja=ninja, pybind=pybind)
    save("configure-command.json", command)
    base.checked(command, output / "configure.log", env)
    base.checked([str(cmake), "--build", str(output / "cmake"), "--parallel", "2"],
                 output / "build.log", env)
    base.checked([str(cmake), "--install", str(output / "cmake"), "--prefix", str(output / "python")],
                 output / "install.log", env)
    package = output / "python/ropesim"
    shutil.copytree(source / "ropesim", package, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    extension, = package.glob("_ropesim.cpython-310-*.so")
    base.checked([str(sdk / "bin/cuobjdump"), "--list-elf", str(extension)],
                 output / "device-code.log", env)
    if "sm_89" not in (output / "device-code.log").read_text():
        raise ValueError("SM89 machine code missing")
    result = dict(source_commit=patch.UPSTREAM, extension=str(extension),
        extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        source_delta_sha256=hashlib.sha256(delta).hexdigest(), python=sys.executable,
        python_version=sys.version.split()[0], torch=torch.__version__, torch_cuda=torch.version.cuda,
        nvcc=version, cuda_architecture="89", getter_schema="dh116.native_parameter_snapshot.v1",
        build_verified=True, solver_executed=False, isaac_coupling_verified=False,
        replaces_installed_backend=False, teacher_qualified=False)
    save("build-result.json", result)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
