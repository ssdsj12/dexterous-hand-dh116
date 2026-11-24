#!/usr/bin/env python3
"""Rebuild copied CUDA sources in a new private directory, never in the repo.

The rebuilt binary always gets a new manifest, even if its SHA matches. This
does not update the pinned production manifest or grant native admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "src/go2_pvcnn/control/dh116/cuda_measurements"
DEFAULT_NVCC = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-f1-20261112-883RS4/sdk/bin/nvcc")


def build(output, nvcc=DEFAULT_NVCC):
    output, nvcc = Path(output).resolve(), Path(nvcc).resolve(strict=True)
    if output == ROOT or ROOT in output.parents:
        raise ValueError("new independent build output must be outside the repository")
    # Refuse existing directories and symlinks: no prior binary can be replaced.
    if output.exists():
        raise ValueError("build output must be a new independent directory")
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "V12.8.61" not in version:
        raise ValueError("validated build requires existing NVCC12.8.61")
    output.mkdir(parents=True, exist_ok=False)
    sdk = nvcc.parent.parent
    binary = output / "libdh116_hull_support.so"
    command = [str(nvcc), "-O3", "-Xptxas=-v", "-std=c++17", "--fmad=false",
        "--prec-div=true", "--prec-sqrt=true", "-arch=sm_89", "-Xcompiler=-fPIC", "-shared",
        str(SOURCES/"hull_support.cu"), str(SOURCES/"pad_support.cu"),
        "-L"+str(sdk/"lib64"), "-Xlinker=-rpath", "-Xlinker="+str(sdk/"lib64"), "-o", str(binary)]
    env = dict(os.environ, TMPDIR=str(output), CUDA_CACHE_PATH=str(output/"cuda_cache"))
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    (output/"build.log").write_text(result.stdout+result.stderr)
    result.check_returncode()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(schema="dh116-cuda-geometry-rebuild-v1", native_ready=False,
        teacher_admission=False, binary_path=str(binary), binary_sha256=sha(binary),
        source_sha256={p.name: sha(p) for p in (SOURCES/"hull_support.cu", SOURCES/"pad_support.cu")},
        command=command, nvcc_version=version, sm=89, fmad=False, fast_math=False,
        production_manifest_updated=False, requires_new_validation=True)
    (output/"rebuild_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new independent private output directory")
    parser.add_argument("--nvcc", type=Path, default=DEFAULT_NVCC)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.nvcc), indent=2))


if __name__ == "__main__":
    main()
