"""Read-only inventory of a supplied DeformX CUDA solver checkout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


TARGET_COMMIT = '96b8514c7c55c724f1abba1fbf6ed6642b7db72b'


def git(root, *args):
    return subprocess.run(
        ['git', '-c', 'core.fsmonitor=false', '-C', str(root), *args],
        env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'GIT_TERMINAL_PROMPT': '0'},
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15,
    )


def inspect_source(root, *, expected_commit=TARGET_COMMIT):
    root = Path(root).resolve()
    result = dict(expected_commit=expected_commit, head=None, blockers=[], source_files={},
                  build_verified=False, solver_executed=False)
    blockers = result['blockers']
    top = git(root, 'rev-parse', '--show-toplevel')
    if top.returncode or Path(top.stdout.strip()).resolve() != root:
        blockers.append('not a standalone Git checkout')
        return result

    head = git(root, 'rev-parse', '--verify', 'HEAD')
    result['head'] = head.stdout.strip() if head.returncode == 0 else None
    if result['head'] != expected_commit:
        blockers.append('target commit not checked out')
    status = git(root, 'status', '--porcelain=v1', '--untracked-files=normal', '--ignore-submodules=none')
    if status.returncode:
        blockers.append('checkout status unavailable')
    elif status.stdout:
        blockers.append('checkout has local changes')
    modules = git(root, 'submodule', 'status', '--recursive')
    result['submodules'] = modules.stdout.splitlines()
    if modules.returncode or any(not line.startswith(' ') for line in result['submodules']):
        blockers.append('recursive submodules unavailable or not at recorded commits')

    tracked = git(root, 'ls-files', '-z', '--recurse-submodules')
    if tracked.returncode:
        blockers.append('recursive tracked source inventory unavailable')
        return result
    files = []
    for name in tracked.stdout.split('\0'):
        path = root / name
        if name and path.is_file() and path.resolve().is_relative_to(root) and path.stat().st_size:
            files.append(path)
    sources = result['source_files']
    sources['cuda'] = [p.relative_to(root).as_posix() for p in files if p.suffix == '.cu']
    sources['build'] = [p.relative_to(root).as_posix() for p in files
                        if p.name in ('CMakeLists.txt', 'pyproject.toml')]
    sources['license'] = [p.relative_to(root).as_posix() for p in files
                          if p.name.lower().split('.')[0] in ('license', 'licence', 'copying')]
    binding_markers = ('PYBIND11_MODULE', 'NB_MODULE', 'PyInit_', 'TORCH_LIBRARY(',
                       'torch.utils.cpp_extension')
    sources['bindings'] = [p.relative_to(root).as_posix() for p in files
                           if p.suffix in ('.cpp', '.cc', '.cxx', '.cu', '.c', '.py')
                           and any(marker in p.read_text(encoding='utf-8', errors='replace')
                                   for marker in binding_markers)]
    if not sources['cuda']:
        blockers.append('tracked CUDA kernels not found')
    if not sources['bindings']:
        blockers.append('tracked binding source not found')
    if {Path(p).name for p in sources['build']} != {'CMakeLists.txt', 'pyproject.toml'}:
        blockers.append('CMakeLists.txt and pyproject.toml not both found')
    if not sources['license']:
        blockers.append('tracked license text not found')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    result = inspect_source(args.checkout)
    print(json.dumps(result, indent=2))
    return 2 if result['blockers'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
