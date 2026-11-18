"""Read-only inventory of a supplied DeformX CUDA solver checkout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess


TARGET_COMMIT = '96b8514c7c55c724f1abba1fbf6ed6642b7db72b'
_BINDING_EXTENSIONS = {'.cpp', '.cc', '.cxx', '.cu', '.c', '.py'}
_BINDING_MARKERS = (b'PYBIND11_MODULE', b'NB_MODULE', b'PyInit_', b'TORCH_LIBRARY(',
                    b'torch.utils.cpp_extension')
_LFS_VERSIONS = (b'version https://git-lfs.github.com/spec/v1',
                 b'version https://hawser.github.com/spec/v1')


def git(root, *args):
    return subprocess.run(
        ['git', '-c', 'core.fsmonitor=false', '-C', str(root), *args],
        env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'GIT_TERMINAL_PROMPT': '0',
             'GIT_NO_LAZY_FETCH': '1', 'GIT_ALLOW_PROTOCOL': '', 'GIT_LFS_SKIP_SMUDGE': '1'},
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15,
    )


def _path_problem(root, name, *, directory=False):
    """Check each component without following symlinks, including same-root links."""
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        return 'invalid_tracked_path'
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except (FileNotFoundError, NotADirectoryError):
            return 'missing_payload'
        except OSError:
            return 'unreadable_payload'
        if stat.S_ISLNK(mode):
            return 'symlink_path'
        is_directory = directory or index < len(relative.parts) - 1
        if is_directory and not stat.S_ISDIR(mode):
            return 'non_directory_path'
        if not is_directory and not stat.S_ISREG(mode):
            return 'non_regular_payload'
    return None


def _tree_entries(root, revision, result, prefix=''):
    """Walk the committed modes and only recurse through matching gitlinks."""
    tree = git(root, 'ls-tree', '-rz', revision)
    if tree.returncode:
        result['blockers'].append('committed source tree unavailable: ' + (prefix.rstrip('/') or '.'))
        return
    for line in tree.stdout.split('\0'):
        if not line:
            continue
        identity, name = line.split('\t', 1)
        mode, kind, oid = identity.split()
        full_name = prefix + name
        if mode != '160000':
            yield full_name, mode
            continue
        pin = dict(expected=oid, actual=None, status='unavailable')
        result['submodule_pins'][full_name] = pin
        problem = _path_problem(root, name, directory=True)
        if not problem and kind == 'commit':
            path = root / name
            top = git(path, 'rev-parse', '--show-toplevel')
            if top.returncode == 0 and Path(top.stdout.strip()).resolve() == path:
                head = git(path, 'rev-parse', '--verify', 'HEAD')
                if head.returncode == 0:
                    pin['actual'] = head.stdout.strip()
        if pin['actual'] != oid:
            pin['status'] = 'wrong_commit' if pin['actual'] is not None else (problem or 'unavailable')
            continue
        pin['status'] = 'verified_pin'
        yield from _tree_entries(root / name, oid, result, full_name + '/')


def _payload_roles(path):
    roles = []
    if path.suffix == '.cu':
        roles.append('cuda')
    if path.suffix == '.cuh':
        roles.append('cuda_headers')
    if path.name in ('CMakeLists.txt', 'pyproject.toml'):
        roles.append('build')
    if path.name.lower().split('.')[0] in ('license', 'licence', 'copying'):
        roles.append('license')
    return roles


def _payload_problem(root, name, mode):
    if mode == '120000':
        # core.symlinks=false can materialize a regular file containing only the link text.
        return 'tracked_symlink', None
    if mode not in ('100644', '100755'):
        return 'non_regular_tracked_mode', None
    problem = _path_problem(root, name)
    if problem:
        return problem, None
    try:
        data = (root / name).read_bytes()
    except OSError:
        return 'unreadable_payload', None
    if not data.strip():
        return 'empty_payload', None
    first_line = data.split(b'\n', 1)[0].removeprefix(b'\xef\xbb\xbf').strip()
    if first_line in _LFS_VERSIONS:
        return 'git_lfs_pointer', None
    return None, data


def inspect_source(root, *, expected_commit=TARGET_COMMIT):
    root = Path(root).resolve()
    result = dict(expected_commit=expected_commit, head=None, blockers=[], source_files={},
                  submodule_pins={}, unavailable_payloads={}, source_inventory_available=False,
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

    sources = result['source_files']
    sources.update({kind: [] for kind in ('cuda', 'cuda_headers', 'build', 'license', 'bindings')})
    if result['head'] is not None:
        for name, mode in _tree_entries(root, result['head'], result):
            path = Path(name)
            roles = _payload_roles(path)
            binding_candidate = path.suffix in _BINDING_EXTENSIONS
            if not roles and not binding_candidate:
                continue
            problem, data = _payload_problem(root, name, mode)
            if problem:
                # An empty __init__.py or other incidental code is not binding evidence.
                if problem != 'empty_payload' or roles:
                    result['unavailable_payloads'][name] = problem
                continue
            for role in roles:
                sources[role].append(name)
            if binding_candidate and any(marker in data for marker in _BINDING_MARKERS):
                sources['bindings'].append(name)

    if (modules.returncode or any(not line.startswith(' ') for line in result['submodules'])
            or any(pin['status'] != 'verified_pin' for pin in result['submodule_pins'].values())):
        blockers.append('recursive submodules unavailable or not at recorded commits')
    if result['unavailable_payloads']:
        blockers.append('tracked solver payload unavailable or a placeholder')
    if not sources['cuda']:
        blockers.append('tracked CUDA kernels not found')
    if not sources['bindings']:
        blockers.append('tracked binding source not found')
    if {Path(p).name for p in sources['build']} != {'CMakeLists.txt', 'pyproject.toml'}:
        blockers.append('CMakeLists.txt and pyproject.toml not both found')
    if not sources['license']:
        blockers.append('tracked license text not found')
    result['source_inventory_available'] = not blockers
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
