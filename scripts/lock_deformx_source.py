"""Verify the immutable upstream source lock against a materialized DeformX checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

if __package__:
    from .verify_dh116_deformx_cuda_source import git, inspect_source
else:
    from verify_dh116_deformx_cuda_source import git, inspect_source


LOCK_SCHEMA = 'dh116-deformx-backend-lock-v1'
DEFAULT_LOCK = Path(__file__).resolve().parents[1] / 'docs/contracts/dh116_deformx_backend_lock.json'


def _relative_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError(f'non-canonical lock path: {name!r}')
    return path


def _inventory(checkout, pin, blockers, gitlinks):
    name = pin['path']
    top = git(checkout, 'rev-parse', '--show-toplevel')
    if not checkout.is_dir() or checkout.resolve() != checkout or (
        top.returncode or Path(top.stdout.strip()).resolve() != checkout
    ):
        blockers.append(f'{name}: missing standalone checkout')
        return
    for label, expected, args in (
        ('HEAD commit', pin['commit'], ('rev-parse', '--verify', 'HEAD')),
        ('HEAD tree', pin['tree'], ('rev-parse', '--verify', 'HEAD^{tree}')),
        ('origin URL', pin['url'], ('config', '--local', '--get', 'remote.origin.url')),
    ):
        actual = git(checkout, *args)
        if actual.returncode or actual.stdout.strip() != expected:
            blockers.append(f'{name}: {label} mismatch')
    status = git(checkout, 'status', '--porcelain=v1', '--untracked-files=normal',
                 '--ignore-submodules=none')
    if status.returncode or status.stdout:
        blockers.append(f'{name}: checkout dirty or status unavailable')
    tree = git(checkout, 'ls-tree', '-rz', 'HEAD')
    if tree.returncode:
        blockers.append(f'{name}: committed tree unavailable')
        return
    regular = set()
    for item in filter(None, tree.stdout.split('\0')):
        identity, path = item.split('\t', 1)
        mode, kind, oid = identity.split()
        if mode == '160000' and kind == 'commit':
            gitlinks[str(PurePosixPath(name) / path)] = oid
        elif mode in ('100644', '100755') and kind == 'blob':
            regular.add(path)

    for path, expected in pin['file_sha256'].items():
        file = checkout / _relative_path(path)
        if path not in regular or file.resolve() != file or not file.is_file():
            blockers.append(f'{name}: locked file unavailable: {path}')
            continue
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            blockers.append(f'{name}: file SHA256 mismatch: {path}')


def inspect_source_lock(parent, lock):
    if lock['schema_version'] != LOCK_SCHEMA:
        raise ValueError('unsupported source lock schema')
    root = Path(parent).resolve()
    repositories = {entry['path']: entry for entry in lock['repositories']}
    backend = lock['cuda_backend_path']
    if '.' not in repositories or backend not in repositories:
        raise ValueError('parent and CUDA backend require repository pins')
    if len(repositories) != len(lock['repositories']):
        raise ValueError('duplicate repository pin')

    blockers = []
    gitlinks = {}
    for relative, pin in repositories.items():
        _inventory(root / _relative_path(relative), pin, blockers, gitlinks)
    locked_gitlinks = {key: pin['commit'] for key, pin in repositories.items() if key != '.'}
    if gitlinks != locked_gitlinks:
        blockers.append('committed recursive gitlinks differ from source lock')
    modules = git(root, 'submodule', 'status', '--recursive')
    if modules.returncode or any(not row.startswith(' ') for row in modules.stdout.splitlines()):
        blockers.append('recursive submodules are uninitialized or differ from their gitlinks')
    cuda_inventory = inspect_source(root / _relative_path(backend),
                                    expected_commit=repositories[backend]['commit'])
    blockers.extend(f'CUDA inventory: {error}' for error in cuda_inventory['blockers'])
    return {
        'source_locked': not blockers,
        'blockers': blockers,
        'gitlinks': gitlinks,
        'cuda_inventory': cuda_inventory,
        'build_verified': False,
        'solver_executed': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    try:
        with args.lock.open(encoding='utf-8') as stream:
            lock = json.load(stream)
        result = inspect_source_lock(args.checkout, lock)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f'Source lock validation failed: {error}\n')
    print(json.dumps(result, indent=2))
    return 0 if result['source_locked'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
