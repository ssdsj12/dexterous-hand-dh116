import hashlib

from scripts.lock_deformx_source import inspect_source_lock
from tests.test_dh116_deformx_cuda_source import MATERIALIZED, commit, git, write_tracked


def test_pins_require_materialized_cuda_and_all_gitlinks(tmp_path):
    parent, cuda, cpu = (tmp_path / name for name in ('parent', 'cuda', 'cpu'))
    for root in (parent, cuda, cpu):
        root.mkdir()
        git(root, 'init', '-q')
    write_tracked(parent, {'LICENSE': 'Parent license\n'})
    write_tracked(cuda, MATERIALIZED)
    write_tracked(cpu, {'LICENSE': 'CPU license\n'})
    commit(cuda)
    commit(cpu)
    for repo, path in ((cuda, 'cuda'), (cpu, 'cpu')):
        git(parent, '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', str(repo), path)
    commit(parent)
    git(parent, 'remote', 'add', 'origin', 'https://example.invalid/parent.git')

    def pin(relative, url):
        root = parent / relative
        license_text = (root / 'LICENSE').read_bytes()
        return {
            'path': relative,
            'url': url,
            'commit': git(root, 'rev-parse', 'HEAD'),
            'tree': git(root, 'rev-parse', 'HEAD^{tree}'),
            'file_sha256': {'LICENSE': hashlib.sha256(license_text).hexdigest()},
        }

    lock = {
        'schema_version': 'dh116-deformx-backend-lock-v1',
        'cuda_backend_path': 'cuda',
        'repositories': [
            pin('.', 'https://example.invalid/parent.git'),
            pin('cuda', str(cuda)),
            pin('cpu', str(cpu)),
        ],
    }
    outcome = inspect_source_lock(parent, lock)
    assert outcome['source_locked'], outcome['blockers']
    assert outcome['cuda_inventory']['source_files']['cuda'] == ['src/solver.cu']
    assert not outcome['build_verified']
    assert not outcome['solver_executed']

    omitted = {**lock, 'repositories': lock['repositories'][:-1]}
    assert not inspect_source_lock(parent, omitted)['source_locked']

    git(parent, 'submodule', 'deinit', '-q', '--', 'cpu')
    outcome = inspect_source_lock(parent, lock)
    assert not outcome['source_locked']
    assert 'cpu: missing standalone checkout' in outcome['blockers']
    git(parent, '-c', 'protocol.file.allow=always', 'submodule', 'update', '--init', '-q', '--', 'cpu')

    (parent / 'cuda' / 'LICENSE').write_text('different license\n')
    outcome = inspect_source_lock(parent, lock)
    assert not outcome['source_locked']
    assert 'cuda: file SHA256 mismatch: LICENSE' in outcome['blockers']


def test_standalone_git_origin_mismatch(tmp_path):
    parent = tmp_path / 'checkout'
    parent.mkdir()
    git(parent, 'init', '-q')
    write_tracked(parent, MATERIALIZED)
    head = commit(parent)
    git(parent, 'remote', 'add', 'origin', 'https://example.invalid/other.git')
    lock = {
        'schema_version': 'dh116-deformx-backend-lock-v1',
        'cuda_backend_path': '.',
        'repositories': [{
            'path': '.',
            'url': 'https://example.invalid/expected.git',
            'commit': head,
            'tree': git(parent, 'rev-parse', 'HEAD^{tree}'),
            'file_sha256': {},
        }],
    }
    outcome = inspect_source_lock(parent, lock)
    assert not outcome['source_locked']
    assert '.: origin URL mismatch' in outcome['blockers']
