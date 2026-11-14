import subprocess

from scripts.verify_dh116_deformx_cuda_source import inspect_source


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def test_empty_checkout_is_blocked(tmp_path):
    assert inspect_source(tmp_path)['blockers'] == ['not a standalone Git checkout']


def test_pinned_source_requires_bindings_kernels_and_clean_checkout(tmp_path):
    git(tmp_path, 'init', '-q')
    files = {'pyproject.toml': '[build-system]\n', 'CMakeLists.txt': 'project(Fixture)\n',
             'LICENSE': 'Fixture only\n', 'src/solver.cu': '__global__ void fixture() {}\n',
             'src/bindings.cpp': 'PYBIND11_MODULE(rope, m) {}\n'}
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(content, encoding='utf-8')
    git(tmp_path, 'add', '--', *files)
    git(tmp_path, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', 'fixture')
    head = git(tmp_path, 'rev-parse', 'HEAD')
    assert 'target commit not checked out' in inspect_source(tmp_path)['blockers']
    assert inspect_source(tmp_path, expected_commit=head)['blockers'] == []
    (tmp_path / 'src/solver.cu').unlink()
    blockers = inspect_source(tmp_path, expected_commit=head)['blockers']
    assert 'checkout has local changes' in blockers
    assert 'tracked CUDA kernels not found' in blockers

def write_tracked(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(data, encoding='utf-8')


def commit(root, message='fixture'):
    git(root, 'add', '--all')
    git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', message)
    return git(root, 'rev-parse', 'HEAD')


MATERIALIZED = {
    'pyproject.toml': '[build-system]\n',
    'CMakeLists.txt': 'project(Fixture)\n',
    'LICENSE': 'Fixture only\n',
    'src/solver.cu': '__global__ void fixture() {}\n',
    'src/bindings.cpp': 'PYBIND11_MODULE(rope, m) {}\n',
}
LFS_POINTER = ('version https://git-lfs.github.com/spec/v1\n'
               'oid sha256:' + '3'*64 + '\nsize 1234\n')


def test_lfs_pointers_never_count_as_solver_payload(tmp_path):
    git(tmp_path, 'init', '-q')
    files = dict.fromkeys(MATERIALIZED, LFS_POINTER)
    write_tracked(tmp_path, files)
    head = commit(tmp_path)
    result = inspect_source(tmp_path, expected_commit=head)
    assert set(result['unavailable_payloads']) == set(files)
    assert set(result['unavailable_payloads'].values()) == {'git_lfs_pointer'}
    assert not any(result['source_files'].values())
    assert not result['source_inventory_available']
    assert not result['build_verified'] and not result['solver_executed']


def test_one_valid_kernel_does_not_hide_another_missing_payload(tmp_path):
    git(tmp_path, 'init', '-q')
    files = dict(MATERIALIZED, **{'src/secondary.cuh':LFS_POINTER})
    write_tracked(tmp_path, files)
    head = commit(tmp_path)
    result = inspect_source(tmp_path, expected_commit=head)
    assert result['source_files']['cuda'] == ['src/solver.cu']
    assert result['unavailable_payloads'] == {'src/secondary.cuh':'git_lfs_pointer'}
    assert not result['source_inventory_available']


def test_git_symlink_mode_is_rejected_even_with_regular_checkout_stub(tmp_path):
    git(tmp_path, 'init', '-q')
    files = dict(MATERIALIZED)
    del files['src/solver.cu']
    files['src/payload.bin'] = '__global__ void fixture() {}\n'
    write_tracked(tmp_path, files)
    (tmp_path / 'src' / 'solver.cu').symlink_to('payload.bin')
    head = commit(tmp_path)
    result = inspect_source(tmp_path, expected_commit=head)
    assert not result['source_files']['cuda']
    assert result['unavailable_payloads']['src/solver.cu'] == 'tracked_symlink'
    (tmp_path / 'src' / 'solver.cu').unlink()
    git(tmp_path, '-c', 'core.symlinks=false', 'checkout-index', '--force', '--', 'src/solver.cu')
    assert not (tmp_path / 'src' / 'solver.cu').is_symlink()
    result = inspect_source(tmp_path, expected_commit=head)
    assert not result['source_files']['cuda']
    assert result['unavailable_payloads']['src/solver.cu'] == 'tracked_symlink'


def test_directory_symlink_is_not_followed(tmp_path):
    git(tmp_path, 'init', '-q')
    write_tracked(tmp_path, MATERIALIZED)
    head = commit(tmp_path)
    (tmp_path / 'src').rename(tmp_path / 'replacement')
    (tmp_path / 'src').symlink_to('replacement', target_is_directory=True)
    result = inspect_source(tmp_path, expected_commit=head)
    assert result['unavailable_payloads']['src/solver.cu'] == 'symlink_path'
    assert result['source_files']['cuda'] == []


def test_source_in_submodule_counts_only_at_recorded_commit(tmp_path):
    upstream, root = tmp_path/'upstream', tmp_path/'root'
    upstream.mkdir()
    root.mkdir()
    for path in (upstream, root):
        git(path, 'init', '-q')
    write_tracked(upstream, {k:v for k,v in MATERIALIZED.items() if k.startswith('src/')})
    upstream_head = commit(upstream)
    write_tracked(root, {k:v for k,v in MATERIALIZED.items() if not k.startswith('src/')})
    git(root, '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q',
        str(upstream), 'deps/solver')
    root_head = commit(root)
    result = inspect_source(root, expected_commit=root_head)
    assert result['source_inventory_available']
    assert result['source_files']['cuda'] == ['deps/solver/src/solver.cu']
    assert result['submodule_pins']['deps/solver']['expected'] == upstream_head
    assert result['submodule_pins']['deps/solver']['actual'] == upstream_head
    write_tracked(root/'deps/solver', {'src/solver.cu':'__global__ void changed() {}\n'})
    changed = commit(root/'deps/solver', 'different checkout')
    result = inspect_source(root, expected_commit=root_head)
    assert result['submodule_pins']['deps/solver']['expected'] == upstream_head
    assert result['submodule_pins']['deps/solver']['actual'] == changed
    assert not result['source_files']['cuda']
    assert not result['source_inventory_available']
