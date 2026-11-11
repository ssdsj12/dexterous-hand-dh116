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
