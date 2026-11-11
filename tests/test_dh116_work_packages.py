import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import run_dh116_work_packages as scheduler


def test_registry_keeps_blocked_work_out_of_parallel_wave():
    packages = {
        'wp1': dict(status='pending', executor='pytest', dependencies=[]),
        'wp2': dict(status='pending', executor='pytest', dependencies=[]),
        'native': dict(status='blocked', executor='gated', dependencies=[], reason='approval'),
        'later': dict(status='pending', executor='pytest', dependencies=['native']),
        'prepared': dict(status='pending', executor='prepared', dependencies=['wp1']),
    }
    assert scheduler.work_queue(packages)['ready'] == ['wp1', 'wp2']
    assert scheduler.work_queue(packages)['blocked'] == {'native': 'approval', 'later': 'waiting for native',
                                                         'prepared': 'waiting for wp1'}


def test_dependency_cycles_and_unknown_dependencies_are_errors():
    packages = {'a': dict(dependencies=['b']), 'b': dict(dependencies=['a'])}
    with pytest.raises(ValueError, match='cycle'):
        scheduler.check_dependencies(packages)
    with pytest.raises(ValueError, match='unknown'):
        scheduler.check_dependencies({'a': dict(dependencies=['missing'])})


@pytest.mark.parametrize('name', ['../user.py', 'artifacts/raw.md', 'runs/failure.json',
                                  'configs/weights.pt', '.git/config', 'C:/temp/source.py'])
def test_release_paths_reject_raw_private_or_outside_files(name):
    with pytest.raises(ValueError, match='publishable'):
        scheduler.release_path(name)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / 'work'
    remote = tmp_path / 'remote'
    root.mkdir()
    remote.mkdir()
    git(remote, 'init', '--bare', '-q')
    git(root, 'init', '-q', '-b', scheduler.BRANCH)
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'remote', 'add', 'origin', str(remote))
    test = root / 'tests' / 'test_fixture.py'
    test.parent.mkdir()
    test.write_text('def test_success():\n    assert True\n', encoding='utf-8')
    for doc in ('README.md', 'CHANGELOG.md'):
        (root / doc).write_text('existing\n', encoding='utf-8')
    private = root / 'artifacts' / 'protected.json'
    write_json(private, {'protected_source_script_config_sha256': {
        'tests/test_fixture.py': hashlib.sha256(test.read_bytes()).hexdigest()}})
    registry = dict(version=1, protection=dict(inventory='artifacts/protected.json',
        inventory_sha256=hashlib.sha256(private.read_bytes()).hexdigest(), count=1, checkpoints={}),
        packages={'wp1': dict(status='pending', executor='pytest', dependencies=[],
                               tests=['tests/test_fixture.py'], title='Verify fixture')})
    write_json(root / scheduler.REGISTRY, registry)
    write_json(root / scheduler.MANIFEST, dict(version=1, releases=[]))
    write_json(root / scheduler.READINESS, {})
    git(root, 'add', '--', 'docs', 'tests', 'README.md', 'CHANGELOG.md')
    git(root, 'commit', '-qm', 'base')
    git(root, 'push', '-q', 'origin', 'HEAD')
    monkeypatch.setattr(scheduler.Repository, 'verify_target', lambda self: None)
    monkeypatch.setattr(scheduler.Repository, 'verify_local_target', lambda self: None)
    return scheduler.Repository(root)


def test_interrupted_push_resumes_without_duplicate_package_commit(repo, monkeypatch):
    root = repo.root
    initial = repo.head()
    result = repo.run_tests('wp1', repo.registry()['packages']['wp1'])
    real_push = repo.push_verified
    monkeypatch.setattr(repo, 'push_verified', lambda *_: (_ for _ in ()).throw(RuntimeError('network')))
    with pytest.raises(RuntimeError, match='network'):
        repo.publish('wp1', result)
    local_package = repo.head()
    assert local_package != initial and repo.remote_head() == initial
    assert repo.journal.exists()
    monkeypatch.setattr(repo, 'push_verified', real_push)
    repo.resume()
    assert not repo.journal.exists() and repo.remote_head() == repo.head()
    assert git(root, 'rev-parse', 'HEAD^') == local_package
    receipt, = json.loads((root / scheduler.MANIFEST).read_text())['releases']
    assert receipt['commit'] == local_package and receipt['passed'] == 1
    assert repo.registry()['packages']['wp1']['status'] == 'passed'


def test_dirty_user_files_block_execution_and_remain_untouched(repo):
    user = repo.root / 'user-notes.md'
    user.write_text('keep me', encoding='utf-8')
    with pytest.raises(RuntimeError, match='working tree'):
        repo.require_scope(set())
    assert user.read_text(encoding='utf-8') == 'keep me'


def test_failed_verification_records_log_and_no_success_receipt(repo):
    package = dict(repo.registry()['packages']['wp1'], tests=['tests/missing.py'])
    result = repo.run_tests('wp1', package)
    assert result['exit_code'] != 0 and Path(result['log']).is_file()
    repo.publish('wp1', result)
    receipt, = json.loads((repo.root / scheduler.MANIFEST).read_text())['releases']
    assert receipt['status'] == 'failed' and receipt['passed'] == 0
    assert repo.registry()['packages']['wp1']['status'] == 'failed'


def test_local_commits_keep_remote_receipts_pending_until_readback(repo):
    initial = repo.head()
    result = repo.run_tests('wp1', repo.registry()['packages']['wp1'])
    receipt = repo.publish('wp1', result, defer=True)
    assert receipt['remote_verified'] is False
    assert repo.remote_head() == initial
    assert repo.pending()[0]['commit'] == repo.head()
    assert json.loads((repo.root / scheduler.MANIFEST).read_text())['releases'] == []
    repo.resume()
    assert repo.remote_head() == repo.head() and repo.pending() == []
    remote_receipt, = json.loads((repo.root / scheduler.MANIFEST).read_text())['releases']
    assert remote_receipt['commit'] == receipt['commit'] and remote_receipt['remote_verified'] is True


def test_scope_does_not_accept_unstaged_renames_or_tracked_private_assets(repo):
    old = repo.root / 'tests' / 'test_fixture.py'
    old.rename(old.with_name('test_moved.py'))
    with pytest.raises(RuntimeError, match='working tree'):
        repo.require_scope({'tests/test_moved.py'})


def test_local_publisher_acknowledges_code_then_manifest_without_new_commit(repo):
    result = repo.run_tests('wp1', repo.registry()['packages']['wp1'])
    pending = repo.publish('wp1', result, defer=True)
    plan = repo.relay_plan()
    receipt = {**plan, 'remote_head': plan['head'], 'draft': True, 'merged': False}
    with pytest.raises(RuntimeError, match='differs'):
        repo.accept_relay_receipt(dict(receipt, draft=False))
    repo.push_verified(pending['parent'], pending['commit'])
    metadata = repo.accept_relay_receipt(receipt)
    assert metadata['attestation_only'] is True and repo.remote_head() == pending['commit']
    repo.push_verified(metadata['parent'], metadata['commit'])
    tracking = 'refs/remotes/origin/' + scheduler.BRANCH
    git(repo.root, 'update-ref', tracking, pending['parent'])
    plan = repo.relay_plan()
    repo.accept_relay_receipt({**plan, 'remote_head': plan['head'], 'draft': True, 'merged': False})
    assert repo.pending() == [] and repo.head() == repo.remote_head() == metadata['commit']
    assert git(repo.root, 'rev-parse', tracking) == metadata['commit']
    release, = json.loads((repo.root / scheduler.MANIFEST).read_text())['releases']
    assert release['commit'] == pending['commit'] and release['verification_route'] == 'local_git_relay'


@pytest.fixture
def relay_cli(repo, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail('Ubuntu scheduler attempted GitHub access')

    monkeypatch.setattr(scheduler.Repository, 'verify_target', no_network)
    monkeypatch.setattr(scheduler.Repository, 'remote_head', no_network)
    monkeypatch.setattr(scheduler.Repository, 'push_verified', no_network)
    monkeypatch.setattr(scheduler, '__file__', str(repo.root / 'scripts' / 'run_dh116_work_packages.py'))
    return repo


def test_relay_cli_plan_uses_only_local_tracking_ref(relay_cli, capsys):
    assert scheduler.main(['--plan']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['publication_route'] == 'windows_relay'
    assert plan['last_verified_remote'] == relay_cli.head()


def test_relay_cli_run_commits_to_queue_without_github_access(relay_cli):
    initial = relay_cli.head()
    assert scheduler.main(['--run']) == 0
    receipt, = relay_cli.pending()
    assert relay_cli.head() == receipt['commit']
    assert git(relay_cli.root, 'rev-parse', 'refs/remotes/origin/' + scheduler.BRANCH) == initial
    assert receipt['passed'] == 1 and receipt['remote_verified'] is False
    assert json.loads((relay_cli.root / scheduler.MANIFEST).read_text())['releases'] == []


def test_relay_cli_rejects_local_commits_outside_pending_queue(relay_cli):
    git(relay_cli.root, 'commit', '-qm', 'independent local change', '--allow-empty')
    with pytest.raises(RuntimeError, match='last verified remote'):
        scheduler.main(['--run'])
    assert relay_cli.pending() == []


def test_relay_cli_resume_reports_queued_work_without_flushing(relay_cli, capsys):
    result = relay_cli.run_tests('wp1', relay_cli.registry()['packages']['wp1'])
    pending = relay_cli.publish('wp1', result, defer=True)
    capsys.readouterr()
    assert scheduler.main(['--resume']) == 0
    status = json.loads(capsys.readouterr().out)
    assert status['publication_route'] == 'windows_relay'
    assert status['pending_publications'] == 1
    assert relay_cli.pending() == [pending]


def test_relay_cli_resume_rejects_direct_publication_journal(relay_cli):
    saved = {'kind': 'package'}
    write_json(relay_cli.journal, saved)
    with pytest.raises(RuntimeError, match='Windows relay'):
        scheduler.main(['--resume'])
    assert json.loads(relay_cli.journal.read_text()) == saved


def test_relay_cli_resume_completes_interrupted_local_commit(relay_cli, monkeypatch):
    result = relay_cli.run_tests('wp1', relay_cli.registry()['packages']['wp1'])
    def interrupted(*args, **kwargs):
        raise RuntimeError('interrupted local commit')

    with monkeypatch.context() as attempt:
        attempt.setattr(scheduler.Repository, '_commit_transaction', interrupted)
        with pytest.raises(RuntimeError, match='interrupted local commit'):
            relay_cli.publish('wp1', result, defer=True)
    assert relay_cli.journal.exists()
    assert scheduler.main(['--resume']) == 0
    assert not relay_cli.journal.exists()
    receipt, = relay_cli.pending()
    assert relay_cli.head() == receipt['commit']


def test_native_student_task_gates_do_not_require_a_solver_for_rigid_runs():
    root = Path(scheduler.__file__).resolve().parents[1]
    packages = scheduler.Repository(root).registry()['packages']

    def dependencies(package):
        direct = packages[package]['dependencies']
        return set(direct).union(*(dependencies(name) for name in direct))

    rigid = dependencies('wp6.rigid-native-final')
    assert 'wp2.rigid-qualified-labels' in rigid
    assert 'wp6.independent-proof' in rigid
    assert 'wp3.frozen-eval' in rigid
    assert 'wp5.solver-native' not in rigid
    assert 'wp5.solver-native' in dependencies('wp6.deformable-native-final')
    assert 'wp2.qualified-labels' in dependencies('wp6.deformable-native-final')
    assert set(packages['wp6.native-final']['dependencies']) == {
        'wp6.rigid-native-final', 'wp6.deformable-native-final',
    }
    for name in ('wp6.rigid-native-final', 'wp6.deformable-native-final'):
        assert packages[name]['executor'] == 'gated'


def test_r0_training_does_not_require_r1_r2_contact_qualification():
    root = Path(scheduler.__file__).resolve().parents[1]
    packages = scheduler.Repository(root).registry()['packages']
    assert 'wp3.mpc-native' in packages['wp3.actor-bc-r0']['dependencies']
    assert 'wp4.contact-native' not in packages['wp3.actor-bc-r0']['dependencies']
    assert packages['wp3.actor-bc-r0']['executor'] == 'gated'
    assert 'wp4.contact-native' in packages['wp3.frozen-eval']['dependencies']
