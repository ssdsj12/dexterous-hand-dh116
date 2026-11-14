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
    monkeypatch.setattr(scheduler, 'NATIVE_SCHEDULER_LOCK', tmp_path / 'scheduler.lock')
    monkeypatch.setattr(scheduler, 'NATIVE_LOCK', tmp_path / 'isaac.lock')
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


def native_fixture(repo, exit_code=0):
    entry = repo.root / 'scripts' / 'native_fixture.py'
    entry.parent.mkdir(exist_ok=True)
    entry.write_text(
        'import json, sys\nfrom pathlib import Path\n'
        'p = Path(sys.argv[-1]); p.mkdir()\n'
        '(p / "summary.json").write_text(json.dumps({"qualified": True}))\n'
        '(p / "physics.jsonl").write_text("{}\\n")\n'
        f'raise SystemExit({exit_code})\n', encoding='utf-8')
    wrapper = repo.root / 'scripts' / 'python.sh'
    wrapper.write_text('exec "$DH116_PYTHON" "$@"\n', encoding='utf-8')
    spec_path = 'configs/dh116_execution/fixture.json'
    write_json(repo.root / spec_path, dict(
        version=1, entrypoint='scripts/native_fixture.py', argv=['--output', '{output}'],
        source_files=[], budget=dict(gpu_index=3, timeout_seconds=3, min_gpu_free_mib=1,
            min_disk_free_bytes=1, physics_steps=1, environments=1, updates=0),
        validator=dict(result_file='summary.json', equals={'qualified': True},
                       artifacts=['summary.json', 'physics.jsonl'], line_counts={'physics.jsonl': 1})))
    native = dict(status='blocked', blocker_kind='waiting_dependency', executor='native_bounded',
        title='Native fixture', dependencies=['wp1'], command_spec=spec_path,
        requires_approval=True, resource_class='isaac_single_gpu', evidence_scope='fixture_native_only',
        dependency_scopes={'wp1': dict(scope='fixture_input', native=True)},
        retry_policy=dict(max_attempts_per_code_config=1, retry_requires_change=True))
    data = repo.registry()
    data['version'] = 2
    data['packages']['native'] = native
    write_json(repo.root / scheduler.REGISTRY, data)
    return native


def native_dependency_pass(repo):
    data = repo.registry()
    data['packages']['wp1'].update(status='passed', last_result=dict(
        native_qualification=True, native_execution_performed=True, qualification_scope='fixture_input',
        outer_exit_code=0))
    write_json(repo.root / scheduler.REGISTRY, data)


def native_approval(repo):
    recipe = repo.native_recipe(repo.registry()['packages']['native'])
    return dict(version=1, batch_id='test-batch', output_root=str(repo.private / 'native-output'),
                packages={'native': {key: recipe[key] for key in (
                    'command_spec_sha256', 'code_config_sha256')}})


def test_native_dependency_scope_and_approval_are_recomputed(repo):
    native_fixture(repo)
    assert repo.native_plan()['blockers']['native']['kind'] == 'waiting_dependency'
    native_dependency_pass(repo)
    assert repo.native_plan()['blockers']['native']['kind'] == 'waiting_approval'
    assert repo.native_plan(approval=native_approval(repo))['native_ready'] == ['native']
    data = repo.registry()
    data['packages']['wp1']['last_result']['native_qualification'] = False
    write_json(repo.root / scheduler.REGISTRY, data)
    assert repo.native_plan(approval=native_approval(repo))['blockers']['native']['kind'] == 'waiting_dependency'


def test_native_failure_artifacts_cannot_override_real_exit_or_enable_retry(repo, monkeypatch):
    native_fixture(repo, exit_code=139)
    native_dependency_pass(repo)
    approval = native_approval(repo)
    monkeypatch.setattr(repo, 'native_resource_blocker', lambda *_: None)
    monkeypatch.setattr(repo, 'native_lock_observed', lambda *_: True)
    intent = repo.register_native_attempt('native', approval)
    result = repo.run_native_worker(intent)
    assert result['outer_exit_code'] == 139
    assert result['native_qualification'] is False
    assert Path(result['output'], 'summary.json').is_file()
    assert repo.native_plan(approval=approval)['blockers']['native']['kind'] == 'failed_native'
    assert repo.native_plan(approval=approval)['native_ready'] == []
    entry = repo.root / 'scripts/native_fixture.py'
    entry.write_text(entry.read_text().replace('SystemExit(139)', 'SystemExit(0)'))
    assert repo.native_plan(approval=approval)['blockers']['native']['kind'] == 'waiting_approval'
    assert repo.native_plan(approval=native_approval(repo))['native_ready'] == ['native']


def test_native_completion_receipt_survives_without_reexecution(repo, monkeypatch):
    native_fixture(repo)
    native_dependency_pass(repo)
    approval = native_approval(repo)
    monkeypatch.setattr(repo, 'native_resource_blocker', lambda *_: None)
    monkeypatch.setattr(repo, 'native_lock_observed', lambda *_: True)
    intent = repo.register_native_attempt('native', approval)
    result = repo.run_native_worker(intent)
    assert result['native_qualification'] is True
    assert result['qualification_scope'] == 'fixture_native_only'
    assert repo.native_plan()['native_results']['native']['result_sha256'] == result['result_sha256']
    with pytest.raises(RuntimeError, match='receipt'):
        repo.run_native_worker(intent)
    assert repo.native_plan(approval=approval)['native_ready'] == []


@pytest.mark.parametrize('changed', ['claimed_success_after_exit_139', 'own_ledger'])
def test_native_resume_revalidates_own_receipt_before_qualification(repo, monkeypatch, changed):
    exit_code = 139 if changed == 'claimed_success_after_exit_139' else 0
    native_fixture(repo, exit_code=exit_code)
    native_dependency_pass(repo)
    approval = native_approval(repo)
    monkeypatch.setattr(repo, 'native_resource_blocker', lambda *_: None)
    monkeypatch.setattr(repo, 'native_lock_observed', lambda *_: True)
    intent = repo.register_native_attempt('native', approval)
    original = repo.run_native_worker(intent)
    receipt = intent.with_name('receipt.json')
    if exit_code:
        rewritten = json.loads(receipt.read_text())
        rewritten.update(native_qualification=True, outer_exit_code=0, exit_code=0)
        rewritten.pop('blocker_kind')
        rewritten.pop('reason')
        recipe = repo.native_recipe(repo.registry()['packages']['native'])
        rewritten.update(scheduler.validate_native_artifacts(recipe['spec'], Path(original['output'])))
        write_json(receipt, rewritten)
    else:
        Path(original['output'], 'physics.jsonl').write_text('{"changed": true}\n')
    bytes_before = receipt.read_bytes()

    resumed = scheduler.Repository(repo.root, evidence_dir=repo.private)
    plan = resumed.native_plan(approval=approval)
    assert plan['native_qualified'] == []
    assert plan['native_ready'] == []
    assert plan['blockers']['native']['kind'] == 'failed_native'
    assert 'native' not in plan['native_results']
    forged = dict(json.loads(bytes_before), result_sha256=scheduler.file_hash(receipt))
    with pytest.raises(RuntimeError, match='durable independently validated receipt'):
        resumed.publish_native('native', forged)
    assert receipt.read_bytes() == bytes_before


def test_native_busy_does_not_consume_attempt_and_cpu_stays_ready(repo, monkeypatch):
    native_fixture(repo)
    native_dependency_pass(repo)
    data = repo.registry()
    data['packages']['cpu'] = dict(status='pending', executor='pytest', dependencies=[], title='CPU')
    write_json(repo.root / scheduler.REGISTRY, data)
    approval = native_approval(repo)
    monkeypatch.setattr(repo, 'native_resource_blocker', lambda *_: 'held global lock')
    intent = repo.register_native_attempt('native', approval)
    result = repo.run_native_worker(intent)
    assert result['blocker_kind'] == 'waiting_resource'
    assert scheduler.scoped_queue(repo.registry())['ready'] == ['cpu']
    assert repo.native_plan(approval=approval)['native_ready'] == ['native']


def test_native_validator_requires_physical_records_and_scope(repo, monkeypatch):
    native_fixture(repo)
    native_dependency_pass(repo)
    recipe = repo.native_recipe(repo.registry()['packages']['native'])
    out = repo.private / 'validator'
    out.mkdir(parents=True)
    (out / 'summary.json').write_text('{"qualified": true}')
    (out / 'physics.jsonl').write_text('[]\n')
    with pytest.raises(ValueError, match='object'):
        scheduler.validate_native_artifacts(recipe['spec'], out)



def test_native_budget_clock_requires_setup_and_measured_steps_to_reconcile(repo, tmp_path):
    native_fixture(repo)
    package = repo.registry()['packages']['native']
    spec_path = repo.root / package['command_spec']
    spec = json.loads(spec_path.read_text())
    spec['budget']['physics_steps'] = 240
    spec['validator']['line_counts']['physics.jsonl'] = 238
    spec['validator']['physics_clock'] = dict(ledger='physics.jsonl', setup_steps=2)
    write_json(spec_path, spec)
    assert repo.native_recipe(package)['spec']['budget']['physics_steps'] == 240

    clock = dict(sdk_initial_step=0, sdk_after_setup_step=2, sdk_first_measurement_step=3,
                 sdk_last_measurement_step=240, sdk_final_step=240, setup_steps=2,
                 measured_steps=238, total_steps=240)
    out = tmp_path / 'physical'
    out.mkdir()

    def write_result(rows, **changes):
        write_json(out / 'summary.json', dict(qualified=True, physics_clock={**clock, **changes}))
        (out / 'physics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    rows = [dict(phase_step=step, sdk_step=step + 2) for step in range(1, 239)]
    write_result(rows)
    assert scheduler.validate_native_artifacts(spec, out)['physics_clock'] == clock
    write_result(rows + [dict(phase_step=239, sdk_step=241), dict(phase_step=240, sdk_step=242)])
    with pytest.raises(ValueError, match='budget'):
        scheduler.validate_native_artifacts(spec, out)
    write_result(rows, sdk_final_step=241)
    with pytest.raises(ValueError, match='budget'):
        scheduler.validate_native_artifacts(spec, out)
    write_result(rows[:72] + [dict(phase_step=73, sdk_step=76)] + rows[73:])
    with pytest.raises(ValueError, match='clock'):
        scheduler.validate_native_artifacts(spec, out)

    spec['validator']['line_counts']['physics.jsonl'] = 240
    write_json(spec_path, spec)
    with pytest.raises(ValueError, match='budget'):
        repo.native_recipe(package)


def test_detailed_native_recipe_freezes_all_phases_and_total_cli_steps(repo):
    native_fixture(repo)
    package = repo.registry()['packages']['native']
    spec_path = repo.root / package['command_spec']
    spec = json.loads(spec_path.read_text())
    spec['budget']['physics_steps'] = 240
    spec['argv'] += ['--steps', '240']
    validator = spec['validator']
    validator['line_counts']['physics.jsonl'] = 238
    validator['artifacts'] += ['shutdown_complete.json', 'ownership.jsonl']
    validator['physics_clock'] = dict(
        ledger='physics.jsonl',
        budget=dict(initialization_physics_steps=2, warmup_physics_steps=0, measurement_physics_steps=238,
                    other_physics_advancing_steps=0, total_physics_steps=240),
        rod_substeps_per_step=12, rod_node_count=21,
        row_equals=dict(side='left', body_path='/link', rod_uid='rod0', mesh_index=0),
        close_marker='shutdown_complete.json', ownership_ledger='ownership.jsonl')
    write_json(spec_path, spec)
    assert repo.native_recipe(package)['spec']['budget']['physics_steps'] == 240
    spec['argv'] += ['--steps=241']
    write_json(spec_path, spec)
    with pytest.raises(ValueError, match='total --steps'):
        repo.native_recipe(package)
    spec['argv'].pop()
    del validator['physics_clock']['budget']['warmup_physics_steps']
    write_json(spec_path, spec)
    with pytest.raises(ValueError, match='declare each phase'):
        repo.native_recipe(package)


def test_budget_clock_new_recipe_cannot_reuse_previous_approval_or_result(repo, monkeypatch):
    native_fixture(repo)
    native_dependency_pass(repo)
    original_approval = native_approval(repo)
    old_path, old = record_qualified_left(repo, monkeypatch, original_approval)

    package = repo.registry()['packages']['native']
    spec_path = repo.root / package['command_spec']
    spec = json.loads(spec_path.read_text())
    spec['budget']['physics_steps'] = 3
    spec['validator']['physics_clock'] = dict(ledger='physics.jsonl', setup_steps=2)
    write_json(spec_path, spec)
    plan = repo.native_plan(approval=original_approval)
    assert plan['native_ready'] == [] and plan['native_qualified'] == []
    assert plan['blockers']['native']['kind'] == 'waiting_approval'
    assert plan['native_identities']['native']['code_config_sha256'] != old['code_config_sha256']
    assert scheduler.file_hash(old_path) == old['result_sha256']


@pytest.mark.parametrize('tamper_before_resume', [False, True])
def test_native_publication_interrupt_resumes_without_a_second_launch(relay_cli, monkeypatch, tamper_before_resume):
    repo = relay_cli
    native_fixture(repo)
    native_dependency_pass(repo)
    git(repo.root, 'add', '--', 'scripts', 'configs', scheduler.REGISTRY)
    git(repo.root, 'commit', '-qm', 'approved fixture recipe')
    git(repo.root, 'update-ref', scheduler.TRACKING_REF, repo.head())
    approval = native_approval(repo)
    monkeypatch.setattr(repo, 'native_resource_blocker', lambda *_: None)
    monkeypatch.setattr(repo, 'native_lock_observed', lambda *_: True)
    result = repo.run_native_worker(repo.register_native_attempt('native', approval))
    with monkeypatch.context() as failed:
        def interrupted(*args, **kwargs):
            raise RuntimeError('local publication interrupted')
        failed.setattr(scheduler.Repository, '_commit_transaction', interrupted)
        with pytest.raises(RuntimeError, match='publication interrupted'):
            repo.publish_native('native', result)
    assert repo.journal.exists()
    if tamper_before_resume:
        previous = repo.head()
        Path(result['output'], 'physics.jsonl').write_text('{"changed": true}\n')
        with pytest.raises(RuntimeError, match='native publication evidence failed revalidation'):
            scheduler.main(['--resume'])
        assert repo.head() == previous and repo.journal.exists()
        assert repo.pending() == []
        return
    assert scheduler.main(['--resume']) == 0
    assert not repo.journal.exists()
    recorded = repo.registry()['packages']['native']['last_result']
    assert recorded['native_qualification'] is True
    assert recorded['result_sha256'] == result['result_sha256']
    assert 'output' not in recorded
    assert repo.native_plan(approval=approval)['native_ready'] == []
    assert len(list(repo.native_attempts('native', result['code_config_sha256']))) == 1


def test_elapsed_budget_preserves_real_child_output_but_blocks_qualification(repo, monkeypatch):
    native_fixture(repo)
    native_dependency_pass(repo)
    path = repo.root / 'configs/dh116_execution/fixture.json'
    spec = json.loads(path.read_text())
    spec['budget']['timeout_seconds'] = 1
    write_json(path, spec)
    entry = repo.root / 'scripts/native_fixture.py'
    entry.write_text('import time; time.sleep(1.08)\n' + entry.read_text())
    approval = native_approval(repo)
    monkeypatch.setattr(repo, 'native_resource_blocker', lambda *_: None)
    monkeypatch.setattr(repo, 'native_lock_observed', lambda *_: True)
    result = repo.run_native_worker(repo.register_native_attempt('native', approval))
    assert result['outer_exit_code'] == 0 and result['duration_seconds'] > 1
    assert result['native_qualification'] is False
    assert Path(result['output'], 'summary.json').exists()
    assert repo.native_plan(approval=approval)['blockers']['native']['kind'] == 'failed_native'


def test_resource_blocker_leaves_cli_cpu_execution_available(relay_cli, monkeypatch):
    repo = relay_cli
    native_fixture(repo)
    native_dependency_pass(repo)
    registry = repo.registry()
    registry['packages']['cpu'] = dict(status='pending', executor='pytest', dependencies=[],
                                       title='Independent CPU', tests=['tests/test_fixture.py'])
    write_json(repo.root / scheduler.REGISTRY, registry)
    git(repo.root, 'add', '--', 'scripts', 'configs', scheduler.REGISTRY)
    git(repo.root, 'commit', '-qm', 'CPU plus native recipe')
    git(repo.root, 'update-ref', scheduler.TRACKING_REF, repo.head())
    approved = repo.private / 'approval.json'
    write_json(approved, native_approval(repo))
    monkeypatch.setattr(scheduler.Repository, 'native_resource_blocker', lambda *_: 'global lock is held')
    monkeypatch.setattr(scheduler.Repository, 'start_native',
                        lambda *_: pytest.fail('busy native package launched a second process'))
    assert scheduler.main(['--run', '--limit', '2', '--native-approval', str(approved)]) == 0
    assert repo.registry()['packages']['cpu']['status'] == 'passed'
    assert repo.native_plan(approval=native_approval(repo), probe_resources=True)['blockers']['native']['kind'] == (
        'waiting_resource')


@pytest.mark.skipif(not Path('/proc/locks').is_file(), reason='Linux lock owner visibility required')
def test_observes_shared_flock_from_real_process_group(repo, monkeypatch, tmp_path):
    import sys
    lock = tmp_path / 'native.lock'
    monkeypatch.setattr(scheduler, 'NATIVE_LOCK', lock)
    command = [sys.executable, '-c',
        'import fcntl, sys; f = open(sys.argv[1], "a+b"); '
        'fcntl.flock(f, fcntl.LOCK_EX); print("ready", flush=True); sys.stdin.readline()',
        str(lock)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                          start_new_session=True) as child:
        assert child.stdout.readline().strip() == 'ready'
        try:
            assert repo.native_lock_observed(child.pid)
        finally:
            child.stdin.write('done\n')
            child.stdin.flush()
    assert child.returncode == 0



def test_duplicate_hidden_worker_cannot_overwrite_same_intent(repo, monkeypatch):
    native_fixture(repo)
    native_dependency_pass(repo)
    approval = native_approval(repo)
    monkeypatch.setattr(repo, 'native_resource_blocker', lambda *_: None)
    intent = repo.register_native_attempt('native', approval)
    monkeypatch.setattr(repo, '_native_child', lambda *_: pytest.fail('duplicate worker launched a child'))
    with scheduler.native_file_lock(intent.parent / 'worker.lock'):
        with pytest.raises(BlockingIOError):
            repo.run_native_worker(intent)
    assert not intent.with_name('receipt.json').exists()



def right_recipe_fixture(repo):
    data = repo.registry()
    left = data['packages']['native']
    right = dict(left, title='Right fixture', command_spec='configs/dh116_execution/right.json',
                 dependencies=['native'], dependency_scopes={'native': dict(scope='fixture_native_only', native=True)},
                 evidence_scope='right_fixture_only')
    spec = json.loads((repo.root / left['command_spec']).read_text())
    spec['dependency_receipt'] = dict(package='native', qualification_scope='fixture_native_only')
    spec['argv'] = ['--left-scheduler-receipt', '{left_scheduler_receipt}',
                    '--left-receipt-sha256', '{left_receipt_sha256}', *spec['argv']]
    write_json(repo.root / right['command_spec'], spec)
    data['packages']['right'] = right
    write_json(repo.root / scheduler.REGISTRY, data)
    return right


def record_qualified_left(repo, monkeypatch, approval):
    monkeypatch.setattr(repo, 'native_resource_blocker', lambda *_: None)
    monkeypatch.setattr(repo, 'native_lock_observed', lambda *_: True)
    intent = repo.register_native_attempt('native', approval)
    result = repo.run_native_worker(intent)
    data = repo.registry()
    data['packages']['native'].update(status='passed', last_result={
        key: value for key, value in result.items() if key != 'output'})
    write_json(repo.root / scheduler.REGISTRY, data)
    return intent.with_name('receipt.json'), result


def test_native_receipt_binding_resolves_frozen_left_and_reuses_after_reopen(repo, monkeypatch):
    native_fixture(repo)
    native_dependency_pass(repo)
    right = right_recipe_fixture(repo)
    approval = native_approval(repo)
    recipe = repo.native_recipe(right)
    approval['packages']['right'] = {key: recipe[key] for key in ('command_spec_sha256', 'code_config_sha256')}
    assert repo.native_plan(approval=approval)['blockers']['right']['kind'] == 'waiting_dependency'
    receipt_path, result = record_qualified_left(repo, monkeypatch, approval)

    resumed = scheduler.Repository(repo.root, evidence_dir=repo.private)
    assert resumed.native_plan(approval=approval)['native_ready'] == ['right']
    intent = resumed.register_native_attempt('right', approval)
    bound = json.loads(intent.read_text())['dependency_receipt']
    assert bound == dict(package='native', qualification_scope='fixture_native_only',
                         path=str(receipt_path), sha256=result['result_sha256'])
    monkeypatch.setattr(resumed, 'native_resource_blocker', lambda *_: None)
    monkeypatch.setattr(resumed, 'native_lock_observed', lambda *_: True)
    right_result = resumed.run_native_worker(intent)
    assert right_result['native_qualification'] is True
    assert right_result['dependency_receipt_sha256'] == result['result_sha256']
    argv = json.loads(intent.with_name('launch_argv.json').read_text())['argv']
    assert argv[argv.index('--left-scheduler-receipt') + 1] == str(receipt_path)
    assert argv[argv.index('--left-receipt-sha256') + 1] == result['result_sha256']
    assert len(list(resumed.native_attempts('native', result['code_config_sha256']))) == 1
    assert resumed.native_plan(approval=approval)['native_qualified'] == ['native', 'right']

    Path(result['output'], 'physics.jsonl').write_text('{"tampered": true}\n')
    invalidated = resumed.native_plan(approval=approval)
    assert invalidated['native_qualified'] == []
    assert invalidated['blockers']['native']['kind'] == 'failed_native'
    assert invalidated['blockers']['right']['kind'] == 'waiting_dependency'
    assert 'right' not in invalidated['native_results']
    assert intent.with_name('receipt.json').exists()


@pytest.mark.parametrize('tamper', ['receipt', 'outer_exit_139', 'close_only', 'ledger'])
def test_native_receipt_binding_rejects_tampered_or_close_only_left(repo, monkeypatch, tamper):
    native_fixture(repo)
    native_dependency_pass(repo)
    right = right_recipe_fixture(repo)
    approval = native_approval(repo)
    recipe = repo.native_recipe(right)
    approval['packages']['right'] = {key: recipe[key] for key in ('command_spec_sha256', 'code_config_sha256')}
    path, result = record_qualified_left(repo, monkeypatch, approval)
    intent = repo.register_native_attempt('right', approval)
    if tamper == 'receipt':
        path.write_text(path.read_text() + '\n')
    elif tamper == 'outer_exit_139':
        outer = path.with_name('outer_exit.json')
        value = json.loads(outer.read_text())
        write_json(outer, dict(value, outer_exit_code=139))
    elif tamper == 'close_only':
        path.with_name('outer_exit.json').unlink()
    else:
        Path(result['output'], 'physics.jsonl').write_text('{"tampered": true}\n')
    with pytest.raises((ValueError, RuntimeError, OSError)):
        repo.run_native_worker(intent)
    assert not intent.with_name('launch_argv.json').exists()



@pytest.mark.skipif(not Path('/usr/bin/gdb').is_file(), reason='Linux debugger unavailable')
@pytest.mark.parametrize('crash', [False, True, 'after_recovered_sigbus'])
def test_native_debugger_preserves_argv_lock_and_real_inferior_status(repo, monkeypatch, crash):
    native_fixture(repo)
    native_dependency_pass(repo)
    monkeypatch.setattr(repo, 'native_resource_blocker', lambda *_: None)
    lock = repo.private / 'debugger-native.lock'
    monkeypatch.setattr(scheduler, 'NATIVE_LOCK', lock)
    spec_path = repo.root / 'configs/dh116_execution/fixture.json'
    spec = json.loads(spec_path.read_text())
    spec.update(debugger='gdb_on_segv', inputs={'/usr/bin/gdb': scheduler.file_hash(Path('/usr/bin/gdb'))})
    spec['argv'] = ['--fixture-words', 'two words and "quoted text"', *spec['argv']]
    write_json(spec_path, spec)
    entry = repo.root / spec['entrypoint']
    source = (
        'import fcntl, json, resource, sys, time\n'
        'resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n'
        f'lock = open({str(lock)!r}, "a+b")\n'
        'fcntl.flock(lock, fcntl.LOCK_EX)\n'
        f'assert sys.argv[1:3] == {spec["argv"][:2]!r}\n'
        'time.sleep(0.25)\n'
    ) + entry.read_text()
    if crash:
        source = source.replace('raise SystemExit(0)', 'import ctypes; ctypes.string_at(0)')
    if crash == 'after_recovered_sigbus':
        source = ('import signal\n'
                  'signal.signal(signal.SIGBUS, lambda *_: print("SIGBUS handled", flush=True))\n'
                  'signal.raise_signal(signal.SIGBUS)\n') + source
    entry.write_text(source)
    approval = native_approval(repo)
    intent = repo.register_native_attempt('native', approval)
    result = repo.run_native_worker(intent)
    assert result['native_lock_observed'] is True
    assert result['debugger_inferior_exit_code'] == (None if crash else 0)
    assert result['debugger_inferior_signal'] == (11 if crash else None)
    assert result['native_qualification'] is (not bool(crash))
    if crash:
        assert result['outer_exit_code'] != 0
        assert result['debugger_backtrace_captured'] is True
        log = intent.with_name('native.log').read_text()
        assert 'DH116_GDB_COMMAND thread apply all bt 20' in log
        assert 'Shared Object Library' in log
        assert 'Mapped address spaces' in log
        assert 'DH116_GDB_COMMAND_ERROR' not in log
        assert 'DH116_GDB_FATAL_BACKTRACE SIGSEGV' in log
        if crash == 'after_recovered_sigbus':
            assert log.index('DH116_GDB_FATAL_BACKTRACE SIGBUS') < log.index('SIGBUS handled')
            assert log.index('SIGBUS handled') < log.index('DH116_GDB_FATAL_BACKTRACE SIGSEGV')
    else:
        assert result['outer_exit_code'] == 0



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


@pytest.fixture
def scoped_relay(relay_cli):
    registry = relay_cli.registry()
    packages = registry['packages']
    packages['solver'] = dict(packages['wp1'], title='Build future solver')
    packages['rigid'] = dict(title='Existing rigid acceptance', executor='gated',
                             status='blocked', dependencies=['wp1'], reason='needs native evidence')
    packages['flexible'] = dict(title='Future flexible acceptance', executor='gated',
                                status='blocked', dependencies=['solver'])
    registry['execution_target'] = dict(name='现有刚性任务完整闭环（复用原环境，不重新建模）',
                                        package='rigid', deferred={'solver': 'future CUDA backend',
                                                                     'flexible': 'future task'})
    write_json(relay_cli.root / scheduler.REGISTRY, registry)
    git(relay_cli.root, 'add', scheduler.REGISTRY)
    git(relay_cli.root, 'commit', '-qm', 'configure scope')
    git(relay_cli.root, 'update-ref', scheduler.TRACKING_REF, relay_cli.head())
    return relay_cli


def test_current_plan_keeps_future_solver_out_of_rigid_queue(scoped_relay, capsys):
    assert scheduler.main(['--plan']) == 0
    current = json.loads(capsys.readouterr().out)
    assert current['execution_target']['package'] == 'rigid'
    assert current['ready'] == ['wp1']
    assert current['blocked'] == {'rigid': 'needs native evidence'}
    assert current['deferred'] == {'solver': 'future CUDA backend', 'flexible': 'future task'}

    assert scheduler.main(['--plan', '--scope', 'all']) == 0
    full = json.loads(capsys.readouterr().out)
    assert full['ready'] == ['wp1', 'solver']
    assert 'flexible' in full['blocked']


def test_scoped_run_never_executes_ready_future_solver(scoped_relay):
    assert scheduler.main(['--run', '--limit', '3']) == 0
    assert [entry['package'] for entry in scoped_relay.pending()] == ['wp1']
    assert scoped_relay.registry()['packages']['solver']['status'] == 'pending'


@pytest.mark.parametrize('deferred', [{'wp1': 'cannot exclude a real dependency'},
                                     {'unknown': 'not registered'},
                                     {'solver': '   '}])
def test_registry_rejects_invalid_scope_without_hiding_dependencies(scoped_relay, deferred):
    registry = scoped_relay.registry()
    registry['execution_target']['deferred'] = deferred
    write_json(scoped_relay.root / scheduler.REGISTRY, registry)
    with pytest.raises(ValueError, match='deferred'):
        scoped_relay.registry()


def test_repository_default_selects_flexible_deformx_and_preserves_rigid_regression():
    root = Path(scheduler.__file__).resolve().parents[1]
    registry = scheduler.Repository(root).registry()
    scoped, description = scheduler.execution_scope(registry)
    assert description["execution_target"]["package"] == "flex.f10-actor-only"
    assert {
        "flex.f0-source-lock", "flex.f1-cuda-build", "flex.f2-standalone-gpu",
        "flex.f3-isaac-rigid", "flex.f4-dh116-collider-offline",
        "flex.f4-dh116-same-rod", "flex.f6-camera-lifecycle",
        "flex.f7-synchronized-data", "flex.f7-physics-state", "flex.f7-stereo-capture",
        "flex.f8-pinch", "flex.f8-tripod", "flex.f8-power", "flex.f8-three-teachers",
        "flex.f9-three-students", "flex.f10-actor-only",
    } <= set(scoped)
    assert "wp6.rigid-native-final" in description["deferred"]
    assert "wp6.rigid-native-final" in registry["packages"]


def test_flexible_capture_gates_follow_task_camera_and_native_cooked_sdf_evidence():
    root = Path(scheduler.__file__).resolve().parents[1]
    registry = scheduler.Repository(root).registry()
    packages = registry["packages"]
    scoped, _ = scheduler.execution_scope(registry)
    assert "flex.f4-cooked-pad-sdf" in packages["flex.f4-dh116-same-rod"]["dependencies"]
    assert set(packages["flex.f7-stereo-capture"]["dependencies"]) == {
        "flex.f7-task-camera-native", "flex.f7-physics-state"}
    assert {"flex.f4-cooked-hull-device-parity", "flex.f4-cooked-pad-sdf",
            "flex.f4-left-contact-native", "flex.f4-right-contact-native"} <= set(scoped)
    assert {"flex.f5-wireseg-pretraining", "flex.f6-render-native"}.isdisjoint(scoped)
