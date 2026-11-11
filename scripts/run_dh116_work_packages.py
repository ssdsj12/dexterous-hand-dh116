"""Run DH116 CPU work packages and stage verified Windows relay publication."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile, mkdtemp
from urllib.request import urlopen


BRANCH = 'codex/dh116-stereo-privileged-dagger'
TRACKING_REF = f'refs/remotes/origin/{BRANCH}'
PROJECT = 'ssdsj12/dexterous-hand-dh116'
BASE_SHA = '6cee7418a5e49f56b7f0090fc5eaf20b143c0b1b'
REGISTRY = 'docs/contracts/dh116_work_package_registry.json'
MANIFEST = 'docs/contracts/dh116_release_manifest.json'
READINESS = 'docs/contracts/stereo_phase0_7_readiness.json'
GENERATED = {REGISTRY, MANIFEST, READINESS, 'README.md', 'CHANGELOG.md'}
PRIVATE = 'artifacts/dh116-work-packages'
SHA = re.compile(r'^[0-9a-f]{40}$')
ID = re.compile(r'^[a-z0-9][a-z0-9._-]{0,70}$')


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text.encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_hash(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def release_path(name):
    path = PurePosixPath(name)
    if (name != path.as_posix() or path.is_absolute() or not path.parts or '..' in path.parts
            or '\\' in name or ':' in name or path.suffix not in {'.py', '.json', '.md'}
            or path.parts[0] not in {'src', 'configs', 'scripts', 'tests', 'docs'} and name not in GENERATED):
        raise ValueError(f'not a publishable work-package path: {name}')
    return name


def check_dependencies(packages):
    completed, visiting = set(), set()

    def visit(name):
        if name not in packages:
            raise ValueError(f'unknown work-package dependency: {name}')
        if name in visiting:
            raise ValueError(f'dependency cycle at {name}')
        if name in completed:
            return
        visiting.add(name)
        for dependency in packages[name]['dependencies']:
            visit(dependency)
        visiting.remove(name)
        completed.add(name)

    for name in packages:
        visit(name)


def work_queue(packages):
    check_dependencies(packages)
    ready, prepared, blocked = [], [], {}
    for name, package in packages.items():
        if package['status'] == 'passed':
            continue
        if package['status'] in {'blocked', 'failed'} or package['executor'] == 'gated':
            blocked[name] = package.get('reason', 'execution gate is closed')
            continue
        waiting = [dep for dep in package['dependencies'] if packages[dep]['status'] != 'passed']
        if waiting:
            blocked[name] = 'waiting for ' + ', '.join(waiting)
        elif package['executor'] == 'pytest':
            ready.append(name)
        elif package['executor'] == 'prepared':
            prepared.append(name)
        else:
            raise ValueError(f'unknown work-package executor: {name}')
    return dict(ready=ready, needs_prepared_changes=prepared, blocked=blocked)


def readiness_text(path, package_id, entry):
    text = path.read_bytes().decode('utf-8').rstrip()
    payload = json.loads(text)
    key = 'autonomous_work_packages'
    existing = payload.get(key, {})
    existing[package_id] = entry
    new = json_text(existing).rstrip().replace('\n', '\n  ')
    token = f'  "{key}": '
    if token in text:
        start = text.index(token) + len(token)
        _, consumed = json.JSONDecoder().raw_decode(text[start:])
        return text[:start] + new + text[start+consumed:] + '\n'
    prefix = text[:-1].rstrip()
    return prefix + (',' if payload.keys() - {key} else '') + '\n' + token + new + '\n}\n'


class Repository:
    def __init__(self, root, *, github_ip=None, evidence_dir=None):
        self.root = Path(root).resolve()
        self.private = Path(evidence_dir).resolve() if evidence_dir else self.root / PRIVATE
        self.journal = self.private / 'transaction.json'
        self.outbox = self.private / 'unpublished.json'
        self.git_options = ['-c', f'safe.directory={self.root}', '-c', 'http.sslVerify=true',
                            '-c', 'http.version=HTTP/1.1']
        if github_ip is not None:
            address = ipaddress.IPv4Address(github_ip)
            self.git_options += ['-c', f'http.curloptResolve=github.com:443:{address}']

    def git(self, *args):
        env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
        env.pop('GIT_SSL_NO_VERIFY', None)
        result = subprocess.run(['git', *self.git_options, '-C', str(self.root), *args],
                                env=env, capture_output=True, encoding='utf-8', timeout=75)
        if result.returncode:
            raise RuntimeError(f'git {args[0]} failed ({result.returncode}): {result.stderr.strip()}')
        return result.stdout

    def head(self):
        return self.git('rev-parse', 'HEAD').strip()

    def verify_local_target(self):
        urls = {f'https://github.com/{PROJECT}.git', f'https://github.com/{PROJECT}',
                f'git@github.com:{PROJECT}.git', f'ssh://git@github.com/{PROJECT}.git'}
        if (self.git('branch', '--show-current').strip() != BRANCH
                or any(url.strip() not in urls for url in (
                    self.git('remote', 'get-url', 'origin'), self.git('remote', 'get-url', '--push', 'origin')))):
            raise RuntimeError('expected DH116 origin and development branch')
    def verify_target(self):
        self.verify_local_target()
        with urlopen(f'https://api.github.com/repos/{PROJECT}/pulls/2', timeout=12) as stream:
            pr = json.load(stream)
        if (pr.get('draft') is not True or pr.get('merged') is not False or pr.get('state') != 'open'
                or pr['head']['ref'] != BRANCH or pr['head']['repo']['full_name'] != PROJECT
                or pr['base']['ref'] != '11-10'):
            raise RuntimeError('expected open Draft PR #2 targeting 11-10')

    def remote_head(self):
        ref = f'refs/heads/{BRANCH}'
        fields = self.git('ls-remote', '--exit-code', '--refs', 'origin', ref).split()
        if len(fields) != 2 or fields[1] != ref or not SHA.fullmatch(fields[0]):
            raise RuntimeError('development remote HEAD is unavailable')
        return fields[0]

    def push_verified(self, parent, commit):
        self.verify_target()
        remote = self.remote_head()
        ref = f'refs/heads/{BRANCH}'
        if remote not in (parent, commit):
            self.git('fetch', '--no-tags', 'origin', ref)
            raise RuntimeError(f'development remote advanced to {remote}; inspect fetched divergence')
        if remote != commit:
            self.git('push', 'origin', f'{commit}:{ref}')
        self.git('fetch', '--no-tags', 'origin', ref)
        if self.git('rev-parse', 'FETCH_HEAD').strip() != commit or self.remote_head() != commit:
            raise RuntimeError('fetched remote commit differs from publication')
        self.git('diff', '--exit-code', commit, 'FETCH_HEAD')
        self.verify_target()

    def registry(self):
        data = json.loads((self.root / REGISTRY).read_text(encoding='utf-8'))
        if data.get('version') != 1 or not isinstance(data.get('packages'), dict):
            raise ValueError('versioned work-package registry required')
        for name, package in data['packages'].items():
            if (not ID.fullmatch(name) or package['status'] not in {'pending', 'blocked', 'failed', 'passed'}
                    or package['executor'] not in {'pytest', 'prepared', 'gated'}
                    or not isinstance(package['dependencies'], list)
                    or not isinstance(package.get('title'), str) or '\n' in package['title']):
                raise ValueError(f'invalid work-package definition: {name}')
        check_dependencies(data['packages'])
        return data

    def require_scope(self, expected, *, allow_staged=False):
        observed = set()
        if self.git('ls-files', '--', 'artifacts/').strip():
            raise RuntimeError('working tree has tracked private artifacts')
        entries = self.git('status', '--porcelain=v1', '-z', '--untracked-files=all', '--no-renames',
                           '--', '.', ':(exclude)artifacts/**').split('\0')
        for row in filter(None, entries):
            state, path = row[:2], row[3:]
            if state == '??' and path.startswith('artifacts/'):
                continue
            if (state not in {'??', ' M', 'M ', 'MM', 'A '} or state[0] not in (' ', '?') and not allow_staged):
                raise RuntimeError(f'working tree has unsupported or staged changes: {path}')
            observed.add(path)
        if observed != set(expected):
            raise RuntimeError(f'working tree scope differs: {sorted(observed ^ set(expected))}')

    def verify_protection(self):
        protection = self.registry()['protection']
        inventory = self.root / protection['inventory']
        if file_hash(inventory) != protection['inventory_sha256']:
            raise RuntimeError('protected source inventory missing or changed')
        files = json.loads(inventory.read_text(encoding='utf-8'))['protected_source_script_config_sha256']
        if len(files) != protection['count']:
            raise RuntimeError('protected source count differs')
        for name, digest in files.items():
            path = self.root / name
            if not path.resolve().is_relative_to(self.root) or file_hash(path) != digest:
                raise RuntimeError(f'protected source changed or missing: {name}')
        for name, digest in protection['checkpoints'].items():
            if file_hash(self.root / name) != digest:
                raise RuntimeError(f'historical Checkpoint changed or missing: {name}')

    def run_tests(self, name, package):
        tests = package.get('tests')
        timeout = package.get('timeout_seconds', 180)
        if (not tests or not isinstance(tests, list) or type(timeout) is not int or not 1 <= timeout <= 900
                or any(not release_path(path).startswith('tests/') or not path.endswith('.py') for path in tests)):
            raise ValueError(f'invalid bounded pytest recipe: {name}')
        self.private.mkdir(parents=True, exist_ok=True)
        tested_head = self.head()
        path = Path(mkdtemp(prefix=f'{name}-', dir=self.private)) / 'pytest.log'
        command = ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests]
        env = {**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'PYTHONDONTWRITEBYTECODE': '1',
               'PYTHONUTF8': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': ''}
        python_paths = [str(self.root / 'src'), str(self.root)]
        if env.get('PYTHONPATH'):
            python_paths.append(env['PYTHONPATH'])
        env['PYTHONPATH'] = os.pathsep.join(python_paths)
        with path.open('xb') as stream:
            try:
                result = subprocess.run([sys.executable, *command], cwd=self.root, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
                stream.write(b'\nCPU verification timed out.\n')
        output = path.read_text(encoding='utf-8', errors='replace')
        counts = {key: max((int(n) for n in re.findall(rf'(\d+) {key}\b', output)), default=0)
                  for key in ('passed', 'failed', 'skipped')}
        if code == 0 and counts['passed'] == 0:
            code = 2
        if self.head() != tested_head:
            raise RuntimeError('HEAD changed during CPU verification')
        return dict(**counts, exit_code=code, command=['python', *command],
                    log=str(path), log_sha256=file_hash(path), tested_head=tested_head)

    def local_tracking_head(self):
        return self.git('for-each-ref', '--format=%(objectname)', TRACKING_REF).strip() or None

    def plan(self):
        self.verify_local_target()
        return dict(branch=BRANCH, head=self.head(), publication_route='windows_relay',
                    last_verified_remote=self.local_tracking_head(),
                    working_tree=self.git('status', '--short').splitlines(),
                    resume_pending=self.journal.exists(), pending_publications=len(self.pending()),
                    **work_queue(self.registry()['packages']))

    def pending(self):
        if not self.outbox.exists():
            return []
        return json.loads(self.outbox.read_text(encoding='utf-8'))

    def _transaction(self, kind, package_id, title, parent, generated, files, retained, evidence):
        writes = {name: {'before': file_hash(self.root / name), 'text': value}
                  for name, value in generated.items()}
        return dict(kind=kind, package=package_id, title=title, parent=parent,
                    writes=writes, prepared={name: file_hash(self.root / name) for name in files},
                    retained={name: file_hash(self.root / name) for name in retained},
                    expected_tree=None, evidence=evidence)

    def _commit_transaction(self, state, *, push=True):
        parent = state['parent']
        changed = set(state['writes']) | set(state['prepared'])
        retained = set(state['retained'])
        head = self.head()
        if head == parent:
            for name, digest in {**state['prepared'], **state['retained']}.items():
                if file_hash(self.root / name) != digest:
                    raise RuntimeError(f'prepared work changed before publication: {name}')
            for name, value in state['writes'].items():
                path = self.root / name
                current = file_hash(path)
                target = hashlib.sha256(value['text'].encode('utf-8')).hexdigest()
                if current not in (value['before'], target):
                    raise RuntimeError(f'generated report was edited: {name}')
                if current != target:
                    atomic_write(path, value['text'])
            self.require_scope(changed | retained, allow_staged=True)
            self.git('add', '--', *sorted(changed))
            self.git('diff', '--cached', '--check')
            if set(self.git('diff', '--cached', '--name-only').splitlines()) != changed:
                raise RuntimeError('staged changes differ from the package scope')
            tree = self.git('write-tree').strip()
            if state['expected_tree'] not in (None, tree):
                raise RuntimeError('staged tree differs from interrupted transaction')
            state['expected_tree'] = tree
            atomic_write(self.journal, json_text(state))
            self.verify_protection()
            self.git('commit', '-m', state['title'])
            head = self.head()
        if (self.git('show', '-s', '--format=%P', head).strip() != parent
                or self.git('show', '-s', '--format=%T', head).strip() != state['expected_tree']
                or set(self.git('diff', '--name-only', parent, head).splitlines()) != changed):
            raise RuntimeError('HEAD is not the expected package or manifest commit')
        self.require_scope(retained)
        self.verify_protection()
        if push:
            self.push_verified(parent, head)
        return head, sorted(changed)

    def publish(self, name, result, prepared=(), *, defer=False):
        if self.journal.exists():
            raise RuntimeError('resume the pending publication first')
        parent = self.head()
        if defer:
            self.verify_local_target()
            pending = self.pending()
            if pending and pending[-1]['commit'] != parent:
                raise RuntimeError('HEAD differs from the deferred publication queue')
        else:
            self.verify_target()
            if self.pending() or self.remote_head() != parent:
                raise RuntimeError('local and remote HEAD must agree before publication')
        self.require_scope(set(prepared))
        self.verify_protection()
        if result['tested_head'] != parent and self.git('diff', '--name-only', result['tested_head'], parent, '--',
                                                        'src', 'scripts', 'tests', 'configs').strip():
            raise RuntimeError('code changed after the CPU verification wave')
        registry = self.registry()
        package = registry['packages'][name]
        status = 'passed' if result['exit_code'] == 0 else 'failed'
        report = f'docs/superpowers/reports/dh116-managed-{name}.md'
        entry = {key: value for key, value in result.items() if key != 'log'}
        entry.update(status=status, report=report, native_qualification=False)
        package['status'] = status
        package['last_result'] = entry
        if status == 'failed':
            package['reason'] = 'CPU regression did not pass; evidence retained outside Git'
        followups = [key for key, item in registry['packages'].items() if name in item['dependencies']]
        report_text = (f'# {package["title"]}\n\n'
                       f'基线 `{result["tested_head"]}`。CPU 状态 `{status}`，通过 {result["passed"]} 项，'
                       f'失败 {result["failed"]} 项，跳过 {result["skipped"]} 项，退出码 {result["exit_code"]}。'
                       '此项不授予原生或训练资格。\n\n'
                       f'命令 `{" ".join(result["command"])}`。私有日志 SHA256 `{result["log_sha256"]}`。'
                       f'后继工作包 {", ".join(followups) or "按登记表继续就绪任务"}。\n')
        if status == 'failed':
            report_text += '\nCPU 配方未通过，保留原始日志与本地代码，依赖任务暂停。\n'
        date = datetime.now(timezone.utc).date().isoformat()
        note = (f'## {date} — {package["title"]}\n\n'
                f'CPU `{status}`，通过 {result["passed"]} 项。详细结果见[报告]({report})。\n\n')
        writes = {REGISTRY: json_text(registry), report: report_text,
                  READINESS: readiness_text(self.root / READINESS, name, entry)}
        for doc in ('README.md', 'CHANGELOG.md'):
            writes[doc] = note + (self.root / doc).read_bytes().decode('utf-8')
        selected = prepared if status == 'passed' else ()
        retained = prepared if status == 'failed' else ()
        state = self._transaction('deferred_package' if defer else 'package', name,
                                  f'{package["title"]} [{name}]', parent, writes, selected, retained, entry)
        atomic_write(self.journal, json_text(state))
        return self.resume()

    def resume(self):
        if not self.journal.exists():
            return self._flush_outbox()
        state = json.loads(self.journal.read_text(encoding='utf-8'))
        if state['kind'] in {'deferred_package', 'relay_manifest'}:
            self.verify_local_target()
            commit, files = self._commit_transaction(state, push=False)
            receipt = {**state['evidence'], 'package': state['package'], 'commit': commit,
                       'parent': state['parent'], 'files': files, 'remote_verified': False}
            if state['kind'] == 'relay_manifest':
                receipt['attestation_only'] = True
                atomic_write(self.outbox, json_text([receipt]))
                self.journal.unlink()
                print(json_text(receipt), end='', flush=True)
                return receipt
            pending = self.pending()
            if not pending or pending[-1]['commit'] != commit:
                if pending and pending[-1]['commit'] != state['parent']:
                    raise RuntimeError('deferred queue has advanced unexpectedly')
                atomic_write(self.outbox, json_text([*pending, receipt]))
            self.journal.unlink()
            print(json_text(receipt), end='', flush=True)
            return receipt
        self.verify_target()
        if state['kind'] == 'package':
            commit, files = self._commit_transaction(state)
            manifest = json.loads((self.root / MANIFEST).read_text(encoding='utf-8'))
            receipt = {**state['evidence'], 'package': state['package'], 'commit': commit,
                       'files': files, 'remote_verified': True}
            manifest['releases'].append(receipt)
            state = self._transaction('manifest', state['package'], f'Attest remote release {state["package"]}',
                                      commit, {MANIFEST: json_text(manifest)}, (), state['retained'], receipt)
            atomic_write(self.journal, json_text(state))
        if state['kind'] != 'manifest':
            raise RuntimeError('unknown transaction phase')
        manifest_commit, _ = self._commit_transaction(state)
        if state.get('clear_outbox') and self.outbox.exists():
            self.outbox.unlink()
        self.journal.unlink()
        result = {**state['evidence'], 'manifest_commit': manifest_commit}
        print(json_text(result), end='', flush=True)
        return result


    def relay_plan(self):
        self.verify_local_target()
        if self.journal.exists():
            raise RuntimeError('resume the interrupted local transaction before relaying')
        self.require_scope(set())
        self.verify_protection()
        pending = self.pending()
        if not pending or self.head() != pending[-1]['commit']:
            raise RuntimeError('no matching pending local commits to relay')
        expected = pending[0]['parent']
        commits = []
        for entry in pending:
            commit = entry['commit']
            if (entry['parent'] != expected
                    or self.git('show', '-s', '--format=%P', commit).strip() != expected
                    or sorted(self.git('diff', '--name-only', expected, commit).splitlines()) != entry['files']):
                raise RuntimeError('deferred commit chain or file scope differs')
            commits.append(dict(commit=commit, parent=expected, files=entry['files'],
                                tree=self.git('show', '-s', '--format=%T', commit).strip()))
            expected = commit
        return dict(project=PROJECT, branch=BRANCH, base_sha=BASE_SHA, head=self.head(), commits=commits)

    def accept_relay_receipt(self, receipt):
        expected = self.relay_plan()
        if receipt != {**expected, 'remote_head': expected['head'], 'draft': True, 'merged': False}:
            raise RuntimeError('local publisher evidence differs from the pending commit chain')
        pending = self.pending()
        verified = [dict(item, remote_verified=True, verification_route='local_git_relay')
                    for item in pending if not item.get('attestation_only')]
        if not verified:
            previous = self.local_tracking_head()
            head = expected['head']
            if previous is not None and previous != head:
                self.git('merge-base', '--is-ancestor', previous, head)
            self.git('update-ref', TRACKING_REF, head, previous or '0' * 40)
            self.outbox.unlink()
            return dict(remote_verified=True, head=expected['head'], pending_publications=0)
        manifest = json.loads((self.root / MANIFEST).read_text(encoding='utf-8'))
        seen = {item['commit'] for item in manifest['releases']}
        if any(item['commit'] in seen for item in verified):
            raise RuntimeError('duplicate remote receipt')
        manifest['releases'].extend(verified)
        state = self._transaction('relay_manifest', 'release-manifest', 'Attest verified local GitHub releases',
                                  expected['head'], {MANIFEST: json_text(manifest)}, (), (),
                                  {'verified_release_count': len(verified)})
        atomic_write(self.journal, json_text(state))
        return self.resume()

    def _flush_outbox(self):
        pending = self.pending()
        if not pending:
            raise RuntimeError('no interrupted or deferred publication')
        self.verify_target()
        self.require_scope(set())
        self.verify_protection()
        if self.head() != pending[-1]['commit']:
            raise RuntimeError('HEAD differs from the deferred publication queue')
        expected = pending[0]['parent']
        for receipt in pending:
            if (receipt['parent'] != expected
                    or self.git('show', '-s', '--format=%P', receipt['commit']).strip() != expected
                    or sorted(self.git('diff', '--name-only', expected, receipt['commit']).splitlines()) != receipt['files']):
                raise RuntimeError('deferred commit chain or file scope differs')
            expected = receipt['commit']
        remote = self.remote_head()
        history = [pending[0]['parent'], *[item['commit'] for item in pending]]
        if remote not in history:
            self.git('fetch', '--no-tags', 'origin', f'refs/heads/{BRANCH}')
            raise RuntimeError('remote advanced outside the deferred queue')
        for receipt in pending[history.index(remote):]:
            self.push_verified(receipt['parent'], receipt['commit'])
        self.push_verified(pending[-1]['parent'], pending[-1]['commit'])
        manifest = json.loads((self.root / MANIFEST).read_text(encoding='utf-8'))
        manifest['releases'].extend(dict(item, remote_verified=True) for item in pending)
        state = self._transaction('manifest', 'deferred-batch', 'Attest deferred DH116 releases', expected,
            {MANIFEST: json_text(manifest)}, (), (), {'verified_releases':len(pending), 'commit':expected})
        state['clear_outbox'] = True
        atomic_write(self.journal, json_text(state))
        return self.resume()


@contextmanager
def publication_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'scheduler.lock').open('a+b') as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--prepared', metavar='PACKAGE')
    modes.add_argument('--resume', action='store_true')
    modes.add_argument('--relay-plan', action='store_true')
    modes.add_argument('--relay-ack', action='store_true', help='Read verified local publisher receipt from stdin')
    parser.add_argument('--file', action='append', default=[])
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--local-only', action='store_true',
                        help='Compatibility option; the scheduler always queues commits for Windows publication')
    parser.add_argument('--evidence-dir', type=Path, help='Separate private directory for logs and recovery journal')
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 12 or not 1 <= args.jobs <= 4 or bool(args.file) != bool(args.prepared):
        parser.error('use 1..12 packages and 1..4 CPU jobs; --prepared needs explicit --file paths')
    repo = Repository(Path(__file__).resolve().parents[1], evidence_dir=args.evidence_dir)
    if args.relay_plan:
        print(json_text(repo.relay_plan()), end='')
        return 0
    if not (args.run or args.prepared or args.resume or args.relay_ack):
        print(json_text(repo.plan()), end='')
        return 0
    with publication_lock(repo.private):
        if args.relay_ack:
            print(json_text(repo.accept_relay_receipt(json.load(sys.stdin))), end='')
            return 0
        if args.resume:
            if not repo.journal.exists():
                pending = repo.pending()
                if not pending:
                    raise RuntimeError('no interrupted local commit or queued Windows publication')
                print(json_text(dict(publication_route='windows_relay',
                                     pending_publications=len(pending), head=repo.head())), end='')
                return 0
            kind = json.loads(repo.journal.read_text(encoding='utf-8'))['kind']
            if kind not in {'deferred_package', 'relay_manifest'}:
                raise RuntimeError('direct publication journal retained; use Windows relay for publication')
            repo.resume()
            return 0
        plan = repo.plan()
        if plan['resume_pending']:
            raise RuntimeError('resume the interrupted local commit before starting another work package')
        pending = repo.pending()
        anchor = pending[-1]['commit'] if pending else plan['last_verified_remote']
        if anchor != plan['head']:
            raise RuntimeError('local HEAD differs from the last verified remote or deferred queue')
        repo.verify_protection()
        if args.prepared:
            if args.prepared not in plan['needs_prepared_changes']:
                raise RuntimeError('prepared package is gated, completed, or waiting on a dependency')
            owned = {release_path(name) for name in args.file}
            if owned & GENERATED or any(not (repo.root / name).is_file()
                    or (repo.root / name).is_symlink() or not (repo.root / name).resolve().is_relative_to(repo.root)
                    or (repo.root / name).stat().st_size > 2_000_000 for name in owned):
                raise RuntimeError('prepared files must be small regular source/config/test/report files')
            repo.require_scope(owned)
            before = {name: file_hash(repo.root / name) for name in owned}
            package = repo.registry()['packages'][args.prepared]
            package['tests'] = sorted(set(package['tests']) | {
                name for name in owned if name.startswith('tests/test_') and name.endswith('.py')})
            result = repo.run_tests(args.prepared, package)
            if before != {name: file_hash(repo.root / name) for name in owned}:
                raise RuntimeError('prepared source changed while tests ran')
            repo.publish(args.prepared, result, sorted(owned), defer=True)
            return 0 if result['exit_code'] == 0 else 2
        repo.require_scope(set())
        failed = False
        remaining = args.limit
        while remaining:
            packages = repo.registry()['packages']
            wave = work_queue(packages)['ready'][:min(remaining, args.jobs)]
            if not wave:
                break
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                futures = [pool.submit(repo.run_tests, name, packages[name]) for name in wave]
                results = [future.result() for future in futures]
            for name, result in zip(wave, results):
                repo.publish(name, result, defer=True)
                failed |= result['exit_code'] != 0
                remaining -= 1
        print(json_text(work_queue(repo.registry()['packages'])), end='')
        return 2 if failed else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f'DH116 scheduler: {error}', file=sys.stderr)
        raise SystemExit(1)
