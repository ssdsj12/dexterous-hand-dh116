"""Run DH116 CPU and bounded native work packages with Windows relay publication."""

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
import shlex
import shutil
import subprocess
import sys
import time
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
        if package['executor'] == 'native_bounded':
            continue
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


def execution_scope(registry):
    packages = registry['packages']
    if 'execution_target' not in registry:
        return packages, {}
    target = registry['execution_target']
    if (not isinstance(target, dict) or target.keys() != {'name', 'package', 'deferred'}
            or not isinstance(target['name'], str) or not target['name'].strip()
            or not isinstance(target['package'], str) or target['package'] not in packages):
        raise ValueError('execution_target requires a name and registered package')
    deferred = target['deferred']
    if (not isinstance(deferred, dict) or any(name not in packages or not isinstance(reason, str)
            or not reason.strip() for name, reason in deferred.items())):
        raise ValueError('execution_target deferred entries require a registered package and reason')

    required = set()
    remaining = [target['package']]
    while remaining:
        name = remaining.pop()
        if name not in required:
            required.add(name)
            remaining.extend(packages[name]['dependencies'])
    overlap = required.intersection(deferred)
    if overlap:
        raise ValueError('execution_target cannot omit deferred dependencies: ' + ', '.join(sorted(overlap)))
    scoped = {name: package for name, package in packages.items() if name in required}
    return scoped, dict(execution_target=dict(name=target['name'], package=target['package'],
                        status=packages[target['package']]['status']), deferred=deferred)


def scoped_queue(registry, scope='current'):
    if scope == 'all':
        return work_queue(registry['packages'])
    packages, description = execution_scope(registry)
    return dict(**description, **work_queue(packages))


def effective_queue(registry, scope, native):
    """Do not let a stale published native label release a CPU dependant."""
    qualified = set(native['native_qualified'])
    packages = {name: (dict(package, status='pending') if package['executor'] == 'native_bounded'
                      and name not in qualified else package)
                for name, package in registry['packages'].items()}
    return scoped_queue(dict(registry, packages=packages), scope)


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


NATIVE_SCOPE = 'isaac_single_gpu'
NATIVE_LOCK = Path('/tmp/dh116-isaac-single-instance.lock')
NATIVE_SCHEDULER_LOCK = Path('/tmp/dh116-native-bounded-scheduler.lock')
SHA256 = re.compile(r'^[0-9a-f]{64}$')
NATIVE_RETRY = {'max_attempts_per_code_config': 1, 'retry_requires_change': True}
NATIVE_RECEIPT_ARGS = {
    '{left_scheduler_receipt}': '--left-scheduler-receipt',
    '{left_receipt_sha256}': '--left-receipt-sha256',
}
NATIVE_BUDGET = {
    'gpu_index': (0, 63), 'timeout_seconds': (1, 900), 'min_gpu_free_mib': (1, 1000000),
    'min_disk_free_bytes': (1, 10**13), 'physics_steps': (0, 240),
    'environments': (1, 2), 'updates': (0, 1),
}


GDB = '/usr/bin/gdb'
GDB_STATUS = 'DH116_GDB_INFERIOR_STATUS '
GDB_BACKTRACE = 'DH116_GDB_FATAL_BACKTRACE'
GDB_COMMANDS = """set pagination off
set debuginfod enabled off
set auto-load off
set print thread-events off
handle SIGSEGV stop print pass
handle SIGBUS stop print pass
python
import gdb
import json

def trace_fatal(event):
    if isinstance(event, gdb.SignalEvent) and event.stop_signal in ("SIGSEGV", "SIGBUS"):
        gdb.write("DH116_GDB_FATAL_BACKTRACE " + event.stop_signal + "\\n")
        for command in ("bt 32", "thread apply all bt 20", "info sharedlibrary", "info proc mappings"):
            gdb.write("DH116_GDB_COMMAND " + command + "\\n")
            try:
                gdb.execute(command)
            except gdb.error as error:
                gdb.write("DH116_GDB_COMMAND_ERROR " + str(error) + "\\n")

def status_variable(name):
    value = gdb.parse_and_eval(name)
    return None if str(value) == "void" else int(value)

gdb.events.stop.connect(trace_fatal)
end
run
python
while gdb.selected_inferior().pid:
    gdb.execute("continue")
gdb.write("DH116_GDB_INFERIOR_STATUS " + json.dumps({
    "exit_code": status_variable("$_exitcode"),
    "signal": status_variable("$_exitsignal"),
}) + "\\n")
end
"""


def native_debugger_wrapper(path, result):
    """Keep python.sh's interpreter probes outside GDB and forward the actual argv."""
    commands = path / 'debugger.gdb'
    atomic_write(commands, GDB_COMMANDS)
    wrapper = path / 'debugger-python'
    interpreter = shlex.quote(sys.executable)
    atomic_write(wrapper, f"""#!/usr/bin/env bash
if [[ "${{1:-}}" == -c ]]; then
    exec {interpreter} "$@"
fi
exec {GDB} -nx -q --batch --return-child-result -x {shlex.quote(str(commands))} --args {interpreter} "$@"
""")
    wrapper.chmod(0o700)
    result.update(debugger='gdb_on_segv', debugger_script_sha256=file_hash(commands),
                  debugger_wrapper_sha256=file_hash(wrapper))
    return str(wrapper)


def debugger_status(log):
    status, traced = None, False
    with log.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if line.startswith(GDB_BACKTRACE):
                traced = True
            if line.startswith(GDB_STATUS):
                try:
                    status = json.loads(line[len(GDB_STATUS):])
                except ValueError:
                    status = None
    if (not isinstance(status, dict) or status.keys() != {'exit_code', 'signal'}
            or sum(type(v) is int for v in status.values()) != 1
            or any(v is not None and type(v) is not int for v in status.values())):
        return dict(debugger_status_verified=False, debugger_backtrace_captured=traced)
    return dict(debugger_status_verified=True, debugger_backtrace_captured=traced,
                debugger_inferior_exit_code=status['exit_code'], debugger_inferior_signal=status['signal'])


def process_is_descendant(pid, ancestor):
    """Only follow the live Linux parent chain of a debugger's lock holder."""
    for _ in range(32):
        if pid == ancestor:
            return True
        if pid <= 1:
            return False
        stat = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
        pid = int(stat.rsplit(')', 1)[1].split()[1])
    return False


def native_relative_path(value):
    path = PurePosixPath(value)
    if (not path.parts or path.is_absolute() or '..' in path.parts or value != path.as_posix()
            or '\\' in value or ':' in value):
        raise ValueError(f'invalid native relative path: {value!r}')
    return path


def native_artifact(output, relative):
    path = output / native_relative_path(relative)
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file():
        raise ValueError(f'expected regular native artifact: {relative}')
    return resolved


def validate_native_artifacts(spec, output):
    """Reopen artifact bytes independently of any producer completion marker."""
    validator = spec['validator']
    artifacts = {name: file_hash(native_artifact(output, name)) for name in validator['artifacts']}
    result = json.loads(native_artifact(output, validator['result_file']).read_text(encoding='utf-8'))
    for selector, expected in validator['equals'].items():
        actual = result
        for part in selector.split('.'):
            if not isinstance(actual, dict) or part not in actual:
                raise ValueError(f'native validator missing result field: {selector}')
            actual = actual[part]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f'native validator field differs: {selector}')
    counts = {}
    for name, expected in validator['line_counts'].items():
        with native_artifact(output, name).open(encoding='utf-8') as stream:
            count = 0
            for line in stream:
                if not line.endswith('\n') or not isinstance(json.loads(line), dict):
                    raise ValueError(f'native ledger must contain complete JSON object rows: {name}')
                count += 1
                if count > expected:
                    raise ValueError(f'native ledger exceeds declared budget: {name}')
        if count != expected:
            raise ValueError(f'native ledger count differs: {name}')
        counts[name] = count
    return dict(artifact_sha256=artifacts, ledger_counts=counts)


def native_file_lock(path, *, acquire=True):
    """Open the shared Linux lock with the existing native launcher's mode."""
    import fcntl

    stream = path.open('a+b')
    try:
        if acquire:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def native_recipe(repo, package):
    name = package.get('command_spec', '')
    path = repo.root / native_relative_path(name)
    if not name.startswith('configs/dh116_execution/') or not name.endswith('.json'):
        raise ValueError('native command spec must be in configs/dh116_execution')
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError('native command spec must be a regular local file')
    raw = path.read_bytes()
    spec = json.loads(raw)
    required = {'version', 'entrypoint', 'argv', 'source_files', 'budget', 'validator'}
    if spec.get('version') != 1 or not required <= spec.keys() <= required | {'inputs', 'environment', 'dependency_receipt', 'debugger'}:
        raise ValueError('invalid native command spec keys or version')
    entrypoint = spec['entrypoint']
    if not isinstance(entrypoint, str) or not release_path(entrypoint).startswith('scripts/') or not entrypoint.endswith('.py'):
        raise ValueError('native entrypoint must be a local Python script')
    binding = spec.get('dependency_receipt')
    placeholders = {'{output}': '--output'}
    if binding is not None:
        if (not isinstance(binding, dict) or binding.keys() != {'package', 'qualification_scope'}
                or not isinstance(binding['package'], str) or not ID.fullmatch(binding['package'])
                or binding['package'] not in package['dependencies']
                or not isinstance(binding['qualification_scope'], str) or not binding['qualification_scope']
                or package.get('dependency_scopes', {}).get(binding['package']) != {
                    'scope': binding['qualification_scope'], 'native': True}):
            raise ValueError('receipt binding must reference a declared scoped native dependency')
        placeholders.update(NATIVE_RECEIPT_ARGS)
    args = spec['argv']
    if (not isinstance(args, list) or any(not isinstance(arg, str) or not arg or '\x00' in arg or '\n' in arg
            or ('{' in arg or '}' in arg) and arg not in placeholders for arg in args)):
        raise ValueError('native argv permits only explicit approved placeholders')
    for token, flag in placeholders.items():
        if (args.count(token) != 1 or args.count(flag) != 1 or args.index(token) == 0
                or args[args.index(token) - 1] != flag or any(arg.startswith(flag + '=') for arg in args)):
            raise ValueError(f'native argv requires one explicit {flag} {token}')
    if binding is None and any(arg.split('=', 1)[0] in NATIVE_RECEIPT_ARGS.values() for arg in args):
        raise ValueError('receipt arguments must be bound to a scheduler dependency')
    budget = spec['budget']
    if (not isinstance(budget, dict) or set(budget) != set(NATIVE_BUDGET)
            or any(type(budget[key]) is not int or not low <= budget[key] <= high
                   for key, (low, high) in NATIVE_BUDGET.items())):
        raise ValueError('native budget is incomplete or outside bounded limits')
    validator = spec['validator']
    if (not isinstance(validator, dict) or validator.keys() != {'result_file', 'equals', 'artifacts', 'line_counts'}
            or not isinstance(validator['artifacts'], list) or not validator['artifacts']
            or not isinstance(validator['equals'], dict) or not validator['equals']
            or not isinstance(validator['line_counts'], dict)
            or not set(validator['line_counts']) <= set(validator['artifacts'])
            or validator['result_file'] not in validator['artifacts']
            or any(type(v) is not int or v < 1 or v > 480 for v in validator['line_counts'].values())
            or budget['physics_steps'] and budget['physics_steps'] not in validator['line_counts'].values()):
        raise ValueError('native validator must bind scoped result fields and exact physical ledger rows')
    for value in validator['artifacts']:
        native_relative_path(value)
    environment = spec.get('environment', {})
    if (not isinstance(environment, dict) or not set(environment) <= {'LD_LIBRARY_PATH', 'DH116_NUM_THREADS'}
            or any(not isinstance(v, str) or '\x00' in v or '\n' in v for v in environment.values())
            or 'LD_LIBRARY_PATH' in environment and any(not Path(v).is_absolute()
                for v in environment['LD_LIBRARY_PATH'].split(os.pathsep))
            or 'DH116_NUM_THREADS' in environment and (not environment['DH116_NUM_THREADS'].isdigit()
                or not 1 <= int(environment['DH116_NUM_THREADS']) <= 64)):
        raise ValueError('unsupported native environment override')
    source_files = spec['source_files']
    if not isinstance(source_files, list):
        raise ValueError('native source_files must be a list')
    files = {'scripts/python.sh', entrypoint, *(release_path(value) for value in source_files),
             *(release_path(value) for value in package.get('tests', ()))}
    scheduler_file = 'scripts/run_dh116_work_packages.py'
    if (repo.root / scheduler_file).is_file():
        files.add(scheduler_file)
    identity = {}
    for relative in sorted(files):
        source = repo.root / relative
        if source.resolve(strict=True) != source or not source.is_file():
            raise ValueError(f'invalid native source: {relative}')
        identity[relative] = file_hash(source)
    inputs = spec.get('inputs', {})
    if not isinstance(inputs, dict):
        raise ValueError('native inputs must map absolute files to SHA256')
    if 'debugger' in spec and (spec['debugger'] != 'gdb_on_segv' or GDB not in inputs):
        raise ValueError('native debugger requires gdb_on_segv and the pinned installed GDB digest')
    for value, expected in inputs.items():
        path = Path(value)
        if (not path.is_absolute() or not isinstance(expected, str) or not SHA256.fullmatch(expected)
                or file_hash(path) != expected):
            raise ValueError(f'native input digest unavailable or differs: {value}')
    spec_digest = hashlib.sha256(raw).hexdigest()
    digest = hashlib.sha256(json.dumps(dict(spec=spec_digest, source=identity), sort_keys=True).encode()).hexdigest()
    return dict(spec=spec, command_spec_sha256=spec_digest, code_config_sha256=digest)


def native_approval(approval, repo):
    if approval is None:
        return None
    if (not isinstance(approval, dict) or approval.keys() != {'version', 'batch_id', 'output_root', 'packages'}
            or approval['version'] != 1 or not isinstance(approval['batch_id'], str)
            or not ID.fullmatch(approval['batch_id']) or not isinstance(approval['packages'], dict)):
        raise ValueError('versioned native execution batch approval required')
    output = Path(approval['output_root'])
    if (not output.is_absolute() or output.resolve() != output or repo.root.is_relative_to(output)
            or output.is_relative_to(repo.root) and not output.is_relative_to(repo.root / 'artifacts')):
        raise ValueError('approved native output root must be private and separate from source')
    for name, approved in approval['packages'].items():
        if (not ID.fullmatch(name) or not isinstance(approved, dict)
                or set(approved) != {'command_spec_sha256', 'code_config_sha256'}
                or any(not isinstance(v, str) or not SHA256.fullmatch(v) for v in approved.values())):
            raise ValueError('native batch requires exact package code and command spec SHA256')
    return approval


def resource_blocker(kind, reason):
    return dict(kind=kind, reason=reason)


def native_dependency_blockers(packages, package, qualified):
    waiting = []
    for dep in package['dependencies']:
        candidate = qualified.get(dep) or packages[dep].get('last_result', {})
        required = package['dependency_scopes'].get(dep)
        if dep not in qualified and (packages[dep]['status'] != 'passed'
                                    or packages[dep]['executor'] == 'native_bounded'):
            waiting.append(dep)
            continue
        if required is None:
            continue
        if (candidate.get('qualification_scope') != required['scope']
                or required['native'] and (
                    candidate.get('native_qualification') is not True
                    or candidate.get('native_execution_performed') is not True
                    or not (candidate.get('outer_exit_code') == 0 or
                            candidate.get('outer_exit_codes') and all(code == 0 for code in candidate['outer_exit_codes']))
                )):
            waiting.append(dep)
    return waiting


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
        if data.get('version') not in {1, 2} or not isinstance(data.get('packages'), dict):
            raise ValueError('versioned work-package registry required')
        for name, package in data['packages'].items():
            if (not ID.fullmatch(name) or package['status'] not in {'pending', 'blocked', 'failed', 'passed'}
                    or package['executor'] not in {'pytest', 'prepared', 'gated', 'native_bounded'}
                    or not isinstance(package['dependencies'], list)
                    or not isinstance(package.get('title'), str) or '\n' in package['title']):
                raise ValueError(f'invalid work-package definition: {name}')
            if package['executor'] == 'native_bounded':
                scopes = package.get('dependency_scopes', {})
                if (package.get('requires_approval') is not True or package.get('resource_class') != NATIVE_SCOPE
                        or not isinstance(package.get('evidence_scope'), str) or not package['evidence_scope']
                        or not isinstance(package.get('command_spec'), str)
                        or package.get('retry_policy') != NATIVE_RETRY
                        or not isinstance(scopes, dict) or not scopes.keys() <= set(package['dependencies'])
                        or any(not isinstance(value, dict) or value.keys() != {'scope', 'native'}
                            or not isinstance(value['scope'], str) or not value['scope']
                            or type(value['native']) is not bool for value in scopes.values())):
                    raise ValueError(f'invalid bounded native work package: {name}')
                package.setdefault('dependency_scopes', {})
        check_dependencies(data['packages'])
        execution_scope(data)
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

    def native_recipe(self, package):
        return native_recipe(self, package)

    def resolve_native_dependency_receipt(self, binding, packages):
        """Locate the published producer's exact private receipt and revalidate its bytes."""
        producer = binding['package']
        scope = binding['qualification_scope']
        package = packages[producer]
        published = package.get('last_result', {})
        recipe = self.native_recipe(package)
        identity = {key: recipe[key] for key in ('command_spec_sha256', 'code_config_sha256')}
        expected = published.get('result_sha256', '')
        if (package['executor'] != 'native_bounded' or package['status'] != 'passed'
                or published.get('native_qualification') is not True
                or published.get('qualification_scope') != scope or not isinstance(expected, str)
                or not SHA256.fullmatch(expected) or any(published.get(k) != v for k, v in identity.items())):
            raise ValueError('producer lacks a current published qualified scheduler receipt')
        base = self.private / 'native-attempts' / producer / identity['code_config_sha256']
        matches = [path for path in base.glob('attempt-*/receipt.json')
                   if path.resolve() == path and file_hash(path) == expected]
        if len(matches) != 1:
            raise ValueError('producer receipt is missing, ambiguous, or has been modified')
        path, = matches
        attempt = path.parent
        receipt = json.loads(path.read_text(encoding='utf-8'))
        intent = json.loads(native_artifact(attempt, 'intent.json').read_text(encoding='utf-8'))
        outer = json.loads(native_artifact(attempt, 'outer_exit.json').read_text(encoding='utf-8'))
        approved = native_approval(intent['approval'], self)
        output = Path(approved['output_root']) / producer / attempt.name / 'native'
        if (intent.get('package') != producer or intent.get('output') != str(output)
                or receipt.get('output') != str(output) or output.resolve(strict=True) != output
                or receipt.get('executor') != 'native_bounded'
                or receipt.get('native_execution_performed') is not True
                or receipt.get('native_qualification') is not True or receipt.get('qualification_scope') != scope
                or approved['packages'].get(producer) != identity
                or receipt.get('tested_head') != intent.get('head') or receipt.get('batch_id') != approved['batch_id']
                or any(intent.get(k) != v or receipt.get(k) != v for k, v in identity.items())):
            raise ValueError('producer receipt identity does not match the scheduled dependency')
        if (any(type(item.get('outer_exit_code')) is not int or item['outer_exit_code'] != 0
                or item.get('native_lock_observed') is not True for item in (outer, receipt))
                or outer.get('log_sha256') != receipt.get('log_sha256')
                or file_hash(native_artifact(attempt, 'native.log')) != receipt.get('log_sha256')):
            raise ValueError('producer requires independent natural exit zero and native lock evidence')
        if recipe['spec'].get('debugger') == 'gdb_on_segv':
            debug = debugger_status(native_artifact(attempt, 'native.log'))
            if (not debug['debugger_status_verified'] or debug.get('debugger_inferior_exit_code') != 0
                    or debug.get('debugger_inferior_signal') is not None
                    or any(item.get(k) != v for item in (outer, receipt) for k, v in debug.items())):
                raise ValueError('producer debugger evidence lacks independent inferior exit zero')
        evidence = validate_native_artifacts(recipe['spec'], output)
        if any(receipt.get(k) != v for k, v in evidence.items()):
            raise ValueError('producer artifacts or measured ledger differ from the qualified receipt')
        return dict(package=producer, qualification_scope=scope, path=str(path), sha256=expected)

    def native_attempts(self, name, fingerprint):
        root = self.private / 'native-attempts' / name / fingerprint
        for directory in sorted(root.glob('attempt-*')):
            intent = directory / 'intent.json'
            receipt = directory / 'receipt.json'
            if not intent.is_file():
                continue
            if not receipt.is_file():
                yield dict(pending=True, intent=str(intent))
                continue
            result = json.loads(receipt.read_text(encoding='utf-8'))
            if result.get('blocker_kind') != 'waiting_resource':
                yield dict(result, result_sha256=file_hash(receipt))

    def native_resource_blocker(self, recipe, output_root, worker=False):
        budget = recipe['spec']['budget']
        try:
            if not worker:
                with native_file_lock(NATIVE_SCHEDULER_LOCK):
                    pass
            with native_file_lock(NATIVE_LOCK):
                pass
            parent = output_root
            while not parent.exists():
                parent = parent.parent
            if shutil.disk_usage(parent).free < budget['min_disk_free_bytes']:
                return 'native output disk free bytes below approved minimum'
            gpu = str(budget['gpu_index'])
            free = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
                                            '--format=csv,noheader,nounits'], text=True, timeout=8)
            if int(free.strip()) < budget['min_gpu_free_mib']:
                return 'GPU memory below approved minimum'
            processes = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid',
                                                 '--format=csv,noheader,nounits'], text=True, timeout=8)
            if processes.strip():
                return 'GPU already has an active compute process'
            # Cover a manual legacy Isaac job even when its launcher did not acquire the shared lock.
            legacy = {'kit', 'isaacsim', 'train_dh116_tabletop_gpu_ppo.py',
                      'collect_dh116_hand_only_mpc_isaac.py'}
            for file in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    args = file.read_bytes().split(b'\0')
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                if any(Path(arg.decode(errors='replace')).name in legacy for arg in args if arg):
                    return 'another Isaac process exists'
        except BlockingIOError:
            return 'shared Isaac native lock or scheduler reservation is occupied'
        except (OSError, ValueError, subprocess.SubprocessError):
            return 'unable to verify native GPU or disk resources'
        return None

    def native_lock_observed(self, process_group, debugger=False):
        """Verify the actual child, rather than the scheduler, holds the shared lock."""
        try:
            stat = NATIVE_LOCK.stat()
            for line in Path('/proc/locks').read_text(encoding='utf-8').splitlines():
                fields = line.split()
                if len(fields) < 8 or fields[1:4] != ['FLOCK', 'ADVISORY', 'WRITE']:
                    continue
                major, minor, inode = fields[5].split(':')
                if (int(major, 16), int(minor, 16), int(inode)) != (
                        os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino):
                    continue
                try:
                    owner = int(fields[4])
                    return (os.getpgid(owner) == process_group
                            or debugger and process_is_descendant(owner, process_group))
                except ProcessLookupError:
                    return False
        except (OSError, ValueError):
            return False
        return False

    def native_plan(self, *, approval=None, scope='current', probe_resources=False):
        approval = native_approval(approval, self)
        registry = self.registry()
        selected = registry['packages'] if scope == 'all' else execution_scope(registry)[0]
        ready, blockers, recipes, pending_results, qualified = [], {}, {}, {}, {}
        identities = {}
        for name, package in selected.items():
            if package['executor'] != 'native_bounded':
                continue
            try:
                recipe = self.native_recipe(package)
            except (OSError, ValueError, TypeError, KeyError) as error:
                blockers[name] = resource_blocker('external_data', str(error))
                continue
            fingerprint = recipe['code_config_sha256']
            identities[name] = {k: recipe[k] for k in ('command_spec_sha256', 'code_config_sha256')}
            recipes[name] = recipe
            current = package.get('last_result', {})
            attempts = list(self.native_attempts(name, fingerprint))
            if len(attempts) > 1:
                raise RuntimeError(f'multiple native attempts consumed for identical code/config: {name}')
            if current.get('code_config_sha256') == fingerprint:
                if current.get('native_qualification') is True and package['status'] == 'passed':
                    qualified[name] = current
                else:
                    blockers[name] = resource_blocker(package.get('blocker_kind', 'failed_native'),
                                                      'same code/config already completed without qualification')
                continue
            if attempts:
                result = attempts[0]
                if result.get('pending'):
                    overdue = Path(result['intent']).with_name('overdue.json').is_file()
                    reason = ('elapsed budget exceeded; child still owns its reservation'
                              if overdue else 'attempt in progress or interrupted; no duplicate launch')
                    blockers[name] = resource_blocker('waiting_resource', reason)
                else:
                    pending_results[name] = result
                    if result.get('native_qualification') is True:
                        qualified[name] = result
                    else:
                        blockers[name] = resource_blocker('failed_native', 'identical code/config has retained failure evidence')
        binding_checks = {}

        def binding_error(name):
            binding = recipes[name]['spec'].get('dependency_receipt')
            if binding is None:
                return None
            if name not in binding_checks:
                try:
                    bound = self.resolve_native_dependency_receipt(binding, registry['packages'])
                    result = qualified.get(name)
                    error = ('qualified result binds a different producer receipt'
                             if result is not None and result.get('dependency_receipt_sha256') != bound['sha256']
                             else None)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                binding_checks[name] = error
            return binding_checks[name]

        # A previously passed native package is usable only while all its scoped
        # inputs remain qualified. This also covers a private result awaiting publication.
        changed = True
        while changed:
            stale = [name for name in qualified
                     if (native_dependency_blockers(registry['packages'], selected[name], qualified)
                         or binding_error(name))]
            changed = bool(stale)
            for name in stale:
                qualified.pop(name)
                pending_results.pop(name, None)
                blockers[name] = resource_blocker('waiting_dependency', 'native prerequisite evidence is no longer current')
        for name in list(pending_results):
            if native_dependency_blockers(registry['packages'], selected[name], qualified):
                pending_results.pop(name)
                blockers[name] = resource_blocker('waiting_dependency', 'retained result awaits current prerequisite evidence')
        for name, recipe in recipes.items():
            if name in qualified or name in blockers or name in pending_results:
                continue
            package = selected[name]
            waiting = native_dependency_blockers(registry['packages'], package, qualified)
            blocker = package.get('blocker_kind')
            previous = package.get('last_result', {}).get('code_config_sha256')
            if waiting:
                blockers[name] = resource_blocker('waiting_dependency', 'waiting for evidence: ' + ', '.join(waiting))
            elif blocker in {'unsupported_case', 'external_data'}:
                blockers[name] = resource_blocker(blocker, package.get('reason', 'native case is unavailable'))
            elif blocker in {'failed_native', 'failed_test'} and not previous:
                blockers[name] = resource_blocker(blocker, 'historical failure has no code/config identity; review evidence')
            elif error := binding_error(name):
                blockers[name] = resource_blocker('waiting_dependency', error)
            if name in blockers:
                continue
            if not approval or approval['packages'].get(name) != identities[name]:
                blockers[name] = resource_blocker('waiting_approval', 'no batch approval for exact code/config and command')
            else:
                reason = self.native_resource_blocker(recipe, Path(approval['output_root'])) if probe_resources else None
                if reason:
                    blockers[name] = resource_blocker('waiting_resource', reason)
                else:
                    ready.append(name)
        return dict(native_ready=ready, blockers=blockers, native_identities=identities,
                    native_results=pending_results, native_qualified=sorted(qualified))

    def register_native_attempt(self, name, approval):
        approval = native_approval(approval, self)
        if name not in self.native_plan(approval=approval, scope='all')['native_ready']:
            raise RuntimeError('native package is blocked, consumed, or unapproved')
        package = self.registry()['packages'][name]
        recipe = self.native_recipe(package)
        fingerprint = recipe['code_config_sha256']
        base = self.private / 'native-attempts' / name / fingerprint
        base.mkdir(parents=True, exist_ok=True)
        directory = Path(mkdtemp(prefix='attempt-', dir=base))
        output = Path(approval['output_root']) / name / directory.name / 'native'
        intent = directory / 'intent.json'
        state = dict(package=name, head=self.head(), output=str(output), approval=approval,
                     command_spec_sha256=recipe['command_spec_sha256'], code_config_sha256=fingerprint)
        binding = recipe['spec'].get('dependency_receipt')
        if binding is not None:
            state['dependency_receipt'] = self.resolve_native_dependency_receipt(binding, self.registry()['packages'])
        atomic_write(intent, json_text(state))
        return intent

    def run_native_worker(self, intent_path):
        intent_path = Path(intent_path).resolve(strict=True)
        if intent_path.name != 'intent.json' or not intent_path.is_relative_to(self.private / 'native-attempts'):
            raise ValueError('native worker requires a registered private intent')
        with native_file_lock(intent_path.parent / 'worker.lock'):
            return self._run_registered_native_worker(intent_path)

    def _run_registered_native_worker(self, intent_path):
        path = intent_path.parent
        intent = json.loads(intent_path.read_text(encoding='utf-8'))
        name = intent['package']
        approval = native_approval(intent['approval'], self)
        packages = self.registry()['packages']
        package = packages[name]
        recipe = self.native_recipe(package)
        identity = {k: recipe[k] for k in ('command_spec_sha256', 'code_config_sha256')}
        output_root = Path(approval['output_root'])
        output = output_root / name / path.name / 'native'
        if (self.head() != intent['head'] or identity != approval['packages'].get(name)
                or any(intent[k] != v for k, v in identity.items()) or str(output) != intent['output']
                or path.parent != self.private / 'native-attempts' / name / identity['code_config_sha256']):
            raise RuntimeError('native worker intent no longer matches approved code/config')
        if native_dependency_blockers(packages, package,
                {key: packages[key].get('last_result', {})
                 for key in self.native_plan(scope='all')['native_qualified']}):
            raise RuntimeError('native worker dependency evidence no longer qualifies')
        binding = recipe['spec'].get('dependency_receipt')
        bound = self.resolve_native_dependency_receipt(binding, packages) if binding is not None else None
        if intent.get('dependency_receipt') != bound:
            raise RuntimeError('producer receipt changed after native attempt registration')
        receipt = path / 'receipt.json'
        if receipt.exists() or (path / 'running.json').exists():
            raise RuntimeError('native attempt already has a receipt or durable running marker')
        result = dict(executor='native_bounded', tested_head=intent['head'], batch_id=approval['batch_id'],
                      qualification_scope=package['evidence_scope'], native_qualification=False,
                      native_execution_performed=False, output=str(output), **identity)
        if bound is not None:
            result['dependency_receipt_sha256'] = bound['sha256']
        try:
            lease = native_file_lock(NATIVE_SCHEDULER_LOCK)
        except BlockingIOError:
            lease = None
        if lease is None:
            result.update(blocker_kind='waiting_resource', reason='another scheduler owns native reservation')
        else:
            with lease:
                if receipt.exists() or (path / 'running.json').exists():
                    raise RuntimeError('native attempt already has a receipt or durable running marker')
                reason = self.native_resource_blocker(recipe, output_root, True)
                if reason:
                    result.update(blocker_kind='waiting_resource', reason=reason)
                else:
                    try:
                        if bound is not None and bound != self.resolve_native_dependency_receipt(binding, packages):
                            raise RuntimeError('producer receipt changed before native launch')
                        self._native_child(path, recipe['spec'], output, result, bound=bound)
                    except (OSError, ValueError, subprocess.SubprocessError):
                        result.update(blocker_kind='failed_native', reason='native worker failed to execute or finish',
                                      log_sha256=file_hash(path / 'native.log'))
        atomic_write(receipt, json_text(result))
        return dict(result, result_sha256=file_hash(receipt))

    def _native_child(self, path, spec, output, result, *, bound=None):
        budget = spec['budget']
        if output.resolve() != output:
            raise ValueError('native output path escapes the approved private root')
        output.parent.mkdir(parents=True, exist_ok=False)
        tokens = {'{output}': str(output)}
        if bound is not None:
            tokens.update({'{left_scheduler_receipt}': bound['path'], '{left_receipt_sha256}': bound['sha256']})
        command = ['bash', 'scripts/python.sh', spec['entrypoint'],
                   *(tokens.get(arg, arg) for arg in spec['argv'])]
        launch_record = path / 'launch_argv.json'
        atomic_write(launch_record, json_text(dict(argv=command, dependency_receipt=bound)))
        result['launch_argv_sha256'] = file_hash(launch_record)
        env = {**os.environ, **spec.get('environment', {}), 'DH116_PYTHON': sys.executable,
               'CUDA_VISIBLE_DEVICES': str(budget['gpu_index']), 'PYTHONDONTWRITEBYTECODE': '1'}
        debugger = spec.get('debugger') == 'gdb_on_segv'
        if debugger:
            env['DH116_PYTHON'] = native_debugger_wrapper(path, result)
        log = path / 'native.log'
        started = time.monotonic()
        observed = False
        with log.open('xb') as stream:
            atomic_write(path / 'running.json', json_text(dict(
                worker_pid=os.getpid(), started_unix_s=time.time(), timeout_seconds=budget['timeout_seconds'])))
            with subprocess.Popen(command, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=subprocess.STDOUT, start_new_session=True) as child:
                while child.poll() is None and time.monotonic() - started <= budget['timeout_seconds']:
                    observed |= self.native_lock_observed(child.pid, debugger)
                    time.sleep(0.05)
                if child.poll() is None:
                    atomic_write(path / 'overdue.json', json_text(dict(
                        worker_pid=os.getpid(), child_pid=child.pid, budget_seconds=budget['timeout_seconds'])))
                code = child.wait()
            stream.flush()
            os.fsync(stream.fileno())
        seconds = time.monotonic() - started
        result.update(native_execution_performed=True, exit_code=code, outer_exit_code=code,
                      log_sha256=file_hash(log), duration_seconds=round(seconds, 3), native_lock_observed=observed)
        debug_status = debugger_status(log) if debugger else {}
        result.update(debug_status)
        atomic_write(path / 'outer_exit.json', json_text({
            **{key: result[key] for key in ('outer_exit_code', 'duration_seconds', 'log_sha256', 'native_lock_observed')},
            **debug_status,
        }))
        debugger_failed = debugger and (not result['debugger_status_verified']
                                        or result.get('debugger_inferior_exit_code') != 0
                                        or result.get('debugger_inferior_signal') is not None)
        if code != 0 or not observed or seconds > budget['timeout_seconds'] or debugger_failed:
            result.update(blocker_kind='failed_native',
                          reason='outer process failure, missing shared lock evidence, or elapsed budget exceeded')
            return
        try:
            artifacts = validate_native_artifacts(spec, output)
        except (OSError, ValueError, TypeError) as error:
            result.update(blocker_kind='failed_native', reason=f'independent validator rejected evidence: {error}')
            return
        result.update(native_qualification=True, **artifacts)

    def start_native(self, name, approval):
        intent = self.register_native_attempt(name, approval)
        budget = self.native_recipe(self.registry()['packages'][name])['spec']['budget']
        with (intent.parent / 'worker.log').open('xb') as log:
            worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--native-worker', str(intent), '--evidence-dir', str(self.private)], cwd=self.root,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            worker.wait(timeout=budget['timeout_seconds'])
        except subprocess.TimeoutExpired:
            # The native recipe enforces its physics/update budget itself.
            # Keep the worker's reservation and evidence; do not signal Isaac.
            return None
        receipt = intent.parent / 'receipt.json'
        if not receipt.is_file():
            return None
        return dict(json.loads(receipt.read_text(encoding='utf-8')), result_sha256=file_hash(receipt))

    def local_tracking_head(self):
        return self.git('for-each-ref', '--format=%(objectname)', TRACKING_REF).strip() or None

    def plan(self, *, scope='current', approval=None, probe_resources=False):
        self.verify_local_target()
        registry = self.registry()
        native = self.native_plan(approval=approval, scope=scope, probe_resources=probe_resources)
        return dict(branch=BRANCH, head=self.head(), publication_route='windows_relay',
                    last_verified_remote=self.local_tracking_head(),
                    working_tree=self.git('status', '--short').splitlines(),
                    resume_pending=self.journal.exists(), pending_publications=len(self.pending()),
                    **effective_queue(registry, scope, native), **native)

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

    def publish_native(self, name, result):
        current = self.native_plan(scope='all')['native_results'].get(name)
        if result != current:
            raise RuntimeError('native publication must bind the durable independently validated receipt')
        package = self.registry()['packages'][name]
        cpu = self.run_tests(name, package) if package.get('tests') else None
        summary = dict(result)
        summary['native_physics_validated'] = result['native_qualification']
        if cpu is not None:
            summary['cpu_regression'] = {k: v for k, v in cpu.items() if k != 'log'}
            if cpu['exit_code'] != 0:
                summary.update(native_qualification=False, blocker_kind='failed_test')
        return self.publish(name, summary, defer=True)

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
        native = result.get('executor') == 'native_bounded'
        if native:
            durable = self.native_plan(scope='all')['native_results'].get(name)
            if (not defer or prepared or durable is None
                    or any(result.get(key) != durable[key] for key in durable if key != 'native_qualification')
                    or result.get('native_qualification') is True and not durable['native_qualification']):
                raise RuntimeError('native publication requires the unchanged approved recipe and Windows relay')
        elif result['tested_head'] != parent and self.git('diff', '--name-only', result['tested_head'], parent, '--',
                                                          'src', 'scripts', 'tests', 'configs').strip():
            raise RuntimeError('code changed after the CPU verification wave')
        registry = self.registry()
        package = registry['packages'][name]
        passed = result.get('native_qualification') is True if native else result['exit_code'] == 0
        status = 'passed' if passed else 'failed'
        report = f'docs/superpowers/reports/dh116-managed-{name}.md'
        if native:
            private_fields = {'log', 'output', 'reason'}
            entry = {key: value for key, value in result.items() if key not in private_fields}
            entry.update(status=status, report=report)
            package['last_native_attempt'] = entry
            if passed:
                package.pop('reason', None)
                package.pop('blocker_kind', None)
            else:
                package['blocker_kind'] = result.get('blocker_kind', 'failed_native')
                package['reason'] = 'native or CPU qualification failed; durable evidence retained outside Git'
            report_text = (f'# {package["title"]}\n\n'
                           f'基线 `{result["tested_head"]}`。原生范围 `{result["qualification_scope"]}`，'
                           f'状态 `{status}`，外层退出码 `{result.get("outer_exit_code")}`。'
                           f'独立私有回执 SHA256 `{result["result_sha256"]}`。\n\n'
                           f'代码与配置 SHA256 `{result["code_config_sha256"]}`。'
                           '范围以独立产物校验和进程退出为准。原始日志留在 Ubuntu 私有证据目录。\n')
            label = '原生'
        else:
            entry = {key: value for key, value in result.items() if key != 'log'}
            entry.update(status=status, report=report, native_qualification=False)
            if not passed:
                package['reason'] = 'CPU regression did not pass; evidence retained outside Git'
            followups = [key for key, item in registry['packages'].items() if name in item['dependencies']]
            report_text = (f'# {package["title"]}\n\n'
                           f'基线 `{result["tested_head"]}`。CPU 状态 `{status}`，通过 {result["passed"]} 项，'
                           f'失败 {result["failed"]} 项，跳过 {result["skipped"]} 项，退出码 {result["exit_code"]}。'
                           '此项不授予原生或训练资格。\n\n'
                           f'命令 `{" ".join(result["command"])}`。私有日志 SHA256 `{result["log_sha256"]}`。'
                           f'后继工作包 {", ".join(followups) or "按登记表继续就绪任务"}。\n')
            if not passed:
                report_text += '\nCPU 配方未通过，保留原始日志与本地代码，依赖任务暂停。\n'
            label = f'CPU 回归 {result["passed"]} 项通过'
        package['status'] = status
        package['last_result'] = entry
        date = datetime.now(timezone.utc).date().isoformat()
        note = (f'## {date} — {package["title"]}\n\n'
                f'{label}，状态 `{status}`。详细结果见[报告]({report})。\n\n')
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
    modes.add_argument('--native-worker', type=Path, help=argparse.SUPPRESS)
    modes.add_argument('--relay-plan', action='store_true')
    modes.add_argument('--relay-ack', action='store_true', help='Read verified local publisher receipt from stdin')
    parser.add_argument('--scope', choices=('current', 'all'), default='current',
                        help='select the configured target or all registered work packages')
    parser.add_argument('--file', action='append', default=[])
    parser.add_argument('--native-approval', type=Path, help='private exact native batch approval JSON')
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--local-only', action='store_true',
                        help='Compatibility option; the scheduler always queues commits for Windows publication')
    parser.add_argument('--evidence-dir', type=Path, help='Separate private directory for logs and recovery journal')
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 12 or not 1 <= args.jobs <= 4 or bool(args.file) != bool(args.prepared):
        parser.error('use 1..12 packages and 1..4 CPU jobs; --prepared needs explicit --file paths')
    repo = Repository(Path(__file__).resolve().parents[1], evidence_dir=args.evidence_dir)
    approval = None
    if args.native_approval is not None:
        approval_path = args.native_approval.resolve(strict=True)
        if approval_path.is_relative_to(repo.root) and not approval_path.is_relative_to(repo.root / 'artifacts'):
            raise ValueError('native batch approval must stay outside published source')
        approval = native_approval(json.loads(approval_path.read_text(encoding='utf-8')), repo)
    if args.native_worker:
        repo.verify_local_target()
        print(json_text(repo.run_native_worker(args.native_worker)), end='', flush=True)
        return 0
    if args.relay_plan:
        print(json_text(repo.relay_plan()), end='')
        return 0
    if not (args.run or args.prepared or args.resume or args.relay_ack):
        print(json_text(repo.plan(scope=args.scope, approval=approval,
                                  probe_resources=approval is not None)), end='')
        return 0
    with publication_lock(repo.private):
        if args.relay_ack:
            print(json_text(repo.accept_relay_receipt(json.load(sys.stdin))), end='')
            return 0
        if args.resume:
            if not repo.journal.exists():
                pending = repo.pending()
                if not pending:
                    native = repo.native_plan(approval=approval, scope=args.scope)
                    if native['native_results'] or any(value['kind'] == 'waiting_resource'
                            for value in native['blockers'].values()):
                        print(json_text(native), end='')
                        return 0
                    raise RuntimeError('no interrupted local commit or queued Windows publication')
                print(json_text(dict(publication_route='windows_relay',
                                     pending_publications=len(pending), head=repo.head())), end='')
                return 0
            kind = json.loads(repo.journal.read_text(encoding='utf-8'))['kind']
            if kind not in {'deferred_package', 'relay_manifest'}:
                raise RuntimeError('direct publication journal retained; use Windows relay for publication')
            repo.resume()
            return 0
        plan = repo.plan(scope=args.scope, approval=approval)
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
        considered = set()
        while remaining:
            registry = repo.registry()
            packages = registry['packages']
            native = repo.native_plan(approval=approval, scope=args.scope, probe_resources=approval is not None)
            wave = effective_queue(registry, args.scope, native)['ready'][:min(remaining, args.jobs)]
            if wave:
                with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                    futures = [pool.submit(repo.run_tests, name, packages[name]) for name in wave]
                    results = [future.result() for future in futures]
                for name, result in zip(wave, results):
                    repo.publish(name, result, defer=True)
                    failed |= result['exit_code'] != 0
                    remaining -= 1
                continue
            retained = native['native_results']
            if retained:
                name, result = next(iter(retained.items()))
            else:
                name = next((key for key in native['native_ready'] if key not in considered), None)
                if name is None:
                    break
                considered.add(name)
                result = repo.start_native(name, approval)
                if result is None or result.get('blocker_kind') == 'waiting_resource':
                    continue
            receipt = repo.publish_native(name, result)
            failed |= receipt['status'] != 'passed'
            remaining -= 1
        native = repo.native_plan(approval=approval, scope=args.scope, probe_resources=approval is not None)
        print(json_text(dict(**effective_queue(repo.registry(), args.scope, native), **native)), end='')
        return 2 if failed else 0



if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f'DH116 scheduler: {error}', file=sys.stderr)
        raise SystemExit(1)
