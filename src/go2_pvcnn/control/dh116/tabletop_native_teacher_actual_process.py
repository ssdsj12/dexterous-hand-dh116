"""Parent-issued, single-use Linux worker launch handoff.

Only a current registered approval, kernel peer credentials and the inferior's
own FLOCK can issue an overlay. No App, GPU or native runtime is imported here.
The caller owns the lock throughout request, execution and cleanup.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import select
import socket
import stat
import struct
import sys
import tempfile
import time

from .tabletop_native_teacher_worker_contract import worker_registered_native_package
from .tabletop_native_teacher_worker_lease import verify_worker_lease

ENVELOPE_SCHEMA = 'dh116-actual-parent-launch-envelope-v1'
REPLY_SCHEMA = 'dh116-actual-worker-launch-reply-v1'
_PACKET_LIMIT = 65536
_FILE_LIMIT = 1024 * 1024
_LEASE_TOKEN = object()


def _encode(value):
    if hasattr(value, '__dataclass_fields__'):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _regular_path(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('regular absolute nonsymlink launch path required')
    return path


def _read(binding):
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
        raise ValueError('pinned launch file binding required')
    path = _regular_path(binding['path'])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _FILE_LIMIT:
            raise ValueError('bounded regular launch file required')
        raw = stream.read(_FILE_LIMIT + 1)
    if _hash(raw) != binding['sha256']:
        raise ValueError('launch file bytes changed')
    return path, raw


def _write(path, value):
    path = Path(path)
    if path.parent.resolve(strict=True) != path.parent or path.exists() or path.is_symlink():
        raise ValueError('exclusive session launch path required')
    raw = _encode(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(path), sha256=_hash(raw))


def _stat_fields(pid):
    raw = Path(f'/proc/{pid}/stat').read_text()
    fields = raw.rsplit(')', 1)[1].split()
    return int(fields[1]), int(fields[19])


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    starttime_ticks: int
    uid: int
    gid: int
    boot_id: str
    exe_device: int
    exe_inode: int
    cmdline_sha256: str


def process_identity(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('positive actual process PID required')
    _, start = _stat_fields(pid)
    proc = Path(f'/proc/{pid}')
    info = proc.stat()
    exe = (proc / 'exe').stat()
    result = ProcessIdentity(pid, start, info.st_uid, info.st_gid,
        Path('/proc/sys/kernel/random/boot_id').read_text().strip(), exe.st_dev, exe.st_ino,
        _hash((proc / 'cmdline').read_bytes()))
    if _stat_fields(pid)[1] != start:
        raise ValueError('process PID was reused during identity read')
    return result


def peer_process_identity(connection):
    pid, uid, gid = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    identity = process_identity(pid)
    if (identity.uid, identity.gid) != (uid, gid):
        raise ValueError('kernel peer and current process credentials differ')
    return identity


def _identity(value):
    if not isinstance(value, dict) or set(value) != set(ProcessIdentity.__dataclass_fields__):
        raise ValueError('complete process identity required')
    return ProcessIdentity(**value)


@dataclass(frozen=True)
class ParentLaunchEnvelope:
    schema: str
    package_name: str
    registered_intent: dict
    core_identity: dict
    actual_identity: dict
    core_source_sha256: str
    semantic_sha256: str
    frozen_config: dict
    parent: ProcessIdentity
    socket_path: str
    nonce: str
    issued_monotonic_s: float
    deadline_monotonic_s: float
    role: str
    device_plan: dict
    checkpoint_origin: object


@dataclass(frozen=True)
class WorkerLaunchReply:
    schema: str
    envelope_sha256: str
    intent_sha256: str
    actual_identity: dict
    nonce: str
    server: ProcessIdentity
    worker: ProcessIdentity
    spawn_root: ProcessIdentity
    lock_device: int
    lock_inode: int
    process_config: dict
    authorization: dict
    resource: dict
    role: str
    checkpoint_origin: object
    issued_monotonic_s: float
    deadline_monotonic_s: float


def _envelope(value):
    if not isinstance(value, dict) or set(value) != set(ParentLaunchEnvelope.__dataclass_fields__):
        raise ValueError('complete parent launch envelope required')
    value = dict(value, parent=_identity(value['parent']))
    return ParentLaunchEnvelope(**value)


def _reply(value):
    if not isinstance(value, dict) or set(value) != set(WorkerLaunchReply.__dataclass_fields__):
        raise ValueError('complete worker launch reply required')
    value = dict(value, **{key: _identity(value[key]) for key in ('server', 'worker', 'spawn_root')})
    return WorkerLaunchReply(**value)


def _registered(repository, package, intent_binding):
    from scripts import run_dh116_work_packages as scheduler
    name, current = worker_registered_native_package(repository, package)
    path, raw = _read(intent_binding)
    intent = json.loads(raw)
    recipe = repository.native_recipe(current)
    launch = repository.actual_launch(current)
    identity = repository.native_execution_identity(current, recipe=recipe)
    if launch is None or identity['actual_identity'] is None:
        raise ValueError('actual registered launch approval required')
    approval = scheduler.native_approval(intent['approval'], repository)
    expected_approval = dict(**identity['core_identity'], actual_identity=identity['actual_identity'])
    expected_parent = repository.private / 'native-attempts' / name / identity['attempt_fingerprint']
    expected_output = Path(approval['output_root']) / name / path.parent.name / 'native'
    if (path.name != 'intent.json' or path.parent.parent != expected_parent
            or not path.parent.name.startswith('attempt-') or intent.get('package') != name
            or intent.get('head') != repository.head() or intent.get('output') != str(expected_output)
            or approval['packages'].get(name) != expected_approval
            or any(intent.get(key) != value for key, value in identity.items())
            or any(intent.get(key) != value for key, value in identity['core_identity'].items())
            or intent.get('actual_launch') != current.get('actual_launch')):
        raise ValueError('registered actual launch intent/source/approval changed')
    return name, recipe, launch, identity, intent


def _verify_scheduler_parent(envelope, repository):
    from scripts import run_dh116_work_packages as scheduler
    pid = envelope.parent.pid
    args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    decoded = [value.decode() for value in args if value]
    expected = ['--native-worker', envelope.registered_intent['path'], '--evidence-dir', str(repository.private)]
    source = 'scripts/run_dh116_work_packages.py'
    if (len(decoded) != 6 or decoded[1] not in (source, str(repository.root/source))
            or decoded[2:] != expected or Path(f'/proc/{pid}/cwd').resolve() != repository.root):
        raise ValueError('actual scheduler native-worker process argv/source required')
    executable = Path(sys.executable).stat()
    if (envelope.parent.exe_device, envelope.parent.exe_inode) != (executable.st_dev, executable.st_ino):
        raise ValueError('actual scheduler interpreter differs')
    # Both existing scheduler locks are held by the issuer itself. Neither is
    # the production native lock, which belongs exclusively to the inferior.
    verify_kernel_lock(pid, Path(envelope.registered_intent['path']).parent/'worker.lock')
    verify_kernel_lock(pid, scheduler.NATIVE_SCHEDULER_LOCK)


def _wall(envelope, recipe):
    issued, deadline, now = envelope.issued_monotonic_s, envelope.deadline_monotonic_s, time.monotonic()
    limit = recipe['spec']['teacher_contract']['limits']['inner_wall_timeout_s']
    if (any(type(v) not in (float, int) or not math.isfinite(v) for v in (issued, deadline))
            or not 0 < issued <= now < deadline or deadline-issued > limit):
        raise ValueError('actual launch monotonic budget expired or enlarged')


def _socket_path(path):
    path = Path(path)
    parent = _regular_path(path.parent)
    info = parent.stat()
    sock = path.lstat()
    if (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700
            or not stat.S_ISDIR(info.st_mode) or not stat.S_ISSOCK(sock.st_mode)
            or stat.S_IMODE(sock.st_mode) != 0o600 or sock.st_uid != os.getuid()):
        raise ValueError('private parent launch socket required')
    return path


def make_parent_launch_envelope(*, repository, package, intent_binding, launch, socket_path, nonce, deadline):
    name, recipe, current, identity, _ = _registered(repository, package, intent_binding)
    if launch != current or not isinstance(nonce, str) or len(nonce) < 32:
        raise ValueError('current actual launch and fresh nonce required')
    contract = recipe['spec']['teacher_contract']
    _socket_path(socket_path)
    result = ParentLaunchEnvelope(ENVELOPE_SCHEMA, name, deepcopy(intent_binding), identity['core_identity'],
        identity['actual_identity'], contract['source_sha256'], contract['semantic_sha256'],
        deepcopy(contract['training_config']), process_identity(os.getpid()), str(socket_path), nonce,
        time.monotonic(), deadline, current['role'], deepcopy(current['device_plan']),
        deepcopy(current['checkpoint_origin']))
    _wall(result, recipe)
    _verify_scheduler_parent(result, repository)
    return result


def _static(envelope, *, repository, package=None, check_socket=True):
    if not isinstance(envelope, ParentLaunchEnvelope) or envelope.schema != ENVELOPE_SCHEMA:
        raise ValueError('versioned actual launch envelope required')
    if package is None:
        package = repository.registry()['packages'][envelope.package_name]
    name, recipe, launch, identity, _ = _registered(repository, package, envelope.registered_intent)
    contract = recipe['spec']['teacher_contract']
    if (envelope.package_name != name or envelope.core_identity != identity['core_identity']
            or envelope.actual_identity != identity['actual_identity']
            or envelope.core_source_sha256 != contract['source_sha256']
            or envelope.semantic_sha256 != contract['semantic_sha256']
            or envelope.frozen_config != contract['training_config']
            or envelope.role != launch['role'] or envelope.device_plan != launch['device_plan']
            or envelope.checkpoint_origin != launch['checkpoint_origin']
            or process_identity(envelope.parent.pid) != envelope.parent
            or envelope.parent.uid != os.getuid()):
        raise ValueError('actual parent envelope current source/process binding differs')
    _read(envelope.frozen_config)
    _wall(envelope, recipe)
    _verify_scheduler_parent(envelope, repository)
    if check_socket:
        _socket_path(envelope.socket_path)
    return package, recipe


def read_launch_binding(binding, *, repository, package=None):
    """Static pre-lock verification. This never connects or acquires a lock."""
    try:
        path, raw = _read(binding)
        envelope = _envelope(json.loads(raw))
        _static(envelope, repository=repository, package=package)
        expected = Path(envelope.registered_intent['path']).parent / 'actual-launch' / 'binding.json'
        if path != expected or _encode(envelope) != raw:
            raise ValueError('binding is not the canonical current attempt envelope')
        return envelope
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError('actual_parent_binding_invalid: ' + str(error)) from error


def validate_parent_launch_envelope(envelope, *, repository, package, connection, expected_binding_sha256):
    try:
        _static(envelope, repository=repository, package=package)
        if (_hash(_encode(envelope)) != expected_binding_sha256
                or peer_process_identity(connection) != envelope.parent):
            raise ValueError('parent kernel peer/envelope SHA differs')
        return envelope
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError('actual_parent_binding_invalid: ' + str(error)) from error


def verify_kernel_lock(pid, lock):
    lock = _regular_path(lock)
    info = lock.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('regular current singleton lock required')
    observed = []
    for row in Path('/proc/locks').read_text().splitlines():
        fields = row.split()
        if len(fields) < 8 or fields[1:4] != ['FLOCK', 'ADVISORY', 'WRITE']:
            continue
        major, minor, inode = fields[5].split(':')
        if (int(major, 16), int(minor, 16), int(inode)) == (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino):
            observed.append(int(fields[4]))
    if observed != [pid]:
        raise ValueError('actual worker kernel singleton lock must be held by peer itself')
    return info.st_dev, info.st_ino


def _verify_ancestry(peer, spawn_root):
    if process_identity(spawn_root.pid) != spawn_root:
        raise ValueError('spawn root identity changed')
    pid = peer.pid
    for _ in range(64):
        current = process_identity(pid)
        parent, start = _stat_fields(pid)
        if (current.uid != peer.uid or current.boot_id != peer.boot_id
                or current.starttime_ticks != start or process_identity(pid) != current):
            raise ValueError('worker ancestor credential/lifetime changed')
        if pid == spawn_root.pid:
            if current != spawn_root:
                raise ValueError('worker ancestor root differs')
            return
        if parent <= 1:
            raise ValueError('peer is not a stable current spawned descendant')
        pid = parent
    raise ValueError('worker ancestry exceeds finite launch depth')


def verify_peer_process(connection, *, spawn_root, repository, package):
    peer = peer_process_identity(connection)
    if peer.pid == os.getpid() or peer.uid != os.getuid() or peer.boot_id != spawn_root.boot_id:
        raise ValueError('worker cannot self-issue or use foreign credentials')
    _verify_ancestry(peer, spawn_root)
    recipe = repository.native_recipe(package)
    spec = recipe['spec']
    executable = Path(sys.executable).stat()
    if (peer.exe_device, peer.exe_inode) != (executable.st_dev, executable.st_ino):
        raise ValueError('worker executable differs from scheduler interpreter')
    args = Path(f'/proc/{peer.pid}/cmdline').read_bytes().split(b'\0')
    decoded = [value.decode() for value in args if value]
    entry = str(repository.root / spec['entrypoint'])
    if len(decoded) < 2 or decoded[1] not in (entry, spec['entrypoint']):
        raise ValueError('kernel worker command is not the frozen entrypoint')
    expected = spec['argv']
    if len(decoded[2:]) != len(expected):
        raise ValueError('kernel worker argv length differs')
    for actual, wanted in zip(decoded[2:], expected):
        if wanted != '{output}' and actual != wanted:
            raise ValueError('kernel worker argv differs from registered recipe')
    # native_recipe hashes entrypoint plus all declared transitive producer code.
    if process_identity(peer.pid) != peer:
        raise ValueError('worker identity changed during ancestry/source verification')
    return peer


def _session_dir(envelope):
    path = Path(envelope.registered_intent['path']).parent / 'actual-launch'
    if (path.resolve(strict=True) != path or stat.S_IMODE(path.stat().st_mode) != 0o700
            or path.stat().st_uid != os.getuid()):
        raise ValueError('private current attempt session directory required')
    return path


def issue_process_overlay(*, repository, package, envelope, connection, spawn_root, hello):
    """Server-only issuer: reread registration, kernel peer, ancestry and FLOCK."""
    from scripts import run_dh116_work_packages as scheduler
    from .tabletop_native_teacher_preflight import authorization_config_digest
    _, recipe, _, _, _ = _registered(repository, package, envelope.registered_intent)
    _static(envelope, repository=repository, package=package)
    if process_identity(os.getpid()) != envelope.parent:
        raise ValueError('only the registered live parent can issue a grant')
    peer = verify_peer_process(connection, spawn_root=spawn_root, repository=repository, package=package)
    _, intent_raw = _read(envelope.registered_intent)
    intended_output = json.loads(intent_raw)['output']
    args = Path(f'/proc/{peer.pid}/cmdline').read_bytes().split(b'\0')
    decoded = [value.decode() for value in args if value]
    output_index = recipe['spec']['argv'].index('{output}') + 2
    if decoded[output_index] != intended_output:
        raise ValueError('kernel worker output differs from approved intent')
    lock_device, lock_inode = verify_kernel_lock(peer.pid, scheduler.NATIVE_LOCK)
    if (not isinstance(hello, dict) or set(hello) != {'nonce', 'lock_device', 'lock_inode'}
            or hello['nonce'] != envelope.nonce
            or type(hello['lock_device']) is not int or hello['lock_device'] != lock_device
            or type(hello['lock_inode']) is not int or hello['lock_inode'] != lock_inode):
        raise ValueError('worker hello nonce/kernel lock identity differs')
    _, raw = _read(envelope.frozen_config)
    training = json.loads(raw)
    config = deepcopy(training['preflight'])
    if config.get('authorization') is not None or config.get('resource_lock') is not None:
        raise ValueError('frozen config must not carry a process authorization')
    spec = recipe['spec']; contract = spec['teacher_contract']; limits = contract['limits']
    budget = dict(max_sdk_steps=spec['budget']['physics_steps'], max_resets=limits['max_resets'],
        max_solver_calls=limits['max_solver_calls'], max_episode_count=limits['max_episode_count'],
        wall_timeout_s=limits['inner_wall_timeout_s'])
    if config.get('budget') != budget:
        raise ValueError('frozen five-key budget must derive from the approved recipe')
    directory = _session_dir(envelope)
    session = envelope.nonce
    auth = dict(schema='dh116-native-teacher-budget-authorization-v1', scope='new_native_teacher_ppo',
        profile=config['profile'], style=config['style'], budget=budget, max_ppo_updates=1,
        config_sha256=authorization_config_digest(config), session_uid=session, owner_pid=peer.pid,
        owner_starttime_ticks=peer.starttime_ticks, boot_id=peer.boot_id, recipe_identity=envelope.core_identity,
        semantic_sha256=envelope.semantic_sha256, source_sha256=envelope.core_source_sha256,
        issued_monotonic_s=envelope.issued_monotonic_s, deadline_monotonic_s=envelope.deadline_monotonic_s,
        actual_identity=envelope.actual_identity, intent_sha256=envelope.registered_intent['sha256'],
        nonce=envelope.nonce, server=asdict(envelope.parent), role=envelope.role)
    authorization = _write(directory / 'authorization.json', auth)
    resource = dict(schema='dh116-native-resource-reservation-v1', session_uid=session,
        resource_class='isaac_native_singleton', authorization_sha256=authorization['sha256'], pid=peer.pid,
        owner_starttime_ticks=peer.starttime_ticks, boot_id=peer.boot_id, lock_device=lock_device,
        lock_inode=lock_inode, recipe_identity=envelope.core_identity)
    reservation = _write(directory / 'resource.json', resource)
    training['preflight'] = dict(config, authorization=authorization, resource_lock=reservation)
    process_config = _write(directory / 'process.json', training)
    # Recheck after exclusive writes; a changed intent or released lock cannot grant.
    _registered(repository, package, envelope.registered_intent)
    if peer_process_identity(connection) != peer or verify_kernel_lock(peer.pid, scheduler.NATIVE_LOCK) != (lock_device, lock_inode):
        raise ValueError('worker changed while issuing launch overlay')
    return WorkerLaunchReply(REPLY_SCHEMA, _hash(_encode(envelope)), envelope.registered_intent['sha256'],
        envelope.actual_identity, envelope.nonce, envelope.parent, peer, spawn_root, lock_device, lock_inode,
        process_config, authorization, reservation, envelope.role, envelope.checkpoint_origin,
        envelope.issued_monotonic_s, envelope.deadline_monotonic_s)


def validate_worker_launch_reply(reply, *, envelope, repository, package, connection):
    from scripts import run_dh116_work_packages as scheduler
    try:
        _static(envelope, repository=repository, package=package)
        current = process_identity(os.getpid())
        device, inode = verify_kernel_lock(current.pid, scheduler.NATIVE_LOCK)
        if (not isinstance(reply, WorkerLaunchReply) or reply.schema != REPLY_SCHEMA
                or reply.envelope_sha256 != _hash(_encode(envelope))
                or reply.intent_sha256 != envelope.registered_intent['sha256']
                or reply.actual_identity != envelope.actual_identity or reply.nonce != envelope.nonce
                or reply.server != peer_process_identity(connection) or reply.server != envelope.parent
                or reply.worker != current or reply.lock_device != device or reply.lock_inode != inode
                or reply.role != envelope.role or reply.checkpoint_origin != envelope.checkpoint_origin
                or reply.issued_monotonic_s != envelope.issued_monotonic_s
                or reply.deadline_monotonic_s != envelope.deadline_monotonic_s):
            raise ValueError('worker reply source/peer/session/role/deadline differs')
        _verify_ancestry(current, reply.spawn_root)
        directory = _session_dir(envelope)
        for field, name in ((reply.process_config, 'process.json'), (reply.authorization, 'authorization.json'), (reply.resource, 'resource.json')):
            path, _ = _read(field)
            if path != directory / name:
                raise ValueError('worker reply file escapes this attempt')
        _, raw = _read(reply.process_config); config = json.loads(raw)
        if config['preflight']['authorization'] != reply.authorization or config['preflight']['resource_lock'] != reply.resource:
            raise ValueError('process config authorization binding differs')
        _, raw = _read(reply.authorization); auth = json.loads(raw)
        if any(auth.get(key) != value for key, value in dict(actual_identity=envelope.actual_identity,
                intent_sha256=envelope.registered_intent['sha256'], nonce=envelope.nonce,
                server=asdict(envelope.parent), role=envelope.role,
                issued_monotonic_s=envelope.issued_monotonic_s,
                deadline_monotonic_s=envelope.deadline_monotonic_s).items()):
            raise ValueError('parent process overlay actual contract differs')
        core = verify_worker_lease(config['preflight'], repo=repository, package=package,
            expected_semantic_sha256=envelope.semantic_sha256)
        return dict(core_lease=core, process_config=config, server=peer_process_identity(connection))
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError('actual_worker_reply_invalid: ' + str(error)) from error


class ActualWorkerLease:
    __slots__ = ('_core', '_config', '_server', '_binding', '_reply_binding', '_envelope', '_reply', '_repository', '_package', '_closed')

    def __init__(self, token=None, *, verified=None, binding=None, reply_binding=None, envelope=None, reply=None, repository=None, package=None):
        if token is not _LEASE_TOKEN:
            raise TypeError('actual worker lease requires verified parent/kernel/source handshake')
        self._core = verified['core_lease']; self._config = deepcopy(verified['process_config'])
        self._server = verified['server']; self._binding = deepcopy(binding); self._reply_binding = deepcopy(reply_binding)
        self._envelope = envelope; self._reply = reply; self._repository = repository; self._package = deepcopy(package)
        self._closed = False

    @property
    def process_config(self):
        return deepcopy(self._config)

    @property
    def holder_pid(self):
        return self._core.holder_pid

    @property
    def role(self):
        return self._reply.role

    @property
    def resume_origin(self):
        return deepcopy(self._reply.checkpoint_origin)

    @property
    def device_plan(self):
        return deepcopy(self._envelope.device_plan)

    @property
    def launch_identity(self):
        return deepcopy(self._reply.actual_identity)

    @property
    def deadline_monotonic_s(self):
        return self._envelope.deadline_monotonic_s

    def revalidate(self):
        if self._closed:
            raise ValueError('actual worker lease is closed')
        self._core.revalidate()
        self._core_source_files()
        if process_identity(self._server.pid) != self._server:
            raise ValueError('original kernel parent peer identity changed')

    def _core_source_files(self):
        # Grant closes IPC. Only original immutable files and live /proc are read.
        _, raw = _read(self._binding)
        envelope = _envelope(json.loads(raw))
        _static(envelope, repository=self._repository, package=self._package, check_socket=False)
        if envelope != self._envelope or _encode(envelope) != raw:
            raise ValueError('parent launch binding changed')
        _, raw = _read(self._reply_binding)
        if _reply(json.loads(raw)) != self._reply:
            raise ValueError('worker launch reply changed')
        for binding in (self._reply.process_config, self._reply.authorization, self._reply.resource):
            _read(binding)

    def close(self):
        self._closed = True
        self._core.close()

    def __reduce__(self):
        raise TypeError('actual process lease cannot be serialized')


def _receive(connection):
    raw, _, flags, _ = connection.recvmsg(_PACKET_LIMIT)
    if not raw or flags & socket.MSG_TRUNC:
        raise ValueError('missing or oversized launch packet')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('launch packet must be one JSON object')
    return value


def request_actual_launch(binding, *, repository, package=None):
    from scripts import run_dh116_work_packages as scheduler
    envelope = read_launch_binding(binding, repository=repository, package=package)
    if package is None:
        package = repository.registry()['packages'][envelope.package_name]
    device, inode = verify_kernel_lock(os.getpid(), scheduler.NATIVE_LOCK)
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        connection.settimeout(max(.001, envelope.deadline_monotonic_s-time.monotonic()))
        connection.connect(envelope.socket_path)
        validate_parent_launch_envelope(envelope, repository=repository, package=package,
            connection=connection, expected_binding_sha256=binding['sha256'])
        connection.sendall(_encode(dict(nonce=envelope.nonce, lock_device=device, lock_inode=inode)))
        reply_binding = _receive(connection)
        path, raw = _read(reply_binding)
        if path != _session_dir(envelope) / 'reply.json':
            raise ValueError('reply path differs from current attempt')
        reply = _reply(json.loads(raw))
        verified = validate_worker_launch_reply(reply, envelope=envelope, repository=repository, package=package, connection=connection)
    return ActualWorkerLease(_LEASE_TOKEN, verified=verified, binding=binding, reply_binding=reply_binding,
        envelope=envelope, reply=reply, repository=repository, package=package)


class ActualLaunchServer:
    """One scheduler-owned listener; no daemon, retry or reconnect."""
    def __init__(self, *, repository, package, intent_binding, launch, directory, deadline):
        self.repository = repository; self.package = deepcopy(package)
        self.intent_binding = deepcopy(intent_binding); self.launch = deepcopy(launch)
        self.directory = Path(directory); self.deadline = deadline
        self.listener = None; self.socket_directory = None; self._served = False

    def __enter__(self):
        expected = Path(self.intent_binding['path']).parent / 'actual-launch'
        if self.directory != expected or self.directory.parent.resolve(strict=True) != self.directory.parent:
            raise ValueError('actual launch metadata must be in current attempt')
        self.directory.mkdir(mode=0o700)
        try:
            # AF_UNIX sockaddr_un has a 108-byte pathname bound. Keep metadata
            # in the attempt and use a separately pinned short private socket.
            self.socket_directory = Path(tempfile.mkdtemp(prefix='dh116-al-'))
            sock = self.socket_directory / 'launch.sock'
            self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self.listener.bind(str(sock)); sock.chmod(0o600); self.listener.listen(1)
            self.envelope = make_parent_launch_envelope(repository=self.repository, package=self.package,
                intent_binding=self.intent_binding, launch=self.launch, socket_path=sock,
                nonce=secrets.token_hex(32), deadline=self.deadline)
            self.binding = _write(self.directory / 'binding.json', self.envelope)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def serve_once(self, child):
        if self._served:
            raise ValueError('actual launch listener already consumed')
        self._served = True
        try:
            spawn_root = process_identity(child.pid)
            while time.monotonic() < self.deadline:
                ready, _, _ = select.select([self.listener], [], [], min(.05, max(0, self.deadline-time.monotonic())))
                if ready:
                    connection, _ = self.listener.accept()
                    self.listener.close(); self.listener = None
                    with connection:
                        connection.settimeout(max(.001, self.deadline-time.monotonic()))
                        hello = _receive(connection)
                        # The Popen root may exec bash -> GDB. PID/starttime/boot
                        # stay fixed; capture executable identity after hello.
                        current_root = process_identity(child.pid)
                        for key in ('pid', 'starttime_ticks', 'uid', 'gid', 'boot_id'):
                            if getattr(current_root, key) != getattr(spawn_root, key):
                                raise ValueError('Popen root lifetime changed before grant')
                        spawn_root = current_root
                        reply = issue_process_overlay(repository=self.repository, package=self.package,
                            envelope=self.envelope, connection=connection, spawn_root=spawn_root, hello=hello)
                        binding = _write(self.directory / 'reply.json', reply)
                        connection.sendall(_encode(binding))
                        return binding
                if child.poll() is not None:
                    raise ValueError('worker exited before actual launch grant')
            raise ValueError('actual launch handshake deadline expired')
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ValueError('actual launch grant rejected: ' + str(error)) from error
        finally:
            if self.listener is not None:
                self.listener.close(); self.listener = None

    def __exit__(self, *exc):
        if self.listener is not None:
            self.listener.close(); self.listener = None
        if self.socket_directory is not None:
            sock = self.socket_directory / 'launch.sock'
            try:
                sock.unlink()
            except FileNotFoundError:
                pass
            self.socket_directory.rmdir()
            self.socket_directory = None
