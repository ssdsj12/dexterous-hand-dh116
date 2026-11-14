"""Private, crash-recoverable forensic ledger for exact completed MPC transitions."""

import base64
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat

import torch

from .tabletop_mpc_privileged_capture import (
    PreActionSample, SUBSTEPS, _same_clock, _valid_clock, file_identity,
)


VERSION = 'dh116-native-mpc-pre-action-journal-v1'
_DTYPES = {'float32': torch.float32, 'float64': torch.float64, 'bool': torch.bool}
_STATUS = {'source_archive_completed', 'quarantined_exception'}


def _encode_tensor(value):
    tensor = value.detach().cpu().contiguous()
    dtype = str(tensor.dtype).removeprefix('torch.')
    if dtype not in _DTYPES:
        raise ValueError('MPC journal requires original float32/float64/bool tensors')
    data = tensor.view(torch.uint8).numpy().tobytes()
    return dict(dtype=dtype, base64=base64.b64encode(data).decode('ascii'))


def _decode_tensor(value, shape, *, boolean=False):
    dtype = _DTYPES[value['dtype']]
    if (dtype == torch.bool) != boolean:
        raise ValueError('journal tensor dtype disagrees with canonical schema')
    raw = bytearray(base64.b64decode(value['base64'], validate=True))
    return torch.frombuffer(raw, dtype=dtype).reshape(shape).clone()


@dataclass(frozen=True)
class RecoveredJournal:
    status: str
    samples: tuple
    identities: tuple
    discarded_partial_line: bool
    qualified_for_bc: bool = False


class _Journal:
    def __init__(self, stream, environments, max_steps):
        self.stream = stream
        self.environments = environments
        self.max_steps = max_steps
        self.steps = 0
        self.rows = 0
        self.run_uuid = None
        self.previous_after = None
        self.known = {}
        self.closed = set()
        self.finished = False
        self._write(dict(kind='header', version=VERSION, environments=environments,
                         max_steps=max_steps, qualified_for_bc=False))

    def _write(self, row):
        self.stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def _identities(self, sample, record):
        labels = record['samples']
        if not torch.equal(labels['actions'], sample.action):
            raise ValueError('journal action differs from actual native submitted action')
        rows = []
        for slot, env in enumerate(labels['env_id']):
            if env is None:
                continue
            run = labels['run_uuid'][slot]
            uid = labels['episode_uid'][slot]
            if (type(env) is not int or env != slot or env >= self.environments
                    or not isinstance(run, str) or not run
                    or not isinstance(uid, str) or not uid or env in self.closed
                    or (self.run_uuid is not None and run != self.run_uuid)
                    or (env in self.known and self.known[env] != uid)
                    or labels['tick'][slot] != self.steps or not labels['valid'][slot]
                    or labels['first_physics_step'][slot] != sample.clock['sdk_step']
                    or labels['policy_timestamp_s'][slot] != sample.clock['sdk_time_s']):
                raise ValueError('journal native first-episode identity or pre-action clock differs')
            self.run_uuid = run
            self.known[env] = uid
            terminal = bool(labels['terminal'][slot])
            if terminal:
                self.closed.add(env)
            rows.append(dict(run_uuid=run, env_id=env, camera_episode_uid=uid, terminal=terminal))
        if not rows or len({row['camera_episode_uid'] for row in rows}) != len(rows):
            raise ValueError('journal requires unique live native first-episode identities')
        return rows

    def append(self, sample, after, sequence_record):
        if self.finished or self.steps >= self.max_steps:
            raise ValueError('journal policy budget exceeded or journal already closed')
        before, after = _valid_clock(sample.clock), _valid_clock(after)
        if (not _same_clock(after, before, SUBSTEPS)
                or (self.previous_after is not None and not _same_clock(before, self.previous_after))
                or before['world_step'] != self.steps * SUBSTEPS
                or sample.state.shape != (self.environments, 371)
                or sample.valid.shape != sample.state.shape
                or sample.action.shape != (self.environments, 24)):
            raise ValueError('journal requires a completed original 60Hz step with exact 480Hz clock')
        identities = self._identities(sample, sequence_record)
        self._write(dict(kind='step', tick=self.steps, before=before, after=after, identities=identities,
                         observation=_encode_tensor(sample.state), validity=_encode_tensor(sample.valid),
                         issued_action=_encode_tensor(sample.action)))
        self.previous_after = after
        self.steps += 1
        self.rows += len(identities)

    def _status(self, status, **detail):
        self._write(dict(kind='status', status=status, steps=self.steps,
                         qualified_for_bc=False, **detail))
        self.finished = True

    def finish(self, result):
        if self.finished or not self.steps or result['qualified_for_bc'] is not False:
            raise ValueError('cannot close an empty, completed, or qualified MPC diagnostic journal')
        if result['rows'] != self.rows:
            raise ValueError('original joined rows disagree with durable native identities')
        self._status('source_archive_completed', joined_rows=self.rows)


@contextmanager
def capture_journal(path, environments, max_steps):
    """Write a new private ledger. A missing closing marker remains quarantined."""
    path = Path(path)
    if (type(environments) is not int or not 1 <= environments <= 8
            or type(max_steps) is not int or not 1 <= max_steps <= 1800):
        raise ValueError('journal environments or bounded policy budget is invalid')
    if stat.S_IMODE(path.parent.stat().st_mode) & 0o077:
        raise ValueError('journal requires a private existing directory (mode 0700)')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
        writer = _Journal(stream, environments, max_steps)
        dirfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
        try:
            yield writer
        except BaseException as error:
            if not writer.finished:
                try:
                    writer._status('quarantined_exception', error_type=type(error).__name__)
                except OSError:
                    # Disk failure may prevent a final marker. The valid prefix
                    # remains inspectable under quarantined_interrupted.
                    pass
            raise


def recover_journal(path, expected_sha256):
    """Reopen only intact records as diagnostic evidence, never expert admission."""
    if file_identity(path)['sha256'] != expected_sha256:
        raise ValueError('MPC journal SHA differs')
    samples, identities = [], []
    status = 'quarantined_interrupted'
    partial = False
    after = None
    run_uuid = None
    active_episodes = {}
    terminal_envs = set()
    with Path(path).open('rb') as stream:
        header = json.loads(next(stream))
        n, limit = header['environments'], header['max_steps']
        if (header['kind'] != 'header' or header['version'] != VERSION
                or header['qualified_for_bc'] is not False or not 1 <= n <= 8 or not 1 <= limit <= 1800):
            raise ValueError('invalid bounded native MPC journal header')
        for line in stream:
            if status != 'quarantined_interrupted':
                raise ValueError('MPC journal data after closing marker')
            if not line.endswith(b'\n'):
                partial = True
                break
            row = json.loads(line)
            tick = len(samples)
            if row['kind'] == 'status':
                if (row['status'] not in _STATUS or row['qualified_for_bc'] is not False
                        or row['steps'] != tick or (row['status'] == 'source_archive_completed'
                        and row['joined_rows'] != sum(map(len, identities)))):
                    raise ValueError('invalid MPC journal closing marker')
                status = row['status']
                continue
            before = _valid_clock(row['before'])
            if (row['kind'] != 'step' or row['tick'] != tick or tick >= limit
                    or before['world_step'] != tick * SUBSTEPS
                    or (after is not None and not _same_clock(before, after))
                    or not _same_clock(row['after'], before, SUBSTEPS)):
                raise ValueError('MPC journal physical clock or bounded sequence differs')
            after = row['after']
            labels = row['identities']
            if not labels or len({label['env_id'] for label in labels}) != len(labels):
                raise ValueError('MPC journal native environment identity is absent or duplicated')
            for label in labels:
                env, run, uid = label['env_id'], label['run_uuid'], label['camera_episode_uid']
                if (type(env) is not int or not 0 <= env < n or env in terminal_envs
                        or not isinstance(run, str) or not run or not isinstance(uid, str) or not uid
                        or run_uuid not in (None, run) or active_episodes.get(env, uid) != uid
                        or type(label['terminal']) is not bool):
                    raise ValueError('MPC journal native first-episode identity differs')
                run_uuid = run
                active_episodes[env] = uid
                if label['terminal']:
                    terminal_envs.add(env)
            state = _decode_tensor(row['observation'], (n, 371))
            valid = _decode_tensor(row['validity'], (n, 371), boolean=True)
            action = _decode_tensor(row['issued_action'], (n, 24))
            if not torch.isfinite(state).all() or not torch.isfinite(action).all() or not (action.abs() <= 1).all():
                raise ValueError('MPC journal contains noncanonical 371D or 24D values')
            samples.append(PreActionSample(state, valid, action, before))
            identities.append(tuple(labels))
    if file_identity(path)['sha256'] != expected_sha256:
        raise ValueError('MPC journal SHA changed during recovery')
    return RecoveredJournal(status, tuple(samples), tuple(identities), partial)
