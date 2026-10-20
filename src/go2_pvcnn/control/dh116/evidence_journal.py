"""Append-only, bounded-memory simulation evidence; never overwrite a run."""
import json
import math
from pathlib import Path
from time import monotonic

from .evidence_json import dumps


class EvidenceJournal:
    def __init__(self, path, flush_policy='audit', flush_every=512, flush_seconds=.5):
        self.path = Path(path)
        self._file = self.path.open('x', encoding='utf-8')
        self._count = 0
        self._last = None
        self.flush_count = 0
        self._pending = 0
        self._last_flush = monotonic()
        self._closed = False
        self.configure(flush_policy=flush_policy, flush_every=flush_every,
                       flush_seconds=flush_seconds)

    def configure(self, *, flush_policy='audit', flush_every=512, flush_seconds=.5):
        if flush_policy not in ('audit', 'buffered'):
            raise ValueError("flush_policy must be 'audit' or 'buffered'")
        if isinstance(flush_every, bool) or not isinstance(flush_every, int) or flush_every < 1:
            raise ValueError('flush_every must be a positive integer')
        if not isinstance(flush_seconds, (int, float)) or isinstance(flush_seconds, bool):
            raise TypeError('flush_seconds must be a real scalar')
        if not math.isfinite(float(flush_seconds)) or flush_seconds <= 0:
            raise ValueError('flush_seconds must be finite and positive')
        self.flush_policy = flush_policy
        self.flush_every = int(flush_every)
        self.flush_seconds = float(flush_seconds)
        return self

    def append(self, record, *, force_flush=False):
        if self._file.closed or self._closed:
            raise ValueError('cannot append to closed evidence journal')
        encoded = dumps(record)
        self._file.write(encoded + '\n')
        self._last = encoded
        self._count += 1
        self._pending += 1
        if (force_flush or self.flush_policy == 'audit'
                or self._pending >= self.flush_every
                or monotonic() - self._last_flush >= self.flush_seconds):
            self.flush()

    def flush(self):
        if self._file.closed:
            return
        self._file.flush()
        self._pending = 0
        self._last_flush = monotonic()
        self.flush_count += 1

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if index != -1 or not self._count:
            raise IndexError('journal retains only the last record; iterate for history')
        return json.loads(self._last)

    def __iter__(self):
        with self.path.open(encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)

    def close(self):
        if self._closed:
            return
        self.flush()
        self._file.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

