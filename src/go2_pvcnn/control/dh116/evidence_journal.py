"""Append-only, bounded-memory simulation evidence; never overwrite a run."""
import json
from pathlib import Path

from .evidence_json import dumps


class EvidenceJournal:
    def __init__(self, path):
        self.path = Path(path)
        self._file = self.path.open('x', encoding='utf-8')
        self._count = 0
        self._last = None

    def append(self, record):
        if self._file.closed:
            raise ValueError('cannot append to closed evidence journal')
        encoded = dumps(record)
        self._file.write(encoded + '\n')
        self._file.flush()
        self._last = encoded
        self._count += 1

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
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

