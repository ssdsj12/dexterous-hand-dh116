"""Preserve owned native batch rows as one typed binary evidence frame per window."""
from dataclasses import fields, is_dataclass
import json
import math
import os
from pathlib import Path
import struct
from tempfile import NamedTemporaryFile

import numpy as np

from .tabletop_deformable_r0_replay import encode_owned


MAGIC = b"DH116B1\0"
HEADER = struct.Struct("<8sQQ")
ARRAY = "__dh116_batch_array__"


def _buffer(value, payload):
    item_type = type(value)
    if value is None or item_type in (str, bool, int):
        return value
    if item_type is float:
        return value if math.isfinite(value) else encode_owned(value)
    if item_type in (list, tuple):
        return [_buffer(item, payload) for item in value]
    if is_dataclass(value):
        return {field.name: _buffer(getattr(value, field.name), payload) for field in fields(value)}
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "bifu":
            raise ValueError("numeric owned arrays only")
        start = len(payload)
        payload.extend(memoryview(np.ascontiguousarray(value)).cast("B") if value.size else b"")
        return {ARRAY: [start, len(payload) - start, value.dtype.str, list(value.shape)]}
    if isinstance(value, dict) or hasattr(value, "items"):
        result = {}
        for key, item in value.items():
            if type(key) not in (int, str) or str(key) in result:
                raise ValueError("distinct native packet field names required")
            result[str(key)] = _buffer(item, payload)
        return result
    return encode_owned(value)


def _restore(value, binary):
    if isinstance(value, dict):
        if set(value) == {ARRAY}:
            start, size, dtype, shape = value[ARRAY]
            dtype = np.dtype(dtype)
            if (type(start) is not int or type(size) is not int or dtype.kind not in "bifu"
                    or any(type(n) is not int or n < 0 for n in shape)
                    or not 0 <= start <= start + size <= len(binary)
                    or math.prod(shape) * dtype.itemsize != size):
                raise ValueError("invalid typed array in batch evidence")
            return np.frombuffer(binary[start:start + size], dtype=dtype).copy().reshape(shape)
        if set(value) == {"__dh116_nonfinite__"}:
            from .tabletop_deformable_r0_replay import decode_owned

            return decode_owned(value)
        return {key: _restore(item, binary) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item, binary) for item in value]
    return value


class LaneEvidence:
    """Keep the existing per-lane journal sequence and scope."""

    scope = "owner_source_unqualified"

    def __init__(self, owner, lane):
        self._owner, self._lane, self._sequence = owner, lane, 0
        self.path = owner.directory / f"lane_{lane:03d}.dhb"

    def emit(self, operation, payload):
        sequence = self._sequence
        self._owner._append(self._lane, sequence, operation, payload)
        self._sequence += 1
        return sequence


class BatchEvidence:
    """Bound in-memory evidence to the setup or current control window."""

    def __init__(self, directory, *, lane_count, full_windows=True):
        if type(lane_count) is not int or not 2 <= lane_count <= 64:
            raise ValueError("two to 64 evidence lanes required")
        self.directory = Path(directory)
        self.full_windows = full_windows
        self._lanes = tuple(LaneEvidence(self, i) for i in range(lane_count))
        self._index = 0
        self._rows = []
        self._payload = bytearray()

    def lane(self, lane):
        return self._lanes[lane]

    def _append(self, lane, sequence, operation, payload):
        if type(operation) is not str or not operation:
            raise ValueError("explicit native evidence operation required")
        if len(self._rows) >= 32 * len(self._lanes):
            raise ValueError("current batch evidence exceeded one window")
        # The batch episode delivers owning rows rather than simulator views.
        # Retain those objects until the control boundary. The full audit
        # mode continues to freeze them at each emit for external readers.
        packet = payload
        if self.full_windows:
            offset = len(self._payload)
            try:
                packet = _buffer(payload, self._payload)
            except BaseException:
                del self._payload[offset:]
                raise
        self._rows.append(dict(lane=lane, sequence=sequence, operation=operation,
                               scope=self._lanes[lane].scope, payload=packet))

    def seal(self, outcome):
        if not self._rows:
            return None
        if not self.full_windows and outcome != "rejected":
            self._rows.clear()
            return None
        if self.full_windows:
            rows, payload = self._rows, self._payload
        else:
            payload = bytearray()
            rows = _buffer(self._rows, payload)
        metadata = json.dumps(dict(index=self._index, outcome=outcome, rows=rows),
                              separators=(",", ":"), allow_nan=False).encode()
        path = self.directory / f"window_{self._index:06d}.dhb"
        temporary = None
        try:
            with NamedTemporaryFile(mode="wb", dir=self.directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(HEADER.pack(MAGIC, len(metadata), len(payload)))
                stream.write(metadata)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self._rows.clear()
        self._payload.clear()
        self._index += 1
        return path


def read_batch_evidence(path):
    """Decode owned numeric arrays without executing serialized payloads."""
    raw = Path(path).read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("truncated batch evidence header")
    magic, meta_size, data_size = HEADER.unpack_from(raw)
    if magic != MAGIC or len(raw) != HEADER.size + meta_size + data_size:
        raise ValueError("unknown or truncated batch evidence")
    end = HEADER.size + meta_size
    return _restore(json.loads(raw[HEADER.size:end]), memoryview(raw)[end:])
