"""Immutable step snapshots and reusable reaction storage for DH116 runs."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import copy


class _FrozenList(list):
    """List-compatible value that rejects in-place mutation."""

    def _readonly(self, *args, **kwargs):
        raise TypeError("snapshot values are immutable")

    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = _readonly
    __iadd__ = __imul__ = _readonly


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            result = np.array(value, copy=True)
            result.setflags(write=False)
            return result
    except ImportError:
        pass
    return copy.deepcopy(value)


@dataclass(frozen=True)
class RuntimeSnapshot:
    step: int
    stage: str
    sample_time: float
    token: int
    values: Mapping[str, object]

    def __post_init__(self):
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("stage must be a nonempty string")
        if not isinstance(self.token, int) or self.token < 0:
            raise ValueError("token must be a nonnegative integer")
        object.__setattr__(self, "values", MappingProxyType(
            {key: _freeze(value) for key, value in dict(self.values).items()}))


class SnapshotStore:
    def __init__(self):
        self._next_token = 0

    def _capture(self, step, stage, sample_time, values):
        token = self._next_token
        self._next_token += 1
        return RuntimeSnapshot(step, stage, float(sample_time), token, values)

    def capture_pre(self, *, step, stage, sample_time, values):
        return self._capture(step, stage, sample_time, values)

    def capture_post(self, *, step, stage, sample_time, values):
        return self._capture(step, stage, sample_time, values)


class ReactionBuffer:
    """Reusable torch arrays for one articulation's body wrenches."""

    def __init__(self, force, torque, position, indices):
        self.force = force
        self.torque = torque
        self.position = position
        self.indices = indices

    @classmethod
    def allocate(cls, device, count, *, dtype=None):
        import torch
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        if dtype is None:
            dtype = torch.float32
        shape = (1, count, 3)
        return cls(
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
            torch.arange(count, device=device, dtype=torch.int32),
        )

    def clear_(self):
        self.force.zero_()
        self.torque.zero_()
        self.position.zero_()
        return self
