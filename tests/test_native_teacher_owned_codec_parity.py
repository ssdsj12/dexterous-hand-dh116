from dataclasses import dataclass
import json
from types import MappingProxyType

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_deformable_r0_replay as legacy
from go2_pvcnn.control.dh116 import tabletop_native_teacher_collector as collector


@pytest.mark.parametrize("source", [
    np.array([-0., 0., 1.25], dtype=np.float16),
    np.array([np.inf, -np.inf, np.nan, -0.]),
    np.arange(20, dtype=">i4")[::3],
    np.asarray(-0.),
    np.empty(0, dtype=np.float32),
    {"tensor": torch.tensor([-0., 1.], dtype=torch.float32), "tuple": (True, np.int32(4), None)},
])
def test_native_collector_preserves_legacy_copy_and_digest_bytes(source):
    assert collector._digest(source) == legacy._digest(source)
    assert legacy._digest(collector._copy(source)) == legacy._digest(legacy._copy(source))


@pytest.mark.parametrize("source", [object(), np.array([1j]), {1: "invalid"}])
def test_native_collector_rejects_non_owned_values(source):
    with pytest.raises(ValueError):
        collector._copy(source)


def test_finite_native_arrays_bypass_legacy_scalar_recursion(monkeypatch, tmp_path):
    value = np.array([[-0., 2.], [1., 3.]], dtype=np.float64)
    expected_digest = legacy._digest(value)
    frozen_bytes = json.dumps(legacy.encode_owned(value), sort_keys=True)
    def reject(_):
        raise AssertionError("finite owned array entered legacy encoder")
    monkeypatch.setattr(legacy, "encode_owned", reject)

    duplicate = collector._copy(value)
    assert not np.shares_memory(duplicate, value)
    assert np.signbit(duplicate[0, 0])
    assert collector._digest(value) == expected_digest

    @dataclass
    class Packet:
        array: object

    p = tmp_path / "events.jsonl"
    journal = collector.NativeJournal(p)
    try:
        assert journal.emit("physical", MappingProxyType({"packet": Packet(value)})) == 0
        row = json.loads(p.read_text())
        assert json.dumps(row["payload"]["packet"]["array"], sort_keys=True) == frozen_bytes
    finally:
        journal.close()
    assert legacy.encode_owned is reject
