"""Exact codec contract on native-shaped CPU values and the inherited journal."""
import json
import math
import numpy as np
import pytest

from go2_pvcnn.control.dh116 import tabletop_deformable_dual_evidence as journal
from go2_pvcnn.control.dh116 import tabletop_deformable_r0_replay as legacy
from go2_pvcnn.control.dh116 import tabletop_physical_evidence_encoding as fast


def as_bytes(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


@pytest.mark.parametrize("dtype", [np.bool_, np.uint64, np.int32, np.float16, np.float32, np.float64, ">f4"])
def test_owned_codec_exact_bytes(dtype):
    base = np.arange(48).reshape(4, 12).astype(dtype)
    if base.dtype.kind == "f":
        base[0, :4] = [-0.0, np.nan, np.inf, -np.inf]
    payload = {"window": [base[:, ::3], base.T, base[:0], base[0, 0].reshape(()),
                          -0.0, math.nan, math.inf, -math.inf, (1, True, None, "测量")]}
    assert as_bytes(fast.encode_owned(payload)) == as_bytes(legacy.encode_owned(payload))


def test_frozen_unsupported_values_rejected():
    for bad in (np.array([object()], dtype=object), np.array([1j]), {1: [1.0]}, object()):
        with pytest.raises(ValueError, match="numeric|explicit|native"):
            fast.encode_owned(bad)


def test_whole_journal_encoder_scoped_and_guard_restored(monkeypatch, tmp_path):
    payload = {"vec": np.array([0.0, -0.0, np.nan], np.float32)}
    original = journal.encode_owned
    class Capture:
        def __init__(self, *args):
            self.terminal = None
        def accept(self, operation, value):
            if operation == "physical":
                self.terminal = "original terminal guard"
            assert as_bytes(legacy.encode_owned(value)) == as_bytes(legacy.encode_owned(payload))
    monkeypatch.setattr(journal, "_Sequence", Capture)

    def create(path):
        with journal.DualHandEvidenceJournal(path, physics_budget_end=10, solver_substeps=12) as writer:
            writer._append("physical", payload)

    create(tmp_path / "original.jsonl")
    with pytest.raises(RuntimeError, match="terminal"):
        with fast.encoding_scope():
            assert journal.encode_owned is fast.encode_owned
            create(tmp_path / "fast.jsonl")
            raise RuntimeError("terminal")
    assert journal.encode_owned is original
    assert (tmp_path / "fast.jsonl").read_bytes() == (tmp_path / "original.jsonl").read_bytes()


def test_finite_float_and_numeric_arrays_bypass_scalar_reencoding(monkeypatch):
    original = fast._LEGACY
    def reject(value):
        raise AssertionError("scalar fallback")
    monkeypatch.setattr(fast, "_LEGACY", reject)
    value = {"geometry": np.full((64, 32, 3), 0.125, np.float64)}
    assert as_bytes(fast.encode_owned(value)) == as_bytes(original(value))
