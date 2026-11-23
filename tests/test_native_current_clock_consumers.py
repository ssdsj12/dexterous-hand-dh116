"""Current-reader CPU probes remain separate from historical native evidence."""

import pytest


def test_current_clock_readers_preserve_owned_stamps_without_claiming_native_readiness():
    from go2_pvcnn.control.dh116 import tabletop_native_clock_consumers_v3 as current
    from go2_pvcnn.control.dh116 import tabletop_native_clock_consumers_v2 as historical

    table = current.probe_current_clock_consumers()
    assert table["schema"] == current.VERSION and table["schema"] != historical.VERSION
    assert table["source_sha256"] != historical.CONSUMER_SOURCE_SHA256
    assert len(table["rows"]) == 32 and all(row["passed"] for row in table["rows"])
    assert any(row["sdk_timestamp_s"] != row["rod_timestamp_s"] for row in table["rows"])
    assert table["native_steps"] == 0
    assert table["actual_sdk_precision_qualified"] is False
    assert table["native_ready"] is False


def test_current_clock_probe_rejects_a_changed_source_pin(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_native_clock_consumers_v3 as current

    changed = dict(current.SOURCE_SHA256)
    key = next(iter(changed))
    changed[key] = "0" * 64
    monkeypatch.setattr(current, "SOURCE_SHA256", changed)
    with pytest.raises(ValueError, match=key):
        current.probe_current_clock_consumers()
