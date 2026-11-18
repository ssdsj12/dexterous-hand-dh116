import json

import pytest

from go2_pvcnn.control.dh116.perf_meter import PerfMeter


def test_perf_meter_records_sections_and_stage_summary(tmp_path):
    meter = PerfMeter(tmp_path / "perf.jsonl", physics_dt=1e-4)
    meter.begin_step(0, "approach", 0.0)
    with meter.section("control"):
        pass
    with meter.section("physx_step"):
        pass
    meter.end_step(completed=True)
    summary = meter.close()
    row = json.loads((tmp_path / "perf.jsonl").read_text().splitlines()[0])
    assert row["completed"] is True
    assert set(row["sections"]) == {"control", "physx_step"}
    assert summary["stages"]["approach"]["steps"] == 1
    assert summary["stages"]["approach"]["rtf"] >= 0.0


def test_perf_meter_marks_incomplete_step(tmp_path):
    meter = PerfMeter(tmp_path / "perf.jsonl", physics_dt=1e-4)
    meter.begin_step(7, "close", 0.0007)
    meter.end_step(completed=False)
    meter.close()
    row = json.loads((tmp_path / "perf.jsonl").read_text().splitlines()[0])
    assert row["completed"] is False


def test_perf_meter_rejects_nested_and_duplicate_sections(tmp_path):
    meter = PerfMeter(tmp_path / "perf.jsonl", physics_dt=1e-4)
    meter.begin_step(0, "settle", 0.0)
    with meter.section("control"):
        with pytest.raises(RuntimeError, match="nested"):
            with meter.section("inner"):
                pass
    with pytest.raises(RuntimeError, match="duplicate"):
        with meter.section("control"):
            pass
    meter.end_step(completed=False)
    meter.close()
