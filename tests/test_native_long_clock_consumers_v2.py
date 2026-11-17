"""Frozen legacy clock-reader compatibility; CPU probes are not native runs."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from go2_pvcnn.control.dh116.tabletop_native_training_clock import ClockPolicyV2

SOURCE_MAP = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/"
    "long-clock-v2-20261115/clock-source-map.json")
MODES = ("fp32_dt_float64_sum", "float64_dt_float64_sum")
STEPS = (240, 258, 2402, 9600)
PREFIX = "long_clock_consumer_incompatible:"


def api():
    from go2_pvcnn.control.dh116 import tabletop_native_clock_consumers_v2
    return tabletop_native_clock_consumers_v2


def test_real_frozen_readers_produce_complete_owned_compatibility_table():
    gate = api()
    table = gate.probe_clock_consumers_v2()
    assert table["schema"] == "dh116-legacy-clock-consumers-v2"
    assert table["native_steps"] == 0
    assert len(table["rows"]) == 32
    assert set(table["source_sha256"]) == set(gate.CONSUMER_SOURCE_SHA256)
    assert gate.validate_clock_consumer_table_v2(table) == table
    assert len(table["table_sha256"]) == 64
    for row in table["rows"]:
        assert row["sdk_timestamp_s"] == gate.accumulated_timestamp_v2(
            row["sdk_step_abs"], row["timestamp_mode"])
        assert row["rod_timestamp_s"] == row["sdk_step_abs"] / 480
    copied = deepcopy(table)
    copied["rows"][0]["passed"] = not copied["rows"][0]["passed"]
    assert gate.probe_clock_consumers_v2() == table


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("step", STEPS)
def test_real_reader_results_preserve_distinct_sdk_and_rod_clock_sources(mode, step):
    gate = api()
    table = gate.probe_clock_consumers_v2()
    rows = {row["consumer"]: row for row in table["rows"]
            if row["timestamp_mode"] == mode and row["sdk_step_abs"] == step}
    assert len(rows) == 4
    for consumer, row in rows.items():
        expected = mode == MODES[1] or step < 2402 or consumer.endswith("bimanual_step._clock")
        assert row["passed"] is expected
        assert row["blocked_reason"] == (None if expected else PREFIX + consumer)
        assert row["timestamp_source"] == (
            "rod_counter_projection" if consumer.endswith("bimanual_step._clock")
            else "actual_sdk_accumulator")
    if mode == MODES[0]:
        assert all(row["sdk_timestamp_s"] != row["rod_timestamp_s"] for row in rows.values())


def test_actual_sdk_stamp_stress_diagnostic_does_not_relabel_nominal_rod_source():
    gate = api()
    table = gate.probe_clock_consumers_v2()
    row = next(row for row in table["rows"] if row["consumer"].endswith("bimanual_step._clock")
        and row["timestamp_mode"] == MODES[0] and row["sdk_step_abs"] == 240)
    assert row["passed"] is True
    assert row["actual_sdk_stamp_diagnostic"]["passed"] is False
    assert "480Hz SDK timestamp" in row["actual_sdk_stamp_diagnostic"]["error"]


def test_table_source_sha_tamper_rejected_even_with_recomputed_table_digest():
    gate = api()
    table = gate.probe_clock_consumers_v2()
    name = next(iter(table["source_sha256"]))
    table["source_sha256"][name] = "0" * 64
    table["table_sha256"] = gate.clock_table_sha256_v2(table)
    with pytest.raises(ValueError, match="clock_consumer_source"):
        gate.validate_clock_consumer_table_v2(table)


def test_result_tamper_rejected_even_with_recomputed_table_digest():
    gate = api()
    table = gate.probe_clock_consumers_v2()
    failed = next(row for row in table["rows"] if not row["passed"])
    failed.update(passed=True, error=None, blocked_reason=None)
    table["table_sha256"] = gate.clock_table_sha256_v2(table)
    with pytest.raises(ValueError, match="clock_consumer_results"):
        gate.validate_clock_consumer_table_v2(table)


def test_table_digest_tamper_rejected():
    gate = api()
    table = gate.probe_clock_consumers_v2()
    table["table_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="clock_consumer_table_sha"):
        gate.validate_clock_consumer_table_v2(table)


@pytest.mark.parametrize("path", [None, "", "/nonexistent/dh116-clock-map.json"])
def test_no_source_map_is_rejected(path):
    with pytest.raises(ValueError, match="clock_source_map_required"):
        api().clock_consumers_gate_v2(clock_policy=ClockPolicyV2(9600, MODES[0]),
            source_map_path=path)


def test_claiming_native_verified_in_caller_map_cannot_upgrade(tmp_path):
    gate = api()
    candidate = tmp_path / "clock-source-map.json"
    source = json.loads(SOURCE_MAP.read_text())
    source["native_callback_precision_verified"] = True
    source["timestamp_mode"] = MODES[1]
    candidate.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="clock_source_map_sha"):
        gate.clock_consumers_gate_v2(clock_policy=ClockPolicyV2(9600, MODES[1]),
            source_map_path=candidate)


def test_historical_short_fp32_observation_does_not_authorize_long_native():
    gate = api()
    result = gate.clock_consumers_gate_v2(clock_policy=ClockPolicyV2(9600, MODES[0]),
        source_map_path=SOURCE_MAP)
    assert result["clock_consumers_ready"] is False
    assert result["native_ready"] is False
    assert result["native_steps"] == 0
    assert result["actual_sdk_mode_verified"] is False
    assert result["historical_observed_mode"] == MODES[0]
    assert result["historical_sdk_step_max"] == 234
    assert "actual_sdk_long_clock_mode_unverified" in result["blocked_reasons"]
    for name in gate.CONSUMERS:
        if name.endswith("bimanual_step._clock"):
            assert PREFIX + name not in result["blocked_reasons"]
        else:
            assert PREFIX + name in result["blocked_reasons"]


def test_successful_v2_math_reset_and_caller_true_cannot_upgrade_gate():
    from go2_pvcnn.control.dh116.tabletop_native_training_clock import validate_sdk_clock
    gate = api()
    policy = ClockPolicyV2(9600, MODES[0])
    reset_clock = validate_sdk_clock(step=2402,
        timestamp_s=gate.accumulated_timestamp_v2(2402, MODES[0]), dt_s=1/480, policy=policy)
    assert reset_clock.sdk_step_abs == 2402
    with pytest.raises(TypeError):
        gate.clock_consumers_gate_v2(clock_policy=policy, source_map_path=SOURCE_MAP,
            clock_consumers_ready=True, native_ready=True)
    result = gate.clock_consumers_gate_v2(clock_policy=policy, source_map_path=SOURCE_MAP)
    assert result["native_ready"] is False


def test_float64_cpu_hypothesis_all_pass_does_not_certify_current_sdk():
    result = api().clock_consumers_gate_v2(clock_policy=ClockPolicyV2(9600, MODES[1]),
        source_map_path=SOURCE_MAP)
    assert all(row["passed"] for row in result["compatibility_table"]["rows"]
               if row["timestamp_mode"] == MODES[1])
    assert result["clock_consumers_ready"] is False and result["native_ready"] is False
    assert "clock_policy_mode_differs_from_historical_sdk_observation" in result["blocked_reasons"]


def test_current_source_mutation_is_rejected_before_any_reader_probe(monkeypatch):
    gate = api()
    read_bytes = Path.read_bytes
    def altered(path):
        data = read_bytes(path)
        return data + b"# altered" if path.name == "tabletop_deformable_live_geometry.py" else data
    monkeypatch.setattr(Path, "read_bytes", altered)
    with pytest.raises(ValueError, match="clock_consumer_source"):
        gate.probe_clock_consumers_v2()



def test_reviewed_source_map_default_is_explicit():
    assert api().DEFAULT_SOURCE_MAP_PATH == SOURCE_MAP
