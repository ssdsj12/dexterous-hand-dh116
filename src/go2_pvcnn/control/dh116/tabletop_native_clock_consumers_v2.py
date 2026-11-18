"""Source-pinned CPU compatibility probes for the frozen physical clock readers.

The SDK accumulator and the runtime's nominal rod timestamp are different source
fields. Probes preserve that production wiring. A separate bimanual diagnostic
shows what would happen if an actual SDK stamp were substituted; it is never used
to claim that the runtime actually supplies that stamp. No SDK/app is created.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import tabletop_deformable_bimanual_step as bimanual
from . import tabletop_deformable_dual_evidence as evidence
from . import tabletop_deformable_dual_hand_episode as episode
from . import tabletop_deformable_live_geometry as geometry
from .tabletop_native_training_clock import ClockPolicyV2, DT, validate_sdk_clock

VERSION = "dh116-legacy-clock-consumers-v2"
MODES = ("fp32_dt_float64_sum", "float64_dt_float64_sum")
BOUNDARIES = (240, 258, 2402, 9600)
CONSUMERS = (
    "tabletop_deformable_bimanual_step._clock",
    "tabletop_deformable_live_geometry.LiveGeometryReader._clock",
    "tabletop_deformable_dual_evidence._sdk_boundary",
    "tabletop_deformable_dual_hand_episode.DualHandPhysicalEpisode._read_state_clock",
)
CONSUMER_SOURCE_SHA256 = {
    "tabletop_deformable_bimanual_step.py":
        "e144640a0310ab4824e482a2d643165239a74bbfdee66cae42289ff8cb3ee7f2",
    "tabletop_deformable_live_geometry.py":
        "19a318aa643d47a73e7dc49e226d2aa6fb4b5275e072dd2257c426e76ec7bbc9",
    "tabletop_deformable_dual_evidence.py":
        "79d3b9b2881df6ab4345410dab7f49e3f60e66445699d99d752d6b8b16bcbb16",
    "tabletop_deformable_dual_hand_episode.py":
        "4c50d414fb983f9189b73cefd3ec2ffe0a7633c105aecd6cf1d01749845e74f7",
    "tabletop_dual_hand_rod_runtime.py":
        "e568a5b1951d43c278bff9949f3773514b1cc8f13e5d948ee0bd982a8c894688",
}
# This reviewed map records unverified callback precision and short historical
# fp32-term observations. A caller cannot replace it with a self-certified map.
DEFAULT_SOURCE_MAP_PATH = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/"
    "long-clock-v2-20261115/clock-source-map.json")
SOURCE_MAP_SHA256 = "9a927be6a7f225bed95df05fda752d4eed3d8f23339c0677ef82818e5d8a0032"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()


def consumer_source_inventory_v2():
    """Validate current on-disk readers and nominal rod timestamp producer."""
    root = Path(__file__).parent
    actual = {name: _sha((root / name).read_bytes()) for name in CONSUMER_SOURCE_SHA256}
    _require(actual == CONSUMER_SOURCE_SHA256, "clock_consumer_source_sha256 differs")
    return actual


def accumulated_timestamp_v2(step, mode):
    """CPU endpoint from repeated addition, with no normalization to n*dt."""
    policy = ClockPolicyV2(max_sdk_step=max(step, 1), timestamp_mode=mode)
    term = float(np.float32(DT)) if mode == MODES[0] else DT
    stamp = 0.
    for _ in range(step):
        stamp += term
    validate_sdk_clock(step=step, timestamp_s=stamp, dt_s=DT, policy=policy)
    return stamp


def _rod_clock(step):
    # This is the exact declared producer contract in DualHandRodRuntime.clock.
    return dict(run_uuid="cpu-clock-consumer-probe", env_id=0,
        episode_uid="cpu:clock-probe", reset_id=1, rod_uid="cpu-rod",
        physics_step=step, rod_tick=12 * (step - 2), timestamp_s=step * DT,
        solver_time_s=(step - 2) * DT, clock_valid=True)


def _outcome(call):
    try:
        call()
    except ValueError as error:
        return dict(passed=False, error=str(error))
    return dict(passed=True, error=None)


def _probe_endpoint(step, mode):
    stamp = accumulated_timestamp_v2(step, mode)
    sim = SimpleNamespace(current_time_step_index=step, current_time=stamp)
    rod_row = _rod_clock(step)
    rod = SimpleNamespace(clock=lambda: deepcopy(rod_row))
    control_row = dict(rod_row, policy_step=0, substep=0)
    # Invoke the actual reader methods without unrelated scene initialization.
    live = SimpleNamespace(_sim=sim, _read_clock=rod.clock)
    physical = SimpleNamespace(_sim=sim, _rod=rod, _control_index=0)
    calls = (
        lambda: bimanual._clock(lambda: [deepcopy(control_row)], (0,)),
        lambda: geometry.LiveGeometryReader._clock(live),
        lambda: evidence._sdk_boundary(dict(sdk_step=step, sdk_time_s=stamp), step),
        lambda: episode.DualHandPhysicalEpisode._read_state_clock(physical),
    )
    result = []
    for index, (name, call) in enumerate(zip(CONSUMERS, calls)):
        outcome = _outcome(call)
        row = dict(consumer=name, timestamp_mode=mode, sdk_step_abs=step,
            sdk_timestamp_s=stamp, rod_timestamp_s=rod_row["timestamp_s"],
            timestamp_source="rod_counter_projection" if index == 0
                else "actual_sdk_accumulator",
            **outcome,
            blocked_reason=None if outcome["passed"] else
                "long_clock_consumer_incompatible:" + name)
        if index == 0:
            sdk_control = dict(control_row, timestamp_s=stamp)
            row["actual_sdk_stamp_diagnostic"] = _outcome(
                lambda: bimanual._clock(lambda: [sdk_control], (0,)))
        result.append(row)
    return result


def clock_table_sha256_v2(table):
    """Digest all table fields except its own checksum."""
    return _sha(_canonical({key: value for key, value in table.items()
                            if key != "table_sha256"}))


def probe_clock_consumers_v2():
    """Run the four frozen clock readers at both declared CPU-mode endpoints."""
    sources = consumer_source_inventory_v2()
    rows = [row for mode in MODES for step in BOUNDARIES
            for row in _probe_endpoint(step, mode)]
    table = dict(schema=VERSION, source_sha256=sources, rows=rows, native_steps=0,
        scope="cpu_endpoint_reader_probe",
        source_fields_distinct=True,
        actual_sdk_precision_qualified=False)
    table["table_sha256"] = clock_table_sha256_v2(table)
    return table


def validate_clock_consumer_table_v2(table):
    """Reject source/result tampering, even when the caller recomputes a digest."""
    _require(type(table) is dict, "owned clock_consumer_table required")
    sources = consumer_source_inventory_v2()
    _require(table.get("source_sha256") == sources,
        "clock_consumer_source_sha256 differs")
    _require(table.get("table_sha256") == clock_table_sha256_v2(table),
        "clock_consumer_table_sha256 differs")
    _require(table == probe_clock_consumers_v2(), "clock_consumer_results differ")
    return deepcopy(table)


def _read_source_map(path):
    _require(path is not None and str(path).strip(), "clock_source_map_required")
    try:
        raw = Path(path).read_bytes()
    except (OSError, TypeError) as error:
        raise ValueError("clock_source_map_required") from error
    _require(_sha(raw) == SOURCE_MAP_SHA256, "clock_source_map_sha256 differs")
    packet = json.loads(raw)
    _require(packet["schema"] == "dh116-clock-static-source-map-v2-task1",
        "clock_source_map schema differs")
    for source in packet["sources"]:
        _require(_sha(Path(source["path"]).read_bytes()) == source["sha256"],
            "clock_source_map installed source SHA differs")
    historical = packet["historical_raw_observation"]
    for path_key, sha_key in (
            ("sdk_clock_path", "sdk_clock_sha256"),
            ("result_path", "result_sha256"),
            ("numeric_reset_path", "numeric_reset_sha256")):
        _require(_sha(Path(historical[path_key]).read_bytes()) == historical[sha_key],
            "clock_source_map historical source SHA differs")
    return packet


def clock_consumers_gate_v2(*, clock_policy, source_map_path, table=None):
    """Return source-checked readiness; the current short audit stays blocked.

    This gate owns neither a native worker nor native authorization. The frozen
    Task1 map has native_callback_precision_verified=false, so even a completely
    compatible CPU hypothesis cannot certify the installed SDK's long clock.
    """
    _require(type(clock_policy) is ClockPolicyV2, "exact ClockPolicyV2 required")
    clock_policy.__post_init__()
    source_map = _read_source_map(source_map_path)
    table = probe_clock_consumers_v2() if table is None else validate_clock_consumer_table_v2(table)
    historical = source_map["historical_raw_observation"]
    # Derive the observation from the pinned audit errors, not a caller mode flag.
    observed_mode = (MODES[0] if historical["max_sdk_time_error_to_fp32_term_sum_s"] == 0.
        and historical["max_sdk_time_error_to_repeated_float64_dt_s"] > 0. else None)
    blocked = []
    if source_map["native_callback_precision_verified"] is not True:
        blocked.append("actual_sdk_long_clock_mode_unverified")
    if observed_mode is None:
        blocked.append("actual_sdk_clock_mode_unknown")
    elif clock_policy.timestamp_mode != observed_mode:
        blocked.append("clock_policy_mode_differs_from_historical_sdk_observation")
    for row in table["rows"]:
        if row["timestamp_mode"] == observed_mode and not row["passed"]:
            if row["blocked_reason"] not in blocked:
                blocked.append(row["blocked_reason"])
    if clock_policy.max_sdk_step > BOUNDARIES[-1]:
        blocked.append("clock_consumer_probe_range_exceeded")
    ready = not blocked
    return dict(schema="dh116-long-clock-consumer-gate-v2",
        clock_policy_sha256=clock_policy.sha256,
        source_map_sha256=SOURCE_MAP_SHA256,
        source_sha256=deepcopy(table["source_sha256"]),
        table_sha256=table["table_sha256"], compatibility_table=deepcopy(table),
        actual_sdk_mode_verified=source_map["native_callback_precision_verified"],
        historical_observed_mode=observed_mode,
        historical_sdk_step_max=historical["sdk_step_max"],
        clock_consumers_ready=ready, native_ready=False,
        blocked_reasons=blocked, native_steps=0,
        scope="static_cpu_clock_compatibility_only")
