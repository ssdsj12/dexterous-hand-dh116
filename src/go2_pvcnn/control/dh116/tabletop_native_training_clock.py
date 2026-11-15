"""Bounded clock mathematics for an independent long-Reset v2.

Neither a selected accumulation hypothesis nor a successful CPU validation
proves the installed SDK callback precision, grants a physics budget, or changes
any frozen v1 reader. Actual SDK timestamps and solver clocks are never repaired.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


DT = 1 / 480
VERSION = "dh116-actual-training-clock-v2"
_MODES = ("fp32_dt_float64_sum", "float64_dt_float64_sum")
_ROD_FIELDS = frozenset(("run_uuid", "env_id", "episode_uid", "reset_id",
    "rod_uid", "physics_step", "rod_tick", "timestamp_s", "solver_time_s", "clock_valid"))
_U = float(np.finfo(np.float64).eps) / 2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    _require(valid, "finite nonnegative actual " + label + " required")
    return value


@dataclass(frozen=True, slots=True)
class ClockPolicyV2:
    """Explicit CPU-math range/hypothesis; not a native mode certification."""

    max_sdk_step: int
    timestamp_mode: str

    def __post_init__(self):
        _require(type(self.max_sdk_step) is int and self.max_sdk_step > 0,
            "positive exact integer max_sdk_step required")
        _require(type(self.timestamp_mode) is str and self.timestamp_mode in _MODES,
            "explicit supported timestamp_mode required")

    @property
    def sha256(self):
        self.__post_init__()
        value = dict(version=VERSION, max_sdk_step=self.max_sdk_step,
            timestamp_mode=self.timestamp_mode, dt_s=DT,
            envelope="n_bias_plus_gamma_n_float64_sum_plus_8eps_v1")
        return hashlib.sha256(json.dumps(value, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SdkClockV2:
    """Owned mathematical validation result, not proof of a native producer."""

    sdk_step_abs: int
    timestamp_s: float
    dt_s: float
    tolerance_s: float
    policy_sha256: str


def validate_sdk_clock(*, step, timestamp_s, dt_s, policy):
    """Check a bounded accumulation envelope; preserve the input timestamp."""
    _require(type(policy) is ClockPolicyV2, "exact ClockPolicyV2 required")
    policy.__post_init__()
    _require(type(step) is int and 0 <= step <= policy.max_sdk_step,
        "exact SDK step outside explicit clock policy range")
    _number(dt_s, "SDK dt")
    _require(dt_s == DT, "actual 480Hz SDK dt required")
    _number(timestamp_s, "SDK timestamp")
    # Compare integers before float arithmetic, including deliberately huge ints.
    _require(step < 2**53, "summation envelope requires n*u < 1")
    term = float(np.float32(dt_s)) if policy.timestamp_mode == _MODES[0] else dt_s
    nu = step * _U
    gamma_n = nu / (1 - nu)
    tolerance = (step * abs(term - dt_s) + gamma_n * step * abs(term)
        + 8 * float(np.finfo(np.float64).eps) * max(1., step * dt_s))
    _require(math.isfinite(tolerance) and tolerance < dt_s / 2,
        "summation envelope must be below half a physical interval")
    _require(abs(timestamp_s - step * dt_s) <= tolerance,
        "actual SDK timestamp outside derived accumulation envelope")
    return SdkClockV2(step, timestamp_s, dt_s, tolerance, policy.sha256)


def _validate_rod(row, sdk, policy):
    _require(type(row) is dict and set(row) == _ROD_FIELDS,
        "exact owned actual rod clock fields required")
    _require(type(row["clock_valid"]) is bool, "exact rod clock_valid required")
    for key in ("run_uuid", "episode_uid", "rod_uid"):
        _require(type(row[key]) is str and bool(row[key].strip()),
            "actual rod " + key + " required")
    for key in ("env_id", "reset_id", "rod_tick"):
        _require(type(row[key]) is int and row[key] >= 0,
            "exact nonnegative rod " + key + " required")
    _number(row["solver_time_s"], "solver clock")
    if row["physics_step"] is None or row["timestamp_s"] is None:
        _require(row["physics_step"] is None and row["timestamp_s"] is None
            and row["clock_valid"] is False and row["reset_id"] == 0
            and row["rod_tick"] == 0 and row["solver_time_s"] == 0.,
            "explicit unregistered constructor rod clock required")
    else:
        checked = validate_sdk_clock(step=row["physics_step"],
            timestamp_s=row["timestamp_s"], dt_s=sdk.dt_s, policy=policy)
        _require(checked.sdk_step_abs == sdk.sdk_step_abs,
            "actual rod and SDK absolute step differ")
    # A poisoned registered clock stays False. Never promote it to live/valid.
    return row


def read_clock_v2(sim, rod, *, policy):
    """Own one actual SDK/rod endpoint, without stepping or retaining owners.

    Constructor None/None/False and a poisoned False clock are preserved. The
    original rod timestamp is a separate field: it is not overwritten with the
    potentially accumulated SDK timestamp, and solver time/tick are not inferred.
    """
    try:
        sdk = validate_sdk_clock(step=sim.current_time_step_index,
            timestamp_s=sim.current_time, dt_s=sim.get_physics_dt(), policy=policy)
        row = _validate_rod(deepcopy(rod.clock()), sdk, policy)
        after = validate_sdk_clock(step=sim.current_time_step_index,
            timestamp_s=sim.current_time, dt_s=sim.get_physics_dt(), policy=policy)
        _require(after == sdk, "actual SDK clock changed during rod clock capture")
        return dict(schema=VERSION, sdk_physics_step=sdk.sdk_step_abs,
            sdk_timestamp_s=sdk.timestamp_s, sdk_dt_s=sdk.dt_s,
            sdk_tolerance_s=sdk.tolerance_s, clock_policy_sha256=sdk.policy_sha256,
            rod=row)
    finally:
        sim = rod = None
