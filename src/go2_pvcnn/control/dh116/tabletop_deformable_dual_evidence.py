"""Durable observations of one bounded dual-hand physical episode."""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .tabletop_deformable_bimanual_step import CLOCK_KEYS, DT, SIDES, _clock
from .tabletop_deformable_r0_replay import decode_owned, encode_owned
from .tabletop_deformable_solver_clock import solver_clock_matches
from .tabletop_teacher_mounted_wrist_source import validate_mounted_observation
from .tabletop_teacher_preaction_readback import PHASE


VERSION = "dh116-dual-physical-evidence-v1"
SETUP_STEPS = 2
CONTROL_SUBSTEPS = 8
BOUNDARY_KEYS = CLOCK_KEYS[:5]


def _clock_row(row):
    return _clock(lambda: [row], (row["env_id"],))[0]


def _sdk_boundary(row, step):
    stamp = row["sdk_time_s"]
    if (type(row["sdk_step"]) is not int or row["sdk_step"] != step
            or type(stamp) not in (int, float)
            or not math.isclose(stamp, step * DT, rel_tol=0, abs_tol=2e-7)):
        raise ValueError("SDK setup boundary differs")


def _rod_boundary(packet, clock, phase):
    snapshot = packet["rod_snapshot"]
    for name in ("rod_tick", "rod_uid", "run_uuid"):
        if packet[name] != clock[name]:
            raise ValueError("rod boundary identity differs")
    expected = dict(env_ids=[clock["env_id"]], episode_uids=[clock["episode_uid"]],
        reset_ids=[clock["reset_id"]], rod_uids=[clock["rod_uid"]],
        physics_step=clock["physics_step"], rod_tick=clock["rod_tick"],
        timestamp_s=clock["timestamp_s"])
    if (packet["boundary"] != phase or any(snapshot.get(k) != v for k, v in expected.items())
            or packet["native_qualified"] is not False or packet["teacher_admission"] is not False):
        raise ValueError("rod snapshot boundary differs")


class _Sequence:
    """Check ordering and physical identities with a single retained boundary."""

    def __init__(self, physics_budget_end, solver_substeps):
        if (type(physics_budget_end) is not int or physics_budget_end < SETUP_STEPS + CONTROL_SUBSTEPS
                or type(solver_substeps) is not int or not 1 <= solver_substeps <= 64):
            raise ValueError("explicit bounded SDK and solver budgets required")
        self.budget = physics_budget_end
        self.solver_substeps = solver_substeps
        self.setup = 0
        self.clock = None
        self.geometry = set()
        self.physical = 0
        self.controls = 0
        self.windows = 0
        self.action = None
        self.substep = 0
        self.terminal = None
        self.rows = 0

    def accept(self, operation, row):
        if operation == "setup":
            phases = ("before_reset", "after_reset")
            if self.setup == len(phases) or row["phase"] != phases[self.setup]:
                raise ValueError("out-of-order SDK setup boundary")
            _sdk_boundary(row, (0, SETUP_STEPS)[self.setup])
            self.setup += 1
        elif operation == "reset":
            if self.setup != 2 or self.clock is not None or row["reset"]["status"] != "complete":
                raise ValueError("one completed reset after SDK setup required")
            clock = _clock_row(dict(row["reset"]["after_clock"], policy_step=0, substep=0))
            if clock["physics_step"] != SETUP_STEPS or clock["rod_tick"] != 0 or clock["reset_id"] < 1:
                raise ValueError("reset clock differs")
            _rod_boundary(row["rod_boundary"], clock, "reset")
            self.clock = clock
        elif operation == "initial_geometry":
            side = row["side"]
            if self.clock is None or side not in SIDES or side in self.geometry or self.action is not None:
                raise ValueError("one initial geometry record per hand after reset required")
            if row["evidence"]["side"] != side:
                raise ValueError("initial geometry side differs")
            _sdk_boundary(row["evidence"]["identity"], self.clock["physics_step"])
            self.geometry.add(side)
        elif operation == "begin":
            self._begin(row)
        elif operation == "physical":
            self._physical(row)
        elif operation == "end":
            self._end(row)
        else:
            raise ValueError("unknown dual-hand evidence operation")
        self.rows += 1

    def _begin(self, row):
        if self.terminal is not None:
            raise ValueError("episode already terminal")
        if self.geometry != set(SIDES) or self.action is not None:
            raise ValueError("both initial geometries and a settled control boundary required")
        if self.clock["physics_step"] + CONTROL_SUBSTEPS > self.budget:
            raise ValueError("complete control exceeds physical budget")
        pre = row["pre_action_state"]
        common = pre["common"]
        validate_mounted_observation(common)
        identity = common["identity"]
        expected = dict(self.clock, policy_step=self.controls, substep=0)
        if identity["phase"] != PHASE or _clock_row(identity) != expected:
            raise ValueError("common state not recorded at the current pre-action clock")
        _rod_boundary(pre["rod_boundary"], expected, "reset" if self.controls == 0 else "post_physics")
        action = np.asarray(row["action"])
        if (action.shape != (1, 24) or action.dtype.kind != "f"
                or not np.isfinite(action).all() or np.any(abs(action) > 1)):
            raise ValueError("one finite normalized 24D decision required")
        self.action = action.copy()
        self.substep = 0

    def _physical(self, row):
        if self.action is None or self.terminal is not None or self.substep >= CONTROL_SUBSTEPS:
            raise ValueError("physical row after terminal or outside the active control window")
        exchange = row["exchange"]
        trace = exchange["action_readback"]
        pre, post = [trace[phase]["identity"] for phase in ("pre", "post")]
        if len(pre) != 1 or len(post) != 1:
            raise ValueError("exactly one physical environment required")
        pre, post = _clock_row(pre[0]), _clock_row(post[0])
        expected = dict(self.clock, policy_step=self.controls, substep=self.substep)
        if (pre != expected or any(pre[k] != post[k] for k in (*BOUNDARY_KEYS, "policy_step", "substep"))
                or post["physics_step"] != pre["physics_step"] + 1
                or post["rod_tick"] != pre["rod_tick"] + self.solver_substeps
                or not solver_clock_matches(pre["solver_time_s"], post["solver_time_s"], steps=1)
                or exchange["policy_step"] != self.controls or exchange["substep"] != self.substep
                or exchange["physics_step"] != post["physics_step"]
                or any(exchange["rod"].get(k) != v for k, v in post.items()
                       if k not in ("policy_step", "substep"))):
            raise ValueError("physical SDK/solver/action clock differs")
        if not np.array_equal(trace["request"]["action24"], self.action):
            raise ValueError("physical action differs from the saved decision")
        _rod_boundary(row["rod_boundary"], post, "post_physics")
        reason = exchange["guard_failure"]
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise ValueError("explicit physical terminal reason required")
        self.clock = post
        self.physical += 1
        self.substep += 1
        self.terminal = reason

    def _end(self, row):
        if (self.action is None or not self.substep or row["control_step"] != self.controls
                or row["physical_samples"] != self.substep or row["termination_reason"] != self.terminal
                or row["teacher_admission"] is not False or row["training_reward"] is not None
                or (self.terminal is None and self.substep != CONTROL_SUBSTEPS)):
            raise ValueError("control settlement differs from the persisted physical rows")
        self.windows += self.substep == CONTROL_SUBSTEPS
        self.controls += 1
        self.action = None

    def summary(self):
        return dict(version=VERSION, sdk_setup_boundaries=self.setup, record_rows=self.rows,
            physics_budget_end=self.budget, solver_substeps=self.solver_substeps,
            reset_recorded=self.clock is not None, geometry_sides=sorted(self.geometry),
            measured_physics_rows=self.physical, last_sdk_step=None if self.clock is None else self.clock["physics_step"],
            settled_control_windows=self.controls, complete_control_windows=self.windows,
            unsettled_control_window=self.action is not None, terminal_reason=self.terminal,
            native_qualification=False, teacher_admission=False, optimizer_updates=0)


class DualHandEvidenceJournal:
    """Persist owner callbacks immediately and fsync every completed boundary."""

    def __init__(self, path, *, physics_budget_end, solver_substeps):
        self._sequence = _Sequence(physics_budget_end, solver_substeps)
        self._file = Path(path).open("xb")
        self._closed = False
        try:
            self._write(dict(version=VERSION, physics_budget_end=physics_budget_end,
                             solver_substeps=solver_substeps), durable=True)
        except BaseException:
            self.close()
            raise

    def _write(self, packet, *, durable):
        data = json.dumps(packet, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        self._file.write(data)
        self._file.flush()
        if durable:
            os.fsync(self._file.fileno())

    def _append(self, operation, row):
        if self._closed:
            raise ValueError("closed physical evidence journal")
        try:
            payload = encode_owned(row)
            self._sequence.accept(operation, decode_owned(payload))
            durable = operation != "physical" or self._sequence.terminal is not None
            self._write(dict(operation=operation, payload=payload), durable=durable)
        except BaseException:
            self.close()
            raise

    def record(self, row):
        """Callback for the builder and DualHandPhysicalEpisode."""
        if row["kind"] not in ("setup", "reset", "initial_geometry", "physical"):
            raise ValueError("expected an original physical owner callback")
        self._append(row["kind"], row)

    def begin(self, pre_action_state, action):
        self._append("begin", dict(pre_action_state=pre_action_state, action=action))

    def end(self, control_result):
        self._append("end", control_result)

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            finally:
                self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _nonstandard_constant(value):
    raise ValueError(f"nonstandard JSON token {value}")


def inspect_evidence(path):
    """Stream and independently validate a complete journal or a valid crash prefix."""
    digest = hashlib.sha256()
    sequence = None
    with Path(path).open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise ValueError("incomplete final physical evidence line")
            digest.update(line)
            entry = json.loads(line, parse_constant=_nonstandard_constant)
            try:
                if sequence is None:
                    if (set(entry) != {"version", "physics_budget_end", "solver_substeps"}
                            or entry["version"] != VERSION):
                        raise ValueError("physical evidence header differs")
                    sequence = _Sequence(entry["physics_budget_end"], entry["solver_substeps"])
                else:
                    if set(entry) != {"operation", "payload"}:
                        raise ValueError("physical evidence row schema differs")
                    operation, payload = entry["operation"], decode_owned(entry["payload"])
                    if operation not in ("begin", "end") and payload.get("kind") != operation:
                        raise ValueError("physical owner callback identity differs")
                    sequence.accept(operation, payload)
            except (KeyError, TypeError, IndexError) as error:
                raise ValueError("invalid physical evidence row") from error
    if sequence is None:
        raise ValueError("missing physical evidence header")
    return dict(sequence.summary(), record_sha256=digest.hexdigest())
