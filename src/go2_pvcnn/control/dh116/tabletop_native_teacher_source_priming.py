"""Explicit one-step source priming over the existing supported episode.

Reset/S0 stays at its original clock. Priming is a zero command with its own
physical receipt and budget, never a policy token, reward window or PPO sample.
No App, solver, scene, model or qualification is created by this module.
"""
from copy import deepcopy

import numpy as np
import torch

from .tabletop_action24_history_owner import _require
from .tabletop_native_teacher_action_history import NativeAction24HistoryOwner
from .tabletop_deformable_r0_replay import _copy, _digest
from .tabletop_readonly_control_cost import _physical_source

VERSION = "dh116-primed-action24-history-v3"
SOURCE = "measured_zero_command_source_priming_then_original_policy_history"


class PrimedNativeActionHistory(NativeAction24HistoryOwner):
    """Keep the actual Reset and prime exactly once before the first policy."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._priming = self._priming_sha = self._priming_pre = None
        self._reset_sha = None
        self._row_hashes = []

    def _check(self):
        super()._check()
        if ((self._reset_sha is not None and _digest(self._reset) != self._reset_sha)
                or (self._priming_sha is not None and _digest(self._priming) != self._priming_sha)):
            self._failed = True
            raise ValueError("priming or original Reset receipt changed outside owner")

    def record(self, row):
        if row.get("kind") == "priming_physical":
            return self.record_priming(row)
        result = super().record(row)
        if row.get("kind") == "reset":
            self._reset_sha = _digest(self._reset)
        return result

    def prepare_priming(self, pre):
        self._check()
        _require(self._reset is not None and self._priming is None and self._priming_pre is None
                 and self._decision is self._settled is self._captured is None
                 and self._sequence.controls == 0 and self._sequence.clock["rod_tick"] == 0,
                 "fresh unprimed Reset before any sampled decision required")
        probe = deepcopy(self._sequence)
        probe.accept("begin", dict(pre_action_state=pre, action=self._seed))
        _require(pre["rod_boundary"]["rod_snapshot"]["backend_sha256"] == self._backend,
                 "priming pre-action backend differs")
        self._priming_pre = _copy(pre)

    def record_priming(self, row):
        self._check()
        try:
            _require(row.get("kind") == "priming_physical" and self._priming_pre is not None
                     and self._priming is None and self._decision is None
                     and self._sequence.controls == 0, "one separately prepared priming physical row required")
            exchange = row["exchange"]
            _require(exchange["guard_failure"] is None
                     and row["rod_boundary"]["rod_snapshot"]["backend_sha256"] == self._backend,
                     "priming guard or backend differs")
            request = np.asarray(exchange["action_readback"]["request"]["action24"])
            _require(request.shape == (1, 24) and request.dtype == np.float64
                     and np.array_equal(request, np.zeros((1, 24), dtype=np.float64)),
                     "priming requires the actual exact zero float64 request")
            # Reuse the original dimensional SDK target/readback and clock
            # validators without inventing a pending policy or partial window.
            hands = _physical_source(exchange, self._priming_pre["common"], self._last_hands, True)
            probe = deepcopy(self._sequence)
            probe.accept("begin", dict(pre_action_state=self._priming_pre, action=self._seed))
            probe.accept("physical", row)
            receipt = dict(version=VERSION, source=SOURCE, owner_uid=self._owner_uid,
                reset_identity=deepcopy(self._reset["identity"]), primed_identity=deepcopy(probe.clock),
                priming_row_sha256=_digest(row), zero_action_sha256=_digest(self._seed),
                ppo_transitions=0, native_qualification=False, teacher_admission=False)
            self._sequence.clock = deepcopy(probe.clock)
            self._sequence.physical += 1
            self._sequence.rows += 1
            self._sequence._first_boundary_phase = "post_physics"
            self._last_hands = hands
            self._priming = receipt
            self._priming_sha = _digest(receipt)
            self._priming_pre = None
            return deepcopy(receipt)
        except BaseException:
            self._failed = True
            raise

    @property
    def priming(self):
        self._check()
        _require(self._priming is not None, "source priming required")
        return deepcopy(self._priming)

    def capture_pre_action(self, pre):
        self._check()
        _require(self._priming is not None, "source priming required before first policy")
        packet = super().capture_pre_action(pre)
        packet.update(version=VERSION, source=SOURCE, priming=self.priming)
        self._captured = _copy(packet)
        return _copy(packet)

    def begin(self, pre, trainer):
        super().begin(pre, trainer)
        self._decision.update(version=VERSION, source=SOURCE,
                              priming_row_sha256=self.priming["priming_row_sha256"])
        return _copy(self._decision)


def prime_supported_sources(episode, *, history, runtime, journal):
    """Advance exactly one real SDK/rod boundary; persist and charge before validation."""
    from .tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
    from .tabletop_deformable_physical_exchange import FlexiblePhysicalExchange
    from .tabletop_native_teacher_session import SessionBudget
    from .tabletop_native_teacher_collector import NativeJournal
    from .tabletop_power_shared_rod import PROXIMAL_POWER_MESH_NAMES

    _require(type(episode) is DualHandPhysicalEpisode and type(history) is PrimedNativeActionHistory
             and type(runtime) is SessionBudget and type(journal) is NativeJournal,
             "original physical episode, primed history, budget and journal required")
    _require(episode._state == "active" and episode._control_index == 0
             and isinstance(episode._exchange, FlexiblePhysicalExchange)
             and history._priming is None and not getattr(runtime, "source_priming_failed", False),
             "fresh active supported episode before source priming required")
    rod, sim = episode._rod, episode._sim
    _require(tuple(rod.mesh_names) == PROXIMAL_POWER_MESH_NAMES
             and runtime.scene == (id(sim), id(rod), id(rod.rope))
             and runtime.sdk_steps == sim.current_time_step_index,
             "same owned supported17 scene and charged SDK setup required")
    substeps = rod._base.cfg["rod"]["solver_substeps"]
    _require(substeps == 12 and min(
        episode._end - sim.current_time_step_index,
        runtime.limits["max_sdk_steps"] - runtime.sdk_steps,
        runtime.limits["max_solver_calls"] - runtime.solver_calls) >= 9
        and runtime.remaining_s() > 9/480, "priming plus first complete policy window exceeds budget")
    pre = episode.pre_action_state()
    history.prepare_priming(pre)
    before = deepcopy(rod.clock())
    row = None
    episode._state = "failed"
    try:
        row = episode._exchange._step(torch.zeros(1, 24, dtype=torch.float64), 0)
        runtime.charge(row, substeps)
        journal.emit("priming_exchange", row)
        packet = episode._boundary.capture(row["rod"])
        event = dict(kind="priming_physical", exchange=row, rod_boundary=deepcopy(packet),
                     native_qualification=False, teacher_admission=False)
        # Existing durable recorder runs before the history can reject this row.
        episode._record(event)
        receipt = history.priming
        episode._rod_packet = packet
        journal.emit("source_priming_complete", dict(receipt=receipt,
            budget_sdk_steps=runtime.sdk_steps, budget_solver_calls=runtime.solver_calls,
            budget_internal_solver_substeps=runtime.internal_solver_substeps))
        episode._state = "active"
        return receipt
    except BaseException as error:
        history._failed = True
        runtime.source_priming_failed = True
        if row is None:
            # The original SDK may have advanced before its operation raised.
            # Record observable progress; unknown solver completion never passes.
            after = rod.clock()
            sdk_delta = sim.current_time_step_index - before["physics_step"]
            tick_delta = after["rod_tick"] - before["rod_tick"]
            runtime.sdk_steps += max(0, sdk_delta)
            if tick_delta > 0:
                # A partially observed call still consumes one call budget.
                # Count only reported ticks; completion remains unverified.
                runtime.solver_calls += (tick_delta + substeps - 1) // substeps
                runtime.internal_solver_substeps += tick_delta
            runtime.last_clock = deepcopy(after)
        try:
            journal.emit("source_priming_failed", dict(type=type(error).__name__, message=str(error),
                before_clock=before, actual_sdk_step=sim.current_time_step_index,
                actual_rod_clock=rod.clock(), solver_completion_verified=row is not None,
                replay_allowed=False, native_qualification=False))
        except BaseException:
            pass
        raise
