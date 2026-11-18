"""Opt-in observer-resource delivery comparison; never a control measurement.

Install/close outside stepping and onUpdate, as required by the audited SDK.
This owner creates exactly one observer: a physics post-step poll OR a full
report callback. It never steps, changes settings, creates reporters or a World.
The existing synchronous provenance poll remains an independent consumer.

All global raw vectors use ContactPairProvenanceCollector's existing copier.
Callback clocks and an active advance bracket are arrival receipts, NOT frame
identifiers or an environment join. SDK counters can update in a different
subscription order. No report from this module grants Native/force/planning/
support/expert qualification. Full callbacks do not supply an actual dt.

The audited SDK demos release subscriptions by dropping their holder reference;
no guessed unsubscribe API is called. close also disables the weak owner
callback, so externally retained callbacks cannot append after closure.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import threading
import time
import weakref

from .tabletop_bimanual_mpc_state import SampleClock
from .tabletop_bimanual_mpc_contact_provenance import (
    ContactPairProvenanceCollector, ContactReport, INSTALLED_BINDING, SourceBinding, StepBracket,
    _bracket, _positive_dt, _plain,
)


@dataclass(frozen=True, slots=True)
class DeliveryCallbackReceipt:
    sequence: int
    mode: str
    active_bracket_sequence: int | None
    active_before_clock_candidate: SampleClock | None
    subscription_api: str
    subscription_order: int | None
    subscription_binding: SourceBinding
    subscription_binding_lines: tuple[int, int]
    arrival_clock_before: SampleClock | None
    arrival_clock_after: SampleClock | None
    actual_callback_dt_s: object
    arrival_monotonic_ns: int
    thread_id: int
    thread_name: str
    attached_stage_before: object
    attached_stage_after: object
    report: ContactReport | None
    errors: tuple[str, ...]
    clock_association: None = field(default=None, init=False)
    pair_env_association: None = field(default=None, init=False)
    native_attested: bool = field(default=False, init=False)
    planning_eligible: bool = field(default=False, init=False)
    native_force_qualified: bool = field(default=False, init=False)
    support_qualified: bool = field(default=False, init=False)
    expert_qualified: bool = field(default=False, init=False)
    receipt_authority: str = field(default='arrival_window_candidate_only', init=False)
    subscription_release_method: str = field(default='audited_SDK_demo_holder_reference_drop', init=False)
    subscription_release_source_sha256: str = field(default=(
        '091262b11a9ff1c35ab92aed6092e662836fbac29d207da0b93f5256a4ece7fe'), init=False)
    subscription_release_source: str = field(default=(
        '/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/'
        'extsPhysics/omni.physx.demos/omni/physxdemos/scenes/ContactReportDemo.py:60-64'), init=False)


@dataclass(frozen=True, slots=True)
class DeliveryStepDiagnostic:
    sequence: int
    mode: str
    bracket: StepBracket | None
    callback_receipts: tuple[DeliveryCallbackReceipt, ...]
    poll_report: ContactReport | None
    attached_stage_before: object
    attached_stage_after: object
    errors: tuple[str, ...]
    clock_association: None = field(default=None, init=False)
    pair_env_association: None = field(default=None, init=False)
    native_attested: bool = field(default=False, init=False)
    planning_eligible: bool = field(default=False, init=False)
    native_force_qualified: bool = field(default=False, init=False)
    support_qualified: bool = field(default=False, init=False)
    expert_qualified: bool = field(default=False, init=False)

    @property
    def callback_count(self):
        return len(self.callback_receipts)


class ContactDeliveryProbe:
    """DI source may expose both subscription APIs; only the selected one is used.

    ``physics_event_source`` is needed only when the existing installed contact
    source and PhysX physics-step interface are separate SDK objects. Injected
    CPU sources never become genuine merely by advertising a source-kind flag.
    ``poll_report`` is supplied by the caller's existing collector; end_step
    never calls its getter, mutates that collector, or requalifies its report.
    """
    def __init__(self, source=None, clock_reader=None, mode='post_step_poll', *,
                 enabled=False, order=0, physics_event_source=None, stage_reader=None):
        if type(enabled) is not bool:
            raise ValueError('enabled must be boolean')
        if mode not in ('post_step_poll', 'full_report_callback'):
            raise ValueError('select exactly one delivery mode')
        if type(order) is not int or order < 0:
            raise ValueError('order must be a nonnegative integer')
        self.enabled, self.mode, self.order = enabled, mode, order
        self._lock = threading.RLock()
        self._source = self._clock_reader = self._stage_reader = None
        self._subscription = self._callback = None
        self._closed = False
        self._pending = None
        self._records, self._steps = [], []
        self._cleanup_errors = []
        self._installation_errors = []
        self._installation_stage = None
        if not enabled:
            return
        if source is None or not callable(clock_reader):
            raise ValueError('explicit source and clock_reader required')
        self._source, self._clock_reader = source, clock_reader
        self._stage_reader = stage_reader or getattr(source, 'get_attached_stage', None)
        reference = weakref.ref(self)
        def callback(*args):
            owner = reference()
            if owner is not None:
                return owner._receive(*args)
        self._callback = callback
        try:
            if mode == 'post_step_poll':
                events = source if physics_event_source is None else physics_event_source
                self._subscription = events.subscribe_physics_on_step_events(
                    callback, pre_step=False, order=order)
            else:
                self._subscription = source.subscribe_full_contact_report_events(callback)
            if self._subscription is None:
                raise RuntimeError('subscription holder was not returned')
            self._installation_stage = self._read_stage(self._installation_errors)
        except BaseException:
            self.close()
            raise

    @property
    def records(self):
        with self._lock:
            return tuple(self._records)

    @property
    def step_diagnostics(self):
        with self._lock:
            return tuple(self._steps)

    @property
    def subscription_active(self):
        with self._lock:
            return self._subscription is not None and not self._closed

    @property
    def cleanup_errors(self):
        return tuple(self._cleanup_errors)

    @property
    def installation_errors(self):
        return tuple(self._installation_errors)

    @property
    def native_attested(self):
        return False

    def _read_clock(self, errors, *, reader=None):
        try:
            result = (self._clock_reader if reader is None else reader)()
            if not isinstance(result, SampleClock):
                raise ValueError('SampleClock required')
            return result
        except Exception as exc:
            errors.append(f'callback_clock_unavailable:{type(exc).__name__}: {exc}')
            return None

    def _read_stage(self, errors, *, reader=None):
        try:
            reader = self._stage_reader if reader is None else reader
            if not callable(reader):
                raise ValueError('actual attached-stage reader unavailable')
            return reader()
        except Exception as exc:
            errors.append(f'attached_stage_unavailable:{type(exc).__name__}: {exc}')
            return None

    def _receive(self, *args):
        # SDK vectors expire after one step. Poll/copy at callback entry BEFORE
        # waiting on the owner bookkeeping lock, retaining only owned DTO data
        # across that wait. A per-callback collector avoids sharing its mutable
        # record list with another callback or with close().
        arrived = time.monotonic_ns()
        pending = self._pending
        if not self.enabled or self._closed:
            return None
        source, clock_reader, stage_reader = self._source, self._clock_reader, self._stage_reader
        mode, order = self.mode, self.order
        errors = []
        before = self._read_clock(errors, reader=clock_reader)
        stage_before = self._read_stage(errors, reader=stage_reader)
        raw, actual_dt = args, None
        if mode == 'post_step_poll':
            actual_dt = _plain(args[0]) if len(args) == 1 else None
            try:
                _positive_dt(actual_dt)
            except ValueError:
                errors.append('callback_dt_invalid')
            else:
                if pending and not math.isclose(actual_dt, pending[2], rel_tol=1e-7, abs_tol=1e-10):
                    errors.append('callback_dt_mismatch')
            try:
                raw = source.get_full_contact_report()
            except Exception as exc:
                raw = None
                errors.append(f'callback_poll_unavailable:{type(exc).__name__}: {exc}')
        # Reuse the existing callback/raw schema/ownership copier in isolation.
        try:
            collector = ContactPairProvenanceCollector(enabled=True, source=source)
            report = collector._capture(raw, 'callback', None, None, None,
                                        ['callback_has_no_frame_identifier'])
        except Exception as exc:
            report = None
            errors.append(f'callback_copy_unavailable:{type(exc).__name__}: {exc}')
        after = self._read_clock(errors, reader=clock_reader)
        stage_after = self._read_stage(errors, reader=stage_reader)
        # Do not retain borrowed vectors/getter results while awaiting commit.
        raw = args = collector = source = clock_reader = stage_reader = None
        if pending is not self._pending:
            errors.append('arrival_window_changed_during_copy')
        if pending is None:
            errors.append('callback_without_active_bracket')
        else:
            if threading.get_ident() != pending[5]:
                errors.append('callback_on_different_thread')
            if any(c is not None and not _same_identity(c, pending[1]) for c in (before, after)):
                errors.append('callback_identity_differs_from_active_bracket')
        if before != after:
            errors.append('callback_clock_changed_during_copy')
        if stage_before != stage_after:
            errors.append('attached_stage_changed_during_copy')
        if report is not None and any(h.stage_id != stage_before or h.stage_id != stage_after
                                      for h in report.headers or ()):
            errors.append('header_stage_differs_from_attached_stage')
        with self._lock:
            if not self.enabled or self._closed:
                return None
            if pending is not self._pending:
                errors.append('arrival_window_closed_before_commit')
            sequence = len(self._records)
            if report is not None:
                report = replace(report, sequence=sequence)
            receipt = DeliveryCallbackReceipt(
                sequence, mode,
                pending[0] if pending else None, pending[1] if pending else None,
                ('PhysX.subscribe_physics_on_step_events' if mode == 'post_step_poll'
                 else 'IPhysxSimulation.subscribe_full_contact_report_events'),
                order if mode == 'post_step_poll' else None,
                INSTALLED_BINDING, ((1845, 1859) if mode == 'post_step_poll' else (1392, 1403)),
                before, after, actual_dt, arrived,
                threading.get_ident(), threading.current_thread().name,
                stage_before, stage_after, report, tuple(errors))
            self._records.append(receipt)
            return receipt

    def begin_step(self, before, *, dt_s):
        with self._lock:
            if not self.enabled:
                return None
            if self._closed:
                raise RuntimeError('delivery probe is closed')
            if not isinstance(before, SampleClock):
                raise ValueError('SampleClock receipt required')
            if self._pending is not None:
                raise ValueError('previous delivery bracket is still active')
            dt_s = _positive_dt(dt_s)
            errors = []
            stage = self._read_stage(errors)
            self._pending = (len(self._steps), before, dt_s, len(self._records),
                             stage, threading.get_ident(), tuple(errors))

    def end_step(self, after, *, step_completed=True, poll_report=None):
        with self._lock:
            if not self.enabled:
                return None
            if self._closed:
                raise RuntimeError('delivery probe is closed')
            if not isinstance(after, SampleClock):
                raise ValueError('SampleClock receipt required')
            if type(step_completed) is not bool:
                raise ValueError('step_completed must be boolean')
            if poll_report is not None and not isinstance(poll_report, ContactReport):
                raise ValueError('poll_report must be the existing ContactReport or None')
            pending, self._pending = self._pending, None
            return self._finish_bracket(pending, after, step_completed, poll_report)

    def _finish_bracket(self, pending, after, completed, poll_report, extra_errors=()):
        bracket = _bracket(pending[1], after, pending[2], completed) if pending else None
        arrivals = tuple(r for r in self._records[pending[3]:]
                         if r.active_bracket_sequence == pending[0]) if pending else ()
        errors = list(extra_errors) + (list(pending[6]) if pending else ['end_without_begin_step'])
        stage_after = self._read_stage(errors)
        if bracket is not None:
            errors.extend(bracket.errors)
        if pending and pending[4] != stage_after:
            errors.append('attached_stage_changed_in_advance')
        if poll_report is not None and poll_report.bracket != bracket:
            errors.append('poll_comparison_bracket_mismatch')
        if not arrivals:
            errors.append('no_callback_in_arrival_window')
        elif len(arrivals) > 1:
            errors.append('multiple_callbacks_in_arrival_window')
        for receipt in arrivals:
            errors.extend(f'callback[{receipt.sequence}]:{error}' for error in receipt.errors)
            if bracket and bracket.after:
                for c in (receipt.arrival_clock_before, receipt.arrival_clock_after):
                    if c is not None and any(not getattr(bracket.before, key) <= getattr(c, key)
                                             <= getattr(bracket.after, key) for key in
                                             ('world_step', 'sdk_step', 'world_time_s', 'sdk_time_s')):
                        errors.append(f'callback[{receipt.sequence}]:clock_outside_advance_receipts')
                        break
        result = DeliveryStepDiagnostic(len(self._steps), self.mode, bracket, arrivals,
                                        poll_report, pending[4] if pending else None,
                                        stage_after, tuple(errors))
        self._steps.append(result)
        return result

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending, self._pending = self._pending, None
            try:
                if pending is not None:
                    self._finish_bracket(pending, None, False, None,
                                         ('probe_closed_with_active_bracket',))
            except BaseException as exc:
                self._cleanup_errors.append(f'close_archive_unavailable:{type(exc).__name__}: {exc}')
                raise
            finally:
                self._subscription = None  # Audited SDK holder-reference release.
                self._callback = self._source = self._clock_reader = self._stage_reader = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except BaseException:
            if exc_value is None:
                raise
        return False


def _same_identity(first, second):
    return all(getattr(first, key) == getattr(second, key) for key in
               ('run_uuid', 'env_id', 'episode_uid', 'reset_generation'))


def _open_installed_delivery_probe(*, source, clock_reader, mode='post_step_poll', order=0):
    # Reuse the existing genuine source; no new simulation interface or World.
    # A self-described injected source cannot select this path. The explicit
    # factory still never issues Native attestation, including in CPU fixtures.
    from .tabletop_bimanual_mpc_contact_provenance import _InstalledContactSource
    if type(source) is not _InstalledContactSource:
        raise ValueError('genuine installed contact source required')
    from pathlib import Path
    import hashlib
    import importlib
    if hashlib.sha256(Path(INSTALLED_BINDING.path).read_bytes()).hexdigest() != INSTALLED_BINDING.sha256:
        raise RuntimeError('installed PhysX binding source SHA differs from audited source')
    physx = importlib.import_module('omni.physx')
    if Path(getattr(physx, '__file__', '')).resolve().parent != Path(INSTALLED_BINDING.path).resolve().parent.parent:
        raise RuntimeError('loaded omni.physx module comes from a different installation')
    physics_events = physx.get_physx_interface() if mode == 'post_step_poll' else None
    return ContactDeliveryProbe(source, clock_reader, mode, enabled=True, order=order,
                                physics_event_source=physics_events,
                                stage_reader=source.interface.get_attached_stage)


def open_installed_delivery_probe(*, enabled=False, **options):
    """Disabled default performs no SDK imports or source/clock access."""
    if type(enabled) is not bool:
        raise ValueError('enabled must be boolean')
    return _open_installed_delivery_probe(**options) if enabled else None
