"""Independent, opt-in PhysX full-report archive; no controller integration.

CPU construction is disabled by default and never imports a simulator. The
explicit installed-source factory lazily acquires the existing PhysX simulation
interface; it does not create a World, step, refresh, prepare reporters, change
settings, apply schemas or write targets. Existing reporters/settings must suffice.

Preferred future call site: begin_step at the ORIGINAL TensorWorld.advance
entry, before advance increments its World counter; original advance executes
its single sim.step; end_step in a pre-reset observer after advance returns and
task.consume has supplied terminal status, before any next step or reset. Do not
place begin_step immediately before sim.step: advance has already incremented
the World counter there, yielding delta World=0/SDK=1 and correctly failing the
single-step check. No wrapper is integrated here. Those receipts are caller
evidence, not proof that this module invoked sim.step. The
installed getter documents CURRENT simulation-step data. Its polling association
requires a single consistent step and equal clocks immediately around the getter.
Callbacks document AFTER simulation step but contain no frame/step identifier:
even callbacks arriving inside a bracket are archived with association NA.

All array ranges belong to actual report headers, not configured filter paths.
uint64 actor/collider path IDs are never Tensor body indices. Only the installed
PhysicsSchemaTools decoder may populate decoded paths, with a roundtrip check.
No report is joined to accepted pads, hand/Wire tensor tuples or MPCState here.
Per-env physical-episode clocks are retained as candidate receipts; assigning a
global report pair to an environment still needs an independently checked join.
The first same-episode False-to-True terminal transition with a completed single
physics step and stable clocks around an attempted getter retains its terminal
metadata. Invalid/no-op/missing-clock attempts remain archived without changing
frame/episode history. Bad raw data on a valid terminal step retains metadata
with report association NA. Later terminal/no-op frames cannot replace it. No
archived report, including a terminal frame, is eligible for planning here.

Binding text does not declare impulse side, frame or SI units. Preserve raw
solver impulses, including signed components; impulse/dt is explicitly derived
and never a qualified ON_HAND/ON_OBJECT force. No upstream capacity/overflow
getter is documented here. Empty events do not prove absence of contact. CPU
fixtures and even an installed-source read cannot grant Native/contact/support/
expert/success qualification. This module has not been run in Native.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real

from .tabletop_bimanual_mpc_state import SampleClock


@dataclass(frozen=True, slots=True)
class SourceBinding:
    path: str
    sha256: str
    poll_lines: tuple[int, int]
    callback_lines: tuple[int, int]
    header_lines: tuple[int, int] = (555, 641)
    point_lines: tuple[int, int] = (424, 483)
    anchor_lines: tuple[int, int] = (791, 808)
    enum_lines: tuple[int, int] = (714, 749)


INSTALLED_BINDING = SourceBinding(
    '/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/'
    'extsPhysics/omni.physx/omni/physx/bindings/_physx.pyi',
    'b75b94176c7e6b766bf6f36058d2115016eaabc0d45d8b36a28c64cbf29f4dd9',
    (1292, 1304), (1392, 1403))


@dataclass(frozen=True, slots=True)
class DecodedPath:
    field_name: str
    path_id: int
    path: str
    stage_id: int
    method: str = 'pxr.PhysicsSchemaTools.intToSdfPath+sdfPathToInt_roundtrip'
    composed_prim_exists: bool | None = None


@dataclass(frozen=True, slots=True)
class ContactEventType:
    raw_repr: str
    name: object
    value: object


@dataclass(frozen=True, slots=True)
class ContactHeader:
    actor0: object
    actor1: object
    collider0: object
    collider1: object
    stage_id: object
    type: object
    proto_index0: object
    proto_index1: object
    contact_data_offset: object
    num_contact_data: object
    friction_anchors_offset: object
    num_friction_anchors_data: object
    decoded_paths: tuple[DecodedPath, ...] = ()


@dataclass(frozen=True, slots=True)
class ContactPoint:
    position: object
    normal: object
    impulse: object
    separation: object
    face_index0: object
    face_index1: object
    material0: object
    material1: object


@dataclass(frozen=True, slots=True)
class FrictionAnchor:
    position: object
    impulse: object


@dataclass(frozen=True, slots=True)
class StepBracket:
    before: SampleClock
    after: SampleClock | None
    dt_s: float
    consistent: bool
    errors: tuple[str, ...]
    receipt_authority: str = 'caller_clock_receipts_not_sim_step_attestation'


@dataclass(frozen=True, slots=True)
class CapacityEvidence:
    header_observed: int | None
    contact_observed: int | None
    anchor_observed: int | None
    header_limit: int | None
    contact_limit: int | None
    anchor_limit: int | None
    at_or_above_limit: bool
    upstream_capacity: int | None = None
    upstream_overflow: bool | None = None
    limit_semantics: str = 'local_capture_budget_only_not_PhysX_capacity'


@dataclass(frozen=True, slots=True)
class ContactReport:
    sequence: int
    origin: str
    source_kind: str
    source_binding: SourceBinding
    runtime_arity: int | None
    headers: tuple[ContactHeader, ...] | None
    points: tuple[ContactPoint, ...] | None
    friction_anchors: tuple[FrictionAnchor, ...] | None
    schema_valid: bool
    ownership_valid: bool
    finite_valid: bool
    raw_valid: bool
    unowned_contact_indices: tuple[int, ...]
    unowned_anchor_indices: tuple[int, ...]
    errors: tuple[str, ...]
    capacity: CapacityEvidence
    bracket: StepBracket | None
    read_before: SampleClock | None
    read_after: SampleClock | None
    clock_association: SampleClock | None
    receipt_clock_candidate: SampleClock | None
    duplicate_candidate: bool = False
    id_namespace: str = field(default='PhysX_uint64_SdfPath_id_not_Tensor_body_index', init=False)
    impulse_unit: str = field(default='SDK_raw_contact_impulse; physical_unit_unspecified_in_binding', init=False)
    position_unit: str = field(default='SDK_raw_contact_position; physical_unit_unspecified_in_binding', init=False)
    frame: str = field(default='NA', init=False)
    force_side: str = field(default='NA', init=False)
    pair_env_association: None = field(default=None, init=False)
    partner_body_id: None = field(default=None, init=False)
    native_attested: bool = field(default=False, init=False)
    measurement_valid: bool = field(default=False, init=False)
    planning_eligible: bool = field(default=False, init=False)
    contact_qualified: bool = field(default=False, init=False)
    support_qualified: bool = field(default=False, init=False)
    expert_qualified: bool = field(default=False, init=False)
    success_qualified: bool = field(default=False, init=False)
    no_contact_attested: bool = field(default=False, init=False)
    upstream_overflow_absence_attested: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class DerivedImpulseRate:
    report_sequence: int
    dt_s: float
    contact_rates: tuple | None
    friction_rates: tuple | None
    kind: str = field(default='derived', init=False)
    unit: str = field(default='SDK_raw_contact_impulse/s; not_qualified_SI_force', init=False)
    force_side: str = field(default='NA', init=False)
    frame: str = field(default='NA', init=False)
    measurement_valid: bool = field(default=False, init=False)


def _finite_scalar(value):
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


def _positive_dt(value):
    if not _finite_scalar(value) or value <= 0:
        raise ValueError('dt_s must be finite and positive')
    return float(value)


def _uint(value, bits=64):
    return not isinstance(value, bool) and isinstance(value, Integral) and 0 <= value < 2**bits


def _plain(value):
    """Detach raw scalars/Float3s, preserving invalid numeric values for audit."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(_plain(v) for v in value)
    # carb.Float3 is iterable; pybind enum is not. Neither object is retained.
    try:
        return tuple(_plain(v) for v in value)
    except TypeError:
        return str(value)


def _read_fields(obj, names, errors, prefix):
    result = {}
    for name in names:
        try:
            raw = getattr(obj, name)
            if name == 'type':
                result[name] = ContactEventType(str(raw),
                    _plain(raw) if isinstance(raw, str) else _plain(getattr(raw, 'name', None)),
                    _plain(getattr(raw, 'value', None)))
            else:
                result[name] = _plain(raw)
        except Exception as exc:
            result[name] = None
            errors.append(f'{prefix}.{name}_unavailable:{type(exc).__name__}')
    return result


def _vector_finite(value):
    return isinstance(value, tuple) and len(value) == 3 and all(_finite_scalar(v) for v in value)


def _bracket(before, after, dt_s, completed):
    errors = []
    if after is None:
        errors.append('step_unfinished')
    else:
        if any(getattr(before, key) != getattr(after, key) for key in
               ('run_uuid', 'env_id', 'episode_uid', 'reset_generation')):
            errors.append('step_identity_changed')
        if before.terminal or before.reset or after.reset:
            errors.append('terminal_or_reset_boundary')
        if after.world_step != before.world_step+1 or after.sdk_step != before.sdk_step+1:
            errors.append('requires_exactly_one_world_and_sdk_physics_step')
        for key in ('world_time_s', 'sdk_time_s'):
            if not math.isclose(getattr(after, key)-getattr(before, key), dt_s, rel_tol=1e-7, abs_tol=1e-10):
                errors.append(f'{key}_delta_mismatch')
        if after.policy_step < before.policy_step or after.policy_step > before.policy_step+1:
            errors.append('policy_step_discontinuity')
    if not completed:
        errors.append('step_not_completed')
    return StepBracket(before, after, dt_s, not errors, tuple(errors))


def derive_impulse_rate(report, *, dt_s):
    """A signed raw impulse/time calculation, explicitly not a qualified force."""
    dt_s = _positive_dt(dt_s)
    if not isinstance(report, ContactReport):
        raise ValueError('ContactReport required')
    def rates(rows):
        if rows is None:
            return None
        return tuple(tuple(v/dt_s for v in row.impulse) if _vector_finite(row.impulse) else None
                     for row in rows)
    return DerivedImpulseRate(report.sequence, dt_s, rates(report.points), rates(report.friction_anchors))


_INSTALLED_SOURCE_KEY = object()


class _InstalledContactSource:
    """Created only by the explicit factory; injection is always CPU-unqualified."""
    def __init__(self, key, interface, schema_tools, stage):
        if key is not _INSTALLED_SOURCE_KEY:
            raise ValueError('use open_installed_contact_source')
        self.interface, self.schema_tools, self.stage = interface, schema_tools, stage

    def get_full_contact_report(self):
        return self.interface.get_full_contact_report()

    def subscribe_full_contact_report_events(self, callback):
        return self.interface.subscribe_full_contact_report_events(callback)

    def decode_header(self, values):
        if self.interface.get_attached_stage() != values['stage_id']:
            raise ValueError('contact stage differs from actually attached stage')
        decoded = []
        for name in ('actor0', 'actor1', 'collider0', 'collider1'):
            path_id = values[name]
            if not _uint(path_id):
                raise ValueError('invalid uint64 path id')
            path = self.schema_tools.intToSdfPath(path_id)
            if (not path.IsAbsolutePath() or not str(path)
                    or self.schema_tools.sdfPathToInt(path) != path_id):
                raise ValueError('PhysicsSchemaTools path roundtrip failed')
            exists = bool(self.stage.GetPrimAtPath(path)) if self.stage is not None else None
            decoded.append(DecodedPath(name, path_id, str(path), values['stage_id'],
                                       composed_prim_exists=exists))
        return tuple(decoded)


def _open_installed_source(stage):
    # Explicit opt-in only. Hash installed text BEFORE acquiring the interface.
    from pathlib import Path
    import hashlib
    import importlib
    if hashlib.sha256(Path(INSTALLED_BINDING.path).read_bytes()).hexdigest() != INSTALLED_BINDING.sha256:
        raise RuntimeError('installed PhysX binding source SHA differs from audited source')
    physx = importlib.import_module('omni.physx')
    if Path(getattr(physx, '__file__', '')).resolve().parent != Path(INSTALLED_BINDING.path).resolve().parent.parent:
        raise RuntimeError('loaded omni.physx module comes from a different installation')
    schema_tools = importlib.import_module('pxr.PhysicsSchemaTools')
    interface = physx.get_physx_simulation_interface()
    return _InstalledContactSource(_INSTALLED_SOURCE_KEY, interface, schema_tools, stage)


def open_installed_contact_source(*, enabled=False, stage=None):
    """No import or source acquisition unless explicitly enabled. No physics writes."""
    if type(enabled) is not bool:
        raise ValueError('enabled must be boolean')
    return _open_installed_source(stage) if enabled else None


class ContactPairProvenanceCollector:
    """Archive copied global reports and proposed env clocks, without a pair join.

    A local budget censors qualification at equality and never truncates raw data.
    Limits are not claims about the SDK's unknown upstream buffers. No automatic
    subscription or fallback to the ambiguous two-vector get_contact_report API.
    """
    def __init__(self, *, enabled=False, source=None, header_limit=None, contact_limit=None, anchor_limit=None):
        if type(enabled) is not bool:
            raise ValueError('enabled must be boolean')
        for limit in (header_limit, contact_limit, anchor_limit):
            if limit is not None and (not _uint(limit) or limit == 0):
                raise ValueError('capture budgets must be positive integers or None')
        self.enabled = enabled
        self._source = source
        self._limits = (header_limit, contact_limit, anchor_limit)
        self._records, self._aborted = [], []
        self._pending = None
        self._subscription = None
        self._first_terminal = None
        self._first_terminal_by_episode = {}
        self._seen_frames, self._closed_episodes, self._latest_identity = set(), set(), {}

    @property
    def records(self):
        return tuple(self._records)

    @property
    def aborted_steps(self):
        return tuple(self._aborted)

    @property
    def first_terminal_record(self):
        return self._first_terminal

    @property
    def first_terminal_records(self):
        """First terminal poll for each run/env/physical episode/reset generation."""
        return tuple(self._first_terminal_by_episode.values())

    def begin_step(self, before, *, dt_s):
        if not self.enabled:
            return
        if not isinstance(before, SampleClock):
            raise ValueError('SampleClock receipt required')
        if self._pending is not None:
            raise ValueError('previous step bracket must end or be explicitly aborted')
        self._pending = (before, _positive_dt(dt_s))

    def abort_step(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('explicit abort reason required')
        if self._pending is not None:
            self._aborted.append((_bracket(self._pending[0], None, self._pending[1], False), reason))
        self._pending = None

    def subscribe(self):
        if not self.enabled:
            return None
        if self._source is None:
            raise ValueError('explicit source required; no automatic SDK acquisition')
        if self._subscription is None:
            self._subscription = self._source.subscribe_full_contact_report_events(self.receive_callback)
        return self._subscription

    def receive_callback(self, headers, contact_data, friction_anchors=None):
        if not self.enabled:
            return None
        bracket = _bracket(self._pending[0], None, self._pending[1], False) if self._pending else None
        report = (headers, contact_data) if friction_anchors is None else (headers, contact_data, friction_anchors)
        return self._capture(report, 'callback', bracket, None, None,
                             ['callback_has_no_frame_identifier'])

    def end_step(self, after, *, poll=True, read_clock=None, step_completed=True):
        if not self.enabled:
            return None
        if not isinstance(after, SampleClock):
            raise ValueError('SampleClock receipt required')
        if type(step_completed) is not bool or type(poll) is not bool:
            raise ValueError('poll/step_completed must be boolean')
        pending, self._pending = self._pending, None
        bracket = _bracket(pending[0], after, pending[1], step_completed) if pending else None
        lifecycle_errors = []
        identity_key = (after.run_uuid, after.env_id)
        episode_key = (*identity_key, after.episode_uid, after.reset_generation)
        frame_key = (*episode_key, after.world_step, after.sdk_step)
        if frame_key in self._seen_frames:
            lifecycle_errors.append('duplicate_step_receipt')
        if episode_key in self._closed_episodes:
            lifecycle_errors.append('physical_episode_already_terminal')
        latest = self._latest_identity.get(identity_key)
        if latest is not None:
            if after.reset_generation < latest[0]:
                lifecycle_errors.append('stale_reset_generation')
            elif after.reset_generation == latest[0] and after.episode_uid != latest[1]:
                lifecycle_errors.append('physical_episode_changed_without_reset_generation')
        if not poll:
            return None
        errors, report, read_before, read_after = lifecycle_errors, None, None, None
        poll_attempted = False
        if bracket is None:
            errors.append('poll_without_begin_step')
        try:
            if read_clock is not None:
                read_before = read_clock()
            if self._source is None:
                raise ValueError('explicit source required; no automatic SDK acquisition')
            getter = self._source.get_full_contact_report
            if not callable(getter):
                raise ValueError('get_full_contact_report must be callable')
            poll_attempted = True
            report = getter()
        except Exception as exc:
            errors.append(f'poll_unavailable:{type(exc).__name__}: {exc}')
        # Copy before any next step, then close the getter clock bracket even
        # when the attempted getter failed. Raw absence never becomes zero.
        return self._capture(report, 'poll', bracket, read_before, read_after,
                             errors, read_clock=read_clock, expected_after=after,
                             poll_attempted=poll_attempted)

    def _capture(self, report, origin, bracket, read_before, read_after, errors, *,
                 read_clock=None, expected_after=None, poll_attempted=False):
        errors = list(errors)
        source_installed = type(self._source) is _InstalledContactSource
        arity = len(report) if isinstance(report, (tuple, list)) else None
        if arity != 3:
            errors.append('full_report_requires_three_vectors')
        headers = points = anchors = None
        def copy_vector(index, row_type, names):
            if arity is None or arity <= index or report[index] is None:
                return None
            try:
                return tuple(row_type(**_read_fields(row, names, errors, f'{row_type.__name__}[{i}]'))
                             for i, row in enumerate(report[index]))
            except Exception as exc:
                errors.append(f'vector_{index}_unavailable:{type(exc).__name__}')
                return None
        headers = copy_vector(0, ContactHeader, tuple(ContactHeader.__dataclass_fields__)[:-1])
        points = copy_vector(1, ContactPoint, tuple(ContactPoint.__dataclass_fields__))
        anchors = copy_vector(2, FrictionAnchor, tuple(FrictionAnchor.__dataclass_fields__))
        schema_valid = arity == 3 and all(v is not None for v in (headers, points, anchors)) and not any(
            '_unavailable:' in e for e in errors)
        decoded_headers = []
        for i, header in enumerate(headers or ()):
            for name in ('actor0', 'actor1', 'collider0', 'collider1', 'stage_id', 'proto_index0', 'proto_index1'):
                if not _uint(getattr(header, name), 32 if name.startswith('proto') else 64):
                    errors.append(f'header[{i}].{name}_invalid_uint')
            event = header.type
            event_values = {'CONTACT_FOUND': 0, 'CONTACT_LOST': 1, 'CONTACT_PERSIST': 2}
            if (not isinstance(event, ContactEventType) or event.name not in event_values
                    or (event.value is not None and (not _uint(event.value) or event.value != event_values[event.name]))
                    or (source_installed and event.value is None)):
                errors.append(f'header[{i}].event_type_invalid')
            if source_installed:
                from dataclasses import replace
                try:
                    values = {name: getattr(header, name) for name in ('actor0', 'actor1', 'collider0', 'collider1', 'stage_id')}
                    header = replace(header, decoded_paths=self._source.decode_header(values))
                except Exception as exc:
                    errors.append(f'header[{i}].path_decode_unavailable:{type(exc).__name__}: {exc}')
            decoded_headers.append(header)
        headers = tuple(decoded_headers) if headers is not None else None
        ownership_errors, owned_contacts, owned_anchors = [], set(), set()
        for rows, offset_name, count_name, owned, label in (
                (points, 'contact_data_offset', 'num_contact_data', owned_contacts, 'contact'),
                (anchors, 'friction_anchors_offset', 'num_friction_anchors_data', owned_anchors, 'anchor')):
            for i, header in enumerate(headers or ()):
                offset, count = getattr(header, offset_name), getattr(header, count_name)
                if (not _uint(offset) or not _uint(count) or rows is None or offset > len(rows)
                        or count > len(rows)-offset):
                    ownership_errors.append(f'header[{i}].{label}_range_invalid')
                    continue
                indices = set(range(offset, offset+count))
                if indices & owned:
                    ownership_errors.append(f'header[{i}].{label}_range_overlap')
                owned.update(indices)
        unowned_contacts = tuple(i for i in range(len(points or ())) if i not in owned_contacts)
        unowned_anchors = tuple(i for i in range(len(anchors or ())) if i not in owned_anchors)
        if unowned_contacts or unowned_anchors:
            ownership_errors.append('unowned_report_rows')
        errors.extend(ownership_errors)
        ownership_valid = schema_valid and not ownership_errors
        finite_errors = []
        for i in owned_contacts:
            point = points[i]
            if not all(_vector_finite(getattr(point, key)) for key in ('position', 'normal', 'impulse')) or not _finite_scalar(point.separation):
                finite_errors.append(f'point[{i}]_nonfinite_or_bad_shape')
            for key in ('face_index0', 'face_index1', 'material0', 'material1'):
                if not _uint(getattr(point, key), 32 if key.startswith('face') else 64):
                    finite_errors.append(f'point[{i}].{key}_invalid_uint')
        for i in owned_anchors:
            if not all(_vector_finite(getattr(anchors[i], key)) for key in ('position', 'impulse')):
                finite_errors.append(f'anchor[{i}]_nonfinite_or_bad_shape')
        errors.extend(finite_errors)
        finite_valid = schema_valid and not finite_errors
        observed = tuple(len(v) if v is not None else None for v in (headers, points, anchors))
        at_limit = any(limit is not None and n is not None and n >= limit for n, limit in zip(observed, self._limits))
        capacity = CapacityEvidence(*observed, *self._limits, at_limit)
        if at_limit:
            errors.append('capture_budget_reached')
        if read_clock is not None:
            try:
                read_after = read_clock()
            except Exception as exc:
                errors.append(f'poll_clock_unavailable:{type(exc).__name__}: {exc}')
        if origin == 'poll':
            if read_before is None or read_after is None:
                errors.append('poll_read_clock_receipts_missing')
            elif (not isinstance(read_before, SampleClock) or not isinstance(read_after, SampleClock)
                  or read_before != read_after or read_after != expected_after):
                errors.append('poll_clock_changed')
        # raw_valid describes copied schema/ownership/numeric/budget, independently
        # of receipt absence and callback's necessarily unavailable frame join.
        raw_errors = [e for e in errors if e not in ('callback_has_no_frame_identifier', 'poll_read_clock_receipts_missing')]
        raw_valid = schema_valid and ownership_valid and finite_valid and not raw_errors
        lifecycle_valid = not any(e in errors for e in (
            'duplicate_step_receipt', 'physical_episode_already_terminal',
            'stale_reset_generation', 'physical_episode_changed_without_reset_generation'))
        step_receipts_valid = (origin == 'poll' and poll_attempted and bracket is not None
            and bracket.consistent and lifecycle_valid and isinstance(read_before, SampleClock)
            and read_before == read_after == expected_after)
        candidate = expected_after if step_receipts_valid and raw_valid else None
        association = candidate if source_installed else None
        duplicate = bool(self._records and origin == 'callback' and self._records[-1].origin == origin
                         and self._records[-1].headers == headers and self._records[-1].points == points
                         and self._records[-1].friction_anchors == anchors)
        result = ContactReport(len(self._records), origin,
            'installed_physx_unqualified' if source_installed else 'cpu_injected_source_unqualified',
            INSTALLED_BINDING, arity, headers, points, anchors, schema_valid, ownership_valid,
            finite_valid, raw_valid, unowned_contacts, unowned_anchors, tuple(errors), capacity,
            bracket, read_before, read_after, association, candidate, duplicate)
        self._records.append(result)
        # Commit lifecycle history only after the entire physics/getter bracket
        # is checked. An invalid attempt must not consume a future valid frame,
        # advance reset history, close an episode, or occupy its terminal pin.
        if step_receipts_valid:
            after = expected_after
            identity_key = (after.run_uuid, after.env_id)
            episode_key = (*identity_key, after.episode_uid, after.reset_generation)
            self._seen_frames.add((*episode_key, after.world_step, after.sdk_step))
            latest = self._latest_identity.get(identity_key)
            if latest is None or after.reset_generation > latest[0]:
                self._latest_identity[identity_key] = (after.reset_generation, after.episode_uid)
            if after.terminal:
                self._closed_episodes.add(episode_key)
                if self._first_terminal is None:
                    self._first_terminal = result
                self._first_terminal_by_episode.setdefault(episode_key, result)
        return result
