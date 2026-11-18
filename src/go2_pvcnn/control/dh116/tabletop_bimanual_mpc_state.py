"""Immutable, CPU-only Teacher state schema, not a simulator or sensor reader.

All values describe one environment of simulation privileged evidence.  No part
of this module is an Actor observation adapter or a hardware measurement claim.
Pose order is xyz + quaternion wxyz; twists are linear then angular, in world.
Hand/digit axes are left/right and thumb/index/middle.  Scalars use shape ().
Dynamic timestamps use the World simulation clock; the independently read SDK
clock is retained without assuming its origin equals the World clock origin.
Static fields carry configuration provenance, no sampling time or sampling Hz;
their clock identifies the snapshot/episode in which configuration applies.

Missing data must be declared explicitly, including its source and reason.
The schema never queries a backend, generates contact or fills missing zeros.
``contact_basis`` identifies the intended evidence category; the string is not
native attestation.  Actual-contact reductions still require the producer's
raw SDK evidence and runner identity chain.  Whole-hand rejected load is never
an alias for per-digit rejected load.  Upward support is signed object-side N;
imbalance denotes the absolute bilateral difference in N, not a ratio.
Terminal/reset snapshots can be archived; eligibility checks are separate and
are not a solver, action-safety, real-time or native qualification certificate.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import ClassVar


HANDS = ('left', 'right')
DIGITS = ('thumb', 'index', 'middle')
STUDENT_OBSERVATION_DIM = 3844


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')


def _finite(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric data')
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be nonnegative')


def _freeze(value):
    """Detach all accepted containers; never retain a caller-owned tensor/list."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    raise ValueError('field data must be scalars, lists or tuples, not mutable/custom objects')


@dataclass(frozen=True, slots=True)
class SampleClock:
    run_uuid: str
    env_id: int
    episode_uid: str
    world_step: int
    world_time_s: float
    sdk_step: int
    sdk_time_s: float
    policy_step: int
    reset_generation: int
    terminal: bool = False
    reset: bool = False

    def __post_init__(self):
        _text(self.run_uuid, 'run_uuid')
        _text(self.episode_uid, 'episode_uid')
        for name in ('env_id', 'world_step', 'sdk_step', 'policy_step', 'reset_generation'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
            object.__setattr__(self, name, int(value))
        for name in ('world_time_s', 'sdk_time_s'):
            _finite(getattr(self, name), name, nonnegative=True)
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ('terminal', 'reset'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')


@dataclass(frozen=True, slots=True)
class StateField:
    value: object
    source: str
    unit: str
    frame: str
    timestamp_s: float | None
    valid: bool
    env_id: int
    episode_uid: str
    clock: SampleClock
    kind: str = 'measured'
    sample_hz: float | None = None
    missing_reason: str | None = None
    force_side: str = 'NA'
    origin: str = 'not_applicable'
    dependencies: tuple[str, ...] = ()
    valid_mask: object = None
    contact_basis: str = 'not_applicable'

    def __post_init__(self):
        if not isinstance(self.clock, SampleClock):
            raise ValueError('field clock must be SampleClock')
        for name in ('source', 'unit', 'frame', 'episode_uid', 'origin'):
            _text(getattr(self, name), name)
        if self.missing_reason is not None:
            _text(self.missing_reason, 'missing_reason')
        if (isinstance(self.env_id, bool) or not isinstance(self.env_id, Integral)
                or self.env_id != self.clock.env_id or self.episode_uid != self.clock.episode_uid):
            raise ValueError('field environment/episode must match its source clock')
        if type(self.valid) is not bool:
            raise ValueError('valid must be boolean')
        if self.kind not in ('measured', 'derived', 'static', 'unavailable'):
            raise ValueError('unknown provenance kind')
        if self.force_side not in ('NA', 'ON_HAND', 'ON_OBJECT'):
            raise ValueError('force_side must explicitly identify force ownership')
        if self.contact_basis not in ('not_applicable', 'actual_contact', 'geometry_only'):
            raise ValueError('contact_basis must distinguish actual contact from geometry')
        if isinstance(self.dependencies, str) or not isinstance(self.dependencies, (tuple, list)):
            raise ValueError('dependencies must be a sequence of source names')
        dependencies = tuple(self.dependencies)
        for dependency in dependencies:
            _text(dependency, 'dependency')
        object.__setattr__(self, 'dependencies', dependencies)
        object.__setattr__(self, 'value', _freeze(self.value))
        object.__setattr__(self, 'valid_mask', _freeze(self.valid_mask))
        if self.valid != (self.value is not None):
            raise ValueError('missing value must be None and valid=False')
        if not self.valid:
            _text(self.missing_reason, 'missing_reason')
            if self.kind != 'unavailable':
                raise ValueError('missing fields must have kind=unavailable')
        elif self.kind == 'unavailable':
            raise ValueError('unavailable fields cannot have values')
        elif self.missing_reason is not None and self.valid_mask is None:
            raise ValueError('valid unmasked data cannot carry a missing reason')
        if self.kind == 'derived' and not dependencies:
            raise ValueError('derived data requires named dependencies')
        if self.kind == 'static':
            if self.timestamp_s is not None or self.sample_hz is not None:
                raise ValueError('configuration is not a sampled measurement')
        elif self.valid:
            _finite(self.timestamp_s, 'timestamp_s', nonnegative=True)
            if self.timestamp_s > self.clock.world_time_s:
                raise ValueError('measurement timestamp is in the future of its source clock')
        elif self.timestamp_s is not None:
            _finite(self.timestamp_s, 'timestamp_s', nonnegative=True)
        if self.sample_hz is not None:
            _finite(self.sample_hz, 'sample_hz')
            if self.sample_hz <= 0:
                raise ValueError('sample_hz must be positive when known')
            object.__setattr__(self, 'sample_hz', float(self.sample_hz))
        if self.timestamp_s is not None:
            object.__setattr__(self, 'timestamp_s', float(self.timestamp_s))
        object.__setattr__(self, 'env_id', int(self.env_id))


@dataclass(frozen=True, slots=True)
class FieldSpec:
    shape: tuple[int | None, ...]
    unit: str
    frame: str
    kinds: tuple[str, ...] = ('measured', 'derived')
    scalar_type: str = 'real'
    force_side: str = 'NA'
    origin: str = 'not_applicable'
    nonnegative: bool = False
    contact_basis: str = 'not_applicable'


def _schema():
    specs = {}
    for hand in HANDS:
        specs[f'robot.{hand}_q'] = FieldSpec((6,), 'rad', 'joint', ('measured',))
        specs[f'robot.{hand}_qd'] = FieldSpec((6,), 'rad/s', 'joint', ('measured',))
        specs[f'wrist.{hand}_pose'] = FieldSpec((7,), 'm|quaternion_wxyz', 'world')
        specs[f'wrist.{hand}_velocity'] = FieldSpec((6,), 'm/s|rad/s', 'world', origin='palm_origin')
        specs[f'wrist.{hand}_base_pose'] = FieldSpec((7,), 'm|quaternion_wxyz', 'world')
        specs[f'wrist.{hand}_base_velocity'] = FieldSpec((6,), 'm/s|rad/s', 'world', origin='base_origin')
        for digit in DIGITS:
            prefix = f'fingertips.{hand}_{digit}'
            specs[f'{prefix}.position'] = FieldSpec((3,), 'm', 'world')
            specs[f'{prefix}.velocity'] = FieldSpec((3,), 'm/s', 'world', origin='pad_reference_point')
            specs[f'{prefix}.normal'] = FieldSpec((3,), 'dimensionless', 'world', ('derived',))
    specs.update({
        'object.centerline': FieldSpec((None, 3), 'm', 'world'),
        'object.centerline_velocity': FieldSpec((None, 3), 'm/s', 'world', origin='centerline_points'),
        'object.local_tangent': FieldSpec((None, 3), 'dimensionless', 'world', ('derived',)),
        'object.local_curvature': FieldSpec((None,), '1/m', 'world', ('derived',), nonnegative=True),
        'object.com': FieldSpec((3,), 'm', 'world'),
        'object.mass': FieldSpec((), 'kg', 'not_applicable', ('measured', 'static'), nonnegative=True),
        'object.bottom_height': FieldSpec((), 'm', 'world'),
        'object.orientation': FieldSpec((4,), 'quaternion_wxyz', 'world'),
        'contact.contact_mask': FieldSpec((2, 3), 'dimensionless', 'dimensionless', ('measured', 'derived'), 'bool'),
        'contact.geometry_candidate': FieldSpec((2, 3), 'dimensionless', 'dimensionless', ('derived',), 'bool', contact_basis='geometry_only'),
        'contact.contact_position': FieldSpec((2, 3, 3), 'm', 'world'),
        'contact.contact_normal': FieldSpec((2, 3, 3), 'dimensionless', 'world'),
        'contact.accepted_load': FieldSpec((2, 3), 'N', 'world', force_side='ON_HAND', nonnegative=True),
        'contact.rejected_load': FieldSpec((2, 3), 'N', 'world', force_side='ON_HAND', nonnegative=True),
        'contact.rejected_load_hand': FieldSpec((2,), 'N', 'world', force_side='ON_HAND', nonnegative=True),
        'contact.accepted_force_on_hand': FieldSpec((2, 3, 3), 'N', 'world', force_side='ON_HAND'),
        'contact.accepted_force_on_object': FieldSpec((2, 3, 3), 'N', 'world', ('derived',), force_side='ON_OBJECT'),
        'contact.friction_force_hand': FieldSpec((2, 3), 'N', 'world', force_side='ON_HAND'),
        'contact.normal_impulse_derived': FieldSpec((2, 3), 'N*s', 'world', ('derived',), force_side='ON_HAND', nonnegative=True),
        'contact.hand_body_id': FieldSpec((2, 3), 'dimensionless', 'not_applicable', ('measured',), 'int', nonnegative=True),
        'contact.partner_body_id': FieldSpec((2, 3), 'dimensionless', 'not_applicable', ('measured',), 'int', nonnegative=True),
        'contact.slip_speed': FieldSpec((2, 3), 'm/s', 'world', ('derived',), nonnegative=True),
        'contact.contact_age': FieldSpec((2, 3), 's', 'not_applicable', ('derived',), nonnegative=True),
        'support.left_upward_support': FieldSpec((), 'N', 'world', ('derived',), force_side='ON_OBJECT'),
        'support.right_upward_support': FieldSpec((), 'N', 'world', ('derived',), force_side='ON_OBJECT'),
        'support.total_upward_support': FieldSpec((), 'N', 'world', ('derived',), force_side='ON_OBJECT'),
        'support.support_margin': FieldSpec((), 'N', 'world', ('derived',), force_side='ON_OBJECT'),
        'support.imbalance': FieldSpec((), 'N', 'world', ('derived',), force_side='ON_OBJECT', nonnegative=True),
        'support.table_reaction': FieldSpec((3,), 'N', 'world', ('measured',), force_side='ON_OBJECT'),
        'support.end_support_reaction': FieldSpec((3,), 'N', 'world', ('measured',), force_side='ON_OBJECT'),
        'safety.joint_margin': FieldSpec((2, 6), 'rad', 'joint', ('derived',)),
        'safety.tracking_error': FieldSpec((2, 6), 'rad', 'joint', ('derived',)),
        'safety.collision_clearance': FieldSpec((), 'm', 'world', ('derived',)),
        'safety.force_limit': FieldSpec((2, 3), 'N', 'world', ('static',), force_side='ON_HAND', nonnegative=True),
        'safety.table_clearance': FieldSpec((), 'm', 'world', ('derived',)),
        # Explicit capability gaps; do not rebrand actuator command/effort as hardware torque/tactile.
        'robot.actual_motor_torque': FieldSpec((2, 6), 'N*m', 'joint', ()),
        'contact.hardware_tactile': FieldSpec((2, 3), 'N', 'world', (), force_side='ON_HAND'),
    })
    for name, spec in tuple(specs.items()):
        if name.startswith('contact.') and name not in ('contact.geometry_candidate', 'contact.hardware_tactile'):
            specs[name] = replace(spec, contact_basis='actual_contact')
    return MappingProxyType(specs)


FIELD_SPECS = _schema()


def missing_field(name: str, clock: SampleClock, reason: str, source: str) -> StateField:
    """Explicit declaration of a capability gap, never an inferred measured zero."""
    if name not in FIELD_SPECS:
        raise ValueError(f'unknown field: {name}')
    spec = FIELD_SPECS[name]
    return StateField(None, source, spec.unit, spec.frame, None, False,
                      clock.env_id, clock.episode_uid, clock, kind='unavailable',
                      missing_reason=reason, force_side=spec.force_side, origin=spec.origin,
                      contact_basis=spec.contact_basis)


def missing_fields(clock: SampleClock, *, reason: str, source: str) -> dict[str, StateField]:
    """Caller-requested explicit NA declarations; make_mpc_state never calls this."""
    return {name: missing_field(name, clock, reason, source) for name in FIELD_SPECS}


def _validate_data(value, shape, spec, name, mask=None):
    if mask is not None and type(mask) is bool:
        if not mask:
            if value is not None:
                raise ValueError(f'{name}: masked missing elements must be None')
            return
        mask = None
    if shape:
        if not isinstance(value, tuple):
            raise ValueError(f'{name}: expected shape {spec.shape}')
        expected = shape[0]
        if (expected is None and len(value) < 2) or (expected is not None and len(value) != expected):
            raise ValueError(f'{name}: expected shape {spec.shape}')
        if mask is not None and (not isinstance(mask, tuple) or len(mask) != len(value)):
            raise ValueError(f'{name}: validity mask must be a shape prefix')
        for i, item in enumerate(value):
            _validate_data(item, shape[1:], spec, name, mask[i] if mask is not None else None)
        return
    if mask is not None:
        raise ValueError(f'{name}: validity mask exceeds data dimensions')
    if spec.scalar_type == 'bool':
        if type(value) is not bool:
            raise ValueError(f'{name}: requires boolean values')
    elif spec.scalar_type == 'int':
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{name}: requires integer body identities')
        if spec.nonnegative and value < 0:
            raise ValueError(f'{name}: unknown body identity requires None with invalid mask')
    else:
        _finite(value, name, nonnegative=spec.nonnegative)


def _mask_flags(mask):
    if type(mask) is bool:
        return (mask,)
    if not isinstance(mask, tuple) or not mask:
        raise ValueError('valid_mask must contain boolean flags')
    return tuple(flag for item in mask for flag in _mask_flags(item))


def _validate_mask_shape(mask, shape, name):
    if type(mask) is bool:
        return ()
    if not shape or not isinstance(mask, tuple):
        raise ValueError(f'{name}: validity mask must be a shape prefix')
    if (shape[0] is None and len(mask) < 2) or (shape[0] is not None and len(mask) != shape[0]):
        raise ValueError(f'{name}: validity mask does not match field shape')
    child_shapes = {_validate_mask_shape(item, shape[1:], name) for item in mask}
    if len(child_shapes) != 1:
        raise ValueError(f'{name}: validity mask cannot be ragged')
    return (len(mask),) + next(iter(child_shapes))


def _vectors(value):
    if value is None:
        return
    if isinstance(value, tuple) and value and all(isinstance(x, Real) and not isinstance(x, bool) for x in value):
        yield value
    elif isinstance(value, tuple):
        for part in value:
            yield from _vectors(part)


def _validate_field(name, f, clock):
    if not isinstance(f, StateField) or f.clock != clock:
        raise ValueError(f'{name}: field clock must match snapshot World/SDK/policy/reset identity')
    spec = FIELD_SPECS[name]
    if (f.unit, f.frame, f.force_side, f.origin, f.contact_basis) != (
            spec.unit, spec.frame, spec.force_side, spec.origin, spec.contact_basis):
        raise ValueError(f'{name}: unit/frame/force ownership/point origin/contact basis mismatch')
    flags = _mask_flags(f.valid_mask) if f.valid_mask is not None else None
    mask_shape = ()
    if f.valid_mask is not None:
        mask_shape = _validate_mask_shape(f.valid_mask, spec.shape, name)
    if not f.valid:
        if flags is not None and any(flags):
            raise ValueError(f'{name}: missing field cannot have valid elements')
        return
    if f.kind not in spec.kinds:
        raise ValueError(f'{name}: unsupported measured/derived/static provenance')
    if flags is not None:
        if not any(flags):
            raise ValueError(f'{name}: all missing must use value=None/valid=False')
        if not all(flags):
            _text(f.missing_reason, 'partial missing_reason')
    _validate_data(f.value, spec.shape, spec, name, f.valid_mask)
    if name.endswith('_pose') or name == 'object.orientation':
        if flags is not None and not all(flags):
            raise ValueError(f'{name}: pose/quaternion must be complete or wholly unavailable')
        quaternion = f.value[3:] if name.endswith('_pose') else f.value
        if not math.isclose(sum(x*x for x in quaternion), 1.0, abs_tol=1e-5):
            raise ValueError(f'{name}: wxyz quaternion must be unit length')
    if name.endswith('.normal') or name in ('contact.contact_normal', 'object.local_tangent'):
        if flags is not None and not all(flags) and len(mask_shape) == len(spec.shape):
            raise ValueError(f'{name}: unit vectors cannot have partially missing components')
        for vector in _vectors(f.value):
            if not math.isclose(sum(x*x for x in vector), 1.0, abs_tol=1e-5):
                raise ValueError(f'{name}: normal/tangent must be unit length')
    if name == 'object.mass' and f.value <= 0:
        raise ValueError('object.mass must be positive')


@dataclass(frozen=True, slots=True)
class MPCState:
    clock: SampleClock
    fields: Mapping[str, StateField]
    observation_scope: ClassVar[str] = 'simulation_privileged'
    teacher_only: ClassVar[bool] = True

    def __post_init__(self):
        if not isinstance(self.clock, SampleClock) or not isinstance(self.fields, Mapping):
            raise ValueError('snapshot requires a SampleClock and explicit field mapping')
        copied = dict(self.fields)
        missing, unknown = set(FIELD_SPECS) - copied.keys(), copied.keys() - set(FIELD_SPECS)
        if missing or unknown:
            raise ValueError(f'explicit field keys required; missing={sorted(missing)}, unknown={sorted(unknown)}')
        for name, f in copied.items():
            _validate_field(name, f, self.clock)
        centerline_lengths = {len(copied[name].value) for name in (
            'object.centerline', 'object.centerline_velocity', 'object.local_tangent',
            'object.local_curvature') if copied[name].valid}
        if len(centerline_lengths) > 1:
            raise ValueError('object sampled centerline fields must share the same K')
        object.__setattr__(self, 'fields', MappingProxyType(copied))

    def to_dict(self):
        """Detached JSON-compatible archive representation, including terminal data."""
        return dict(clock=asdict(self.clock), fields={name: asdict(f) for name, f in self.fields.items()},
                    observation_scope=self.observation_scope, teacher_only=self.teacher_only)


def make_mpc_state(clock: SampleClock, fields: Mapping[str, StateField]) -> MPCState:
    return MPCState(clock, fields)


def _same_episode(first: SampleClock, second: SampleClock):
    for name in ('run_uuid', 'env_id', 'episode_uid', 'reset_generation'):
        if getattr(first, name) != getattr(second, name):
            raise ValueError(f'cannot cross {name}')


def validate_transition(previous: MPCState, current: MPCState) -> None:
    """Validate a contiguous-identity history edge; first terminal may be the last."""
    a, b = previous.clock, current.clock
    _same_episode(a, b)
    if a.terminal or a.reset or b.reset:
        raise ValueError('cannot cross a reset or continue after first terminal')
    if (b.world_step <= a.world_step or b.sdk_step <= a.sdk_step
            or b.world_time_s <= a.world_time_s or b.sdk_time_s <= a.sdk_time_s
            or b.policy_step < a.policy_step):
        raise ValueError('history clocks must advance without stale or reversed samples')


def validate_planning_state(state: MPCState, current_clock: SampleClock, *,
                            max_age_s: float, required_fields: tuple[str, ...]) -> None:
    """Check only identity, availability and freshness under caller-declared needs.

    No implicit tolerance or list of solver inputs is invented here.  Passing
    this check does not certify contact completeness or a feasible/safe action.
    """
    _finite(max_age_s, 'max_age_s', nonnegative=True)
    a, b = state.clock, current_clock
    _same_episode(a, b)
    if a.terminal or b.terminal or a.reset or b.reset:
        raise ValueError('terminal/reset snapshots are archive-only')
    if (b.world_step < a.world_step or b.sdk_step < a.sdk_step or b.policy_step < a.policy_step
            or not 0 <= b.world_time_s-a.world_time_s <= max_age_s
            or not 0 <= b.sdk_time_s-a.sdk_time_s <= max_age_s):
        raise ValueError('snapshot is stale or ahead of the current World/SDK clocks')
    if ((b.world_step == a.world_step) != (b.world_time_s == a.world_time_s)
            or (b.sdk_step == a.sdk_step) != (b.sdk_time_s == a.sdk_time_s)):
        raise ValueError('World/SDK step progression must agree with its own time progression')
    if isinstance(required_fields, str) or not isinstance(required_fields, (tuple, list)):
        raise ValueError('required_fields must explicitly name solver inputs')
    for name in required_fields:
        if name not in state.fields or not state.fields[name].valid:
            raise ValueError(f'required planning input unavailable: {name}')
        f = state.fields[name]
        if f.valid_mask is not None and not all(_mask_flags(f.valid_mask)):
            raise ValueError(f'required planning input has missing elements: {name}')
    for name, f in state.fields.items():
        if f.valid and f.kind != 'static' and not 0 <= b.world_time_s-f.timestamp_s <= max_age_s:
            raise ValueError(f'{name}: stale measurement timestamp')
