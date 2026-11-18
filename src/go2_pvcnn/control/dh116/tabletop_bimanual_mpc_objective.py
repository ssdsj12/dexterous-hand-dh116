"""Opt-in CPU cost arithmetic, with no controller, dynamics, solver or wiring.

U is eight fixed 60 Hz CONTROL periods by canonical24Tanh coordinates (dimensionless),
not eight physics substeps. No action is produced, mapped or clipped here.
Hands are left/right; pads are thumb/index/middle. Joint axes are explicitly
named by the caller, with variable D; a finger-only subset cannot claim full
mounted-articulation coverage (which may include six additional wrist axes).

Inputs are already declared residuals, not a replacement FK/contact model:
geometry is distance(pad, source-bound target_region) in m; contact is a
nonnegative valid_contact_error cost, restricted by an explicit accepted or
required-hold contact mask; support is signed ON_OBJECT upward N per hand;
slip is actual-contact tangential speed m/s. Bimanual residual is translation
and SO(3) log error of inv(T_left_grasp) @ T_right_grasp versus its target, in
the left grasp frame (m then rad; q and -q must yield the same upstream log).
Joint data are min(q-lower, upper-q) margins in each declared coordinate's SI
unit (m for prismatic, rad for revolute). The barrier is the dimensionless
sum((reference_per_dof / margin_per_dof)**2), with strictly positive margins
and a sourced experimental reference in the SAME unit on each axis. Reference
values only normalize this cost; they are not physical limits. Non-SI backend
stage_distance must be source-qualified/converted upstream; no conversion is
invented here. This cost is not a joint-limit hard constraint. Risk inputs are
explicitly normalized nonnegative collision, overload, tracking residuals;
their source must identify the normalization, not an actuator-budget proxy.

Measured residuals contain ONE current sample. Predicted residuals contain
H=8 future control endpoints and identify the exact candidate digest. This
binding and a candidate_dependent label are declarations, NOT model validation.
Predicted support/slip require an upstream contact predictor; none is supplied
or qualified by this module. Measured support/slip are constant diagnostics.
No evidence here grants Native, prediction, solve or action admissibility.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
from itertools import product
import math
from numbers import Real
import struct
from types import MappingProxyType
from typing import ClassVar

from .tabletop_bimanual_mpc_state import SampleClock


TERM_NAMES = ('geometry', 'contact', 'support', 'slip', 'bimanual', 'joint', 'smooth', 'risk')
HORIZON = 8
ACTION_DIM = 24
CONTROL_PERIOD_S = 1 / 60
_SPECS = {
    'geometry': ((2, 3), 'm', 'world'),
    'contact': ((2, 3), 'dimensionless', 'declared_residual'),
    'support': ((2,), 'N', 'world'),
    'slip': ((2, 3), 'm/s', 'world'),
    'bimanual': ((6,), 'm|rad', 'relative_left_grasp'),
    'joint': ((2, None), 'joint_coordinate_per_dof', 'joint'),
    'collision': ((2,), 'dimensionless', 'declared_residual'),
    'overload': ((2, 3), 'dimensionless', 'declared_residual'),
    'tracking': ((2, None), 'dimensionless', 'joint'),
}
_RAW_UNITS = dict(geometry='m^2', contact='dimensionless', support='N^2',
                  slip='m^2/s^2', bimanual='dimensionless', joint='dimensionless',
                  smooth='dimensionless', risk='dimensionless')


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')


def _finite(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite real data')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be nonnegative')


def _square(value):
    result = value * value
    _finite(result, 'squared cost')
    return result


def _control_period(value):
    _finite(value, 'control period', positive=True)
    if value != CONTROL_PERIOD_S:
        raise ValueError('control period must be fixed 60 Hz (1/60 s)')


def _coordinate_units(value):
    if (not isinstance(value, (tuple, list)) or not value
            or any(unit not in ('m', 'rad') for unit in value)):
        raise ValueError('joint coordinate units must explicitly be m or rad per DOF')
    return tuple(value)


def _sum(values):
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError('summed cost must be finite') from exc
    _finite(result, 'summed cost')
    return result


def _freeze(value):
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if value is None or type(value) is bool:
        return value
    _finite(value, 'array element')
    return float(value)


def _shape(value, shape, name, *, boolean=False, nullable=False):
    if shape:
        if not isinstance(value, tuple) or len(value) != shape[0]:
            raise ValueError(f'{name} expected shape {shape}')
        for item in value:
            _shape(item, shape[1:], name, boolean=boolean, nullable=nullable)
    elif boolean:
        if type(value) is not bool:
            raise ValueError(f'{name} mask must contain booleans')
    elif value is None and nullable:
        return
    else:
        _finite(value, name)


def _indices(shape):
    return product(*(range(n) for n in shape))


def _get(array, index):
    for i in index:
        array = array[i]
    return array


def _nested(shape, values, prefix=()):
    if not shape:
        return values.get(prefix)
    return tuple(_nested(shape[1:], values, prefix + (i,)) for i in range(shape[0]))


def _check_clock(sample, current, max_age_s=None):
    if not isinstance(sample, SampleClock) or not isinstance(current, SampleClock):
        raise ValueError('clock must be SampleClock')
    for name in ('run_uuid', 'env_id', 'episode_uid', 'reset_generation'):
        if getattr(sample, name) != getattr(current, name):
            raise ValueError(f'clock crosses {name}')
    if sample.terminal or current.terminal or sample.reset or current.reset:
        raise ValueError('terminal/reset clocks are not objective inputs')
    if current.policy_step < sample.policy_step:
        raise ValueError('clock policy step reversed')
    for step, time in (('world_step', 'world_time_s'), ('sdk_step', 'sdk_time_s')):
        delta = getattr(current, time) - getattr(sample, time)
        if (getattr(current, step) < getattr(sample, step) or delta < 0
                or ((getattr(current, step) == getattr(sample, step)) != (delta == 0))
                or (max_age_s is not None and delta > max_age_s)):
            raise ValueError('clock is stale, ahead or inconsistent')


def candidate_digest(actions) -> str:
    """Value identity only; never certifies a rollout's physical correctness."""
    array = _freeze(actions)
    _shape(array, (HORIZON, ACTION_DIM), 'actions')
    packed = b''.join(struct.pack('!d', v) for row in array for v in row)
    return hashlib.sha256(packed).hexdigest()


@dataclass(frozen=True, slots=True)
class ResidualEvidence:
    values: object
    valid_mask: object
    source: str
    unit: str
    frame: str
    clock: SampleClock
    scope: str
    qualification: str
    candidate_digest: str | None = None
    missing_reason: str | None = None
    contact_basis: str = 'not_applicable'
    contact_mask: object = None
    control_period_s: float | None = None
    coordinate_units: tuple[str, ...] | None = None

    def __post_init__(self):
        for name in ('source', 'unit', 'frame', 'qualification'):
            _text(getattr(self, name), name)
        if not isinstance(self.clock, SampleClock):
            raise ValueError('evidence clock must be SampleClock')
        if self.scope not in ('measured_current', 'predicted_rollout'):
            raise ValueError('scope must be measured_current or predicted_rollout')
        if self.scope == 'predicted_rollout':
            _text(self.candidate_digest, 'candidate_digest')
            _control_period(self.control_period_s)
        elif self.candidate_digest is not None:
            raise ValueError('measured data cannot depend on a candidate')
        elif self.control_period_s is not None:
            raise ValueError('measured data has no future control period')
        if self.unit == 'joint_coordinate_per_dof':
            object.__setattr__(self, 'coordinate_units', _coordinate_units(self.coordinate_units))
        elif self.coordinate_units is not None:
            raise ValueError('coordinate units require unit=joint_coordinate_per_dof')
        if self.missing_reason is not None:
            _text(self.missing_reason, 'missing_reason')
        if self.values is None and self.missing_reason is None:
            raise ValueError('unavailable evidence requires missing_reason')
        for name in ('values', 'valid_mask', 'contact_mask'):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    """Explicit experimental weights; no defaults, fitted/optimal claim or tuning."""
    geometry: float
    contact: float
    support: float
    slip: float
    bimanual: float
    joint: float
    smooth: float
    risk: float

    def __post_init__(self):
        for field in fields(self):
            _finite(getattr(self, field.name), f'{field.name} weight', nonnegative=True)


@dataclass(frozen=True, slots=True)
class ObjectiveInput:
    clock: SampleClock
    current_clock: SampleClock
    max_age_s: float
    control_period_s: float
    actions: object
    previous_action: object
    previous_action_clock: SampleClock
    previous_action_source: str
    active_hands: object
    active_pads: object
    required_support_N: float
    bimanual_translation_scale_m: float
    bimanual_rotation_scale_rad: float
    joint_dof_names: tuple[str, ...]
    joint_scope_source: str
    joint_coordinate_units: tuple[str, ...]
    joint_reference_per_dof: object
    joint_reference_source: str
    residuals: Mapping[str, ResidualEvidence]

    def __post_init__(self):
        _finite(self.max_age_s, 'max_age_s', nonnegative=True)
        _check_clock(self.clock, self.current_clock, self.max_age_s)
        _check_clock(self.previous_action_clock, self.clock)
        _text(self.previous_action_source, 'previous_action_source (last actually issued)')
        _control_period(self.control_period_s)
        for name in ('bimanual_translation_scale_m', 'bimanual_rotation_scale_rad'):
            _finite(getattr(self, name), name, positive=True)
        _finite(self.required_support_N, 'required_support_N', nonnegative=True)
        if (not isinstance(self.joint_dof_names, (tuple, list)) or not self.joint_dof_names
                or len(set(self.joint_dof_names)) != len(self.joint_dof_names)):
            raise ValueError('joint_dof_names must explicitly name unique local axes')
        for name in self.joint_dof_names:
            _text(name, 'joint_dof_names')
        _text(self.joint_scope_source, 'joint_scope_source')
        object.__setattr__(self, 'joint_dof_names', tuple(self.joint_dof_names))
        units = _coordinate_units(self.joint_coordinate_units)
        if len(units) != len(self.joint_dof_names):
            raise ValueError('joint coordinate units must match joint_dof_names')
        object.__setattr__(self, 'joint_coordinate_units', units)
        try:
            references = _freeze(self.joint_reference_per_dof)
        except ValueError as exc:
            raise ValueError('joint_reference_per_dof must contain finite real scales') from exc
        _shape(references, (len(self.joint_dof_names),), 'joint_reference_per_dof')
        for value in references:
            _finite(value, 'joint_reference_per_dof', positive=True)
        _text(self.joint_reference_source, 'joint_reference_source (same-unit experimental scales)')
        object.__setattr__(self, 'joint_reference_per_dof', references)
        for name, shape, boolean in (
                ('actions', (HORIZON, ACTION_DIM), False),
                ('previous_action', (ACTION_DIM,), False),
                ('active_hands', (2,), True), ('active_pads', (2, 3), True)):
            value = _freeze(getattr(self, name))
            _shape(value, shape, name, boolean=boolean)
            object.__setattr__(self, name, value)
        if not isinstance(self.residuals, Mapping):
            raise ValueError('residuals must be an explicit mapping')
        copied = dict(self.residuals)
        if copied.keys() - _SPECS.keys():
            raise ValueError('unknown residual names')
        if any(not isinstance(v, ResidualEvidence) for v in copied.values()):
            raise ValueError('residuals must contain ResidualEvidence')
        object.__setattr__(self, 'residuals', MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class CostTerm:
    raw: float | None
    weight: float
    weighted: float | None
    available: bool
    missing_reason: str | None
    optimization_effect: str
    raw_unit: str
    input_units: tuple[str, ...]
    source: tuple[str, ...]
    qualification: tuple[str, ...]
    prediction_scope: tuple[str, ...]
    valid_mask: Mapping
    required_mask: Mapping
    element_costs: Mapping
    observed_raw: float | None
    clock: SampleClock


@dataclass(frozen=True, slots=True)
class ObjectiveReport:
    terms: Mapping[str, CostTerm]
    active_costs: tuple[str, ...]
    total: float | None
    full_spec_cost_valid: bool
    clock: SampleClock
    control_period_s: float
    candidate_digest: str
    joint_dof_names: tuple[str, ...]
    joint_scope_source: str
    joint_coordinate_units: tuple[str, ...]
    joint_reference_per_dof: tuple[float, ...]
    joint_reference_source: str
    native_qualified: ClassVar[bool] = False
    prediction_validated: ClassVar[bool] = False
    solve_qualified: ClassVar[bool] = False
    action_admissible: ClassVar[bool] = False


def _selected(data, name, index):
    if name == 'bimanual':
        return all(data.active_hands)
    hand = index[1]
    if name in ('geometry', 'contact', 'slip', 'overload'):
        return data.active_hands[hand] and data.active_pads[hand][index[2]]
    return data.active_hands[hand]


def _tail_shape(data, name):
    return tuple(len(data.joint_dof_names) if n is None else n for n in _SPECS[name][0])


def _component(data, name, digest):
    """Validate one source and keep missing cells distinct from excluded cells."""
    evidence = data.residuals.get(name)
    if evidence is None:
        shape = (1,) + _tail_shape(data, name)
        return ({}, {i: bool(_selected(data, name, i)) for i in _indices(shape)},
                {i: False for i in _indices(shape)}, (f'{name}: not supplied',), None)
    if evidence.clock != data.clock:
        raise ValueError(f'{name}: evidence clock must exactly match input snapshot clock')
    _, unit, frame = _SPECS[name]
    tail = _tail_shape(data, name)
    if (evidence.unit, evidence.frame) != (unit, frame):
        raise ValueError(f'{name}: unit/frame mismatch')
    if name == 'joint' and evidence.coordinate_units != data.joint_coordinate_units:
        raise ValueError('joint: evidence coordinate units must match the named input axes')
    if evidence.scope == 'predicted_rollout' and evidence.candidate_digest != digest:
        raise ValueError(f'{name}: candidate digest mismatch')
    if evidence.scope == 'predicted_rollout' and evidence.control_period_s != data.control_period_s:
        raise ValueError(f'{name}: predicted control period mismatch')
    count = HORIZON if evidence.scope == 'predicted_rollout' else 1
    shape = (count,) + tail
    _shape(evidence.valid_mask, shape, f'{name}.valid_mask', boolean=True)
    if evidence.values is None:
        if any(_get(evidence.valid_mask, i) for i in _indices(shape)):
            raise ValueError(f'{name}: unavailable data cannot have valid elements')
        return ({}, {i: bool(_selected(data, name, i)) for i in _indices(shape)},
                {i: False for i in _indices(shape)}, (f'{name}: {evidence.missing_reason}',), evidence)
    _shape(evidence.values, shape, name, nullable=True)
    if name == 'contact':
        if evidence.contact_basis not in ('accepted_contact', 'required_hold_contact_model'):
            return {}, {}, {}, ('contact: contact basis is not actual accepted/required hold',), evidence
        _shape(evidence.contact_mask, shape, 'contact.contact_mask', boolean=True)
    costs, required, valid = {}, {}, {}
    missing = []
    for index in _indices(shape):
        value, flag = _get(evidence.values, index), _get(evidence.valid_mask, index)
        if flag != (value is not None):
            raise ValueError(f'{name}: invalid elements require None, valid elements require finite values')
        selected = _selected(data, name, index)
        if name == 'contact':
            selected = selected and _get(evidence.contact_mask, index)
        required[index] = bool(selected)
        valid[index] = bool(selected and flag)
        if value is not None:
            _finite(value, name, positive=name == 'joint',
                    nonnegative=name not in ('support', 'bimanual', 'joint'))
        if selected and not flag:
            if not evidence.missing_reason:
                raise ValueError(f'{name}: partial unavailable data requires missing_reason')
            missing.append(f'{name}: {evidence.missing_reason}')
        if not selected or not flag:
            continue
        if name == 'contact' or name == 'support':
            costs[index] = value  # contact is already valid_contact_error; support reduced below.
        elif name == 'joint':
            costs[index] = _square(data.joint_reference_per_dof[index[2]] / value)
        elif name == 'bimanual':
            scale = (data.bimanual_translation_scale_m if index[1] < 3
                     else data.bimanual_rotation_scale_rad)
            costs[index] = _square(value / scale)
        else:
            costs[index] = _square(value)
        _finite(costs[index], f'{name} element cost')
    if name == 'support':
        combined, combined_required, combined_valid = {}, {}, {}
        for t in range(count):
            selected_hands = [h for h in range(2) if required[t, h]]
            combined_required[t,] = bool(selected_hands)
            combined_valid[t,] = bool(selected_hands) and all(valid[t, h] for h in selected_hands)
            if combined_valid[t,]:
                combined[t,] = _square(max(0., data.required_support_N - _sum(costs[t, h] for h in selected_hands)))
                _finite(combined[t,], 'support element cost')
        # Retain hand masks in the caller report; scalar costs use the total selected hands.
        costs = combined
        if not any(combined_valid.values()) and not missing:
            missing.append('support: no complete selected-hand support sample')
    if not any(required.values()):
        missing.append(f'{name}: no active eligible elements')
    return costs, required, valid, tuple(dict.fromkeys(missing)), evidence


def evaluate_objective(data: ObjectiveInput, weights: ObjectiveWeights, *,
                       active_costs: tuple[str, ...]) -> ObjectiveReport:
    """Evaluate only a declared cost set; any active NA makes total=None.

    raw=None denotes an incomplete term. observed_raw/element_costs retain only
    available selected cells as diagnostics, never a substitute objective.
    Different raw units are converted to caller-defined cost units by explicit
    dimensional weights. A zero weight never changes term/full-spec validity.
    """
    if not isinstance(data, ObjectiveInput) or not isinstance(weights, ObjectiveWeights):
        raise ValueError('ObjectiveInput and ObjectiveWeights are required')
    if (not isinstance(active_costs, tuple) or not active_costs
            or len(set(active_costs)) != len(active_costs)
            or any(n not in TERM_NAMES for n in active_costs)):
        raise ValueError('active_costs must explicitly name unique known terms')
    digest = candidate_digest(data.actions)
    components = {n: _component(data, n, digest) for n in _SPECS}
    terms = {}
    for name in TERM_NAMES:
        weight = getattr(weights, name)
        if name == 'smooth':
            cells = {}
            for t in range(HORIZON):
                previous = data.previous_action if t == 0 else data.actions[t - 1]
                for j in range(ACTION_DIM):
                    cells[t, j] = _square(data.actions[t][j] - previous[j])
            raw = _sum(cells.values())
            _finite(raw, 'smooth raw')
            flags = _nested((HORIZON, ACTION_DIM), {i: True for i in cells})
            terms[name] = CostTerm(raw, weight, raw * weight, True, None,
                'candidate_dependent', _RAW_UNITS[name], ('dimensionless',),
                ('candidate canonical24Tanh sequence', data.previous_action_source),
                ('arithmetic_only; last issued action is caller-declared',),
                ('candidate_sequence',), MappingProxyType({'smooth': flags}),
                MappingProxyType({'smooth': flags}), MappingProxyType({'smooth': _nested((HORIZON, ACTION_DIM), cells)}),
                raw, data.clock)
        else:
            names = ('collision', 'overload', 'tracking') if name == 'risk' else (name,)
            observed, reasons, sources, units, qualifications, scopes = [], [], [], [], [], []
            masks, required_masks, elements = {}, {}, {}
            for component in names:
                costs, required, valid, missing, evidence = components[component]
                observed.extend(costs.values())
                reasons.extend(missing)
                count = HORIZON if evidence and evidence.scope == 'predicted_rollout' else 1
                shape = (count,) + _tail_shape(data, component)
                masks[component], required_masks[component] = _nested(shape, valid), _nested(shape, required)
                elements[component] = _nested((count,) if component == 'support' else shape, costs)
                sources.append(evidence.source if evidence else f'unavailable:{component}')
                units.append(_SPECS[component][1])
                qualifications.append(evidence.qualification if evidence else 'unavailable')
                scopes.append(evidence.scope if evidence else 'unavailable')
                if component in ('joint', 'tracking'):
                    sources.append(data.joint_scope_source)
                    qualifications.append('declared_joint_axes_only; completeness_not_certified')
                if component == 'joint':
                    sources.append(data.joint_reference_source)
                    qualifications.append('same_unit_per_dof_experimental_reference; not_physical_limits')
            available = not reasons
            observed_raw = _sum(observed) if observed else None
            raw = observed_raw if available else None
            effect = ('unavailable' if not available else 'candidate_dependent'
                      if 'predicted_rollout' in scopes else 'constant_wrt_candidate')
            terms[name] = CostTerm(raw, weight, None if raw is None else raw * weight,
                available, '; '.join(dict.fromkeys(reasons)) if reasons else None, effect,
                _RAW_UNITS[name], tuple(units), tuple(sources), tuple(qualifications), tuple(scopes),
                MappingProxyType(masks), MappingProxyType(required_masks), MappingProxyType(elements),
                observed_raw, data.clock)
        if terms[name].weighted is not None:
            _finite(terms[name].weighted, f'{name} weighted cost')
    total = (_sum(terms[n].weighted for n in active_costs)
             if all(terms[n].available for n in active_costs) else None)
    if total is not None:
        _finite(total, 'total cost')
    return ObjectiveReport(MappingProxyType(terms), active_costs, total,
                           all(t.available for t in terms.values()), data.clock,
                           data.control_period_s, digest, data.joint_dof_names,
                           data.joint_scope_source, data.joint_coordinate_units,
                           data.joint_reference_per_dof, data.joint_reference_source)
