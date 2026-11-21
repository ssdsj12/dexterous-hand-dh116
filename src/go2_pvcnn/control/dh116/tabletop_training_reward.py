"""Approved reward arithmetic and owned CPU settlement, without native admission.

No SDK targets, sources, solvers, PPO updates or native qualification are created.
C01/C04 must bind real readers and the evaluator to this endpoint contract. Source
labels describe evidence; they never prove that a reader ran. CPU fixture packets
are useful for software tests only, and cannot admit a native Teacher session.
"""
from dataclasses import dataclass, field, fields
import hashlib
import json
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Mapping

PHYSICS_HZ = 480
CONTROL_HZ = 60
WINDOW_SUBSTEPS = 8
DESIGN_SHA256 = "a6a04d7dab20048d22b152762eb07018c3269cd63bfad77ed2b11799d7ed97d9"
SCHEMA_VERSION = "dh116-training-reward-endpoints-v1"
SCHEMA_SHA256 = hashlib.sha256((SCHEMA_VERSION + ":owned-clock-measurement-request24-fixed-material-tangent-reference:fixed-required-pads:cpu-unqualified").encode()).hexdigest()
SUCCESS = MappingProxyType({"net_lift_m": .03, "hold_s": 5., "max_slip_m": .0005})
SMOKE = "approach_wiring_smoke_v1"
FULL = "flexible_train_r1_full"
CURRICULUM = "flexible_train_curriculum_v1"
_SCOPES = ("cpu_fixture", "unqualified_source")
_BASE_UNITS = {"distance": "m", "numerical_health": "dimensionless", "constraint_failure": "dimensionless"}
_FULL_UNITS = {"median_z": "m", "min_z": "m", "coverage_left": "dimensionless", "coverage_right": "dimensionless",
               "opposed_left": "dimensionless", "opposed_right": "dimensionless", "bilateral_support": "dimensionless",
               "off_table": "dimensionless", "no_external_support": "dimensionless", "hold_motion": "dimensionless",
               "prohibited_support": "dimensionless"}
_BOOL_FIELDS = frozenset((set(_BASE_UNITS) - {"distance"}) | (set(_FULL_UNITS) - {"median_z", "min_z", "coverage_left", "coverage_right"}))


def _finite(value):
    """Reject nonprimitive numbers before conversion; overflow is invalid input."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _solver_interval_matches(previous, current, *, native_smoke):
    step = 1 / PHYSICS_HZ
    interval = current.solver_time_s - previous.solver_time_s
    if math.isclose(interval, step, rel_tol=0., abs_tol=1e-9):
        return True
    if not (native_smoke and current.scope == "unqualified_source"
            and current.source in ("policy_owned_pinch_endpoint_v1","policy_owned_tripod_endpoint_v1")):
        return False
    times = (previous.solver_time_s, current.solver_time_s)
    # The native RopeBatch getter returns binary32 seconds. Keep these
    # exact observations in the reward stamp and bound only their rounding.
    if any(not 0. <= value <= float.fromhex("0x1.fffffep+127") for value in times):
        return False
    rounded = [struct.unpack(">f", struct.pack(">f", value))[0] for value in times]
    if tuple(rounded) != times:
        return False
    step32 = struct.unpack(">f", struct.pack(">f", step))[0]
    ulps = [max(2. ** -149, math.ldexp(1., math.frexp(value)[1] - 24))
           if value else 2. ** -149 for value in times]
    error_bound = min(1e-6, .5 * sum(ulps) + abs(step32 - step))
    return math.isclose(interval, step, rel_tol=0., abs_tol=error_bound)


def _owned_raw(value):
    """Retained scalar/vector evidence has only immutable, finite leaves."""
    pending = [value]
    while pending:
        part = pending.pop()
        if type(part) is tuple:
            pending.extend(part)
        elif part is not None and type(part) is not bool and not _finite(part):
            return False
    return True


def _identity_text(value):
    """Own primitive identity text; comparison-compatible mutable objects fail."""
    return type(value) is str and bool(value.strip())


def _clip(value, low=0., high=1.):
    if any(not _finite(v) for v in (value, low, high)):
        raise ValueError("finite derived value required before clipping")
    return min(high, max(low, value))


def command_rate(action, previous):
    """Mean squared normalized submitted-request delta, not energy or SDK motion."""
    if len(action) != 24 or len(previous) != 24:
        raise ValueError("normalized request24 required")
    if any(not _finite(v) or abs(v) > 1 for v in (*action, *previous)):
        raise ValueError("finite bounded normalized request required")
    return math.fsum(((a - b) / 2.) ** 2 for a, b in zip(action, previous)) / 24.


def smoke_potential(distance_m, distance_ref_m):
    if not _finite(distance_m) or distance_m < 0:
        raise ValueError("finite actual distance required")
    if not _finite(distance_ref_m) or distance_ref_m < .005:
        raise ValueError("fixed reset-derived distance scale required")
    return 2. * (1. - _clip(distance_m / distance_ref_m))


def control_total(phi_pre, phi_post, cost_per_second, gamma, event, terminated):
    """Integrate only actual 480Hz intervals; absorb the terminal potential once."""
    if not 1 <= len(cost_per_second) <= WINDOW_SUBSTEPS:
        raise ValueError("one to eight actual SDK samples required")
    if any(not _finite(v) for v in (phi_pre, phi_post, gamma, event, *cost_per_second)):
        raise ValueError("finite reward components required")
    if not 0 < gamma < 1 or any(v < 0 for v in cost_per_second):
        raise ValueError("bounded gamma and nonnegative costs required")
    try:
        total = gamma * (0. if terminated else phi_post) - phi_pre - math.fsum(cost_per_second) / PHYSICS_HZ + event
    except OverflowError as exc:
        raise ValueError("finite integrated reward required; cost sum overflow") from exc
    if not _finite(total):
        raise ValueError("finite integrated reward required")
    return total


def slip_status(drift_uncertainty_pairs, threshold_m=.0005):
    """Original three-valued conservative classification, independent of termination."""
    if not drift_uncertainty_pairs:
        raise ValueError("applicable anchor measurements required")
    if any(not _finite(x) or x < 0 for pair in drift_uncertainty_pairs for x in pair):
        raise ValueError("finite nonnegative measured drift and uncertainty required")
    if any(not _finite(d + u) or not _finite(d - u) for d, u in drift_uncertainty_pairs):
        raise ValueError("finite derived slip bounds required")
    if any(d - u > threshold_m for d, u in drift_uncertainty_pairs):
        return False
    if all(d + u <= threshold_m for d, u in drift_uncertainty_pairs):
        return True
    return None


def conservative_slip(drift_uncertainty_pairs):
    """Single pure upper-bound implementation for new endpoint consumers."""
    slip_status(drift_uncertainty_pairs)
    return max(d + u for d, u in drift_uncertainty_pairs)


class SourceFailure(ValueError):
    """Corrupt/unknown applicable evidence: caller must quarantine its rollout."""


def _source_math(function, *args):
    """Translate only checked reward arithmetic failures at the source boundary."""
    try:
        return function(*args)
    except (ValueError, OverflowError) as exc:
        raise SourceFailure("nonfinite derived source arithmetic: " + str(exc)) from exc


def _source_value(value):
    if not _finite(value):
        raise SourceFailure("finite derived source value required")
    return value


@dataclass(frozen=True)
class SourceStamp:
    episode_uid: str
    physics_tick: int
    solver_time_s: float
    scope: str
    source: str
    frame: str = "world"
    env_id: int = 0
    reset_uid: str = "cpu-reset"
    rod_uid: str = "cpu-rod"

    @property
    def clock_identity(self):
        return (self.episode_uid, self.physics_tick, self.solver_time_s, self.scope,
                self.frame, self.env_id, self.reset_uid, self.rod_uid)


@dataclass(frozen=True)
class Measurement:
    raw: float | bool | None
    unit: str
    stamp: SourceStamp
    valid: bool = True
    applicable: bool = True
    history_reason: str | None = None


@dataclass(frozen=True)
class SlipAnchor:
    side: str
    kind: str
    reference_uid: str
    reference: tuple
    reference_tick: int
    drift: Measurement
    uncertainty: Measurement

    def __post_init__(self):
        object.__setattr__(self, "reference", tuple(self.reference))


@dataclass(frozen=True)
class RewardEndpoint:
    stamp: SourceStamp
    target_ids: tuple[str, ...]
    fields: Mapping[str, Measurement]
    anchors: tuple[SlipAnchor, ...] = ()
    required_pad_ids: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "target_ids", tuple(self.target_ids))
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "anchors", tuple(self.anchors))
        object.__setattr__(self, "required_pad_ids", tuple((side, tuple(pads)) for side, pads in self.required_pad_ids))


@dataclass(frozen=True)
class SubmittedRequest:
    token: str
    policy_uid: str
    stamp: SourceStamp
    values: tuple[float, ...]
    representation: str

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True)
class RewardConfig:
    profile: str
    profile_version: int
    ppo: Mapping
    weights: Mapping
    costs: Mapping
    events: Mapping
    config_sha256: str
    schema_sha256: str
    cost_max_per_second: float
    discounted_cost_bound: float
    success: Mapping = field(default_factory=lambda: SUCCESS)
    # Data construction/replacement cannot inherit the loader's source binding.
    _source_path: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ("ppo", "weights", "costs", "events", "success"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def gamma(self):
        return self.ppo["gamma"]


def _profile_spec(profile):
    full = profile in (FULL, CURRICULUM)
    weights = {"distance": 2., "contact": 2., "persistence": .5, "geometry": 1., "lift": 4., "hold": 4.} if full else {"distance": 2.}
    slip_cost = 0.25 if profile == CURRICULUM else (1. if profile == FULL else None)
    costs = {"command_rate": .05, "time": .10, "slip": slip_cost, "velocity": 0. if full else None}
    events = {"trusted_constraint_failure": -40., "prohibited_support": -50. if full else None, "success": 100. if full else None}
    ppo = dict(gae_lambda=.97 if full else .95, rollout_steps=256 if full else 32,
               learning_rate=1e-4, epochs=3 if full else 1, minibatches=4 if full else 1,
               clip=.2, entropy=.005, value=.5, max_grad_norm=.5, target_kl=.02, kl_backtrack_steps=3)
    return weights, costs, events, ppo


def _config_mapping(value, expected, name):
    # Approved maps are flat. Only primitive numeric/NA leaves are admissible;
    # nested containers, custom equality and bool-as-number are never retained.
    if (type(value) not in (dict, MappingProxyType) or any(type(k) is not str for k in value)
            or set(value) != set(expected)):
        raise ValueError("approved config mapping schema required: " + name)
    for key, wanted in expected.items():
        actual = value[key]
        if wanted is None:
            valid = actual is None
        elif type(wanted) is int:
            valid = type(actual) is int and actual == wanted
        else:
            valid = _finite(actual) and actual == wanted
        if not valid:
            raise ValueError("approved primitive config value required: " + name + "." + key)


def _config_ppo(value, expected):
    if type(value) not in (dict, MappingProxyType):
        raise ValueError("approved PPO config mapping required")
    gamma = value.get("gamma")
    if not _finite(gamma) or not 0 < gamma < 1:
        raise ValueError("same finite PPO gamma required")
    _config_mapping(value, {**expected, "gamma": gamma}, "ppo")
    return gamma


def _sha256_text(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_reward_config(config):
    if not _identity_text(config.profile) or config.profile not in (SMOKE, FULL, CURRICULUM):
        raise ValueError("loaded approved config profile required")
    if type(config.profile_version) is not int or config.profile_version != 1:
        raise ValueError("exact approved config version required")
    if not _sha256_text(config.config_sha256):
        raise ValueError("primitive full config SHA256 identity required")
    if not _sha256_text(config.schema_sha256) or config.schema_sha256 != SCHEMA_SHA256:
        raise ValueError("loaded approved config schema SHA256 required")
    weights, costs, events, ppo = _profile_spec(config.profile)
    _config_mapping(config.weights, weights, "weights")
    _config_mapping(config.costs, costs, "costs")
    _config_mapping(config.events, events, "events")
    gamma = _config_ppo(config.ppo, ppo)
    slip_max = 2. * config.costs["slip"] if config.profile in (FULL, CURRICULUM) else 0.
    maximum = .05 + .10 + slip_max
    bound = maximum / (CONTROL_HZ * (1. - gamma))
    if (not _finite(config.cost_max_per_second)
            or not math.isclose(config.cost_max_per_second, maximum, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("loaded config finite cost maximum mismatch")
    if (not _finite(config.discounted_cost_bound) or bound > 40.
            or not math.isclose(bound, config.discounted_cost_bound, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("loaded config discounted cost bound mismatch")


def _config_payload(config):
    return tuple(getattr(config, f.name) for f in fields(RewardConfig) if f.init)


def load_config(path, *, ppo_gamma=None):
    """Read actual file bytes; only this loader binds a configuration to a source."""
    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    data = json.loads(raw)
    if type(data) is not dict:
        raise ValueError("approved config JSON object required")
    if data.get("gamma_source") != "ppo.gamma" or "reward_gamma" in data or "gamma" in data:
        raise ValueError("reward gamma must come only from ppo.gamma")
    profile = data.get("profile")
    if not _identity_text(profile) or profile not in (SMOKE, FULL, CURRICULUM):
        raise ValueError("independent approved reward profile required")
    success = data.get("success")
    if profile == CURRICULUM:
        if (type(success) is not dict or set(success) != set(SUCCESS)
                or any(not _finite(success[k]) or success[k] <= 0 for k in SUCCESS)):
            raise ValueError("positive finite curriculum success thresholds required")
    elif success != SUCCESS:
        raise ValueError("original success thresholds are immutable")
    if (data.get("schema_sha256") != SCHEMA_SHA256 or data.get("design_document_sha256") != DESIGN_SHA256
            or data.get("weight_status") != "design_approved_native_unverified"
            or type(data.get("profile_version")) is not int or data["profile_version"] != 1):
        raise ValueError("approved design/version/schema provenance required")
    if data.get("legacy_r1_enabled") is not False or data.get("legacy_r2_enabled") is not False:
        raise ValueError("legacy R1/R2 remain OFF")
    deadline = data.get("finite_task_deadline")
    if (type(deadline) is not dict or set(deadline) != {"enabled", "remaining_s"}
            or deadline["enabled"] is not False or deadline["remaining_s"] is not None):
        raise ValueError("first profiles disable finite task deadline; external caps are truncations")
    weights, costs, events, expected_ppo = _profile_spec(profile)
    ppo = data.get("ppo")
    gamma = _config_ppo(ppo, expected_ppo)
    if ppo_gamma is not None and (not _finite(ppo_gamma) or ppo_gamma != gamma):
        raise ValueError("same finite PPO gamma required")
    _config_mapping(data.get("potential_weights"), weights, "weights")
    _config_mapping(data.get("costs_per_second"), costs, "costs")
    _config_mapping(data.get("events"), events, "events")
    slip_max = 2. * costs["slip"] if profile in (FULL, CURRICULUM) else 0.
    maximum = .05 + .10 + slip_max
    bound = maximum / (CONTROL_HZ * (1. - gamma))
    if abs(events["trusted_constraint_failure"]) < bound:
        raise ValueError("failure penalty does not cover discounted cost bound; recompute approved bounds")
    result = RewardConfig(profile, 1, ppo, weights, costs, events, hashlib.sha256(raw).hexdigest(),
                          SCHEMA_SHA256, maximum, bound, success)
    _validate_reward_config(result)
    object.__setattr__(result, "_source_path", str(source_path))
    return result


def _term(raw, unit, source, contribution=0., *, applicable=True, history_reason=None, valid=True):
    if not _owned_raw(raw) or not _finite(contribution):
        raise SourceFailure("finite owned term raw and contribution required")
    return MappingProxyType(dict(raw=raw, unit=unit, valid=valid, applicable=applicable,
                                 source=source, contribution=contribution, history_reason=history_reason))


def _excluded():
    return _term(None, "dimensionless", "profile_excluded", applicable=False, history_reason="profile_excluded")


@dataclass(frozen=True)
class RewardPacket:
    token: str
    policy_uid: str
    profile: str
    profile_version: int
    config_sha256: str
    schema_sha256: str
    reward: float
    duration_s: float
    phi_pre: float
    phi_post_raw: float
    endpoint_phi: float
    cost_per_second: tuple[float, ...]
    integrated_cost: float
    event: float
    event_reasons: tuple[str, ...]
    endpoint: RewardEndpoint
    post_context: Mapping
    terms: Mapping
    substep_terms: tuple
    terminated: bool
    truncated: bool
    buffer_full: bool
    done: bool
    scope: str
    gamma: float
    pre_endpoint: RewardEndpoint
    pre_context: Mapping
    pre_terms: Mapping
    event_terms: Mapping

    @property
    def native_ready(self): return False
    @property
    def teacher_admission(self): return False
    @property
    def native_qualification(self): return False

    def bootstrap(self, shaped_value):
        if not _finite(shaped_value):
            raise ValueError("finite shaped value from actual post_context required")
        return 0. if self.terminated else self.gamma * shaped_value


class TrainingReward:
    """Exactly-once endpoint consumer; never emits/resubmits physical actions.

    begin reserves a token, observe consumes one actual interval, settle_once
    returns one packet. SourceFailure poisons this instance until a new reset
    episode. The caller owns rollout quarantine and must not update on failure.
    """
    def __init__(self, config):
        if type(config) is not RewardConfig:
            raise TypeError("loaded RewardConfig required")
        _validate_reward_config(config)
        if not _identity_text(config._source_path) or not Path(config._source_path).is_absolute():
            raise ValueError("actual file-backed loader provenance required")
        try:
            actual = load_config(config._source_path, ppo_gamma=config.gamma)
        except (OSError, ValueError) as exc:
            raise ValueError("reward config source file no longer validates against loaded bytes") from exc
        if _config_payload(config) != _config_payload(actual):
            raise ValueError("reward config source bytes/SHA/content mismatch")
        # Hold the fresh file-derived, fully owned instance, not caller metadata.
        self.config = actual
        self._tokens = set()
        self._episodes = set()
        self._last = None
        self._request = None
        self._closed = False
        self.quarantined = False
        self.failed_endpoint = None
        self._context = MappingProxyType({})

    @property
    def native_ready(self): return False
    @property
    def teacher_admission(self): return False
    @property
    def native_qualification(self): return False
    @property
    def post_context(self): return self._context

    def _fail(self, reason, endpoint=None):
        self.quarantined = True
        self.failed_endpoint = endpoint
        raise SourceFailure(reason)

    def _stamp(self, stamp):
        if (type(stamp) is not SourceStamp or any(not _identity_text(v) for v in (stamp.episode_uid, stamp.reset_uid, stamp.rod_uid, stamp.scope, stamp.source, stamp.frame))
                or type(stamp.physics_tick) is not int or stamp.physics_tick < 0 or not _finite(stamp.physics_tick) or not _finite(stamp.solver_time_s)
                or stamp.scope not in _SCOPES
                or stamp.frame != "world" or type(stamp.env_id) is not int or stamp.env_id < 0 or not _finite(stamp.env_id)):
            raise SourceFailure("explicit owned source/clock/world/scope identity required")

    def _retained_measurement(self, value):
        # Ownership is independent of profile applicability. Excluded evidence
        # may be invalid/NA, have another unit/clock or change its reader.
        if (type(value) is not Measurement or not _owned_raw(value.raw)
                or not _identity_text(value.unit)
                or (value.history_reason is not None and not _identity_text(value.history_reason))
                or type(value.valid) is not bool or type(value.applicable) is not bool):
            raise SourceFailure("immutable finite measurement schema required")
        self._stamp(value.stamp)

    def _measurement(self, row, name, unit, *, nullable=False, source_key=None):
        value = row.fields.get(name)
        if type(value) is not Measurement:
            raise SourceFailure("missing applicable source: " + name)
        self._stamp(value.stamp)
        if (value.stamp.clock_identity != row.stamp.clock_identity or not _identity_text(value.unit)
                or value.unit != unit or (value.history_reason is not None and not _identity_text(value.history_reason))):
            raise SourceFailure("same-clock units and identity required: " + name)
        if type(value.valid) is not bool or type(value.applicable) is not bool or not value.valid:
            raise SourceFailure("invalid applicable source: " + name)
        # Proven NA still has a real history reader whose episode identity is fixed.
        source = self._sources.setdefault(name if source_key is None else source_key, value.stamp.source)
        if source != value.stamp.source:
            raise SourceFailure("episode source changed: " + name)
        if value.raw is None:
            if nullable and not value.applicable and value.history_reason == "no_bilateral_contact_since_reset":
                return value
            raise SourceFailure("unknown applicable source: " + name)
        if not value.applicable:
            raise SourceFailure("not-applicable raw must be None: " + name)
        if name in _BOOL_FIELDS or name.startswith("original_guard_"):
            if type(value.raw) is not bool:
                raise SourceFailure("original boolean predicate required: " + name)
        elif not _finite(value.raw):
            raise SourceFailure("nonfinite/corrupt numeric source: " + name)
        return value

    def _validate(self, row, *, reset=False):
        if type(row) is not RewardEndpoint:
            raise SourceFailure("owned RewardEndpoint required")
        if any(not _identity_text(name) for name in row.fields):
            raise SourceFailure("named measurement schema required")
        # Even profile-excluded and unconsumed fields are retained in a packet.
        # Check ownership, without requiring their validity/applicability or
        # binding their source/clock to the active reward profile.
        for value in row.fields.values():
            self._retained_measurement(value)
        for anchor in row.anchors:
            if (type(anchor) is not SlipAnchor
                    or any(not _identity_text(v) for v in (anchor.side, anchor.kind, anchor.reference_uid))
                    or type(anchor.reference_tick) is not int or not _finite(anchor.reference_tick)
                    or any(not _finite(v) for v in anchor.reference)):
                raise SourceFailure("immutable anchor reference identity required")
            self._retained_measurement(anchor.drift)
            self._retained_measurement(anchor.uncertainty)
        if any(not _identity_text(side) or any(not _identity_text(v) for v in ids)
               for side, ids in row.required_pad_ids):
            raise SourceFailure("immutable required-pad identity schema required")
        self._stamp(row.stamp)
        if not row.target_ids or any(not _identity_text(s) for s in row.target_ids) or len(set(row.target_ids)) != len(row.target_ids):
            raise SourceFailure("fixed actual material target IDs required")
        if not reset:
            prior = self._last.stamp
            if (row.stamp.episode_uid != prior.episode_uid or row.stamp.scope != prior.scope
                    or (row.stamp.env_id, row.stamp.reset_uid, row.stamp.rod_uid, row.stamp.source) !=
                       (prior.env_id, prior.reset_uid, prior.rod_uid, prior.source)
                    or row.stamp.physics_tick != prior.physics_tick + 1
                    or not _solver_interval_matches(prior, row.stamp, native_smoke=self.config.profile == SMOKE)):
                raise SourceFailure("same episode and consecutive actual 480Hz clock required")
            if row.target_ids != self._last.target_ids:
                raise SourceFailure("episode material target IDs changed")
        units = dict(_BASE_UNITS)
        if self.config.profile in (FULL, CURRICULUM):
            units.update(_FULL_UNITS)
        guards = frozenset(name for name in row.fields if name.startswith("original_guard_"))
        if reset:
            self._guards = guards
        elif guards != self._guards:
            raise SourceFailure("original guard bundle schema changed or guard source missing")
        units.update((name, "dimensionless") for name in guards)
        values = {name: self._measurement(row, name, unit).raw for name, unit in units.items()}
        if values["distance"] < 0:
            raise SourceFailure("nonnegative actual distance required")
        if self.config.profile in (FULL, CURRICULUM):
            pads = row.required_pad_ids
            if (len(pads) != 2 or any(not _identity_text(side) for side, _ in pads)
                    or {side for side, _ in pads} != {"left", "right"}
                    or any(not ids or any(not _identity_text(v) for v in ids) or len(set(ids)) != len(ids) for _, ids in pads)):
                raise SourceFailure("fixed required-pad denominator IDs required")
            if not reset and pads != self._last.required_pad_ids:
                raise SourceFailure("required-pad denominator changed during episode")
            if any(not 0 <= values[name] <= 1 for name in ("coverage_left", "coverage_right")):
                raise SourceFailure("required-pad coverage must be in [0,1]")
            self._read_anchors(row, values)
        return values

    def _read_anchors(self, row, values):
        contact = values["coverage_left"] == values["coverage_right"] == 1 and values["opposed_left"] and values["opposed_right"]
        if not row.anchors:
            na = self._measurement(row, "slip", "m", nullable=True)
            if na.raw is not None or na.applicable:
                raise SourceFailure("missing anchors require explicit proven NA applicability")
            if self._anchor_refs or self._ever_bilateral or contact:
                raise SourceFailure("latched/applicable anchors cannot become not applicable")
            self._slip = None
            self._slip_status = None
            return
        if any(type(a) is not SlipAnchor or not _identity_text(a.side) or not _identity_text(a.kind)
               or type(a.drift) is not Measurement or type(a.uncertainty) is not Measurement for a in row.anchors):
            raise SourceFailure("owned fixed-anchor measurement schema required")
        expected = {(side, kind) for side in ("left", "right") for kind in ("material", "tangential")}
        if len(row.anchors) != 4 or {(a.side, a.kind) for a in row.anchors} != expected:
            raise SourceFailure("all applicable material/tangential anchors required")
        pairs = []
        for anchor in row.anchors:
            key = (anchor.side, anchor.kind)
            self._stamp(anchor.drift.stamp)
            self._stamp(anchor.uncertainty.stamp)
            if (not _identity_text(anchor.reference_uid) or type(anchor.reference_tick) is not int
                    or not self._reset_tick <= anchor.reference_tick <= row.stamp.physics_tick
                    or not anchor.reference or any(not _finite(v) for v in anchor.reference)):
                raise SourceFailure("finite fixed anchor reference identity required")
            reference = (anchor.reference_uid, anchor.reference, anchor.reference_tick,
                         anchor.drift.stamp.source, anchor.uncertainty.stamp.source)
            if self._anchor_refs.setdefault(key, reference) != reference:
                raise SourceFailure("material/tangential anchor reference changed; reanchoring forbidden")
            temporary = RewardEndpoint(row.stamp, row.target_ids,
                                       {"anchor_drift": anchor.drift, "anchor_uncertainty": anchor.uncertainty})
            d = self._measurement(temporary, "anchor_drift", "m", source_key=(key, "drift")).raw
            u = self._measurement(temporary, "anchor_uncertainty", "m", source_key=(key, "uncertainty")).raw
            if d < 0 or u < 0:
                raise SourceFailure("nonnegative anchored slip and uncertainty required")
            pairs.append((d, u))
        self._slip = _source_math(conservative_slip, pairs)
        self._slip_status = _source_math(slip_status, pairs)
        self._ever_bilateral = self._ever_bilateral or contact

    def _check_request(self, request, row):
        if type(request) is not SubmittedRequest:
            raise SourceFailure("owned submitted request required")
        self._stamp(request.stamp)
        if request.stamp.clock_identity != row.stamp.clock_identity:
            raise SourceFailure("submitted request must match actual PRE endpoint clock")
        if not _identity_text(request.representation) or request.representation != "normalized_submitted_command24_preprojection":
            raise SourceFailure("owner normalized submitted preprojection request required")
        if not _identity_text(request.token) or not _identity_text(request.policy_uid):
            raise SourceFailure("explicit request token and frozen policy identity required")
        try:
            command_rate(request.values, request.values)
        except ValueError as exc:
            raise SourceFailure(str(exc)) from exc

    def reset_from_verified_state(self, endpoint, *, previous_request):
        """Verify schema/reset references; this name grants no native attestation."""
        if type(endpoint) is not RewardEndpoint or type(endpoint.stamp) is not SourceStamp:
            self._fail("owned reset endpoint/clock required", endpoint)
        if self._request is not None and not self._closed and not self.quarantined:
            raise RuntimeError("cannot reset an unfinished control window")
        try:
            self._stamp(endpoint.stamp)
        except SourceFailure as exc:
            self._fail(str(exc), endpoint)
        if endpoint.stamp.episode_uid in self._episodes:
            raise RuntimeError("episode reset UID must be new; cannot clear fixed references")
        self._sources = {}
        self._anchor_refs = {}
        self._ever_bilateral = False
        self._reset_tick = endpoint.stamp.physics_tick
        self._contact_start = self._hold_start = None
        self._contact_s = self._hold_s = 0.
        self._slip = self._slip_status = None
        try:
            values = self._validate(endpoint, reset=True)
            self._check_request(previous_request, endpoint)
            if not values["numerical_health"] or values["constraint_failure"] or any(v for n, v in values.items() if n.startswith("original_guard_")):
                raise SourceFailure("reset original health/constraint guards failed")
        except SourceFailure as exc:
            self._fail(str(exc), endpoint)
        self._episodes.add(endpoint.stamp.episode_uid)
        self._last = endpoint
        self._request = None
        self._closed = self.quarantined = False
        self.failed_endpoint = None
        self._dref = max(values["distance"], .005)
        self._reset_median = values.get("median_z")
        self._reset_min = values.get("min_z")
        self._tokens.add(previous_request.token)
        self._previous_token = previous_request.token
        self._previous_policy = previous_request.policy_uid
        self._previous = previous_request.values
        try:
            self._phi, self._terms = self._potential(endpoint, values)
            self._context = self._make_context(endpoint)
        except SourceFailure as exc:
            self._fail(str(exc), endpoint)

    def _potential(self, row, values):
        src = lambda name: row.fields[name].stamp.source
        pd = _source_math(smoke_potential, values["distance"], self._dref) / 2.
        terms = {name: _excluded() for name in ("distance", "contact", "persistence", "geometry", "lift", "hold", "slip", "velocity", "prohibited_support", "task_deadline")}
        terms["distance"] = _term(values["distance"], "m", src("distance"), 2. * pd)
        phi = 2. * pd
        if self.config.profile in (FULL, CURRICULUM):
            tick = row.stamp.physics_tick
            contact = values["coverage_left"] == values["coverage_right"] == 1 and values["opposed_left"] and values["opposed_right"]
            self._contact_start = (tick if self._contact_start is None else self._contact_start) if contact else None
            self._contact_s = _source_value(0. if self._contact_start is None else (tick - self._contact_start) / PHYSICS_HZ)
            # Validate both deltas before min; either overflow is corrupt evidence.
            median_delta = _source_value(values["median_z"] - self._reset_median)
            min_delta = _source_value(values["min_z"] - self._reset_min)
            h = min(median_delta, min_delta)
            succ = self.config.success
            is_curr = self.config.profile == CURRICULUM
            genuine = contact and values["bilateral_support"] and values["off_table"] and values["no_external_support"] and not values["prohibited_support"]
            if is_curr:
                lift_h = 0.5 * (median_delta + min_delta)
                lift_ok = values["bilateral_support"] and values["no_external_support"] and not values["prohibited_support"]
                pl = _source_math(_clip, lift_h / succ["net_lift_m"]) if lift_ok else 0.
                hold = (lift_ok and lift_h >= succ["net_lift_m"] and values["hold_motion"]
                        and self._slip_status is not False and values["numerical_health"] and not values["constraint_failure"])
                if hold:
                    self._hold_s = _source_value(self._hold_s + 1. / PHYSICS_HZ)
                else:
                    self._hold_s = _source_value(max(0., self._hold_s - 2. / PHYSICS_HZ))
            else:
                pl = _source_math(_clip, h / succ["net_lift_m"]) if genuine else 0.
                hold = (genuine and h >= succ["net_lift_m"] and values["hold_motion"] and self._slip_status is True
                        and values["numerical_health"] and not values["constraint_failure"])
                self._hold_start = (tick if self._hold_start is None else self._hold_start) if hold else None
                self._hold_s = _source_value(0. if self._hold_start is None else (tick - self._hold_start) / PHYSICS_HZ)
            pc = (values["coverage_left"] + values["coverage_right"]) / 2.
            pg = (values["opposed_left"] + values["opposed_right"]) / 2.
            pp, ph = _source_math(_clip, self._contact_s / .25), _source_math(_clip, self._hold_s / succ["hold_s"])
            terms.update(contact=_term(pc, "coverage_fraction", src("coverage_left") + ";" + src("coverage_right"), 2. * pc),
                         persistence=_term(self._contact_s, "s", "derived:continuous_required_bilateral_original_opposed", .5 * pp),
                         geometry=_term(pg, "dimensionless", src("opposed_left") + ";" + src("opposed_right"), pg),
                         lift=_term(h, "m", src("median_z") + ";" + src("min_z"), 4. * pl),
                         hold=_term(self._hold_s, "s", "derived:continuous_instantaneous_hold", 4. * ph),
                         prohibited_support=_term(values["prohibited_support"], "dimensionless", src("prohibited_support")))
            terms["slip"] = (_term(None, "m", row.fields["slip"].stamp.source, applicable=False,
                                    history_reason="no_bilateral_contact_since_reset") if self._slip is None else
                              _term(self._slip, "m", ";".join(a.drift.stamp.source for a in row.anchors), _source_math(_clip, self._slip / .0005, 0., 2.)))
            phi += 2. * pc + .5 * pp + pg + 4. * pl + 4. * ph
        for name in values:
            m = row.fields[name]
            terms["raw:" + name] = _term(m.raw, m.unit, m.stamp.source, applicable=m.applicable, history_reason=m.history_reason, valid=m.valid)
        for a in row.anchors if self.config.profile in (FULL, CURRICULUM) else ():
            for name, m in (("drift", a.drift), ("uncertainty", a.uncertainty)):
                terms["raw:anchor:" + a.side + ":" + a.kind + ":" + name] = _term(m.raw, m.unit, m.stamp.source)
        return _source_value(phi), MappingProxyType(terms)

    def _make_context(self, row):
        numeric = (row.stamp.physics_tick, row.stamp.solver_time_s, self._dref,
                   self._reset_median, self._reset_min, self._contact_s, self._hold_s,
                   self._previous, tuple((ref[1], ref[2]) for ref in self._anchor_refs.values()))
        if not _owned_raw(numeric):
            raise SourceFailure("finite owned numeric post_context required")
        return MappingProxyType(dict(episode_uid=row.stamp.episode_uid, physics_tick=row.stamp.physics_tick,
            solver_time_s=row.stamp.solver_time_s, target_ids=row.target_ids, distance_ref_m=self._dref,
            reset_median_z_m=self._reset_median, reset_min_z_m=self._reset_min,
            contact_s=self._contact_s, hold_s=self._hold_s, anchors=tuple(sorted(self._anchor_refs.items())),
            slip_status=self._slip_status, previous_request=self._previous, previous_request_token=self._previous_token,
            previous_policy_uid=self._previous_policy, required_pad_ids=row.required_pad_ids, finite_task_remaining_s=None,
            config_sha256=self.config.config_sha256, schema_sha256=SCHEMA_SHA256, scope=row.stamp.scope))

    def begin(self, submitted_request, *, remaining_substeps):
        if self.quarantined:
            raise SourceFailure("quarantined rollout; reset to a new verified episode required")
        if self._last is None or self._closed:
            raise RuntimeError("active verified reset episode required")
        if self._request is not None:
            raise RuntimeError("prior token has an unfinished window")
        if type(submitted_request) is not SubmittedRequest:
            self._fail("owned submitted request schema required", self._last)
        if type(remaining_substeps) is not int or remaining_substeps < 8:
            raise ValueError("eight (8) budgeted substeps required before starting a window")
        try:
            self._check_request(submitted_request, self._last)
        except SourceFailure as exc:
            self._fail(str(exc), self._last)
        if submitted_request.token in self._tokens:
            raise RuntimeError("request token already reserved/settled")
        self._tokens.add(submitted_request.token)
        self._request = submitted_request
        self._qcmd = command_rate(submitted_request.values, self._previous)
        self._phi_pre = self._phi
        self._pre_endpoint, self._pre_context, self._pre_terms = self._last, self._context, self._terms
        self._costs, self._substep_terms = [], []
        self._terminal = False
        self._event = 0.
        self._reasons = ()

    def observe(self, endpoint):
        if self.quarantined:
            raise SourceFailure("quarantined rollout has no usable PPO packet")
        if self._request is None or self._terminal or len(self._costs) >= 8:
            raise RuntimeError("no open physical interval; terminal/full window cannot advance")
        try:
            values = self._validate(endpoint)
            phi, terms = self._potential(endpoint, values)
            failures = []
            if not values["numerical_health"]: failures.append("numerical_health")
            if values["constraint_failure"]: failures.append("constraint_failure")
            failures.extend(n for n, v in values.items() if n.startswith("original_guard_") and v)
            full = self.config.profile in (FULL, CURRICULUM)
            if full and values["prohibited_support"]: failures.append("prohibited_support")
            slip_w = (self.config.costs["slip"] or 0.) if full else 0.
            slip_thr = self.config.success["max_slip_m"]
            cost = .05 * self._qcmd + .10 + (slip_w * _source_math(_clip, self._slip / slip_thr, 0., 2.) if full and self._slip is not None else 0.)
            _source_value(cost)
            self._costs.append(cost)
            costs = MappingProxyType({"command_rate": _term(self._qcmd, "dimensionless", self._request.stamp.source, .05 * self._qcmd / PHYSICS_HZ),
                                      "time": _term(1 / PHYSICS_HZ, "s", endpoint.stamp.source, .10 / PHYSICS_HZ),
                                      "slip": _term(self._slip, "m", "derived:conservative_fixed_anchors", (cost - .05 * self._qcmd - .10) / PHYSICS_HZ,
                                                    applicable=self._slip is not None, history_reason="no_bilateral_contact_since_reset" if full and self._slip is None else None) if full else _excluded()})
            self._substep_terms.append(costs)
            if failures:
                self._event = -50. if "prohibited_support" in failures else -40.
                self._reasons = tuple(failures)
                self._terminal = True
            elif full and self._hold_s >= self.config.success["hold_s"]:
                self._event, self._reasons, self._terminal = 100., ("success",), True
            self._last, self._phi, self._terms = endpoint, phi, terms
            self._previous = self._request.values
            self._previous_token = self._request.token
            self._previous_policy = self._request.policy_uid
            self._context = self._make_context(endpoint)
            return self._terminal or len(self._costs) == 8
        except SourceFailure as exc:
            self._fail(str(exc), endpoint)

    def settle_once(self, token, *, truncated=False, buffer_full=False):
        if self.quarantined:
            raise SourceFailure("quarantined rollout has no usable PPO packet")
        if self._request is not None and not _identity_text(token):
            self._fail("explicit immutable settlement token required", self._last)
        if self._request is None or token != self._request.token:
            raise RuntimeError("token absent or already settled")
        if not self._costs or (len(self._costs) < 8 and not self._terminal):
            raise RuntimeError("incomplete window: eight (8) samples required unless trusted terminal")
        if type(truncated) is not bool or type(buffer_full) is not bool or (truncated and buffer_full) or (self._terminal and truncated):
            raise ValueError("one unambiguous terminal/truncation/buffer boundary required")
        try:
            event_terms = MappingProxyType({
                "trusted_constraint_failure": _term(bool(self._reasons and self._event < 0), "dimensionless",
                     ";".join(self._last.fields[n].stamp.source for n in self._reasons if n in self._last.fields),
                     self._event if self._event < 0 and "prohibited_support" not in self._reasons else 0.),
                "prohibited_support": _term(self._last.fields["prohibited_support"].raw, "dimensionless",
                     self._last.fields["prohibited_support"].stamp.source, self._event if "prohibited_support" in self._reasons else 0.) if self.config.profile in (FULL, CURRICULUM) else _excluded(),
                "success": _term(self._event == 100., "dimensionless", "derived:independent_reward_hold_predicate",
                     self._event if self._event > 0 else 0.) if self.config.profile in (FULL, CURRICULUM) else _excluded()})
            packet = RewardPacket(token, self._request.policy_uid, self.config.profile, self.config.profile_version,
                self.config.config_sha256, SCHEMA_SHA256,
                _source_math(control_total, self._phi_pre, self._phi, self._costs, self.config.gamma, self._event, self._terminal),
                len(self._costs) / PHYSICS_HZ, self._phi_pre, self._phi, 0. if self._terminal else self._phi,
                tuple(self._costs), math.fsum(self._costs) / PHYSICS_HZ, self._event, self._reasons,
                self._last, self._context, self._terms, tuple(self._substep_terms), self._terminal, truncated, buffer_full,
                self._terminal or truncated, self._last.stamp.scope, self.config.gamma,
                self._pre_endpoint, self._pre_context, self._pre_terms, event_terms)
        except SourceFailure as exc:
            self._fail(str(exc), self._last)
        self._closed = self._terminal or truncated
        self._request = None
        return packet
