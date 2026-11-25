"""File-bound approved reward configuration and exact tensor eight-substep settlement."""
from dataclasses import dataclass as tenso_train_rewar__dataclass
from dataclasses import dataclass as train_rewar__dataclass
from types import MappingProxyType as tenso_train_rewar__MappingProxyType
from dataclasses import field
import math as tenso_train_rewar__math
from dataclasses import fields
import struct as tenso_train_rewar__struct
import hashlib
import torch
import json
import math as train_rewar__math
from pathlib import Path
import struct as train_rewar__struct
import weakref
from types import MappingProxyType as train_rewar__MappingProxyType
from typing import Mapping
from dh116.env.scene import ERROR_FIELDS as train_rewar__ERROR_FIELDS
from dh116.env.scene import FingerGoal
from dh116.env.scene import load_finger_goal
train_rewar__PHYSICS_HZ = 480
CONTROL_HZ = 60
train_rewar__WINDOW_SUBSTEPS = 8
DESIGN_SHA256 = 'a6a04d7dab20048d22b152762eb07018c3269cd63bfad77ed2b11799d7ed97d9'
SCHEMA_VERSION = 'dh116-training-reward-endpoints-v1'
train_rewar__SCHEMA_SHA256 = hashlib.sha256((SCHEMA_VERSION + ':owned-clock-measurement-request24-fixed-material-tangent-reference:fixed-required-pads:cpu-unqualified').encode()).hexdigest()
train_rewar__SUCCESS = train_rewar__MappingProxyType({'net_lift_m': 0.03, 'hold_s': 5.0, 'max_slip_m': 0.0005})
train_rewar__CONTACT = 'approach_contact_prelift_v1'
train_rewar__SMOKE = 'approach_wiring_smoke_v1'
train_rewar__FULL = 'flexible_train_r1_full'
train_rewar__GOAL_FULL = 'flexible_train_goal_full_v3'
GOAL_SHAPING_FIELDS = ('load_progress_left', 'load_progress_right', 'support_unload', 'hand_lift_load',
                       'balanced_hand_lift_load')
GOAL_TERMINATIONS = ('cable_fall', 'joint_dislocation', 'pad_backface')
# Bounded costs per second of each unit-normalized measured excess.
GOAL_SOFT_COSTS = {
    'soft_pad_depth': 0.8,
    'soft_non_target_contact': 0.4,
    'soft_wrist_tracking': 0.3,
    'soft_mimic': 0.3,
    'soft_joint_limit': 0.2,
    'soft_reaction_force': 0.4,
}
train_rewar__CURRICULUM = 'flexible_train_curriculum_v1'
_SCOPES = ('cpu_fixture', 'unqualified_source')
_FULL_PROFILES = (train_rewar__FULL, train_rewar__CURRICULUM, train_rewar__GOAL_FULL)
train_rewar___BASE_UNITS = {'distance': 'm', 'numerical_health': 'dimensionless', 'constraint_failure': 'dimensionless'}
train_rewar___FULL_UNITS = {'median_z': 'm', 'min_z': 'm', 'coverage_left': 'dimensionless', 'coverage_right': 'dimensionless', 'opposed_left': 'dimensionless', 'opposed_right': 'dimensionless', 'bilateral_support': 'dimensionless', 'off_table': 'dimensionless', 'no_external_support': 'dimensionless', 'hold_motion': 'dimensionless', 'prohibited_support': 'dimensionless'}
_CONTACT_UNITS = {name: train_rewar___FULL_UNITS[name] for name in ('coverage_left', 'coverage_right', 'opposed_left', 'opposed_right', 'bilateral_support')}
_CONTACT_UNITS['table_contact_failure'] = 'dimensionless'
train_rewar___BOOL_FIELDS = frozenset(set(train_rewar___BASE_UNITS) - {'distance'} | set(train_rewar___FULL_UNITS) - {'median_z', 'min_z', 'coverage_left', 'coverage_right'} | {'table_contact_failure'})

def _finite(value):
    """Reject nonprimitive numbers before conversion; overflow is invalid input."""
    if type(value) not in (int, float):
        return False
    try:
        return train_rewar__math.isfinite(value)
    except OverflowError:
        return False

def _solver_interval_matches(previous, current, *, native_smoke):
    step = 1 / train_rewar__PHYSICS_HZ
    interval = current.solver_time_s - previous.solver_time_s
    if train_rewar__math.isclose(interval, step, rel_tol=0.0, abs_tol=1e-09):
        return True
    if not (native_smoke and current.scope == 'unqualified_source' and (current.source in ('policy_owned_pinch_endpoint_v1', 'policy_owned_tripod_endpoint_v1', 'policy_owned_power_endpoint_v1'))):
        return False
    times = (previous.solver_time_s, current.solver_time_s)
    if any((not 0.0 <= value <= float.fromhex('0x1.fffffep+127') for value in times)):
        return False
    rounded = [train_rewar__struct.unpack('>f', train_rewar__struct.pack('>f', value))[0] for value in times]
    if tuple(rounded) != times:
        return False
    step32 = train_rewar__struct.unpack('>f', train_rewar__struct.pack('>f', step))[0]
    ulps = [max(2.0 ** (-149), train_rewar__math.ldexp(1.0, train_rewar__math.frexp(value)[1] - 24)) if value else 2.0 ** (-149) for value in times]
    error_bound = min(1e-06, 0.5 * sum(ulps) + abs(step32 - step))
    return train_rewar__math.isclose(interval, step, rel_tol=0.0, abs_tol=error_bound)

def _owned_raw(value):
    """Retained scalar/vector evidence has only immutable, finite leaves."""
    pending = [value]
    while pending:
        part = pending.pop()
        if type(part) is tuple:
            pending.extend(part)
        elif part is not None and type(part) is not bool and (not _finite(part)):
            return False
    return True

def _identity_text(value):
    """Own primitive identity text; comparison-compatible mutable objects fail."""
    return type(value) is str and bool(value.strip())

def _clip(value, low=0.0, high=1.0):
    if any((not _finite(v) for v in (value, low, high))):
        raise ValueError('finite derived value required before clipping')
    return min(high, max(low, value))

def command_rate(action, previous):
    """Mean squared normalized submitted-request delta, not energy or SDK motion."""
    if len(action) != 24 or len(previous) != 24:
        raise ValueError('normalized request24 required')
    if any((not _finite(v) or abs(v) > 1 for v in (*action, *previous))):
        raise ValueError('finite bounded normalized request required')
    return train_rewar__math.fsum((((a - b) / 2.0) ** 2 for (a, b) in zip(action, previous))) / 24.0

def smoke_potential(distance_m, distance_ref_m):
    if not _finite(distance_m) or distance_m < 0:
        raise ValueError('finite actual distance required')
    if not _finite(distance_ref_m) or distance_ref_m < 0.005:
        raise ValueError('fixed reset-derived distance scale required')
    return 2.0 * (1.0 - _clip(distance_m / distance_ref_m))

def control_total(phi_pre, phi_post, cost_per_second, gamma, event, terminated):
    """Integrate only actual 480Hz intervals; absorb the terminal potential once."""
    if not 1 <= len(cost_per_second) <= train_rewar__WINDOW_SUBSTEPS:
        raise ValueError('one to eight actual SDK samples required')
    if any((not _finite(v) for v in (phi_pre, phi_post, gamma, event, *cost_per_second))):
        raise ValueError('finite reward components required')
    if not 0 < gamma < 1 or any((v < 0 for v in cost_per_second)):
        raise ValueError('bounded gamma and nonnegative costs required')
    try:
        total = gamma * (0.0 if terminated else phi_post) - phi_pre - train_rewar__math.fsum(cost_per_second) / train_rewar__PHYSICS_HZ + event
    except OverflowError as exc:
        raise ValueError('finite integrated reward required; cost sum overflow') from exc
    if not _finite(total):
        raise ValueError('finite integrated reward required')
    return total

def slip_status(drift_uncertainty_pairs, threshold_m=0.0005):
    """Original three-valued conservative classification, independent of termination."""
    if not drift_uncertainty_pairs:
        raise ValueError('applicable anchor measurements required')
    if any((not _finite(x) or x < 0 for pair in drift_uncertainty_pairs for x in pair)):
        raise ValueError('finite nonnegative measured drift and uncertainty required')
    if any((not _finite(d + u) or not _finite(d - u) for (d, u) in drift_uncertainty_pairs)):
        raise ValueError('finite derived slip bounds required')
    if any((d - u > threshold_m for (d, u) in drift_uncertainty_pairs)):
        return False
    if all((d + u <= threshold_m for (d, u) in drift_uncertainty_pairs)):
        return True
    return None

def conservative_slip(drift_uncertainty_pairs):
    """Single pure upper-bound implementation for new endpoint consumers."""
    slip_status(drift_uncertainty_pairs)
    return max((d + u for (d, u) in drift_uncertainty_pairs))

class SourceFailure(ValueError):
    """Corrupt/unknown applicable evidence: caller must quarantine its rollout."""

def _source_math(function, *args):
    """Translate only checked reward arithmetic failures at the source boundary."""
    try:
        return function(*args)
    except (ValueError, OverflowError) as exc:
        raise SourceFailure('nonfinite derived source arithmetic: ' + str(exc)) from exc

def _source_value(value):
    if not _finite(value):
        raise SourceFailure('finite derived source value required')
    return value

@train_rewar__dataclass(frozen=True)
class SourceStamp:
    episode_uid: str
    physics_tick: int
    solver_time_s: float
    scope: str
    source: str
    frame: str = 'world'
    env_id: int = 0
    reset_uid: str = 'cpu-reset'
    rod_uid: str = 'cpu-rod'

    @property
    def clock_identity(self):
        return (self.episode_uid, self.physics_tick, self.solver_time_s, self.scope, self.frame, self.env_id, self.reset_uid, self.rod_uid)

@train_rewar__dataclass(frozen=True)
class Measurement:
    raw: float | bool | None
    unit: str
    stamp: SourceStamp
    valid: bool = True
    applicable: bool = True
    history_reason: str | None = None

@train_rewar__dataclass(frozen=True)
class SlipAnchor:
    side: str
    kind: str
    reference_uid: str
    reference: tuple
    reference_tick: int
    drift: Measurement
    uncertainty: Measurement

    def __post_init__(self):
        object.__setattr__(self, 'reference', tuple(self.reference))

@train_rewar__dataclass(frozen=True)
class RewardEndpoint:
    stamp: SourceStamp
    target_ids: tuple[str, ...]
    fields: Mapping[str, Measurement]
    anchors: tuple[SlipAnchor, ...] = ()
    required_pad_ids: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'target_ids', tuple(self.target_ids))
        object.__setattr__(self, 'fields', train_rewar__MappingProxyType(dict(self.fields)))
        object.__setattr__(self, 'anchors', tuple(self.anchors))
        object.__setattr__(self, 'required_pad_ids', tuple(((side, tuple(pads)) for (side, pads) in self.required_pad_ids)))

@train_rewar__dataclass(frozen=True)
class SubmittedRequest:
    token: str
    policy_uid: str
    stamp: SourceStamp
    values: tuple[float, ...]
    representation: str

    def __post_init__(self):
        object.__setattr__(self, 'values', tuple(self.values))

@train_rewar__dataclass(frozen=True)
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
    success: Mapping = field(default_factory=lambda : train_rewar__SUCCESS)
    finger_goal: FingerGoal | None = None
    _source_path: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ('ppo', 'weights', 'costs', 'events', 'success'):
            object.__setattr__(self, name, train_rewar__MappingProxyType(dict(getattr(self, name))))

    @property
    def gamma(self):
        return self.ppo['gamma']

def _profile_spec(profile):
    full = profile in _FULL_PROFILES
    long_rollout = full and profile != train_rewar__GOAL_FULL
    weights = {'distance': 2.0, 'contact': 2.0, 'persistence': 0.5, 'geometry': 1.0, 'lift': 4.0, 'hold': 4.0} if full else {'distance': 2.0}
    if profile == train_rewar__GOAL_FULL:
        weights.update(lift=3.2, prelift=2.4, load_progress=1.0)
    if profile == train_rewar__CONTACT:
        weights = {'distance': 2.0, 'contact': 2.0, 'persistence': 0.5, 'geometry': 1.0}
    slip_cost = 0.25 if profile == train_rewar__CURRICULUM else 1.0 if profile in (train_rewar__FULL, train_rewar__GOAL_FULL) else None
    costs = {'command_rate': 0.05, 'time': 0.1, 'slip': slip_cost, 'velocity': 0.0 if full else None}
    events = {'trusted_constraint_failure': -40.0, 'prohibited_support': -50.0 if full else None, 'success': 100.0 if full else None}
    ppo = dict(gae_lambda=0.97 if full else 0.95, rollout_steps=256 if long_rollout else 32, learning_rate=0.0001, epochs=3 if full else 1, minibatches=4 if full else 1, clip=0.2, entropy=0.005, value=0.5, max_grad_norm=0.5, target_kl=0.02, kl_backtrack_steps=3)
    return (weights, costs, events, ppo)

def _config_mapping(value, expected, name):
    if type(value) not in (dict, train_rewar__MappingProxyType) or any((type(k) is not str for k in value)) or set(value) != set(expected):
        raise ValueError('approved config mapping schema required: ' + name)
    for (key, wanted) in expected.items():
        actual = value[key]
        if wanted is None:
            valid = actual is None
        elif type(wanted) is int:
            valid = type(actual) is int and actual == wanted
        else:
            valid = _finite(actual) and actual == wanted
        if not valid:
            raise ValueError('approved primitive config value required: ' + name + '.' + key)

def _config_ppo(value, expected):
    if type(value) not in (dict, train_rewar__MappingProxyType):
        raise ValueError('approved PPO config mapping required')
    gamma = value.get('gamma')
    if not _finite(gamma) or not 0 < gamma < 1:
        raise ValueError('same finite PPO gamma required')
    _config_mapping(value, {**expected, 'gamma': gamma}, 'ppo')
    return gamma

def _sha256_text(value):
    return type(value) is str and len(value) == 64 and all((c in '0123456789abcdef' for c in value))

def _validate_reward_config(config):
    if not _identity_text(config.profile) or config.profile not in (train_rewar__SMOKE, train_rewar__FULL, train_rewar__CURRICULUM, train_rewar__CONTACT, train_rewar__GOAL_FULL):
        raise ValueError('loaded approved config profile required')
    if type(config.profile_version) is not int or config.profile_version != 1:
        raise ValueError('exact approved config version required')
    if not _sha256_text(config.config_sha256):
        raise ValueError('primitive full config SHA256 identity required')
    if not _sha256_text(config.schema_sha256) or config.schema_sha256 != train_rewar__SCHEMA_SHA256:
        raise ValueError('loaded approved config schema SHA256 required')
    if config.finger_goal is not None and (config.profile not in (train_rewar__SMOKE, train_rewar__GOAL_FULL) or type(config.finger_goal) is not FingerGoal):
        raise ValueError('finger goal requires a declared goal-conditioned profile')
    if config.profile == train_rewar__GOAL_FULL and config.finger_goal is None:
        raise ValueError('complete grasp reward requires its grasp-specific reference')
    (weights, costs, events, ppo) = _profile_spec(config.profile)
    _config_mapping(config.weights, weights, 'weights')
    _config_mapping(config.costs, costs, 'costs')
    _config_mapping(config.events, events, 'events')
    gamma = _config_ppo(config.ppo, ppo)
    slip_max = 2.0 * config.costs['slip'] if config.profile in _FULL_PROFILES else 0.0
    maximum = 0.05 + 0.1 + slip_max
    if config.profile == train_rewar__GOAL_FULL:
        maximum += sum(GOAL_SOFT_COSTS.values())
    bound = maximum / (CONTROL_HZ * (1.0 - gamma))
    if not _finite(config.cost_max_per_second) or not train_rewar__math.isclose(config.cost_max_per_second, maximum, rel_tol=0, abs_tol=1e-12):
        raise ValueError('loaded config finite cost maximum mismatch')
    if not _finite(config.discounted_cost_bound) or bound > 40.0 or (not train_rewar__math.isclose(bound, config.discounted_cost_bound, rel_tol=0, abs_tol=1e-12)):
        raise ValueError('loaded config discounted cost bound mismatch')

def _config_payload(config):
    return tuple((getattr(config, f.name) for f in fields(RewardConfig) if f.init))

def load_config(path, *, ppo_gamma=None, full_grasp=False):
    """Read actual file bytes; only this loader binds a configuration to a source."""
    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    data = json.loads(raw)
    if type(data) is not dict:
        raise ValueError('approved config JSON object required')
    if data.get('gamma_source') != 'ppo.gamma' or 'reward_gamma' in data or 'gamma' in data:
        raise ValueError('reward gamma must come only from ppo.gamma')
    profile = data.get('profile')
    if not _identity_text(profile) or profile not in (train_rewar__SMOKE, train_rewar__FULL, train_rewar__CURRICULUM, train_rewar__CONTACT, train_rewar__GOAL_FULL):
        raise ValueError('independent approved reward profile required')
    success = data.get('success')
    if profile == train_rewar__CURRICULUM:
        if type(success) is not dict or set(success) != set(train_rewar__SUCCESS) or any((not _finite(success[k]) or success[k] <= 0 for k in train_rewar__SUCCESS)):
            raise ValueError('positive finite curriculum success thresholds required')
    elif success != train_rewar__SUCCESS:
        raise ValueError('original success thresholds are immutable')
    if data.get('schema_sha256') != train_rewar__SCHEMA_SHA256 or data.get('design_document_sha256') != DESIGN_SHA256 or data.get('weight_status') != 'design_approved_native_unverified' or (type(data.get('profile_version')) is not int) or (data['profile_version'] != 1):
        raise ValueError('approved design/version/schema provenance required')
    if data.get('legacy_r1_enabled') is not False or data.get('legacy_r2_enabled') is not False:
        raise ValueError('legacy R1/R2 remain OFF')
    deadline = data.get('finite_task_deadline')
    if type(deadline) is not dict or set(deadline) != {'enabled', 'remaining_s'} or deadline['enabled'] is not False or (deadline['remaining_s'] is not None):
        raise ValueError('first profiles disable finite task deadline; external caps are truncations')
    (weights, costs, events, expected_ppo) = _profile_spec(profile)
    ppo = data.get('ppo')
    gamma = _config_ppo(ppo, expected_ppo)
    if ppo_gamma is not None and (not _finite(ppo_gamma) or ppo_gamma != gamma):
        raise ValueError('same finite PPO gamma required')
    _config_mapping(data.get('potential_weights'), weights, 'weights')
    _config_mapping(data.get('costs_per_second'), costs, 'costs')
    _config_mapping(data.get('events'), events, 'events')
    if full_grasp and profile != train_rewar__GOAL_FULL:
        if profile != train_rewar__SMOKE or data.get('enable_full_grasp') is not True:
            raise ValueError('current grasp configuration must explicitly enable full reward')
        profile = train_rewar__GOAL_FULL
        weights, costs, events, expected_ppo = _profile_spec(profile)
        overrides = {name: expected_ppo[name] for name in ('gae_lambda', 'epochs', 'minibatches')}
        _config_mapping(data.get('full_grasp_ppo'), overrides, 'full_grasp_ppo')
        ppo = {**ppo, **overrides}
        _config_ppo(ppo, expected_ppo)
        raw += b'\0' + profile.encode()
    slip_max = 2.0 * costs['slip'] if profile in _FULL_PROFILES else 0.0
    maximum = 0.05 + 0.1 + slip_max
    if profile == train_rewar__GOAL_FULL:
        maximum += sum(GOAL_SOFT_COSTS.values())
    bound = maximum / (CONTROL_HZ * (1.0 - gamma))
    if abs(events['trusted_constraint_failure']) < bound:
        raise ValueError('failure penalty does not cover discounted cost bound; recompute approved bounds')
    goal = load_finger_goal(data['finger_goal']) if 'finger_goal' in data else None
    result = RewardConfig(profile, 1, ppo, weights, costs, events, hashlib.sha256(raw).hexdigest(), train_rewar__SCHEMA_SHA256, maximum, bound, success, goal)
    _validate_reward_config(result)
    object.__setattr__(result, '_source_path', str(source_path))
    return result

def _term(raw, unit, source, contribution=0.0, *, applicable=True, history_reason=None, valid=True):
    if not _owned_raw(raw) or not _finite(contribution):
        raise SourceFailure('finite owned term raw and contribution required')
    return train_rewar__MappingProxyType(dict(raw=raw, unit=unit, valid=valid, applicable=applicable, source=source, contribution=contribution, history_reason=history_reason))

def _excluded():
    return _term(None, 'dimensionless', 'profile_excluded', applicable=False, history_reason='profile_excluded')

@train_rewar__dataclass(frozen=True)
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
    def native_ready(self):
        return False

    @property
    def teacher_admission(self):
        return False

    @property
    def native_qualification(self):
        return False

    def bootstrap(self, shaped_value):
        if not _finite(shaped_value):
            raise ValueError('finite shaped value from actual post_context required')
        return 0.0 if self.terminated else self.gamma * shaped_value

def validate_reward_config(config):
    """Repeat the original file-backed constructor checks without an episode owner."""
    if type(config) is not RewardConfig:
        raise TypeError('loaded RewardConfig required')
    _validate_reward_config(config)
    if not _identity_text(config._source_path) or not Path(config._source_path).is_absolute():
        raise ValueError('actual file-backed loader provenance required')
    try:
        actual = load_config(config._source_path, ppo_gamma=config.gamma,
                             full_grasp=config.profile == train_rewar__GOAL_FULL)
    except (OSError, ValueError) as exc:
        raise ValueError('reward config source file no longer validates against loaded bytes') from exc
    if _config_payload(config) != _config_payload(actual):
        raise ValueError('reward config source bytes/SHA/content mismatch')
    return actual
tenso_train_rewar__validate_reward_config = validate_reward_config
tenso_train_rewar__SourceFailure = SourceFailure
tenso_train_rewar__SMOKE = train_rewar__SMOKE
tenso_train_rewar__FULL = train_rewar__FULL
tenso_train_rewar__CURRICULUM = train_rewar__CURRICULUM
tenso_train_rewar__CONTACT = train_rewar__CONTACT
tenso_train_rewar__SUCCESS = train_rewar__SUCCESS
tenso_train_rewar__SCHEMA_SHA256 = train_rewar__SCHEMA_SHA256
tenso_train_rewar__PHYSICS_HZ = train_rewar__PHYSICS_HZ
tenso_train_rewar__WINDOW_SUBSTEPS = train_rewar__WINDOW_SUBSTEPS
tenso_train_rewar___BASE_UNITS = train_rewar___BASE_UNITS
tenso_train_rewar___FULL_UNITS = train_rewar___FULL_UNITS
tenso_train_rewar___BOOL_FIELDS = train_rewar___BOOL_FIELDS
tenso_train_rewar__ERROR_FIELDS = train_rewar__ERROR_FIELDS
_STEP32 = tenso_train_rewar__struct.unpack('>f', tenso_train_rewar__struct.pack('>f', 1.0 / tenso_train_rewar__PHYSICS_HZ))[0]

@tenso_train_rewar__dataclass(frozen=True)
class TensorLaneSchema:
    """Actual owner's immutable identity and reader declarations for one lane.

    field_specs contains (name, unit, reader_source), including proven-NA slip
    in FULL. anchor_specs is ordered left/material, left/tangential,
    right/material, right/tangential and contains (reference_uid, drift_reader,
    uncertainty_reader). It may first become available at a window boundary;
    once declared it cannot change. Numeric anchor reference coordinates/ticks
    are consumed from the frame and latched at their first actual observation.
    """
    episode_uid: str
    env_id: int
    reset_uid: str
    rod_uid: str
    scope: str
    source: str
    target_ids: tuple
    field_specs: tuple
    required_pad_ids: tuple = ()
    anchor_specs: tuple = ()
    frame: str = 'world'

@tenso_train_rewar__dataclass(frozen=True)
class TensorRewardFrame:
    """One actual 480 Hz observation; tensors have leading dimension N.

    fields supplies each declared finite scalar/boolean field except FULL's
    separately represented slip. FULL requires explicit anchor_present[N] and
    drift/uncertainty[N,4], reference[N,4,K], reference_tick[N,4]. Absent anchor
    entries may contain NaN; absence is proven only before any latched anchor
    and before bilateral contact, never substituted with numeric zero.
    """
    physics_tick: torch.Tensor
    solver_time_s: torch.Tensor
    fields: dict
    anchor_present: torch.Tensor | None = None
    anchor_drift: torch.Tensor | None = None
    anchor_uncertainty: torch.Tensor | None = None
    anchor_reference: torch.Tensor | None = None
    anchor_reference_tick: torch.Tensor | None = None

@tenso_train_rewar__dataclass(frozen=True)
class TensorRewardPacket:
    tokens: tuple
    policy_uids: tuple
    config_sha256: str
    schema_sha256: str
    profile: str
    profile_version: int
    gamma: float
    reward: torch.Tensor
    duration_s: torch.Tensor
    phi_pre: torch.Tensor
    phi_post_raw: torch.Tensor
    endpoint_phi: torch.Tensor
    cost_per_second: torch.Tensor
    integrated_cost: torch.Tensor
    event: torch.Tensor
    reason_masks: object
    substeps: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    buffer_full: torch.Tensor
    done: torch.Tensor
    post_context: object
    terms: object
    pre_context: object
    pre_terms: object
    substep_terms: object
    event_terms: object
    schemas: tuple

    @property
    def native_ready(self):
        return False

    @property
    def teacher_admission(self):
        return False

    @property
    def native_qualification(self):
        return False

    def bootstrap(self, shaped_value):
        if not isinstance(shaped_value, torch.Tensor) or shaped_value.shape != self.reward.shape or shaped_value.device != self.reward.device or (not shaped_value.is_floating_point()) or bool((~torch.isfinite(shaped_value)).any()):
            raise ValueError('finite shaped value from actual post_context required')
        return torch.where(self.terminated, 0.0, self.gamma * shaped_value)

    def event_reasons(self, lane):
        """Explicit cold-path materialization, never called by settlement."""
        return tuple((name for (name, mask) in self.reason_masks.items() if bool(mask[lane])))

def measured_finger_errors(actual_active_q, finger_goal):
    """Named active SDK q [N,2,6] -> max participating errors [N,2].

    The owner must gather the six ACTIVE joints in their original named order.
    Validation of finite/nonnegative outputs is fused into observe/reset guard.
    """
    if not isinstance(actual_active_q, torch.Tensor) or actual_active_q.ndim != 3 or tuple(actual_active_q.shape[1:]) != (2, 6) or (not actual_active_q.is_floating_point()):
        raise ValueError('actual named active q [N,2,6] required')
    q = actual_active_q.to(dtype=torch.float64)
    target = torch.tensor(finger_goal.target_rad, device=q.device, dtype=q.dtype)
    error = (q - target).abs()[:, :, finger_goal.participating].amax(dim=2)
    return torch.where(torch.isfinite(q).all(dim=(1, 2))[:, None], error, torch.nan)

def _text(value):
    return type(value) is str and bool(value.strip())

class TensorTrainingReward:
    """Exactly once batch windows, with reset-selected lane state on device.

    reset_selected(ids, schemas, frame, previous_request=..., tokens=...,
                   policy_uids=...) consumes only K selected reset observations.
    begin(schemas, request=[N,24], tokens=..., policy_uids=...,
          physics_tick=[N], solver_time_s=[N], remaining_substeps=...) reserves
    all N lanes against the actual PRE clock. observe(frame) consumes one real
    interval for each nonterminal lane and returns a device readiness mask.
    settle_once(tokens, truncated=[N], buffer_full=[N]) returns an owned packet.

    remaining_substeps is a Python int >= 8 or actual int64 tensor [N]. Token
    and episode identity sets persist across selected resets. A corrupt frame
    quarantines the batch; selected reset can recover only the reset lanes.
    """

    def __init__(self, config, *, num_envs, device='cpu', defer_checks=False):
        self.config = tenso_train_rewar__validate_reward_config(config)
        self.defer_checks = defer_checks
        self._source_terms = {}
        if self.config.profile == tenso_train_rewar__CONTACT:
            raise tenso_train_rewar__SourceFailure('CONTACT requires original live contact owner; tensor attestation unavailable')
        if dict(self.config.success) != dict(tenso_train_rewar__SUCCESS):
            raise ValueError('original success thresholds required; no relaxed tensor success')
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError('positive batch size required')
        (self.n, self.device) = (num_envs, torch.device(device))
        self.full = self.config.profile in _FULL_PROFILES
        self.schemas = [None] * num_envs
        self._tokens = [set() for _ in range(num_envs)]
        self._episodes = [set() for _ in range(num_envs)]
        self._pending = None
        self._poisoned_lanes = set()
        self._policies = None
        self._f = lambda value=0.0: torch.full((num_envs,), value, device=self.device, dtype=torch.float64)
        self._b = lambda value=False: torch.full((num_envs,), value, device=self.device, dtype=torch.bool)
        self._i = lambda value=0: torch.full((num_envs,), value, device=self.device, dtype=torch.int64)
        (self._tick, self._time) = (self._i(), self._f())
        self._reset_tick = self._i()
        (self._dref, self._median, self._min) = (self._f(tenso_train_rewar__math.nan), self._f(tenso_train_rewar__math.nan), self._f(tenso_train_rewar__math.nan))
        (self._contact_start, self._hold_start) = (self._i(-1), self._i(-1))
        (self._contact_s, self._hold_s) = (self._f(), self._f())
        (self._anchor_latched, self._ever_bilateral) = (self._b(), self._b())
        self._anchor_ref = None
        self._anchor_tick = torch.full((num_envs, 4), -1, dtype=torch.int64, device=self.device)
        self._anchor_drift = torch.full((num_envs, 4), tenso_train_rewar__math.nan, device=self.device, dtype=torch.float64)
        self._anchor_uncertainty = self._anchor_drift.clone()
        (self._slip, self._slip_status) = (self._f(tenso_train_rewar__math.nan), self._i(-1))
        (self._dwell, self._arrived) = (self._i(), self._b())
        self._previous = torch.full((num_envs, 24), tenso_train_rewar__math.nan, device=self.device, dtype=torch.float64)
        (self._previous_tokens, self._previous_policies) = ([None] * num_envs, [None] * num_envs)
        (self._phi, self._closed, self._quarantined) = (self._f(tenso_train_rewar__math.nan), self._b(True), self._b())
        self._terminal = self._b()
        self._terms = {}
        self._last_values = {}

    @property
    def native_ready(self):
        return False

    @property
    def teacher_admission(self):
        return False

    @property
    def native_qualification(self):
        return False

    @property
    def quarantined(self):
        return self._quarantined.clone()

    @property
    def post_context(self):
        return self._context()

    def _tensor(self, value, shape, *, boolean=False, integer=False):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.device != self.device or value.requires_grad or (boolean and value.dtype != torch.bool) or (integer and value.dtype != torch.int64) or (not boolean and (not integer) and (not value.is_floating_point())):
            raise tenso_train_rewar__SourceFailure('actual device tensor shape/dtype/ownership mismatch')
        return value if boolean or integer else value.to(torch.float64)

    def _guard(self, bad, message):
        if bool(bad.any()):
            self._quarantined[:] = True
            self._poisoned_lanes = set(range(self.n))
            raise tenso_train_rewar__SourceFailure(message)

    def _schema(self, schemas, ids, *, reset):
        if type(schemas) is not tuple or len(schemas) != len(ids):
            raise tenso_train_rewar__SourceFailure('one immutable actual lane schema per selected lane required')
        previous_names = None
        for (lane, s) in zip(ids, schemas):
            if type(s) is not TensorLaneSchema or any((not _text(v) for v in (s.episode_uid, s.reset_uid, s.rod_uid, s.scope, s.source))) or type(s.env_id) is not int or (s.env_id < 0) or (type(s.frame) is not str) or (s.frame != 'world') or (s.scope not in ('cpu_fixture', 'unqualified_source')) or (type(s.target_ids) is not tuple) or (not s.target_ids) or any((not _text(v) for v in s.target_ids)) or (len(set(s.target_ids)) != len(s.target_ids)) or (type(s.field_specs) is not tuple) or any((type(v) is not tuple or len(v) != 3 or any((not _text(x) for x in v)) for v in s.field_specs)):
                raise tenso_train_rewar__SourceFailure('immutable original identity/source/material schema required')
            names = tuple((name for (name, unit, source) in s.field_specs))
            if len(set(names)) != len(names) or (previous_names is not None and names != previous_names):
                raise tenso_train_rewar__SourceFailure('homogeneous field schema required')
            previous_names = names
            units = dict(tenso_train_rewar___BASE_UNITS)
            if self.full:
                units.update(tenso_train_rewar___FULL_UNITS)
            units.update(((name, 'dimensionless') for name in names if name.startswith('original_guard_')))
            if self.config.finger_goal is not None:
                units.update(((name, 'rad') for name in tenso_train_rewar__ERROR_FIELDS))
            if self.config.profile == train_rewar__GOAL_FULL:
                units.update(dict.fromkeys(GOAL_SHAPING_FIELDS, 'dimensionless'))
                units.update(soft_non_target_contact='dimensionless', soft_reaction_force='dimensionless')
                if s.scope == 'unqualified_source':
                    units.update(wrist_goal_m='m', wrist_goal_rad='rad')
                    units.update(dict.fromkeys((*GOAL_SOFT_COSTS, *GOAL_TERMINATIONS), 'dimensionless'))
            if any((name not in names for name in units)):
                raise tenso_train_rewar__SourceFailure('missing applicable original reward field')
            if any((name in units and unit != units[name] for (name, unit, source) in s.field_specs)):
                raise tenso_train_rewar__SourceFailure('original source units mismatch')
            if self.full:
                pads = s.required_pad_ids
                if type(pads) is not tuple or len(pads) != 2 or {x[0] for x in pads} != {'left', 'right'} or any((type(p) is not tuple or not p or any((not _text(v) for v in p)) or (len(set(p)) != len(p)) for (side, p) in pads)) or ('slip' not in names) or (dict(((n, u) for (n, u, _) in s.field_specs))['slip'] != 'm'):
                    raise tenso_train_rewar__SourceFailure('fixed actual pad denominator and slip history schema required')
            if type(s.anchor_specs) is not tuple or len(s.anchor_specs) not in (0, 4) or any((type(v) is not tuple or len(v) != 3 or any((not _text(x) for x in v)) for v in s.anchor_specs)):
                raise tenso_train_rewar__SourceFailure('fixed four anchor reader/UID schema required')
            if not reset:
                prior = self.schemas[lane]
                if prior is None or any((getattr(prior, k) != getattr(s, k) for k in ('episode_uid', 'env_id', 'reset_uid', 'rod_uid', 'scope', 'source', 'frame', 'target_ids', 'field_specs', 'required_pad_ids'))):
                    raise tenso_train_rewar__SourceFailure('episode identity/source/schema changed before window')
                if prior.anchor_specs and prior.anchor_specs != s.anchor_specs:
                    raise tenso_train_rewar__SourceFailure('fixed anchor UID/source changed; reanchoring forbidden')

    def _frame(self, frame, schemas=None, *, field_names=None, anchor_available=None):
        k = self.n if schemas is None else len(schemas)
        if type(frame) is not TensorRewardFrame or type(frame.fields) is not dict:
            raise tenso_train_rewar__SourceFailure('actual batched frame required')
        tick = self._tensor(frame.physics_tick, (k,), integer=True)
        time = self._tensor(frame.solver_time_s, (k,))
        names = field_names if field_names is not None else {n for (n, _, _) in schemas[0].field_specs} - ({'slip'} if self.full else set())
        if set(frame.fields) != names:
            raise tenso_train_rewar__SourceFailure('actual value bundle differs from prevalidated schema')
        values = {n: self._tensor(v, (k,), boolean=n in tenso_train_rewar___BOOL_FIELDS or n in GOAL_TERMINATIONS or n.startswith('original_guard_')) for (n, v) in frame.fields.items()}
        bad = (tick < 0) | ~torch.isfinite(time)
        for value in values.values():
            if value.dtype != torch.bool:
                bad |= ~torch.isfinite(value)
        bad |= values['distance'] < 0
        if self.config.finger_goal is not None:
            for n in tenso_train_rewar__ERROR_FIELDS:
                bad |= values[n] < 0
        anchors = None
        if self.full:
            present = self._tensor(frame.anchor_present, (k,), boolean=True)
            drift = self._tensor(frame.anchor_drift, (k, 4))
            uncertainty = self._tensor(frame.anchor_uncertainty, (k, 4))
            if frame.anchor_reference is None or frame.anchor_reference.ndim != 3 or frame.anchor_reference.shape[2] < 1:
                raise tenso_train_rewar__SourceFailure('actual fixed anchor reference coordinates required')
            ref = self._tensor(frame.anchor_reference, (k, 4, frame.anchor_reference.shape[2]))
            rt = self._tensor(frame.anchor_reference_tick, (k, 4), integer=True)
            available = torch.tensor([bool(s.anchor_specs) for s in schemas], device=self.device) if anchor_available is None else anchor_available
            bad |= present & (~available | (~torch.isfinite(drift) | ~torch.isfinite(uncertainty) | (drift < 0) | (uncertainty < 0) | ~torch.isfinite(drift + uncertainty) | ~torch.isfinite(drift - uncertainty)).any(dim=1) | ~torch.isfinite(ref).all(dim=(1, 2)))
            for n in ('coverage_left', 'coverage_right', *GOAL_SHAPING_FIELDS, *GOAL_SOFT_COSTS):
                if n in values:
                    bad |= (values[n] < 0) | (values[n] > 1)
            anchors = (present, drift, uncertainty, ref, rt)
        return (tick, time, values, anchors, bad)

    def _shape_potential(self, ids, v, state, *, reset=False):
        distance_ratio = v['distance'] / self._dref[ids]
        terms = {'distance': 2.0 * (1.0 - distance_ratio.clamp(0.0, 1.0))}
        if self.config.finger_goal is not None:
            goal = self.config.finger_goal
            errors = torch.stack([v[name] for name in tenso_train_rewar__ERROR_FIELDS], dim=1)
            pg = torch.exp(-(errors - goal.tolerance_rad).clamp_min(0.0) / goal.scale_rad).mean(1)
            if self.config.profile == train_rewar__GOAL_FULL:
                if 'wrist_goal_m' in v:
                    wrist = torch.exp(-v['wrist_goal_m'] / 0.025 - v['wrist_goal_rad'] / 0.35)
                    pg = (pg + wrist) * 0.5
                if not reset:
                    pg = torch.where((self._arrived | self._arrival_pending)[ids], 1.0, pg)
            terms['finger_goal'] = pg
        if self.full:
            success = self.config.success
            md = v['median_z'] - self._median[ids]
            mn = v['min_z'] - self._min[ids]
            if self.config.profile == train_rewar__CURRICULUM:
                lift_ok = v['bilateral_support'] & v['no_external_support'] & ~v['prohibited_support']
                pl = torch.where(lift_ok, ((md + mn) * 0.5 / success['net_lift_m']).clamp(0, 1), 0)
            elif self.config.profile == train_rewar__GOAL_FULL:
                net_lift = torch.minimum(md, mn) / success['net_lift_m']
                transfer = torch.minimum(v['support_unload'], v['hand_lift_load'])
                carrying = transfer.masked_fill(~(self._opposed_contact(v) & v['bilateral_support'])
                                                 | v['prohibited_support'], 0)
                pl = carrying * net_lift.clamp(0, 1)
                left, right = v['load_progress_left'], v['load_progress_right']
                weaker_hand = torch.minimum(left, right)
                # Transfer can start while the sagging rod is below its reset height.
                clean = (v['off_table'] & ~v['prohibited_support']
                         & (v['soft_non_target_contact'] == 0) & (v['soft_reaction_force'] < 1))
                recovery = (1 + net_lift).clamp(0, 1)
                balanced_transfer = torch.minimum(v['support_unload'], v['balanced_hand_lift_load'])
                prelift = (weaker_hand * balanced_transfer * recovery).masked_fill(~clean, 0)
                terms['prelift'] = self.config.weights['prelift'] * prelift
                bilateral_progress = (left + right + 2 * weaker_hand) * 0.25
                terms['load_progress'] = self.config.weights['load_progress'] * bilateral_progress
            else:
                pl = torch.where(self._genuine_contact(v),
                                 (torch.minimum(md, mn) / success['net_lift_m']).clamp(0, 1), 0)
            pc = (v['coverage_left'] + v['coverage_right']) * 0.5
            geom = (v['opposed_left'].double() + v['opposed_right'].double()) * 0.5
            terms.update(
                contact=2.0 * pc, persistence=0.5 * (state['_contact_s'] / 0.25).clamp(0, 1),
                geometry=geom, lift=self.config.weights['lift'] * pl,
                hold=4.0 * (state['_hold_s'] / success['hold_s']).clamp(0, 1),
            )
        return sum(terms.values()), terms

    @staticmethod
    def _opposed_contact(values):
        return ((values['coverage_left'] == 1) & (values['coverage_right'] == 1)
                & values['opposed_left'] & values['opposed_right'])

    def _genuine_contact(self, values):
        return (self._opposed_contact(values) & values['bilateral_support'] & values['off_table']
                & values['no_external_support'] & ~values['prohibited_support'])

    def _potential(self, ids, tick, v, anchors, *, reset=False, shape=True):
        state = {'_derived_bad': torch.zeros_like(tick, dtype=torch.bool)}
        if self.full:
            present, d, u, ref, rt = anchors
            contact = self._opposed_contact(v)
            if self.config.profile == train_rewar__GOAL_FULL:
                contact &= present | v['bilateral_support']
            cs = torch.where(contact, torch.where(self._contact_start[ids] < 0, tick, self._contact_start[ids]), -1)
            contact_s = torch.where(cs < 0, 0.0, (tick - cs).double() / tenso_train_rewar__PHYSICS_HZ)
            slip_observed = (present & contact & v['bilateral_support']
                             if self.config.profile == train_rewar__GOAL_FULL else present)
            slip = torch.where(slip_observed, (d + u).amax(dim=1), torch.nan)
            slip_false = (d - u > 0.0005).any(dim=1)
            slip_true = (d + u <= 0.0005).all(dim=1)
            status = torch.where(slip_observed, torch.where(slip_false, 0, torch.where(slip_true, 1, -1)), -1)
            md, mn = v['median_z'] - self._median[ids], v['min_z'] - self._min[ids]
            success = self.config.success
            valid_step = v['hold_motion'] & v['numerical_health'] & ~v['constraint_failure']
            if self.config.profile == train_rewar__GOAL_FULL:
                # Exploration pays the two soft costs. The continuous Hold interval
                # requires clean non-target contact and unsaturated raw hand force.
                valid_step &= (v['soft_non_target_contact'] == 0) & (v['soft_reaction_force'] < 1.0)
            if self.config.profile == train_rewar__CURRICULUM:
                lift_ok = v['bilateral_support'] & v['no_external_support'] & ~v['prohibited_support']
                hold = lift_ok & ((md + mn) * 0.5 >= success['net_lift_m']) & valid_step & (status != 0)
                hs = self._hold_start[ids]
                hold_s = torch.where(hold, self._hold_s[ids] + 1.0 / tenso_train_rewar__PHYSICS_HZ,
                                     (self._hold_s[ids] - 2.0 / tenso_train_rewar__PHYSICS_HZ).clamp_min(0.0))
            else:
                hold = (self._genuine_contact(v) & (torch.minimum(md, mn) >= success['net_lift_m'])
                        & valid_step & (status == 1))
                hs = torch.where(hold, torch.where(self._hold_start[ids] < 0, tick, self._hold_start[ids]), -1)
                hold_s = torch.where(hs < 0, 0.0, (tick - hs).double() / tenso_train_rewar__PHYSICS_HZ)
            state.update(_contact_start=cs, _contact_s=contact_s, _hold_start=hs, _hold_s=hold_s,
                         _slip=slip, _slip_status=status, _ever_bilateral=self._ever_bilateral[ids] | contact,
                         _anchor_latched=self._anchor_latched[ids] | present)
            bad = ~torch.isfinite(md) | ~torch.isfinite(mn) | ~torch.isfinite(hold_s) | ~torch.isfinite(contact_s)
            bad |= slip_observed & ~torch.isfinite(slip / success['max_slip_m'])
            bad |= ~present & (self._anchor_latched[ids] | self._ever_bilateral[ids] | contact)
            bad |= present & ((rt < self._reset_tick[ids, None]) | (rt > tick[:, None])).any(dim=1)
            if self._anchor_ref is not None:
                if self._anchor_ref.shape[2] != ref.shape[2]:
                    raise tenso_train_rewar__SourceFailure('fixed reference vector schema changed')
                bad |= present & self._anchor_latched[ids] & (
                    (ref != self._anchor_ref[ids]).any(dim=(1, 2)) | (rt != self._anchor_tick[ids]).any(dim=1))
            state['_derived_bad'] = bad
        if not shape:
            return None, {}, state
        phi, terms = self._shape_potential(ids, v, state, reset=reset)
        return phi, terms, state

    def _commit(self, ids, active, phi, terms, state, tick, time, values, anchors):

        def put(target, value):
            target[ids] = torch.where(active, value, target[ids])
        for (name, value) in state.items():
            if name != '_derived_bad':
                put(getattr(self, name), value)
        if anchors is not None:
            (present, d, u, ref, rt) = anchors
            if self._anchor_ref is None:
                self._anchor_ref = torch.full((self.n, 4, ref.shape[2]), tenso_train_rewar__math.nan, device=self.device, dtype=torch.float64)
            mask = (active & present)[:, None, None]
            self._anchor_ref[ids] = torch.where(mask, ref, self._anchor_ref[ids])
            self._anchor_tick[ids] = torch.where((active & present)[:, None], rt, self._anchor_tick[ids])
            self._anchor_drift[ids] = torch.where((active & present)[:, None], d, self._anchor_drift[ids])
            self._anchor_uncertainty[ids] = torch.where((active & present)[:, None], u, self._anchor_uncertainty[ids])
        if phi is not None:
            put(self._phi, phi)
        put(self._tick, tick)
        put(self._time, time)
        for (name, value) in terms.items():
            if name not in self._terms:
                self._terms[name] = self._f(tenso_train_rewar__math.nan)
            put(self._terms[name], value)
        for (name, value) in values.items():
            if name not in self._last_values:
                self._last_values[name] = self._b() if value.dtype == torch.bool else self._f(tenso_train_rewar__math.nan)
            put(self._last_values[name], value)

    def reset_selected(self, lane_ids, schemas, frame, *, previous_request, tokens, policy_uids):
        if self._pending is not None and (not self._poisoned_lanes):
            raise RuntimeError('cannot reset unfinished window')
        if self._poisoned_lanes:
            self._pending = None
            self._policies = None
        ids = self._ids(lane_ids)
        self._schema(schemas, lane_ids, reset=True)
        self._identities(tokens, policy_uids, len(lane_ids))
        self._source_terms.clear()
        for (lane, s, token) in zip(lane_ids, schemas, tokens):
            if s.episode_uid in self._episodes[lane]:
                raise RuntimeError('episode reset UID must be new')
        (tick, time, values, anchors, bad) = self._frame(frame, schemas)
        previous = self._tensor(previous_request, (len(lane_ids), 24))
        bad |= (~torch.isfinite(previous) | (previous.abs() > 1)).any(dim=1)
        bad |= ~values['numerical_health'] | values['constraint_failure']
        for (n, v) in values.items():
            if n.startswith('original_guard_'):
                bad |= v
        self._guard(bad, 'reset numerical/health/source validation failed')
        self._reset_tick[ids] = tick
        self._dref[ids] = values['distance'].clamp_min(0.005)
        for name in ('_contact_start', '_hold_start'):
            getattr(self, name)[ids] = -1
        for name in ('_contact_s', '_hold_s', '_dwell'):
            getattr(self, name)[ids] = 0
        for name in ('_anchor_latched', '_ever_bilateral', '_arrived'):
            getattr(self, name)[ids] = False
        self._slip[ids] = tenso_train_rewar__math.nan
        self._slip_status[ids] = -1
        self._anchor_tick[ids] = -1
        self._anchor_drift[ids] = tenso_train_rewar__math.nan
        self._anchor_uncertainty[ids] = tenso_train_rewar__math.nan
        if self._anchor_ref is not None:
            self._anchor_ref[ids] = tenso_train_rewar__math.nan
        if self.full:
            self._median[ids] = values['median_z']
            self._min[ids] = values['min_z']
        (phi, terms, state) = self._potential(ids, tick, values, anchors, reset=True)
        self._guard(~torch.isfinite(phi) | state.get('_derived_bad', torch.zeros_like(bad)), 'reset finite derived source required')
        active = torch.ones_like(bad)
        self._commit(ids, active, phi, terms, state, tick, time, values, anchors)
        self._previous[ids] = previous
        self._closed[ids] = False
        self._terminal[ids] = False
        self._quarantined[ids] = False
        self._poisoned_lanes.difference_update(lane_ids)
        self._previous_tokens = list(self._previous_tokens)
        self._previous_policies = list(self._previous_policies)
        for (lane, s, token, policy) in zip(lane_ids, schemas, tokens, policy_uids):
            self.schemas[lane] = s
            self._episodes[lane].add(s.episode_uid)
            self._tokens[lane].add(token)
            self._previous_tokens[lane] = token
            self._previous_policies[lane] = policy

    def _ids(self, ids):
        if type(ids) is not tuple or not ids or any((type(i) is not int or not 0 <= i < self.n for i in ids)) or (len(set(ids)) != len(ids)):
            raise ValueError('unique selected lane IDs required')
        return torch.tensor(ids, device=self.device, dtype=torch.int64)

    @staticmethod
    def _identities(tokens, policies, n):
        if type(tokens) is not tuple or type(policies) is not tuple or len(tokens) != n or (len(policies) != n) or any((not _text(v) for v in tokens + policies)):
            raise tenso_train_rewar__SourceFailure('explicit immutable token and policy identities required')

    def begin(self, schemas, *, request, tokens, policy_uids, physics_tick, solver_time_s, remaining_substeps):
        if self._poisoned_lanes:
            raise tenso_train_rewar__SourceFailure('quarantined rollout requires verified reset')
        if self._pending is not None:
            raise RuntimeError('prior token has unfinished window')
        if any((s is None for s in self.schemas)):
            raise RuntimeError('active verified reset required')
        try:
            self._schema(schemas, tuple(range(self.n)), reset=False)
            self._identities(tokens, policy_uids, self.n)
            action = self._tensor(request, (self.n, 24))
            tick = self._tensor(physics_tick, (self.n,), integer=True)
            time = self._tensor(solver_time_s, (self.n,))
        except tenso_train_rewar__SourceFailure:
            self._quarantined[:] = True
            self._poisoned_lanes = set(range(self.n))
            raise
        for (lane, token) in enumerate(tokens):
            if token in self._tokens[lane]:
                raise RuntimeError('request token already reserved/settled')
        if type(remaining_substeps) is int:
            if remaining_substeps < tenso_train_rewar__WINDOW_SUBSTEPS:
                raise ValueError('eight (8) budgeted substeps required')
            budget_bad = self._b()
        else:
            budget = self._tensor(remaining_substeps, (self.n,), integer=True)
            budget_bad = budget < tenso_train_rewar__WINDOW_SUBSTEPS
        bad = (~torch.isfinite(action) | (action.abs() > 1)).any(dim=1) | ~torch.isfinite(time) | (tick != self._tick) | (time != self._time)
        bad |= self._quarantined | self._closed | budget_bad
        self._guard(bad, 'quarantined/closed, budget or actual PRE clock/request validation failed')
        self.schemas = list(schemas)
        self._window_fields = {n for (n, _, _) in schemas[0].field_specs} - ({'slip'} if self.full else set())
        self._window_anchors = torch.tensor([bool(s.anchor_specs) for s in schemas], device=self.device)
        self._native_float32_clock = torch.tensor([s.scope == 'unqualified_source' and s.source in ('policy_owned_pinch_endpoint_v1', 'policy_owned_tripod_endpoint_v1', 'policy_owned_power_endpoint_v1') for s in schemas], device=self.device)
        (self._pending, self._policies) = (tokens, policy_uids)
        for (lane, token) in enumerate(tokens):
            self._tokens[lane].add(token)
        self._request = action.clone()
        self._qcmd = (((action - self._previous) / 2.0) ** 2).sum(dim=1) / 24.0
        self._phi_pre = self._phi.clone()
        self._pre_context = self._context()
        self._pre_terms = self._term_payload()
        (self._count, self._terminal, self._event) = (self._i(), self._b(), self._f())
        self._arrival_pending = self._b()
        self._costs = torch.full((self.n, 8), tenso_train_rewar__math.nan, device=self.device, dtype=torch.float64)
        self._slip_history = torch.full_like(self._costs, tenso_train_rewar__math.nan)
        self._reasons = {name: self._b() for name in ('numerical_health', 'constraint_failure')}
        self._reasons.update(((n, self._b()) for (n, _, _) in schemas[0].field_specs if n.startswith('original_guard_')))
        if self.full:
            self._reasons.update(prohibited_support=self._b(), success=self._b())
        self._soft_fields = tuple(k for k in GOAL_SOFT_COSTS if k in self._window_fields)
        self._soft_history = {k: torch.zeros_like(self._costs) for k in self._soft_fields}
        for key in GOAL_TERMINATIONS:
            if key in self._window_fields:
                self._reasons[key] = self._b()
        self._window_bad = self._b()
        self._observations_finalized = False
        self._observe_calls = 0

    def _interval(self, time):
        interval = time - self._time
        match = (interval - 1.0 / tenso_train_rewar__PHYSICS_HZ).abs() <= 1e-09
        if self.config.profile == tenso_train_rewar__SMOKE:
            native = self._native_float32_clock
            both = torch.stack((self._time, time), dim=1)
            f32 = both.to(torch.float32).to(torch.float64)
            (_, exp) = torch.frexp(both)
            ulp = torch.maximum(torch.full_like(both, 2.0 ** (-149)), torch.ldexp(torch.ones_like(both), exp - 24))
            ulp = torch.where(both == 0, 2.0 ** (-149), ulp)
            bound = (0.5 * ulp.sum(dim=1) + abs(_STEP32 - 1.0 / tenso_train_rewar__PHYSICS_HZ)).clamp_max(1e-06)
            match |= native & ((both >= 0) & (both <= float.fromhex('0x1.fffffep+127')) & (both == f32)).all(dim=1) & ((interval - 1.0 / tenso_train_rewar__PHYSICS_HZ).abs() <= bound)
        if self.config.profile == train_rewar__GOAL_FULL:
            # The installed RopeBatch stores time in float; accept its exact
            # representable next time after the original one-step addition.
            current_f32 = self._time.float()
            next_f32 = (current_f32 + _STEP32).double()
            match |= self._native_float32_clock & (self._time == current_f32.double()) & (
                (time == next_f32) & (next_f32 > self._time)
            )
        return match

    def observe(self, frame):
        if self._poisoned_lanes:
            raise tenso_train_rewar__SourceFailure('quarantined rollout has no usable PPO packet')
        if self._pending is None or self._observe_calls >= 8:
            raise RuntimeError('no open physical interval; terminal/full window cannot advance')
        try:
            (tick, time, v, anchors, bad) = self._frame(frame, field_names=self._window_fields, anchor_available=self._window_anchors)
            active = ~self._terminal & (self._count < 8)
            if self.config.profile == train_rewar__GOAL_FULL:
                bad |= ~v['numerical_health']
            bad |= (tick != self._tick + 1) | ~self._interval(time)
            ids = torch.arange(self.n, device=self.device)
            phi, terms, state = self._potential(ids, tick, v, anchors, shape=not self.defer_checks)
            bad |= state['_derived_bad']
            if phi is not None:
                bad |= ~torch.isfinite(phi)
            cost = 0.05 * self._qcmd + 0.1
            if self.full:
                slip = state['_slip']
                cost = cost + torch.where(torch.isnan(slip), 0.0, self.config.costs['slip'] * (slip / self.config.success['max_slip_m']).clamp(0.0, 2.0))
            for key in self._soft_fields:
                extra = v[key] * GOAL_SOFT_COSTS[key]
                cost = cost + extra
                self._soft_history[key][:, self._observe_calls] = torch.where(active, extra, 0)
            bad = self._quarantined | ((bad | ~torch.isfinite(cost)) & active)
            if self.defer_checks:
                self._window_bad |= bad
            else:
                self._guard(bad, 'nonfinite/numerical source, fixed anchor or consecutive actual 480Hz clock failure')
            failures = {'numerical_health': ~v['numerical_health'], 'constraint_failure': v['constraint_failure']}
            failures.update((n, value) for n, value in v.items()
                            if n.startswith('original_guard_') or n in GOAL_TERMINATIONS)
            if self.full:
                failures['prohibited_support'] = v['prohibited_support']
            failure = self._b()
            for mask in failures.values():
                failure |= mask
            if self.config.finger_goal is not None:
                goal = self.config.finger_goal
                tol = ~failure
                for name in tenso_train_rewar__ERROR_FIELDS:
                    tol &= v[name] <= goal.tolerance_rad
                if 'wrist_goal_m' in v:
                    tol &= (v['wrist_goal_m'] <= 0.010) & (v['wrist_goal_rad'] <= 0.15)
                dwell = torch.where(tol, self._dwell + 1, 0)
                change = active & ~self._arrived
                self._dwell = torch.where(change, dwell, self._dwell)
                self._arrival_pending |= change & (dwell >= goal.arrival_steps)
            success = ~failure & (state['_hold_s'] >= self.config.success['hold_s']) if self.full else self._b()
            event = torch.where(failure, -40.0, torch.where(success, 100.0, 0.0)).to(torch.float64)
            if self.full:
                event = torch.where(v['prohibited_support'], -50.0, event)
            for (name, mask) in failures.items():
                self._reasons[name] |= active & mask
            if self.full:
                self._reasons['success'] |= active & success
            self._event = torch.where(active, event, self._event)
            self._costs[:, self._observe_calls] = torch.where(active, cost, torch.nan)
            if self.full:
                self._slip_history[:, self._observe_calls] = torch.where(active, state['_slip'], torch.nan)
            self._count += active.to(torch.int64)
            self._terminal |= active & (failure | success)
            self._commit(ids, active, phi, terms, state, tick, time, v, anchors)
            self._previous = torch.where(active[:, None], self._request, self._previous)
            self._previous_tokens = self._pending
            self._previous_policies = self._policies
            self._observe_calls += 1
            return (self._count == 8) | self._terminal
        except tenso_train_rewar__SourceFailure:
            self._quarantined[:] = True
            self._poisoned_lanes = set(range(self.n))
            raise

    def finalize_observations(self):
        if not self.defer_checks or self._observations_finalized:
            return
        ids = torch.arange(self.n, device=self.device)
        state = {'_contact_s': self._contact_s, '_hold_s': self._hold_s}
        phi, terms = self._shape_potential(ids, self._last_values, state)
        self._guard(self._window_bad | ~torch.isfinite(phi), 'invalid source, fixed anchor or consecutive 480Hz clock in control window')
        self._phi = phi
        self._terms = terms
        self._observations_finalized = True

    def settle_once(self, tokens, *, truncated=None, buffer_full=None):
        if self._poisoned_lanes:
            raise tenso_train_rewar__SourceFailure('quarantined rollout has no usable PPO packet')
        if self._pending is None or tokens != self._pending:
            raise RuntimeError('token absent or already settled')
        if self._observe_calls == 0:
            raise RuntimeError('incomplete window: eight samples unless trusted terminal')
        self.finalize_observations()
        tr = self._b() if truncated is None else self._tensor(truncated, (self.n,), boolean=True)
        bf = self._b() if buffer_full is None else self._tensor(buffer_full, (self.n,), boolean=True)
        incomplete = (self._count == 0) | (self._count < 8) & ~self._terminal
        boundary = tr & bf | self._terminal & tr
        event = self._event.clone()
        arrived = self._arrival_pending & ~self._terminal
        if self.config.finger_goal is not None:
            event += arrived.to(torch.float64) * self.config.finger_goal.arrival_reward
        intervals = torch.arange(8, device=self.device)[None, :] < self._count[:, None]
        integrated = torch.where(intervals, self._costs, 0.0).sum(dim=1) / tenso_train_rewar__PHYSICS_HZ
        endpoint_phi = torch.where(self._terminal, 0.0, self._phi)
        reward = self.config.gamma * endpoint_phi - self._phi_pre - integrated + event
        numerical = ~torch.isfinite(reward) | ~torch.isfinite(integrated)
        if bool((incomplete | boundary | numerical).any()):
            if bool(incomplete.any()):
                raise RuntimeError('incomplete window: eight samples unless trusted terminal')
            if bool(boundary.any()):
                raise ValueError('ambiguous terminal/truncation/buffer boundary')
            self._quarantined[:] = True
            self._poisoned_lanes = set(range(self.n))
            raise tenso_train_rewar__SourceFailure('finite integrated reward required')
        if self.config.finger_goal is not None:
            self._arrived |= arrived
        context = self._context()
        term_payload = self._term_payload()
        event_terms = {'trusted_constraint_failure': self._term(self._event < 0, 'dimensionless', 'derived:original_failure_predicates', torch.where((self._event < 0) & (self._event != -50.0), self._event, 0.0))}
        if self.full:
            event_terms.update(prohibited_support=self._term(self._last_values['prohibited_support'], 'dimensionless', self._sources('prohibited_support'), torch.where(self._event == -50.0, self._event, 0.0)), success=self._term(self._event == 100.0, 'dimensionless', 'derived:independent_reward_hold_predicate', torch.where(self._event > 0, self._event, 0.0)))
        else:
            event_terms.update(prohibited_support=self._excluded(), success=self._excluded())
        if self.config.finger_goal is not None:
            event_terms['finger_goal_arrived_once'] = self._term(arrived, 'dimensionless', 'derived:measured_bilateral_active_joint_goal', arrived.to(torch.float64) * self.config.finger_goal.arrival_reward)
        contribution = lambda value: torch.where(intervals, value, torch.nan)
        substep_terms = {'command_rate': self._term(contribution(self._qcmd[:, None]), 'dimensionless', 'submitted_request', contribution(0.05 * self._qcmd[:, None] / tenso_train_rewar__PHYSICS_HZ), applicable=intervals), 'time': self._term(contribution(torch.full_like(self._costs, 1.0 / tenso_train_rewar__PHYSICS_HZ)), 's', 'actual_endpoint_clock', contribution(torch.full_like(self._costs, 0.1 / tenso_train_rewar__PHYSICS_HZ)), applicable=intervals)}
        if self.full:
            slip_cost = self.config.costs['slip'] * (self._slip_history / self.config.success['max_slip_m']).clamp(0, 2)
            substep_terms['slip'] = self._term(self._slip_history, 'm', 'derived:conservative_fixed_anchors', contribution(slip_cost.nan_to_num() / tenso_train_rewar__PHYSICS_HZ), applicable=intervals & ~torch.isnan(self._slip_history), history_reason='no_bilateral_contact_since_reset')
        else:
            substep_terms['slip'] = self._excluded(shape=self._costs.shape)
        for key in self._soft_fields:
            cost = self._soft_history[key]
            substep_terms[key] = self._term(contribution(cost / GOAL_SOFT_COSTS[key]), 'dimensionless',
                                            self._sources(key), contribution(cost / tenso_train_rewar__PHYSICS_HZ), applicable=intervals)
        packet = TensorRewardPacket(self._pending, self._policies, self.config.config_sha256, tenso_train_rewar__SCHEMA_SHA256, self.config.profile, self.config.profile_version, self.config.gamma, reward.clone(), self._count.to(torch.float64) / tenso_train_rewar__PHYSICS_HZ, self._phi_pre.clone(), self._phi.clone(), endpoint_phi.clone(), self._costs.clone(), integrated.clone(), event.clone(), self._copy(self._reasons), self._count.clone(), self._terminal.clone(), tr.clone(), bf.clone(), (self._terminal | tr).clone(), context, term_payload, self._pre_context, self._pre_terms, tenso_train_rewar__MappingProxyType(substep_terms), tenso_train_rewar__MappingProxyType(event_terms), tuple(self.schemas))
        self._closed = self._terminal | tr
        self._pending = None
        self._policies = None
        return packet

    @staticmethod
    def _copy(values):
        return tenso_train_rewar__MappingProxyType({name: value.clone() for (name, value) in values.items()})

    def _term(self, raw, unit, source, contribution=None, *, applicable=None, history_reason=None):
        shape = (self.n,) if contribution is None else contribution.shape
        return tenso_train_rewar__MappingProxyType(dict(raw=raw.clone() if isinstance(raw, torch.Tensor) else raw, unit=unit, source=source, valid=torch.ones(shape, dtype=torch.bool, device=self.device), applicable=torch.ones(shape, dtype=torch.bool, device=self.device) if applicable is None else applicable.clone(), contribution=torch.zeros(shape, dtype=torch.float64, device=self.device) if contribution is None else contribution.clone(), history_reason=history_reason))

    def _excluded(self, shape=None):
        shape = (self.n,) if shape is None else shape
        return self._term(None, 'dimensionless', 'profile_excluded', torch.zeros(shape, device=self.device, dtype=torch.float64), applicable=torch.zeros(shape, device=self.device, dtype=torch.bool), history_reason='profile_excluded')

    def _sources(self, *names):
        if names not in self._source_terms:
            rows = ({key: source for key, _, source in schema.field_specs} for schema in self.schemas)
            self._source_terms[names] = tuple(';'.join(row[name] for name in names) for row in rows)
        return self._source_terms[names]

    def _term_payload(self):
        terms = {name: self._excluded() for name in ('distance', 'contact', 'persistence', 'geometry', 'lift', 'hold', 'slip', 'velocity', 'prohibited_support', 'task_deadline')}
        terms['distance'] = self._term(self._last_values['distance'], 'm', self._sources('distance'), self._terms['distance'])
        if self.config.finger_goal is not None:
            errors = torch.stack([self._last_values[n] for n in tenso_train_rewar__ERROR_FIELDS], dim=1)
            terms['finger_goal'] = self._term(errors, 'rad', self._sources(tenso_train_rewar__ERROR_FIELDS[0]), self._terms['finger_goal'])
        if self.full:
            v = self._last_values
            terms.update(contact=self._term((v['coverage_left'] + v['coverage_right']) / 2.0, 'coverage_fraction', self._sources('coverage_left', 'coverage_right'), self._terms['contact']), persistence=self._term(self._contact_s, 's', 'derived:continuous_required_bilateral_original_opposed', self._terms['persistence']), geometry=self._term((v['opposed_left'].double() + v['opposed_right'].double()) / 2.0, 'dimensionless', self._sources('opposed_left', 'opposed_right'), self._terms['geometry']), lift=self._term(torch.minimum(v['median_z'] - self._median, v['min_z'] - self._min), 'm', self._sources('median_z', 'min_z'), self._terms['lift']), hold=self._term(self._hold_s, 's', 'derived:continuous_instantaneous_hold', self._terms['hold']), prohibited_support=self._term(v['prohibited_support'], 'dimensionless', self._sources('prohibited_support')), slip=self._term(self._slip, 'm', tuple((';'.join((a[1] for a in s.anchor_specs)) for s in self.schemas)), torch.where(torch.isnan(self._slip), 0.0, (self._slip / 0.0005).clamp(0.0, 2.0)), applicable=~torch.isnan(self._slip), history_reason='no_bilateral_contact_since_reset'))
        if self.config.profile == train_rewar__GOAL_FULL:
            for name, sources in (('load_progress', GOAL_SHAPING_FIELDS[:2]),
                                  ('prelift', ('median_z', 'min_z', *GOAL_SHAPING_FIELDS[2:]))):
                contribution = self._terms[name]
                terms[name] = self._term(contribution / self.config.weights[name], 'dimensionless',
                                         self._sources(*sources), contribution)
        for (name, unit, source) in self.schemas[0].field_specs:
            raw = self._slip if self.full and name == 'slip' else self._last_values[name]
            if self.full and name == 'slip':
                terms['raw:slip'] = self._term(None, unit, self._sources(name), applicable=self._b(), history_reason='no_bilateral_contact_since_reset')
            else:
                terms['raw:' + name] = self._term(raw, unit, self._sources(name))
        if self.full:
            for (j, (side, kind)) in enumerate(((s, k) for s in ('left', 'right') for k in ('material', 'tangential'))):
                for (label, raw, source_index) in (('drift', self._anchor_drift, 1), ('uncertainty', self._anchor_uncertainty, 2)):
                    terms['raw:anchor:' + side + ':' + kind + ':' + label] = self._term(raw[:, j], 'm', tuple((s.anchor_specs[j][source_index] if s.anchor_specs else None for s in self.schemas)), applicable=self._anchor_latched)
        return tenso_train_rewar__MappingProxyType(terms)

    def _context(self):
        values = dict(physics_tick=self._tick, solver_time_s=self._time, distance_ref_m=self._dref, reset_median_z_m=self._median, reset_min_z_m=self._min, contact_s=self._contact_s, hold_s=self._hold_s, slip_status=self._slip_status, slip_m=self._slip, previous_request=self._previous, anchor_reference_tick=self._anchor_tick)
        if self._anchor_ref is not None:
            values['anchor_reference'] = self._anchor_ref
        result = {name: value.clone() for (name, value) in values.items()}
        result.update(previous_request_token=tuple(self._previous_tokens), previous_policy_uid=tuple(self._previous_policies), schemas=tuple(self.schemas), finite_task_remaining_s=None, config_sha256=self.config.config_sha256, schema_sha256=tenso_train_rewar__SCHEMA_SHA256)
        return tenso_train_rewar__MappingProxyType(result)
