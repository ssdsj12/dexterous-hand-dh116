from dh116.env.contracts import source_file
"""Current Teacher network, pure six-key/context/material schema, tensor inputs and source-context owner; contains no policy collector."""
from dataclasses import dataclass as tenso_nativ_input__dataclass
from copy import deepcopy
import json
from dataclasses import dataclass as tenso_teach_schem__dataclass
import math as tenso_nativ_input__math
from pathlib import Path as tenso_nativ_teach__Path
import math as tenso_sourc_owner__math
import math as tenso_teach_schem__math
import torch as tenso_nativ_input__torch
import torch as tenso_nativ_teach__torch
from pathlib import Path as tenso_sourc_owner__Path
from types import MappingProxyType
from dh116.env.contracts import linea_coupl__ACTIVE as ACTIVE
from torch import nn as tenso_nativ_teach__nn
import sys
import torch as tenso_teach_schem__torch
from dh116.env.contracts import JOINTS
from dh116.env.contracts import EagerRuntimePolicy
from time import perf_counter
from torch import nn as tenso_teach_schem__nn
from dh116.env.contracts import MIMIC
from dh116.env.contracts import validate_runtime_execution
import uuid
from torch.nn.functional import pad
from dh116.env.contracts import active_limits
from dh116.rl.ppo import PrivilegedTeacher
import numpy as np
from dh116.env.contracts import runti_io__digest as tenso_teach_schem__digest
from dh116.hand.kinematics import WRIST_LINEAR
import torch as tenso_sourc_owner__torch
MODEL_SCHEMA_VERSION = 'dh116-native-context-teacher-tanh-v1'
from dh116.hand.kinematics import WRIST_ANGULAR
from dh116.env.contracts import runti_io__ROOT as ROOT
INPUT_SCHEMA_VERSION = 'dh116-native-context-teacher-input-v1'
from dh116.env.contracts import runti_io__digest as tenso_sourc_owner__digest
CONTEXT_SCHEMA_VERSION = 'dh116-native-teacher-context50-v1'
tenso_teach_schem__INPUT_KEYS = ('common371', 'common_valid', 'rod_nodes', 'rod_mask', 'task_context', 'task_context_valid')
tenso_teach_schem__FAMILY = 'shared_batched_solver_v1'

def common_selection():
    return tuple(range(24)) + tuple(range(78, 88)) + tuple(range(153, 185))
ROD_FEATURES = ('position_world_x_m', 'position_world_y_m', 'position_world_z_m', 'velocity_world_x_m_s', 'velocity_world_y_m_s', 'velocity_world_z_m_s', 'tangent_world_x', 'tangent_world_y', 'tangent_world_z', 'curvature_m_inv', 'normalized_rest_material_station', 'adjacent_segment_strain', 'adjacent_segment_strain_valid')

def _mask(mask, shape, device):
    if not isinstance(mask, tenso_teach_schem__torch.Tensor) or mask.dtype != tenso_teach_schem__torch.bool or mask.shape != shape or (mask.device != device):
        raise ValueError('explicit boolean rod node mask shape/device differs')
    if bool((mask.sum(dim=1) < 3).any()):
        raise ValueError('each rod requires at least three valid material nodes')
    return mask

def _rod_tensor(snapshot, name, reference, shape):
    value = snapshot.get(name)
    if not isinstance(value, tenso_teach_schem__torch.Tensor) or value.shape != shape or value.dtype != reference.dtype or (value.device != reference.device):
        raise ValueError(f'rod {name} tensor shape/dtype/device differs')
    return value

def tenso_teach_schem___features(snapshot, common, mask):
    nodes = snapshot.get('node_positions')
    if not isinstance(nodes, tenso_teach_schem__torch.Tensor) or nodes.ndim != 3:
        raise ValueError('rod node_positions shape must be [batch,nodes,3]')
    (batch, count) = (common.shape[0], nodes.shape[1])
    mask = _mask(mask, (batch, count), common.device)
    shape = (batch, count, 3)
    fields = []
    for name in ('node_positions', 'velocities', 'tangent'):
        value = _rod_tensor(snapshot, name, common, shape)
        if not bool(tenso_teach_schem__torch.isfinite(value[mask]).all()):
            raise ValueError(f'valid rod {name} contains unavailable values')
        if name == 'tangent':
            norms = tenso_teach_schem__torch.linalg.vector_norm(value[mask], dim=-1)
            if not tenso_teach_schem__torch.allclose(norms, tenso_teach_schem__torch.ones_like(norms), atol=0.0001, rtol=0.0):
                raise ValueError('valid rod nodes require a unit tangent')
        fields.append(tenso_teach_schem__torch.where(mask[..., None], value, 0.0))
    curvature = _rod_tensor(snapshot, 'curvature', common, (batch, count))
    if not bool((tenso_teach_schem__torch.isfinite(curvature[mask]) & (curvature[mask] >= 0)).all()):
        raise ValueError('valid curvature must be finite and nonnegative')
    lengths = _rod_tensor(snapshot, 'rest_lengths_m', common, (batch, count - 1))
    if not bool((tenso_teach_schem__torch.isfinite(lengths) & (lengths > 0)).all()):
        raise ValueError('rest material lengths require finite positive segment geometry')
    station = pad(lengths.cumsum(-1), (1, 0)) / lengths.sum(-1, keepdim=True)
    strain = _rod_tensor(snapshot, 'deformation', common, (batch, count - 1))
    segment_valid = mask[:, :-1] & mask[:, 1:]
    if not bool(tenso_teach_schem__torch.isfinite(strain[segment_valid]).all()):
        raise ValueError('valid adjacent segment strain is unavailable')
    observed_strain = tenso_teach_schem__torch.where(segment_valid, strain, 0.0)
    weights = segment_valid.to(dtype=common.dtype)
    count_valid = pad(weights, (1, 0)) + pad(weights, (0, 1))
    average_strain = (pad(observed_strain, (1, 0)) + pad(observed_strain, (0, 1))) / count_valid.clamp_min(1.0)
    scalars = tenso_teach_schem__torch.stack((tenso_teach_schem__torch.where(mask, curvature, 0.0), station, average_strain, (count_valid > 0).to(dtype=common.dtype)), dim=-1)
    return tenso_teach_schem__torch.where(mask[..., None], tenso_teach_schem__torch.cat((*fields, scalars), dim=-1), 0.0)

class _MaterialEncoder(tenso_teach_schem__nn.Module):

    def __init__(self, width):
        super().__init__()
        self.per_node = tenso_teach_schem__nn.Sequential(tenso_teach_schem__nn.Linear(len(ROD_FEATURES), width), tenso_teach_schem__nn.ELU(), tenso_teach_schem__nn.Linear(width, width), tenso_teach_schem__nn.ELU())

    def forward(self, nodes, mask):
        safe = tenso_teach_schem__torch.where(mask[..., None], nodes, 0.0)
        encoded = self.per_node(safe)
        active = mask[..., None]
        mean = tenso_teach_schem__torch.where(active, encoded, 0.0).sum(1) / mask.sum(1, keepdim=True)
        maximum = encoded.masked_fill(~active, -tenso_teach_schem__torch.inf).amax(1)
        return tenso_teach_schem__torch.cat((mean, maximum), dim=-1)
CONTACT_PRELIFT = 'approach_contact_prelift_v1'
MODES = ('approach_wiring_smoke_v1', 'flexible_train_r1_full', CONTACT_PRELIFT)
CONTACT_PRELIFT_EXCLUDED = (7, 47)
SCOPES = ('cpu_fixture', 'owner_source_unqualified')
tenso_teach_schem__SMOKE_EXCLUDED = (4, 5, 6, 7, 47)

def _schema():
    rows = []

    def add(name, unit, scale, source, applicability='required'):
        rows.append(MappingProxyType(dict(index=len(rows), name=name, unit=unit, scale=scale, source=source, validity='finite when valid; unknown NaN/false', applicability=applicability)))
    for phase in ('Approach', 'Close', 'Lift', 'Hold'):
        add('phase_one_hot.' + phase, 'bool', 1.0, 'physical_phase_state_machine')
    for side in ('L', 'R'):
        add('required_contact_fraction.' + side, '1', 1.0, 'qualified_dual_hand_contact', 'full_only')
    add('bilateral_contact_duration', 's', 0.25, 'qualified_contact_history', 'full_only')
    add('continuous_hold_duration', 's', 5.0, 'qualified_task_history', 'full_only')
    for side in ('L', 'R'):
        add('fixed_material_anchor_station.' + side, 'm', 'rest_length_m', 'fixed_material_anchor_history', 'anchored_only')
    for side in ('L', 'R'):
        add('fixed_anchor_valid.' + side, 'bool', 1.0, 'fixed_material_anchor_history')
    for i in range(24):
        add('previous_normalized_request24.' + str(i), '1', 1.0, 'Action24HistoryOwner')
    for name in ('reset_median_node_z', 'reset_min_node_z'):
        add(name, 'm', 1.0, 'verified_reset_41_nodes')
    add('distance_ref', 'm', 1.0, 'max_actual_reset_distance_0.005m')
    add('actual_reset_distance_mean', 'm', 1.0, 'verified_reset_distance')
    for (name, unit) in (('lift_target', 'm'), ('hold_target', 's'), ('slip_limit', 'm')):
        add(name, unit, 1.0, 'success_contract_30mm_5s_0.5mm')
    add('episode_elapsed', 's', 'explicit_episode_cap_s', 'physical_clock')
    add('external_time_remaining', 's', 'explicit_external_cap_s', 'external_budget_clock')
    add('finite_task_deadline_enabled', 'bool', 1.0, 'task_contract_no_finite_deadline')
    add('finite_task_time_remaining', 's', 1.0, 'task_contract_no_finite_deadline', 'not_applicable')
    add('prohibited_support_latched', 'bool', 1.0, 'qualified_support_history', 'full_only')
    add('success_event_emitted', 'bool', 1.0, 'task_event_history')
    add('trusted_failure_event_emitted', 'bool', 1.0, 'task_event_history')
    return tuple(rows)
CONTEXT_FIELDS = _schema()
CONTEXT_SCHEMA_SHA256 = tenso_teach_schem__digest(dict(version=CONTEXT_SCHEMA_VERSION, fields=[dict(field) for field in CONTEXT_FIELDS]))

@tenso_teach_schem__dataclass(frozen=True, slots=True)
class ContextIdentity:
    run_uuid: str
    env_id: int
    episode_uid: str
    reset_id: int
    rod_uid: str
    physics_step: int
    rod_tick: int
    timestamp_s: float
    solver_time_s: float

    def __post_init__(self):
        for key in ('run_uuid', 'episode_uid', 'rod_uid'):
            if type(getattr(self, key)) is not str or not getattr(self, key).strip():
                raise ValueError('immutable nonempty context identity required')
        for key in ('env_id', 'reset_id', 'physics_step', 'rod_tick'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError('explicit integer context identity required')
        for key in ('timestamp_s', 'solver_time_s'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not tenso_teach_schem__math.isfinite(value) or value < 0:
                raise ValueError('finite context clock required')

    @property
    def epoch(self):
        return (self.run_uuid, self.env_id, self.episode_uid, self.reset_id, self.rod_uid)

@tenso_teach_schem__dataclass(frozen=True, slots=True)
class ContextMetadata:
    owner_uid: str
    identities: tuple
    backend_profile_sha256: str
    scope: str
    mode: str
    sources: tuple
    applicable: tuple
    reasons: tuple
    reference_uids: tuple
    previous_request_token: str
    grasp: str = 'pinch'

    def __post_init__(self):
        if type(self.identities) is not tuple or not self.identities or any((type(x) is not ContextIdentity for x in self.identities)) or (len({x.env_id for x in self.identities}) != len(self.identities)) or (self.scope not in SCOPES) or (self.mode not in MODES) or (self.grasp not in ('pinch', 'tripod', 'power')):
            raise ValueError('immutable complete context identities/scope required')
        if any((type(x) is not str or not x.strip() for x in (self.owner_uid, self.previous_request_token, self.backend_profile_sha256))):
            raise ValueError('context owner/profile/previous request token required')
        for value in (self.sources, self.applicable, self.reasons, self.reference_uids):
            if type(value) is not tuple:
                raise ValueError('immutable context metadata tuples required')
        if len(self.sources) != 50 or len(self.applicable) != 50 or len(self.reasons) != 50 or any((type(x) is not bool for x in self.applicable)) or any((type(x) is not str or not x for x in self.sources)) or any((x is not None and (type(x) is not str or not x.strip()) for x in self.reasons)) or (not self.reference_uids) or any((type(x) is not tuple or len(x) != 2 or any((type(y) is not str or not y for y in x)) for x in self.reference_uids)) or (len({x[0] for x in self.reference_uids}) != len(self.reference_uids)):
            raise ValueError('complete immutable context source/applicability/reference metadata required')

@tenso_teach_schem__dataclass(frozen=True, slots=True)
class TaskContextPacket:
    values: tenso_teach_schem__torch.Tensor
    valid: tenso_teach_schem__torch.Tensor
    metadata: ContextMetadata

    def owned(self):
        return TaskContextPacket(self.values.clone(), self.valid.clone(), self.metadata)

    @property
    def native_qualified(self):
        return False

def encode_context(values, valid):
    """Generic last-dimension mask representation; never modifies the raw values."""
    if not isinstance(values, tenso_teach_schem__torch.Tensor) or not values.is_floating_point() or values.ndim < 1 or (not isinstance(valid, tenso_teach_schem__torch.Tensor)) or (valid.dtype != tenso_teach_schem__torch.bool) or (valid.shape != values.shape) or (valid.device != values.device) or (not bool(tenso_teach_schem__torch.isfinite(values[valid]).all())) or (not bool(tenso_teach_schem__torch.isnan(values[~valid]).all())):
        raise ValueError('context values/validity require finite valid and raw NaN unknown')
    return tenso_teach_schem__torch.cat((tenso_teach_schem__torch.where(valid, values, 0.0), valid.to(values.dtype)), dim=-1)

def validate_context_tensors(values, valid, *, mode=None):
    if not isinstance(values, tenso_teach_schem__torch.Tensor) or values.dtype != tenso_teach_schem__torch.float32 or values.ndim != 2 or (values.shape[1] != 50) or (not len(values)):
        raise ValueError('context requires float32[N,50]')
    encode_context(values, valid)
    required = [i for i in range(50) if i not in (*tenso_teach_schem__SMOKE_EXCLUDED, 8, 9, 46)]
    if not bool(valid[:, required].all()):
        raise ValueError('required applicable context source is unavailable')
    if bool(valid[:, 46].any()) or not bool((values[:, 45] == 0).all()):
        raise ValueError('finite task deadline is disabled by this task contract')
    for i in (0, 1, 2, 3, 10, 11, 45, 48, 49):
        if not bool(((values[:, i] == 0) | (values[:, i] == 1)).all()):
            raise ValueError('boolean context field differs')
    if not bool((values[:, :4].sum(-1) == 1).all()):
        raise ValueError('one actual physical phase required')
    if not bool((abs(values[:, 12:36]) <= 1).all()):
        raise ValueError('previous normalized request outside action24')
    for j in (0, 1):
        anchored = values[:, 10 + j] == 1
        if not tenso_teach_schem__torch.equal(valid[:, 8 + j], anchored):
            raise ValueError('anchor station validity differs from actual anchor state')
        if not bool(((values[anchored, 8 + j] >= 0) & (values[anchored, 8 + j] <= 1)).all()):
            raise ValueError('anchor material station outside rest geometry')
    expected = values.new_tensor([0.03, 5.0, 0.0005])
    if not tenso_teach_schem__torch.equal(values[:, 40:43], expected.expand(len(values), 3)) or not bool((values[:, 39] >= 0).all()) or (not tenso_teach_schem__torch.allclose(values[:, 38], values[:, 39].clamp_min(0.005), rtol=0, atol=1e-08)) or (not bool((values[:, 36] >= values[:, 37]).all())) or (not bool((values[:, 43:45] >= 0).all())) or (not bool((values[:, 44] <= 1).all())):
        raise ValueError('fixed reset reference, success contract or explicit budget context differs')
    if mode == MODES[0]:
        if bool(valid[:, tenso_teach_schem__SMOKE_EXCLUDED].any()) or not bool((values[:, 0] == 1).all()):
            raise ValueError('smoke context excludes contact/support and remains physical Approach')
    elif mode == MODES[1]:
        if not bool(valid[:, tenso_teach_schem__SMOKE_EXCLUDED].all()):
            raise ValueError('full context requires qualified contact/support/history sources')
    elif mode == CONTACT_PRELIFT:
        if not bool(valid[:, [4, 5, 6]].all()) or bool(valid[:, CONTACT_PRELIFT_EXCLUDED].any()) or (not bool((values[:, 1] == 1).all())) or bool(valid[:, 8:10].any()) or (not bool((values[:, 10:12] == 0).all())) or (not bool((values[:, 48] == 0).all())):
            raise ValueError('contact-only Close requires measured contact, no anchors/hold/support latch/success')
    for i in (4, 5):
        if not bool(((values[valid[:, i], i] >= 0) & (values[valid[:, i], i] <= 1)).all()):
            raise ValueError('contact fraction outside [0,1]')
    if any((not bool((values[valid[:, i], i] >= 0).all()) for i in (6, 7))):
        raise ValueError('continuous context durations must be nonnegative')
    if not bool(((values[valid[:, 47], 47] == 0) | (values[valid[:, 47], 47] == 1)).all()):
        raise ValueError('support latch must be boolean')

def validate_context(packet, *, backend_profile, identities=None):
    if type(packet) is not TaskContextPacket or type(packet.metadata) is not ContextMetadata:
        raise ValueError('typed context packet and immutable sidecar required')
    meta = packet.metadata
    if meta.backend_profile_sha256 != backend_profile.digest:
        raise ValueError('context backend profile differs')
    validate_context_tensors(packet.values, packet.valid, mode=meta.mode)
    if len(meta.identities) != len(packet.values):
        raise ValueError('context identity batch differs')
    if identities is not None and meta.identities != tuple(identities):
        raise ValueError('context time/episode identity differs')
    if meta.sources != tuple((x['source'] for x in CONTEXT_FIELDS)):
        raise ValueError('required context source schema differs')
    applicable = list(meta.applicable)
    wanted = [True] * 50
    wanted[46] = False
    for j in (0, 1):
        anchors = packet.valid[:, 8 + j]
        if not bool((anchors == anchors[0]).all()):
            raise ValueError('different anchor applicability needs separate context packet')
        wanted[8 + j] = bool(anchors[0])
    if meta.mode == MODES[0]:
        for i in tenso_teach_schem__SMOKE_EXCLUDED:
            wanted[i] = False
    elif meta.mode == CONTACT_PRELIFT:
        for i in CONTACT_PRELIFT_EXCLUDED:
            wanted[i] = False
    if applicable != wanted:
        raise ValueError('context applicability differs from task/source profile')
    for (i, applies) in enumerate(applicable):
        if applies and (not bool(packet.valid[:, i].all())) or (not applies and (bool(packet.valid[:, i].any()) or not meta.reasons[i])):
            raise ValueError('required applicable unknown or missing explicit NA reason')
    return packet

@tenso_teach_schem__dataclass(frozen=True, slots=True)
class SharedSourceIdentity:
    parent_solver_id: int
    parent_num_envs: int
    env_rows: tuple
    run_uuid: str
    episode_uids: tuple
    rod_uids: tuple
    reset_ids: tuple
    rod_ticks: tuple
    sdk_step: int
    sdk_time_s: float
    sdk_dt_s: float
    sdk_timer_dt_s: float
    policy_time_s: float
    solver_time_s: float
    profile_sha256: str
    scope: str
tenso_nativ_input__common_selection = common_selection
tenso_nativ_input___features = tenso_teach_schem___features
tenso_nativ_input__ContextIdentity = ContextIdentity
tenso_nativ_input__validate_context_tensors = validate_context_tensors
tenso_nativ_input__FAMILY = tenso_teach_schem__FAMILY
tenso_nativ_input__SharedSourceIdentity = SharedSourceIdentity
tenso_nativ_teach__INPUT_KEYS = tenso_teach_schem__INPUT_KEYS
tenso_nativ_teach___MaterialEncoder = _MaterialEncoder
tenso_nativ_teach__common_selection = common_selection
tenso_nativ_teach__encode_context = encode_context
tenso_nativ_teach__validate_context_tensors = validate_context_tensors
tenso_sourc_owner__ContextIdentity = ContextIdentity
tenso_sourc_owner__SMOKE_EXCLUDED = tenso_teach_schem__SMOKE_EXCLUDED
tenso_sourc_owner__SharedSourceIdentity = SharedSourceIdentity
VERSION = 'dh116-shared-tensor-native-inputs-v1'
from dh116.physics.solver import TensorBackendProfile as tenso_nativ_teach__TensorBackendProfile
from dh116.physics.solver import validate_snapshot

@tenso_nativ_input__dataclass(frozen=True, slots=True)
class TensorTeacherPacket:
    """Owned six-key input with the unchanged shared parent/row source identity."""
    inputs: dict
    source: tenso_nativ_input__SharedSourceIdentity
    grasp: str
    context_identities: tuple
    source_family: str = tenso_nativ_input__FAMILY

    @property
    def native_qualified(self):
        return False

    def validate_fresh(self, current_identity):
        if type(current_identity) is not tenso_nativ_input__SharedSourceIdentity or current_identity != self.source:
            raise ValueError('shared source parent/row/episode Reset epoch or clock changed')
        return self

    def for_teacher(self, model, *, current_identity):
        """New explicit hook; the old pack_shared class gate stays untouched."""
        self.validate_fresh(current_identity)
        _tenso_nativ_input_import__TensorNativeTeacher = TensorNativeTeacher
        if type(model) is not _tenso_nativ_input_import__TensorNativeTeacher:
            raise ValueError('explicit current tensor-source Teacher required')
        identity = model.model_identity
        if identity['source_family'] != self.source_family or identity['parent_num_envs'] != self.source.parent_num_envs or identity['grasp'] != self.grasp or (identity['backend_profile_sha256'] != self.source.profile_sha256):
            raise ValueError('shared Teacher source/grasp/profile/environment contract differs')
        model._parts(self.inputs)
        return {key: value.detach().clone() for (key, value) in self.inputs.items()}
from dh116.env.contracts import runti_io__digest as tenso_nativ_teach__digest
from dh116.physics.solver import TensorRodOwner

def _tensor(name, value, shape, device, *, dtype=None, finite=True):
    if not isinstance(value, tenso_nativ_input__torch.Tensor) or value.shape != shape or value.device != device or (value.dtype != dtype if dtype is not None else value.dtype not in (tenso_nativ_input__torch.float32, tenso_nativ_input__torch.float64)):
        raise ValueError(f'{name} requires same-device {shape} tensor with original units/dtype')
    if finite and (not bool(tenso_nativ_input__torch.isfinite(value).all())):
        raise ValueError(f'{name} requires finite native tensor values')
    return value.detach()
from dh116.env.contracts import runti_io__file_sha256 as file_sha256
from dh116.physics.solver import TensorBackendProfile as tenso_sourc_owner__TensorBackendProfile

def _source(source, n, contexts, profile_sha, grasp, scope):
    if type(source) is not tenso_nativ_input__SharedSourceIdentity or source.parent_num_envs != n or source.env_rows != tuple(range(n)) or (not 2 <= n <= 512) or (source.scope != scope) or (scope not in ('cpu_fixture', 'owner_source_unqualified')) or (source.profile_sha256 != profile_sha) or (type(profile_sha) is not str) or (len(profile_sha) != 64) or (grasp not in ('pinch', 'tripod', 'power')) or any((type(x) is not tuple or len(x) != n for x in (source.episode_uids, source.rod_uids, source.reset_ids, source.rod_ticks))) or (type(contexts) is not tuple) or (len(contexts) != n) or any((type(c) is not tenso_nativ_input__ContextIdentity for c in contexts)):
        raise ValueError('explicit same shared parent rows/profile/grasp/source scope required')
    expected = tuple((tenso_nativ_input__ContextIdentity(source.run_uuid, i, source.episode_uids[i], source.reset_ids[i], source.rod_uids[i], source.sdk_step, source.rod_ticks[i], source.policy_time_s, source.solver_time_s) for i in range(n)))
    if contexts != expected:
        raise ValueError('common/rod/context shared parent row/time/episode source differs')
from dh116.rl.reward import validate_reward_config as validate_reward_config
_EXCLUDED = (*tenso_sourc_owner__SMOKE_EXCLUDED, 8, 9, 46)

@tenso_nativ_input__torch.no_grad()
def pack_shared_tensor_inputs(q, qd, *, joint_names, body_link_pose_w, body_com_velocity_w, root_seed_w, env_origins, rod_positions_w, rod_velocities_w, rest_lengths_m, task_context, task_context_valid, source_identity, context_identities, backend_profile_sha256, grasp, scope):
    """Pack [N,2,17] native q/qd into the original common66+rod41x13+context50.

    ``joint_names`` is a pair of complete, source-verified 17DOF name tuples,
    one per hand; q/qd angular units are rad and rad/s, prismatic units m/m/s.
    Pose is measured base_link WXYZ (no sign canonicalization); velocity is
    the original body COM world linear/angular velocity in m/s and rad/s.
    ``root_seed_w`` excludes reset root_offset, exactly as the original source.
    Context values/masks are copied from the existing owner, never synthesized.
    The caller rechecks its live shared boundary before consuming the packet.
    """
    if not isinstance(q, tenso_nativ_input__torch.Tensor) or q.ndim != 3 or q.shape[1:] != (2, 17):
        raise ValueError('q requires complete native [N,2,17] tensor')
    (n, device) = (len(q), q.device)
    _source(source_identity, n, context_identities, backend_profile_sha256, grasp, scope)
    complete = set(JOINTS + WRIST_LINEAR + WRIST_ANGULAR)
    if type(joint_names) not in (tuple, list) or len(joint_names) != 2 or any((type(names) not in (tuple, list) or len(names) != 17 or any((type(name) is not str for name in names)) or (len(set(names)) != 17) or (set(names) != complete) for names in joint_names)):
        raise ValueError('complete original named 17DOF joint source required for both hands')
    q = _tensor('q', q, (n, 2, 17), device).double()
    qd = _tensor('qd', qd, (n, 2, 17), device).double()
    pose = _tensor('base_link WXYZ pose', body_link_pose_w, (n, 2, 7), device).double()
    velocity = _tensor('body COM world velocity', body_com_velocity_w, (n, 2, 6), device).double()
    seed = _tensor('authored root seed', root_seed_w, (n, 2, 7), device).double()
    origins = _tensor('environment origins', env_origins, (n, 3), device)
    if not bool((origins[:, 2] == 0).all()) or not bool((abs(tenso_nativ_input__torch.linalg.vector_norm(pose[..., 3:], dim=-1) - 1.0) <= 1e-05).all()):
        raise ValueError('horizontal origins and raw unit WXYZ quaternion required')
    context = _tensor('task context', task_context, (n, 50), device, dtype=tenso_nativ_input__torch.float32, finite=False)
    context_valid = _tensor('task context validity', task_context_valid, (n, 50), device, dtype=tenso_nativ_input__torch.bool)
    tenso_nativ_input__validate_context_tensors(context, context_valid, mode='approach_wiring_smoke_v1')
    ledger = tenso_nativ_input__torch.full((n, 371), tenso_nativ_input__torch.nan, dtype=tenso_nativ_input__torch.float32, device=device)
    valid = tenso_nativ_input__torch.zeros((n, 371), dtype=tenso_nativ_input__torch.bool, device=device)
    (lower, upper) = active_limits()
    (lower, upper) = (q.new_tensor(lower), q.new_tensor(upper))
    for (hi, names) in enumerate(joint_names):
        ids = [names.index(name) for name in ACTIVE]
        ledger[:, 6 * hi:6 * (hi + 1)] = (2 * (q[:, hi, ids] - lower) / (upper - lower) - 1).float()
        ledger[:, 12 + 6 * hi:18 + 6 * hi] = (qd[:, hi, ids] / 0.3).float()
        follower = [names.index(f) for (f, _, _, _) in MIMIC]
        master = [names.index(m) for (_, m, _, _) in MIMIC]
        ratio = q.new_tensor([r for (_, _, r, _) in MIMIC])
        offset = q.new_tensor([o for (_, _, _, o) in MIMIC])
        ledger[:, 78 + 5 * hi:83 + 5 * hi] = ((q[:, hi, follower] - ratio * q[:, hi, master] - offset) / 0.02).float()
        normalized_pose = pose[:, hi].clone()
        normalized_pose[:, :3] = (pose[:, hi, :3] - seed[:, hi, :3]) / 0.05
        ledger[:, 153 + 7 * hi:160 + 7 * hi] = normalized_pose.float()
        ledger[:, 167 + 6 * hi:173 + 6 * hi] = (velocity[:, hi] / q.new_tensor([0.05, 0.05, 0.05, 1.0, 1.0, 1.0])).float()
        ledger[:, 179 + 3 * hi:182 + 3 * hi] = (q[:, hi, [names.index(k) for k in WRIST_ANGULAR]] / (tenso_nativ_input__math.pi / 2)).float()
    valid[:, tenso_nativ_input__common_selection()] = True
    if not bool(tenso_nativ_input__torch.isfinite(ledger[valid]).all()):
        raise ValueError('native common normalization produced nonfinite values')
    p = _tensor('rod world positions', rod_positions_w, (n, 41, 3), device).float()
    v = _tensor('rod world velocities', rod_velocities_w, (n, 41, 3), device).float()
    rest = _tensor('material rest lengths', rest_lengths_m, (n, 40), device).float()
    if not bool((rest > 0).all()):
        raise ValueError('finite positive material rest lengths required')
    edges = p.diff(dim=1)
    lengths = tenso_nativ_input__torch.linalg.vector_norm(edges, dim=-1)
    if not bool((lengths > 0).all()):
        raise ValueError('degenerate shared material segment')
    direction = edges / lengths[..., None]
    tangent = tenso_nativ_input__torch.cat((direction[:, :1], direction[:, :-1] + direction[:, 1:], direction[:, -1:]), 1)
    tangent /= tenso_nativ_input__torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    curvature = tenso_nativ_input__torch.zeros((n, 41), dtype=p.dtype, device=device)
    curvature[:, 1:-1] = tenso_nativ_input__torch.linalg.vector_norm(direction.diff(dim=1), dim=-1) / ((lengths[:, :-1] + lengths[:, 1:]) / 2)
    snapshot = dict(node_positions=p - origins.to(dtype=p.dtype)[:, None], velocities=v, tangent=tangent, curvature=curvature, rest_lengths_m=rest, deformation=lengths / rest - 1)
    mask = tenso_nativ_input__torch.ones((n, 41), dtype=tenso_nativ_input__torch.bool, device=device)
    nodes = tenso_nativ_input___features(snapshot, ledger[:, tenso_nativ_input__common_selection()], mask)
    inputs = dict(common371=ledger, common_valid=valid, rod_nodes=nodes, rod_mask=mask, task_context=context.clone(), task_context_valid=context_valid.clone())
    return TensorTeacherPacket(inputs, source_identity, grasp, context_identities)
from dh116.rl.reward import train_rewar__SMOKE as SMOKE
_REQUIRED = tuple((i for i in range(50) if i not in _EXCLUDED))

def _common(ledger, valid):
    if not isinstance(ledger, tenso_nativ_teach__torch.Tensor) or ledger.dtype != tenso_nativ_teach__torch.float32 or ledger.ndim != 2 or (ledger.shape[1] != 371) or (not len(ledger)) or (not isinstance(valid, tenso_nativ_teach__torch.Tensor)) or (valid.dtype != tenso_nativ_teach__torch.bool) or (valid.shape != ledger.shape) or (valid.device != ledger.device):
        raise ValueError('raw common371 requires float32[N,371] and boolean validity')
    wanted = tenso_nativ_teach__torch.zeros_like(valid)
    wanted[:, tenso_nativ_teach__common_selection()] = True
    if not tenso_nativ_teach__torch.equal(wanted, valid) or not bool(tenso_nativ_teach__torch.isfinite(ledger[valid]).all()) or (not bool(tenso_nativ_teach__torch.isnan(ledger[~valid]).all())):
        raise ValueError('common fixed measured66 mask/finite and 305 raw NaN required')
    return ledger[:, tenso_nativ_teach__common_selection()]

class TensorSourceOwner:
    """Owner of source identities and GPU smoke reset/action references.

    ``profile`` is the source-bound current TensorBackendProfile. Its manifest
    path and digest are fixed for this owner. Profile verification and native parameter
    validator run at construction/Reset, never at substep/context capture.

    ``contexts()`` returns (float32[N,50], bool[N,50], ContextIdentity tuple).
    ``reset(env_ids, distance_m)`` accepts K primitive row IDs and a device
    floating [K] tensor measured at each actual fresh Reset. Only those GPU
    reference rows change. Wall time remains external and elapsed time comes
    from the original SDK-derived policy timestamp, not a fitted offset.
    """

    def __init__(self, runtime, sim, profile, env_origins, device, grasp, episode_cap_s, wall_cap_s, *, scope=None, rod_uids=None):
        if type(runtime) is not TensorRodOwner:
            raise ValueError('actual TensorRodOwner required')
        if grasp not in ('pinch', 'tripod', 'power') or any((type(x) not in (float, int) or not tenso_sourc_owner__math.isfinite(x) or x <= 0 for x in (episode_cap_s, wall_cap_s))):
            raise ValueError('original grasp and finite source-owned episode/wall caps required')
        (self.runtime, self.sim, self.profile) = (runtime, sim, profile)
        self.device = tenso_sourc_owner__torch.device(device)
        self.scope = ('cpu_fixture' if self.device.type == 'cpu' else 'owner_source_unqualified') if scope is None else scope
        if self.scope not in ('cpu_fixture', 'owner_source_unqualified'):
            raise ValueError('explicit fixture or unqualified owner source scope required')
        if runtime.device != self.device:
            raise ValueError('source and actual tensor rod owner device differ')
        self._profile_sha256 = getattr(profile, 'digest', None)
        self._manifest_path = getattr(profile, 'manifest_path', None)
        self._verify_profile()
        n = runtime.solver.num_envs
        self.n = n
        self._parent_solver_id = id(runtime.solver)
        self.rod_uids = tuple(('rod' + str(i) for i in range(n))) if rod_uids is None else tuple(rod_uids)
        if len(self.rod_uids) != n or len(set(self.rod_uids)) != n or any((type(x) is not str or not x.strip() for x in self.rod_uids)):
            raise ValueError('one source-owned material rod identity per actual parent row required')
        self._origins = tenso_sourc_owner__torch.as_tensor(env_origins, device=self.device, dtype=tenso_sourc_owner__torch.float64).detach().clone()
        if self._origins.shape != (n, 3) or not bool(tenso_sourc_owner__torch.isfinite(self._origins).all()) or (not bool((self._origins[:, 2] == 0).all())):
            raise ValueError('source-owned finite horizontal environment origins required')
        if self.scope != 'cpu_fixture':
            loaded = sys.modules.get('ropesim._ropesim')
            if loaded is None or not getattr(loaded, '__file__', None) or tenso_sourc_owner__Path(loaded.__file__).resolve() != tenso_sourc_owner__Path(self.profile.binary_path).resolve() or (type(runtime.solver) is not getattr(loaded, 'RopeBatch', None)):
                raise ValueError('loaded actual native flat candidate shared solver required')
        self.grasp = grasp
        (self.episode_cap_s, self.wall_cap_s) = (float(episode_cap_s), float(wall_cap_s))
        self.started = perf_counter()
        self.owner_uid = uuid.uuid4().hex
        self._reset_identities = [None] * n
        self._previous_tokens = [None] * n
        self._reset_z = tenso_sourc_owner__torch.full((n, 2), tenso_sourc_owner__torch.nan, dtype=tenso_sourc_owner__torch.float64, device=self.device)
        self._reset_distance = tenso_sourc_owner__torch.full((n,), tenso_sourc_owner__torch.nan, dtype=tenso_sourc_owner__torch.float64, device=self.device)
        self._reference_sdk = tenso_sourc_owner__torch.full((n,), -1, dtype=tenso_sourc_owner__torch.int64, device=self.device)
        self._reference_policy_time = tenso_sourc_owner__torch.full((n,), tenso_sourc_owner__torch.nan, dtype=tenso_sourc_owner__torch.float64, device=self.device)
        self._previous = tenso_sourc_owner__torch.full((n, 24), tenso_sourc_owner__torch.nan, dtype=tenso_sourc_owner__torch.float32, device=self.device)
        boundary = self._boundary()
        (self._rest, self._rest_total) = self._read_parameters(boundary)
        if self._boundary() != boundary:
            raise ValueError('source boundary moved during startup parameter capture')

    @property
    def native_qualified(self):
        return False

    @property
    def teacher_admission(self):
        return False

    @property
    def native_qualification(self):
        return False

    @property
    def origins(self):
        return self._origins.clone()

    @property
    def rest_lengths_m(self):
        return self._rest.clone()

    @property
    def rest_length_m(self):
        return self._rest_total.clone()

    @property
    def reset_z(self):
        return self._reset_z.clone()

    @property
    def reset_distance_m(self):
        return self._reset_distance.clone()

    @property
    def distance_ref_m(self):
        return self._reset_distance.clamp_min(0.005)

    @property
    def previous_request(self):
        return self._previous.clone()

    @property
    def previous_request_tokens(self):
        return tuple(self._previous_tokens)

    @property
    def reference_sdk(self):
        return self._reference_sdk.clone()

    def _verify_profile(self):
        verify = getattr(self.profile, 'verify', None)
        sha = getattr(self.profile, 'digest', None)
        if type(self.profile) is not tenso_sourc_owner__TensorBackendProfile or self.profile.manifest_path != self._manifest_path or (not callable(verify)) or (type(sha) is not str) or (len(sha) != 64) or any((c not in '0123456789abcdef' for c in sha)):
            raise ValueError('explicit verifiable flat candidate profile digest required')
        verify()
        manifest = json.loads(tenso_sourc_owner__Path(self._manifest_path).read_text())
        if manifest.get('schema') != 'dh116-runtime-backend-v2' or manifest.get('teacher_admission') is not False or manifest.get('native_mechanical_receipt') is not None or (manifest.get('physics_kernel_changes') is not False) or (manifest.get('flat_abi') != 'contact_pair_samples_flat_cuda_v1') or (tenso_sourc_owner__digest(manifest) != sha) or (sha != self._profile_sha256):
            raise ValueError('strictly unqualified source-locked flat candidate profile required')

    def _boundary(self):
        if getattr(self.profile, 'digest', None) != self._profile_sha256 or id(self.runtime.solver) != self._parent_solver_id or self.runtime.solver.num_envs != self.n:
            raise ValueError('candidate profile or actual parent solver changed')
        (r, solver) = (self.runtime, self.runtime.solver)
        getter = getattr(self.sim, 'get_physics_dt', None)
        dt = float(getter()) if callable(getter) else 1 / 480
        fp32_dt = float(np.float32(1 / 480))
        timer_dt = float(np.float32(dt)) if callable(getter) else dt
        if dt not in (1 / 480, fp32_dt) or not np.isfinite(float(self.sim.current_time)):
            raise ValueError('actual 480Hz SDK dt and finite native time required')
        if r._failed or solver.last_error != 0 or float(solver.time) != r._time or (self.sim.current_time_step_index != r.physics_step) or (abs(float(self.sim.current_time) - r.physics_step * timer_dt) > max(1e-09, r.physics_step * 1e-14)):
            raise ValueError('shared SDK/solver source boundary changed or failed')
        return tenso_sourc_owner__SharedSourceIdentity(id(solver), solver.num_envs, tuple(range(solver.num_envs)), r.run_uuid, r.episodes, self.rod_uids, tuple(r.resets), tuple(r.ticks), r.physics_step, float(self.sim.current_time), dt, timer_dt, r.physics_step / 480, float(solver.time), self.profile.digest, self.scope)

    def _parameters(self, raw, boundary):
        import json
        checked = validate_snapshot(raw, expected_num_envs=boundary.parent_num_envs, expected_nodes=41, expected_time_s=boundary.solver_time_s)
        cfg = json.loads((ROOT / 'configs/tasks' / (self.grasp + '.json')).read_text())['rod']
        expected = dict(contact_radius_m=cfg['radius_m'], gravity_world_m_s2=cfg['gravity_world_m_s2'], contact_mu_static=cfg['mu_s'], contact_mu_kinetic=cfg['mu_k'], contact_v_slip_m_s=cfg['v_slip_m_s'], contact_kappa=cfg['kappa'], stretch_ea_n=cfg['axial_stiffness_n'], bend_ei_n_m2=cfg['bend_stiffness_n_m2'])
        if checked['fixed_node_indices'] or any((not np.allclose(raw[k], v, atol=1e-09, rtol=2e-06) for (k, v) in expected.items())):
            raise ValueError('native shared material/gravity/radius/free-root source differs')
        rest = np.asarray(raw['rest_lengths_m'], dtype=np.float64)
        mass = np.concatenate((rest[:, :1] / 2, (rest[:, :-1] + rest[:, 1:]) / 2, rest[:, -1:] / 2), 1) * cfg['linear_density_kg_m']
        if not np.allclose(1 / np.asarray(raw['inverse_masses_kg_inv']), mass, rtol=1e-06, atol=1e-10):
            raise ValueError('actual shared device masses differ')
        return tenso_sourc_owner__torch.as_tensor(rest, dtype=tenso_sourc_owner__torch.float32, device=self.device).clone()

    def _read_parameters(self, boundary):
        raw = self.runtime.solver.parameter_snapshot()
        rest = self._parameters(raw, boundary)
        total = tenso_sourc_owner__torch.as_tensor(np.asarray(raw['rest_lengths_m']).sum(axis=1), dtype=tenso_sourc_owner__torch.float64, device=self.device).clone()
        return (rest, total)

    def _selected(self, env_ids):
        ids = tuple(env_ids)
        if not ids or any((type(i) is not int or not 0 <= i < self.n for i in ids)) or len(set(ids)) != len(ids):
            raise ValueError('unique actual selected Reset row IDs required')
        return (ids, tenso_sourc_owner__torch.tensor(ids, device=self.device, dtype=tenso_sourc_owner__torch.int64))

    def _floating(self, value, shape, name):
        if not isinstance(value, tenso_sourc_owner__torch.Tensor) or tuple(value.shape) != shape or value.device != self.device or (not value.is_floating_point()):
            raise ValueError('actual same-device ' + name + ' tensor required')
        return value.detach()

    @staticmethod
    def _identities(boundary):
        return tuple((tenso_sourc_owner__ContextIdentity(boundary.run_uuid, i, boundary.episode_uids[i], boundary.reset_ids[i], boundary.rod_uids[i], boundary.sdk_step, boundary.rod_ticks[i], boundary.policy_time_s, boundary.solver_time_s) for i in range(boundary.parent_num_envs)))

    def reset(self, env_ids, distance_m):
        (ids, selected) = self._selected(env_ids)
        distance = self._floating(distance_m, (len(ids),), 'measured Reset distance').double()
        if bool((~tenso_sourc_owner__torch.isfinite(distance) | (distance < 0)).any()):
            raise ValueError('actual finite nonnegative measured Reset distance required')
        self._verify_profile()
        boundary = self._boundary()
        identities = self._identities(boundary)
        for lane in ids:
            prior = self._reset_identities[lane]
            current = identities[lane]
            if current.rod_tick != 0 or (prior is not None and (current.reset_id == prior.reset_id or current.episode_uid == prior.episode_uid)):
                raise ValueError('fresh zero-tick nodes and a new native Reset epoch required')
        (rest, total) = self._read_parameters(boundary)
        keep = tenso_sourc_owner__torch.ones(self.n, dtype=tenso_sourc_owner__torch.bool, device=self.device)
        keep[selected] = False
        if not tenso_sourc_owner__torch.equal(rest[keep], self._rest[keep]) or not tenso_sourc_owner__torch.equal(total[keep], self._rest_total[keep]):
            raise ValueError('selected Reset changed peer material rest references')
        p = tenso_sourc_owner__torch.as_tensor(self.runtime.solver.positions(), device=self.device).detach()
        if p.shape != (self.n, 41, 3):
            raise ValueError('actual complete source-owned Reset nodes required')
        positions = p[selected]
        z = tenso_sourc_owner__torch.stack((positions[:, :, 2].median(dim=1).values, positions[:, :, 2].amin(dim=1)), dim=1).double()
        if not bool(tenso_sourc_owner__torch.isfinite(positions).all()):
            raise ValueError('actual complete finite selected Reset node positions required')
        if self._boundary() != boundary:
            raise ValueError('source boundary moved during selected Reset capture')
        self._reset_z[selected] = z
        self._reset_distance[selected] = distance
        self._reference_sdk[selected] = boundary.sdk_step
        self._reference_policy_time[selected] = boundary.policy_time_s
        self._previous[selected] = 0.0
        self._rest[selected] = rest[selected]
        self._rest_total[selected] = total[selected]
        for lane in ids:
            self._reset_identities[lane] = identities[lane]
            self._previous_tokens[lane] = 'measured-reset:' + identities[lane].episode_uid
        return tuple((identities[i] for i in ids))

    def set_previous_request(self, actions, token):
        values = self._floating(actions, (self.n, 24), 'normalized actual request')
        if type(token) is not str or not token.strip() or bool((~tenso_sourc_owner__torch.isfinite(values) | (values.abs() > 1)).any()):
            raise ValueError('one actual finite bounded normalized request per row and token required')
        if any((c is None for c in self._reset_identities)):
            raise ValueError('verified Reset references required before recording requests')
        self._boundary()
        self._previous = values.float().clone()
        self._previous_tokens = [token] * self.n

    def contexts(self):
        boundary = self._boundary()
        identities = self._identities(boundary)
        for (current, reset) in zip(identities, self._reset_identities):
            if reset is None or current.epoch != reset.epoch:
                raise ValueError('new native Reset epoch requires fresh source references')
        remaining = max(0.0, self.wall_cap_s - (perf_counter() - self.started)) / self.wall_cap_s
        if not tenso_sourc_owner__math.isfinite(remaining) or not 0.0 <= remaining <= 1.0:
            raise ValueError('finite original external budget clock required')
        values = tenso_sourc_owner__torch.full((self.n, 50), tenso_sourc_owner__torch.nan, dtype=tenso_sourc_owner__torch.float32, device=self.device)
        valid = tenso_sourc_owner__torch.zeros((self.n, 50), dtype=tenso_sourc_owner__torch.bool, device=self.device)
        valid[:, _REQUIRED] = True
        values[:, :4] = values.new_tensor([1.0, 0.0, 0.0, 0.0])
        values[:, 10:12] = 0.0
        values[:, 12:36] = self._previous
        values[:, 36:38] = self._reset_z.float()
        values[:, 38] = self._reset_distance.clamp_min(0.005).float()
        values[:, 39] = self._reset_distance.float()
        values[:, 40:43] = values.new_tensor([0.03, 5.0, 0.0005])
        values[:, 43] = ((boundary.policy_time_s - self._reference_policy_time) / self.episode_cap_s).float()
        values[:, 44] = remaining
        values[:, 45] = 0.0
        values[:, 48:50] = 0.0
        if bool((valid & ~tenso_sourc_owner__torch.isfinite(values) | ~valid & ~tenso_sourc_owner__torch.isnan(values)).any()):
            raise ValueError('finite actual context with explicit NA values required')
        if self._boundary() != boundary:
            raise ValueError('source boundary moved during tensor context capture')
        return (values, valid, identities)

    def validate_fresh(self, packet):
        source = packet if type(packet) is tenso_sourc_owner__SharedSourceIdentity else getattr(packet, 'source', None)
        if type(source) is not tenso_sourc_owner__SharedSourceIdentity or source != self._boundary():
            raise ValueError('shared parent/row/episode Reset epoch or clock changed')
        return packet

def _all_nodes(mask, batch, device):
    if not isinstance(mask, tenso_nativ_teach__torch.Tensor) or mask.shape != (batch, 41) or mask.dtype != tenso_nativ_teach__torch.bool or (mask.device != device) or (not bool(mask.all())):
        raise ValueError('all 41 original material nodes required; no mask may hide a bad node')
    return mask

def initialize_native_network(self, identity, backend_profile, context_mode, initial_std):
    PrivilegedTeacher.__init__(self, initial_std=initial_std)
    self._model_identity = identity
    self._backend_profile = backend_profile
    self.backend_sha256 = backend_profile.binary_sha256
    self.context_mode = context_mode

    def common_encoder():
        return tenso_nativ_teach__nn.Sequential(tenso_nativ_teach__nn.Linear(66, 256), tenso_nativ_teach__nn.ELU(), tenso_nativ_teach__nn.Linear(256, 256), tenso_nativ_teach__nn.ELU())

    def context_encoder():
        return tenso_nativ_teach__nn.Sequential(tenso_nativ_teach__nn.Linear(100, 64), tenso_nativ_teach__nn.ELU(), tenso_nativ_teach__nn.Linear(64, 64), tenso_nativ_teach__nn.ELU())
    (self.actor, self.critic) = (common_encoder(), common_encoder())
    (self.actor_rod, self.critic_rod) = (tenso_nativ_teach___MaterialEncoder(64), tenso_nativ_teach___MaterialEncoder(64))
    (self.actor_context, self.critic_context) = (context_encoder(), context_encoder())
    (self.actor_head, self.critic_head) = (tenso_nativ_teach__nn.Linear(448, 24), tenso_nativ_teach__nn.Linear(448, 1))

class TensorNativeTeacher(EagerRuntimePolicy, PrivilegedTeacher):

    def __init__(self, config, *, backend_profile, num_envs, reward_config):
        validate_runtime_execution(config=config)
        if type(backend_profile) is not tenso_nativ_teach__TensorBackendProfile:
            raise ValueError('explicit independently verified flat CUDA profile required')
        backend_profile.verify()
        reward = validate_reward_config(reward_config)
        model = config.get('model', {})
        grasp = config.get('grasp')
        expected = dict(common_size=371, action_size=24, hidden=[256, 256], rod_feature_size=13, rod_hidden=64)
        if type(num_envs) is not int or not 2 <= num_envs <= 512 or grasp not in ('pinch', 'tripod', 'power') or (config.get('task_kind') != 'deformable_rod') or (config.get('use_cameras') is not False) or (set(model) != set(expected) | {'initial_std'}) or any((model[k] != v for (k, v) in expected.items())) or (type(model['initial_std']) not in (int, float)) or (not 0.001 <= model['initial_std'] <= 0.3) or (reward.profile != SMOKE) or (reward.finger_goal is None) or (reward.finger_goal.grasp != grasp):
            raise ValueError('unchanged shared finger-goal model/reward configuration required')
        identity = dict(policy_version='dh116-tensor-native-context-teacher-v1', source_family='shared_batched_solver_v1', grasp=grasp, task_kind='deformable_rod', parent_num_envs=num_envs, action_size=24, common_size=371, measured_common_size=66, raw_common_unknown='NaN_false_305', context_size=50, context_mode=SMOKE, input_keys=list(tenso_nativ_teach__INPUT_KEYS), backend_sha256=backend_profile.binary_sha256, backend_profile_sha256=backend_profile.digest, source_tree_sha256=backend_profile.source_tree_sha256, config_semantic_sha256=tenso_nativ_teach__digest(config), reward_config_sha256=reward.config_sha256, rod_position_frame='world_minus_environment_origin', other_rod_vector_frames='world', physics_coordinate_frame='world', checkpoint_namespace='tensor_native_teacher_' + grasp + '_v1', native_ready=False, teacher_admission=False, grasp_qualified=False, backend_scope='flat_candidate_unqualified', architecture=dict(common=[66, 256, 256], rod=[13, 64, 64], context=[100, 64, 64], fusion=448, actor=24, critic=1))
        identity['dependency_source_sha256'] = {name: file_sha256(source_file(name)) for name in ('rl/teacher.py','rl/device.py','physics/solver.py','rl/ppo.py','env/contracts.py')}
        identity['model_identity_sha256'] = tenso_nativ_teach__digest(identity)
        initialize_native_network(self, identity, backend_profile, SMOKE, model['initial_std'])

    @property
    def model_identity(self):
        return deepcopy(self._model_identity)

    def _parts(self, observations):
        if not isinstance(observations, dict) or set(observations) != set(tenso_nativ_teach__INPUT_KEYS):
            raise ValueError('new Teacher requires exact six input keys')
        common = _common(observations['common371'], observations['common_valid'])
        nodes = observations['rod_nodes']
        if not isinstance(nodes, tenso_nativ_teach__torch.Tensor) or nodes.shape != (len(common), 41, 13) or nodes.dtype != common.dtype or (nodes.device != common.device) or (not bool(tenso_nativ_teach__torch.isfinite(nodes).all())):
            raise ValueError('all 41x13 packed material features must be finite')
        mask = _all_nodes(observations['rod_mask'], len(common), common.device)
        context = observations['task_context']
        tenso_nativ_teach__validate_context_tensors(context, observations['task_context_valid'], mode=self.context_mode)
        if context.device != common.device or len(context) != len(common):
            raise ValueError('context/common batch/device differs')
        return (common, nodes, mask)

    def latent_mean(self, observations):
        (common, nodes, mask) = self._parts(observations)
        context = tenso_nativ_teach__encode_context(observations['task_context'], observations['task_context_valid'])
        return self.actor_head(tenso_nativ_teach__torch.cat((self.actor(common), self.actor_rod(nodes, mask), self.actor_context(context)), dim=-1))

    def value(self, observations):
        (common, nodes, mask) = self._parts(observations)
        context = tenso_nativ_teach__encode_context(observations['task_context'], observations['task_context_valid'])
        return self.critic_head(tenso_nativ_teach__torch.cat((self.critic(common), self.critic_rod(nodes, mask), self.critic_context(context)), dim=-1)).squeeze(-1)

    def get_extra_state(self):
        return self.model_identity

    def set_extra_state(self, state):
        if state != self._model_identity:
            raise ValueError('tensor Teacher checkpoint identity differs')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.set_extra_state(state_dict.get('_extra_state'))
        return tenso_nativ_teach__nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)
