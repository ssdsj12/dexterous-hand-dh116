"""Opt-in full-articulation readback, independent of the controller/main path.

Use the two EXISTING hand.root_physx_view objects after the original refresh,
before reset. This module never refreshes, advances, writes or creates a World.
No SDK/Torch imports or simulation-context construction occur, even when enabled.
``source_reader(side, actual_articulation_path)`` optionally reads the existing
stage identity/units and installed/active SDK source identities. Its evidence,
like an SDK-shaped CPU fixture, cannot attest Native execution.

Primary source: installed omni.physics.tensors/impl/api.py, SHA256
bb5e684dce25e0ad7f3e03ec395ec4729d55203f3b3a58bfa824b8d106c0e54b:
529-708 metadata/types; 744-963 limits/drives; 1379-1435 q/qd;
1645-1760 targets/explicit input/solver projection; 1825-1977 M/C/G;
2043-2175 link wrench/mass/COM/inertia. Source audit identity is retained below.
These are API/source references, NOT a claim this runner loaded those files.

M/C/G are current-state raw SDK values, with compensation sign, gravity flags,
armature inclusion, floating-root ordering and dynamics qualification unresolved.
They are not candidate-dependent future dynamics or a PhysX-equivalent model.
Solver forces are not isolated motors, and no field is hardware measurement.
Implicit PD estimates are deliberately NA: their pre-step calculation clock and
actuator clipping limits are not established by a post-step backend read.
Raw evidence may contain NaN/Inf; invalid DTO fields remain NA. Raw is not a
strict-JSON export: the caller's archive encoder must retain nonfinite paths/bits.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from .linear_coupling import ACTIVE, JOINTS
from .tabletop_bimanual_mpc_state import SampleClock, StateField


SOURCE_AUDIT_SHA256 = '4ba01cef794145679c3733a375154b383fe610bd8abb2a83238a0e0ebde19f9f'
_HANDS = ('left', 'right')
# Names from tabletop_palm_mount.py:9-10; import would unnecessarily import Torch.
# Names are matched to actual metadata, never treated as observed DOF ordering.
_WRIST = ('wrist_x', 'wrist_y', 'wrist_z', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
_META = ('fixed_base', 'link_count', 'link_names', 'link_indices', 'joint_count',
         'joint_names', 'joint_indices', 'joint_types', 'joint_dof_offsets',
         'joint_dof_counts', 'dof_count', 'dof_names', 'dof_indices', 'dof_types')
# field, actual getter, trailing dimensions, explicit raw unit family.
_GETTERS = (
    ('q', 'get_dof_positions', 'D', 'mixed_coordinate'),
    ('qd', 'get_dof_velocities', 'D', 'mixed_coordinate/s'),
    ('backend_position_target', 'get_dof_position_targets', 'D', 'mixed_coordinate'),
    ('backend_velocity_target', 'get_dof_velocity_targets', 'D', 'mixed_coordinate/s'),
    ('explicit_effort_input', 'get_dof_actuation_forces', 'D', 'mixed_generalized_effort'),
    ('dof_type', 'get_dof_types', 'D', 'enum_rotation0_translation1'),
    ('drive_type', 'get_drive_types', 'D', 'enum_none0_force1_acceleration2'),
    ('stiffness', 'get_dof_stiffnesses', 'D', 'drive_type_dependent_mixed_gain'),
    ('damping', 'get_dof_dampings', 'D', 'drive_type_dependent_mixed_gain*s'),
    ('drive_effort_cap', 'get_dof_max_forces', 'D', 'drive_type_dependent_mixed_cap'),
    ('velocity_cap', 'get_dof_max_velocities', 'D', 'mixed_coordinate/s'),
    ('limits', 'get_dof_limits', 'D2', 'mixed_coordinate_lower_upper'),
    ('armature', 'get_dof_armatures', 'D', 'mixed_inertia_raw_unqualified'),
    ('joint_friction', 'get_dof_friction_coefficients', 'D', 'joint_friction_coefficient_not_contact_mu'),
    ('generalized_mass_matrix_raw', 'get_generalized_mass_matrices', 'GG', 'mixed_generalized_inertia_raw_unqualified'),
    ('coriolis_compensation_raw', 'get_coriolis_and_centrifugal_compensation_forces', 'G', 'mixed_generalized_compensation_raw_unqualified'),
    ('gravity_compensation_raw', 'get_gravity_compensation_forces', 'G', 'mixed_generalized_compensation_raw_unqualified'),
    ('solver_projected_joint_force_raw', 'get_dof_projected_joint_forces', 'D', 'mixed_generalized_effort_not_motor_only'),
    ('incoming_joint_wrench_raw', 'get_link_incoming_joint_force', 'L6', 'force_xyz|torque_xyz_raw_unqualified'),
    ('link_mass_raw', 'get_masses', 'L', 'backend_mass_raw_unqualified'),
    ('link_com_pose_xyzw_raw', 'get_coms', 'L7', 'stage_distance_xyz|quaternion_xyzw'),
    ('link_inertia_9_raw', 'get_inertias', 'L9', 'backend_inertia9_frame_unqualified'),
)
_GAPS = {
    'implicit_pd_estimate': 'implicit PD estimate only; calculation clock and actuator-model clipping limit unavailable; never actual motor/hardware torque',
    'actual_motor_torque': 'SDK explicit input and solver joint-force projection do not isolate implicit drive motor torque; no hardware measurement',
    'candidate_future_dynamics': 'getters read current physical state only; no arbitrary candidate q/qd evaluator or qualified forward dynamics',
}


class _ClockChanged(RuntimeError):
    pass


class _SourceChanged(RuntimeError):
    pass


def _source_record(value):
    # Only common stage/units are compared ACROSS hands. Full per-hand records
    # remain independent, so different left/right asset identities are allowed.
    common = {k:value.get(k) for k in ('stage_identity','meters_per_unit','kilograms_per_unit')}
    for key in ('meters_per_unit','kilograms_per_unit'):
        if isinstance(common[key],Real) and not isinstance(common[key],bool):
            common[key] = float(common[key])
    normalized = dict(value,**common)
    def fingerprint(record):
        payload = json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()
    return dict(value=value,available=True,fingerprint=fingerprint(normalized),
                common=common,common_fingerprint=fingerprint(common))


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return deepcopy(value)


def _owned(value):
    """Copy immediately, before another SDK getter can reuse internal storage."""
    dtype = str(getattr(value, 'dtype', 'unknown'))
    device = str(getattr(value, 'device', 'cpu' if isinstance(value, np.ndarray) else 'unknown'))
    if hasattr(value, 'detach'):
        value = value.detach().clone().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    array = np.array(value, copy=True)
    return dict(value=array.tolist(), raw_shape=list(array.shape),
                raw_dtype=dtype, source_device=device, available=True, missing_reason=None)


def _enum_value(value):
    """Normalize exact integers/known enum members, retaining raw type evidence."""
    evidence = dict(raw_type=type(value).__module__+'.'+type(value).__qualname__,
                    raw_value=_plain(value) if isinstance(value,(str,bool,Real)) else repr(value),
                    raw_repr=repr(value),raw_name=None,raw_enum_value=None,
                    integer_conversion=dict(attempted=False,value=None,error=None),
                    normalization=None,normalization_basis=None,
                    missing_reason=None)
    def type_name(item):
        return type(item).__module__+'.'+type(item).__qualname__
    def converted_integer(item,receipt):
        receipt['attempted'] = True
        try:receipt['value'] = int(item)
        except Exception as exc:
            receipt['error'] = type(exc).__name__+': '+str(exc)
            raise
        return receipt['value']
    try:
        if isinstance(value,(bool,np.bool_)):
            raise ValueError('boolean is not an integer enum')
        if isinstance(value,Integral) and not isinstance(value,Enum):
            normalized, kind = converted_integer(value,evidence['integer_conversion']), 'integer'
            evidence['normalization_basis'] = 'exact_integral_scalar'
        elif isinstance(value,Real) and not isinstance(value,Enum):
            if not math.isfinite(value) or value != math.trunc(value):
                raise ValueError('nonfinite/fractional value is not an exact integer enum')
            normalized, kind = converted_integer(value,evidence['integer_conversion']), 'exact_integral_real'
            evidence['normalization_basis'] = 'exact_integral_real_scalar'
        else:
            # Protocol evidence belongs only to enum-shaped inputs, avoiding a
            # copy of this schema for every ordinary integer getter element.
            evidence.update(raw_name_type=None,raw_enum_value_type=None,
                member_mapping_available=False,member_mapping_type=None,declared_member_keys=[],
                same_type_member_count=0,identity_member_names=[],equality_member_names=[],
                equality_result_types=[],declared_member_names=[],matched_member_checks=[])
            missing = object()
            name,raw = getattr(value,'name',missing),getattr(value,'value',missing)
            evidence['raw_name_available'],evidence['raw_enum_value_available'] = name is not missing,raw is not missing
            if name is not missing:
                evidence['raw_name'] = name if name is None or isinstance(name,str) else repr(name)
                evidence['raw_name_type'] = type_name(name)
            if raw is not missing:
                evidence['raw_enum_value'] = _plain(raw) if raw is None or isinstance(raw,(str,bool,Real)) else repr(raw)
                evidence['raw_enum_value_type'] = type_name(raw)
            members = getattr(type(value),'__members__',missing)
            evidence['member_mapping_available'] = members is not missing
            if members is not missing:evidence['member_mapping_type'] = type_name(members)
            if not isinstance(members,Mapping):
                raise ValueError('known enum requires a declared member mapping')
            evidence['declared_member_keys'] = [str(key) for key in members]
            if isinstance(raw,(bool,np.bool_)) or not isinstance(raw,Integral):
                raise ValueError('known enum must have an exact integer value')
            normalized = int(raw)
            matches = []
            for member_name,member in members.items():
                if type(member) is not type(value):
                    continue
                evidence['same_type_member_count'] += 1
                if value is member:evidence['identity_member_names'].append(str(member_name))
                equal = value == member
                equality_type = type_name(equal)
                if equality_type not in evidence['equality_result_types']:
                    evidence['equality_result_types'].append(equality_type)
                # Require an actual bool True, without coercing arrays, integer
                # truthiness, numpy.bool_ or NotImplemented to membership.
                if equal is not True:
                    continue
                evidence['equality_member_names'].append(str(member_name))
                evidence['normalization_basis'] = 'same_type_declared_member_equality_and_exact_integer_agreement'
                member_raw = getattr(member,'value',None)
                check = dict(name=str(member_name),raw_value_type=type_name(member_raw),
                    integer_value=_plain(member_raw) if isinstance(member_raw,Real) else repr(member_raw),
                    identity_match=value is member,equality_is_bool_true=True,
                    integer_conversion=dict(attempted=False,value=None,error=None))
                evidence['matched_member_checks'].append(check)
                if isinstance(member_raw,(bool,np.bool_)) or not isinstance(member_raw,Integral):
                    raise ValueError('declared enum member must have an exact integer value')
                # Membership is checked BEFORE conversion. Equal pybind wrappers
                # need not share Python identity. Neither repr nor name supplies
                # an integer; every accepted wrapper/member value must agree.
                if int(member_raw)!=normalized:
                    raise ValueError('equal enum member integer values disagree')
                actual_integer = converted_integer(value,evidence['integer_conversion'])
                member_integer = converted_integer(member,check['integer_conversion'])
                if actual_integer!=normalized or member_integer!=normalized:
                    raise ValueError('enum integer conversion disagrees with declared member value')
                matches.append(str(member_name))
            if not matches:
                raise ValueError('not an integer or known enum member')
            kind = 'known_enum_member'
            evidence['declared_member_names'] = matches
            evidence['raw_value'] = dict(name=evidence['raw_name'],integer_value=normalized)
        evidence['normalization'] = kind
        return normalized, evidence
    except Exception as exc:
        evidence['missing_reason'] = str(exc)
        return evidence['raw_value'], evidence


def _metatype(value):
    result, evidence, errors = {}, {}, []
    for key in _META:
        item = getattr(value, key)
        if key in ('joint_types', 'dof_types'):
            pairs = [_enum_value(v) for v in item]
            item, evidence[key] = [p[0] for p in pairs], [p[1] for p in pairs]
            errors.extend(f'{key}[{i}]: '+p[1]['missing_reason'] for i,p in enumerate(pairs) if p[1]['missing_reason'])
        elif key.endswith('_indices'):
            item = dict(item)
        result[key] = _plain(item)
    result['enum_evidence'], result['enum_errors'] = evidence, errors
    return result


def _paths(values):
    result = []
    for value in values:
        if isinstance(value, str):
            if not value.startswith('/') or any(c in value for c in '*?[]'):
                raise ValueError('exact absolute actual paths required')
            result.append(value)
        elif isinstance(value, (list, tuple)):
            result.extend(_paths(value))
        else:
            raise ValueError('actual SDK path list required')
    if len(set(result)) != len(result):
        raise ValueError('duplicate actual paths')
    return result


def _indices(meta, prefix, count):
    names = meta[prefix+'_names']
    indices = meta[prefix+'_indices']
    if (len(names) != count or len(set(names)) != count or any(not isinstance(n,str) or not n for n in names)
            or indices != {n:i for i,n in enumerate(names)}):
        raise ValueError(prefix+' names/indices/count do not uniquely agree')
    return names


def _binding(metadata, *, side, clock, hand_names):
    """Validate every selected mapping against actual SDK names and path set."""
    paths = _paths(metadata['prim_paths'])
    n = metadata['count']
    if type(n) is not int or n < 1 or n != len(paths):
        raise ValueError('actual view count/path mismatch')
    scope = f'/World/envs/env_{clock.env_id}/{side}Hand'
    candidates = [i for i,p in enumerate(paths) if p == scope or p.startswith(scope+'/')]
    if len(candidates) != 1:
        raise ValueError('actual environment/hand articulation path binding missing or ambiguous')
    index = candidates[0]
    meta = metadata['metatypes'][index]
    if meta['enum_errors'] or metadata['shared_metatype']['enum_errors']:
        raise ValueError('invalid actual metadata enum: '+'; '.join(meta['enum_errors'] or metadata['shared_metatype']['enum_errors']))
    d, l = meta['dof_count'], meta['link_count']
    if (metadata['is_homogeneous'] is not True or type(d) is not int or d < 1
            or type(l) is not int or l < 1 or d != metadata['max_dofs'] or l != metadata['max_links']
            or type(meta['fixed_base']) is not bool):
        raise ValueError('unsupported/unknown actual homogeneous DOF/link/fixed-base layout')
    if any(meta[key] != metadata['shared_metatype'][key] for key in _META):
        raise ValueError('actual per-articulation metatype disagrees with shared metatype')
    names = _indices(meta, 'dof', d)
    links = _indices(meta, 'link', l)
    joints = _indices(meta, 'joint', meta['joint_count'])
    types = meta['dof_types']
    if len(types) != d or any(t not in (0,1) for t in types):
        raise ValueError('unknown actual DOF type; no mixed-unit assumption')
    offsets, counts = meta['joint_dof_offsets'], meta['joint_dof_counts']
    if len(offsets) != len(joints) or len(counts) != len(joints) or len(meta['joint_types']) != len(joints):
        raise ValueError('actual joint DOF offset/count metadata incomplete')
    coverage = []
    for offset,count in zip(offsets,counts):
        if (type(offset) is not int or type(count) is not int or count < 0
                or (count > 0 and (offset < 0 or offset+count > d))):
            raise ValueError('invalid actual joint DOF offset/count')
        # A fixed/zero-DOF joint's raw offset is not used as an array index.
        coverage.extend(range(offset,offset+count))
    if sorted(coverage) != list(range(d)):
        raise ValueError('actual joint offsets must cover each DOF exactly once')
    for key, actual in (('joint_names',names), ('body_names',links)):
        supplied = hand_names[key]
        if len(supplied) != len(actual) or len(set(supplied)) != len(supplied) or set(supplied) != set(actual):
            raise ValueError('actual hand '+key+' does not match backend names uniquely')
    dof_paths, link_paths = _paths(metadata['dof_paths']), _paths(metadata['link_paths'])
    def named_paths(names, actual):
        selected = []
        for name in names:
            matches = [p for p in actual if p.startswith(scope+'/') and p.rsplit('/',1)[-1] == name]
            if len(matches) != 1:
                raise ValueError('actual named path unbound/ambiguous: '+name+'; multi-DOF path conventions require further source qualification')
            selected.append(matches[0])
        return selected
    groups = {}
    for key, expected in (('full11',JOINTS), ('active6',ACTIVE), ('passive5',tuple(n for n in JOINTS if n not in ACTIVE)), ('wrist6',_WRIST)):
        available = all(n in names for n in expected)
        groups[key] = dict(names=list(expected), available=available,
            indices=[names.index(n) for n in expected] if available else None,
            source='tabletop_palm_mount named contract' if key == 'wrist6' else 'linear_coupling named contract',
            binding_source='actual get_metatype DOF names/indices matched uniquely',
            missing_reason=None if available else 'named source group not fully present in actual articulation')
    return dict(available=True, missing_reason=None, selected_view_index=index,
        articulation_path=paths[index], source_prim_paths=paths, dof_count=d, link_count=l,
        fixed_base=meta['fixed_base'], dof_names=names, dof_types=types,
        dof_paths=named_paths(names,dof_paths), link_names=links, link_paths=named_paths(links,link_paths),
        groups=groups, metatype=meta,
        hand_joint_to_dof=[names.index(n) for n in hand_names['joint_names']],
        hand_body_to_link=[links.index(n) for n in hand_names['body_names']])


def _units(binding, source, drive_types):
    types = binding.get('dof_types', [])
    si = (isinstance(source.get('stage_identity'),str) and bool(source['stage_identity'].strip())
          and all(isinstance(source.get(k),Real) and not isinstance(source[k],bool) and source[k] == 1.
                  for k in ('meters_per_unit','kilograms_per_unit')))
    force = [('N*m' if t == 0 else 'N') if si else
             ('backend_torque_unqualified' if t == 0 else 'backend_force_unqualified') for t in types]
    inertia = None
    if binding.get('fixed_base') is True and si:
        inertia = [[('kg*m^2' if a == b == 0 else 'kg' if a == b == 1 else 'kg*m') for b in types] for a in types]
    gains = []
    for i,t in enumerate(types):
        drive = drive_types[i] if drive_types is not None else None
        gains.append(('N*m/rad' if t == 0 else 'N/m') if si and drive == 1 else
                     'acceleration_drive_gain_unqualified' if drive == 2 else 'drive_gain_unqualified')
    return dict(position=['rad' if t == 0 else 'stage_distance' for t in types],
        velocity=['rad/s' if t == 0 else 'stage_distance/s' for t in types],
        generalized_force=force, force_drive_stiffness=gains,
        force_drive_damping=[g+'*s' for g in gains], mass_matrix=inertia,
        generalized_root_order=None, stage_meters_per_unit=source.get('meters_per_unit'),
        stage_kilograms_per_unit=source.get('kilograms_per_unit'), si_stage_units_verified=si,
        non_SI_backend_normalization_qualified=False,
        note='no SI conversion applied; SDK angular coordinates rad, USD authored angles may be degrees; floating root units/order unqualified')


def _field(clock, side, name, unit, *, value=None, reason=None, source='articulation readback'):
    return StateField(value=value, source=source, unit=unit,
        frame='joint_attachment_unqualified' if name == 'incoming_joint_wrench_raw' else 'backend_generalized_or_link_coordinates',
        timestamp_s=clock.world_time_s if value is not None else None, valid=value is not None,
        env_id=clock.env_id, episode_uid=clock.episode_uid, clock=clock,
        kind='measured' if value is not None else 'unavailable', missing_reason=reason,
        origin='unqualified_joint_attachment' if name == 'incoming_joint_wrench_raw' else 'not_applicable')


@dataclass(frozen=True)
class ArticulationMeasurementSample:
    clock: SampleClock
    fields: object
    raw: dict
    qualification: object


class ArticulationMeasurementProvider:
    """One synchronous full-view transaction, no persistent tensor/sample cache.

    clock_reader(env_id) must read actual current World/SDK/episode identity.
    capture_batch brackets EVERY read against ALL requested live clocks and
    commits history only after the entire batch succeeds. Inactive/reset slots
    emit NA without selecting backend values. Terminal snapshots can be archived
    but never grant planning permission. Rebuild after stop/play/source changes.
    """
    def __init__(self, *, hands, clock_reader, source_reader=None, enabled=False, physics_dt=1/480):
        if type(enabled) is not bool:
            raise ValueError('explicit bool enabled required')
        if isinstance(physics_dt,bool) or not math.isfinite(physics_dt) or not math.isclose(physics_dt,1/480,rel_tol=0.,abs_tol=1e-12):
            raise ValueError('original 480 Hz physics dt required')
        # Disabled construction must not inspect supplied backends/callables.
        self.hands, self.clock_reader, self.source_reader = hands, clock_reader, source_reader
        self.enabled, self.physics_dt = enabled, float(physics_dt)
        self._previous, self._bindings = {}, {}
        self._sources, self._common_source = {}, None
        self.last_sample = None
        self._capturing = False

    def capture(self, *, clock, active=True):
        return self.capture_batch(clocks=(clock,), active=(active,))[0]

    def _transition(self, clock):
        previous = self._previous.get(clock.env_id)
        if previous is None:
            return
        if previous.run_uuid != clock.run_uuid:
            raise ValueError('run identity changed; rebuild provider')
        elapsed = clock.world_step-previous.world_step
        if elapsed <= 0 or clock.sdk_step <= previous.sdk_step:
            raise ValueError('stale/repeated physical clock')
        if clock.sdk_step-previous.sdk_step != elapsed or any(not math.isclose(
                getattr(clock,k)-getattr(previous,k), elapsed*self.physics_dt, rel_tol=0.,abs_tol=1e-7)
                for k in ('world_time_s','sdk_time_s')):
            raise ValueError('World/SDK clock step/time alignment changed')
        if clock.episode_uid != previous.episode_uid:
            if clock.reset_generation <= previous.reset_generation:
                raise ValueError('episode changed without advanced reset generation')
        elif clock.reset_generation != previous.reset_generation or previous.terminal:
            raise ValueError('reset/terminal episode cannot resume')
        if clock.policy_step < previous.policy_step:
            raise ValueError('policy step moved backwards')

    def capture_batch(self, *, clocks, active=None):
        self.last_sample = None
        clocks = tuple(clocks)
        if not clocks or not all(isinstance(c,SampleClock) for c in clocks):
            raise ValueError('nonempty SampleClock sequence required')
        if len({c.env_id for c in clocks}) != len(clocks):
            raise ValueError('duplicate environment in capture batch')
        if len({(c.run_uuid,c.world_step,c.sdk_step,c.world_time_s,c.sdk_time_s) for c in clocks}) != 1:
            raise ValueError('all batch environments must share one physics frame')
        active = (True,)*len(clocks) if active is None else tuple(active)
        if len(active) != len(clocks) or any(type(v) is not bool for v in active):
            raise ValueError('boolean active mask required per environment')
        if self._capturing:
            raise ValueError('nested/concurrent articulation capture unsupported')
        live = tuple(c for c,a in zip(clocks,active) if self.enabled and a and not c.reset)
        for c in live:
            self._transition(c)
        brackets = []
        def current():
            for expected in live:
                actual = self.clock_reader(expected.env_id)
                if not isinstance(actual,SampleClock) or actual != expected:
                    raise _ClockChanged('actual source clock/episode changed during articulation capture')
        def read(name, function):
            current()
            before = [asdict(c) for c in live]
            try:
                return function()
            finally:
                current()
                brackets.append(dict(name=name,before=before,after=[asdict(c) for c in live]))
        def source_read(side,path):
            try:
                if path is None:
                    raise ValueError('source identity unavailable without unique actual articulation path')
                if self.source_reader is None:
                    raise ValueError('no actual stage/SDK source reader supplied')
                value = read(side+'.source_identity',lambda: _plain(self.source_reader(side,path)))
                if not isinstance(value,dict):
                    raise ValueError('actual source identity dictionary required')
                return _source_record(value)
            except _ClockChanged:
                raise
            except Exception as exc:
                return dict(value=dict(missing_reason=type(exc).__name__+': '+str(exc)),
                    available=False,fingerprint=None,common=None,common_fingerprint=None)
        def validate_sources(records):
            common = {r['common_fingerprint'] for r in records.values() if r['available']}
            if len(common) > 1:
                raise _SourceChanged('common stage/source units disagree across hands or environments')
            if self._common_source is not None and common and common != {self._common_source}:
                raise _SourceChanged('common stage/source units changed; rebuild provider')
            for identity,record in records.items():
                if identity in self._sources and (not record['available'] or record['fingerprint'] != self._sources[identity]):
                    raise _SourceChanged('actual per-hand source identity changed or became unavailable; rebuild provider')
        pending_bindings = dict(self._bindings)
        self._capturing = True
        try:
            current()
            shared, views, source_paths, source_before = {}, {}, {}, {}
            # Acquire source identities BEFORE metadata/numeric getters, and
            # re-read AFTER the whole batch. No old identity/units are replayed.
            for side in _HANDS if live else ():
                hand = self.hands[side]
                views[side] = read(side+'.root_physx_view',lambda: hand.root_physx_view)
                try:
                    paths = read(side+'.source_prim_paths',lambda: _paths(views[side].prim_paths))
                except _ClockChanged:
                    raise
                except (ValueError,TypeError,AttributeError):
                    paths = []
                for c in live:
                    scope = f'/World/envs/env_{c.env_id}/{side}Hand'
                    matches = [p for p in paths if p == scope or p.startswith(scope+'/')]
                    path = matches[0] if len(matches) == 1 else None
                    identity = (side,c.env_id)
                    source_paths[identity] = path
                    source_before[identity] = source_read(side,path)
            validate_sources(source_before)
            for side in _HANDS if live else ():
                hand = self.hands[side]
                view = views[side]
                metadata = {}
                try:
                    for name in ('count','max_dofs','max_links','is_homogeneous','prim_paths','dof_paths','link_paths'):
                        metadata[name] = read(side+'.'+name, lambda name=name: _plain(getattr(view,name)))
                    metadata['shared_metatype'] = read(side+'.shared_metatype',lambda: _metatype(view.shared_metatype))
                    metadata['metatypes'] = [read(side+f'.get_metatype({i})',lambda i=i: _metatype(view.get_metatype(i))) for i in range(metadata['count'])]
                    hand_names = read(side+'.hand_names', lambda: dict(joint_names=list(hand.joint_names),body_names=list(hand.body_names)))
                    metadata_error = None
                except _ClockChanged:
                    raise
                except Exception as exc:
                    hand_names = {}
                    metadata_error = type(exc).__name__+': '+str(exc)
                getters = {}
                for field,name,shape,unit in _GETTERS:
                    try:
                        record = read(side+'.'+name, lambda name=name: _owned(getattr(view,name)()))
                    except _ClockChanged:
                        raise
                    except Exception as exc:
                        record = dict(value=None,raw_shape=None,raw_dtype=None,source_device=None,
                            available=False,missing_reason=type(exc).__name__+': '+str(exc))
                    record.update(source=f'hand[{side}].root_physx_view.{name}()', unit=unit)
                    getters[name] = record
                shared[side] = dict(metadata=metadata,metadata_error=metadata_error,
                    hand_names=hand_names,getters=getters)
            samples = []
            for c,a in zip(clocks,active):
                reason = 'provider disabled' if not self.enabled else 'inactive slot' if not a else 'reset snapshot' if c.reset else None
                fields = {}
                for side in _HANDS:
                    for name,getter,shape,unit in _GETTERS:
                        fields[side+'.'+name] = _field(c,side,name,unit,reason=reason or 'source unavailable')
                    for name,gap in _GAPS.items():
                        fields[side+'.'+name] = _field(c,side,name,'unqualified',reason=reason or gap)
                raw = dict(source_scope='simulation_privileged_current_state_readback_not_hardware',
                    source_audit_sha256=SOURCE_AUDIT_SHA256,
                    array_scope='batch-view raw arrays in actual source_prim_paths order; fields select one environment',
                    read_brackets=deepcopy(brackets),hands={})
                if not reason:
                    for side in _HANDS:
                        data = deepcopy(shared[side])
                        try:
                            if data['metadata_error']:
                                raise ValueError(data['metadata_error'])
                            binding = _binding(data['metadata'],side=side,clock=c,hand_names=data['hand_names'])
                        except (ValueError,KeyError,TypeError) as exc:
                            binding = dict(available=False,missing_reason=str(exc))
                        if binding['available']:
                            identity = (side,c.env_id)
                            if binding['articulation_path'] != source_paths[identity]:
                                raise _SourceChanged('actual source path and metadata binding changed during capture')
                            previous = self._bindings.get(identity)
                            if previous is not None and previous != binding:
                                raise ValueError('actual articulation path/name/metatype binding changed; rebuild provider')
                            pending_bindings[identity] = deepcopy(binding)
                        source = deepcopy(source_before[(side,c.env_id)]['value'])
                        data.update(binding=binding,source_identity=source)
                        data['semantics'] = dict(current_state_only=True,compensation_sign_qualified=False,
                            armature_inclusion_qualified=False,gravity_flags_qualified=False,solver_force_motor_isolated=False,
                            incoming_wrench_frame='joint attachment frame unqualified; not world COM wrench',
                            implicit_pd='estimate only; no post-solver measurement claimed')
                        drive = None
                        if binding['available']:
                            n,d,l = data['metadata']['count'],binding['dof_count'],binding['link_count']
                            g = d if binding['fixed_base'] else d+6
                            dims = dict(D=(n,d),D2=(n,d,2),G=(n,g),GG=(n,g,g),L=(n,l),L6=(n,l,6),L7=(n,l,7),L9=(n,l,9))
                            i = binding['selected_view_index']
                            type_record = data['getters']['get_dof_types']
                            if type_record['available']:
                                types = np.asarray(type_record['value'])
                                normalized_types = []
                                enum_errors = []
                                if types.shape == (n,d):
                                    pairs = [_enum_value(t) for t in types[i].tolist()]
                                    normalized_types = [p[0] for p in pairs]
                                    type_record['selected_enum_evidence'] = [p[1] for p in pairs]
                                    enum_errors = [p[1]['missing_reason'] for p in pairs if p[1]['missing_reason']]
                                if types.shape != (n,d) or enum_errors or normalized_types != binding['dof_types']:
                                    binding.update(available=False,missing_reason='actual DOF type getter and metatype disagree')
                            for name,getter,shape,unit in _GETTERS:
                                record = data['getters'][getter]
                                error = binding.get('missing_reason') or record['missing_reason']
                                value = None
                                if error is None:
                                    array = np.asarray(record['value'])
                                    if (array.shape != dims[shape] or not np.issubdtype(array.dtype,np.number)
                                            or np.issubdtype(array.dtype,np.complexfloating) or not np.isfinite(array[i]).all()):
                                        error = 'actual getter shape or selected-environment finite-data validation failed'
                                    elif name in ('drive_type','dof_type') and not np.isin(array[i],(0,1,2) if name == 'drive_type' else (0,1)).all():
                                        error = 'unknown actual enum value'
                                    else:
                                        value = array[i].tolist()
                                if name == 'drive_type':
                                    drive = value
                                fields[side+'.'+name] = _field(c,side,name,unit,value=value,reason=error,source=record['source'])
                        else:
                            for name,getter,shape,unit in _GETTERS:
                                fields[side+'.'+name] = _field(c,side,name,unit,reason=binding['missing_reason'])
                        data['unit_layout'] = _units(binding,source,drive)
                        raw['hands'][side] = data
                raw['read_brackets'] = deepcopy(brackets)
                qualification = MappingProxyType(dict(native_attested=False,dynamics_qualified=False,
                    solver_motor_torque_qualified=False,candidate_future_dynamics_qualified=False,planning_eligible=False))
                samples.append(ArticulationMeasurementSample(c,MappingProxyType(fields),raw,qualification))
            current()
            source_after = {identity:source_read(identity[0],path) for identity,path in source_paths.items()}
            validate_sources(source_after)
            for identity,before in source_before.items():
                after = source_after[identity]
                if before['available'] != after['available'] or before['fingerprint'] != after['fingerprint']:
                    raise _SourceChanged('actual source identity/units changed during articulation capture')
            current()
            for sample in samples:
                sample.raw['read_brackets'] = deepcopy(brackets)
                for side,data in sample.raw['hands'].items():
                    identity = (side,sample.clock.env_id)
                    data['source_brackets'] = dict(path=source_paths[identity],
                        before=deepcopy(source_before[identity]),after=deepcopy(source_after[identity]))
            self._previous = dict(self._previous)
            for c in live:
                self._previous[c.env_id] = c
            self._bindings = pending_bindings
            self._sources = dict(self._sources)
            for identity,record in source_before.items():
                if record['available']:
                    self._sources[identity] = record['fingerprint']
                    self._common_source = record['common_fingerprint']
            self.last_sample = samples[-1]
            return tuple(samples)
        finally:
            self._capturing = False
