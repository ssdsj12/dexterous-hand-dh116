"""Opt-in, read-only scene evidence for the privileged Teacher (CPU import safe).

This module never refreshes/steps simulation, changes a schema/property/setting,
adds a sensor to the existing 34 sensors, or modifies an Actor observation.
``ComposedUSDScene`` imports USD only when explicitly instantiated by a future
native runner. The runner supplies its *existing* physics SimulationView and
Wire root rigid view. No second simulation context or high-level sensor is made.

``capture(clock=...)`` returns a separate auxiliary DTO, not a replacement for
MPCState. Observed tuple reductions are distinct from qualified ON_OBJECT force:
static-filter delivery, sign/frame and upstream completeness need independent
native gates. There is deliberately no native=True flag/callback. All native,
passive-support and airborne qualification remains false. A numeric observed
zero never proves absent support. Tensor tuples contain no partner body IDs.

Installed SDK source audit: omni.physics.tensors/impl/api.py:263-312,4540-4831;
get_masses:3075-3098, get_coms:3125-3149, get_inertias:3151-3177. Inertia frame
comments conflict; all scopes and both candidates are retained. A common world
tensor is provided only when the two interpretations numerically coincide.
Backend SI fields and force reductions require actual stage meters/kilograms
per unit both equal to one; non-SI normalization is unproven and stays raw/NA.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral
from types import MappingProxyType

import numpy as np

from .tabletop_bimanual_mpc_state import FIELD_SPECS, SampleClock, StateField


SURFACES = ('table', 'negative_y', 'positive_y')


class _ClockChanged(RuntimeError):
    """Distinguish torn capture from an unavailable backend getter."""


INERTIA_FRAME_SCOPES = (
    ('omni.physics.tensors/impl/api.py:3154', 'center of mass local frame'),
    ('omni.physics.tensors/impl/api.py:3160', 'object frame'),
    ('isaaclab/assets/rigid_object/rigid_object_data.py:113-120', 'about COM, expressed in actor frame'),
)
_FIELD_META = {
    'geometry.radius': ('m', 'shape', 'NA'),
    'geometry.height': ('m', 'shape', 'NA'),
    'geometry.axis_body': ('dimensionless', 'actor', 'NA'),
    'object.centerline': ('m', 'world', 'NA'),
    'object.local_tangent': ('dimensionless', 'world', 'NA'),
    'object.centerline_velocity': ('m/s', 'world', 'NA'),
    'object.bottom_height': ('m', 'world', 'NA'),
    'object.com': ('m', 'world', 'NA'),
    'object.mass': ('kg', 'not_applicable', 'NA'),
    'rigid.inertia_raw_9': ('kg*m^2', 'ambiguous_actor_or_COM', 'NA'),
    'rigid.inertia_world': ('kg*m^2', 'world', 'NA'),
    'rigid.com_local_pose_xyzw': ('m|quaternion_xyzw', 'actor', 'NA'),
    'rigid.actor_world_pose_wxyz': ('m|quaternion_wxyz', 'world', 'NA'),
    'rigid.com_world_pose_wxyz': ('m|quaternion_wxyz', 'world', 'NA'),
    'support.table_reaction': ('N', 'world', 'ON_OBJECT'),
    'support.negative_y_reaction': ('N', 'world', 'ON_OBJECT'),
    'support.positive_y_reaction': ('N', 'world', 'ON_OBJECT'),
    'support.end_support_reaction': ('N', 'world', 'ON_OBJECT'),
    'contact.partner_body_id': ('dimensionless', 'not_applicable', 'NA'),
}


def _array(value):
    """Own SDK storage immediately; no Torch/warp dependency is imported here."""
    if hasattr(value, 'detach'):
        value = value.detach().clone().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    return np.array(value, copy=True)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return deepcopy(value)


def _finite(value, shape):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'finite data of shape {shape} required')
    return a


def _rotation(q_xyzw):
    q = _finite(q_xyzw, (4,))
    if not np.isclose(np.linalg.norm(q), 1., rtol=0., atol=1e-6):
        raise ValueError('unit quaternion required')
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _multiply_quat(a, b):
    av, aw, bv, bw = a[:3], a[3], b[:3], b[3]
    return np.r_[aw*bv+bw*av+np.cross(av, bv), aw*bw-np.dot(av, bv)]


def _wxyz_pose(pose):
    return np.r_[pose[:3], pose[6], pose[3:6]].tolist()


def _wire_paths(view):
    paths = list(view.prim_paths)
    if not paths or len(set(paths)) != len(paths) or any(
            not isinstance(p, str) or not p.startswith('/World/envs/env_') or not p.endswith('/Wire')
            or any(c in p for c in '*?[]') for p in paths):
        raise ValueError('exact actual Wire root view paths required')
    return paths


def _validate_scene(record, wire_path):
    if record.get('wire_path') != wire_path or not record.get('stage_identity'):
        raise ValueError('actual stage/Wire binding missing')
    if record.get('wire_reporter_exists') is not True:
        raise ValueError('existing Wire contact reporter unproven; no API will be applied')
    if record.get('contact_processing_enabled') is not True:
        raise ValueError('contact processing disabled/unknown; no setting will be changed')
    env = wire_path.rsplit('/', 1)[0]
    roots = {'table': env+'/Table', 'negative_y': env+'/EndSupports/negative_y',
             'positive_y': env+'/EndSupports/positive_y'}
    filters = {}
    for surface in SURFACES:
        colliders = record.get('supports', {}).get(surface, [])
        if len(colliders) != 1:
            raise ValueError(f'{surface}: exactly one actual collider required')
        c = colliders[0]; path = c.get('path', '')
        if (c.get('collision_api') is not True or c.get('collision_enabled') is not True
                or c.get('rigid_ancestor') is not None or not c.get('source_layers')
                or not (path == roots[surface] or path.startswith(roots[surface]+'/'))
                or (surface == 'table' and path == roots[surface])
                or (surface != 'table' and (path != roots[surface] or c.get('type_name') != 'Cube'))):
            raise ValueError(f'{surface}: unproven static CollisionAPI collider/source binding')
        filters[surface] = path
    return filters


@dataclass(frozen=True)
class PassiveContactViews:
    enabled: bool
    views: object
    wire_paths: tuple[str, ...]
    filter_paths: object
    stage_identities: tuple[str, ...]
    scene: object = None
    rigid_body_view: object = None
    simulation_view: object = None


def create_passive_contact_views(*, scene, simulation_view, rigid_body_view,
                                 enabled=False, capacity_per_wire=0):
    """Explicit opt-in, three independent LOW-LEVEL views of the existing sim.

    Preflight *all* environments before creating anything. Only
    ``simulation_view.create_rigid_contact_view`` is called. No import of core.api,
    create_simulation_view, Apply/Set, prepare_contact_sensors, refresh or step.
    ``scene`` may be a CPU fixture; neither type nor handle is native attestation.
    Unsupported backend/filter expansion fails closed at capture, preserving raw.
    """
    if type(enabled) is not bool:
        raise ValueError('enabled must be explicit bool')
    if not enabled:
        return PassiveContactViews(False, MappingProxyType({}), (), MappingProxyType({}), ())
    if isinstance(capacity_per_wire, bool) or not isinstance(capacity_per_wire, Integral) or capacity_per_wire < 1:
        raise ValueError('explicit positive contact capacity per Wire required')
    wires = _wire_paths(rigid_body_view)
    records = [scene.snapshot(p) for p in wires]
    filters = [_validate_scene(r, p) for r, p in zip(records, wires)]
    expected = {s: tuple(f[s] for f in filters) for s in SURFACES}
    views = {}
    for s in SURFACES:
        views[s] = simulation_view.create_rigid_contact_view(
            pattern=list(wires), filter_patterns=[[p] for p in expected[s]],
            max_contact_data_count=len(wires)*int(capacity_per_wire))
    return PassiveContactViews(True, MappingProxyType(views), tuple(wires),
        MappingProxyType(expected), tuple(r['stage_identity'] for r in records),
        scene, rigid_body_view, simulation_view)


@dataclass(frozen=True)
class SceneMeasurementSample:
    clock: SampleClock
    fields: object
    raw: dict
    qualification: object


def _field(clock, name, value=None, *, reason=None, kind='measured', source='scene measurement provider', dependencies=()):
    unit, frame, side = _FIELD_META[name]
    spec = FIELD_SPECS.get(name)
    if spec is not None:
        unit, frame, side = spec.unit, spec.frame, spec.force_side
    return StateField(value=_plain(value), source=source, unit=unit, frame=frame,
        timestamp_s=None if kind == 'static' else clock.world_time_s,
        valid=value is not None, env_id=clock.env_id, episode_uid=clock.episode_uid,
        clock=clock, kind=kind if value is not None else 'unavailable',
        missing_reason=reason, force_side=side, dependencies=dependencies,
        origin=spec.origin if spec else 'not_applicable',
        contact_basis=spec.contact_basis if spec else 'not_applicable')


def _owned(buffers, *, normal, n, capacity, active_mask=None):
    # Optional audit selection never weakens full-batch ownership/capacity checks.
    # Inactive result rows are analysis-only NaN placeholders, not measured zero.
    if active_mask is not None:
        active_mask=tuple(active_mask)
        if len(active_mask)!=n or any(type(v) is not bool for v in active_mask):
            raise ValueError('one actual payload-selection bool per sensor row required')
    size = 6 if normal else 4
    if len(buffers) != size:
        raise ValueError(f'expected {size}-tuple')
    counts, starts = buffers[-2:]
    if (counts.shape != (n, 1) or starts.shape != (n, 1)
            or counts.dtype.kind not in 'iu' or starts.dtype.kind not in 'iu'):
        raise ValueError('integer count/start shape (sensor_count,1) required')
    widths = (1, 3, 3, 1) if normal else (3, 3)
    for value, width in zip(buffers[:-2], widths):
        if value.shape != (capacity, width):
            raise ValueError('buffer capacity/layout mismatch')
    if int(counts.sum(dtype=np.int64)) >= capacity:
        raise ValueError('capacity saturated or exceeded; overflow cannot be excluded')
    occupied = set(); result = []
    for i in range(n):
        count, start = int(counts[i, 0]), int(starts[i, 0])
        if count < 0 or start < 0 or start > capacity or start+count > capacity:
            raise ValueError('owned interval outside capacity')
        slots = set(range(start, start+count))
        if occupied.intersection(slots):
            raise ValueError('owned intervals overlap between sensor/filter cells')
        occupied.update(slots)
        if active_mask is not None and not active_mask[i]:
            result.append(np.full(3,np.nan));continue
        for value in buffers[:-2]:
            if not np.isfinite(value[start:start+count]).all():
                raise ValueError('nonfinite owned contact data')
        vectors = buffers[0][start:start+count]
        if normal:
            normals = buffers[2][start:start+count]
            if count and not np.allclose(np.linalg.norm(normals, axis=1), 1., rtol=1e-5, atol=1e-5):
                raise ValueError('owned normals are not unit vectors')
            vectors = vectors * normals
        result.append(vectors.sum(axis=0))
    return np.asarray(result)


def _contact_observation(view, *, wires, filters, dt, read, si_units, passive_tap=None):
    out = dict(observed_data_available=False, normal_available=False, friction_available=False,
        static_filter_support_qualified=False, on_object_sign_frame_qualified=False,
        upstream_overflow_absence_attested=False, partner_body_id=None,
        qualification_missing_reason='Native static-filter delivery, ON_OBJECT sign/frame and upstream completeness unverified')
    try:
        def obtain(name,function):
            return function() if passive_tap is None else passive_tap(name,function)
        def actual_metadata():
            return {name: _plain(getattr(view,name)) for name in (
                'sensor_count','filter_count','max_contact_data_count','sensor_paths','filter_paths')}
        # Preserve the actual getter structures. Expected paths never replace readback.
        metadata = read('metadata', lambda: obtain('metadata',actual_metadata))
        out['metadata'] = metadata
        if read('check', lambda: obtain('check',view.check)) is not True:
            raise ValueError('contact view check failed; stop/play lifecycle may have invalidated it')
        n = len(wires); cap = metadata['max_contact_data_count']
        actual_filters = metadata['filter_paths']
        # Accept only unambiguous exact readback, either nested-per-sensor or flat.
        if actual_filters == list(filters):
            actual_filters = [[p] for p in actual_filters]
        if (metadata['sensor_count'] != n or metadata['filter_count'] != 1
                or metadata['sensor_paths'] != list(wires)
                or actual_filters != [[p] for p in filters]
                or isinstance(cap, bool) or not isinstance(cap, Integral) or cap < 1):
            raise ValueError('actual sensor/filter mapping or capacity unproven')
        errors = []
        matrix = normal = friction = None
        for name, function in (
                ('matrix', lambda: _array(obtain('matrix',lambda:view.get_contact_force_matrix(dt=dt)))),
                ('normal', lambda: tuple(_array(v) for v in obtain('normal',lambda:view.get_contact_data(dt=dt)))),
                ('friction', lambda: tuple(_array(v) for v in obtain('friction',lambda:view.get_friction_data(dt=dt))))):
            try:
                value = read(name, function)
                # normal ownership is copied before the friction getter can run.
                out[name if name == 'matrix' else name+'_tuple'] = _plain(value)
                if name == 'matrix': matrix = value
                elif name == 'normal': normal = value
                else: friction = value
            except _ClockChanged:
                raise
            except Exception as exc:
                errors.append(f'{name}: {type(exc).__name__}: {exc}')
        if passive_tap is not None:
            read('metadata_after',lambda:obtain('metadata_after',actual_metadata))
        if not si_units:
            raise ValueError('backend unit normalization unproven for non-SI/unknown stage units; raw only')
        try:
            if normal is None or matrix is None:
                raise ValueError('normal tuple or matrix unavailable')
            normal_sum = _owned(normal, normal=True, n=n, capacity=cap)
            _finite(matrix, (n, 1, 3))
            if not np.allclose(normal_sum, matrix[:, 0], rtol=1e-4, atol=1e-5):
                raise ValueError('raw normal reduction differs from force matrix')
            out.update(normal_available=True, normal_sum_n=_plain(normal_sum))
        except (ValueError, TypeError, OverflowError) as exc:
            errors.append('normal: '+str(exc))
        try:
            if friction is None:
                raise ValueError('friction tuple unavailable')
            friction_sum = _owned(friction, normal=False, n=n, capacity=cap)
            out.update(friction_available=True, friction_sum_n=_plain(friction_sum))
        except (ValueError, TypeError, OverflowError) as exc:
            errors.append('friction: '+str(exc))
        if errors:
            raise ValueError('; '.join(errors))
        out.update(observed_data_available=True,
                   resultant_sensor_side_n=_plain(normal_sum+friction_sum), missing_reason=None)
    except _ClockChanged:
        raise  # actual clock failure must invalidate the entire capture
    except Exception as exc:
        out['missing_reason'] = f'{type(exc).__name__}: {exc}'
    return out


class SceneMeasurementProvider:
    """A CPU-testable facade over actual composed stage and backend getters.

    Clock reader returns the runner's *current* SampleClock for the requested
    environment. It must read actual World/SDK clocks, not return the supplied
    request unchanged. Each getter is bracketed; identity is never native proof.
    Capture once per live env/physical step, after original refresh/before reset.
    Reset/inactive slots emit NA without reading. Terminal values may be archived.
    Rebuild provider and views after stop/play; no cached values are replayed.
    """
    def __init__(self, *, scene, rigid_body_view, clock_reader, contact_views=None,
                 enabled=False, physics_dt=1/480, passive_pair_observer=None):
        if type(enabled) is not bool or not callable(clock_reader):
            raise ValueError('explicit bool enabled and actual clock reader required')
        if not math.isfinite(physics_dt) or not math.isclose(physics_dt, 1/480, rel_tol=0., abs_tol=1e-12):
            raise ValueError('original 480 Hz physics dt required; policy dt is not contact dt')
        self.scene, self.rigid_body_view = scene, rigid_body_view
        self.clock_reader, self.contact_views = clock_reader, contact_views
        self.enabled, self.physics_dt = enabled, float(physics_dt)
        self.last_sample = None
        self._previous = {}
        self._paths = None
        self._batch_clocks = None
        self._batch_cache = None
        self.passive_pair_observer=passive_pair_observer
        self.last_passive_pair_snapshot=None
        self.last_passive_pair_error=None

    def capture_batch(self, *, clocks, active=None):
        """Own full-view getters once, then select immutable per-env fields.

        All clocks must describe one physical frame; episode/reset identity may
        differ per environment. Every actual getter checks every requested live
        clock. The cache exists only inside this synchronous call and never
        survives a physics frame. A torn batch restores all per-env history.
        Raw evidence remains explicitly batch-scoped with exact view-path order.
        """
        clocks = tuple(clocks)
        if not clocks or not all(isinstance(c, SampleClock) for c in clocks):
            raise ValueError('nonempty SampleClock sequence required')
        if len({c.env_id for c in clocks}) != len(clocks):
            raise ValueError('duplicate environment in scene capture batch')
        frames = {(c.run_uuid, c.world_step, c.world_time_s, c.sdk_step, c.sdk_time_s) for c in clocks}
        if len(frames) != 1:
            raise ValueError('all environments must share one physics frame')
        active = (True,)*len(clocks) if active is None else tuple(active)
        if len(active) != len(clocks) or any(type(v) is not bool for v in active):
            raise ValueError('one boolean active mask per clock required')
        if self._batch_clocks is not None:
            raise ValueError('nested/concurrent scene capture batch unsupported')
        previous = dict(self._previous)
        self._batch_clocks = tuple(c for c, live in zip(clocks, active) if live and not c.reset)
        self._batch_cache = {}
        self.last_passive_pair_snapshot=None;self.last_passive_pair_error=None
        observer=self.passive_pair_observer;begun=False
        try:
            if observer is not None and observer.enabled:
                begun=observer.begin(clocks=clocks,active_mask=active)
            return tuple(self.capture(clock=c, active=live) for c, live in zip(clocks, active))
        except Exception:
            self._previous = previous
            self.last_sample = None
            raise
        finally:
            try:
                if begun:
                    import sys
                    unwinding=sys.exc_info()[0] is not None
                    try:self.last_passive_pair_snapshot=observer.finish()
                    except Exception as error:
                        self.last_passive_pair_error=repr(error)
                        if not unwinding:raise
            finally:self._batch_clocks = self._batch_cache = None

    def capture(self, *, clock, active=True):
        self.last_sample = None
        if self.passive_pair_observer is not None and self.passive_pair_observer.enabled and self._batch_clocks is None:
            raise ValueError('passive sidecar requires explicit full capture_batch')
        if not isinstance(clock, SampleClock) or type(active) is not bool:
            raise ValueError('SampleClock and bool active required')
        reason = 'provider disabled' if not self.enabled else ('inactive slot' if not active else 'reset snapshot' if clock.reset else None)
        fields = {name: _field(clock, name, reason=reason or 'source unavailable') for name in _FIELD_META}
        qualification = MappingProxyType(dict(native_attested=False, passive_support_qualified=False,
                                             airborne_qualified=False))
        raw = dict(source_scope='simulation_privileged_readback_not_hardware', read_brackets=[],
                   rigid_getters={}, passive_views={})
        if reason:
            sample = SceneMeasurementSample(clock, MappingProxyType(fields), raw, qualification)
            self.last_sample = sample
            return sample
        def current():
            for expected in self._batch_clocks if self._batch_clocks is not None else (clock,):
                actual = self.clock_reader(expected.env_id)
                if not isinstance(actual, SampleClock) or actual != expected:
                    raise _ClockChanged('actual source clock/episode changed during scene capture')
            return clock
        def read(name, function):
            before = current()
            key = (name, clock.env_id if name == 'composed_stage' else None)
            cached = self._batch_cache is not None and key in self._batch_cache
            try:
                if cached:
                    result = self._batch_cache[key]
                else:
                    result = function()
                    if self._batch_cache is not None:
                        self._batch_cache[key] = result
            finally:
                after = current()
                raw['read_brackets'].append(dict(name=name, before=repr(before), after=repr(after),
                    reused_within_same_capture_batch=cached))
            return result
        current()
        previous = self._previous.get(clock.env_id)
        if previous:
            if clock.run_uuid != previous.run_uuid:
                raise ValueError('run identity changed; rebuild provider')
            if clock.world_step <= previous.world_step or clock.sdk_step <= previous.sdk_step:
                raise ValueError('stale/repeated physics clock')
            elapsed_steps = clock.world_step-previous.world_step
            if clock.sdk_step-previous.sdk_step != elapsed_steps or any(
                    not math.isclose(elapsed, elapsed_steps*self.physics_dt, rel_tol=0., abs_tol=1e-7)
                    for elapsed in (clock.world_time_s-previous.world_time_s,
                                    clock.sdk_time_s-previous.sdk_time_s)):
                raise ValueError('World/SDK clock alignment or physical elapsed time changed')
            if clock.episode_uid != previous.episode_uid:
                if clock.reset_generation <= previous.reset_generation:
                    raise ValueError('episode change without reset generation')
            elif clock.reset_generation != previous.reset_generation or previous.terminal:
                raise ValueError('reset/terminal episode identity cannot resume')
            if clock.policy_step < previous.policy_step:
                raise ValueError('policy clock moved backwards')
        paths = read('rigid.prim_paths', lambda: _wire_paths(self.rigid_body_view))
        if self._paths is None:
            self._paths = tuple(paths)
        if tuple(paths) != self._paths:
            raise ValueError('Wire path binding changed')
        expected_path = f'/World/envs/env_{clock.env_id}/Wire'
        if expected_path not in paths:
            raise ValueError('environment Wire path binding missing')
        env_index = paths.index(expected_path)
        raw.update(source_wire_paths=paths[:], selected_wire_index=env_index,
                   array_scope='batch-view raw arrays ordered by source_wire_paths; StateFields select one Wire')
        bundle = self.contact_views
        if bundle and bundle.enabled and (tuple(paths) != bundle.wire_paths
                or bundle.scene is not self.scene or bundle.rigid_body_view is not self.rigid_body_view):
            raise ValueError('passive view source binding mismatch')
        record = read('composed_stage', lambda: _plain(self.scene.snapshot(expected_path)))
        raw['scene'] = record
        if self.passive_pair_observer is not None and self.passive_pair_observer.enabled:
            self.passive_pair_observer.observe_scene(clock.env_id,record)
        if record.get('wire_path') != expected_path or not record.get('stage_identity'):
            raise ValueError('actual composed stage/Wire binding missing')
        # Installed TensorAPI does not establish normalization for non-SI stages.
        # Preserve raw backend values rather than mixing them with SI geometry.
        si_units = record.get('meters_per_unit') == 1. and record.get('kilograms_per_unit') == 1.
        raw['backend_unit_evidence'] = dict(stage_meters_per_unit=record.get('meters_per_unit'),
            stage_kilograms_per_unit=record.get('kilograms_per_unit'), si_stage_units_verified=si_units,
            non_SI_backend_normalization_qualified=False)
        getter_arrays = {}
        for name in ('get_masses', 'get_inertias', 'get_coms', 'get_transforms', 'get_velocities'):
            try:
                value = read(name, lambda name=name: _array(getattr(self.rigid_body_view, name)()))
                getter_arrays[name] = value
                raw['rigid_getters'][name] = dict(available=True, value=_plain(value),
                    shape=list(value.shape), dtype=str(value.dtype), source='actual root_physx_view.'+name)
            except _ClockChanged:
                raise
            except Exception as exc:
                raw['rigid_getters'][name] = dict(available=False, value=None, missing_reason=str(exc))
        def row(name, width):
            if not si_units:
                raise ValueError('backend unit normalization unproven for non-SI/unknown stage units')
            array = getter_arrays.get(name)
            if array is None or array.shape != (len(paths), width):
                raise ValueError(f'{name}: exact per-Wire backend layout required')
            return _finite(array[env_index], (width,))
        try:
            mass = row('get_masses', 1)[0]
            if mass <= 0: raise ValueError('backend mass must be positive')
            fields['object.mass'] = _field(clock, 'object.mass', float(mass), source='backend.get_masses (SI kg)')
        except ValueError as exc:
            fields['object.mass'] = _field(clock, 'object.mass', reason=str(exc))
        actor = com = rotation = world_com = None
        try:
            actor = row('get_transforms', 7); rotation = _rotation(actor[3:])
            fields['rigid.actor_world_pose_wxyz'] = _field(clock, 'rigid.actor_world_pose_wxyz', _wxyz_pose(actor), source='backend.get_transforms xyzw -> wxyz')
        except ValueError:
            actor = rotation = None
        try:
            com = row('get_coms', 7); com_rotation = _rotation(com[3:])
            fields['rigid.com_local_pose_xyzw'] = _field(clock, 'rigid.com_local_pose_xyzw', com, source='backend.get_coms raw local xyzw')
            if actor is not None:
                world_com = actor[:3]+rotation@com[:3]
                world_q = _multiply_quat(actor[3:], com[3:])
                fields['object.com'] = _field(clock, 'object.com', world_com, kind='derived', source='backend actor pose composed with actual COM local pose', dependencies=('get_transforms', 'get_coms'))
                fields['rigid.com_world_pose_wxyz'] = _field(clock, 'rigid.com_world_pose_wxyz', _wxyz_pose(np.r_[world_com, world_q]), kind='derived', source='backend actor/COM local pose composition', dependencies=('get_transforms', 'get_coms'))
        except ValueError:
            com = None
        raw['inertia_frame_evidence'] = dict(source_scopes=INERTIA_FRAME_SCOPES,
            raw_order='Ixx,Iyx,Izx,Ixy,Iyy,Izy,Ixz,Iyz,Izz',
            candidates_numerically_equivalent=False, rtol=1e-6, atol_kg_m2=1e-10,
            frame_semantics_qualified=False)
        try:
            inertia = row('get_inertias', 9)
            fields['rigid.inertia_raw_9'] = _field(clock, 'rigid.inertia_raw_9', inertia, source='backend.get_inertias raw9; conflicting frame source scopes retained')
            tensor = inertia.reshape(3, 3, order='F')
            if not np.allclose(tensor, tensor.T, rtol=1e-6, atol=1e-10) or np.linalg.eigvalsh(tensor).min() <= 0:
                raise ValueError('backend inertia tensor is not symmetric positive definite')
            if actor is None or com is None: raise ValueError('actual actor/COM poses unavailable')
            a = rotation@tensor@rotation.T
            rc = rotation@com_rotation; b = rc@tensor@rc.T
            equivalent = bool(np.allclose(a, b, rtol=1e-6, atol=1e-10))
            raw['inertia_frame_evidence'].update(actor_scope_world=_plain(a), com_scope_world=_plain(b),
                                               candidates_numerically_equivalent=equivalent)
            if not equivalent: raise ValueError('inertia frame conflict changes world tensor')
            fields['rigid.inertia_world'] = _field(clock, 'rigid.inertia_world', a, kind='derived',
                source='conditional numerical equivalence of actor/COM source scopes; not native qualification',
                dependencies=('get_inertias', 'get_transforms', 'get_coms'))
        except ValueError as exc:
            fields['rigid.inertia_world'] = _field(clock, 'rigid.inertia_world', reason=str(exc))
        self._geometry(record, actor, rotation, world_com, row, clock, fields)
        contact_reason = 'Native static-filter delivery/ON_OBJECT sign/frame/completeness gate not available'
        try:
            expected_filters = _validate_scene(record, expected_path)
            if not bundle or not bundle.enabled:
                raise ValueError('passive contact view factory disabled')
            if record['stage_identity'] != bundle.stage_identities[env_index]:
                raise ValueError('stage identity changed since view creation')
            if any(expected_filters[s] != bundle.filter_paths[s][env_index] for s in SURFACES):
                raise ValueError('actual composed collider mapping changed')
            for s in SURFACES:
                options={}
                if self.passive_pair_observer is not None and self.passive_pair_observer.enabled:
                    options['passive_tap']=lambda name,fn,s=s:self.passive_pair_observer.read(s,name,bundle.views[s],fn)
                raw['passive_views'][s] = _contact_observation(bundle.views[s], wires=paths,
                    filters=bundle.filter_paths[s], dt=self.physics_dt,
                    si_units=si_units,
                    read=lambda name, fn, s=s: read(s+'.'+name, fn),**options)
        except ValueError as exc:
            contact_reason += '; '+str(exc)
        for name in _FIELD_META:
            if name.startswith('support.'):
                fields[name] = _field(clock, name, reason=contact_reason)
        fields['contact.partner_body_id'] = _field(clock, 'contact.partner_body_id',
            reason='Tensor normal/friction tuples contain no partner body ID; view paths are not per-contact identity')
        if not si_units:
            for name in ('object.com', 'object.mass', 'object.centerline', 'object.local_tangent',
                         'object.centerline_velocity', 'object.bottom_height', 'rigid.inertia_raw_9',
                         'rigid.inertia_world', 'rigid.com_local_pose_xyzw',
                         'rigid.actor_world_pose_wxyz', 'rigid.com_world_pose_wxyz'):
                fields[name] = _field(clock, name, reason='backend unit normalization unproven for non-SI/unknown stage units; raw retained')
        current()
        sample = SceneMeasurementSample(clock, MappingProxyType(fields), raw, qualification)
        self._previous[clock.env_id] = clock
        self.last_sample = sample
        return sample

    @staticmethod
    def _geometry(record, actor, rotation, world_com, row, clock, fields):
        names = ('geometry.radius', 'geometry.height', 'geometry.axis_body', 'object.centerline',
                 'object.local_tangent', 'object.centerline_velocity', 'object.bottom_height')
        try:
            shapes = record.get('wire_colliders', [])
            if len(shapes) != 1: raise ValueError('single actual solid Cylinder collider required')
            shape = shapes[0]
            if (shape.get('type_name') != 'Cylinder' or shape.get('collision_api') is not True
                    or shape.get('collision_enabled') is not True or not shape.get('source_layers')
                    or shape.get('rigid_ancestor') != record.get('wire_path')):
                raise ValueError('actual composed Cylinder CollisionAPI/source binding unproven')
            # SDK actor poses contain no scale. Cancelling USD's body-world
            # transform would otherwise silently drop an inherited body scale.
            body_linear = _finite(record['body_world_linear_for_scale_only'], (3, 3))
            if not np.allclose(body_linear.T@body_linear, np.eye(3), rtol=1e-6, atol=1e-10) or np.linalg.det(body_linear) <= 0:
                raise ValueError('scaled/sheared/reflected rigid ancestor unsupported; actual body scale retained')
            units = float(record['meters_per_unit'])
            radius, height = float(shape['radius']), float(shape['height'])
            if not all(math.isfinite(v) and v > 0 for v in (units, radius, height)):
                raise ValueError('finite positive composed dimensions/units required')
            transform = _finite(shape['local_to_body'], (4, 4))
            if not np.allclose(transform[3], [0., 0., 0., 1.], rtol=0., atol=1e-10):
                raise ValueError('invalid shape-to-body affine transform')
            linear = transform[:3, :3]; scale = np.linalg.norm(linear[:, 0])
            if scale <= 0 or not np.allclose(linear.T@linear, np.eye(3)*scale**2, rtol=1e-6, atol=1e-10) or np.linalg.det(linear) <= 0:
                raise ValueError('nonuniform scale/shear/reflection unsupported; raw geometry retained')
            if shape['axis'] not in ('X', 'Y', 'Z'): raise ValueError('actual Cylinder axis unknown')
            axis = np.eye(3)['XYZ'.index(shape['axis'])]
            tangent_body = linear@axis/scale
            center_body = transform[:3, 3]*units
            radius *= units*scale; height *= units*scale
            for name, value in (('geometry.radius', radius), ('geometry.height', height), ('geometry.axis_body', tangent_body)):
                fields[name] = _field(clock, name, value, kind='derived', source='actual composed USD rigid Cylinder (SI)', dependencies=('composed_Cylinder', 'local_to_body', 'meters_per_unit'))
            if actor is None: raise ValueError('actual backend actor pose unavailable')
            center = actor[:3]+rotation@center_body; tangent = rotation@tangent_body
            points = np.array([center-tangent*height/2, center+tangent*height/2])
            bottom = center[2]-abs(tangent[2])*height/2-radius*math.sqrt(max(0., 1-tangent[2]**2))
            for name, value in (('object.centerline', points), ('object.local_tangent', [tangent, tangent]), ('object.bottom_height', bottom)):
                fields[name] = _field(clock, name, value, kind='derived', source='source-bound rigid Cylinder geometry; no flexible dynamics claim', dependencies=('composed_Cylinder', 'local_to_body', 'get_transforms'))
            try:
                if world_com is None: raise ValueError('actual COM pose unavailable')
                velocity = row('get_velocities', 6)
                point_v = velocity[:3]+np.cross(velocity[3:], points-world_com)
                fields['object.centerline_velocity'] = _field(clock, 'object.centerline_velocity', point_v, kind='derived', source='rigid v_COM + omega cross (point-COM)', dependencies=('object.centerline', 'get_velocities', 'object.com'))
            except ValueError as exc:
                fields['object.centerline_velocity'] = _field(clock, 'object.centerline_velocity', reason=str(exc))
        except (KeyError, TypeError, ValueError) as exc:
            for name in names:
                if name.startswith('geometry.') and fields[name].valid:
                    continue
                fields[name] = _field(clock, name, reason=str(exc))


class ComposedUSDScene:
    """Read the caller's actual composed stage; imports USD only on explicit use.

    The settings handle must be the caller's existing carb settings interface.
    Its contact-processing key is read, never written. Instance proxies and
    property stacks are traversed. No cfg radius, spawn path or bounding box is
    substituted for CollisionAPI geometry. Native runtime compatibility remains
    to be checked by the sole native runner after the current job has ended.
    """
    def __init__(self, *, stage, settings):
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
        self.stage, self.settings = stage, settings
        self.Usd, self.UsdGeom, self.UsdPhysics, self.PhysxSchema = Usd, UsdGeom, UsdPhysics, PhysxSchema

    def snapshot(self, wire_path):
        Usd, Geom, Physics = self.Usd, self.UsdGeom, self.UsdPhysics
        stage = self.stage; wire = stage.GetPrimAtPath(wire_path)
        if not wire or not wire.IsValid() or not wire.HasAPI(Physics.RigidBodyAPI):
            raise ValueError('actual stage Wire RigidBodyAPI not found')
        cache = Geom.XformCache(Usd.TimeCode.Default())
        body_world = cache.GetLocalToWorldTransform(wire)
        def colliders(root_path):
            root = stage.GetPrimAtPath(root_path)
            if not root or not root.IsValid(): return []
            result = []
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(Physics.CollisionAPI): continue
                ancestor = prim
                while ancestor and not ancestor.HasAPI(Physics.RigidBodyAPI):
                    ancestor = ancestor.GetParent()
                # Gf matrices use row vectors; expose an explicit column-vector transform.
                relative = cache.GetLocalToWorldTransform(prim)*body_world.GetInverse()
                c = dict(path=str(prim.GetPath()), type_name=prim.GetTypeName(),
                    collision_api=True, collision_enabled=Physics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                    rigid_ancestor=str(ancestor.GetPath()) if ancestor else None,
                    local_to_body=np.asarray(relative, dtype=float).T.tolist(),
                    source_layers=[[spec.layer.identifier, str(spec.path)] for spec in prim.GetPrimStack()],
                    applied_schemas=list(prim.GetAppliedSchemas()), is_instance_proxy=prim.IsInstanceProxy(),
                    asset_info={str(k): str(v) for k, v in prim.GetAssetInfo().items()})
                if prim.IsA(Geom.Cylinder):
                    cylinder = Geom.Cylinder(prim)
                    c.update(radius=cylinder.GetRadiusAttr().Get(), height=cylinder.GetHeightAttr().Get(), axis=str(cylinder.GetAxisAttr().Get()))
                result.append(c)
            return result
        env = wire_path.rsplit('/', 1)[0]
        disabled = self.settings.get('/physics/disableContactProcessing')
        return dict(wire_path=wire_path, meters_per_unit=Geom.GetStageMetersPerUnit(stage),
            kilograms_per_unit=Physics.GetStageKilogramsPerUnit(stage),
            body_world_linear_for_scale_only=np.asarray(body_world, dtype=float).T[:3, :3].tolist(),
            stage_identity=str(stage.GetRootLayer().identifier)+';session='+str(stage.GetSessionLayer().identifier)+';handle='+hex(id(stage)),
            source_scope='actual composed stage read; native qualification external',
            wire_reporter_exists=wire.HasAPI(self.PhysxSchema.PhysxContactReportAPI),
            contact_processing_enabled=(not disabled) if type(disabled) is bool else None,
            wire_colliders=colliders(wire_path), supports={
                'table': colliders(env+'/Table'),
                'negative_y': colliders(env+'/EndSupports/negative_y'),
                'positive_y': colliders(env+'/EndSupports/positive_y')})
