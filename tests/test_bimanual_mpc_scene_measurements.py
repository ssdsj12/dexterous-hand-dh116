"""CPU fixtures only: no SDK imports, live views, or native/support qualification."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import SampleClock


def api():
    path = Path(__file__).resolve().parents[1] / 'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_scene_measurements.py'
    assert path.is_file(), 'read-only scene measurement provider is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_scene_measurements')


def clock(**changes):
    values = dict(run_uuid='CPU-fixture', env_id=0, episode_uid='ep0', world_step=480,
                  world_time_s=1., sdk_step=1480, sdk_time_s=11., policy_step=60, reset_generation=0)
    values.update(changes)
    return SampleClock(**values)


def scene_record(index=0):
    root = f'/World/envs/env_{index}'
    wire = root + '/Wire'
    def collider(path, typ, rigid=None):
        return dict(path=path, type_name=typ, collision_api=True, collision_enabled=True,
                    rigid_ancestor=rigid, local_to_body=np.eye(4).tolist(),
                    source_layers=[['fixture.usda', path]], applied_schemas=['PhysicsCollisionAPI'])
    cylinder = collider(wire+'/geometry/mesh', 'Cylinder', wire)
    cylinder.update(radius=.012, height=.7, axis='Y')
    return dict(wire_path=wire, meters_per_unit=1., kilograms_per_unit=1.,
                body_world_linear_for_scale_only=np.eye(3).tolist(),
                stage_identity='fixture-stage', source_scope='CPU_fixture_not_native',
                wire_reporter_exists=True, contact_processing_enabled=True,
                wire_colliders=[cylinder], supports={
                    'table': [collider(root+'/Table/geometry/mesh', 'Cube')],
                    'negative_y': [collider(root+'/EndSupports/negative_y', 'Cube')],
                    'positive_y': [collider(root+'/EndSupports/positive_y', 'Cube')]})


class Scene:
    def __init__(self, n=2):
        self.records = [scene_record(i) for i in range(n)]
        self.reads = 0

    def snapshot(self, wire_path):
        self.reads += 1
        return deepcopy(next(s for s in self.records if s['wire_path'] == wire_path))


class Rigid:
    def __init__(self, n=2):
        self.prim_paths = [f'/World/envs/env_{i}/Wire' for i in range(n)]
        self.mass = np.array([[.21], [.34]])[:n]
        self.inertia = np.tile(np.diag([1., 2., 3.]).reshape(1, 9), (n, 1))
        self.com = np.tile([.03, 0., 0., 0., 0., 0., 1.], (n, 1)).astype(float)
        self.actor = np.tile([1., 2., 3., 0., 0., 0., 1.], (n, 1)).astype(float)
        self.velocity = np.tile([0., 0., 0., 0., 0., 2.], (n, 1)).astype(float)
        self.reads = []

    def _read(self, name, value):
        self.reads.append(name)
        return value

    def get_masses(self): return self._read('mass', self.mass)
    def get_inertias(self): return self._read('inertia', self.inertia)
    def get_coms(self): return self._read('com', self.com)
    def get_transforms(self): return self._read('actor', self.actor)
    def get_velocities(self): return self._read('velocity', self.velocity)


class Contact:
    """Reproduces SDK metadata aliasing, separate anchors, signed normal data."""
    def __init__(self, wires, filters, capacity=8):
        self.sensor_count = len(wires)
        self.filter_count = 1
        self.sensor_paths = list(wires)
        self.filter_paths = deepcopy(filters)
        self.max_contact_data_count = capacity
        self.ok = True
        self.calls = []
        n = len(wires)
        self.count = np.zeros((n, 1), dtype=np.uint32)
        self.start = self.count.copy()
        self.normal_count = np.ones((n, 1), dtype=np.uint32)
        self.normal_start = np.arange(n, dtype=np.uint32).reshape(n, 1)
        self.friction_count = np.ones((n, 1), dtype=np.uint32)
        self.friction_start = (np.arange(n)+3).astype(np.uint32).reshape(n, 1)
        self.scalars = np.full((capacity, 1), np.nan)
        self.points = np.full((capacity, 3), np.nan)
        self.normals = np.full((capacity, 3), np.nan)
        self.separation = np.full((capacity, 1), np.nan)
        self.friction = np.full((capacity, 3), np.nan)
        self.anchors = np.full((capacity, 3), np.nan)
        for i in range(n):
            self.scalars[i] = -2. if i == 0 else 3.
            self.normals[i] = [0., 0., 1.]
            self.points[i] = [0., 0., .2]
            self.separation[i] = -.001
            self.friction[i+3] = [.5, 0., -.1]
            self.anchors[i+3] = [0., 0., .3]
        self.matrix = self.scalars[:n, None, :] * self.normals[:n, None, :]

    def check(self): return self.ok
    def get_contact_force_matrix(self, dt):
        self.calls.append(('matrix', dt))
        return self.matrix
    def get_contact_data(self, dt):
        self.calls.append(('normal', dt))
        self.count[:] = self.normal_count; self.start[:] = self.normal_start
        return self.scalars, self.points, self.normals, self.separation, self.count, self.start
    def get_friction_data(self, dt):
        self.calls.append(('friction', dt))
        self.count[:] = self.friction_count; self.start[:] = self.friction_start
        return self.friction, self.anchors, self.count, self.start


class Simulation:
    def __init__(self): self.created = []
    def create_rigid_contact_view(self, **kwargs):
        self.created.append(deepcopy(kwargs))
        return Contact(kwargs['pattern'], kwargs['filter_patterns'], kwargs['max_contact_data_count'])


def fixture(*, enabled=True, contacts=True):
    m = api(); scene = Scene(); rigid = Rigid(); sim = Simulation(); now = [clock()]
    views = m.create_passive_contact_views(scene=scene, simulation_view=sim,
        rigid_body_view=rigid, enabled=contacts, capacity_per_wire=8)
    provider = m.SceneMeasurementProvider(scene=scene, rigid_body_view=rigid,
        clock_reader=lambda env_id: now[env_id], contact_views=views,
        enabled=enabled, physics_dt=1/480)
    return NS(m=m, scene=scene, rigid=rigid, sim=sim, views=views, provider=provider, now=now)


def test_missing_provider_is_a_real_red_and_import_has_no_native_dependencies():
    before = set(sys.modules)
    m = api()
    assert hasattr(m, 'SceneMeasurementProvider')
    assert not any(name == 'pxr' or name.startswith(('pxr.', 'isaacsim', 'omni.', 'torch'))
                   for name in set(sys.modules)-before)


def test_disabled_factory_and_provider_do_no_reads_or_creates():
    m = api(); scene = Scene(); rigid = Rigid(); sim = Simulation()
    views = m.create_passive_contact_views(scene=scene, simulation_view=sim,
        rigid_body_view=rigid, capacity_per_wire=8)
    assert not sim.created and not views.enabled
    p = m.SceneMeasurementProvider(scene=scene, rigid_body_view=rigid,
        clock_reader=lambda _: (_ for _ in ()).throw(AssertionError('clock read while disabled')))
    sample = p.capture(clock=clock())
    assert not rigid.reads and not scene.reads
    assert all(f.value is None and f.missing_reason for f in sample.fields.values())


def test_factory_binds_actual_collision_children_and_three_independent_views():
    f = fixture()
    assert len(f.sim.created) == 3
    for surface, request in zip(('table', 'negative_y', 'positive_y'), f.sim.created):
        assert request['pattern'] == f.rigid.prim_paths
        expected = [[f.scene.records[i]['supports'][surface][0]['path']] for i in range(2)]
        assert request['filter_patterns'] == expected
        assert request['max_contact_data_count'] == 16
        assert all(path[0].endswith('/geometry/mesh') for path in expected) if surface == 'table' else True
    assert len({id(v) for v in f.views.views.values()}) == 3


@pytest.mark.parametrize('fault', ['extra_table', 'dynamic_support', 'no_reporter', 'processing_unknown', 'disabled_collision'])
def test_factory_refuses_ambiguous_or_unprepared_scene_without_creating(fault):
    m = api(); s = Scene(); r = Rigid(); sim = Simulation()
    rec = s.records[1]
    if fault == 'extra_table': rec['supports']['table'].append(deepcopy(rec['supports']['table'][0]))
    elif fault == 'dynamic_support': rec['supports']['negative_y'][0]['rigid_ancestor'] = '/World/dynamic'
    elif fault == 'no_reporter': rec['wire_reporter_exists'] = False
    elif fault == 'processing_unknown': rec['contact_processing_enabled'] = None
    else: rec['supports']['table'][0]['collision_enabled'] = False
    with pytest.raises(ValueError):
        m.create_passive_contact_views(scene=s, simulation_view=sim, rigid_body_view=r,
                                     enabled=True, capacity_per_wire=8)
    assert not sim.created


def test_actual_geometry_uses_composed_shape_transform_and_sdk_com_origin():
    f = fixture(); rec = f.scene.records[0]; shape = rec['wire_colliders'][0]
    shape['local_to_body'][0][3] = .1
    sample = f.provider.capture(clock=clock())
    assert sample.fields['geometry.radius'].value == pytest.approx(.012)
    assert sample.fields['geometry.height'].value == pytest.approx(.7)
    assert np.array(sample.fields['object.centerline'].value) == pytest.approx(np.array([[1.1, 1.65, 3.], [1.1, 2.35, 3.]]))
    assert sample.fields['object.local_tangent'].value == ((0., 1., 0.), (0., 1., 0.))
    assert sample.fields['object.centerline'].kind == 'derived'
    assert 'rigid' in sample.fields['object.centerline'].source
    assert sample.fields['object.com'].value == pytest.approx([1.03, 2., 3.])
    assert np.array(sample.fields['object.centerline_velocity'].value) == pytest.approx(np.array([[.7, .14, 0.], [-.7, .14, 0.]]))
    assert sample.raw['scene']['wire_colliders'][0]['radius'] == .012
    assert sample.raw['rigid_getters']['get_coms']['value'][0][:3] == [.03, 0., 0.]


@pytest.mark.parametrize('fault', ['Mesh', 'two_colliders', 'nonuniform_scale', 'no_collision', 'invalid_axis', 'missing_source'])
def test_unsupported_geometry_stays_na_and_retains_actual_raw(fault):
    f = fixture(); rec = f.scene.records[0]; shape = rec['wire_colliders'][0]
    if fault == 'Mesh': shape['type_name'] = 'Mesh'
    elif fault == 'two_colliders': rec['wire_colliders'].append(deepcopy(shape))
    elif fault == 'nonuniform_scale': shape['local_to_body'][0][0] = 2.
    elif fault == 'no_collision': shape['collision_api'] = False
    elif fault == 'invalid_axis': shape['axis'] = 'unknown'
    else: shape['source_layers'] = []
    s = f.provider.capture(clock=clock())
    assert not s.fields['object.centerline'].valid
    assert s.fields['object.centerline'].missing_reason
    assert s.fields['object.mass'].value == .21
    assert s.raw['scene']['wire_colliders']


def test_backend_parameters_preserve_raw9_and_both_inertia_frame_scopes():
    f = fixture(); s = f.provider.capture(clock=clock())
    assert s.fields['object.mass'].value == .21
    assert s.fields['rigid.inertia_raw_9'].value == (1., 0., 0., 0., 2., 0., 0., 0., 3.)
    assert s.fields['rigid.com_local_pose_xyzw'].value == (.03, 0., 0., 0., 0., 0., 1.)
    evidence = s.raw['inertia_frame_evidence']
    assert len(evidence['source_scopes']) == 3
    assert evidence['candidates_numerically_equivalent']
    assert np.array(s.fields['rigid.inertia_world'].value) == pytest.approx(np.diag([1., 2., 3.]))
    assert not s.qualification['native_attested']
    f.rigid.mass[:] = 99.; f.rigid.com[:] = 99.
    assert s.raw['rigid_getters']['get_masses']['value'][0] == [.21]


def test_rotated_com_with_distinct_inertia_candidates_is_na():
    f = fixture(); q = np.sqrt(.5); f.rigid.com[0, 3:] = [0., 0., q, q]
    s = f.provider.capture(clock=clock())
    assert not s.fields['rigid.inertia_world'].valid
    assert 'frame' in s.fields['rigid.inertia_world'].missing_reason
    assert not s.raw['inertia_frame_evidence']['candidates_numerically_equivalent']
    assert s.fields['rigid.com_world_pose_wxyz'].value[:3] == pytest.approx([1.03, 2., 3.])


def test_signed_normal_and_friction_reduce_separate_owned_ranges_and_keep_alias_snapshot():
    f = fixture(); s = f.provider.capture(clock=clock())
    observation = s.raw['passive_views']['table']
    assert observation['observed_data_available']
    assert observation['normal_sum_n'][0] == [0., 0., -2.]
    assert observation['friction_sum_n'][0] == [.5, 0., -.1]
    assert observation['resultant_sensor_side_n'][0] == [.5, 0., -2.1]
    assert observation['normal_tuple'][5][0] == [0]
    assert observation['friction_tuple'][3][0] == [3]
    assert s.fields['support.table_reaction'].value is None
    assert 'Native' in s.fields['support.table_reaction'].missing_reason
    assert not s.qualification['passive_support_qualified']
    assert not s.qualification['airborne_qualified']
    assert s.fields['contact.partner_body_id'].value is None
    assert f.views.views['table'].calls == [('matrix', 1/480), ('normal', 1/480), ('friction', 1/480)]


@pytest.mark.parametrize('fault', ['overlap', 'overflow', 'saturation', 'fractional_count', 'nonfinite',
                                  'normal_mismatch', 'wrong_filter', 'wildcard', 'dead_view', 'missing_friction'])
def test_malformed_or_incomplete_passive_data_fail_closed(fault):
    f = fixture(); view = f.views.views['table']
    if fault == 'overlap': view.normal_start[1] = 0
    elif fault == 'overflow': view.normal_start[1] = 999
    elif fault == 'saturation': view.normal_count[0] = 16
    elif fault == 'fractional_count':
        view.normal_count = np.array([[.5], [1.]])
        view.count = np.zeros((2, 1), dtype=float)
    elif fault == 'nonfinite': view.scalars[0] = np.nan
    elif fault == 'normal_mismatch': view.matrix[0, 0, 2] = 99.
    elif fault == 'wrong_filter': view.filter_paths[0] = [view.filter_paths[1][0]]
    elif fault == 'wildcard': view.filter_paths[0] = ['/World/envs/env_.*/Table/geometry/mesh']
    elif fault == 'dead_view': view.ok = False
    else: view.get_friction_data = None
    s = f.provider.capture(clock=clock())
    obs = s.raw['passive_views']['table']
    assert not obs['observed_data_available']
    assert obs['missing_reason']
    assert s.fields['support.table_reaction'].value is None
    assert s.raw['passive_views']['negative_y']['observed_data_available']


def test_unproven_zero_is_observed_but_never_support_absence_or_native():
    f = fixture()
    for v in f.views.views.values():
        v.normal_count[:] = 0; v.friction_count[:] = 0; v.matrix[:] = 0
    s = f.provider.capture(clock=clock())
    obs = s.raw['passive_views']['table']
    assert obs['observed_data_available']
    assert obs['resultant_sensor_side_n'][0] == [0., 0., 0.]
    assert not obs['static_filter_support_qualified']
    assert not obs['upstream_overflow_absence_attested']
    assert s.fields['support.table_reaction'].value is None
    assert not any(s.qualification.values())


def test_clock_change_refuses_capture_and_clears_previous_evidence():
    f = fixture(); f.provider.capture(clock=clock())
    f.now[0] = clock(world_step=481, sdk_step=1481, world_time_s=1.+1/480, sdk_time_s=11.+1/480)
    original = f.rigid.get_coms
    def change_clock():
        result = original(); f.now[0] = clock(world_step=482, sdk_step=1482); return result
    f.rigid.get_coms = change_clock
    with pytest.raises(RuntimeError, match='clock'):
        f.provider.capture(clock=clock(world_step=481, sdk_step=1481, world_time_s=1.+1/480, sdk_time_s=11.+1/480))
    assert f.provider.last_sample is None


def test_repeat_reset_inactive_and_path_mismatch_cannot_reuse_state():
    f = fixture(); f.provider.capture(clock=clock())
    with pytest.raises(ValueError, match='repeat|stale'):
        f.provider.capture(clock=clock())
    f = fixture(); s = f.provider.capture(clock=clock(reset=True))
    assert all(not field.valid for field in s.fields.values())
    f = fixture(); s = f.provider.capture(clock=clock(), active=False)
    assert all(not field.valid for field in s.fields.values())
    assert not f.rigid.reads
    f = fixture(); f.rigid.prim_paths.reverse()
    with pytest.raises(ValueError, match='path|binding'):
        f.provider.capture(clock=clock())


def test_each_environment_keeps_distinct_mass_and_bad_getter_does_not_fill_zero():
    f = fixture(); f.now.append(clock(env_id=1, episode_uid='ep1'))
    s = f.provider.capture(clock=f.now[1]); assert s.fields['object.mass'].value == .34
    f = fixture(); f.rigid.get_masses = lambda: np.array([[np.nan], [.34]])
    s = f.provider.capture(clock=clock())
    assert s.fields['object.mass'].value is None
    assert s.raw['rigid_getters']['get_masses']['value'][1] == [.34]


def test_no_native_attestation_callable_can_be_supplied():
    m = api()
    with pytest.raises(TypeError):
        m.SceneMeasurementProvider(scene=Scene(), rigid_body_view=Rigid(),
            clock_reader=lambda _: clock(), native_attestation=lambda: True)


def test_backend_runtime_errors_are_na_and_not_misreported_as_clock_changes():
    f = fixture()
    def unsupported(): raise RuntimeError('backend unsupported')
    f.rigid.get_masses = unsupported
    f.views.views['table'].get_friction_data = unsupported
    s = f.provider.capture(clock=clock())
    assert not s.fields['object.mass'].valid
    assert not s.raw['passive_views']['table']['observed_data_available']
    assert 'unsupported' in s.raw['passive_views']['table']['missing_reason']
    assert s.raw['passive_views']['table']['normal_available']


@pytest.mark.parametrize('change', [dict(world_time_s=.9), dict(sdk_time_s=10.9),
                                   dict(sdk_step=1490), dict(world_time_s=4.)])
def test_advancing_steps_cannot_hide_broken_world_sdk_clock_alignment(change):
    f = fixture(); f.provider.capture(clock=clock())
    values = dict(world_step=481, sdk_step=1481, world_time_s=1+1/480, sdk_time_s=11+1/480)
    values.update(change); f.now[0] = clock(**values)
    with pytest.raises(ValueError, match='clock|time|step'):
        f.provider.capture(clock=f.now[0])


def test_missing_friction_keeps_independently_available_normal_observation():
    f = fixture(); f.views.views['table'].get_friction_data = None
    s = f.provider.capture(clock=clock())
    obs = s.raw['passive_views']['table']
    assert obs['normal_available']
    assert not obs['friction_available']
    assert obs['normal_sum_n'][0] == [0., 0., -2.]


def test_raw_batch_evidence_explicitly_binds_paths_and_selected_environment_index():
    f = fixture(); f.rigid.prim_paths = list(reversed(f.rigid.prim_paths))
    f.rigid.mass = f.rigid.mass[::-1].copy()
    # Rebuild views against actual reverse order, including filters.
    f.views = f.m.create_passive_contact_views(scene=f.scene, simulation_view=f.sim,
        rigid_body_view=f.rigid, enabled=True, capacity_per_wire=8)
    f.provider = f.m.SceneMeasurementProvider(scene=f.scene, rigid_body_view=f.rigid,
        clock_reader=lambda _: clock(), contact_views=f.views, enabled=True)
    s = f.provider.capture(clock=clock())
    assert s.fields['object.mass'].value == .21
    assert s.raw['source_wire_paths'] == ['/World/envs/env_1/Wire', '/World/envs/env_0/Wire']
    assert s.raw['selected_wire_index'] == 1
    assert 'batch' in s.raw['array_scope']


def test_wrong_scene_identity_cannot_supply_valid_geometry_or_rigid_parameters():
    f = fixture(); f.scene.records[0]['wire_path'] = '/World/envs/env_9/Wire'
    f.scene.snapshot = lambda _: deepcopy(f.scene.records[0])
    with pytest.raises(ValueError, match='binding|Wire'):
        f.provider.capture(clock=clock())


def test_capture_batch_owns_one_backend_snapshot_and_emits_distinct_per_env_fields():
    f = fixture(); f.now.append(clock(env_id=1, episode_uid='ep1'))
    result = f.provider.capture_batch(clocks=tuple(f.now))
    assert [s.clock.env_id for s in result] == [0, 1]
    assert [s.fields['object.mass'].value for s in result] == [.21, .34]
    assert f.rigid.reads == ['mass', 'inertia', 'com', 'actor', 'velocity']
    for view in f.views.views.values():
        assert len(view.calls) == 3
    assert result[0].raw is not result[1].raw
    result[0].raw['rigid_getters']['get_masses']['value'][0][0] = 99.
    assert result[1].raw['rigid_getters']['get_masses']['value'][0][0] == .21


def test_capture_batch_torn_sibling_clock_rolls_back_all_capture_history():
    f = fixture(); f.now.append(clock(env_id=1, episode_uid='ep1'))
    original = f.rigid.get_coms
    def changed_sibling():
        result = original(); f.now[1] = clock(env_id=1, world_step=481); return result
    f.rigid.get_coms = changed_sibling
    with pytest.raises(RuntimeError, match='clock'):
        f.provider.capture_batch(clocks=(clock(), clock(env_id=1, episode_uid='ep1')))
    assert f.provider._previous == {}
    assert f.provider.last_sample is None


def test_capture_batch_rejects_duplicate_env_or_mixed_physics_frames():
    f = fixture()
    with pytest.raises(ValueError, match='environment|frame'):
        f.provider.capture_batch(clocks=(clock(), clock()))
    with pytest.raises(ValueError, match='frame'):
        f.provider.capture_batch(clocks=(clock(), clock(env_id=1, world_step=481)))


def test_scaled_rigid_ancestor_cannot_be_lost_by_cancelling_usd_body_world_transform():
    f = fixture(); f.scene.records[0]['body_world_linear_for_scale_only'] = (2*np.eye(3)).tolist()
    s = f.provider.capture(clock=clock())
    assert not s.fields['geometry.radius'].valid
    assert not s.fields['object.centerline'].valid
    assert 'scale' in s.fields['object.centerline'].missing_reason


@pytest.mark.parametrize('unit_key,unit_value', [('meters_per_unit', .01), ('kilograms_per_unit', .001),
                                               ('kilograms_per_unit', None)])
def test_unproven_backend_unit_normalization_keeps_raw_and_rejects_si_world_fields(unit_key, unit_value):
    f = fixture(); rec = f.scene.records[0]; rec[unit_key] = unit_value
    if unit_key == 'meters_per_unit':
        rec['wire_colliders'][0].update(radius=1.2, height=70.)
    s = f.provider.capture(clock=clock())
    assert s.fields['geometry.radius'].value == pytest.approx(.012)
    for name in ('object.mass', 'object.com', 'object.centerline', 'rigid.inertia_world', 'rigid.com_local_pose_xyzw'):
        assert not s.fields[name].valid
        assert 'unit' in s.fields[name].missing_reason
    assert s.raw['rigid_getters']['get_coms']['value'][0][:3] == [.03, 0., 0.]
    assert not s.raw['passive_views']['table']['observed_data_available']
    assert s.raw['passive_views']['table']['normal_tuple']
    assert 'unit' in s.raw['passive_views']['table']['missing_reason']


def test_every_shared_field_preserves_existing_mpc_state_metadata_contract():
    from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import FIELD_SPECS, make_mpc_state, missing_fields
    f = fixture(); s = f.provider.capture(clock=clock())
    fields = missing_fields(s.clock, source='CPU fixture', reason='CPU fixture source unavailable')
    fields.update({name: field for name, field in s.fields.items() if name in FIELD_SPECS})
    result = make_mpc_state(s.clock, fields)
    assert result.fields['object.centerline_velocity'].origin == 'centerline_points'
    assert result.fields['contact.partner_body_id'].contact_basis == 'actual_contact'
