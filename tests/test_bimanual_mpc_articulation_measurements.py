"""CPU-only SDK-shaped fixtures; none attest Native or future dynamics."""
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS
from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import SampleClock


WRIST = ('wrist_x', 'wrist_y', 'wrist_z', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')


def api():
    path = Path(__file__).resolve().parents[1] / 'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_articulation_measurements.py'
    assert path.is_file(), 'read-only articulation measurement provider is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_articulation_measurements')


def clock(**changes):
    values = dict(run_uuid='CPU-fixture', env_id=0, episode_uid='ep0', world_step=480,
                  world_time_s=1., sdk_step=1480, sdk_time_s=11., policy_step=60, reset_generation=0)
    values.update(changes)
    return SampleClock(**values)


class View:
    def __init__(self, side, names, *, floating=False):
        self.count, self.max_dofs, self.max_links = 2, len(names), len(names)+1
        self.is_homogeneous = True
        self.prim_paths = [f'/World/envs/env_{i}/{side}Hand/wrist_fixed' for i in (1, 0)]
        roots = [p.rsplit('/', 1)[0] for p in self.prim_paths]
        links = ['anchor'] + ['link_'+name for name in names]
        self.dof_paths = [root+'/'+n for root in roots for n in ('root_joint',)+tuple(names)]
        self.link_paths = [root+'/'+n for root in roots for n in links]
        types = [1 if n in WRIST[:3] else 0 for n in names]
        self.shared_metatype = NS(fixed_base=not floating, link_count=self.max_links,
            dof_count=self.max_dofs, joint_count=self.max_dofs+1,
            link_names=links, link_indices={n:i for i,n in enumerate(links)},
            dof_names=list(names), dof_indices={n:i for i,n in enumerate(names)}, dof_types=types,
            joint_names=['root_joint']+list(names), joint_indices={n:i for i,n in enumerate(['root_joint']+list(names))},
            joint_types=[0]+[1]*len(names), joint_dof_offsets=[0]+list(range(len(names))),
            joint_dof_counts=[0]+[1]*len(names))
        self.metatypes = [deepcopy(self.shared_metatype) for _ in range(2)]
        self.calls = []
        n, d, l = self.count, self.max_dofs, self.max_links
        g = d+6 if floating else d
        vector = np.stack([100+np.arange(d), 200+np.arange(d)]).astype(np.float32)
        self.arrays = {key: vector.copy() for key in (
            'get_dof_positions', 'get_dof_velocities', 'get_dof_position_targets',
            'get_dof_velocity_targets', 'get_dof_actuation_forces', 'get_dof_stiffnesses',
            'get_dof_dampings', 'get_dof_max_forces', 'get_dof_max_velocities',
            'get_dof_armatures', 'get_dof_friction_coefficients', 'get_dof_projected_joint_forces')}
        self.arrays.update(get_dof_types=np.tile(types, (n, 1)).astype(np.uint8),
            get_drive_types=np.ones((n,d), dtype=np.uint8),
            get_dof_limits=np.tile([-1., 1.], (n,d,1)),
            get_generalized_mass_matrices=np.tile(2*np.eye(g), (n,1,1)),
            get_coriolis_and_centrifugal_compensation_forces=np.full((n,g), 3.),
            get_gravity_compensation_forces=np.full((n,g), 4.),
            get_link_incoming_joint_force=np.full((n,l,6), 5.),
            get_masses=np.full((n,l), .1), get_coms=np.tile([0,0,0,0,0,0,1.], (n,l,1)),
            get_inertias=np.tile(np.eye(3).reshape(9), (n,l,1)))

    def get_metatype(self, i):
        self.calls.append(('metatype', i))
        return self.metatypes[i]

    def __getattr__(self, name):
        if name in self.arrays:
            def read():
                self.calls.append(name)
                return self.arrays[name]
            return read
        raise AttributeError(name)


def fixture(*, names=None, floating=False, enabled=True):
    m = api()
    names = tuple(reversed(JOINTS+WRIST)) if names is None else tuple(names)
    hands = {s:NS(root_physx_view=View(s, names, floating=floating), joint_names=list(reversed(names)),
                  body_names=list(reversed(['anchor']+['link_'+n for n in names]))) for s in ('left','right')}
    now = [clock(), clock(env_id=1, episode_uid='ep1')]
    sources = dict(stage_identity='CPU-fixture-stage', meters_per_unit=1., kilograms_per_unit=1.,
                   sdk_sources=[dict(path='/fixture/api.py', sha256='a'*64, version='fixture')])
    reads = []
    def source_reader(side, path):
        reads.append((side,path))
        return deepcopy(sources)
    provider = m.ArticulationMeasurementProvider(hands=hands, clock_reader=lambda i:now[i],
        source_reader=source_reader, enabled=enabled)
    return NS(m=m, hands=hands, provider=provider, now=now, sources=sources, source_reads=reads)


def test_disabled_never_touches_backend_or_clock_or_source():
    m = api()
    class Bomb:
        def __getattribute__(self, name): raise AssertionError(name)
    p = m.ArticulationMeasurementProvider(hands=Bomb(), clock_reader=Bomb(), source_reader=Bomb())
    s = p.capture(clock=clock())
    assert all(not v.valid and 'disabled' in v.missing_reason for v in s.fields.values())
    assert not any(s.qualification.values())


def test_full_actual_d_order_group_maps_and_distinct_environment_selection():
    f = fixture(); samples = f.provider.capture_batch(clocks=f.now)
    s = samples[0]
    assert len(s.fields['left.q'].value) == 17
    assert s.fields['left.q'].value[0] == 200.
    assert samples[1].fields['left.q'].value[0] == 100.
    binding = s.raw['hands']['left']['binding']
    assert binding['selected_view_index'] == 1
    assert binding['dof_names'] == list(reversed(JOINTS+WRIST))
    assert binding['groups']['active6']['indices'] == [binding['dof_names'].index(n) for n in ACTIVE]
    assert binding['groups']['passive5']['names'] == [n for n in JOINTS if n not in ACTIVE]
    assert binding['groups']['wrist6']['names'] == list(WRIST)
    assert binding['dof_paths'][0].endswith('/wrist_yaw')
    assert len(binding['hand_joint_to_dof']) == 17
    assert s.raw['array_scope'].startswith('batch')
    assert all(not v for v in s.qualification.values())


def test_batch_reads_once_per_hand_and_owns_each_getter_before_next_aliasing_call():
    f = fixture(); v = f.hands['left'].root_physx_view
    original = v.get_dof_velocities
    def aliasing():
        v.arrays['get_dof_positions'][:] = -99
        return original()
    v.get_dof_velocities = aliasing
    samples = f.provider.capture_batch(clocks=f.now)
    assert samples[0].fields['left.q'].value[0] == 200.
    raw = samples[0].raw['hands']['left']['getters']['get_dof_positions']
    assert raw['raw_shape'] == [2,17] and raw['raw_dtype'] == 'float32'
    assert raw['source_device'] == 'cpu'
    assert raw['value'][1][0] == 200.
    assert v.calls.count('get_dof_positions') == 1
    assert v.calls.count('get_generalized_mass_matrices') == 1
    raw['value'][0][0] = 999
    assert samples[1].raw['hands']['left']['getters']['get_dof_positions']['value'][0][0] == 100.


def test_unmounted_full11_retains_actual_dimension_and_missing_wrist_map():
    f = fixture(names=JOINTS); s = f.provider.capture(clock=clock())
    assert len(s.fields['left.q'].value) == 11
    group = s.raw['hands']['left']['binding']['groups']['wrist6']
    assert not group['available'] and group['indices'] is None and group['missing_reason']


def test_mixed_coordinate_units_and_raw_dynamics_never_claim_motor_or_future_model():
    f = fixture(); s = f.provider.capture(clock=clock())
    raw = s.raw['hands']['left']; units = raw['unit_layout']
    names = raw['binding']['dof_names']
    assert units['position'][names.index('wrist_x')] == 'stage_distance'
    assert units['position'][names.index('finger11')] == 'rad'
    assert units['generalized_force'][names.index('wrist_x')] == 'N'
    assert units['generalized_force'][names.index('finger11')] == 'N*m'
    assert units['mass_matrix'][names.index('wrist_x')][names.index('finger11')] == 'kg*m'
    assert s.fields['left.generalized_mass_matrix_raw'].value[0][0] == 2.
    assert s.fields['left.coriolis_compensation_raw'].value[0] == 3.
    for name in ('actual_motor_torque', 'implicit_pd_estimate', 'candidate_future_dynamics'):
        assert not s.fields['left.'+name].valid
    assert raw['source_identity']['sdk_sources'][0]['sha256'] == 'a'*64
    assert raw['semantics']['current_state_only']
    assert not raw['semantics']['compensation_sign_qualified']
    assert not raw['semantics']['armature_inclusion_qualified']
    assert not raw['semantics']['solver_force_motor_isolated']


@pytest.mark.parametrize('name', ['get_generalized_mass_matrices', 'get_gravity_compensation_forces',
    'get_coriolis_and_centrifugal_compensation_forces', 'get_dof_positions'])
def test_unsupported_or_raising_actual_getter_is_explicit_na_without_zero_or_legacy_fallback(name):
    f = fixture(); v = f.hands['left'].root_physx_view
    def unsupported(): raise RuntimeError('backend unsupported on this device')
    setattr(v, name, unsupported)
    v.get_mass_matrices = lambda: pytest.fail('no deprecated fallback')
    s = f.provider.capture(clock=clock())
    raw = s.raw['hands']['left']['getters'][name]
    assert raw['value'] is None and 'unsupported' in raw['missing_reason']
    assert s.fields['right.q'].valid
    assert not s.qualification['dynamics_qualified']


@pytest.mark.parametrize('fault', ['wrong_shape', 'nonfinite', 'drive_enum', 'type_disagrees'])
def test_malformed_getters_keep_raw_but_do_not_create_valid_field(fault):
    f = fixture(); v = f.hands['left'].root_physx_view
    name = 'get_dof_positions'
    field = 'left.q'
    if fault == 'wrong_shape': v.arrays[name] = np.ones((2,11))
    elif fault == 'nonfinite': v.arrays[name][1,0] = np.nan
    elif fault == 'drive_enum':
        name, field = 'get_drive_types', 'left.drive_type'; v.arrays[name][1,0] = 255
    else:
        name, field = 'get_dof_types', 'left.q'; v.arrays[name][1,0] = 1
    s = f.provider.capture(clock=clock())
    assert not s.fields[field].valid and s.fields[field].missing_reason
    assert s.raw['hands']['left']['getters'][name]['value'] is not None


def test_floating_root_dimensions_retained_without_invented_axis_order_or_units():
    f = fixture(floating=True); s = f.provider.capture(clock=clock())
    assert len(s.fields['left.generalized_mass_matrix_raw'].value) == 23
    raw = s.raw['hands']['left']
    assert raw['binding']['fixed_base'] is False
    assert raw['unit_layout']['generalized_root_order'] is None
    assert raw['unit_layout']['mass_matrix'] is None
    assert not s.qualification['dynamics_qualified']


def test_missing_source_or_non_si_units_retains_native_raw_units_without_si_assumption():
    f = fixture(); f.sources['meters_per_unit'] = .01
    s = f.provider.capture(clock=clock())
    raw = s.raw['hands']['left']
    assert raw['unit_layout']['position'][0] == 'rad'
    assert set(raw['unit_layout']['generalized_force']) == {'backend_force_unqualified', 'backend_torque_unqualified'}
    assert not raw['unit_layout']['si_stage_units_verified']
    f = fixture(); f.provider.source_reader = None; s = f.provider.capture(clock=clock())
    assert s.raw['hands']['left']['source_identity']['missing_reason']
    assert not s.qualification['native_attested']


@pytest.mark.parametrize('fault', ['duplicate_name', 'index_disagrees', 'missing_path', 'wrong_body_names'])
def test_unbound_actual_layout_is_na_instead_of_guessing_dof_indices(fault):
    f = fixture(); v = f.hands['left'].root_physx_view
    if fault == 'duplicate_name': v.metatypes[1].dof_names[0] = v.metatypes[1].dof_names[1]
    elif fault == 'index_disagrees': v.metatypes[1].dof_indices['finger11'] = 0
    elif fault == 'missing_path': v.dof_paths = [p for p in v.dof_paths if '/env_0/' not in p or not p.endswith('/finger11')]
    else: f.hands['left'].body_names[0] = 'invented'
    s = f.provider.capture(clock=clock())
    assert not s.fields['left.q'].valid
    assert s.raw['hands']['left']['binding']['missing_reason']
    assert s.raw['hands']['left']['getters']['get_dof_positions']['value'] is not None


def test_clock_change_in_sibling_rolls_back_entire_transaction_and_evidence():
    f = fixture(); v = f.hands['left'].root_physx_view
    original = v.get_dof_positions
    def torn():
        result = original(); f.now[1] = clock(env_id=1, world_step=481); return result
    v.get_dof_positions = torn
    with pytest.raises(RuntimeError, match='clock'):
        f.provider.capture_batch(clocks=tuple(f.now))
    assert f.provider.last_sample is None and f.provider._previous == {}


def test_repeat_and_terminal_episode_cannot_be_reused_for_planning():
    f = fixture(); f.provider.capture(clock=clock())
    with pytest.raises(ValueError, match='stale|repeat'):
        f.provider.capture(clock=clock())
    f = fixture(); f.now[0] = clock(terminal=True)
    s = f.provider.capture(clock=f.now[0]); assert s.fields['left.q'].valid
    assert not s.qualification['planning_eligible']
    f.now[0] = clock(world_step=481, sdk_step=1481, world_time_s=1+1/480, sdk_time_s=11+1/480)
    with pytest.raises(ValueError, match='terminal'):
        f.provider.capture(clock=f.now[0])


@pytest.mark.parametrize('changes', [dict(sdk_step=1482), dict(world_time_s=.9), dict(reset_generation=1),
                                    dict(episode_uid='other')])
def test_clock_or_episode_transition_requires_original_physical_alignment(changes):
    f = fixture(); f.provider.capture(clock=clock())
    c = dict(world_step=481, sdk_step=1481, world_time_s=1+1/480, sdk_time_s=11+1/480)
    c.update(changes); f.now[0] = clock(**c)
    with pytest.raises(ValueError, match='clock|time|step|reset|episode'):
        f.provider.capture(clock=f.now[0])


@pytest.mark.parametrize('changes', [dict(reset=True), dict(active=False)])
def test_reset_inactive_are_na_without_backend_reads(changes):
    f = fixture(); active = changes.pop('active', True)
    s = f.provider.capture(clock=clock(**changes), active=active)
    assert all(not value.valid for value in s.fields.values())
    assert not f.hands['left'].root_physx_view.calls and not f.source_reads


def test_batch_rejects_duplicates_mixed_frames_and_nonboolean_mask():
    f = fixture()
    for clocks, active in (([clock(),clock()],None), ([clock(),clock(env_id=1,world_step=481)],None),
                            ([clock()],[1])):
        with pytest.raises(ValueError): f.provider.capture_batch(clocks=clocks,active=active)
    assert not f.hands['left'].root_physx_view.calls


def test_import_is_cpu_only_and_no_simulator_mutators_or_candidate_api_exist():
    import ast
    m = api(); tree = ast.parse(Path(m.__file__).read_text(encoding='utf-8'))
    imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert not any(n.startswith(('torch','pxr','isaac','omni')) for n in imports)
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not any(n.startswith(('set_', 'write_', 'create_', 'Set', 'Apply')) or n in ('step','refresh') for n in calls)
    # A generic `.update` also matches ordinary dictionary operations. Check the
    # backend receiver explicitly rather than rejecting Python container use.
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr == 'update'
                   and isinstance(n.func.value,ast.Name) and n.func.value.id in ('view','hand') for n in ast.walk(tree))
    assert not hasattr(m.ArticulationMeasurementProvider, 'evaluate_candidate')


def test_arbitrary_actual_d_and_zero_dof_joint_offset_sentinel_do_not_drop_extra_rows():
    f = fixture(names=JOINTS+WRIST+('extra_joint',))
    for hand in f.hands.values():
        hand.root_physx_view.shared_metatype.joint_dof_offsets[0] = -1
        for meta in hand.root_physx_view.metatypes: meta.joint_dof_offsets[0] = -1
    s = f.provider.capture(clock=clock())
    assert len(s.fields['left.q'].value) == 18
    assert s.raw['hands']['left']['binding']['dof_names'][-1] == 'extra_joint'


def test_nonnumeric_malformed_getter_is_na_with_raw_evidence_preserved():
    f = fixture(); v = f.hands['left'].root_physx_view
    v.arrays['get_dof_positions'] = np.full((2,17),'invalid')
    s = f.provider.capture(clock=clock())
    assert not s.fields['left.q'].valid
    assert s.raw['hands']['left']['getters']['get_dof_positions']['value'][0][0] == 'invalid'
    assert s.fields['right.q'].valid


def test_all_batch_samples_retain_complete_source_read_brackets_and_same_physical_frame():
    f = fixture(); samples = f.provider.capture_batch(clocks=f.now)
    assert samples[0].raw['read_brackets'] == samples[1].raw['read_brackets']
    assert [r['name'] for r in samples[0].raw['read_brackets']].count('right.source_identity') == 4
    assert all(len(r['before']) == 2 and r['before'] == r['after'] for r in samples[0].raw['read_brackets'])


def test_clock_reader_requires_actual_sample_clock_type_and_source_change_is_torn():
    f = fixture()
    class Spoof:
        def __eq__(self, other): return True
    f.provider.clock_reader = lambda _: Spoof()
    with pytest.raises(RuntimeError, match='clock'): f.provider.capture(clock=clock())
    f = fixture(); original = f.provider.source_reader
    def torn(side,path):
        record = original(side,path); f.now[1] = clock(env_id=1,episode_uid='changed'); return record
    f.provider.source_reader = torn
    with pytest.raises(RuntimeError, match='clock'): f.provider.capture_batch(clocks=tuple(f.now))
    assert f.provider._previous == {} and f.provider.last_sample is None


@pytest.mark.parametrize('change', [dict(stage_identity=None),dict(meters_per_unit=True)])
def test_si_unit_layout_requires_real_stage_identity_and_numeric_unit_metadata(change):
    f = fixture(); f.sources.update(change)
    s = f.provider.capture(clock=clock())
    assert not s.raw['hands']['left']['unit_layout']['si_stage_units_verified']


def advance(f):
    f.now[:] = [clock(env_id=i,episode_uid='ep'+str(i),world_step=481,sdk_step=1481,
                     world_time_s=1+1/480,sdk_time_s=11+1/480) for i in range(2)]


@pytest.mark.parametrize('change', ['stage', 'units', 'sdk', 'missing'])
def test_source_identity_or_units_cannot_change_across_legal_physical_frames(change):
    f = fixture(); f.provider.capture(clock=clock())
    previous, bindings = dict(f.provider._previous), deepcopy(f.provider._bindings)
    sources, common = dict(f.provider._sources), f.provider._common_source
    if change == 'stage': f.sources['stage_identity'] = 'replacement-stage'
    elif change == 'units': f.sources['meters_per_unit'] = .01
    elif change == 'sdk': f.sources['sdk_sources'][0]['sha256'] = 'b'*64
    else: f.provider.source_reader = None
    advance(f)
    with pytest.raises((ValueError,RuntimeError),match='source|stage|unit'):
        f.provider.capture(clock=f.now[0])
    assert f.provider._previous == previous and f.provider._bindings == bindings
    assert f.provider._sources == sources and f.provider._common_source == common
    assert f.provider.last_sample is None


@pytest.mark.parametrize('change', ['stage', 'units'])
def test_common_stage_units_must_agree_between_hands_and_all_batch_environments(change):
    f = fixture(); original = f.provider.source_reader
    def inconsistent(side,path):
        record = original(side,path)
        if side == 'right' or '/env_1/' in path:
            record['stage_identity' if change == 'stage' else 'kilograms_per_unit'] = 'other-stage' if change == 'stage' else .001
        return record
    f.provider.source_reader = inconsistent
    with pytest.raises((ValueError,RuntimeError),match='source|stage|unit'):
        f.provider.capture_batch(clocks=f.now)
    assert f.provider._previous == {} and f.provider._bindings == {}
    assert f.provider._sources == {} and f.provider._common_source is None
    assert f.provider.last_sample is None


def test_source_drift_during_backend_read_is_torn_and_does_not_commit_any_history():
    f = fixture(); v = f.hands['right'].root_physx_view
    original = v.get_dof_velocities
    def changed_source():
        result = original(); f.sources['sdk_sources'][0]['sha256'] = 'c'*64; return result
    v.get_dof_velocities = changed_source
    with pytest.raises((ValueError,RuntimeError),match='source'):
        f.provider.capture_batch(clocks=f.now)
    assert f.provider._previous == {} and f.provider._bindings == {}
    assert f.provider._sources == {} and f.provider._common_source is None
    assert f.provider.last_sample is None


def test_each_hand_keeps_distinct_stable_asset_source_and_source_brackets():
    f = fixture(); original = f.provider.source_reader
    def distinct(side,path):
        record = original(side,path)
        record['asset_source'] = dict(path='/fixture/'+side+'.usd',sha256=('a' if side == 'left' else 'b')*64)
        return record
    f.provider.source_reader = distinct
    samples = f.provider.capture_batch(clocks=f.now)
    for s in samples:
        assert s.fields['left.q'].valid and s.fields['right.q'].valid
        for side in ('left','right'):
            evidence = s.raw['hands'][side]['source_brackets']
            assert evidence['before']['fingerprint'] == evidence['after']['fingerprint']
            assert evidence['before']['available']
    advance(f)
    assert f.provider.capture(clock=f.now[0]).fields['left.q'].valid


def test_always_unavailable_source_keeps_raw_unqualified_and_never_reuses_previous_units():
    f = fixture()
    def unsupported(*args): raise RuntimeError('source unsupported')
    f.provider.source_reader = unsupported
    s = f.provider.capture(clock=clock())
    assert s.fields['left.q'].valid
    assert 'unsupported' in s.raw['hands']['left']['source_identity']['missing_reason']
    assert not s.raw['hands']['left']['unit_layout']['si_stage_units_verified']
    advance(f); assert f.provider.capture(clock=f.now[0]).fields['left.q'].valid


@pytest.mark.parametrize('invalid', [.5,True,False,'0',float('nan'),float('inf'),-float('inf')])
def test_metadata_enum_never_truncates_or_coerces_invalid_raw_types(invalid):
    f = fixture(); v = f.hands['left'].root_physx_view
    for meta in [v.shared_metatype]+v.metatypes: meta.dof_types[0] = invalid
    s = f.provider.capture(clock=clock())
    assert not s.fields['left.q'].valid
    assert s.raw['hands']['left']['metadata_error'] or s.raw['hands']['left']['binding']['missing_reason']
    assert s.fields['right.q'].valid


def test_metadata_integer_enums_preserve_compatibility_and_invalid255_remains_raw_na():
    from enum import IntEnum
    class DofType(IntEnum):
        Rotation = 0
        Translation = 1
        Invalid = 255
    f = fixture(); v = f.hands['left'].root_physx_view
    for meta in [v.shared_metatype]+v.metatypes: meta.dof_types[:] = [DofType(t) for t in meta.dof_types]
    assert f.provider.capture(clock=clock()).fields['left.q'].valid
    f = fixture(); v = f.hands['left'].root_physx_view
    for meta in [v.shared_metatype]+v.metatypes: meta.dof_types[0] = DofType.Invalid
    s = f.provider.capture(clock=clock())
    assert not s.fields['left.q'].valid
    assert s.raw['hands']['left']['metadata']['shared_metatype']['dof_types'][0] == 255


def test_metadata_exact_integral_real_is_explicitly_classified_without_fractional_loss():
    f = fixture(); v = f.hands['left'].root_physx_view
    for meta in [v.shared_metatype]+v.metatypes: meta.dof_types[0] = 0.
    s = f.provider.capture(clock=clock())
    assert s.fields['left.q'].valid
    evidence = s.raw['hands']['left']['metadata']['shared_metatype']['enum_evidence']['dof_types'][0]
    assert evidence['raw_type'] == 'builtins.float'
    assert evidence['normalization'] == 'exact_integral_real'


def test_known_pybind_style_enum_member_is_accepted_without_sdk_import():
    class DofType:
        def __init__(self,value): self.value=value
        def __int__(self): return self.value
    DofType.__members__ = {'Rotation':DofType(0),'Translation':DofType(1),'Invalid':DofType(255)}
    f = fixture(); v = f.hands['left'].root_physx_view
    for meta in [v.shared_metatype]+v.metatypes:
        meta.dof_types[:] = [DofType.__members__['Rotation' if t == 0 else 'Translation'] for t in meta.dof_types]
    assert f.provider.capture(clock=clock()).fields['left.q'].valid


def sdk_enum_wrapper(name, declared):
    """SDK-shaped enum wrappers compare by type/value, without singleton identity."""
    class Wrapper:
        def __init__(self, value, member_name=None):
            self.value = value
            self.name = member_name
        def __int__(self):
            return int(self.value)
        def __eq__(self, other):
            return type(self) is type(other) and self.value == other.value
        def __repr__(self):
            return f'<{name}.{self.name}: {self.value}>'
    Wrapper.__name__ = Wrapper.__qualname__ = name
    Wrapper.__module__ = 'omni.physics.tensors.bindings._physicsTensors'
    Wrapper.__members__ = {key: Wrapper(value, key) for key, value in declared.items()}
    return Wrapper


def test_distinct_sdk_enum_wrappers_normalize_declared_member_equality_for_actual_binding():
    JointType = sdk_enum_wrapper('JointType', {'Fixed': 0, 'Revolute': 1, 'Prismatic': 2})
    DofType = sdk_enum_wrapper('DofType', {'Rotation': 0, 'Translation': 1, 'Invalid': 255})
    f = fixture()
    for hand in f.hands.values():
        v = hand.root_physx_view
        for meta in [v.shared_metatype]+v.metatypes:
            meta.dof_types[:] = [DofType(t, 'Rotation' if t == 0 else 'Translation') for t in meta.dof_types]
            meta.joint_types[:] = [JointType(0, 'Fixed')] + [
                JointType(2 if t.value == 1 else 1, 'Prismatic' if t.value == 1 else 'Revolute')
                for t in meta.dof_types]
    actual = f.hands['left'].root_physx_view.shared_metatype.dof_types[0]
    assert actual is not DofType.__members__[actual.name]
    assert actual == DofType.__members__[actual.name]
    sample = f.provider.capture(clock=clock())
    assert sample.fields['left.q'].valid and sample.fields['right.q'].valid
    assert sample.fields['left.generalized_mass_matrix_raw'].valid
    raw = sample.raw['hands']['left']['metadata']['shared_metatype']
    evidence = raw['enum_evidence']['dof_types'][0]
    assert evidence['raw_type'] == 'omni.physics.tensors.bindings._physicsTensors.DofType'
    assert evidence['raw_name'] == actual.name
    assert evidence['raw_enum_value'] == actual.value
    assert evidence['member_mapping_type'] == 'builtins.dict'
    assert evidence['member_mapping_available'] is True
    assert evidence['declared_member_keys'] == ['Rotation', 'Translation', 'Invalid']
    assert evidence['integer_conversion'] == {'attempted': True, 'value': actual.value, 'error': None}
    assert evidence['identity_member_names'] == []
    assert evidence['equality_member_names'] == [actual.name]
    assert evidence['declared_member_names'] == [actual.name]
    assert evidence['normalization_basis'] == 'same_type_declared_member_equality_and_exact_integer_agreement'
    assert not any(sample.qualification.values())


@pytest.mark.parametrize('defect', ['unknown', 'fraction', 'bool_value', 'integer_disagrees',
                                   'wrong_member_type', 'only_int', 'repr_name_only',
                                   'nonbool_equality', 'false_identity_equality', 'conversion_error',
                                   'no_int_declared_enum'])
def test_sdk_enum_wrapper_rejects_unknown_fractional_boolean_or_unverified_conversion(defect):
    DofType = sdk_enum_wrapper('DofType', {'Rotation': 0, 'Translation': 1})
    value = DofType(0, 'Rotation')
    if defect == 'unknown':
        value = DofType(8, 'Unknown')
    elif defect == 'fraction':
        value = DofType(.5, 'Rotation')
        DofType.__eq__ = lambda self, other: type(self) is type(other) and int(self.value) == int(other.value)
    elif defect == 'bool_value':
        value = DofType(False, 'Rotation')
    elif defect == 'integer_disagrees':
        DofType.__int__ = lambda self: 1 if self is value else int(self.value)
    elif defect == 'wrong_member_type':
        Other = sdk_enum_wrapper('OtherDofType', {'Rotation': 0, 'Translation': 1})
        DofType.__members__ = Other.__members__
        DofType.__eq__ = lambda self, other: getattr(other, 'value', None) == self.value
    elif defect == 'only_int':
        class Unknown:
            def __int__(self): return 0
        value = Unknown()
    elif defect == 'repr_name_only':
        class Unknown:
            name = 'Rotation'
            def __repr__(self): return '<DofType.Rotation: 0>'
        value = Unknown()
    elif defect == 'nonbool_equality':
        DofType.__eq__ = lambda self, other: np.bool_(self.value == other.value)
    elif defect == 'false_identity_equality':
        value = DofType.__members__['Rotation']
        DofType.__eq__ = lambda self, other: False
    elif defect == 'no_int_declared_enum':
        from enum import Enum
        class Declared(Enum):
            Rotation = 0
            Translation = 1
        value = Declared.Rotation
    else:
        def unavailable(self):
            raise TypeError('fixture enum int conversion unavailable')
        DofType.__int__ = unavailable
    f = fixture(); view = f.hands['left'].root_physx_view
    for meta in [view.shared_metatype]+view.metatypes:
        meta.dof_types[0] = value
    sample = f.provider.capture(clock=clock())
    assert not sample.fields['left.q'].valid
    assert sample.fields['right.q'].valid
    assert sample.raw['hands']['left']['getters']['get_dof_positions']['raw_shape'] == [2,17]
    evidence = sample.raw['hands']['left']['metadata']['shared_metatype']['enum_evidence']['dof_types'][0]
    assert evidence['missing_reason']
    assert evidence['member_mapping_type'] == (None if defect in ('only_int', 'repr_name_only')
        else 'builtins.mappingproxy' if defect == 'no_int_declared_enum' else 'builtins.dict')
    assert evidence['member_mapping_available'] is (defect not in ('only_int', 'repr_name_only'))
    assert evidence['declared_member_keys'] == ([] if defect in ('only_int', 'repr_name_only') else ['Rotation', 'Translation'])
    if defect not in ('only_int', 'repr_name_only'):
        assert evidence['raw_name'] == value.name
        assert evidence['raw_enum_value'] == value.value
    if defect == 'integer_disagrees':
        assert evidence['integer_conversion'] == {'attempted': True, 'value': 1, 'error': None}
        assert evidence['equality_member_names'] == ['Rotation']
    elif defect in ('conversion_error', 'no_int_declared_enum'):
        assert evidence['integer_conversion']['attempted']
        assert evidence['integer_conversion']['error']
    else:
        assert not evidence['integer_conversion']['attempted']
    assert not any(sample.qualification.values())


def test_new_source_failure_during_postread_rolls_back_and_same_frame_can_be_retried():
    f = fixture(); original = f.provider.source_reader; calls = [0]
    def changed(side,path):
        calls[0] += 1
        if calls[0] > 2: raise RuntimeError('source disappeared during read')
        return original(side,path)
    f.provider.source_reader = changed
    with pytest.raises((ValueError,RuntimeError),match='source'):
        f.provider.capture(clock=clock())
    assert f.provider._previous == f.provider._bindings == f.provider._sources == {}
    assert f.provider._common_source is None and f.provider.last_sample is None
    f.provider.source_reader = original
    assert f.provider.capture(clock=clock()).fields['left.q'].valid


def test_boolean_backend_dof_type_array_cannot_validate_generalized_coordinate_units():
    f = fixture(); v = f.hands['left'].root_physx_view
    v.arrays['get_dof_types'] = v.arrays['get_dof_types'].astype(bool)
    s = f.provider.capture(clock=clock())
    assert not s.fields['left.q'].valid
    assert s.fields['right.q'].valid


def test_source_path_preread_and_actual_binding_cannot_diverge_inside_transaction():
    f = fixture(); original = f.provider.source_reader
    def moved(side,path):
        record = original(side,path)
        if side == 'left': f.hands[side].root_physx_view.prim_paths[1] += '_replaced'
        return record
    f.provider.source_reader = moved
    with pytest.raises((ValueError,RuntimeError),match='source|binding|path'):
        f.provider.capture(clock=clock())
    assert f.provider._previous == f.provider._bindings == f.provider._sources == {}
