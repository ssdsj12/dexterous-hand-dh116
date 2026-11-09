"""Fresh-metadata CPU facades only; no SDK, numerical contacts or Native truth."""
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import SampleClock

MODULE = 'go2_pvcnn.control.dh116.tabletop_bimanual_mpc_filtered_contact_measurements'


def api():
    assert importlib.util.find_spec(MODULE) is not None, 'fresh F0 metadata provider is missing'
    return importlib.import_module(MODULE)


def clock(env=0, **changes):
    return replace(SampleClock('CPU-only', env, f'ep:{env}', 10, 10/480,
                              1010, 5+10/480, 1, 0), **changes)


class ContactView:
    def __init__(self, side, body):
        self.paths = [f'/World/envs/env_{i}/{side}Hand/finger_{body}_link' for i in (1, 0)]
        self.filters = [[f'/World/envs/env_{i}/Wire'] for i in (1, 0)]
        self.borrowed_paths, self.borrowed_filters = [], []
        self.calls, self.faults, self.hook = [], {}, None

    def read(self, name, value):
        self.calls.append(name)
        if self.hook is not None:
            self.hook(name)
        if name in self.faults:
            fault = self.faults[name]
            if isinstance(fault, Exception):
                raise fault
            return fault
        return value

    @property
    def sensor_paths(self):
        self.borrowed_paths[:] = self.paths
        return self.read('sensor_paths', self.borrowed_paths)

    @property
    def filter_paths(self):
        self.borrowed_paths[:] = ['BUFFER_REUSED']
        self.borrowed_filters[:] = deepcopy(self.filters)
        return self.read('filter_paths', self.borrowed_filters)

    @property
    def filter_count(self):
        if self.borrowed_filters:
            self.borrowed_filters[0][:] = ['BUFFER_REUSED']
        return self.read('filter_count', 1)

    @property
    def sensor_count(self):
        return self.read('sensor_count', 2)


class BodyView:
    def __init__(self, contact):
        self.contact = contact
    @property
    def prim_paths(self):
        self.contact.borrowed_paths[:] = ['BUFFER_REUSED']
        return self.contact.read('body_prim_paths', self.contact.paths)


def fixture(**options):
    views = [ContactView(side, body) for side in ('left', 'right') for body in range(17)]
    sensors = [NS(contact_physx_view=view, body_physx_view=BodyView(view),
                  cfg=NS(filter_prim_paths_expr=['CONFIG_ONLY_TRAP'])) for view in views]
    now = [clock(0), clock(1)]
    source = dict(stage_identity='CPU-stage:13', contact_view_api=dict(
        path='/CPU-only/api.py', sha256='a'*64,
        python_type=type(views[0]).__module__+'.'+type(views[0]).__qualname__))
    source_reads, clock_reads = [], []
    def read_source():
        source_reads.append(True)
        return source
    def read_clocks():
        clock_reads.append(True)
        return list(now)
    wire = NS(prim_paths=['/World/envs/env_0/Wire', '/World/envs/env_1/Wire'])
    arguments = dict(sensors=sensors, wire_root_view=wire, clock_reader=read_clocks,
                     source_reader=read_source, enabled=True)
    arguments.update(options)
    provider = api().FilteredPairMetadataProvider(**arguments)
    return NS(provider=provider, sensors=sensors, views=views, now=now, source=source,
              source_reads=source_reads, clock_reads=clock_reads, wire=wire)


def assert_invalid(result):
    assert not result.metadata_binding_valid and result.pairs == () and result.errors
    assert not any(result.qualification.values()) and not result.contact_values_audited


def test_default_disabled_does_not_inspect_sources_or_clocks():
    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError(name)
    p = api().FilteredPairMetadataProvider(sensors=Forbidden(), wire_root_view=Forbidden(),
        clock_reader=Forbidden(), source_reader=Forbidden())
    result = p.capture(Forbidden())
    assert_invalid(result)
    assert result.raw['acquisition']['enabled'] is False


def test_fresh_actual_metadata_binds_all_rows_without_config_or_numeric_qualification():
    f = fixture()
    result = f.provider.capture(f.now)
    assert result.metadata_binding_valid and result.errors == ()
    assert len(result.pairs) == 68
    assert result.pairs[0].clock.env_id == 1 and result.pairs[0].sensor_row == 0
    assert result.pairs[0].filter_path == '/World/envs/env_1/Wire'
    assert len(f.source_reads) == 2 and f.clock_reads
    assert result.raw['sensor_sources']['sensors'][0]['sensor_count'] == 2
    assert result.raw['sensor_sources']['sensors'][0]['configured_filter_expressions'] == ('CONFIG_ONLY_TRAP',)
    assert all(p.actor_id is p.collider_id is p.partner_body_id is None for p in result.pairs)
    assert not any(result.qualification.values()) and not result.contact_values_audited


def test_each_borrowed_getter_is_owned_before_the_next_getter_reuses_its_buffer():
    f = fixture(); result = f.provider.capture(f.now)
    assert result.metadata_binding_valid
    raw = result.raw['sensor_sources']['sensors'][0]
    assert raw['sensor_paths'] == tuple(f.views[0].paths)
    assert raw['filter_paths']['value'] == (('/World/envs/env_1/Wire',), ('/World/envs/env_0/Wire',))
    f.views[0].paths[:] = ['/MUTATED']
    f.source['contact_view_api']['sha256'] = 'b'*64
    f.wire.prim_paths[:] = ['/MUTATED']
    assert result.pairs[0].sensor_path != '/MUTATED'
    assert result.raw['source_identity']['before']['contact_view_api']['sha256'] == 'a'*64
    with pytest.raises(TypeError):
        raw['sensor_paths'][0] = '/MUTATED'
    exported = result.to_dict(); exported['raw']['source_identity']['before']['contact_view_api']['sha256'] = 'c'*64
    assert result.raw['source_identity']['before']['contact_view_api']['sha256'] == 'a'*64


@pytest.mark.parametrize('getter', ['sensor_paths', 'body_prim_paths', 'filter_paths', 'filter_count', 'sensor_count'])
def test_getter_failure_is_explicit_invalid_and_preserves_other_raw_without_last_valid_fallback(getter):
    f = fixture(); previous = f.provider.capture(f.now)
    assert previous.metadata_binding_valid
    f.now[:] = [replace(c, world_step=11, sdk_step=1011, world_time_s=11/480, sdk_time_s=5+11/480) for c in f.now]
    f.views[0].faults[getter] = RuntimeError('CPU getter unavailable')
    result = f.provider.capture(f.now, previous=previous)
    assert_invalid(result)
    assert any(getter in error and 'CPU getter unavailable' in error for error in result.errors)
    assert result.raw['sensor_sources']['sensors'][1]['sensor_paths']
    assert previous.metadata_binding_valid and previous.pairs[0].clock.world_step == 10


@pytest.mark.parametrize('defect', ['flat_filter', 'regex_filter', 'count_bool', 'count_shape'])
def test_wrong_actual_shape_or_config_cannot_be_replaced_with_a_plausible_path(defect):
    f = fixture()
    if defect == 'flat_filter': f.views[0].faults['filter_paths'] = ['/World/envs/env_1/Wire', '/World/envs/env_0/Wire']
    elif defect == 'regex_filter': f.views[0].faults['filter_paths'] = [['/World/envs/env_.*/Wire']]*2
    elif defect == 'count_bool': f.views[0].faults['sensor_count'] = True
    else: f.views[0].faults['sensor_count'] = [2]
    result = f.provider.capture(f.now)
    assert_invalid(result)
    assert result.raw['sensor_sources']['sensors'][0]['configured_filter_expressions'] == ('CONFIG_ONLY_TRAP',)


@pytest.mark.parametrize('drift', ['clock', 'source', 'view'])
def test_drift_during_metadata_getter_returns_raw_and_no_pair_credit(drift):
    f = fixture()
    def changed(name):
        if name == 'sensor_count':
            if drift == 'clock': f.now[0] = replace(f.now[0], sdk_step=1011)
            elif drift == 'source': f.source['stage_identity'] = 'changed-stage'
            else: f.sensors[0].contact_physx_view = ContactView('left', 0)
    f.views[0].hook = changed
    result = f.provider.capture(f.now)
    assert_invalid(result)
    assert result.raw['acquisition']['read_brackets']
    assert result.raw['sensor_sources']['sensors'][0]['sensor_paths']


def test_source_reader_last_receipt_cannot_hide_a_view_lifecycle_change():
    f = fixture(); calls = []
    def source():
        calls.append(True)
        if len(calls) == 2:
            f.sensors[0].contact_physx_view = ContactView('left', 0)
        return f.source
    f.provider.source_reader = source
    result = f.provider.capture(f.now)
    assert_invalid(result)
    assert any('view identity changed' in error for error in result.errors)


def test_reset_refuses_getters_and_terminal_previous_cannot_be_retimestamped():
    f = fixture(); f.now[:] = [replace(c, reset=True) for c in f.now]
    result = f.provider.capture(f.now)
    assert_invalid(result)
    assert result.raw['acquisition']['expected_clocks'][0]['reset'] is True
    assert not f.clock_reads and not f.source_reads and not any(view.calls for view in f.views)
    f = fixture(); f.now[:] = [replace(c, terminal=True) for c in f.now]
    terminal = f.provider.capture(f.now)
    assert terminal.metadata_binding_valid
    f.now[:] = [replace(c, world_step=11, sdk_step=1011, world_time_s=11/480, sdk_time_s=5+11/480) for c in f.now]
    result = f.provider.capture(f.now, previous=terminal)
    assert_invalid(result)
    assert terminal.pairs[0].clock.terminal and terminal.pairs[0].clock.world_step == 10


@pytest.mark.parametrize('reader', ['clock_reader', 'source_reader'])
def test_missing_or_raising_required_reader_is_invalid_without_reusing_data(reader):
    def unavailable():
        raise RuntimeError('fixture reader unavailable')
    for value in (None, unavailable):
        f = fixture(**{reader: value})
        result = f.provider.capture(f.now)
        assert_invalid(result)
        assert any('unavailable' in error or 'TypeError' in error for error in result.errors)


def stage_fixture():
    class Stage:
        def GetRootLayer(self):
            return NS(identifier='CPU-root.usda', realPath='/CPU-only/root.usda')
        def GetSessionLayer(self):
            return NS(identifier='anon:CPU-session', realPath='')
    return Stage()


def test_source_reader_uses_stage_runtime_all_actual_view_types_and_owned_file_identity(tmp_path, monkeypatch):
    m = api(); f = fixture(); stage = stage_fixture()
    path = tmp_path/'actual_view_wrapper.py'; path.write_text('# CPU source fixture\n')
    inspected = []
    def actual_file(cls):
        inspected.append(cls)
        return str(path)
    monkeypatch.setattr(m, 'inspect', NS(getfile=actual_file))
    reader = m.make_source_reader(stage=stage, sensors=f.sensors, stage_reader=lambda: stage)
    first = reader()
    assert first['contact_view_api']['path'] == str(path.resolve())
    assert first['contact_view_api']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert first['contact_view_api']['python_type'] == ContactView.__module__+'.'+ContactView.__qualname__
    assert first['stage_python_runtime_id'] == id(stage)
    first['contact_view_api']['sha256'] = 'forged'
    second = reader()
    assert second['contact_view_api']['sha256'] != 'forged'
    assert inspected == [ContactView, ContactView]
    other_stage = stage_fixture()
    other_reader = m.make_source_reader(stage=other_stage, sensors=f.sensors, stage_reader=lambda: other_stage)
    assert other_reader()['stage_identity'] != second['stage_identity']


def test_source_reader_checks_stat_each_time_and_rehashes_only_changed_files(tmp_path, monkeypatch):
    m = api(); f = fixture(); stage = stage_fixture()
    path = tmp_path/'view.py'; path.write_bytes(b'first')
    monkeypatch.setattr(m, 'inspect', NS(getfile=lambda cls: str(path)))
    real_read, reads = Path.read_bytes, []
    def reading(p):
        if p == path:
            reads.append(True)
        return real_read(p)
    monkeypatch.setattr(Path, 'read_bytes', reading)
    reader = m.make_source_reader(stage=stage, sensors=f.sensors, stage_reader=lambda: stage)
    first, second = reader(), reader()
    assert first == second and len(reads) == 1
    path.write_bytes(b'changed file with another size')
    third = reader()
    assert len(reads) == 2 and third['contact_view_api']['sha256'] != first['contact_view_api']['sha256']


def test_source_reader_rejects_file_changed_while_hashing_then_retries_fresh(tmp_path, monkeypatch):
    m = api(); f = fixture(); stage = stage_fixture()
    path = tmp_path/'view.py'; path.write_bytes(b'first')
    monkeypatch.setattr(m, 'inspect', NS(getfile=lambda cls: str(path)))
    real_read, changed = Path.read_bytes, [False]
    def reading(p):
        data = real_read(p)
        if p == path and not changed[0]:
            changed[0] = True
            path.write_bytes(b'changed while old bytes were being read')
        return data
    monkeypatch.setattr(Path, 'read_bytes', reading)
    reader = m.make_source_reader(stage=stage, sensors=f.sensors, stage_reader=lambda: stage)
    with pytest.raises(RuntimeError, match='changed'):
        reader()
    result = reader()
    assert result['contact_view_api']['sha256'] == hashlib.sha256(real_read(path)).hexdigest()


@pytest.mark.parametrize('defect', ['stage', 'class', 'sensor_count', 'missing_file'])
def test_source_reader_refuses_drift_or_missing_actual_protocol(tmp_path, monkeypatch, defect):
    m = api(); f = fixture(); stage = stage_fixture()
    current = [stage]
    path = tmp_path/'view.py'; path.write_bytes(b'CPU-only source')
    monkeypatch.setattr(m, 'inspect', NS(getfile=lambda cls: str(path)))
    reader = m.make_source_reader(stage=stage, sensors=f.sensors, stage_reader=lambda: current[0])
    reader()
    if defect == 'stage': current[0] = stage_fixture()
    elif defect == 'class':
        class Other(ContactView): pass
        f.sensors[-1].contact_physx_view = Other('right', 16)
    elif defect == 'sensor_count': f.sensors.pop()
    else: path.unlink()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        reader()


@pytest.mark.parametrize('drift', ['file', 'view'])
def test_source_reader_rechecks_cached_file_and_views_after_layer_getters(tmp_path, monkeypatch, drift):
    m = api(); f = fixture(); stage = stage_fixture()
    path = tmp_path/'view.py'; path.write_bytes(b'cached source')
    monkeypatch.setattr(m, 'inspect', NS(getfile=lambda cls: str(path)))
    reader = m.make_source_reader(stage=stage, sensors=f.sensors, stage_reader=lambda: stage)
    reader()
    original = stage.GetRootLayer
    def changed():
        if drift == 'file':
            path.write_bytes(b'API source changed during a cached identity read')
        else:
            f.sensors[0].contact_physx_view = ContactView('left', 0)
        return original()
    monkeypatch.setattr(stage, 'GetRootLayer', changed)
    with pytest.raises(RuntimeError, match='changed'):
        reader()


def test_provider_import_contains_no_sdk_or_numeric_contact_getter(tmp_path):
    import ast
    m = api(); tree = ast.parse(Path(m.__file__).read_text())
    imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
    assert not any(name.startswith(('omni', 'isaac', 'pxr', 'torch', 'numpy')) for name in imports)
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not any(name in calls for name in ('step', 'refresh', 'Apply', 'Set', 'get_net_contact_forces',
                                             'get_contact_force_data', 'get_friction_data'))
