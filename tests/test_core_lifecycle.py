"""CPU lifetime contracts at the production owner boundary; no SDK qualification."""
import builtins
import gc
import json
import os
import sys
import weakref
from types import ModuleType
from types import SimpleNamespace
import pytest
from dh116.env.lifecycle import NativeOwnershipTrace
from dh116.env.lifecycle import TimelineStopOwner
from dh116.env.lifecycle import run_lifecycle
from dh116.env.lifecycle import run_lifecycle_with_stat_cache

@pytest.fixture(autouse=True)
def forbid_forced_exit(monkeypatch):
    attempted = []

    def reject(status):
        attempted.append(status)
        raise AssertionError('native cleanup must return or raise without forced exit')
    monkeypatch.setattr(os, '_exit', reject)
    yield
    assert attempted == []

class _Sim:

    def __init__(self, step):
        self.current_time_step_index = step
        self.current_time = step / 480

class _Runtime:

    def __init__(self, sim, cached_frames):
        self.sim = sim
        self.cycle = self
        if cached_frames is not None:
            cached_frames.append(sys._getframe())

class _Composer:

    def __init__(self, events):
        self.events = events

    def reset(self):
        self.events.append('composer_reset')

class _ObservedTrace(NativeOwnershipTrace):

    def __init__(self, path, events):
        super().__init__(path)
        self.events = events

    def checkpoint(self, phase):
        self.events.append('checkpoint:' + phase)
        super().checkpoint(phase)

class _Run:

    def __init__(self, output, *, damage=None, step=2, cached=False):
        (self.output, self.damage, self.step) = (output, damage, step)
        (self.events, self.refs, self.held) = ([], {}, [])
        self.cached_frames = [] if cached else None
        self.failed_frames = []
        self.completed = None
        self.cache_calls = []
        self.app = SimpleNamespace(update=self.update, close=self.close)
        self.kwargs = dict(make_sim=self.make_sim, acquire=self.acquire, synchronize=self.synchronize, clear_sim=self.clear, stop=self.stop, manager_is_none=self.manager, freeze=self.freeze, output=output, cleanup='explicit', owned_names=('runtime', 'composer'), trace_factory=lambda path: _ObservedTrace(path, self.events))

    def hit(self, name):
        self.events.append(name)
        if self.damage == name:
            raise RuntimeError('injected ' + name)

    def make_sim(self):
        self.hit('make_sim')
        sim = _Sim(self.step)
        self.refs['simulation'] = weakref.ref(sim)
        return sim

    def acquire(self, sim, owners, trace, cleanup):
        runtime = owners.keep('runtime', _Runtime(sim, self.cached_frames))
        composer = owners.keep('composer', _Composer(self.events))
        for (name, value) in (('runtime', runtime), ('composer', composer)):
            self.refs[name] = weakref.ref(value)
            trace.watch(name, value, owner='CPU fixture', release='owned scope then STOP')
        if self.damage == 'retained_owner':
            self.held.append(runtime)
        if self.damage == 'acquire':
            self.failed_frames.append(sys._getframe())
        self.hit('acquire')
        self.completed = self.step
        return {'sdk_step': self.step, 'physics_rows': 0}

    def synchronize(self):
        self.hit('synchronize')

    def release_cache(self, sim, *, expected_sdk_step):
        assert sim is self.refs['simulation']()
        assert self.events[-1] == 'synchronize'
        assert all((frame.f_locals == {} for frame in self.failed_frames))
        self.cache_calls.append(expected_sdk_step)
        self.events.append('cache_release')
        if self.cached_frames is not None:
            assert self.refs['runtime']() is not None
            self.cached_frames.clear()

    def clear(self, sim):
        self.hit('clear')

    def stop(self):
        self.hit('stop')

    def freeze(self):
        self.events.append('freeze')
        return False

    def manager(self):
        self.events.append('manager')
        return None if self.damage == 'manager_unknown' else True

    def update(self):
        self.hit('update')

    def close(self):
        self.hit('close')

    def read_sim(self):
        reference = self.refs.get('simulation')
        return reference() if reference is not None else None

    def with_cache(self, **overrides):
        options = dict(cache_release=self.release_cache, read_sim=self.read_sim, read_completed_sdk_step=lambda : self.completed, max_sdk_step=66)
        options.update(overrides)
        return run_lifecycle_with_stat_cache(self.app, lifecycle=run_lifecycle, **options, **self.kwargs)

def _read(output, name):
    return json.loads((output / name).read_text())

def _assert_unqualified(result):
    assert result['native_qualification'] is False
    assert result['teacher_admission'] is False
    assert result['grasp_qualified'] is False
    assert result['process_exit_verified'] is False

@pytest.mark.parametrize('source_failure', [False, True])
def test_cache_release_and_gc_drop_frame_owned_resources_before_stop(tmp_path, source_failure):
    run = _Run(tmp_path, damage='acquire' if source_failure else None, cached=True)
    if source_failure:
        with pytest.raises(RuntimeError, match='injected acquire') as raised:
            run.with_cache()
        assert raised.value.__context__ is None
    else:
        run.with_cache()
    result = _read(tmp_path, 'result.json')
    _assert_unqualified(result)
    assert result['close_mode'] == 'normal'
    assert bool(result['failure']) is source_failure
    assert run.cache_calls == [2]
    order = ['composer_reset', 'synchronize', 'cache_release', 'checkpoint:owned hand and solver references dropped', 'clear', 'stop', 'manager', 'update', 'close']
    assert [run.events.index(name) for name in order] == sorted((run.events.index(name) for name in order))
    assert run.events.count('stop') == run.events.count('update') == run.events.count('close') == 1
    assert all((reference() is None for reference in run.refs.values()))
    rows = [json.loads(row) for row in (tmp_path / 'ownership.jsonl').read_text().splitlines()]
    dropped = next((row for row in rows if row['phase'] == 'owned hand and solver references dropped'))
    assert dropped['python_liveness']['runtime'] is False
    assert dropped['python_liveness']['composer'] is False
    assert dropped['python_liveness']['simulation'] is True
    assert _read(tmp_path, 'app_close_returned.json')['app_close_returned'] is True

@pytest.mark.parametrize('damage', ['clear', 'stop', 'manager_unknown', 'retained_owner'])
def test_unsafe_stop_never_pumps_app_and_still_attempts_one_close(tmp_path, damage):
    run = _Run(tmp_path, damage=damage)
    try:
        with pytest.raises(RuntimeError):
            run_lifecycle(run.app, **run.kwargs)
        assert 'update' not in run.events
        assert run.events.count('stop') == run.events.count('close') == 1
        result = _read(tmp_path, 'result.json')
        assert result['close_mode'] == 'fault' and result['failure']
        _assert_unqualified(result)
        stages = _read(tmp_path, 'cleanup_stages.json')['stages']
        assert stages['post_stop_update']['attempted'] is False
        assert _read(tmp_path, 'fault_app_close_returned.json')['app_close_returned'] is True
        if damage == 'retained_owner':
            assert run.refs['runtime']() is not None
        else:
            assert all((reference() is None for reference in run.refs.values()))
    finally:
        run.held.clear()
        gc.collect()

@pytest.mark.parametrize('damage', ['before_marker', 'close'])
def test_evidence_or_close_failure_cannot_skip_or_repeat_app_close(tmp_path, monkeypatch, damage):
    import dh116.env.contracts as runtime_io
    run = _Run(tmp_path, damage='close' if damage == 'close' else None)
    if damage == 'before_marker':
        original = runtime_io.write_new

        def write(path, value):
            if path.name == 'before_app_close.json':
                raise OSError('injected before marker')
            return original(path, value)
        monkeypatch.setattr(runtime_io, 'write_new', write)
    with pytest.raises(RuntimeError, match='injected'):
        run_lifecycle(run.app, **run.kwargs)
    assert run.events.count('close') == 1
    assert run.events.count('stop') == 1
    result = _read(tmp_path, 'result.json')
    assert result['failure']
    _assert_unqualified(result)
    assert all((reference() is None for reference in run.refs.values()))
    assert (tmp_path / 'app_close_returned.json').exists() is (damage != 'close')

class _Timeline:

    def __init__(self, events, clock, damage=None):
        (self.events, self.clock, self.damage) = (events, clock, damage)
        self.pending = self.stopped = False

    def stop(self):
        self.events.append('timeline_stop')
        self.pending = True

    def commit(self):
        self.events.append('timeline_commit')
        if self.damage == 'commit':
            raise RuntimeError('injected timeline commit')
        self.stopped = self.pending
        if self.damage == 'clock':
            self.clock['sdk_step'] += 1

    def is_stopped(self):
        return self.stopped

def _timeline_owner(output, *, damage=None):
    (events, references, held) = ([], [], [])
    clock = {'sdk_step': 66, 'sdk_time_s': 66 / 480}

    def acquire():
        timeline = _Timeline(events, clock, damage)
        references.append(weakref.ref(timeline))
        if damage == 'retained':
            held.append(timeline)
        return timeline

    def freeze():
        events.append('freeze')
        return False
    return (TimelineStopOwner(acquire=acquire, freeze=freeze, read_clock=lambda : dict(clock), output=output), events, references, held)

def test_stop_commit_drops_temporary_timeline_without_advancing_sdk(tmp_path):
    (owner, events, references, _) = _timeline_owner(tmp_path)
    result = owner.stop()
    assert events == ['freeze', 'timeline_stop', 'timeline_commit', 'freeze']
    assert result['clock_before'] == result['clock_after'] == {'sdk_step': 66, 'sdk_time_s': 66 / 480}
    assert references[0]() is None
    assert owner.before_close() is True
    report = _read(tmp_path, 'timeline_release.json')
    assert report['local_timeline_weakref_alive'] is False
    assert report['local_timeline_release_verified'] is True
    assert report['native_release_verified'] is report['process_exit_verified'] is report['qualification_granted'] is False
    with pytest.raises(RuntimeError, match='once only'):
        owner.stop()
    with pytest.raises(RuntimeError, match='once only'):
        owner.before_close()
    assert events.count('timeline_commit') == 1

@pytest.mark.parametrize('damage', ['retained', 'clock', 'commit'])
def test_failed_or_retained_timeline_cannot_attest_release(tmp_path, damage):
    (owner, _, references, held) = _timeline_owner(tmp_path, damage=damage)
    try:
        if damage == 'retained':
            owner.stop()
            assert references[0]() is not None
        else:
            with pytest.raises(RuntimeError, match='clock|commit'):
                owner.stop()
            assert references[0]() is None
        with pytest.raises(RuntimeError, match='retained|not verified'):
            owner.before_close()
        report = _read(tmp_path, 'timeline_release.json')
        assert report['local_timeline_release_verified'] is False
        assert report['native_release_verified'] is report['qualification_granted'] is False
    finally:
        held.clear()
        gc.collect()

@pytest.mark.parametrize('step,completed,missing_sim', [(0, None, False), (67, None, False), (2, 3, False), (2, None, True)])
def test_cache_release_refuses_unbound_or_out_of_budget_sdk_clock(tmp_path, step, completed, missing_sim):
    run = _Run(tmp_path, step=step)
    read_sim = (lambda : None) if missing_sim else run.read_sim
    with pytest.raises(RuntimeError, match='SDK clock|SDK budget|clock differs|clock owner'):
        run.with_cache(read_sim=read_sim, read_completed_sdk_step=lambda : completed)
    assert run.cache_calls == []
    assert run.events.count('stop') == run.events.count('close') == 1
    assert all((reference() is None for reference in run.refs.values()))
    _assert_unqualified(_read(tmp_path, 'result.json'))

def test_completed_sdk_step_at_inclusive_budget_is_released_once(tmp_path):
    run = _Run(tmp_path, step=66)
    result = run.with_cache()
    assert run.cache_calls == [66]
    assert result['physics_steps'] == 66 and result['failure'] is None
    assert run.events.count('close') == 1
    _assert_unqualified(result)

def test_native_session_restores_arguments_and_closes_after_post_launch_import_failure(tmp_path, monkeypatch):
    from dh116.env.lifecycle import run_native_session
    events = []
    app = SimpleNamespace(close=lambda **kwargs: events.append(('close', kwargs)))

    class Launcher:

        def __init__(self, **kwargs):
            events.append(('launch', tuple(sys.argv)))
            self.app = app
    package = ModuleType('isaaclab')
    package.__path__ = []
    launcher_module = ModuleType('isaaclab.app')
    launcher_module.AppLauncher = Launcher
    package.app = launcher_module
    monkeypatch.setitem(sys.modules, 'isaaclab', package)
    monkeypatch.setitem(sys.modules, 'isaaclab.app', launcher_module)
    original_import = builtins.__import__

    def import_without_sdk(name, *args, **kwargs):
        if name == 'carb.settings':
            raise ImportError('injected post-launch dependency failure')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', import_without_sdk)
    argv = ['teacher', '--external-argument']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ImportError, match='post-launch dependency failure'):
        run_native_session(tmp_path, SimpleNamespace(experience='fixture.kit'), 0, 2, tmp_path, max_sdk_step=66, author_and_exercise=lambda *args: pytest.fail('authoring preceded SDK startup'), owned_names=('runtime',), runtime_source_audit=False)
    assert sys.argv is argv
    assert events == [('launch', ('teacher',)), ('close', {'wait_for_replicator': False})]


@pytest.mark.parametrize(('count', 'columns', 'width'), [(512, 23, 3), (1024, 32, 4)])
def test_scene_lanes_order_and_stage_paths(count, columns, width):
    from dh116.env.lifecycle import scene_lanes

    lanes = scene_lanes(count, (1.5, 1.0, 0.1))
    assert len(lanes) == count
    assert all(
        path == f'/World/DH116BatchAudit/lane_{i:0{width}d}'
        and origin == (2.5 * (i % columns), 2.5 * (i // columns), 0.0)
        for i, (path, origin) in enumerate(lanes)
    )
    with pytest.raises(ValueError):
        scene_lanes(1025, (1.5, 1.0, 0.1))


def test_1024_physical_lanes_reach_regex_scene_authoring(tmp_path, monkeypatch):
    from dh116.env import scene
    from dh116.env.lifecycle import NativeOwners, scene_lanes
    from dh116.hand.articulation import prepare_regex_task

    class ReachedAuthoring(Exception):
        pass

    def stop_at_scene_inputs(*args):
        raise ReachedAuthoring

    monkeypatch.setattr(scene, 'spawn_parameters', stop_at_scene_inputs)
    sim = SimpleNamespace(current_time_step_index=0)
    archive = SimpleNamespace(style='pinch', scene_inputs={})
    lanes = scene_lanes(1024, (1.5, 1.0, 0.1))
    with pytest.raises(ReachedAuthoring):
        prepare_regex_task(sim, NativeOwners(), None, tmp_path, {}, archive, lanes)
    with pytest.raises(ValueError, match='original physical lanes'):
        prepare_regex_task(sim, NativeOwners(), None, tmp_path, {}, archive, lanes + [lanes[-1]])


@pytest.fixture
def physical_gpus(tmp_path, monkeypatch):
    import subprocess
    from dh116.env import lifecycle

    pytest.importorskip('fcntl')
    uuids = {
        0: 'GPU-b7b79e46-2e95-fd6b-2ac9-38fad64fe99c',
        1: 'GPU-1d290889-01ad-85ec-f69d-bfe9e1762c5e',
    }
    memory = {}
    occupancy_queries = []
    monkeypatch.setattr(lifecycle, 'NATIVE_LOCK', tmp_path / 'dh116-isaac-single-instance.lock')

    def query(argv, *, text):
        assert text and argv[0] == 'nvidia-smi' and argv[1] == '-i'
        selector = argv[2]
        if argv[3] == '--query-gpu=index,uuid':
            return f'{selector}, {uuids[int(selector)]}\n'
        assert argv[3] == '--query-gpu=uuid,memory.used'
        assert selector in uuids.values()
        occupancy_queries.append(selector)
        return f'{selector}, {memory.get(selector, 0)}\n'

    monkeypatch.setattr(subprocess, 'check_output', query)
    return SimpleNamespace(uuids=uuids, memory=memory, occupancy_queries=occupancy_queries)


def test_native_gpu_lock_excludes_same_gpu_and_allows_distinct_gpus(physical_gpus, monkeypatch):
    from dh116.env.lifecycle import native_gpu_lock

    first, second = physical_gpus.uuids.values()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', first)
    with native_gpu_lock(0) as gpu_uuid:
        assert gpu_uuid == first
        with pytest.raises(BlockingIOError), native_gpu_lock(0):
            pytest.fail('the physical GPU already has an owner')
        assert physical_gpus.occupancy_queries == [first]
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', second)
        with native_gpu_lock(1) as gpu_uuid:
            assert gpu_uuid == second
    assert physical_gpus.occupancy_queries == [first, second]


def test_native_gpu_lock_conflicts_with_legacy_lock_in_both_directions(physical_gpus, monkeypatch):
    from dh116.env.lifecycle import NATIVE_LOCK, native_file_lock, native_gpu_lock

    gpu_uuid = physical_gpus.uuids[0]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', gpu_uuid)
    with native_file_lock(NATIVE_LOCK):
        with pytest.raises(BlockingIOError), native_gpu_lock(0):
            pytest.fail('the legacy launch already has an owner')
        assert physical_gpus.occupancy_queries == []
    with native_gpu_lock(0):
        with pytest.raises(BlockingIOError), native_file_lock(NATIVE_LOCK):
            pytest.fail('the legacy launch overlaps a per-GPU launch')


@pytest.mark.parametrize('selector', ['1', 'GPU-b7b79e46', 'GPU-1d290889-01ad-85ec-f69d-bfe9e1762c5e'])
def test_native_gpu_lock_requires_verified_full_cuda_uuid(physical_gpus, monkeypatch, selector):
    from dh116.env.lifecycle import native_gpu_lock

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', selector)
    with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES'), native_gpu_lock(0):
        pytest.fail('the CUDA device has not been bound to this physical GPU')
    assert physical_gpus.occupancy_queries == []


def test_physical_gpu_lookup_rejects_invalid_or_mismatched_inventory(monkeypatch):
    import subprocess
    from dh116.env.lifecycle import resolve_physical_gpu_uuid

    for response in ('0, GPU-short', '1, GPU-b7b79e46-2e95-fd6b-2ac9-38fad64fe99c'):
        monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: response)
        with pytest.raises(ValueError, match='physical GPU'):
            resolve_physical_gpu_uuid(0)


def test_native_gpu_lock_releases_both_locks_on_occupied_gpu(physical_gpus, monkeypatch):
    from dh116.env.lifecycle import NATIVE_LOCK, native_file_lock, native_gpu_lock

    gpu_uuid = physical_gpus.uuids[0]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', gpu_uuid)
    physical_gpus.memory[gpu_uuid] = 513
    with pytest.raises(RuntimeError, match='physical GPU is occupied'), native_gpu_lock(0):
        pytest.fail('an occupied physical GPU cannot launch')
    with native_file_lock(NATIVE_LOCK):
        pass
    physical_gpus.memory[gpu_uuid] = 512
    with native_gpu_lock(0) as verified:
        assert verified == gpu_uuid
