"""Single-instance ownership, lifetime release and source/SDK cleanup auditing."""
import gc as nativ_owner__gc, json as nativ_owner__json, os, time, types as nativ_owner__types, weakref as nativ_owner__weakref
import sys as nativ_sessi__sys
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path as nativ_resou_lock__Path
from pathlib import Path as runti_lifec__Path
import hashlib as runti_sourc_audit__hashlib
import gc as simul_owner_clean__gc
import hashlib as stat_cache_relea__hashlib
from pathlib import Path as nativ_owner__Path
import math as nativ_sessi__math
NATIVE_LOCK = nativ_resou_lock__Path('/tmp/dh116-isaac-single-instance.lock')
import traceback as runti_lifec__traceback
import json as runti_sourc_audit__json
import hashlib as simul_owner_clean__hashlib
import functools
from dh116.env.contracts import write_new as write_json
from pathlib import Path as nativ_sessi__Path

def native_file_lock(path, *, acquire=True, shared=False):
    """Open a nonblocking Linux native launch lock."""
    import fcntl
    stream = path.open('a+b')
    try:
        if acquire:
            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(stream.fileno(), mode | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def _physical_gpu_query(selector, fields):
    query = [
        'nvidia-smi', '-i', str(selector), f'--query-gpu={fields}',
        '--format=csv,noheader,nounits',
    ]
    rows = subprocess.check_output(query, text=True).strip().splitlines()
    values = [value.strip() for value in rows[0].split(',')] if len(rows) == 1 else []
    if len(values) != 2 or not all(values):
        raise ValueError(f'expected one physical GPU record for {selector}')
    return values


def resolve_physical_gpu_uuid(gpu):
    """Resolve an nvidia-smi physical index to its full CUDA GPU UUID."""
    if type(gpu) is not int or gpu < 0:
        raise ValueError('nonnegative physical GPU index required')
    index, gpu_uuid = _physical_gpu_query(gpu, 'index,uuid')
    if index != str(gpu) or re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', gpu_uuid) is None:
        raise ValueError(f'nvidia-smi did not verify the physical GPU UUID for index {gpu}')
    return 'GPU-' + gpu_uuid[4:].lower()


@contextmanager
def native_gpu_lock(gpu):
    """Hold the legacy shared lock and the physical GPU's exclusive launch lock."""
    gpu_uuid = resolve_physical_gpu_uuid(gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must match the verified physical GPU UUID')
    gpu_path = NATIVE_LOCK.with_name(f'dh116-isaac-{gpu_uuid}.lock')
    with native_file_lock(NATIVE_LOCK, shared=True), native_file_lock(gpu_path):
        reported_uuid, memory_used = _physical_gpu_query(gpu_uuid, 'uuid,memory.used')
        if reported_uuid != gpu_uuid or not memory_used.isascii() or not memory_used.isdecimal():
            raise ValueError(f'nvidia-smi did not verify physical GPU occupancy for {gpu_uuid}')
        if int(memory_used) > 512:
            raise RuntimeError(f'physical GPU is occupied: {gpu_uuid}')
        yield gpu_uuid


import weakref as runti_lifec__weakref
from pathlib import Path as runti_sourc_audit__Path
from pathlib import Path as simul_owner_clean__Path
import importlib
from dh116.env.contracts import clock as nativ_owner__clock
from types import SimpleNamespace
runti_lifec__VERSION = 'dh116-shared-native-session-v1'
runti_sourc_audit__VERSION = 'dh116-attempt-source-freeze-audit-v1'
import traceback as simul_owner_clean__traceback
import math as stat_cache_relea__math
nativ_owner__VERSION = 'dh116-native-lifetime-audit-v1'
from weakref import ref

def commit_preview_stop(timeline, *, freeze, read_clock):
    """Dispatch real pending STOP callbacks without rendering or App.update.

    The installed Timeline.commit API applies pending changes and callbacks on
    the main thread. Manager release is still checked by the existing lifecycle.
    Never write the manager's view field or invent a successful readback.
    """
    import threading
    try:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('STOP commit requires the main thread')
        if freeze() is not False:
            raise RuntimeError('STOP physics freeze failed')
        before = read_clock()
        timeline.stop()
        timeline.commit()
        if timeline.is_stopped() is not True:
            raise RuntimeError('timeline STOP did not read back true')
        if freeze() is not False:
            raise RuntimeError('STOP physics freeze failed after commit')
        after = read_clock()
        if after is not None and after != before:
            raise RuntimeError('STOP commit changed physical clock')
        return dict(stop_commit_returned=True, clock_before=before, clock_after=after, app_updates=0, explicit_physics_steps_requested=0, physical_clock_unchanged=True if before is not None and after == before else None, native_release_verified=False)
    finally:
        timeline = freeze = read_clock = None
SOURCE_NAMES = ()
import types as simul_owner_clean__types
from pathlib import Path as stat_cache_relea__Path
TRACE_VERSION = nativ_owner__VERSION
import numpy as np

def _detach_exception_frames(error, seen=None):
    """Drop completed frames throughout a chained native failure, retaining text only."""
    seen = set() if seen is None else seen
    if id(error) in seen:
        return
    seen.add(id(error))
    related = (error.__cause__, error.__context__, *getattr(error, 'exceptions', ()))
    runti_lifec__traceback.clear_frames(error.__traceback__)
    error.__traceback__ = None
    error.__cause__ = error.__context__ = None
    for nested in related:
        if nested is not None:
            _detach_exception_frames(nested, seen)

def _sha(path):
    return runti_sourc_audit__hashlib.sha256(path.read_bytes()).hexdigest()
import weakref as simul_owner_clean__weakref
import sys as stat_cache_relea__sys
BOUNDARY_VERSION = 'dh116-pinch-import-boundary-v3'

def scene_lanes(count, table_size):
    size = np.asarray(table_size, dtype=float)
    if count not in (2, 32, 64, 128, 512) or size.shape != (3,) or (not np.isfinite(size).all()) or (size <= 0).any():
        raise ValueError('2, 32, 64, 128 or 512 lanes and a positive original table size required')
    columns = nativ_sessi__math.ceil(nativ_sessi__math.sqrt(count))
    spacing = float(max(size[:2]) + 1.0)
    return [(f'/World/DH116BatchAudit/lane_{i:03d}', (spacing * (i % columns), spacing * (i // columns), 0.0)) for i in range(count)]

class TimelineStopOwner:
    """Own the temporary STOP wrapper only, never an articulation or CUDA view."""

    def __init__(self, *, acquire, freeze, read_clock, output):
        (self._acquire, self._freeze, self._read_clock) = (acquire, freeze, read_clock)
        self.output = runti_lifec__Path(output)
        self._reference = None
        self._attempted = self._committed = self._release_attempted = False
        self._weakref_error = None
        self._type = None

    def stop(self):
        if self._attempted:
            raise RuntimeError('timeline STOP is once only')
        self._attempted = True
        timeline = None
        fault = None
        result = None
        try:
            timeline = self._acquire()
            self._type = type(timeline).__module__ + '.' + type(timeline).__qualname__
            try:
                self._reference = runti_lifec__weakref.ref(timeline)
            except TypeError as error:
                self._weakref_error = str(error)
            result = commit_preview_stop(timeline, freeze=self._freeze, read_clock=self._read_clock)
            self._committed = True
            if self._reference is None:
                raise RuntimeError('timeline weakref unsupported; release is unknown')
        except BaseException as error:
            fault = runti_lifec__traceback.format_exc()
            _detach_exception_frames(error)
        finally:
            timeline = None
        if fault is not None:
            raise RuntimeError(fault) from None
        return result

    def before_close(self):
        if self._release_attempted:
            raise RuntimeError('timeline release check is once only')
        self._release_attempted = True
        self._acquire = self._freeze = self._read_clock = None
        alive = self._reference() is not None if self._reference is not None else None
        verified = self._committed and alive is False and (self._weakref_error is None)
        report = dict(version='dh116-local-timeline-release-v1', stop_commit_returned=self._committed, local_timeline_type=self._type, local_timeline_weakref_supported=self._reference is not None, local_timeline_weakref_alive=alive, local_timeline_release_verified=verified, weakref_error=self._weakref_error, sdk_global_owners_modified=False, native_release_verified=False, process_exit_verified=False, qualification_granted=False)
        from dh116.env.contracts import write_new
        write_new(self.output / 'timeline_release.json', report)
        if not verified:
            raise RuntimeError('timeline wrapper retained' if alive else 'timeline release not verified')
        return True

class SourceFreezeAudit:

    def __init__(self, *, project, attempt, output):
        (self.project, self.attempt, self.output) = (runti_sourc_audit__Path(project), runti_sourc_audit__Path(attempt), runti_sourc_audit__Path(output))
        if self.attempt.is_symlink():
            raise ValueError('source attempt must be an owned regular file')
        self.attempt_sha256 = _sha(self.attempt)
        inventory = runti_sourc_audit__json.loads(self.attempt.read_text()).get('source_sha256')
        if not isinstance(inventory, dict) or not inventory or any((not name.startswith('src/') or '..' in runti_sourc_audit__Path(name).parts for name in inventory)):
            raise ValueError('complete shared runtime source inventory required')
        self.expected = dict(inventory)
        self._started = self._finished = False

    def _inspect(self):
        (drift, actual) = ([], {})
        for (name, expected) in self.expected.items():
            path = self.project / name
            value = _sha(path) if path.is_file() and (not path.is_symlink()) else None
            actual[name] = value
            if value != expected:
                drift.append(dict(path=name, before=expected, after=value))
        value = _sha(self.attempt)
        if value != self.attempt_sha256:
            drift.append(dict(path='attempt.json', before=self.attempt_sha256, after=value))
        return (actual, drift)

    def _write(self, actual, drift, *, admissible):
        from dh116.env.contracts import write_new
        write_new(self.output / 'source_freeze_audit.json', dict(version=runti_sourc_audit__VERSION, attempt_sha256=self.attempt_sha256, source_sha256=actual, drift=drift, admissible=admissible, startup_checked=self._started, process_exit_verified=False, qualification_granted=False))

    def check_startup(self):
        if self._started:
            raise ValueError('source freeze startup audit is once only')
        (actual, drift) = self._inspect()
        if drift:
            self._write(actual, drift, admissible=False)
            raise ValueError('source freeze changed before native acquisition')
        self._started = True

    def before_close(self):
        if self._finished:
            raise ValueError('source freeze pre-close audit is once only')
        self._finished = True
        (actual, drift) = self._inspect()
        admissible = self._started and (not drift)
        self._write(actual, drift, admissible=admissible)
        if not admissible:
            raise ValueError('source freeze changed or startup audit missing')
        return True
import types as stat_cache_relea__types
import sys as nativ_owner__sys

def run_native_session(project, archive, gpu, num_envs, output, *, max_sdk_step, author_and_exercise,
                       owned_names, runtime_source_audit=True, enable_cameras=False):
    _nativ_sessi_import__assert_clean_import_boundary = assert_clean_import_boundary
    _nativ_sessi_import__assert_clean_import_boundary()
    from isaaclab.app import AppLauncher
    argv = nativ_sessi__sys.argv
    try:
        nativ_sessi__sys.argv = argv[:1]
        # The offscreen camera experience caps the renderer at one adapter.
        # Its default selector maps the single visible CUDA UUID to Vulkan.
        experience = '' if enable_cameras else archive.experience
        app = AppLauncher(headless=True, device='cuda:0', enable_cameras=enable_cameras,
                          fast_shutdown=False, experience=experience).app
    finally:
        nativ_sessi__sys.argv = argv
    entered = False
    try:
        import carb.settings
        import isaaclab.sim as sim_utils
        from isaacsim.core.simulation_manager import SimulationManager
        import omni.timeline
        import torch
        _nativ_sessi_import__TimelineStopOwner = TimelineStopOwner
        _nativ_sessi_import__run_lifecycle_with_stat_cache = run_lifecycle_with_stat_cache
        _nativ_sessi_import__clear_known_simulation_instance = clear_known_simulation_instance
        _nativ_sessi_import__simulation_source_contract = simulation_source_contract
        _nativ_sessi_import__stat_cache_source_contract = stat_cache_source_contract
        _nativ_sessi_import__run_lifecycle = run_lifecycle
        _nativ_sessi_import__StartupTrace = StartupTrace
        from dh116.env.contracts import clock
        from dh116.env.contracts import write_new
        _nativ_sessi_import__make_before_collect = make_before_collect
        settings = carb.settings.get_settings()
        clear_contract = _nativ_sessi_import__simulation_source_contract()
        cache = _nativ_sessi_import__make_before_collect(dict(output=str(output), stat_cache_release=dict(source_contract=_nativ_sessi_import__stat_cache_source_contract(), max_reset_calls=1)))
        sim_reference = None
        completed_step = None

        def freeze():
            settings.set_bool('/app/player/playSimulations', False)
            return settings.get('/app/player/playSimulations')

        def make_sim():
            nonlocal sim_reference
            sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 480, device='cuda:0', render_interval=1, gravity=(0.0, 0.0, -9.81), physx=sim_utils.PhysxCfg(solver_type=0)))
            sim_reference = ref(sim)
            return sim

        def acquire(sim, owners, trace, cleanup):
            nonlocal completed_step
            from dh116.env.scene import prepared_scene
            if source_audit is not None:
                source_audit.check_startup()
            prepared = prepared_scene(archive.scene_inputs)
            lanes = scene_lanes(num_envs, prepared['table']['size_m'])
            result = author_and_exercise(sim, owners, trace, project, prepared, archive, lanes, output, gpu, max_sdk_step)
            result['style'] = archive.style
            completed_step = result['sdk_step']
            return result

        def read_sim():
            return sim_reference() if sim_reference is not None else None

        def read_clock():
            simulation = None
            try:
                simulation = read_sim()
                return clock(simulation) if simulation is not None else None
            finally:
                simulation = None
        _nativ_sessi_import__SourceFreezeAudit = SourceFreezeAudit
        source_audit = _nativ_sessi_import__SourceFreezeAudit(project=project, attempt=output.parent / 'attempt.json', output=output) if runtime_source_audit else None
        timeline_owner = _nativ_sessi_import__TimelineStopOwner(acquire=omni.timeline.get_timeline_interface, freeze=freeze, read_clock=read_clock, output=output)

        def stop():
            write_new(output / 'stop_commit.json', timeline_owner.stop())

        def release_before_close():
            errors = []
            callbacks = ((source_audit.before_close,) if source_audit is not None else ()) + (timeline_owner.before_close,)
            for callback in callbacks:
                try:
                    if callback() is not True:
                        errors.append('runtime pre-close audit returned unknown')
                except BaseException as error:
                    errors.append(str(error))
                finally:
                    callback = None
            if errors:
                raise RuntimeError('; '.join(errors)) from None
            return True
        entered = True
        return _nativ_sessi_import__run_lifecycle_with_stat_cache(SimpleNamespace(update=app.update, close=lambda: app.close(wait_for_replicator=False)), lifecycle=_nativ_sessi_import__run_lifecycle, cache_release=cache, read_sim=read_sim, read_completed_sdk_step=lambda : completed_step, max_sdk_step=max_sdk_step, make_sim=make_sim, acquire=acquire, synchronize=lambda : torch.cuda.synchronize(0), clear_sim=lambda sim: _nativ_sessi_import__clear_known_simulation_instance(sim, contract=clear_contract), stop=stop, manager_is_none=lambda : SimulationManager.get_physics_sim_view() is None, freeze=freeze, output=output, cleanup='explicit', owned_names=tuple(owned_names), trace_factory=_nativ_sessi_import__StartupTrace, before_close_guard=release_before_close)
    finally:
        if not entered:
            app.close(wait_for_replicator=False)

def run_lifecycle_with_stat_cache(app, *, lifecycle, cache_release, read_sim, read_completed_sdk_step, max_sdk_step, **kwargs):
    """Release the source-bound cache before owned GC on acquisition exit.

    Earlier make_sim/watch failures do not enter run_owned_call and remain
    failures of the original lifecycle. No native release or exit is inferred.
    """
    _runti_lifec_import__run_owned_call = run_owned_call
    if type(max_sdk_step) is not int or max_sdk_step < 2:
        raise ValueError('frozen worker SDK bound required')
    original = _runti_lifec_import__run_owned_call
    called = False

    def before_collect():
        nonlocal called
        if called:
            raise RuntimeError('worker stat-cache release is once only')
        called = True
        simulation = None
        try:
            simulation = read_sim()
            if simulation is None:
                raise ValueError('live worker SDK clock owner required')
            step = simulation.current_time_step_index
            end = read_completed_sdk_step()
            if type(step) is not int or not 2 <= step <= max_sdk_step or (end is not None and (type(end) is not int or end != step)):
                raise ValueError('actual SDK clock differs from frozen/completed worker budget')
            cache_release(simulation, expected_sdk_step=step)
        finally:
            simulation = None

    def run_owned(operation, **options):
        if 'before_collect' in options:
            raise ValueError('worker owns the source-bound pre-GC callback')
        return original(operation, before_collect=before_collect, **options)
    return lifecycle(app, owned_call=run_owned, **kwargs)
stat_cache_relea__VERSION = 'dh116-kit-stat-cache-release-v1'

class NativeOwnershipTrace:
    """Record Python liveness separately from SDK release and process exit."""

    def __init__(self, path):
        self.path = path
        self.references = {}
        path.touch(exist_ok=False)

    def event(self, phase, **details):
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(nativ_owner__json.dumps(dict(phase=phase, time_unix_s=time.time(), **details), allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def watch(self, name, obj, *, owner, release, gpu_view=False, callbacks=False, composer=False):
        if name in self.references:
            raise ValueError(f'duplicate native resource name: {name}')
        try:
            self.references[name] = nativ_owner__weakref.ref(obj)
        except TypeError:
            self.references[name] = None
        self.event('acquire', resource=name, object_type=type(obj).__name__, owner=owner, release=release, gpu_view=gpu_view, callbacks=callbacks, composer=composer, python_weakref_available=self.references[name] is not None)

    def checkpoint(self, phase):
        self.event(phase, python_liveness={name: ref() is not None if ref is not None else None for (name, ref) in self.references.items()})
import gc as runti_lifec__gc
MODULE = 'omni.ext._impl.stat_cache'

def nativ_owner___type_name(value):
    kind = type(value)
    return kind.__module__ + '.' + kind.__qualname__

def run_lifecycle(app, *, make_sim, acquire, synchronize, clear_sim, stop, manager_is_none, freeze, output, cleanup, stage_observer=None, before_close_guard=None, owned_names=(), trace_factory=None, owned_call=None):
    """Record every cleanup boundary and close once, then rethrow text-only faults.

    CPU callback success grants no native, grasp, process-exit or Teacher proof.
    Bindings on previous are read at call time for existing scoped callers.
    """
    _runti_lifec_import__batch_snapshot = batch_snapshot
    _runti_lifec_import__StartupTrace = StartupTrace
    _runti_lifec_import__run_owned_call = run_owned_call
    from dh116.env.contracts import clock
    from dh116.env.contracts import write_new
    if trace_factory is None:
        trace_factory = _runti_lifec_import__StartupTrace
    if owned_call is None:
        owned_call = _runti_lifec_import__run_owned_call
    if not owned_names:
        raise ValueError('explicit current owned resource names required')
    if cleanup != 'explicit':
        raise ValueError('unsupported cleanup case')
    output = runti_lifec__Path(output)
    sim = sim_ref = trace = None
    (failures, clocks, stages) = ([], [], {})
    acquired = {}

    def notify(name, event, error=None):
        if stage_observer is None:
            return
        try:
            stage_observer(dict(stage=name, event=event, error=error))
        except BaseException as sink_error:
            text = runti_lifec__traceback.format_exc()
            failures.append(text)
            stages['diagnostic:' + name + ':' + event] = dict(attempted=True, returned=False, error=text)
            runti_lifec__traceback.clear_frames(sink_error.__traceback__)
            sink_error.__traceback__ = None

    def attempt(name, callback):
        record = dict(attempted=True, returned=False, error=None)
        stages[name] = record
        notify(name, 'start')
        try:
            value = callback()
            record['returned'] = True
            notify(name, 'returned')
            return value
        except BaseException as error:
            text = runti_lifec__traceback.format_exc()
            record['error'] = text
            failures.append(text)
            notify(name, 'error', text)
            runti_lifec__traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
            return None
        finally:
            callback = None

    def skip(name, reason):
        stages[name] = dict(attempted=False, returned=False, error=None, skipped_reason=reason)
        notify(name, 'skipped')

    def frozen():
        if freeze() is not False:
            raise RuntimeError('Kit physics freeze did not read back false')
        return True

    def observe(phase):
        target = sim_ref() if sim_ref is not None else None
        try:
            clocks.append(dict(phase=phase, clock_available=target is not None, **clock(target) if target is not None else {}))
        finally:
            target = None
        trace.checkpoint(phase)

    def acquire_owned():
        nonlocal sim, sim_ref, acquired
        if trace is None:
            raise RuntimeError('ownership trace unavailable')
        sim = make_sim()
        sim_ref = runti_lifec__weakref.ref(sim)
        trace.watch('simulation', sim, owner=runti_lifec__VERSION, release='clear local alias then STOP', callbacks=True)
        received = owned_call(lambda owners: acquire(sim, owners, trace, cleanup), synchronize=synchronize, ownership=trace)
        if not isinstance(received, dict):
            raise ValueError('owned acquisition dictionary required')
        acquired = received
        if acquired.get('failure'):
            raise RuntimeError(str(acquired['failure']))

    def drop_alias():
        nonlocal sim
        sim = None
        runti_lifec__gc.collect()

    def ownership_readback():
        resources = _runti_lifec_import__batch_snapshot(trace.references)['resources']
        names = tuple(dict.fromkeys((*owned_names, 'simulation')))
        return bool(names) and all((isinstance(resources.get(name), dict) and resources[name].get('alive') is False for name in names))

    def checked(name, callback, message):
        verified = attempt(name, callback) is True
        if not verified:

            def reject():
                raise RuntimeError(message)
            attempt(name + '_guard', reject)
        else:
            skip(name + '_guard', 'readback verified')
        return verified
    trace = attempt('trace', lambda : trace_factory(output / 'ownership.jsonl'))
    attempt('acquire', acquire_owned)
    initial_frozen = attempt('freeze_before_stop', frozen) is True
    attempt('observe_before_stop', lambda : observe('before_stop'))
    if sim is not None:
        attempt('clear', lambda : clear_sim(sim))
    else:
        skip('clear', 'simulation was not created')
    attempt('drop_alias', drop_alias)
    attempt('stop', stop)
    after_stop_frozen = attempt('freeze_after_stop', frozen) is True
    attempt('observe_after_stop', lambda : observe('after_stop'))
    manager_released = checked('manager_readback', manager_is_none, 'manager view release false or unknown')
    released = checked('ownership_readback', ownership_readback, 'owned resource release false or unknown')
    attempt('stop_event', lambda : trace.event('SDK STOP', manager_view_released=manager_released))

    def cleanup_has_errors():
        return any((row['error'] for (name, row) in stages.items() if name not in ('acquire',)))
    safe_to_update = stages['clear']['returned'] and stages['drop_alias']['returned'] and stages['stop']['returned'] and initial_frozen and after_stop_frozen and manager_released and released and (not cleanup_has_errors())
    if safe_to_update:
        attempt('post_stop_update', app.update)
    else:
        skip('post_stop_update', 'clear/STOP/freeze/manager/ownership not verified')
    manager_at_close = checked('manager_before_close', manager_is_none, 'manager view release false or unknown before close')
    released_at_close = checked('ownership_before_close', ownership_readback, 'owned resource release false or unknown before close')
    close_frozen = attempt('freeze_before_close', frozen) is True
    attempt('observe_before_close', lambda : observe('before_close_check'))
    runtime_released = checked('runtime_before_close', before_close_guard, 'pre-close runtime release false or unknown') if before_close_guard is not None else True
    normal = bool(runtime_released and safe_to_update and stages['post_stop_update']['returned'] and manager_at_close and released_at_close and close_frozen and (not cleanup_has_errors()))
    mode = 'normal' if normal else 'fault'
    before_name = 'before_app_close.json' if normal else 'fault_before_app_close.json'
    returned_name = 'app_close_returned.json' if normal else 'fault_app_close_returned.json'
    before = dict(case=cleanup, clocks=clocks, close_mode=mode, lifetime_released=released_at_close, play_simulations=False if close_frozen else None, process_exit_verified=False, qualification_granted=False)
    attempt('before_close_marker', lambda : write_new(output / before_name, before))
    attempt('close', app.close)
    if stages['close']['returned']:
        attempt('close_returned_marker', lambda : write_new(output / returned_name, dict(case=cleanup, app_close_returned=True, close_mode=mode, process_exit_verified=False, qualification_granted=False)))
    else:
        skip('close_returned_marker', 'app.close did not return')
    attempt('observe_after_close', lambda : observe('after_app_close'))
    if not normal and (not failures):
        failures.append('normal close admission failed')
    steps = [row['sdk_step'] for row in clocks if row.get('clock_available')]
    result = dict(version='dh116-shared-native-session-v1', cleanup_version=runti_lifec__VERSION, case=cleanup, acquisition=acquired, clocks=clocks, close_mode=mode, physics_steps=max(steps) if steps else None, measurement_steps=acquired.get('physics_rows'), rod_steps=acquired.get('physics_rows'), code_config_sha256=acquired.get('code_config_sha256'), physics_clock=acquired.get('physics_clock'), physics_budget=acquired.get('physics_budget'), ppo_updates=0, play_simulations=False if close_frozen else None, lifetime_released_before_close=released_at_close, failure='\n'.join(failures) or None, process_exit_verified=False, native_qualification=False, grasp_qualified=False, teacher_admission=False)
    ledger = dict(version=runti_lifec__VERSION, close_mode=mode, stages=dict(stages), qualification_granted=False, failure=result['failure'])
    attempt('write_cleanup_stages', lambda : write_new(output / 'cleanup_stages.json', ledger))
    result['failure'] = '\n'.join(failures) or None
    attempt('write_result', lambda : write_new(output / 'result.json', result))
    if failures:
        raise RuntimeError('\n'.join(failures)) from None
    return result
BASE = stat_cache_relea__Path('/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/omni/kernel/py/omni/ext/_impl')

def nativ_owner___describe_referrer(value, target):
    result = {'object_id': id(value), 'type': nativ_owner___type_name(value)}
    if type(value) is dict:
        keys = []
        for (index, (key, item)) in enumerate(value.items()):
            if index >= 128:
                break
            if item is target and type(key) is str and (len(keys) < 8):
                keys.append(key[:160])
        result.update(kind='dict', matching_keys=keys, keys_scan_truncated=len(value) > 128)
    elif type(value) is nativ_owner__types.CellType:
        result['kind'] = 'closure_cell'
    elif type(value) is nativ_owner__types.FrameType:
        result.update(kind='frame', function=value.f_code.co_name, filename=value.f_code.co_filename, line=value.f_lineno)
    elif type(value) is nativ_owner__types.FunctionType:
        result.update(kind='function', function=value.__qualname__, filename=value.__code__.co_filename, line=value.__code__.co_firstlineno)
    elif type(value) is nativ_owner__types.MethodType:
        result.update(kind='bound_method', function=value.__func__.__qualname__)
    else:
        result['kind'] = 'other'
    return result
SOURCES = {'cache': (BASE / 'stat_cache.py', 'd09905d34ad52577c95f997d1f69fa152b3b3d71dc7e8570e777842ef064cd6b'), 'settings': (BASE / 'ext_settings.py', '9ae0326d15a7573db685a1b6bed9d22d4a31d9008c2743e6bfacecb766f37fd4'), 'startup': (BASE / '_internal.py', '6dfde3b8a85cbb33746e0c506e5a9ba5cd2a7ad90caeb568098c274c14126283')}

def snapshot_weakrefs(references, *, max_referrers=16):
    """Return scalar observations without retaining referents in the result.

    Call at an already authorized lifecycle boundary. This function does not
    collect or release anything; it must not be treated as a teardown fix.
    """
    if not 1 <= max_referrers <= 64 or not 1 <= len(references) <= 32:
        raise ValueError('diagnostic bounds require 1..32 owners and 1..64 referrers')
    if any((type(name) is not str or (ref is not None and type(ref) is not nativ_owner__weakref.ReferenceType) for (name, ref) in references.items())):
        raise TypeError('only named weak references or explicit unsupported None are accepted')
    result = {'version': nativ_owner__VERSION, 'native_release_verified': False, 'resources': {}}
    for (name, ref) in references.items():
        if ref is None:
            result['resources'][name] = {'alive': None, 'reason': 'weakref_unsupported'}
            continue
        target = ref()
        if target is None:
            result['resources'][name] = {'alive': False}
            continue
        referrers = nativ_owner__gc.get_referrers(target)
        summaries = []
        for index in range(min(len(referrers), max_referrers)):
            summaries.append(nativ_owner___describe_referrer(referrers[index], target))
        result['resources'][name] = {'alive': True, 'object_id': id(target), 'type': nativ_owner___type_name(target), 'referrer_count': len(referrers), 'referrers_truncated': len(referrers) > max_referrers, 'referrers': summaries}
        del referrers, target
    return result
FUNCTIONS = {'_os_stat_cached': 11, 'refresh_stat_cache': 26, 'reset_stat_cache': 43}

class NativeOwners:

    def __init__(self):
        self.resources = {}

    def keep(self, name, value):
        if name in self.resources:
            raise ValueError('duplicate native owner')
        self.resources[name] = value
        return value

def stat_cache_source_contract():
    sources = {}
    for (name, (path, expected)) in SOURCES.items():
        actual = stat_cache_relea__hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('installed stat cache source differs: ' + name)
        sources[name] = {'path': str(path.resolve(strict=True)), 'sha256': actual}
    return {'version': stat_cache_relea__VERSION, 'module': MODULE, 'sources': sources, 'functions': dict(FUNCTIONS), 'max_reset_calls': 1, 'app_updates_added': 0, 'native_release_verified': False}

def run_owned_call(operation, *, synchronize, ownership, before_collect=None):
    """Release solver/view/composer owners even when an SDK frame raises.

    The completed failing frame is explicitly cleared before STOP/App close;
    propagating its original traceback would keep closure/SDK owners alive.
    """
    import gc
    import traceback
    owners = NativeOwners()
    result = None
    failure = None
    failed_tb = None
    failed_exception = None
    cleanup_errors = []
    hook_interrupt = None
    try:
        result = operation(owners)
    except BaseException as error:
        failure = traceback.format_exc()
        (failed_tb, failed_exception) = (error.__traceback__, error)
    finally:
        try:
            if 'composer' in owners.resources:
                owners.resources['composer'].reset()
        except BaseException:
            cleanup_errors.append(traceback.format_exc())
        try:
            synchronize()
        except BaseException:
            cleanup_errors.append(traceback.format_exc())
        if failed_tb is not None:
            traceback.clear_frames(failed_tb)
            failed_exception.__traceback__ = None
            failed_tb = failed_exception = None
        for name in ('provider', 'runtime', 'view', 'composer', 'hand'):
            owners.resources.pop(name, None)
        owners.resources.clear()
        if before_collect is not None:
            try:
                before_collect()
            except (KeyboardInterrupt, SystemExit) as error:
                error.__traceback__ = None
                hook_interrupt = error
            except Exception as error:
                cleanup_errors.append(traceback.format_exc())
                error.__traceback__ = None
        gc.collect()
        ownership.checkpoint('owned hand and solver references dropped')
        if hook_interrupt is not None:
            raise hook_interrupt from None
    if failure or cleanup_errors:
        raise RuntimeError('\n'.join(([failure] if failure else []) + cleanup_errors)) from None
    return result

def _source_codes(path):
    compiled = compile(stat_cache_relea__Path(path).read_bytes(), str(path), 'exec', dont_inherit=True, optimize=stat_cache_relea__sys.flags.optimize)
    return {code.co_name: code for code in compiled.co_consts if type(code) is stat_cache_relea__types.CodeType}

class LifetimeTrace(NativeOwnershipTrace):

    def checkpoint(self, phase):
        super().checkpoint(phase)
        name = phase.replace(' ', '_').replace(',', '')
        write_json(self.path.parent / (name + '.refs.json'), batch_snapshot(self.references))
        with (self.path.parent / (name + '.maps')).open('x') as stream:
            stream.write(nativ_owner__Path('/proc/self/maps').read_text())
            stream.flush()
            os.fsync(stream.fileno())

def _validate_function(fn, fields, path, expected, *, defaults=None):
    if type(fn) is not stat_cache_relea__types.FunctionType or fn.__globals__ is not fields or stat_cache_relea__Path(fn.__code__.co_filename).resolve() != stat_cache_relea__Path(path) or (fn.__code__ != expected) or (fn.__defaults__ != defaults) or (fn.__kwdefaults__ is not None):
        raise ValueError('loaded stat cache function source/body differs: ' + expected.co_name)

def assert_clean_import_boundary():
    loaded = sorted((name for name in nativ_owner__sys.modules if name.split('.')[0] in ('pxr', 'PhysxSchema')))
    if loaded:
        raise RuntimeError('USD modules loaded before AppLauncher: ' + ', '.join(loaded))
    return dict(version=BOUNDARY_VERSION, phase='before_isaaclab_app_import', loaded_usd_modules=[], native_execution_performed=False)

def _validated_fields(module, contract):
    if contract != stat_cache_source_contract():
        raise ValueError('stat cache source contract differs')
    if type(module) is not stat_cache_relea__types.ModuleType or module.__name__ != MODULE or stat_cache_relea__sys.modules.get(MODULE) is not module:
        raise ValueError('loaded stat cache module identity differs')
    fields = vars(module)
    path = contract['sources']['cache']['path']
    if stat_cache_relea__Path(fields.get('__file__', '')).resolve() != stat_cache_relea__Path(path):
        raise ValueError('loaded stat cache module source differs')
    codes = _source_codes(path)
    for name in FUNCTIONS:
        _validate_function(fields.get(name), fields, path, codes[name])
    if importlib._bootstrap_external._path_stat is not fields['_os_stat_cached']:
        raise ValueError('installed stat cache hook is not active; refusing to enable it')
    settings_name = 'omni.ext._impl.ext_settings'
    settings = stat_cache_relea__sys.modules.get(settings_name)
    settings_path = contract['sources']['settings']['path']
    if type(settings) is not stat_cache_relea__types.ModuleType or settings.__name__ != settings_name or stat_cache_relea__Path(vars(settings).get('__file__', '')).resolve() != stat_cache_relea__Path(settings_path):
        raise ValueError('loaded settings source identity differs')
    settings_fields = vars(settings)
    settings_codes = _source_codes(settings_path)
    _validate_function(settings_fields.get('_get_setting'), settings_fields, settings_path, settings_codes['_get_setting'], defaults=(None,))
    wrapper_type = type(functools.lru_cache()(stat_cache_source_contract))
    for name in ('_is_reentrant_test_mode', '_is_path_stat_cache_enabled'):
        predicate = settings_fields.get(name)
        if type(predicate) is not wrapper_type:
            raise ValueError('source settings predicate wrapper required')
        _validate_function(predicate.__wrapped__, settings_fields, settings_path, settings_codes[name])
    predicate = settings_fields['_is_path_stat_cache_enabled']
    if fields.get('_is_path_stat_cache_enabled') is not predicate:
        raise ValueError('source settings predicate identity differs')
    if predicate() is not True:
        raise ValueError('stat cache must already be enabled')
    if type(fields.get('_stat_cache')) is not dict:
        raise ValueError('ordinary source stat cache dictionary required')
    return fields

def snapshot_for_startup(references):
    if not references:
        return dict(version=TRACE_VERSION, resources={}, native_release_verified=False, observation='no_registered_owners')
    return snapshot_weakrefs(references)

def _exception_record(error):
    tb = frame = current = None
    try:
        tb = error.__traceback__
        records = []
        seen = set()
        tb_count = 0
        truncated = False
        while tb is not None and tb_count < 32:
            tb_count += 1
            frame = tb.tb_frame
            current = frame
            depth = 0
            while current is not None and depth < 32:
                identity = id(current)
                if identity in seen:
                    break
                if len(records) >= 128:
                    truncated = True
                    break
                seen.add(identity)
                records.append({'object_id': identity, 'function': current.f_code.co_name, 'filename': current.f_code.co_filename, 'line': current.f_lineno, 'f_back_id': id(current.f_back) if current.f_back is not None else None})
                current = current.f_back
                depth += 1
            if current is not None and depth >= 32:
                truncated = True
            tb = tb.tb_next
        if tb is not None:
            truncated = True
        related = any((r['filename'].endswith('/torch/cuda/__init__.py') and r['function'] in ('_lazy_init', '_register_triton_kernels') or r['filename'].endswith('/isaaclab/sim/simulation_context.py') or r['filename'].endswith('[force-diagnostic-only]') for r in records))
        return {'exception_id': id(error), 'exception_type': type(error).__name__, 'traceback_nodes_examined': tb_count, 'frame_chain': records, 'frame_descriptions_truncated': truncated, 'runtime_related': related}
    finally:
        error = tb = frame = current = None

class StartupTrace(LifetimeTrace):

    def checkpoint(self, phase):
        if self.references:
            return super().checkpoint(phase)
        NativeOwnershipTrace.checkpoint(self, phase)
        name = phase.replace(' ', '_').replace(',', '')
        write_json(self.path.parent / (name + '.refs.json'), snapshot_for_startup({}))
        with (self.path.parent / (name + '.maps')).open('x') as stream:
            stream.write(nativ_owner__Path('/proc/self/maps').read_text())
            stream.flush()
            os.fsync(stream.fileno())

def snapshot_stat_cache(module, *, contract, max_entries=4096, max_records=16):
    if type(max_entries) is not int or not 1 <= max_entries <= 4096:
        raise ValueError('cache scan budget must be 1..4096')
    if type(max_records) is not int or not 1 <= max_records <= 16:
        raise ValueError('cache record budget must be 1..16')
    fields = cache = key = value = error = None
    try:
        fields = _validated_fields(module, contract)
        cache = fields['_stat_cache']
        primary = []
        other = []
        examined = exceptions = related = unsupported = 0
        for key in reversed(cache):
            if examined >= max_entries:
                break
            examined += 1
            value = cache[key]
            if type(value) is not tuple or len(value) != 2:
                unsupported += 1
                continue
            error = value[1]
            if error is None:
                continue
            exceptions += 1
            if not isinstance(error, BaseException) or type(error).__module__ != 'builtins':
                unsupported += 1
                continue
            record = _exception_record(error)
            record['key_sha256'] = stat_cache_relea__hashlib.sha256(key.encode('utf-8', 'surrogatepass') if type(key) is str else key).hexdigest() if type(key) in (str, bytes) else None
            if record['runtime_related']:
                related += 1
                if len(primary) < max_records:
                    primary.append(record)
            elif len(other) < max_records:
                other.append(record)
        records = primary + other[:max_records - len(primary)]
        epoch = fields.get('_stat_cache_update_number')
        return {'version': stat_cache_relea__VERSION, 'source_validated': True, 'active_patch_id': id(fields['_os_stat_cached']), 'cache_entries_total': len(cache), 'entries_scanned': examined, 'entries_scan_truncated': len(cache) > examined, 'cached_exceptions': exceptions, 'runtime_related_entries': related, 'exception_records': records, 'exception_records_truncated': exceptions > len(records), 'unsupported_entries': unsupported, 'cache_epoch': epoch if type(epoch) is int else None, 'native_release_verified': False, 'root_cause_proven': False}
    finally:
        module = fields = cache = key = value = error = None

def batch_snapshot(references):
    """Apply the existing lifetime observer in bounded chunks with no sampling."""
    items = list(references.items())
    if len(items) > 8 * 128 + 1:
        raise ValueError('resource witnesses exceed the bounded 128-lane probe')
    summary = snapshot_for_startup({})
    for first in range(0, len(items), 32):
        snapshot = snapshot_for_startup(dict(items[first:first + 32]))
        summary['resources'].update(snapshot['resources'])
    if items:
        summary.pop('observation', None)
    return summary

def _clock(reader, expected_sdk_step):
    row = reader()
    if type(row) is not dict or set(row) != {'sdk_step', 'sdk_time_s', 'app_update_number'}:
        raise ValueError('complete SDK/App clock required')
    (step, t, update) = (row['sdk_step'], row['sdk_time_s'], row['app_update_number'])
    if type(expected_sdk_step) is not int or expected_sdk_step < 2 or type(step) is not int or (step != expected_sdk_step) or (type(t) not in (float, int)) or (not stat_cache_relea__math.isfinite(t)) or (not stat_cache_relea__math.isclose(t, expected_sdk_step / 480, rel_tol=0, abs_tol=2e-06)) or (type(update) is not int) or (update < 0):
        raise ValueError(f'SDK step {expected_sdk_step} and valid non-sentinel App clock required')
    return dict(row)
simul_owner_clean___describe_referrer = nativ_owner___describe_referrer
simul_owner_clean___type_name = nativ_owner___type_name

def verify_stat_cache_empty(module, *, contract, clock_reader, expected_clock, expected_sdk_step=2):
    """Read-only final guard; never reset twice, write evidence or advance App."""
    fields = None
    try:
        fields = _validated_fields(module, contract)
        expected = _clock(lambda : expected_clock, expected_sdk_step)
        before = _clock(clock_reader, expected_sdk_step)
        snapshot = snapshot_stat_cache(module, contract=contract)
        after = _clock(clock_reader, expected_sdk_step)
        if before != expected or after != expected:
            raise ValueError('clock changed after stat cache evidence IO')
        if type(fields.get('_stat_cache')) is not dict or len(fields['_stat_cache']) != 0:
            raise ValueError('stat cache repopulated after evidence IO; no reset retry')
        return snapshot
    finally:
        module = fields = clock_reader = None
simul_owner_clean__VERSION = 'dh116-simulation-owner-cleanup-cpu-v1'

def release_owned_stat_cache(module, *, contract, clock_reader, observer=None, expected_sdk_step=2):
    fields = None
    io_errors = []

    def emit(phase, row):
        if observer is None:
            return
        try:
            observer(phase, row)
        except Exception as error:
            io_errors.append(type(error).__name__ + ': ' + str(error))
            error.__traceback__ = None
    try:
        fields = _validated_fields(module, contract)
        before = snapshot_stat_cache(module, contract=contract)
        clock_before = _clock(clock_reader, expected_sdk_step)
        try:
            emit('before', dict(before, clock=clock_before))
        finally:
            fields['reset_stat_cache']()
        clock_after = _clock(clock_reader, expected_sdk_step)
        if clock_after != clock_before:
            raise ValueError('clock changed during stat cache API')
        after = snapshot_stat_cache(module, contract=contract)
        if after['cache_entries_total'] != 0:
            raise ValueError('stat cache reset did not leave an empty cache')
        receipt = {'version': stat_cache_relea__VERSION, 'source_validated': True, 'reset_api_calls': 1, 'reset_api_returned': True, 'before': before, 'after': after, 'clock_before': clock_before, 'clock_after': clock_after, 'patch_preserved': before['active_patch_id'] == after['active_patch_id'], 'native_release_verified': False, 'root_cause_proven': False}
        emit('after', dict(after, clock=clock_after, reset_api_calls=1))
        if io_errors:
            raise RuntimeError('stat cache evidence IO failed: ' + '; '.join(io_errors))
        receipt['post_observer_check'] = verify_stat_cache_empty(module, contract=contract, clock_reader=clock_reader, expected_clock=clock_before, expected_sdk_step=expected_sdk_step)
        return receipt
    finally:
        module = fields = clock_reader = observer = None
CORE = simul_owner_clean__Path('/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py')
import sys as stat_cache_relea__sys, weakref as stat_cache_relea__weakref
LAB = simul_owner_clean__Path('/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py')
from dh116.env.contracts import clock as stat_cache_relea__clock
EVENTS = simul_owner_clean__Path('/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/omni/kernel/py/carb/events/_events.pyi')
from dh116.env.contracts import write_new
HASHES = dict(core='0be3b0eaa44e6110cec520be9ac2c08cea17ba97db14b1b0aa60d87824deceb3', lab='d95f2eaa138b4fff845ac72439caa2eca5deab4f0d5609e06f163d20eef4df93', events='5b65d0b62734824f532de87aca92829a8f5d025b2f02d33378335e62afcc501f')

def make_before_collect(plan):
    output = stat_cache_relea__Path(plan['output'])
    contract = plan['stat_cache_release']['source_contract']
    if plan['stat_cache_release']['max_reset_calls'] != 1:
        raise ValueError('one reset budget required')
    called = False

    def before_collect(simulation, *, expected_sdk_step=2):
        nonlocal called
        if called:
            raise RuntimeError('stat cache reset is once only')
        called = True
        reference = None
        module = None
        try:
            reference = stat_cache_relea__weakref.ref(simulation)
            simulation = None
            module = stat_cache_relea__sys.modules.get(MODULE)
            if module is None:
                raise ValueError('already loaded SDK cache module required')
            app_module = stat_cache_relea__sys.modules.get('omni.kit.app')
            if app_module is None:
                raise ValueError('already loaded Kit app module required')

            def read_clock():
                target = None
                try:
                    target = reference()
                    if target is None:
                        raise ValueError('live SDK clock owner unavailable')
                    row = stat_cache_relea__clock(target)
                    row['app_update_number'] = app_module.get_app().get_update_number()
                    return row
                finally:
                    target = None

            def observe(phase, row):
                write_new(output / ('stat_cache_' + phase + '.json'), row)
            receipt = release_owned_stat_cache(module, contract=contract, clock_reader=read_clock, observer=observe, expected_sdk_step=expected_sdk_step)
            receipt['final_post_write_guard_required'] = True
            receipt['final_post_write_guard_attested_by_this_file'] = False
            write_new(output / 'stat_cache_release.json', receipt)
            verify_stat_cache_empty(module, contract=contract, clock_reader=read_clock, expected_clock=receipt['clock_before'], expected_sdk_step=expected_sdk_step)
        except Exception as error:
            message = type(error).__name__ + ': ' + str(error)
            error.__traceback__ = None
            try:
                write_new(output / 'stat_cache_error.json', {'error': message, 'native_release_verified': False, 'reset_retried': False})
            except Exception:
                pass
            raise RuntimeError(message) from None
        finally:
            simulation = reference = module = None
    return before_collect

def simulation_source_contract(*, core_source=CORE, lab_source=LAB, events_source=EVENTS):
    sources = {}
    for (name, path) in (('core', core_source), ('lab', lab_source), ('events', events_source)):
        path = simul_owner_clean__Path(path).resolve(strict=True)
        sha = simul_owner_clean__hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != HASHES[name]:
            raise ValueError(name + ' installed source differs')
        sources[name] = dict(path=str(path), sha256=sha)
    return dict(version=simul_owner_clean__VERSION, sources=sources, subscription_field='_on_post_physics_ready_callback', bound_method='_on_post_physics_ready', original_clear_omits_ready_subscription=True, constructor_frame_upstream_owner_known=False, native_qualification=False)

def clear_known_simulation_instance(sim, *, contract):
    """Detach only the known ready subscription, then call original clear_instance.

    Future approved owners may use this as their clear_sim callback. Source and
    method-file guards run before mutation. A failed unsubscribe does not clear
    the subscription field; original singleton/callback cleanup still runs.
    Completion means the API calls returned, never that native release is proven.
    All callback/native aliases are dropped even when an exception is retained.
    """
    handle = callback = clear = code = None
    validated = False
    errors = []
    result = dict(version=simul_owner_clean__VERSION, subscription_status='unavailable', unsubscribe_returned=False, subscription_field_cleared=False, original_clear_instance_returned=False, extra_gc_calls=0, native_release_verified=False, simulation_liveness_after_return=None, constructor_frame_owner_resolved=False, physics_steps_executed=0)
    try:
        expected = simulation_source_contract(**{name + '_source': contract['sources'][name]['path'] for name in ('core', 'lab', 'events')})
        if contract != expected:
            raise ValueError('source contract differs')
        callback = getattr(sim, contract['bound_method'], None)
        code = getattr(getattr(callback, '__func__', None), '__code__', None)
        if code is None or simul_owner_clean__Path(code.co_filename).resolve() != simul_owner_clean__Path(contract['sources']['core']['path']):
            raise ValueError('actual SimulationContext callback source differs')
        clear = type(sim).clear_instance
        code = getattr(getattr(clear, '__func__', None), '__code__', None)
        paths = {simul_owner_clean__Path(contract['sources'][name]['path']) for name in ('core', 'lab')}
        if code is None or simul_owner_clean__Path(code.co_filename).resolve() not in paths:
            raise ValueError('actual clear_instance source differs')
        validated = True
        field = contract['subscription_field']
        if not hasattr(sim, field):
            raise ValueError('ready subscription ownership is unavailable')
        handle = getattr(sim, field)
        if handle is None:
            result.update(subscription_status='already_absent', subscription_field_cleared=True)
        else:
            unsubscribe = getattr(handle, 'unsubscribe', None)
            try:
                if not callable(unsubscribe):
                    raise ValueError('installed ISubscription.unsubscribe API unavailable')
                unsubscribe()
                result['unsubscribe_returned'] = True
                setattr(sim, field, None)
                result.update(subscription_status='explicitly_unsubscribed', subscription_field_cleared=True)
            finally:
                unsubscribe = None
    except BaseException as error:
        errors.append(type(error).__name__ + ': ' + str(error))
        simul_owner_clean__traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    try:
        if validated:
            clear()
            result['original_clear_instance_returned'] = True
    except BaseException as error:
        errors.append(type(error).__name__ + ': ' + str(error))
        simul_owner_clean__traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    finally:
        sim = handle = callback = clear = code = None
    if errors:
        raise RuntimeError('\n'.join(errors)) from None
    return result

def runtime_owner_ids(values, namespace):
    """Correlate already-owned factory dictionaries using process-local IDs only.

    IDs may be reused after object death. This is boundary evidence, not native
    release or crash attribution. No callback, container or resource is retained.
    """
    if type(values) is not dict or type(namespace) is not dict:
        raise TypeError('exact factory values and namespace dictionaries required')
    known = {}
    for name in ('sim', 'left_hands', 'right_hands', 'rope', 'rope_pos_view', 'selected_scene'):
        if name in values:
            known[name] = id(values[name])
    state = values.get('force_state')
    return dict(version=simul_owner_clean__VERSION, factory_locals_id=id(values), namespace_id=id(namespace), force_state_id=id(state) if state is not None else None, known_object_ids=known, ids_scope='single process and construction boundary; IDs may be reused after death', root_cause_proven=False, native_release_verified=False)

def _frame_callers(frame, limit):
    """Bounded scalar f_back descriptions; never materialize frame locals."""
    callers = []
    current = None
    try:
        current = frame.f_back
        for _ in range(limit):
            if current is None:
                break
            callers.append(dict(object_id=id(current), kind='frame', function=current.f_code.co_name, filename=current.f_code.co_filename, line=current.f_lineno))
            current = current.f_back
        return callers
    finally:
        current = frame = None

def _probe_cuda_frame_owners(frame, *, excluded_ids, max_referrers, max_frames=1):
    """Walk only frame referrers with a hard node budget; emit scalar records."""
    pending = parents = parent = current = None
    try:
        pending = [(frame, 0, None)]
        queued_ids = {id(frame)}
        packets = []
        duplicates = 0
        while pending and len(packets) < max_frames:
            (current, depth, previous_id) = pending.pop(0)
            parents = simul_owner_clean__gc.get_referrers(current)
            descriptions = []
            for i in range(min(len(parents), max_referrers)):
                parent = parents[i]
                if id(parent) in excluded_ids or parent is pending:
                    continue
                if type(parent) is simul_owner_clean__types.FrameType and parent.f_code.co_filename == __file__ and (parent.f_code.co_name in ('probe_simulation_owner_edges', '_probe_cuda_frame_owners')):
                    continue
                descriptions.append(simul_owner_clean___describe_referrer(parent, current))
                if type(parent) is simul_owner_clean__types.FrameType:
                    identity = id(parent)
                    if identity in queued_ids:
                        duplicates += 1
                    else:
                        queued_ids.add(identity)
                        pending.append((parent, depth + 1, id(current)))
            packets.append(dict(frame=simul_owner_clean___describe_referrer(current, None), owners=descriptions, parent_frame_id=previous_id, depth_from_cuda_frame=depth, enumerated_count=len(parents), scanned_count=min(len(parents), max_referrers), truncated=len(parents) > max_referrers, observation='python_owners_observed' if descriptions else 'no_visible_python_owner', count_scope='scanned non-diagnostic Python referrers only; native holders may be invisible', native_owner_qualified=False))
            parents = parent = current = None
        return dict(packets=packets, frames_followed=len(packets), frame_walk_truncated=bool(pending), pending_frame_candidates_at_limit=len(pending), frame_walk_duplicates_skipped=duplicates)
    finally:
        frame = pending = parents = parent = current = None

def probe_simulation_owner_edges(reference, *, max_referrers=16, max_frame_depth=4, trace_cuda_init_owners=False, cuda_frame_owner_budget=1):
    """Scalar two-hop trace with an optional bounded upstream CUDA-frame walk.

    Never reads f_locals, clears frames, collects objects or follows native
    threads. gc.get_referrers enumerates Python owners, not all C++ holders.
    The returned record retains no referents, methods, frames or native handles.
    """
    if type(reference) is not simul_owner_clean__weakref.ReferenceType:
        raise TypeError('one explicit weak reference required')
    if type(max_referrers) is not int or not 1 <= max_referrers <= 64:
        raise ValueError('max_referrers must be 1..64')
    if type(max_frame_depth) is not int or not 1 <= max_frame_depth <= 8:
        raise ValueError('max_frame_depth must be 1..8')
    if type(trace_cuda_init_owners) is not bool:
        raise ValueError('trace_cuda_init_owners must be boolean')
    if type(cuda_frame_owner_budget) is not int or not 1 <= cuda_frame_owner_budget <= 8:
        raise ValueError('cuda_frame_owner_budget must be 1..8')
    cuda_seen_ids = set()
    target = direct = owner = upstream = parent = None
    result = dict(version=simul_owner_clean__VERSION, alive=None, direct_referrers=[], selected_owner_edges=[], root_cause_proven=False, native_release_verified=False, native_thread_attribution=None, frame_locals_read=False, diagnostic_handles_retained=False, max_hops=2 + cuda_frame_owner_budget if trace_cuda_init_owners else 2, max_referrers=max_referrers, max_frame_depth=max_frame_depth, enumeration_limit='gc.get_referrers enumerates all direct Python referrers; descriptions and traversal are bounded')
    if trace_cuda_init_owners:
        result['cuda_frame_owner_trace'] = dict(max_frames=cuda_frame_owner_budget, frames_followed=0, root_frames_followed=0, frame_walk_truncated=False, pending_frame_candidates_at_limit=0, frame_walk_duplicates_skipped=0, candidate_occurrences_seen=0, duplicate_candidates_skipped=0, additional_candidates_not_followed=0, selector='_lazy_init function name', selector_is_not_source_qualification=True)
    try:
        target = reference()
        result['alive'] = target is not None
        if target is None:
            return result
        direct = simul_owner_clean__gc.get_referrers(target)
        result.update(object_id=id(target), referrer_count=len(direct), referrers_truncated=len(direct) > max_referrers)
        for i in range(min(len(direct), max_referrers)):
            owner = direct[i]
            description = simul_owner_clean___describe_referrer(owner, target)
            result['direct_referrers'].append(description)
            selected = type(owner) is simul_owner_clean__types.FrameType and owner.f_code.co_name == '__init__' or (type(owner) is simul_owner_clean__types.MethodType and owner.__func__.__name__ == '_on_post_physics_ready')
            if not selected:
                continue
            if type(owner) is simul_owner_clean__types.FrameType:
                description['caller_chain'] = _frame_callers(owner, max_frame_depth)
            upstream = simul_owner_clean__gc.get_referrers(owner)
            summaries = []
            frame_owner_packets = []
            visible = 0
            for j in range(min(len(upstream), max_referrers)):
                parent = upstream[j]
                if parent is direct:
                    continue
                if type(parent) is simul_owner_clean__types.FrameType and parent.f_code.co_filename == __file__ and (parent.f_code.co_name == 'probe_simulation_owner_edges'):
                    continue
                visible += 1
                if len(summaries) < max_referrers:
                    summaries.append(simul_owner_clean___describe_referrer(parent, owner))
                if trace_cuda_init_owners and type(parent) is simul_owner_clean__types.FrameType and (parent.f_code.co_name == '_lazy_init'):
                    stats = result['cuda_frame_owner_trace']
                    stats['candidate_occurrences_seen'] += 1
                    identity = id(parent)
                    if identity in cuda_seen_ids:
                        stats['duplicate_candidates_skipped'] += 1
                    elif stats['root_frames_followed'] >= 1:
                        stats['additional_candidates_not_followed'] += 1
                    else:
                        cuda_seen_ids.add(identity)
                        walk = _probe_cuda_frame_owners(parent, excluded_ids={id(direct), id(upstream)}, max_referrers=max_referrers, max_frames=cuda_frame_owner_budget)
                        frame_owner_packets.extend(walk['packets'])
                        for key in ('frames_followed', 'frame_walk_truncated', 'pending_frame_candidates_at_limit', 'frame_walk_duplicates_skipped'):
                            stats[key] = walk[key]
                        stats['root_frames_followed'] += 1
            result['selected_owner_edges'].append(dict(direct=description, upstream=summaries, upstream_referrer_count=visible, upstream_count_scope='scanned non-diagnostic owners', upstream_enumerated_count=len(upstream), upstream_scanned_count=min(len(upstream), max_referrers), upstream_truncated=len(upstream) > max_referrers))
            if trace_cuda_init_owners:
                result['selected_owner_edges'][-1]['upstream_frame_owners'] = frame_owner_packets
            upstream = parent = None
        return result
    finally:
        target = direct = owner = upstream = parent = None

class TensorViewOwners:
    """Weak-only observations of source-confirmed Python instance fields.

    The installed source/line/SHA map is separate evidence. Object identities and
    Python wrapper liveness never establish native backend release.
    """

    def __init__(self):
        self.identities = {}
        self.references = {}

    @staticmethod
    def _field(owner, name):
        if owner is None:
            return (dict(state='unknown', reason='owner_unavailable'), None)
        try:
            fields = type.__getattribute__(owner, '__dict__') if isinstance(owner, type) else object.__getattribute__(owner, '__dict__')
        except (AttributeError, TypeError):
            return (dict(state='unknown', reason='dictionary_unavailable'), None)
        if type(fields) not in (dict, simul_owner_clean__types.MappingProxyType):
            return (dict(state='unknown', reason='dictionary_unavailable'), None)
        if name not in fields:
            return (dict(state='unknown', reason='field_not_in_dictionary'), None)
        value = fields[name]
        if value is None:
            return (dict(state='none', field_present=True), None)
        return (dict(state='present', field_present=True, object_id=id(value), type=simul_owner_clean___type_name(value)), value)

    def _watch(self, name, row, value):
        record = dict(row)
        reference = None
        if row['state'] == 'present':
            try:
                reference = simul_owner_clean__weakref.ref(value)
                record['weakref_status'] = 'supported'
            except TypeError:
                record['weakref_status'] = 'unsupported'
        else:
            record['weakref_status'] = 'unavailable'
        self.identities[name] = record
        self.references[name] = reference

    def capture(self, runtime, manager):
        """Register known aliases once; retain only scalar records and weakrefs."""
        hand = data = view = backend = owner = None
        try:
            if self.identities:
                raise ValueError('tensor view owners already registered')
            (row, view) = self._field(manager, '_physics_sim_view')
            self._watch('manager_sim_view', row, view)
            (row, backend) = self._field(view, '_backend')
            self._watch('manager_sim_view_backend', row, backend)
            for side in ('left', 'right'):
                (row, hand) = self._field(runtime, side + '_hands')
                self._watch(side + '_asset', row, hand)
                (row, data) = self._field(hand, '_data')
                self._watch(side + '_data', row, data)
                for (origin, owner) in (('asset', hand), ('data', data)):
                    (row, view) = self._field(owner, '_physics_sim_view')
                    name = side + '_' + origin + '_sim_view'
                    self._watch(name, row, view)
                    (row, backend) = self._field(view, '_backend')
                    self._watch(name + '_backend', row, backend)
            return self.observe(manager)
        finally:
            runtime = manager = hand = data = view = backend = owner = None

    def _current_field(self, resource, field):
        reference = self.references.get(resource)
        target = None
        try:
            if reference is None:
                return dict(state='unknown', reason='weakref_unavailable')
            target = reference()
            if target is None:
                return dict(state='unknown', reason='owner_gone')
            return self._field(target, field)[0]
        finally:
            target = None

    def observe(self, manager):
        """Observe cached wrapper/backend weakrefs and current plain fields."""
        target = view = None
        try:
            result = dict(version=simul_owner_clean__VERSION, resources={}, current_fields={}, native_release_verified=False, root_cause_proven=False, observation_scope='Python dictionaries and weakrefs only; native release unknown', identity_scope='one process; IDs may be reused after object death')
            (result['manager_current'], view) = self._field(manager, '_physics_sim_view')
            result['manager_backend_current'] = self._field(view, '_backend')[0]
            for (name, record) in self.identities.items():
                row = dict(record)
                reference = self.references[name]
                if reference is None:
                    row['python_liveness'] = 'not_applicable' if row['state'] == 'none' else 'unknown'
                else:
                    target = reference()
                    row['python_liveness'] = 'alive' if target is not None else 'gone'
                    target = None
                result['resources'][name] = row
            for side in ('left', 'right'):
                result['current_fields'][side + '_asset._physics_sim_view'] = self._current_field(side + '_asset', '_physics_sim_view')
                result['current_fields'][side + '_asset._data'] = self._current_field(side + '_asset', '_data')
                result['current_fields'][side + '_data._physics_sim_view'] = self._current_field(side + '_data', '_physics_sim_view')
                for origin in ('asset', 'data'):
                    name = side + '_' + origin + '_sim_view'
                    result['current_fields'][name + '._backend'] = self._current_field(name, '_backend')
            return result
        finally:
            target = view = manager = None
