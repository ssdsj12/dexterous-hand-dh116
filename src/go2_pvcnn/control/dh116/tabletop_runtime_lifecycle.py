"""Policy-independent native shutdown for Teacher and Student runtimes.

The original owned-call/stat-cache/clear/STOP ordering is preserved. A dropped
Python Timeline wrapper is not proof that every SDK singleton/native factory
has been released, or that the outer process will exit successfully.
"""
from pathlib import Path
import traceback
import weakref

from .tabletop_preview_launcher import commit_preview_stop


def _detach_exception_frames(error, seen=None):
    """Drop completed frames throughout a chained native failure, retaining text only."""
    seen = set() if seen is None else seen
    if id(error) in seen:
        return
    seen.add(id(error))
    related = (error.__cause__, error.__context__, *getattr(error, "exceptions", ()))
    traceback.clear_frames(error.__traceback__)
    error.__traceback__ = None
    error.__cause__ = error.__context__ = None
    for nested in related:
        if nested is not None:
            _detach_exception_frames(nested, seen)


class TimelineStopOwner:
    """Own the temporary STOP wrapper only, never an articulation or CUDA view."""
    def __init__(self, *, acquire, freeze, read_clock, output):
        self._acquire, self._freeze, self._read_clock = acquire, freeze, read_clock
        self.output = Path(output)
        self._reference = None
        self._attempted = self._committed = self._release_attempted = False
        self._weakref_error = None
        self._type = None

    def stop(self):
        if self._attempted:
            raise RuntimeError("timeline STOP is once only")
        self._attempted = True
        timeline = None
        fault = None
        result = None
        try:
            timeline = self._acquire()
            self._type = type(timeline).__module__ + "." + type(timeline).__qualname__
            try:
                self._reference = weakref.ref(timeline)
            except TypeError as error:
                self._weakref_error = str(error)
            result = commit_preview_stop(timeline, freeze=self._freeze, read_clock=self._read_clock)
            self._committed = True
            if self._reference is None:
                raise RuntimeError("timeline weakref unsupported; release is unknown")
        except BaseException as error:
            fault = traceback.format_exc()
            _detach_exception_frames(error)
        finally:
            timeline = None
        # Outside the handler: Python must not automatically retain its context.
        if fault is not None:
            raise RuntimeError(fault) from None
        return result

    def before_close(self):
        if self._release_attempted:
            raise RuntimeError("timeline release check is once only")
        self._release_attempted = True
        # These callbacks belong to this run. Do not mutate SDK global owners.
        self._acquire = self._freeze = self._read_clock = None
        alive = self._reference() is not None if self._reference is not None else None
        verified = self._committed and alive is False and self._weakref_error is None
        report = dict(version="dh116-local-timeline-release-v1", stop_commit_returned=self._committed,
            local_timeline_type=self._type, local_timeline_weakref_supported=self._reference is not None,
            local_timeline_weakref_alive=alive, local_timeline_release_verified=verified,
            weakref_error=self._weakref_error, sdk_global_owners_modified=False,
            native_release_verified=False, process_exit_verified=False, qualification_granted=False)
        from scripts.diagnose_dh116_teacher_initialization import write_new
        write_new(self.output / "timeline_release.json", report)
        if not verified:
            raise RuntimeError("timeline wrapper retained" if alive else "timeline release not verified")
        return True


def run_lifecycle_with_stat_cache(app, *, lifecycle, cache_release, read_sim,
                                   read_completed_sdk_step, max_sdk_step, **kwargs):
    """Release the source-bound cache before owned GC on acquisition exit.

    Earlier make_sim/watch failures do not enter run_owned_call and remain
    failures of the original lifecycle. No native release or exit is inferred.
    """
    from scripts import dh116_pinch_acquisition_v2 as owned
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

    if type(max_sdk_step) is not int or max_sdk_step<2:
        raise ValueError("frozen worker SDK bound required")
    original=owned.run_owned_call
    called=False

    def before_collect():
        nonlocal called
        if called:
            raise RuntimeError("worker stat-cache release is once only")
        called=True
        simulation=None
        try:
            simulation=read_sim()
            if simulation is None:
                raise ValueError("live worker SDK clock owner required")
            step=simulation.current_time_step_index
            end=read_completed_sdk_step()
            if (type(step) is not int or not 2<=step<=max_sdk_step
                    or end is not None and (type(end) is not int or end!=step)):
                raise ValueError("actual SDK clock differs from frozen/completed worker budget")
            cache_release(simulation,expected_sdk_step=step)
        finally:
            simulation=None

    def run_owned(operation, **options):
        if "before_collect" in options:
            raise ValueError("worker owns the source-bound pre-GC callback")
        return original(operation,before_collect=before_collect,**options)

    with bindings((owned,"run_owned_call",run_owned)):
        return lifecycle(app,**kwargs)
