"""Explicit opt-in lifecycle repair; historical native recipes keep their identity."""
from contextlib import contextmanager
import gc
from pathlib import Path
import traceback
import weakref

from scripts import dh116_pinch_acquisition_v2 as previous
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

VERSION = "dh116-fault-safe-lifecycle-v3"


def run_lifecycle(app, *, make_sim, acquire, synchronize, clear_sim, stop,
                  manager_is_none, freeze, output, cleanup):
    """Record every cleanup boundary and close once, then rethrow text-only faults.

    CPU callback success grants no native, grasp, process-exit or Teacher proof.
    Bindings on previous are read at call time for existing scoped callers.
    """
    if cleanup != "explicit":
        raise ValueError("unsupported cleanup case")
    output = Path(output)
    sim = sim_ref = trace = None
    failures, clocks, stages = [], [], {}
    acquired = {}

    def attempt(name, callback):
        record = dict(attempted=True, returned=False, error=None)
        stages[name] = record
        try:
            value = callback()
            record["returned"] = True
            return value
        except BaseException as error:
            # No exception object or completed callback traceback may cross STOP.
            text = traceback.format_exc()
            record["error"] = text
            failures.append(text)
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
            return None
        finally:
            callback = None

    def skip(name, reason):
        stages[name] = dict(attempted=False, returned=False, error=None,
                            skipped_reason=reason)

    def frozen():
        if freeze() is not False:
            raise RuntimeError("Kit physics freeze did not read back false")
        return True

    def observe(phase):
        target = sim_ref() if sim_ref is not None else None
        try:
            clocks.append(dict(phase=phase, clock_available=target is not None,
                               **(previous.clock(target) if target is not None else {})))
        finally:
            target = None
        trace.checkpoint(phase)

    def acquire_owned():
        nonlocal sim, sim_ref, acquired
        if trace is None:
            raise RuntimeError("ownership trace unavailable")
        sim = make_sim()
        sim_ref = weakref.ref(sim)
        trace.watch("simulation", sim, owner=VERSION,
                    release="clear local alias then STOP", callbacks=True)
        received = previous.run_owned_call(
            lambda owners: acquire(sim, owners, trace, cleanup),
            synchronize=synchronize, ownership=trace)
        if not isinstance(received, dict):
            raise ValueError("owned acquisition dictionary required")
        acquired = received
        if acquired.get("failure"):
            raise RuntimeError(str(acquired["failure"]))

    def drop_alias():
        nonlocal sim
        sim = None
        gc.collect()

    def ownership_readback():
        resources = previous.snapshot_weakrefs(trace.references)["resources"]
        names = tuple(dict.fromkeys((*previous.OWNED_NAMES, "simulation")))
        return bool(names) and all(
            isinstance(resources.get(name), dict) and
            resources[name].get("alive") is False for name in names)

    def checked(name, callback, message):
        verified = attempt(name, callback) is True
        if not verified:
            def reject():
                raise RuntimeError(message)
            attempt(name + "_guard", reject)
        else:
            skip(name + "_guard", "readback verified")
        return verified

    trace = attempt("trace", lambda: previous.LifetimeTrace(output / "ownership.jsonl"))
    attempt("acquire", acquire_owned)
    initial_frozen = attempt("freeze_before_stop", frozen) is True
    attempt("observe_before_stop", lambda: observe("before_stop"))
    if sim is not None:
        attempt("clear", lambda: clear_sim(sim))
    else:
        skip("clear", "simulation was not created")
    # This boundary is independent of clear/observe/freeze success.
    attempt("drop_alias", drop_alias)
    attempt("stop", stop)
    after_stop_frozen = attempt("freeze_after_stop", frozen) is True
    attempt("observe_after_stop", lambda: observe("after_stop"))
    manager_released = checked(
        "manager_readback", manager_is_none, "manager view release false or unknown")
    released = checked(
        "ownership_readback", ownership_readback, "owned resource release false or unknown")
    attempt("stop_event", lambda: trace.event(
        "SDK STOP", manager_view_released=manager_released))

    def cleanup_has_errors():
        return any(row["error"] for name, row in stages.items()
                   if name not in ("acquire",))

    # STOP returning does not establish that clear worked or views were released.
    # In particular, a failed clear may leave dangling TensorViews: never pump Kit.
    safe_to_update = (
        stages["clear"]["returned"] and stages["drop_alias"]["returned"] and
        stages["stop"]["returned"] and initial_frozen and after_stop_frozen and
        manager_released and released and not cleanup_has_errors())
    if safe_to_update:
        attempt("post_stop_update", app.update)
    else:
        skip("post_stop_update", "clear/STOP/freeze/manager/ownership not verified")

    # An update can change manager/ownership/freeze; recheck before normal close.
    manager_at_close = checked(
        "manager_before_close", manager_is_none, "manager view release false or unknown before close")
    released_at_close = checked(
        "ownership_before_close", ownership_readback, "owned resource release false or unknown before close")
    close_frozen = attempt("freeze_before_close", frozen) is True
    attempt("observe_before_close", lambda: observe("before_close_check"))
    normal = bool(
        safe_to_update and stages["post_stop_update"]["returned"] and
        manager_at_close and released_at_close and close_frozen and not cleanup_has_errors())
    mode = "normal" if normal else "fault"
    before_name = "before_app_close.json" if normal else "fault_before_app_close.json"
    returned_name = "app_close_returned.json" if normal else "fault_app_close_returned.json"
    before = dict(
        case=cleanup, clocks=clocks, close_mode=mode,
        lifetime_released=released_at_close,
        play_simulations=False if close_frozen else None,
        process_exit_verified=False, qualification_granted=False)
    attempt("before_close_marker", lambda: previous.write_json(output / before_name, before))
    # Marker/trace failures cannot skip the one app.close() attempt.
    attempt("close", app.close)
    if stages["close"]["returned"]:
        attempt("close_returned_marker", lambda: previous.write_json(
            output / returned_name, dict(
                case=cleanup, app_close_returned=True, close_mode=mode,
                process_exit_verified=False, qualification_granted=False)))
    else:
        skip("close_returned_marker", "app.close did not return")
    attempt("observe_after_close", lambda: observe("after_app_close"))
    if not normal and not failures:
        failures.append("normal close admission failed")
    steps = [row["sdk_step"] for row in clocks if row.get("clock_available")]
    result = dict(
        version=previous.VERSION, cleanup_version=VERSION, case=cleanup,
        acquisition=acquired, clocks=clocks, close_mode=mode,
        physics_steps=max(steps) if steps else None,
        measurement_steps=acquired.get("physics_rows"), rod_steps=acquired.get("physics_rows"),
        code_config_sha256=acquired.get("code_config_sha256"),
        physics_clock=acquired.get("physics_clock"), physics_budget=acquired.get("physics_budget"),
        ppo_updates=0, play_simulations=False if close_frozen else None,
        lifetime_released_before_close=released_at_close,
        failure="\n".join(failures) or None, process_exit_verified=False,
        native_qualification=False, grasp_qualified=False, teacher_admission=False)
    # Snapshot cleanup before final writes: a file cannot certify its own write.
    # Keep each output independent, and never let an OSError replace prior faults.
    ledger = dict(version=VERSION, close_mode=mode, stages=dict(stages),
                  qualification_granted=False, failure=result["failure"])
    attempt("write_cleanup_stages", lambda: previous.write_json(
        output / "cleanup_stages.json", ledger))
    result["failure"] = "\n".join(failures) or None
    attempt("write_result", lambda: previous.write_json(output / "result.json", result))
    # Even if neither file is writable, the final exception retains every cause.
    if failures:
        raise RuntimeError("\n".join(failures)) from None
    return result


@contextmanager
def repaired_lifecycle_scope():
    """Temporarily opt an existing caller in; nested/exception paths restore it."""
    with bindings((previous, "run_lifecycle", run_lifecycle)):
        yield
