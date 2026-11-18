#!/usr/bin/env python3
"""Paired lifecycle-only acquisition; native execution requires its dedicated worker."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import weakref

from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
from go2_pvcnn.control.dh116.tabletop_deformable_rod_bridge import NativeOwnershipTrace
from go2_pvcnn.control.dh116.tabletop_native_lifetime_audit import snapshot_weakrefs

VERSION = "dh116-physx-lifetime-probe-v1"
WORKER_ENV = "DH116_PHYSX_LIFETIME_WORKER_V1"
WRAPPER = "scripts/run_dh116_physx_lifetime_work_packages.py"
OWNED_NAMES = ("hand", "asset_sim_view", "manager_sim_view", "hand_view",
               "composer", "provider", "runtime", "rope")
CASES = ("registry", "explicit")
DT = 1 / 480


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def clock(sim):
    return {"sdk_step": int(sim.current_time_step_index), "sdk_time_s": float(sim.current_time)}


def record_reset(sim, output):
    """Two actual observed boundaries, never a fabricated per-step ledger."""
    before = {"phase": "before_reset", **clock(sim)}
    if before["sdk_step"] != 0:
        raise RuntimeError("lifetime reset must start at zero and advance exactly two steps")
    with (Path(output) / "initialization_boundaries.jsonl").open("x") as stream:
        stream.write(json.dumps(before) + "\n"); stream.flush(); os.fsync(stream.fileno())
        sim.reset()
        after = {"phase": "after_reset", **clock(sim)}
        stream.write(json.dumps(after) + "\n"); stream.flush(); os.fsync(stream.fileno())
    if after["sdk_step"] - before["sdk_step"] != 2:
        raise RuntimeError("SDK reset must advance exactly two initialization steps")
    return before, after


class LifetimeTrace(NativeOwnershipTrace):
    def checkpoint(self, phase):
        super().checkpoint(phase)
        name = phase.replace(" ", "_").replace(",", "")
        write_json(self.path.parent / (name + ".refs.json"), snapshot_weakrefs(self.references))
        with (self.path.parent / (name + ".maps")).open("x") as stream:
            stream.write(Path("/proc/self/maps").read_text())
            stream.flush(); os.fsync(stream.fileno())


def run_lifecycle(app, *, make_sim, acquire, synchronize, clear_sim, stop,
                  manager_is_none, freeze, output, cleanup):
    if cleanup not in CASES:
        raise ValueError("unsupported cleanup case")
    output = Path(output)
    trace = LifetimeTrace(output / "ownership.jsonl")
    sim = None
    sim_ref = None
    failure = []
    clocks = []
    acquired = {}
    released = False

    def observe(phase):
        target = sim_ref() if sim_ref is not None else None
        clocks.append({"phase": phase, "clock_available": target is not None,
                       **(clock(target) if target is not None else {})})
        del target
        trace.checkpoint(phase)

    try:
        sim = make_sim()
        sim_ref = weakref.ref(sim)
        trace.watch("simulation", sim, owner="lifecycle", release="clear then STOP", callbacks=True)
        acquired = run_owned_call(lambda owners: acquire(sim, owners, trace, cleanup),
                                  synchronize=synchronize, ownership=trace)
    except BaseException as error:
        failure.append(traceback.format_exc())
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    # The active exception must end before STOP and close; only text survives.
    try:
        if freeze() is not False:
            raise RuntimeError("Kit physics freeze did not read back false")
        if sim is not None:
            observe("before_stop")
            clear_sim(sim)
            sim = None
            gc.collect()
            stop()
            if freeze() is not False:
                raise RuntimeError("Kit physics resumed during STOP")
            app.update()
            observe("after_stop")
            trace.event("SDK STOP", manager_view_released=manager_is_none())
        observe("before_app_close")
        state = snapshot_weakrefs(trace.references)["resources"]
        released = all(state.get(name, {}).get("alive") is False for name in OWNED_NAMES)
    except BaseException as error:
        failure.append(traceback.format_exc())
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    try:
        if freeze() is not False:
            raise RuntimeError("Kit physics must remain frozen before close")
        write_json(output / "before_app_close.json",
                   {"case": cleanup, "clocks": clocks, "lifetime_released": released,
                    "play_simulations": False, "process_exit_verified": False})
        app.close()
        write_json(output / "app_close_returned.json",
                   {"case": cleanup, "app_close_returned": True, "process_exit_verified": False})
        observe("after_app_close")
    except BaseException as error:
        failure.append(traceback.format_exc())
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    steps = [row["sdk_step"] for row in clocks if row.get("clock_available")]
    result = {"version": VERSION, "case": cleanup, "acquisition": acquired, "clocks": clocks,
              "physics_steps": max(steps) if steps else None, "measurement_steps": 0,
              "rod_steps": 0, "ppo_updates": 0, "play_simulations": False,
              "lifetime_released_before_close": released, "failure": "\n".join(failure) or None,
              "process_exit_verified": False, "grasp_qualified": False, "teacher_admission": False}
    write_json(output / "result.json", result)
    if failure:
        raise RuntimeError("\n".join(failure)) from None
    return result


def acquire_graph(sim, owners, trace, cleanup, build):
    """Same acquisition in both cases; only explicit local deletion differs."""
    import numpy as np
    bundle = build(sim)
    for name in OWNED_NAMES:
        value = owners.keep(name, getattr(bundle, name))
        trace.watch(name, value, owner="lifetime acquisition", release="owned scope",
                    gpu_view=name in ("asset_sim_view", "manager_sim_view", "hand_view", "rope"))
    del value
    hand, view, composer = bundle.hand, bundle.hand_view, bundle.composer
    runtime, provider, rope = bundle.runtime, bundle.provider, bundle.rope
    asset_sim_view, manager_sim_view = bundle.asset_sim_view, bundle.manager_sim_view

    def host(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.array(value, dtype=float, copy=True).tolist()

    def snapshot():
        data = hand.data
        return {"q": host(data.joint_pos), "qd": host(data.joint_vel),
                "body_link_pos_w": host(data.body_link_pos_w)}

    metadata = dict(bundle.metadata, state=snapshot(), case=cleanup)
    trace.checkpoint("resources_acquired")
    if cleanup == "explicit":
        del snapshot, hand, view, composer, runtime, provider, rope
        del asset_sim_view, manager_sim_view, bundle
    return metadata


def execute_native(args, contract, cfg, project):
    from scripts.dh116_physx_lifetime_acquisition import execute
    return execute(args, contract, cfg, project)


def require_worker(args, project):
    try:
        value = json.loads(os.environ[WORKER_ENV])
    except (KeyError, ValueError):
        raise RuntimeError("use the dedicated PhysX lifetime scheduler with exact approval") from None
    expected = {"version": VERSION, "case": args.cleanup, "output": str(args.output.resolve()),
                "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "wrapper_sha256": hashlib.sha256((project / WRAPPER).read_bytes()).hexdigest()}
    if value != expected or os.environ.get("DH116_PHYSX_LIFETIME_GDB") != "1":
        raise RuntimeError("dedicated worker/capture identity differs")


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup", choices=CASES, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args()
    if args.execute_native:
        if args.output is None or not args.headless or args.device != "cuda:0":
            parser.error("native lifetime requires output, headless and cuda:0")
        require_worker(args, project)
    from scripts.verify_dh116_deformx_pinch_contact import default_arguments, preflight
    native_args = default_arguments(project)
    contract, cfg = preflight(native_args, project)
    if not args.execute_native:
        print(json.dumps({"version": VERSION, "case": args.cleanup, "physics_steps": 2,
                          "measurement_steps": 0, "rod_steps": 0, "ppo_updates": 0,
                          "native_execution_performed": False,
                          "source_acquisition_sha256": contract["code_config_sha256"]}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    args.ropesim_python = native_args.ropesim_python
    sys.path.insert(0, str(args.ropesim_python))
    execute_native(args, contract, cfg, project)


if __name__ == "__main__":
    main()
