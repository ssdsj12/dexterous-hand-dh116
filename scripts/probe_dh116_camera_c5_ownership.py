#!/usr/bin/env python3
"""C5 diagnostic: release the existing telemetry hook pair before C4 shutdown."""

import argparse
import faulthandler
import os
from pathlib import Path
import sys

try:
    from scripts import probe_dh116_simulation_app_lifecycle as base
except ModuleNotFoundError:
    import probe_dh116_simulation_app_lifecycle as base

OWNER_MODULE = "omni.kit.app._impl.telemetry_helpers"
HOLDER_TYPE = "omni.ext._extensions.IHookHolder"


def observe_telemetry_hook():
    """Read already-loaded Python ownership only; return no native references."""
    module = sys.modules.get(OWNER_MODULE)
    value = vars(module).get("_telemetry_hook") if module is not None else None
    pair = value if type(value) in (tuple, list) else ()
    return {
        "module_present": module is not None,
        "hook_present": value is not None,
        "container_type": type(value).__name__,
        "hook_types": [type(item).__module__ + "." + type(item).__name__ for item in pair],
        "hook_ids": [id(item) for item in pair],
    }


def release_telemetry_hook(observed):
    # Do not keep the pair in a local variable or copy it into diagnostic output.
    if (not observed["module_present"] or not observed["hook_present"]
            or observed["hook_types"] != [HOLDER_TYPE, HOLDER_TYPE]):
        raise RuntimeError("C5 requires an existing telemetry IHookHolder pair")
    sys.modules[OWNER_MODULE].__dict__["_telemetry_hook"] = None
    if observe_telemetry_hook()["hook_present"]:
        raise RuntimeError("telemetry owner was not cleared")


def write_full_maps(path):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(Path("/proc/self/maps").read_text(encoding="utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def native_snapshot(app, case, phase):
    # Reuse the frozen C4 evidence fields, including actual default-window readback.
    return {**base._native_snapshot(app, "C4", phase), "case": case}


def run_lifecycle(app, output, *, snapshot=native_snapshot):
    applied = False
    try:
        write_full_maps(output / "before_explicit_update.maps")
        base.write_json(output / "before_explicit_update.json",
                        snapshot(app, "C5", "before_explicit_update"))
        app.update()
        write_full_maps(output / "before_hook_release.maps")
        observed = observe_telemetry_hook()
        base.write_json(output / "before_hook_release.json", observed)
        release_telemetry_hook(observed)
        applied = True
        write_full_maps(output / "before_app_close.maps")
        base.write_json(output / "before_app_close.json", {
            **snapshot(app, "C5", "before_app_close"),
            "owner_release_applied": applied,
            "hook_absent_before_close": not observe_telemetry_hook()["hook_present"],
        })
    finally:
        app.close()
        write_full_maps(output / "after_app_close.maps")
        after = observe_telemetry_hook()
        base.write_json(output / "app_close_returned.json", {
            "case": "C5", "app_close_returned": True, "process_exit_verified": False,
            "owner_release_applied": applied,
            "hook_absent_after_close": not after["hook_present"],
            "ownership_after_close": after,
        })
    if after["hook_present"]:
        raise RuntimeError("telemetry hook was reacquired during shutdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args()
    config, options = base.launch_options("C4")
    preloaded = base.preloaded_runtime_modules(sys.modules)
    args.output.mkdir(parents=False, exist_ok=False)
    base.write_json(args.output / "plan.json", {
        "case": "C5", "control_case": "C4",
        "ownership_variable": OWNER_MODULE + "._telemetry_hook",
        "native_execution_requested": args.execute_native,
        "native_execution_performed": False,
        "launch_config": config, "launch_options": options,
        "preloaded_runtime_modules": preloaded,
        "environment": base._environment(),
        "explicit_application_updates": 1, "physics_steps": 0, "ppo_updates": 0,
    })
    if not args.execute_native:
        return
    if os.environ.get("DH116_CAMERA_C5_GDB") != "1":
        raise RuntimeError("C5 debugger capture required; use run_dh116_camera_c5_work_packages.py")
    if preloaded:
        raise RuntimeError(f"native/project modules loaded before SimulationApp: {preloaded}")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("camera lifecycle controls require an empty display environment")
    base._hold_instance_lock()
    faulthandler.enable(all_threads=True)
    from isaacsim import SimulationApp
    run_lifecycle(SimulationApp(config, **options), args.output)


if __name__ == "__main__":
    main()
