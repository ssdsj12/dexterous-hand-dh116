#!/usr/bin/env python3
"""Bounded C0–C4 lifecycle controls for the installed Isaac Sim 4.5 launcher."""

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


CASES = ("C0", "C1", "C2", "C3", "C4")
KIT_ROOT = Path("/mnt/mydisk/xk/projects/IsaacLab/apps/isaacsim_4_5")
RENDERING_KIT = KIT_ROOT / "isaaclab.python.headless.rendering.kit"
NO_CAMERA_KIT = KIT_ROOT / "isaaclab.python.headless.kit"
KIT_SHA256 = {
    RENDERING_KIT: "978bd89e90f2ca00b32805a70dd0a15c5b8b48edeb42cd042f8eb679d33fa4ea",
    NO_CAMERA_KIT: "b4a60880219827b4843f3ec3d6f9cbad8028c1c9df588a89c1796a60b8030a92",
}
INSTANCE_LOCK = Path("/tmp/dh116-isaac-single-instance.lock")
ORCHESTRATOR_ENABLED = "/exts/omni.replicator.core/Orchestrator/enabled"
AUTOCREATE_APPWINDOW = "/exts/omni.appwindow/autocreateAppWindow"
ENVIRONMENT_KEYS = (
    "DISPLAY", "WAYLAND_DISPLAY", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
    "PYTHONPATH", "CARB_APP_PATH", "ISAAC_PATH", "EXP_PATH",
)
# Retain the descriptor through interpreter teardown, even after app.close returns.
_instance_lock_fd = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def launch_options(case):
    config = {"headless": True, "fast_shutdown": False}
    if case == "C0":
        return config, {}
    experience = NO_CAMERA_KIT if case == "C3" else RENDERING_KIT
    if sha256(experience) != KIT_SHA256[experience]:
        raise ValueError(f"installed Kit SHA256 differs: {experience}")
    config["create_new_stage"] = case == "C2"
    config["extra_args"] = [f"--{ORCHESTRATOR_ENABLED}=true", "--/app/settings/persistent=false"]
    if case == "C4":
        config["extra_args"].append(f"--{AUTOCREATE_APPWINDOW}=false")
    return config, {"experience": str(experience)}


def preloaded_runtime_modules(module_names):
    roots = {"carb", "omni", "pxr", "isaacsim", "torch", "go2_pvcnn"}
    return sorted(name for name in module_names if name.split(".", 1)[0] in roots)


def mapped_runtime_libraries(lines):
    libraries = set()
    terms = ("appwindow", "carb", "physx", "omni", "kit", "libcuda", "nvidia")
    for line in lines:
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or not parts[5].startswith("/"):
            continue
        mapped = parts[5].rstrip()
        name = Path(mapped).name.lower()
        if ".so" in name and any(term in name for term in terms):
            libraries.add(mapped)
    return sorted(libraries)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _hold_instance_lock():
    import fcntl

    global _instance_lock_fd
    fd = os.open(INSTANCE_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise
    _instance_lock_fd = fd


def _environment():
    return {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _scene_snapshot():
    # Keep native wrappers local, with no object held across app.close().
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    products = [] if stage is None else [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "RenderProduct"
    ]
    viewport_module = sys.modules.get("omni.kit.viewport.utility")
    viewport = viewport_module.get_active_viewport() if viewport_module is not None else None
    path = getattr(viewport, "render_product_path", None) if viewport is not None else None
    return {
        "stage_present": stage is not None,
        "usd_render_product_paths": sorted(products),
        "default_viewport_present": viewport is not None if viewport_module is not None else None,
        "default_viewport_render_product_path": str(path) if path else None,
    }


def _kit_package():
    carb_root = os.environ.get("CARB_APP_PATH")
    path = Path(carb_root) / "PACKAGE-INFO.yaml" if carb_root else None
    if path is None or not path.is_file():
        return None
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key in {"Package", "Version", "Commit"}:
            fields[key.lower()] = value.strip()
    return {"path": str(path.resolve()), "sha256": sha256(path), **fields}


def _default_appwindow_present():
    # Read the SDK default pointer without importing or retaining a window wrapper.
    module = sys.modules.get("omni.appwindow")
    return None if module is None else module.get_default_app_window() is not None


def _native_snapshot(app, case, phase):
    import carb.settings

    config = app.config
    experience = Path(config["experience"]).resolve(strict=True)
    native_app = app.app
    manager = native_app.get_extension_manager()
    settings = carb.settings.get_settings()
    isaac_root = Path(sys.modules["isaacsim"].__file__).resolve().parent
    version_file = isaac_root / "VERSION"
    snapshot = {
        "case": case,
        "phase": phase,
        "monotonic_ns": time.monotonic_ns(),
        "headless": config["headless"],
        "effective_create_new_stage": config["create_new_stage"],
        "fast_shutdown": settings.get_as_bool("/app/fastShutdown"),
        "effective_extra_args": config["extra_args"],
        "orchestrator_enabled": settings.get_as_bool(ORCHESTRATOR_ENABLED),
        "effective_experience": str(experience),
        "effective_experience_sha256": sha256(experience),
        "isaac_sdk_version": version_file.read_text().strip() if version_file.is_file() else None,
        "kit_build_version": native_app.get_build_version(),
        "kit_kernel_version": native_app.get_kernel_version(),
        "kit_package": _kit_package(),
        "enabled_extensions": sorted(ext["id"] for ext in manager.get_extensions() if ext["enabled"]),
        "environment": _environment(),
        "mapped_runtime_libraries": mapped_runtime_libraries(Path("/proc/self/maps").read_text().splitlines()),
        **_scene_snapshot(),
    }
    if case == "C4":
        snapshot.update(
            appwindow_extension_enabled=manager.is_extension_enabled("omni.appwindow"),
            autocreate_appwindow_setting=settings.get(AUTOCREATE_APPWINDOW),
            default_appwindow_present=_default_appwindow_present(),
        )
    return snapshot


def run_lifecycle(app, case, output, *, snapshot=_native_snapshot):
    try:
        write_json(output / "before_explicit_update.json", snapshot(app, case, "before_explicit_update"))
        app.update()
        write_json(output / "before_app_close.json", snapshot(app, case, "before_app_close"))
    finally:
        app.close()
        write_json(output / "app_close_returned.json", {
            "case": case, "app_close_returned": True, "process_exit_verified": False,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args()

    config, options = launch_options(args.case)
    preloaded = preloaded_runtime_modules(sys.modules)
    args.output.mkdir(parents=False, exist_ok=False)
    write_json(args.output / "plan.json", {
        "case": args.case,
        "native_execution_requested": args.execute_native,
        "native_execution_performed": False,
        "launch_config": config,
        "launch_options": options,
        "explicit_experience_sha256": sha256(options["experience"]) if options else None,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "isaacsim_package_version": _version("isaacsim"),
        "isaacsim_kernel_package_version": _version("isaacsim-kernel"),
        "preloaded_runtime_modules": preloaded,
        "environment": _environment(),
        "explicit_application_updates": 1,
        "physics_steps": 0,
    })
    if not args.execute_native:
        return
    if preloaded:
        raise RuntimeError(f"native/project modules loaded before SimulationApp: {preloaded}")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("camera lifecycle controls require an empty display environment")

    _hold_instance_lock()
    faulthandler.enable(all_threads=True)
    # Installed extension APIs are available only after the constructor returns.
    from isaacsim import SimulationApp

    run_lifecycle(SimulationApp(config, **options), args.case, args.output)


if __name__ == "__main__":
    main()
