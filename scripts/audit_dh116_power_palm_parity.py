#!/usr/bin/env python3
"""Capture and compare the original left and right Power palm cook on CPU and GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from go2_pvcnn.control.dh116.usd_surface import read_surface
from scripts import audit_dh116_cooked_hull_device_parity as parity
from scripts import verify_dh116_cooked_convex_hulls_native as probe
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings


ROOT = Path(__file__).resolve().parents[1]
LINKS = ("base_link",)
STEPS = 2
NATIVE_FILES = ("left_base_link.json", "right_base_link.json", "physics_device.json",
                "reference.json", "result.json", "simulation_context_cleared.json", "shutdown_complete.json")
write_json = probe.write_json


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_palm_source(stage, body_path):
    from pxr import Sdf, UsdGeom, UsdPhysics

    path = Sdf.Path(body_path)
    if (not path.IsAbsolutePath() or not path.IsPrimPath() or path.name != "base_link"
            or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or UsdGeom.GetStageMetersPerUnit(stage) != 1.):
        raise ValueError("absolute original metre Z-up base_link required")
    source = read_surface(stage, body_path, branch="collisions")
    mesh = stage.GetPrimAtPath(source["mesh_path"])
    ancestor = mesh
    while ancestor.GetPath() != path:
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("original palm collision belongs to a nested rigid body")
        ancestor = ancestor.GetParent()
    if source["collision_approximation"] != "convexHull":
        raise ValueError("original palm convexHull approximation required")
    vertices = np.asarray(source["points_body"], dtype=np.float32)
    triangles = np.asarray(source["triangles"], dtype=np.int64)
    if source["orientation"] == "leftHanded":
        triangles = triangles[:, [0, 2, 1]]
    row = {k: source[k] for k in ("body_path", "mesh_path", "collision_owner", "collision_approximation",
                                   "orientation", "mesh_to_body", "geometry_sha256")}
    row.update(instance_proxy=mesh.IsInstanceProxy(), frame="PhysX_link_origin", unit="m")
    return vertices, triangles, row


def checked_capture(folder, device):
    native = folder / "native"
    receipt = json.loads((folder / "outer.json").read_text())
    if (type(receipt["exit_code"]) is not int or receipt["exit_code"] != 0
            or receipt["forced_exit"] is not False or receipt["device"] != device
            or receipt["native"] != str(native)):
        raise ValueError("successful independent Power palm capture identity required")
    for name, digest in receipt["files"].items():
        artifact = native / name
        if Path(name).name != name or artifact.is_symlink() or file_sha(artifact) != digest:
            raise ValueError("Power native file differs from outer receipt")
    return native


def compare(cpu, gpu):
    runs = (checked_capture(cpu, "cpu"), checked_capture(gpu, "cuda:0"))
    for run in runs:
        for name in NATIVE_FILES:
            if name not in json.loads((run.parent / "outer.json").read_text())["files"]:
                raise ValueError("incomplete Power palm lifecycle or source files")
        result = json.loads((run / "result.json").read_text())
        if (result["clock"]["steps_commanded"] != STEPS or result["sdk_pad_bodies_found"] != 2
                or result["pads_with_live_hulls"] != 2):
            raise ValueError("two original same-clock palm collider rows required")
    with bindings((parity, "DISTAL_PAD_LINKS", LINKS)):
        result = parity.compare_native_runs(*runs)
    result["palms"] = result.pop("pads")
    result["scope"] = "original_dh116_live_power_palm_cpu_cuda0_cooking_parity"
    result["palm_sdf_registered"] = False
    result["force_contact_qualified"] = False
    return result


def capture_pair(root):
    root.mkdir()
    frozen = {str(Path(path).resolve()): file_sha(Path(path))
              for path in (__file__, probe.__file__, parity.__file__)}
    write_json(root / "attempt.json", dict(
        source_sha256=frozen, source_assets=probe._sources(ROOT),
        serial_devices=["cpu", "cuda:0"], physics_steps_per_device=STEPS,
        soft_deadline_s=180, auto_retry=False, auto_kill=False))
    for label, device in (("cpu", "cpu"), ("gpu", "cuda:0")):
        if any(file_sha(Path(path)) != expected for path, expected in frozen.items()):
            raise ValueError("Power capture implementation changed during the attempt")
        folder = root / label
        folder.mkdir()
        out = folder / "native"
        argv = [sys.executable, "-B", "-m", __spec__.name, "--capture", str(out), "--device", device]
        with (folder / "native.log").open("xb") as log, subprocess.Popen(
                argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT) as worker:
            try:
                status = worker.wait(timeout=180)
            except subprocess.TimeoutExpired:
                write_json(folder / "soft_deadline.json", {"pid": worker.pid})
                status = worker.wait()
            log.flush()
            os.fsync(log.fileno())
        write_json(folder / "outer.json", dict(exit_code=status, forced_exit=False,
            device=device, native=str(out), files={p.name: file_sha(p) for p in out.glob("*.json")}))
        if status != 0:
            return status
    result = compare(root / "cpu", root / "gpu")
    write_json(root / "parity.json", result)
    return 0 if result["parity_passed"] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--capture", type=Path)
    modes.add_argument("--capture-pair", type=Path)
    modes.add_argument("--compare", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda:0"))
    args = p.parse_args(argv)
    if args.capture is not None:
        if not args.device:
            p.error("a capture needs the physical device")
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0], "--output", str(args.capture), "--execute-native", "--headless",
                        "--steps", str(STEPS), "--device", args.device]
            with bindings((probe, "DISTAL_PAD_LINKS", LINKS), (probe, "read_pad_triangles", read_palm_source)):
                probe.main()
        finally:
            sys.argv = old_argv
        return 0
    folder = (args.capture_pair or args.compare).resolve()
    if args.capture_pair is not None:
        return capture_pair(folder)
    result = compare(folder / "cpu", folder / "gpu")
    print(json.dumps(result, indent=2))
    return int(not result["parity_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
