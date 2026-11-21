#!/usr/bin/env python3
"""Read an original little-finger proximal collider and its native ray witnesses."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np

from scripts import audit_dh116_power_palm_parity as prior
from scripts.audit_dh116_power_palm_raycast import ray_audit
from scripts.diagnose_dh116_teacher_initialization import digest, sha, write_new

LINK = "finger52_link"
ROOT = Path(__file__).resolve().parents[1]


def record_rays(original, stage, physx, hands, output, clock):
    from omni.physx import get_physx_scene_query_interface
    from pxr import Gf

    rows = original(stage, physx, hands, output, clock)
    if len(rows) != 2 or {row["side"] for row in rows} != {"left", "right"}:
        raise ValueError("two independently captured proximal colliders required")
    interface = get_physx_scene_query_interface()

    def cast(origin, direction, limit):
        return interface.raycast_closest(Gf.Vec3f(*origin), Gf.Vec3f(*direction), limit)

    reports = {}
    for row in rows:
        # The same-frame raw fan selects ray targets. The native hit independently
        # witnesses the actual scene shape; device parity is audited after both runs.
        if "error" in row:
            raise ValueError(f"proximal collider acquisition failed: {row['error']}")
        reports[row["side"]] = ray_audit(row, row, cast, expected_link=LINK)
    write_new(output/"proximal_rays.json", dict(link=LINK, clock=clock, reports=reports,
        sdf_registered=False, native_contact_qualified=False, teacher_admission=False))
    if not all(row["owned_rays"] for row in reports.values()):
        raise ValueError("both proximal links must have owned native ray hits")
    return rows


def _receipt(root, requested_device):
    path = root/"outer.json"
    outer = json.loads(path.read_bytes())
    if (outer["exit_code"] != 0 or (outer["debugger_inferior_exit_code"] != 0)
            or outer["debugger_inferior_signal"] is not None or outer["forced_exit"] or outer["exceeded_budget"]):
        raise ValueError("successful natural native exit required before parity")
    if sha(root/"native.log") != outer["native_log_sha256"]:
        raise ValueError("native ray log SHA differs")
    n = root/"native"
    for name, value in outer["artifact_sha256"].items():
        if Path(name).name != name or (n/name).is_symlink() or sha(n/name) != value:
            raise ValueError("independent proximal artifact differs")
    result = json.loads((n/"result.json").read_bytes())
    if result["physics_device"]["requested_device"] != requested_device or result["clock"]["steps_commanded"] != 2:
        raise ValueError("exact native device and two-step capture required")
    return n


def compare(cpu, gpu):
    outputs = (_receipt(cpu, "cpu"), _receipt(gpu, "cuda:0"))
    with prior.bindings((prior.parity, "DISTAL_PAD_LINKS", (LINK,))):
        device_parity = prior.parity.compare_native_runs(*outputs)
    rays = {}
    for output, label in zip(outputs, ("cpu", "gpu"), strict=True):
        record = json.loads((output/"proximal_rays.json").read_bytes())
        rays[label] = {}
        for side, side_record in record["reports"].items():
            rows = [row for row in side_record["rays"] if row["native_hit_distance_m"] is not None]
            if not rows:
                raise ValueError(f"no independently owned {side} {label} PhysX ray hits")
            rays[label][side] = dict(total=side_record["total_rays"], owned=len(rows),
                original_fan_abs_error_max_m=float(max(abs(row["native_hit_distance_m"] -
                    row["original_fan_m"]) for row in rows)),
                original_fan_abs_error_p95_m=float(np.percentile([abs(row["native_hit_distance_m"] -
                    row["original_fan_m"]) for row in rows], 95)))
    return dict(parity=device_parity, rays=rays, sdf_registered=False,
                native_contact_qualified=False, grasp_qualified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--approved-frozen-sha256")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("CPU", "GPU"))
    args = parser.parse_args(argv)
    if args.compare:
        result = compare(*args.compare)
        write_new(args.output, result)
        return int(not result["parity"]["parity_passed"])
    if args.device is None or args.plan is None:
        parser.error("a frozen native capture requires --device and --plan")
    files = (Path(__file__), Path(prior.__file__), Path(prior.probe.__file__), Path(prior.parity.__file__),
             ROOT/"scripts/audit_dh116_power_palm_raycast.py")
    plan = dict(output=str(args.output.resolve()), device=args.device, link=LINK,
                frozen_files={str(path.resolve()):sha(path) for path in files},
                source_assets=prior.probe._sources(ROOT),
                budget=dict(sdk_steps=2, solver_steps=0, control_steps=0, ppo_updates=0,
                            soft_wall_seconds=180, auto_retries=0, auto_kill=False))
    if args.approved_frozen_sha256 is None:
        write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=digest(plan), budget=plan["budget"])))
        return 0
    if args.approved_frozen_sha256 != digest(plan) or json.loads(args.plan.read_bytes()) != plan:
        raise ValueError("the native proximal geometry recipe changed")
    original = sys.argv
    try:
        sys.argv = [original[0], "--output", str(args.output), "--execute-native", "--headless",
                    "--steps", "2", "--device", args.device]
        recorder = partial(record_rays, prior.probe._record_pads)
        read = partial(prior.read_collision_source, expected_link=LINK)
        with prior.bindings((prior.probe, "DISTAL_PAD_LINKS", (LINK,)),
                            (prior.probe, "read_pad_triangles", read),
                            (prior.probe, "_record_pads", recorder)):
            prior.probe.main(scene_queries=True)
    finally:
        sys.argv = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
