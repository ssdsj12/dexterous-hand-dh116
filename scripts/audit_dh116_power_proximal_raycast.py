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
SETUP_SDK_STEPS = 2
CAPTURE_SDK_STEPS = 2


def owned_tensor_pose(view, path):
    if view.count != 1 or list(view.prim_paths) != [path]:
        raise ValueError("tensor pose did not bind the exact original rigid body")
    raw = view.get_transforms()
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    values = np.asarray(raw, dtype=float).copy()
    if (values.shape != (1, 7) or not np.isfinite(values).all()
            or abs(np.linalg.norm(values[0, 3:]) - 1.) > 1.e-5):
        raise ValueError("finite original unit-XYZW tensor pose required")
    return dict(position_m=values[0, :3].tolist(), quaternion_xyzw=values[0, 3:].tolist())


def record_rays(original, stage, physx, hands, output, clock, *, tensor_pose_control=False):
    from omni.physx import get_physx_scene_query_interface
    from pxr import Gf

    rows = original(stage, physx, hands, output, clock)
    if len(rows) != 2 or {row["side"] for row in rows} != {"left", "right"}:
        raise ValueError("two independently captured proximal colliders required")
    interface = get_physx_scene_query_interface()

    def cast(origin, direction, limit):
        return interface.raycast_closest(Gf.Vec3f(*origin), Gf.Vec3f(*direction), limit)

    reports, tensor_comparison = {}, {}
    if tensor_pose_control:
        from isaaclab.sim import SimulationContext
        from isaacsim.core.simulation_manager import SimulationManager

        sim = SimulationContext.instance()
        simulation_view = SimulationManager.get_physics_sim_view()
        start_clock = (sim.current_time_step_index, float(sim.current_time))
    for row in rows:
        # The same-frame raw fan selects ray targets. The native hit independently
        # witnesses the actual scene shape; device parity is audited after both runs.
        if "error" in row:
            raise ValueError(f"proximal collider acquisition failed: {row['error']}")
        side = row["side"]
        reports[side] = ray_audit(row, row, cast, expected_link=LINK)
        if tensor_pose_control:
            path = row["source"]["body_path"]
            view = simulation_view.create_rigid_body_view(path)
            try:
                tensor_pose = owned_tensor_pose(view, path)
            finally:
                view = None
            aligned = dict(row, sdk_body=dict(row["sdk_body"], **tensor_pose))
            tensor_comparison[side] = dict(rigid_pose=row["sdk_body"], tensor_pose=tensor_pose,
                rays=ray_audit(aligned, row, cast, expected_link=LINK))
    if tensor_pose_control:
        if start_clock != (sim.current_time_step_index, float(sim.current_time)):
            raise ValueError("SDK clock advanced during the paired pose observations")
        write_new(output/"proximal_tensor_pose_rays.json", dict(link=LINK, sdk_clock=start_clock,
            evidence_scope="same_frame_read_only_pose_comparison", comparisons=tensor_comparison,
            sdf_registered=False, teacher_admission=False))
    write_new(output/"proximal_rays.json", dict(link=LINK, clock=clock, reports=reports,
        sdf_registered=False, native_contact_qualified=False, teacher_admission=False))
    if not all(row["owned_rays"] for row in reports.values()):
        raise ValueError("both proximal links must have owned native ray hits")
    return rows


def _receipt(root, requested_device):
    path = root/"outer.json"
    outer = json.loads(path.read_bytes())
    if (outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or outer["debugger_status_verified"] is not True or outer["debugger_inferior_signal"] is not None
            or outer["forced_exit"] or outer["exceeded_budget"]):
        raise ValueError("successful natural native exit required before parity")
    if sha(root/"native.log") != outer["native_log_sha256"]:
        raise ValueError("native ray log SHA differs")
    n = root/"native"
    for name, value in outer["artifact_sha256"].items():
        if Path(name).name != name or (n/name).is_symlink() or sha(n/name) != value:
            raise ValueError("independent proximal artifact differs")
    result = json.loads((n/"result.json").read_bytes())
    if (result["physics_device"]["requested_device"] != requested_device
            or result["clock"]["steps_commanded"] != CAPTURE_SDK_STEPS
            or result["clock"]["physics_step"] != SETUP_SDK_STEPS + CAPTURE_SDK_STEPS):
        raise ValueError("exact native device and setup/capture clock required")
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
    parser.add_argument("--tensor-pose-control", action="store_true")
    parser.add_argument("--approved-frozen-sha256")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("CPU", "GPU"))
    args = parser.parse_args(argv)
    if args.compare:
        result = compare(*args.compare)
        write_new(args.output, result)
        return int(not result["parity"]["parity_passed"])
    if args.device is None or args.plan is None:
        parser.error("a frozen native capture requires --device and --plan")
    if args.tensor_pose_control and args.device != "cuda:0":
        parser.error("the paired Tensor View pose comparison uses CUDA PhysX")
    files = (Path(__file__), Path(prior.__file__), Path(prior.probe.__file__), Path(prior.parity.__file__),
             ROOT/"scripts/audit_dh116_power_palm_raycast.py")
    plan = dict(output=str(args.output.resolve()), device=args.device, link=LINK,
                tensor_pose_control=args.tensor_pose_control,
                frozen_files={str(path.resolve()):sha(path) for path in files},
                source_assets=prior.probe._sources(ROOT),
                budget=dict(sdk_steps=SETUP_SDK_STEPS+CAPTURE_SDK_STEPS, sdk_setup_steps=SETUP_SDK_STEPS,
                            sdk_capture_steps=CAPTURE_SDK_STEPS, solver_steps=0, control_steps=0, ppo_updates=0,
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
                    "--steps", str(CAPTURE_SDK_STEPS), "--device", args.device]
        recorder = partial(record_rays, prior.probe._record_pads, tensor_pose_control=args.tensor_pose_control)
        read = partial(prior.read_collision_source, expected_link=LINK)
        with prior.bindings((prior.probe, "DISTAL_PAD_LINKS", (LINK,)),
                            (prior.probe, "read_pad_triangles", read),
                            (prior.probe, "_record_pads", recorder)):
            prior.probe.main(scene_queries=True, exact_setup_steps=SETUP_SDK_STEPS)
    finally:
        sys.argv = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
