#!/usr/bin/env python3
"""Run one native lifetime-isolation group in one fresh Isaac process."""
import argparse
import faulthandler
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from go2_pvcnn.control.dh116.tabletop_deformable_rod_bridge import NativeOwnershipTrace
from scripts.verify_dh116_deformx_rigid_bridge import DT, run_phases, write_json


COMMIT = "96b8514c7c55c724f1abba1fbf6ed6642b7db72b"
EXTENSION = "c47942c4b15b3dfedc47212e71f8193825c0347df3c1d73fb76b648f7bdedb53"
BRIDGE = "7a593ec2b56f4115e3c04ebe5cd6969294c7047ec8c577012bfcb3c1bf649eae"


def group_features(group):
    """Extension import, CUDA rod, RopeColliders, staged reaction."""
    index = "ABCDE".index(group)
    return tuple(index >= threshold for threshold in range(1, 5))


def run_tensor_probe(sim, args, ownership, create_rod):
    import omni.physics.tensors as tensors
    import torch
    import isaaclab.sim as sim_utils

    cfg = sim_utils.CuboidCfg(
        size=(0.20, 0.20, 0.20), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    cfg.func("/World/Probe", cfg, translation=(0.3, 0.0, 1.0))
    sim.reset()
    view = tensors.create_simulation_view("torch")
    ownership.watch("explicit_simulation_view", view, owner="probe",
                    release="invalidate then drop reference", gpu_view=True)
    body = view.create_rigid_body_view("/World/Probe")
    ownership.watch("explicit_body_view", body, owner="probe view",
                    release="drop before view invalidation", gpu_view=True)
    if body.count != 1:
        raise RuntimeError("expected one native tensor body")
    rope = None
    if create_rod:
        import numpy as np
        import ropesim

        positions = np.zeros((1, 21, 3), dtype=np.float32)
        positions[0, :, 0] = np.arange(21) * 0.025
        positions[:, :, 2] = 2
        pin = np.zeros((1, 21), dtype=bool)
        pin[0, 0] = True
        rope = ropesim.RopeBatch.from_positions(
            positions, 0.11714786, 11472.95064, 0.004479735,
            DT / 12, 16, pinned=pin,
        )
        ownership.watch("rope_batch", rope, owner="probe", release="drop after CUDA synchronize",
                        gpu_view=True)
        root = torch.as_tensor(positions[:, 0].copy(), device=args.device)
        speed = torch.zeros_like(root)
    with (args.output / "physics.jsonl").open("x", encoding="utf-8") as ledger:
        for tick in range(10):
            sim.step(render=False)
            row = dict(step=tick + 1, sdk_step=int(sim.current_time_step_index),
                       sdk_time_s=float(sim.current_time),
                       body_pose=body.get_transforms().detach().cpu().tolist())
            if rope is not None:
                rope.step(root, speed, DT, 12, root_pos_end=root)
                row["rod_tip"] = torch.as_tensor(rope.positions())[0, -1].detach().cpu().tolist()
            ledger.write(json.dumps(row, allow_nan=False) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())
    torch.cuda.synchronize(args.device)
    ownership.event("ledger fsynced and CUDA synchronized")
    del body
    view.invalidate()
    ownership.event("explicit simulation view invalidated", sdk_valid=bool(view.is_valid))
    del view, rope
    gc.collect()
    ownership.checkpoint("probe views and solver references dropped")
    return dict(steps=10, rod_steps=10 if create_rod else 0)


def launch_one(args, launcher):
    from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import close_calibration_app
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock

    ownership = NativeOwnershipTrace(args.output / "ownership.jsonl")
    with native_instance_lock():
        app = launcher(args, fast_shutdown=False).app
        faulthandler.enable(file=sys.__stderr__, all_threads=True)
        sim = None
        try:
            import carb.settings
            import isaaclab.sim as sim_utils
            from isaacsim.core.simulation_manager import SimulationManager

            if carb.settings.get_settings().get("/app/fastShutdown") is not False:
                raise RuntimeError("native shutdown probe requires natural app teardown")
            load_extension, create_rod, create_bridge, reaction = group_features(args.group)
            if load_extension:
                import ropesim
                ownership.event("pinned ropesim extension imported", module=ropesim.__file__,
                                python_extension_statics_release="interpreter finalization")
            sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
                dt=DT, render_interval=1, device=args.device, gravity=(0.0, 0.0, 0.0),
            ))
            ownership.watch("simulation_context", sim, owner="probe / singleton",
                            release="clear_instance before timeline STOP", callbacks=True)
            if create_bridge:
                args.steps_per_phase = getattr(args, "steps_per_phase", 120) if reaction else 10
                result = run_phases(sim, args, control_only=not reaction, ownership=ownership)
            else:
                result = run_tensor_probe(sim, args, ownership, create_rod)
            ownership.checkpoint("native sampling returned")
            write_json(args.output / "result.json", dict(group=args.group, **result))
        except Exception:
            write_json(args.output / "failure.json", {"traceback": traceback.format_exc()})
            raise
        finally:
            try:
                if sim is not None:
                    import omni.timeline

                    native_view = SimulationManager.get_physics_sim_view()
                    if native_view is not None:
                        ownership.watch("manager_view", native_view, owner="SimulationManager",
                                        release="timeline STOP callback", gpu_view=True, callbacks=True)
                        del native_view
                    type(sim).clear_instance()
                    sim = None
                    gc.collect()
                    timeline = omni.timeline.get_timeline_interface()
                    timeline.stop()
                    app.update()
                    released = SimulationManager.get_physics_sim_view() is None
                    del timeline
                    ownership.event("SDK STOP callback returned", manager_view_released=released)
                    if not released:
                        raise RuntimeError("SimulationManager tensor view survived STOP")
                    gc.collect()
                    ownership.checkpoint("before app close")
            finally:
                close_calibration_app(app, args.output)
                ownership.checkpoint("app.close returned; outer exit not yet observed")


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=list("ABCDE"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ropesim-python", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.group != "A":
        if args.ropesim_python is None:
            parser.error("groups B-E require pinned ropesim module directory")
        backend = args.ropesim_python.resolve(strict=True)
        extensions = list((backend / "ropesim").glob("_ropesim.cpython-310-*.so"))
        if (len(extensions) != 1 or hashlib.sha256(extensions[0].read_bytes()).hexdigest() != EXTENSION
                or hashlib.sha256((backend / "ropesim/isaac_bridge.py").read_bytes()).hexdigest() != BRIDGE):
            parser.error("pinned ropesim extension or bridge digest mismatch")
        sys.path.insert(0, str(backend))
    args.backend_commit = COMMIT
    args.extension_sha256 = EXTENSION
    args.bridge_sha256 = BRIDGE
    launch_one(args, AppLauncher)


if __name__ == "__main__":
    main()
