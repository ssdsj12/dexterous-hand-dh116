#!/usr/bin/env python3
"""Bounded native validation of a DeformX rod and one isolated rigid body."""

import argparse
import faulthandler
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


DT = 1 / 480
SOLVER_SUBSTEPS = 12
MASS = 0.5


def rotation_wxyz(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def wrench_at_com(wrench, link, com):
    wrench = np.asarray(wrench, dtype=float)
    return np.r_[wrench[:3], wrench[3:] - np.cross(np.subtract(com, link), wrench[:3])]


def angular_momentum(inertia, quaternion, omega):
    rotation = rotation_wxyz(quaternion)
    return rotation @ inertia @ rotation.T @ omega


def impulse_audit(rows, reaction):
    previous = np.zeros(6)
    delivered, momentum = np.zeros(6), np.zeros(6)
    linear_residual, angular_residual = [], []
    for index, row in enumerate(rows):
        if (row["phase_step"] != index + 1 or index and row["sdk_step"] != rows[index - 1]["sdk_step"] + 1
                or not np.isfinite(row["dt_s"]) or row["dt_s"] <= 0):
            raise ValueError("non-contiguous physics clock")
        applied = np.asarray(row["applied_wrench_com_w"], dtype=float)
        if not np.allclose(applied, previous, atol=3e-5, rtol=2e-5):
            raise ValueError("applied wrench must match previous rod step")
        delta = np.r_[row["delta_momentum_w"], row["delta_angular_momentum_com_w"]]
        impulse = applied * row["dt_s"]
        delivered += impulse
        momentum += delta
        linear_residual.append(float(np.linalg.norm(delta[:3] - impulse[:3])))
        angular_residual.append(float(np.linalg.norm(delta[3:] - impulse[3:])))
        if reaction:
            previous = np.asarray(row["object_wrench_com_w"], dtype=float)
    return dict(force_impulse_w=delivered[:3].tolist(), torque_impulse_com_w=delivered[3:].tolist(),
                measured_momentum_change_w=momentum[:3].tolist(),
                measured_angular_momentum_change_com_w=momentum[3:].tolist(),
                linear_net_residual=float(np.linalg.norm(momentum[:3] - delivered[:3])),
                angular_net_residual=float(np.linalg.norm(momentum[3:] - delivered[3:])),
                max_linear_residual=max(linear_residual), max_angular_residual=max(angular_residual))


def stage_owned_wrench(bridge, composer):
    """Refresh the exclusively owned test composer link pose before a world-wrench write."""
    composer.reset()
    bridge.apply_wrenches()


def write_json(path, data):
    serialized = json.dumps(data, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as file:
        file.write(serialized + "\n")
        file.flush()
        os.fsync(file.fileno())


def run_phases(sim, args):
    import importlib.metadata
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    import ropesim
    from ropesim.isaac_bridge import ColliderSpec, RopeColliders

    @configclass
    class FreeBoxScene(InteractiveSceneCfg):
        body = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Body",
            spawn=sim_utils.CuboidCfg(
                size=(0.20, 0.20, 0.20),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.0, angular_damping=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 1.0)),
        )

    scene = InteractiveScene(FreeBoxScene(num_envs=1, env_spacing=3.0))
    sim.reset()
    body = scene["body"]
    device = torch.device(args.device)
    composer = body.permanent_wrench_composer
    node_count, segment, radius = 21, 0.025, 0.00635

    def numpy(value):
        return value.detach().cpu().numpy().astype(float, copy=True)

    def snapshot():
        d = body.data
        return dict(link=numpy(d.root_link_pos_w[0]), link_q=numpy(d.root_link_quat_w[0]),
                    com=numpy(d.root_com_pos_w[0]), com_q=numpy(d.root_com_quat_w[0]),
                    v=numpy(d.root_com_lin_vel_w[0]), omega=numpy(d.root_com_ang_vel_w[0]))

    def supplied_wrench(pose):
        if not composer.active:
            return np.zeros(6)
        rotate = rotation_wxyz(pose["link_q"])
        return np.r_[rotate @ numpy(composer.composed_force_as_torch[0, 0]),
                     rotate @ numpy(composer.composed_torque_as_torch[0, 0])]

    phases = {}
    with (args.output / "physics.jsonl").open("x", encoding="utf-8") as ledger:
        for name, reaction in (("control", False), ("dynamic", True)):
            scene.reset()
            initial = body.data.default_root_state.clone()
            body.write_root_pose_to_sim(initial[:, :7])
            body.write_root_velocity_to_sim(initial[:, 7:])
            scene.update(0.0)
            pose = snapshot()
            mass = float(body.root_physx_view.get_masses().reshape(-1)[0])
            inertia = numpy(body.root_physx_view.get_inertias()[0]).reshape(3, 3)
            if abs(mass - MASS) > 1e-6:
                raise ValueError("unexpected PhysX mass")
            start = pose["link"] + np.array([-0.1 - 0.035 - (node_count-1)*segment, 0.025, -0.025])
            pos = start + np.arange(node_count)[:, None] * np.array([segment, 0, 0])
            pin = np.zeros((1, node_count), dtype=bool)
            pin[0, 0] = True
            rope = ropesim.RopeBatch.from_positions(
                pos[None].astype(np.float32), 0.11714786, 11472.95064, 0.004479735,
                DT / SOLVER_SUBSTEPS, 16, pinned=pin,
            )
            rope.damping = 0.0
            rope.drag = 0.0
            rope.drag_quad = 0.0
            rope.set_gravity(np.zeros(3))
            rope.set_contact(radius=radius, mu_s=0.5, mu_k=0.4, v_slip=1e-3, kappa=2.0)
            bridge = RopeColliders(rope, scene, [ColliderSpec(
                "body", dynamic=reaction, mu_s=0.8, mu_k=0.7, spacing=0.005, margin=0.03,
            )])
            rope.enable_contact_forces(True)
            bridge.sync_poses(reset_env_ids=torch.tensor([0], device=device))
            root = torch.tensor(pos[:1], dtype=torch.float32, device=device)
            velocity = torch.tensor([[0.3, 0.0, 0.0]], device=device)
            rows = []
            for tick in range(args.steps_per_phase):
                before = snapshot()
                applied = supplied_wrench(before)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(DT)
                after = snapshot()
                bridge.sync_poses()
                pa, pb = root + tick * DT * velocity, root + (tick + 1) * DT * velocity
                rope.step(pa, velocity, DT, SOLVER_SUBSTEPS, root_pos_end=pb)
                node_pos = numpy(torch.as_tensor(rope.positions())[0])
                node_force = numpy(torch.as_tensor(rope.contact_forces_view())[0])
                object_link_wrench = numpy(bridge.wrenches()[0, 0])
                object_com_wrench = wrench_at_com(object_link_wrench, after["link"], after["com"])
                if reaction:
                    stage_owned_wrench(bridge, composer)
                delta_l = angular_momentum(inertia, after["com_q"], after["omega"]) - angular_momentum(
                    inertia, before["com_q"], before["omega"]
                )
                row = dict(phase=name, phase_step=tick + 1,
                           sdk_step=int(sim.current_time_step_index), sdk_time_s=float(sim.current_time),
                           rod_substep=(tick + 1) * SOLVER_SUBSTEPS, dt_s=DT,
                           box_link_w=after["link"].tolist(), box_com_w=after["com"].tolist(),
                           box_link_quat_wxyz=after["link_q"].tolist(), box_com_quat_wxyz=after["com_q"].tolist(),
                           box_com_linear_velocity_w=after["v"].tolist(), box_angular_velocity_w=after["omega"].tolist(),
                           object_wrench_link_w=object_link_wrench.tolist(), object_wrench_com_w=object_com_wrench.tolist(),
                           applied_wrench_com_w=applied.tolist(), rod_node_position_w=node_pos.tolist(),
                           rod_vertex_contact_force_w=node_force.tolist(), rod_contact_force_w=node_force.sum(0).tolist(),
                           rod_contact_torque_link_origin_end_pose_estimate_w=np.cross(
                               node_pos - after["link"], node_force
                           ).sum(0).tolist(),
                           delta_momentum_w=(mass * (after["v"] - before["v"])).tolist(),
                           delta_angular_momentum_com_w=delta_l.tolist(),
                           pair_force_residual=float(np.linalg.norm(node_force.sum(0) + object_link_wrench[:3])))
                rows.append(row)
                ledger.write(json.dumps(row, allow_nan=False) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            summary = impulse_audit(rows, reaction)
            summary.update(steps=len(rows), first_sdk_step=rows[0]["sdk_step"], last_sdk_step=rows[-1]["sdk_step"],
                           initial_body_link_w=pose["link"].tolist(), final_body_link_w=after["link"].tolist(),
                           initial_rod_tip_w=pos[-1].tolist(), contact_steps=int(sum(
                               np.linalg.norm(row["object_wrench_link_w"][:3]) > 1e-5 for row in rows)),
                           max_pair_force_residual=max(row["pair_force_residual"] for row in rows),
                           peak_object_force_n=float(max(np.linalg.norm(row["object_wrench_link_w"][:3]) for row in rows)),
                           peak_object_torque_nm=float(max(np.linalg.norm(row["object_wrench_com_w"][3:]) for row in rows)),
                           mass_kg=mass, inertia_com_kgm2=inertia.tolist())
            phases[name] = summary
            del bridge, rope
    control, dynamic = phases["control"], phases["dynamic"]
    if (not np.allclose(control["initial_body_link_w"], dynamic["initial_body_link_w"], atol=1e-7, rtol=0)
            or not np.allclose(control["initial_rod_tip_w"], dynamic["initial_rod_tip_w"], atol=1e-7, rtol=0)
            or not control["contact_steps"] or not dynamic["contact_steps"]
            or np.linalg.norm(control["measured_momentum_change_w"]) > 1e-6
            or np.linalg.norm(dynamic["measured_momentum_change_w"]) < 1e-3
            or np.linalg.norm(dynamic["measured_angular_momentum_change_com_w"]) < 1e-5
            or dynamic["linear_net_residual"] > 1e-3 or dynamic["angular_net_residual"] > 1e-4
            or dynamic["max_pair_force_residual"] > 1e-3 or control["max_pair_force_residual"] > 1e-3):
        raise ValueError(f"isolated bidirectional coupling invariants failed: {phases}")
    return dict(scope="minimal_single_rigid_body_rod_only", bidirectional_rigid_body_passed=True,
                dh116_hand_qualified=False, backend_commit=args.backend_commit,
                extension_sha256=args.extension_sha256, bridge_source_sha256=args.bridge_sha256,
                isaac_sim=importlib.metadata.version("isaacsim"), isaac_lab=importlib.metadata.version("isaaclab"),
                torch_version=torch.__version__, gpu=torch.cuda.get_device_name(device),
                dt_s=DT, rod_substeps_per_exchange=SOLVER_SUBSTEPS, phases=phases,
                ledger_sha256=hashlib.sha256((args.output / "physics.jsonl").read_bytes()).hexdigest())


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ropesim-python", type=Path, required=True)
    parser.add_argument("--backend-commit", required=True, choices=["96b8514c7c55c724f1abba1fbf6ed6642b7db72b"])
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--bridge-sha256", required=True)
    parser.add_argument("--steps-per-phase", type=int, default=120)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.steps_per_phase <= 120:
        parser.error("combined physics budget is capped at 240 steps")
    backend = args.ropesim_python.resolve(strict=True)
    extensions = list((backend / "ropesim").glob("_ropesim.cpython-310-*.so"))
    if (len(extensions) != 1 or hashlib.sha256(extensions[0].read_bytes()).hexdigest() != args.extension_sha256
            or hashlib.sha256((backend / "ropesim/isaac_bridge.py").read_bytes()).hexdigest() != args.bridge_sha256):
        parser.error("pinned ropesim extension and bridge digests must match")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(backend))
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock

    with native_instance_lock():
        native_run(args, AppLauncher)


def native_run(args, launcher):
    app = launcher(args, fast_shutdown=False).app
    faulthandler.enable(file=sys.__stderr__, all_threads=True)
    simulation = None
    try:
        import carb.settings
        import isaaclab.sim as sim_utils

        shutdown_mode = carb.settings.get_settings().get("/app/fastShutdown")
        if shutdown_mode is not False:
            raise RuntimeError("Isaac app must allow natural shutdown")
        write_json(args.output / "shutdown_mode.json", {"fast_shutdown": shutdown_mode})
        simulation = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=DT, render_interval=1, device=args.device, gravity=(0.0, 0.0, 0.0),
        ))
        write_json(args.output / "result.json", run_phases(simulation, args))
    except Exception:
        write_json(args.output / "failure.json", {"exception": traceback.format_exc()})
        raise
    finally:
        from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import close_calibration_app

        if simulation is None:
            close_calibration_app(app, args.output)
        else:
            release_owned_native_view(app, simulation, args.output)


def release_owned_native_view(app, simulation, output):
    import omni.timeline
    from isaacsim.core.simulation_manager import SimulationManager
    from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import close_calibration_app

    cls = type(simulation)
    if cls.instance() is not simulation:
        raise RuntimeError("expected the probe-owned SimulationContext")
    # Clear the Lab STOP hook first, then dispatch the SDK STOP event so the
    # SimulationManager releases its tensor view while its plugin is live.
    cls.clear_instance()
    del simulation
    gc.collect()
    timeline = omni.timeline.get_timeline_interface()
    try:
        timeline.stop()
        app.update()
        if SimulationManager.get_physics_sim_view() is not None:
            raise RuntimeError("owned SDK simulation view survived STOP")
        write_json(output / "simulation_context_cleared.json", {
            "public_api": "SimulationContext.clear_instance then timeline.stop",
            "simulation_manager_tensor_view_released": True,
        })
    finally:
        del timeline
        close_calibration_app(app, output)


if __name__ == "__main__":
    main()
