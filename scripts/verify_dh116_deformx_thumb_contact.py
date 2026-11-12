#!/usr/bin/env python3
"""One original DH116 thumb and one locked flexible rod, with bounded native evidence."""

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

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_rod_bridge import NativeOwnershipTrace
from go2_pvcnn.control.dh116.tabletop_deformable_rod_contact import (
    DT, MOVE_STEPS, RADIUS_M, STEPS, SUBSTEPS, ZERO_STEPS, command_at_step, min_grid_distance,
    plan_thumb_approach, schedule_summary,
)
from scripts.bake_dh116_deformx_pad_sdfs import checked_bytes, checked_json
from scripts.verify_dh116_cooked_convex_hulls_native import _sources
from scripts.verify_dh116_deformx_rigid_bridge import rotation_wxyz, wrench_at_com, write_json


_SDF_SUMMARY_SHA = "db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab"
_GPU_RESULT_SHA = "2de331289145d4be824243db0eeb64b796f24a92cee7b93ee61a37244e9bbde5"
_THUMB = "finger14_link"


def left_output_from_scheduler_receipt(receipt_path, expected_sha):
    """Reopen the scheduler-owned outer process and qualified output evidence."""
    receipt_path = receipt_path.resolve(strict=True)
    attempt = receipt_path.parent
    if (receipt_path.name != "receipt.json" or not attempt.name.startswith("attempt-")
            or attempt.parent.parent.name != "flex.f4-left-contact-native"
            or attempt.parent.parent.parent.name != "native-attempts"):
        raise ValueError("left receipt must belong to the scheduler's left-thumb attempt")
    if expected_sha is None:
        raise ValueError("expected scheduler receipt SHA256 is required")
    receipt = checked_json(receipt_path, expected_sha)
    intent = json.loads((attempt / "intent.json").read_text())
    outer = json.loads((attempt / "outer_exit.json").read_text())
    output = Path(receipt["output"])
    if (receipt["executor"] != "native_bounded" or receipt["native_execution_performed"] is not True
            or receipt["native_qualification"] is not True or intent["package"] != attempt.parent.parent.name
            or not output.is_absolute() or output.resolve(strict=True) != output
            or (output.name, output.parent.name, output.parent.parent.name)
               != ("native", attempt.name, intent["package"])
            or intent["output"] != receipt["output"]
            or intent["code_config_sha256"] != attempt.parent.name
            or any(intent[key] != receipt[key] for key in ("code_config_sha256", "command_spec_sha256"))):
        raise ValueError("left scheduler receipt lacks qualified attempt identity")
    if (type(outer["outer_exit_code"]) is not int or outer["outer_exit_code"] != 0
            or receipt["outer_exit_code"] != 0 or outer["native_lock_observed"] is not True
            or receipt["native_lock_observed"] is not True or outer["log_sha256"] != receipt["log_sha256"]):
        raise ValueError("left scheduler attempt requires independently recorded outer exit 0")
    checked_bytes(attempt / "native.log", outer["log_sha256"])
    digests = receipt["artifact_sha256"]
    ledger = checked_bytes(output / "physics.jsonl", digests["physics.jsonl"])
    if receipt["ledger_counts"] != {"physics.jsonl": STEPS} or ledger.count(b"\n") != STEPS:
        raise ValueError("left scheduler ledger count differs")
    result = checked_json(output / "result.json", digests["result.json"])
    marker = checked_json(output / "shutdown_complete.json", digests["shutdown_complete.json"])
    if (result["side"] != "left" or result["sample_validation_passed"] is not True
            or result["physics_rows"] != STEPS or result["failure_reasons"]
            or marker["close_returned"] is not True):
        raise ValueError("right requires qualified left contact and shutdown")
    return output


def preflight(args, project):
    if args.steps != STEPS:
        raise ValueError("exact authorization is 240 measured physics rows per side")
    lock_path = project / "docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json"
    lock = json.loads(lock_path.read_text())
    sources = _sources(project)
    backend = args.ropesim_python.resolve(strict=True)
    for name, digest in lock["backend_files"].items():
        checked_bytes(backend / "ropesim" / name, digest)
    key = f"{args.side}.{_THUMB}"
    sdf_summary = checked_json(args.sdf_dir / "result.json", _SDF_SUMMARY_SHA)
    if (sdf_summary["rods_per_env"] != 1 or sdf_summary["isaac_launched"]
            or sdf_summary["simulation_steps"] != 0 or sdf_summary["contact_verified"]):
        raise ValueError("unexpected initial standalone SDF qualification")
    pad = sdf_summary["pads"][key]
    if (pad["frame"] != "original_pad_coincident_local_basis_m"
            or pad["rod_radius_m"] != RADIUS_M or not pad["closed_welded_hull"]):
        raise ValueError("unexpected locked pad metric frame or radius")
    path = args.sdf_dir / f"{key}.npz"
    checked_bytes(path, pad["sdf_file_sha256"])
    captured = args.capture_dir / "gpu"
    result = checked_json(captured / "result.json", _GPU_RESULT_SHA)
    row = checked_json(captured / f"{args.side}_{_THUMB}.json", lock["gpu_rows"][key])
    if (pad["gpu_row_sha256"] != lock["gpu_rows"][key]
            or row["clock"] != result["clock"] or not row["sdk_body"]["sdk_rigid_body_found"]):
        raise ValueError("SDF does not bind the actual GPU pad identity")
    if args.side == "right":
        if args.left_scheduler_receipt is not None:
            left_output_from_scheduler_receipt(args.left_scheduler_receipt, args.left_receipt_sha256)
        else:
            if args.left_evidence is None or (args.left_evidence / "exit_code").read_text().strip() != "0":
                raise ValueError("right-thumb launch requires left process exit 0")
            prior = json.loads((args.left_evidence / "native/result.json").read_text())
            marker = json.loads((args.left_evidence / "native/shutdown_complete.json").read_text())
            if prior["side"] != "left" or not prior["sample_validation_passed"] or not marker["close_returned"]:
                raise ValueError("right-thumb launch requires validated left contact and shutdown")
    contract = {"source_usd": sources[args.side], "side": args.side, "steps": STEPS,
                "baked_sdf_sha256": pad["sdf_file_sha256"], "native_hull_sha256": lock["gpu_rows"][key],
                "backend_files": lock["backend_files"], "thumb_link": _THUMB}
    return contract, path, row


def verify_thumb_ledger_identity(rows, *, side, body_path, slot):
    """Bind the measured pad, rod, and canonical finger12 command to this side."""
    action_index = {"left": 1, "right": 7}[side]
    if body_path != f"/World/DeformXThumb/{side}Hand/{_THUMB}" or slot != 0:
        raise ValueError("wrong measured pad identity")
    for row in rows:
        if (row["side"], row["body_path"], row["mesh_index"], row["rod_uid"]) != (
                side, body_path, slot, "rod0"):
            raise ValueError("thumb ledger has mismatched pad identity")
        action = np.asarray(row["canonical_action24"])
        moving = ZERO_STEPS < row["phase_step"] <= ZERO_STEPS + MOVE_STEPS
        expected = [action_index] if moving else []
        if (action.shape != (24,) or not np.isfinite(action).all()
                or np.flatnonzero(action).tolist() != expected):
            raise ValueError("thumb ledger has mismatched canonical action")
    return action_index


def _pose_matrix(p, q):
    transform = np.eye(4)
    transform[:3, :3] = rotation_wxyz(q)
    transform[:3, 3] = p
    return transform


def _collect_owned_references(ownership, phase):
    gc.collect()
    ownership.checkpoint(phase)


def _run(sim, args, contract, sdf_path, capture, ownership):
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from pxr import Usd, UsdPhysics
    from ropesim import RopeBatch, sdf
    from go2_pvcnn.control.dh116.deformx_dh116_colliders import read_pad_triangles
    from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
    from go2_pvcnn.control.dh116.runtime_mimic import install_hand_mimic, configure_mimic_compliance
    from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure
    from go2_pvcnn.control.dh116.tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S

    def host(value):
        return value.detach().cpu().numpy().astype(float, copy=True)

    device = torch.device(args.device)
    # Isolated calibration placement, preserving original joint/collision USD.
    start = np.array([.35, .35, .4, .4, .4, .4])
    path = f"/World/DeformXThumb/{args.side}Hand"
    config = ArticulationCfg(
        prim_path=path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=contract["source_usd"]["path"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True, enabled_self_collisions=False,
                solver_position_iteration_count=64, solver_velocity_iteration_count=16,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., .65), joint_pos=dict(zip(JOINTS, map(float, expand(start))))),
        actuators={
            "active": ImplicitActuatorCfg(joint_names_expr=list(ACTIVE), stiffness=10., damping=.2,
                                          effort_limit_sim=.15, velocity_limit_sim=ACTIVE_VELOCITY_LIMIT_RAD_S),
            "passive": ImplicitActuatorCfg(joint_names_expr=[j for j in JOINTS if j not in ACTIVE],
                                           stiffness=0., damping=0., effort_limit_sim=0., velocity_limit_sim=3.),
        },
    )
    hand = Articulation(config)
    stage = sim.stage
    install_hand_mimic(stage, path)
    configure_mimic_compliance(stage, path, 400., 1.)
    sim.reset()
    hand.reset()
    view = hand.root_physx_view
    composer = hand.permanent_wrench_composer
    ownership.watch("hand", hand, owner="probe", release="asset STOP callback", callbacks=True)
    ownership.watch("hand_tensor_view", view, owner="Articulation", release="asset STOP callback", gpu_view=True)
    ownership.watch("hand_composer", composer, owner="Articulation", release="exclusive reset", composer=True)
    if set(hand.joint_names) != set(JOINTS) or not hand.is_fixed_base:
        raise RuntimeError("unexpected original DH116 articulation contract")
    body = hand.body_names.index(_THUMB)
    base = hand.body_names.index("base_link")
    body_path = view.link_paths[0][body]
    if body_path != f"{path}/{_THUMB}":
        raise RuntimeError("actual PhysX body link identity differs")
    _, _, actual_geometry = read_pad_triangles(stage, body_path)
    if actual_geometry["geometry_sha256"] != capture["source"]["geometry_sha256"]:
        raise RuntimeError("runtime thumb collision source SHA differs from the locked SDF")
    active = [hand.joint_names.index(name) for name in ACTIVE]
    full = hand.data.default_joint_pos.clone()
    for name, value in zip(JOINTS, expand(start)):
        full[:, hand.joint_names.index(name)] = float(value)
    hand.write_joint_state_to_sim(full, torch.zeros_like(full))
    hand.set_joint_position_target(full[:, active], joint_ids=active)
    hand.update(0.0)

    def snapshot():
        data = hand.data
        return {
            "link": host(data.body_link_pos_w[0, body]), "link_q": host(data.body_link_quat_w[0, body]),
            "com": host(data.body_com_pos_w[0, body]), "com_q": host(data.body_com_quat_w[0, body]),
            "q": host(data.joint_pos[0]), "qd": host(data.joint_vel[0]),
            "link_velocity_w": host(data.body_link_vel_w[0, body]),
        }

    with np.load(sdf_path, allow_pickle=False) as packet:
        grid = sdf.SdfGrid(packet["values"].copy(), packet["origin"].copy(), float(packet["spacing"]))
    hull = capture["queries"][capture["source"]["collision_owner"]]["live_simulation_cooked"]["hulls"][0]
    vertices, faces = np.asarray(hull["vertices"]), np.asarray(hull["faces"])
    pose = snapshot()
    q0 = pose["q"][active]
    if np.max(np.abs(q0-start)) > 1e-5:
        raise RuntimeError("SDK did not read back planned initial joint state")
    base_pose = host(hand.data.body_link_pose_w[0, base])
    fk = HandPointFK(contract["source_usd"]["path"], args.side, base_pose)
    def thumb_pose(q):
        return next(value for key, value in fk.poses(q).items() if key.endswith("/" + _THUMB))
    lo, hi = active_limits()
    limits = np.stack((lo, hi), axis=-1)
    endpoint, nodes, planner = plan_thumb_approach(
        grid, vertices, faces, q0, thumb_pose, limits, _pose_matrix(pose["link"], pose["link_q"]),
    )
    pin = np.zeros(21, dtype=bool)
    pin[[0, -1]] = True
    rope = RopeBatch.from_positions(nodes[None].astype(np.float32), .065, 800., .018,
                                    DT/SUBSTEPS, 8, pinned=pin, deterministic=True)
    rope.set_gravity(np.zeros(3))
    rope.set_contact(radius=RADIUS_M, mu_s=.5, mu_k=.4, v_slip=.001, kappa=2.)
    ownership.watch("rope", rope, owner="probe", release="drop RopeBatch", gpu_view=True)
    def mesh_pose(state):
        return state["link"].astype(np.float32), state["link_q"][[1, 2, 3, 0]].astype(np.float32)
    pos, quat = mesh_pose(pose)
    slot = rope.add_mesh(*grid.as_args(), pos=pos, quat=quat)
    if slot != 0 or rope.num_envs != 1:
        raise RuntimeError("expected one explicitly registered pad on one continuous rod")
    rope.enable_contact_forces(True)
    collisions = [
        {"path": str(p.GetPath()), "physx_enabled": UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get(),
         "deformx_slot": slot if str(p.GetPath()) == actual_geometry["collision_owner"] else None}
        for p in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    write_json(args.output / "plan.json", {
        **contract, **planner, "body_index": body, "actual_body_path": body_path,
        "active_limits_rad": limits.tolist(), "q0": q0.tolist(), "endpoint": endpoint.tolist(),
        "joint_names": hand.joint_names, "initial_nodes_w": nodes.tolist(), "rod_uid": "rod0",
        "rod_radius_m": RADIUS_M, "calibration_pinned_node_ids": [0, 20],
        "registered_pad_slot": slot, "physx_self_collisions": False, "collision_inventory": collisions,
    })
    rows, previous = [], np.zeros(6)
    failures = []
    with (args.output / "physics.jsonl").open("x", encoding="utf-8") as log:
        for step in range(1, STEPS+1):
            before = snapshot()
            q_target, action = command_at_step(q0, endpoint, step=step, side=args.side)
            hand.set_joint_position_target(torch.as_tensor(q_target[None], dtype=torch.float32, device=device),
                                           joint_ids=active)
            composer.reset()
            f, t = (torch.as_tensor(v[None, None], dtype=torch.float32, device=device)
                    for v in (previous[:3], previous[3:]))
            composer.set_forces_and_torques(forces=f, torques=t, body_ids=[body], is_global=True)
            rot = rotation_wxyz(before["link_q"])
            applied = np.r_[rot @ host(composer.composed_force_as_torch[0, body]),
                            rot @ host(composer.composed_torque_as_torch[0, body])]
            hand.write_data_to_sim()
            sdk_target = host(view.get_dof_position_targets()[0])
            sim.step(render=False)
            hand.update(DT)
            after = snapshot()
            pos, quat = mesh_pose(after)
            rope.set_mesh_poses(slot, pos, quat, teleport=False)
            rope.step_free(DT, SUBSTEPS)
            node_pos = host(torch.as_tensor(rope.positions())[0])
            node_force = np.asarray(rope.contact_forces()[0], dtype=float)
            hand_wrench = np.asarray(rope.mesh_wrenches()[0, slot], dtype=float)
            hand_com_wrench = wrench_at_com(hand_wrench, after["link"], after["com"])
            force_on_rod = node_force.sum(0)
            failure = joint_safety_failure(after["q"][active], limits, np.asarray([
                after["q"][hand.joint_names.index(follower)] -
                (ratio * after["q"][hand.joint_names.index(master)] + offset)
                for follower, master, ratio, offset in MIMIC]))
            if np.linalg.norm(hand_wrench[:3]) > 30:
                failure = failure or "excess_force"
            if not np.allclose(sdk_target[active], q_target, atol=1e-7, rtol=0):
                failure = failure or "sdk_target_mismatch"
            row = {
                "phase_step": step, "sdk_step": int(sim.current_time_step_index), "sdk_time_s": float(sim.current_time),
                "rod_substep": step*SUBSTEPS, "dt_s": DT, "side": args.side, "rod_uid": "rod0",
                "body_path": body_path, "mesh_index": slot,
                "canonical_action24": action.tolist(), "sdk_joint_target_rad": sdk_target.tolist(),
                "pre_state": {k: v.tolist() for k, v in before.items()},
                "post_state": {k: v.tolist() for k, v in after.items()},
                "rod_on_hand_link_w": hand_wrench.tolist(), "rod_on_hand_com_w": hand_com_wrench.tolist(),
                "applied_wrench_com_w": applied.tolist(), "applied_source_phase_step": step-1,
                "wrench_write_calls": 1, "hand_on_rod_force_w": force_on_rod.tolist(),
                "hand_on_rod_torque_end_pose_estimate_about_link_w": np.cross(
                    node_pos - after["link"], node_force).sum(0).tolist(),
                "rod_nodes_w": node_pos.tolist(), "rod_node_contact_forces_w": node_force.tolist(),
                "contact_node_ids": np.flatnonzero(np.linalg.norm(node_force, axis=-1) > 1e-5).tolist(),
                "pair_force_residual_n": float(np.linalg.norm(force_on_rod+hand_wrench[:3])),
                "min_signed_distance_m": min_grid_distance(
                    grid, node_pos, _pose_matrix(after["link"], after["link_q"])),
                "unmeasured_articulation_joint_and_root_reactions": None, "failure": failure,
            }
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False)+"\n")
            if step % 16 == 0 or failure:
                log.flush()
                os.fsync(log.fileno())
            if failure:
                failures.append(failure)
                break
            previous = hand_com_wrench
        log.flush()
        os.fsync(log.fileno())
    composer.reset()
    torch.cuda.synchronize(device)
    summary = schedule_summary(rows)
    action_index = verify_thumb_ledger_identity(rows, side=args.side, body_path=body_path, slot=slot)
    summary.update(side=args.side, scope="one_thumb_original_articulation_calibration",
                   peak_force_n=max((np.linalg.norm(r["rod_on_hand_link_w"][:3]) for r in rows), default=0),
                   peak_torque_nm=max((np.linalg.norm(r["rod_on_hand_com_w"][3:]) for r in rows), default=0),
                   standalone_sdf_sha256=contract["baked_sdf_sha256"], rod_uid="rod0",
                   source_usd_sha256=contract["source_usd"]["sha256"],
                   native_hull_sha256=contract["native_hull_sha256"],
                   registered_pad_slot=slot, thumb_action_index=action_index,
                   body_path=body_path, failure=failures[0] if failures else None,
                   outer_exit_verified=False, grasp_qualified=False)
    # Match the qualified F3 ownership order: drop the probe-owned solver and
    # view references before the SDK STOP callback and App teardown.
    del snapshot, mesh_pose, thumb_pose, fk, f, t, rope, composer, view, hand
    _collect_owned_references(ownership, "owned hand and solver references dropped")
    return summary


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ropesim-python", type=Path, required=True)
    parser.add_argument("--sdf-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    left = parser.add_mutually_exclusive_group()
    left.add_argument("--left-evidence", type=Path)
    left.add_argument("--left-scheduler-receipt", type=Path)
    parser.add_argument("--left-receipt-sha256")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--execute-native", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    contract, sdf_path, capture = preflight(args, Path(__file__).resolve().parents[1])
    if not args.execute_native:
        print(json.dumps({"native_started": False, **contract}))
        return
    if not args.headless or args.enable_cameras or args.device != "cuda:0":
        parser.error("expected headless, cameras disabled and cuda:0")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.ropesim_python.resolve(strict=True)))
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock

    ownership = NativeOwnershipTrace(args.output / "ownership.jsonl")
    with native_instance_lock():
        app = AppLauncher(args, fast_shutdown=False).app
        faulthandler.enable(file=sys.__stderr__, all_threads=True)
        sim = None
        try:
            import carb.settings
            import isaaclab.sim as sim_utils
            from isaacsim.core.simulation_manager import SimulationManager

            if carb.settings.get_settings().get("/app/fastShutdown") is not False:
                raise RuntimeError("expected natural teardown")
            sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
                dt=DT, render_interval=1, device=args.device, gravity=(0., 0., 0.),
                physx=sim_utils.PhysxCfg(solver_type=0)))
            ownership.watch("simulation", sim, owner="probe", release="clear then STOP", callbacks=True)
            result = _run(sim, args, contract, sdf_path, capture, ownership)
            write_json(args.output / "result.json", result)
        except Exception:
            write_json(args.output / "failure.json", {"traceback": traceback.format_exc()})
            raise
        finally:
            from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import close_calibration_app

            try:
                if sim is not None:
                    import omni.timeline
                    type(sim).clear_instance()
                    sim = None
                    gc.collect()
                    timeline = omni.timeline.get_timeline_interface()
                    timeline.stop()
                    app.update()
                    del timeline
                    released = SimulationManager.get_physics_sim_view() is None
                    ownership.event("SDK STOP", manager_view_released=released)
                    if not released:
                        raise RuntimeError("owned SimulationManager view survived STOP")
                    _collect_owned_references(ownership, "before app close")
            finally:
                close_calibration_app(app, args.output)
                ownership.checkpoint("app.close returned, outer exit pending")


if __name__ == "__main__":
    main()
