"""IsaacLab smoke validation for the decoupled DH116 hand-only replay.

This scene intentionally has no Panda or approach controller.  The wire is a
rigid cylindrical proxy for contact plumbing; it is not a deformable Cosserat
validation.  A successful run only reports that replay actions, USD hands,
PhysX and contact sensors execute together.
"""
from __future__ import annotations

import argparse, json, time, traceback
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument("--replay", type=Path)
source_group.add_argument("--checkpoint", type=Path,
                          help="Run a fitted hand-only policy checkpoint using Isaac joint feedback")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps-per-action", type=int, default=20)
parser.add_argument("--max-actions", type=int, default=48,
                    help="Maximum replay rows for a bounded Isaac smoke run")
parser.add_argument("--episode-index", type=int, default=0,
                    help="Select one complete episode from the step-major teacher/student replay")
parser.add_argument("--joint-rate-limit", type=float, default=0.3,
                    help="Active joint target slew limit in rad/s")
parser.add_argument("--wire-z", type=float, default=0.46,
                    help="Initial center height of the rigid wire proxy in metres")
parser.add_argument("--hold-window-s", type=float, default=0.0,
                    help="Keep the Isaac window open after the bounded demo (display only)")
parser.add_argument("--candidate3-pose", type=Path,
                    help="Use candidate3 DH116 pose and reference wire/table layout")
parser.add_argument("--native-pose-summary", type=Path,
                    help="Use DH116 world roots/joints resolved by the native pose preview")
parser.add_argument("--left-grasp-roll-deg", type=float, default=76.0,
                    help="Roll native-pose left hand around its fixed thumb/index target axis")
parser.add_argument("--right-grasp-roll-deg", type=float, default=82.0,
                    help="Roll native-pose right hand around its fixed thumb/index target axis")
parser.add_argument("--pose-check-only", action="store_true",
                    help="Initialize and measure the pose without executing replay or policy actions")
parser.add_argument("--tabletop-sequence", action="store_true",
                    help="Run bounded approach, close, hold, and contact-gated lift on the tabletop wire")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("refusing to overwrite evidence")
args.output.mkdir(parents=True)
app = AppLauncher(args).app
result = {"passed": False, "physics_stepped": False, "arm_state_included": False,
          "task": "dh116_hand_only_grasp_lift", "wire_proxy": True,
          "failure_reason": None, "failure_action_index": None,
          "physics_steps": 0, "completed_actions": 0}
runtime_error = None
try:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import Usd, UsdGeom
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "assets/fixed_dual_panda_dh116_v1/manifest.json").read_text())
    sources = {(x["side"], x["kind"]): Path(x["path"]) for x in manifest["sources"]}
    from go2_pvcnn.control.dh116.teacher_student import LinearImitationPolicy
    from go2_pvcnn.control.dh116.hand_only_scene import (
        assess_table_clearance, candidate3_hand_only_layout, native_pose_hand_only_layout)
    candidate = None
    if args.candidate3_pose and args.native_pose_summary:
        raise ValueError("choose only one hand pose source")
    if args.candidate3_pose:
        candidate = candidate3_hand_only_layout(json.loads(args.candidate3_pose.read_text()))
        result["initial_pose_source"] = str(args.candidate3_pose)
        result["initial_pose_scope"] = "offline_candidate3_not_physics_qualified"
        result["candidate3_clearance_passed"] = False
    elif args.native_pose_summary:
        candidate = native_pose_hand_only_layout(
            json.loads(args.native_pose_summary.read_text()),
            roll_degrees={"left": args.left_grasp_roll_deg, "right": args.right_grasp_roll_deg})
        result["initial_pose_source"] = str(args.native_pose_summary)
        result["initial_pose_scope"] = "native_resolved_world_pose_physics_not_grasp_qualified"
    policy = None
    if args.checkpoint:
        policy = LinearImitationPolicy.load(args.checkpoint, expected_role="student")
        rows = [None] * min(args.max_actions, 24)
    else:
        all_rows = [json.loads(line) for line in args.replay.read_text().splitlines() if line.strip()]
        manifest_path = args.replay.parent / "manifest.json"
        episodes = int(json.loads(manifest_path.read_text()).get("episodes", 1)) if manifest_path.exists() else 1
        if not 0 <= args.episode_index < episodes:
            raise ValueError(f"episode-index must be in [0, {episodes})")
        rows = all_rows[args.episode_index::episodes][:args.max_actions]
        if not rows:
            raise ValueError("replay is empty")
        if any(len(row["action"]) != 12 for row in rows):
            raise ValueError("replay action must be 12D")
    dt = 1.0 / 240.0
    if args.joint_rate_limit <= 0:
        raise ValueError("joint-rate-limit must be positive")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
    if not args.headless:
        sim.set_camera_view(eye=(0.43, -1.05, 0.58), target=(0.43, 0.0, 0.34))
        sim_utils.DomeLightCfg(intensity=1800.0).func(
            "/World/HandOnlyLight", sim_utils.DomeLightCfg(intensity=1800.0))
    if candidate:
        table_cfg = sim_utils.CuboidCfg(size=candidate["table_size"])
        table_cfg.func("/World/Table", table_cfg, translation=candidate["table_center"])
    hands = []
    sensors = []
    for side, x in (("left", -0.22), ("right", 0.22)):
        source = sources[(side, "dh116")]
        cfg = ArticulationCfg(
            prim_path=f"/World/{side.title()}Hand",
            spawn=sim_utils.UsdFileCfg(usd_path=str(source), activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True, enabled_self_collisions=False,
                    solver_position_iteration_count=16, solver_velocity_iteration_count=4)),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(candidate["hand_roots"][side][:3]) if candidate else (x, 0.0, 0.24),
                rot=tuple(candidate["hand_roots"][side][3:]) if candidate else (0.0, 1.0, 0.0, 0.0),
                joint_pos=dict(zip(JOINTS, map(float, expand(candidate["active_joints"][side]
                    if candidate else np.zeros(6) + 0.2))))),
            actuators={
                "active": ImplicitActuatorCfg(joint_names_expr=list(ACTIVE), stiffness=1.0,
                    damping=0.05, effort_limit_sim=0.1, velocity_limit_sim=0.3),
                "passive": ImplicitActuatorCfg(joint_names_expr=[n for n in JOINTS if n not in ACTIVE],
                    stiffness=0.0, damping=0.0, effort_limit_sim=0.0, velocity_limit_sim=0.3)},
        )
        hands.append(Articulation(cfg))
        sensors.append(ContactSensor(ContactSensorCfg(prim_path=f"/World/{side.title()}Hand/.*", update_period=0.0)))
    wire_cfg = RigidObjectCfg(prim_path="/World/WireProxy", spawn=sim_utils.CylinderCfg(
        radius=0.008, height=0.5, axis="Y" if candidate else "X", activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.02)) if candidate else None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.12)),
        # Place the wire just above the neutral hand envelope; closure can
        # approach it without starting in penetration.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.43, 0.0, 0.308) if candidate
                                                 else (0.0, 0.0, args.wire_z)))
    wire = RigidObject(wire_cfg)
    wire_sensor = ContactSensor(ContactSensorCfg(prim_path="/World/WireProxy", update_period=0.0))
    sim.reset()
    for hand, sensor in zip(hands, sensors):
        hand.reset(); sensor.reset()
        reset_positions = hand.data.default_joint_pos.clone()
        if candidate:
            side = "left" if hand is hands[0] else "right"
            reset_ids = [hand.joint_names.index(name) for name in JOINTS]
            reset_active = candidate["active_joints"][side]
            reset_full = expand(reset_active)
            reset_positions[:, reset_ids] = torch.as_tensor(
                reset_full, device=args.device, dtype=torch.float32)[None, :]
        hand.write_joint_state_to_sim(reset_positions,
                                       torch.zeros_like(hand.data.default_joint_vel))
        reset_active_ids = [hand.joint_names.index(name) for name in ACTIVE]
        hand.set_joint_position_target(reset_positions[:, reset_active_ids],
                                       joint_ids=reset_active_ids)
        hand.write_data_to_sim()
    wire.reset(); wire_sensor.reset()
    active_ids = [[hand.joint_names.index(n) for n in ACTIVE] for hand in hands]
    if candidate:
        for hand in hands:
            hand.update(dt)
        wire.update(dt)
        table_top_z = candidate["table_center"][2] + candidate["table_size"][2] / 2.0
        stage = omni.usd.get_context().get_stage()
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        collision_bottoms = {}
        collision_counts = {}
        collision_bound_samples = {}
        for side in ("left", "right"):
            prefix = f"/World/{side.title()}Hand/"
            bottoms = []
            root_prim = stage.GetPrimAtPath(prefix.rstrip("/"))
            for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
                path = str(prim.GetPath())
                if (not path.startswith(prefix) or "/collisions/" not in path
                        or not prim.IsA(UsdGeom.Mesh)):
                    continue
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                if points is None or len(points) == 0:
                    continue
                local_points = np.asarray(points, dtype=np.float64)
                local_min, local_max = local_points.min(axis=0), local_points.max(axis=0)
                transform = xform_cache.GetLocalToWorldTransform(prim)
                matrix = np.asarray([[float(transform[row][col]) for col in range(4)]
                                     for row in range(4)], dtype=np.float64)
                world_h = np.concatenate((local_points, np.ones((len(local_points), 1))), axis=1) @ matrix
                world_min_z = float(world_h[:, 2].min())
                if np.isfinite(world_min_z) and abs(world_min_z) < 10.0:
                    bottoms.append(world_min_z)
                if len(collision_bound_samples.get(side, [])) < 3:
                    collision_bound_samples.setdefault(side, []).append({
                        "path": path, "point_count": 0 if points is None else len(points),
                        "world_min_z_from_vertices": world_min_z,
                        "local_min": local_min.tolist(),
                        "local_max": local_max.tolist(),
                    })
            collision_bottoms[side] = bottoms
            collision_counts[side] = len(bottoms)
        result["table_clearance_measurement"] = "usd_collision_world_aabb_min_z"
        result["collision_shape_counts"] = collision_counts
        result["collision_bound_samples"] = collision_bound_samples
        clearance_report = assess_table_clearance(collision_bottoms, table_top_z, minimum_clearance_m=0.002)
        result["table_clearance"] = clearance_report
        result["minimum_hand_body_z_m"] = clearance_report["minimum_body_z_m"]
        result["table_top_z_m"] = table_top_z
        result["minimum_table_clearance_m"] = clearance_report["minimum_clearance_m"]
        result["table_clearance_passed"] = clearance_report["passed"]
        if not clearance_report["passed"]:
            result["failure_reason"] = (
                "candidate_pose_table_penetration"
                if clearance_report["failure_reason"] == "tabletop_penetration"
                else clearance_report["failure_reason"])
            result["numerically_stable"] = True
            result["contact_observed"] = False
            result["physics_steps"] = 0
        wire_bottom_z = float(wire.data.root_pos_w[0, 2].item()) - candidate["wire_radius_m"]
        result["wire_bottom_z_m"] = wire_bottom_z
        result["wire_table_clearance_m"] = wire_bottom_z - table_top_z
        if result["wire_table_clearance_m"] < -1e-4:
            result["failure_reason"] = "wire_initially_below_table_surface"
        result["table_clearance_passed"] = not bool(result["failure_reason"])
        result["physics_steps"] = 0
    active_lower, active_upper = active_limits()
    contact_peak = [0.0, 0.0]
    command = [candidate["active_joints"][side].copy() if candidate else
               np.full(6, 0.2, dtype=np.float32) for side in ("left", "right")]
    wire_z_start = float(wire.data.root_pos_w[0, 2].item())
    if candidate:
        result["wire_visual"] = "single_candidate3_reference_orange"
    previous_policy_action = np.full(12, 0.2, dtype=np.float32)
    action_rows = () if (args.pose_check_only or args.tabletop_sequence
                         or not result.get("table_clearance_passed", True)) else rows
    for action_index, row in enumerate(action_rows):
        if policy is not None:
            observed = np.concatenate([
                hand.data.joint_pos[0, active_ids[i]].detach().cpu().numpy() for i, hand in enumerate(hands)
            ]).astype(np.float32)
            grasp_progress = min(1.0, (action_index + 1) / 12.0)
            observation = np.concatenate((observed, previous_policy_action, [grasp_progress]))
            action = policy.predict(observation)
            previous_policy_action = action.copy()
        else:
            action = np.asarray(row["action"], dtype=np.float32)
        for _ in range(args.steps_per_action):
            for i, hand in enumerate(hands):
                desired = action[i * 6:(i + 1) * 6]
                max_delta = args.joint_rate_limit * dt
                command[i] += np.clip(desired - command[i], -max_delta, max_delta)
                # USD joint order is not assumed to equal the contract order.
                # Drive the six active joints by name; passive followers are
                # supplied by the USD's affine mimic joints.
                target = torch.as_tensor(command[i], device=args.device, dtype=torch.float32)[None]
                hand.set_joint_position_target(target, joint_ids=active_ids[i])
                hand.write_data_to_sim()
            sim.step(render=False)
            result["physics_steps"] += 1
            for i, (hand, sensor) in enumerate(zip(hands, sensors)):
                hand.update(dt); sensor.update(dt, force_recompute=True)
                if not bool(torch.isfinite(hand.data.joint_pos).all().item() and
                            torch.isfinite(hand.data.joint_vel).all().item() and
                            torch.isfinite(hand.data.body_pos_w).all().item()):
                    result["failure_reason"] = "non_finite_hand_state"
                    result["failure_action_index"] = action_index
                    break
                force = float(torch.linalg.vector_norm(sensor.data.net_forces_w[0]).item())
                contact_peak[i] = max(contact_peak[i], force)
            if result["failure_reason"]:
                break
            if candidate and action_index >= 12:
                lift_fraction = min((action_index - 11) / 12.0, 1.0)
                for i, hand in enumerate(hands):
                    root_state = hand.data.root_state_w.clone()
                    root_state[:, 2] = candidate["hand_roots"][("left", "right")[i]][2] + 0.03 * lift_fraction
                    hand.write_root_pose_to_sim(root_state[:, :7])
            wire.update(dt); wire_sensor.update(dt, force_recompute=True)
            if not args.headless:
                app.update()
            result["physics_stepped"] = True
            current_wire_z = float(wire.data.root_pos_w[0, 2].item())
            if not np.isfinite(current_wire_z) or not np.isfinite(contact_peak).all():
                result["failure_reason"] = "non_finite_wire_or_contact_state"
                result["failure_action_index"] = action_index
                break
            if current_wire_z < -0.1 or current_wire_z > 1.5 or max(contact_peak) > 100.0:
                result["failure_reason"] = "contact_impulse_or_wire_escape"
                result["failure_action_index"] = action_index
                break
        if result["failure_reason"]:
            break
        result["completed_actions"] += 1
    wire_z = float(wire.data.root_pos_w[0, 2].item())
    hand_finite = all(bool(torch.isfinite(hand.data.joint_pos).all().item() and
                           torch.isfinite(hand.data.joint_vel).all().item() and
                           torch.isfinite(hand.data.body_pos_w).all().item()) for hand in hands)
    stable = bool(np.isfinite(wire_z) and abs(wire_z) < 2.0 and hand_finite and
                  np.isfinite(contact_peak).all() and max(contact_peak) < 100.0)
    contact_observed = bool(max(contact_peak) > 0.01)
    result.update({"passed": bool(stable and contact_observed and result["failure_reason"] is None), "replay_rows": len(rows),
                   "contact_peak_n": contact_peak, "contact_observed": contact_observed,
                   "wire_z_m": wire_z, "numerically_stable": stable,
                   "wire_z_start_m": wire_z_start,
                   "wire_lift_delta_m": wire_z - wire_z_start,
                   "joint_rate_limit_rad_s": args.joint_rate_limit,
                   "contact_sensor_bodies": [s.body_names for s in sensors],
                   "hand_state_finite": hand_finite,
                   "hand_body_positions_w": [
                       {name: ([float(x) if np.isfinite(float(x)) else None for x in hand.data.body_pos_w[0, idx].detach().cpu().tolist()])
                        for idx, name in enumerate(hand.body_names)} for hand in hands]})
except Exception as exc:
    result["failure_reason"] = f"runtime_exception:{type(exc).__name__}:{exc}"
    result["runtime_traceback"] = traceback.format_exc()
finally:
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DH116_HAND_ONLY_WIRE", json.dumps(result), flush=True)
    if args.hold_window_s < 0 or not np.isfinite(args.hold_window_s):
        raise ValueError("hold-window-s must be finite and non-negative")
    if args.hold_window_s and not args.headless:
        deadline = time.monotonic() + args.hold_window_s
        while time.monotonic() < deadline:
            app.update()
    app.close(wait_for_replicator=False, skip_cleanup=True)
