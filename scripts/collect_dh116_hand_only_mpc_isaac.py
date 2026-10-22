#!/usr/bin/env python3
"""Collect measured DH116 hand-only MPC trials in an Isaac rigid-wire probe.

This is deliberately arm-free and starts in grasp.  The proxy is rigid and
gravity is disabled, so the run is useful for contact/action feedback only; it
does not qualify lifting a flexible, loaded wire.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--student-checkpoint", type=Path,
                    help="Execute this trained 25D hand-only student; MPC remains logged as the reference")
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--actions-per-episode", type=int, default=24)
parser.add_argument("--steps-per-action", type=int, default=20)
parser.add_argument("--wire-z-min", type=float, default=0.40)
parser.add_argument("--wire-z-max", type=float, default=0.44)
parser.add_argument("--hand-root-z", type=float, default=None,
                    help="Optional fixed palm height; otherwise sweep high-to-low")
parser.add_argument("--hand-root-z-min", type=float, default=0.55)
parser.add_argument("--hand-root-z-max", type=float, default=0.62)
parser.add_argument("--joint-rate-limit", type=float, default=0.3)
parser.add_argument("--abort-force-n", type=float, default=30.0,
                    help="Numerical contact-force abort cap for this uncalibrated proxy scene")
parser.add_argument("--contact-stiffness-n-per-m", type=float, default=1000.0,
                    help="Simulation-only compliant contact stiffness for distal pads and wire")
parser.add_argument("--contact-damping-n-s-per-m", type=float, default=25.0,
                    help="Simulation-only compliant contact damping for distal pads and wire")
parser.add_argument("--joint-initial-min", type=float, default=0.2)
parser.add_argument("--joint-initial-max", type=float, default=0.2)
parser.add_argument("--grasp-fraction-min", type=float, default=0.45)
parser.add_argument("--grasp-fraction-max", type=float, default=0.65)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("refusing to overwrite evidence")
if args.episodes <= 0 or args.actions_per_episode < 2 or args.steps_per_action <= 0:
    raise ValueError("episodes and steps must be positive; at least two actions are required for grasp and lift")
if not 0 < args.wire_z_min <= args.wire_z_max < 1.5 or args.joint_rate_limit <= 0 or args.abort_force_n <= 0:
    raise ValueError("invalid wire height sweep or joint-rate limit")
if not 0 <= args.joint_initial_min <= args.joint_initial_max <= 0.8:
    raise ValueError("invalid initial active-joint sweep")
if not 0 <= args.grasp_fraction_min <= args.grasp_fraction_max <= 0.75:
    raise ValueError("invalid MPC grasp fraction sweep")
if not 0 < args.hand_root_z_min <= args.hand_root_z_max < 1.5:
    raise ValueError("invalid hand-root height sweep")
if args.hand_root_z is not None and not 0 < args.hand_root_z < 1.5:
    raise ValueError("invalid fixed hand-root height")
if args.contact_stiffness_n_per_m <= 0 or args.contact_damping_n_s_per_m <= 0:
    raise ValueError("simulation contact stiffness and damping must be positive")
if args.student_checkpoint is not None and not args.student_checkpoint.is_file():
    raise ValueError("student checkpoint does not exist")
args.output.mkdir(parents=True)
app = AppLauncher(args).app
summary = {"task": "dh116_hand_only_grasp_lift", "source": "parallel_joint_mpc",
           "planner_execution": "preplan_once_then_replay_by_measured_progress",
           "action_source": "student" if args.student_checkpoint is not None else "parallel_mpc",
           "student_checkpoint": None if args.student_checkpoint is None else str(args.student_checkpoint),
           "student_checkpoint_sha256": None,
           "arm_state_included": False, "wire_proxy": "rigid_cylinder_on_kinematic_table",
           "contact_material": "simulation_only_compliant_probe",
           "contact_stiffness_n_per_m": args.contact_stiffness_n_per_m,
           "contact_damping_n_s_per_m": args.contact_damping_n_s_per_m,
           "deformable_wire_validated": False, "episodes": [], "passed": False,
           "stage": "starting"}
try:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material
    from isaaclab.sim.utils.prims import bind_physics_material
    from go2_pvcnn.control.dh116.hand_only_env import HandOnlyState
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, active_limits, expand
    from go2_pvcnn.control.dh116.mpc_feedback import (
        advance_progress, initial_active_action, peak_body_force, to_numpy,
        top_down_hand_pose, trajectory_action_index,
    )
    from go2_pvcnn.control.dh116.hand_collisions import disable_nonpad_collisions
    from go2_pvcnn.control.dh116.contact_model import (
        compliant_contact_parameters, table_center_z_for_wire,
    )
    from go2_pvcnn.control.dh116.parallel_mpc import ParallelJointMPC
    from go2_pvcnn.control.dh116.teacher_student import (
        LinearImitationPolicy, make_transition, qualify_proxy_lift, score_physics_outcome, write_replay,
    )

    student_policy = None
    if args.student_checkpoint is not None:
        student_policy = LinearImitationPolicy.load(args.student_checkpoint, expected_role="student")
        summary["student_checkpoint_sha256"] = hashlib.sha256(args.student_checkpoint.read_bytes()).hexdigest()

    root = Path(__file__).resolve().parents[1]
    summary["stage"] = "loading_manifest"
    manifest = json.loads((root / "assets/fixed_dual_panda_dh116_v1/manifest.json").read_text())
    sources = {(x["side"], x["kind"]): Path(x["path"]) for x in manifest["sources"]}
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 240.0, device=args.device))
    summary["stage"] = "creating_scene"
    hands, sensors = [], []
    collision_filter_report = {}
    for side, x in (("left", -0.22), ("right", 0.22)):
        root_pose = top_down_hand_pose(x, args.hand_root_z or args.hand_root_z_max)
        pad_material_paths = [
            f"{link}/collisions/{link}/node_STL_BINARY_"
            for link in ("finger14_link", "finger23_link", "finger33_link",
                         "finger43_link", "finger53_link")
        ]
        def spawn_with_pad_only_collisions(prim_path, cfg, translation=None, orientation=None, **kwargs):
            prim = spawn_from_usd(prim_path, cfg, translation=translation,
                                  orientation=orientation, **kwargs)
            # Collision meshes are referenced instance branches in the source
            # USD. De-instance only the pad branches before binding a runtime
            # material so the factory asset remains untouched.
            for link in ("finger14_link", "finger23_link", "finger33_link",
                         "finger43_link", "finger53_link"):
                branch = sim.stage.GetPrimAtPath(f"{prim_path}/{link}/collisions")
                if branch and branch.IsValid() and branch.IsInstance():
                    branch.SetInstanceable(False)
            material_cfg = sim_utils.RigidBodyMaterialCfg(**compliant_contact_parameters(
                args.contact_stiffness_n_per_m, args.contact_damping_n_s_per_m))
            material_path = f"/World/PhysicsMaterials/{side.title()}PadProbe"
            if not sim.stage.GetPrimAtPath("/World/PhysicsMaterials").IsValid():
                sim.stage.DefinePrim("/World/PhysicsMaterials", "Scope")
            spawn_rigid_body_material(material_path, material_cfg)
            for relative_path in pad_material_paths:
                bind_physics_material(f"{prim_path}/{relative_path}", material_path,
                                      stage=sim.stage)
            disabled = disable_nonpad_collisions(sim.stage, str(prim.GetPath()))
            collision_filter_report[side] = {"disabled_shapes": len(disabled),
                                               "disabled_paths": list(disabled)}
            return prim
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=str(sources[(side, "dh116")]), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,
                enabled_self_collisions=False, solver_position_iteration_count=16,
                solver_velocity_iteration_count=4))
        spawn_cfg.func = spawn_with_pad_only_collisions
        cfg = ArticulationCfg(
            prim_path=f"/World/{side.title()}Hand",
            spawn=spawn_cfg,
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(root_pose[:3]), rot=tuple(root_pose[3:]),
                joint_pos=dict(zip(JOINTS, map(float, expand(np.zeros(6) + 0.2))))),
            actuators={
                "active": ImplicitActuatorCfg(joint_names_expr=list(ACTIVE), stiffness=1.0,
                    damping=0.05, effort_limit_sim=0.1, velocity_limit_sim=0.3),
                "passive": ImplicitActuatorCfg(joint_names_expr=[n for n in JOINTS if n not in ACTIVE],
                    stiffness=0.0, damping=0.0, effort_limit_sim=0.0, velocity_limit_sim=0.3)})
        hands.append(Articulation(cfg))
        sensors.append(ContactSensor(ContactSensorCfg(prim_path=f"/World/{side.title()}Hand/.*", update_period=0.0)))
    wire_radius_m = 0.008
    table_thickness_m = 0.02
    table = RigidObject(RigidObjectCfg(prim_path="/World/TableProxy", spawn=sim_utils.CuboidCfg(
        size=(0.8, 0.5, table_thickness_m),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0,
            table_center_z_for_wire(args.wire_z_min, wire_radius_m, table_thickness_m)))))
    wire = RigidObject(RigidObjectCfg(prim_path="/World/WireProxy", spawn=sim_utils.CylinderCfg(
        radius=wire_radius_m, height=0.5, axis="X", activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.12)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args.wire_z_max))))
    wire_sensor = ContactSensor(ContactSensorCfg(prim_path="/World/WireProxy", update_period=0.0))
    sim.reset()
    summary["stage"] = "scene_reset"
    active_ids = [[hand.joint_names.index(n) for n in ACTIVE] for hand in hands]
    lower, upper = active_limits()
    summary["stage"] = "ready_for_rollout"
    transitions = []

    for episode in range(args.episodes):
        episode_transition_start = len(transitions)
        wire_start = float(np.linspace(args.wire_z_min, args.wire_z_max, args.episodes)[episode])
        hand_root_z = (args.hand_root_z if args.hand_root_z is not None else
                       float(np.linspace(args.hand_root_z_max, args.hand_root_z_min, args.episodes)[episode]))
        initial_joint = float(np.linspace(args.joint_initial_min, args.joint_initial_max,
                                          args.episodes)[episode])
        grasp_fraction = float(np.linspace(args.grasp_fraction_min, args.grasp_fraction_max,
                                           args.episodes)[episode])
        planner = ParallelJointMPC(device=args.device, grasp_fraction=grasp_fraction)
        for (hand, sensor), x in zip(zip(hands, sensors), (-0.22, 0.22)):
            full_joint_values = dict(zip(JOINTS, map(float, expand(np.full(6, initial_joint)))))
            joint_ids = [hand.joint_names.index(name) for name in JOINTS]
            initial_positions = hand.data.default_joint_pos.clone()
            initial_positions[:, joint_ids] = torch.as_tensor(
                [full_joint_values[name] for name in JOINTS], device=args.device, dtype=torch.float32)
            hand.write_joint_state_to_sim(initial_positions, torch.zeros_like(hand.data.default_joint_vel))
            hand.reset(); sensor.reset()
            root_pose = top_down_hand_pose(x, hand_root_z)
            hand.write_root_pose_to_sim(torch.as_tensor(root_pose, device=args.device)[None])
        table.reset()
        table_center_z = table_center_z_for_wire(wire_start, wire_radius_m, table_thickness_m)
        table.write_root_pose_to_sim(torch.tensor([[0.0, 0.0, table_center_z,
                                                    1.0, 0.0, 0.0, 0.0]], device=args.device))
        table.write_root_velocity_to_sim(torch.zeros((1, 6), device=args.device))
        wire.reset(); wire_sensor.reset()
        wire.write_root_pose_to_sim(torch.tensor([[0.0, 0.0, wire_start, 1.0, 0.0, 0.0, 0.0]],
                                                 device=args.device))
        wire.write_root_velocity_to_sim(torch.zeros((1, 6), device=args.device))
        previous = initial_active_action(initial_joint, hand_count=len(hands))
        commands = [previous[index * 6:(index + 1) * 6].copy()
                    for index in range(len(hands))]
        grasp_progress = lift_progress = 0.0
        contact_latched = False
        contact_peak = 0.0
        contact_peak_by_body = {side: {"body": None, "force_n": 0.0}
                                for side in ("left", "right")}
        abort_reason = None
        successful_action = False
        # Separate initial interpenetration from MPC-driven motion.  Any
        # contact already over the cap is evidence against this pose, not an
        # action outcome.
        sim.step(render=False)
        for side, hand, sensor in zip(("left", "right"), hands, sensors):
            hand.update(1.0 / 240.0); sensor.update(1.0 / 240.0, force_recompute=True)
            body_name, body_force = peak_body_force(sensor.body_names,
                sensor.data.net_forces_w[0].detach().cpu().numpy())
            contact_peak = max(contact_peak, body_force)
            if body_force > contact_peak_by_body[side]["force_n"]:
                contact_peak_by_body[side] = {"body": body_name, "force_n": body_force}
        wire.update(1.0 / 240.0); wire_sensor.update(1.0 / 240.0, force_recompute=True)
        wire_z_after_settle = float(wire.data.root_pos_w[0, 2].item())
        distal_pad_poses = {}
        for side, hand in zip(("left", "right"), hands):
            distal_pad_poses[side] = {}
            for link_name in ("finger14_link", "finger23_link", "finger33_link",
                              "finger43_link", "finger53_link"):
                body_index = hand.body_names.index(link_name)
                distal_pad_poses[side][link_name] = {
                    "position_w_m": hand.data.body_pos_w[0, body_index].detach().cpu().tolist(),
                    "orientation_w_wxyz": hand.data.body_quat_w[0, body_index].detach().cpu().tolist(),
                }
        if contact_peak > args.abort_force_n or abs(wire_z_after_settle - wire_start) > 0.05:
            abort_reason = "initial_pose_contact_or_wire_displacement"
        grasp_steps = max(1, args.actions_per_episode // 2)
        lift_steps = args.actions_per_episode - grasp_steps
        planned_trajectory = None
        if abort_reason is None:
            initial_observed = np.concatenate([
                hand.data.joint_pos[0, ids].detach().cpu().numpy()
                for hand, ids in zip(hands, active_ids)
            ]).astype(np.float32)
            initial_poses = np.stack([
                np.concatenate((hand.data.root_pos_w[0].detach().cpu().numpy(),
                                hand.data.root_quat_w[0].detach().cpu().numpy()))
                for hand in hands
            ]).astype(np.float32)
            initial_state = HandOnlyState(
                initial_poses, initial_observed.reshape(2, 6),
                np.zeros(2, dtype=np.float32), previous.copy(),
            )
            planned_trajectory, _plan_diagnostics = planner.plan_trajectory(
                initial_state, grasp_steps=grasp_steps, lift_steps=lift_steps,
                previous_action=previous.copy(),
            )
            planned_trajectory = to_numpy(planned_trajectory).astype(np.float32, copy=False)
            summary["planner_execution"] = "preplanned_once_per_episode"
            summary["planned_trajectory_shape"] = list(planned_trajectory.shape)
        action_index = -1
        for action_index in range(args.actions_per_episode) if abort_reason is None else ():
            observed = np.concatenate([hand.data.joint_pos[0, ids].detach().cpu().numpy()
                                       for hand, ids in zip(hands, active_ids)]).astype(np.float32)
            poses = np.stack([np.concatenate((hand.data.root_pos_w[0].detach().cpu().numpy(),
                                               hand.data.root_quat_w[0].detach().cpu().numpy()))
                              for hand in hands]).astype(np.float32)
            obs_student = np.concatenate((observed, previous, [grasp_progress])).astype(np.float32)
            obs_privileged = np.concatenate((obs_student, [lift_progress])).astype(np.float32)
            trajectory_index = trajectory_action_index(
                grasp_progress, lift_progress,
                grasp_steps=grasp_steps, lift_steps=lift_steps,
            )
            mpc_action = planned_trajectory[0, trajectory_index].copy()
            action = (mpc_action if student_policy is None else
                      student_policy.predict(obs_student).astype(np.float32, copy=False))
            force_this_action = 0.0
            for _ in range(args.steps_per_action):
                for index, hand in enumerate(hands):
                    desired = action[index * 6:(index + 1) * 6]
                    commands[index] += np.clip(desired - commands[index],
                                               -args.joint_rate_limit / 240.0,
                                               args.joint_rate_limit / 240.0)
                    hand.set_joint_position_target(torch.as_tensor(commands[index], device=args.device)[None],
                                                  joint_ids=active_ids[index])
                    hand.write_data_to_sim()
                sim.step(render=False)
                for side, hand, sensor in zip(("left", "right"), hands, sensors):
                    hand.update(1.0 / 240.0); sensor.update(1.0 / 240.0, force_recompute=True)
                    body_name, body_force = peak_body_force(sensor.body_names,
                        sensor.data.net_forces_w[0].detach().cpu().numpy())
                    force_this_action = max(force_this_action, body_force)
                    if body_force > contact_peak_by_body[side]["force_n"]:
                        contact_peak_by_body[side] = {"body": body_name, "force_n": body_force}
                wire.update(1.0 / 240.0); wire_sensor.update(1.0 / 240.0, force_recompute=True)
                z = float(wire.data.root_pos_w[0, 2].item())
                if not np.isfinite(z) or not np.isfinite(force_this_action):
                    abort_reason = "non_finite_physics_state"
                    break
                if force_this_action > args.abort_force_n or z < -0.1 or z > 1.5:
                    abort_reason = "contact_impulse_or_wire_escape"
                    break
            contact_peak = max(contact_peak, force_this_action)
            z = float(wire.data.root_pos_w[0, 2].item())
            success, reward = score_physics_outcome(contact_force_n=contact_peak,
                                                     wire_lift_m=z - wire_start)
            transitions.append(make_transition(
                obs_student=obs_student, obs_teacher_privileged=obs_privileged,
                previous_action=previous, mpc_action=mpc_action, action=action, reward=reward,
                done=abort_reason is not None or action_index == args.actions_per_episode - 1,
                phase="grasp" if lift_progress == 0 else "lift",
                contact_force_n=force_this_action, wire_lift_m=z - wire_start,
                physics_valid=abort_reason is None and contact_peak <= args.abort_force_n))
            successful_action = successful_action or success
            if abort_reason:
                break
            contact_latched = contact_latched or force_this_action >= 1.0
            grasp_progress, lift_progress = advance_progress(
                grasp_progress, lift_progress, contact_force_n=force_this_action,
                contact_latched=contact_latched)
            previous = action.copy()
        wire_end = float(wire.data.root_pos_w[0, 2].item())
        episode_success = qualify_proxy_lift(contact_peak_n=contact_peak,
            wire_lift_m=wire_end - wire_start, abort_reason=abort_reason,
            max_contact_force_n=args.abort_force_n)
        if abort_reason:
            from dataclasses import replace
            for row_index in range(episode_transition_start, len(transitions)):
                transitions[row_index] = replace(transitions[row_index], reward=-1.0, physics_valid=False)
        summary["episodes"].append({"episode": episode, "wire_z_start_m": wire_start,
            "wire_z_end_m": wire_end, "wire_lift_delta_m": wire_end - wire_start,
            "wire_z_after_settle_m": wire_z_after_settle,
            "table_top_z_m": table_center_z + table_thickness_m / 2.0,
            "contact_peak_n": contact_peak, "actions_completed": action_index + 1,
            "contact_peak_by_body": contact_peak_by_body,
            "distal_pad_poses_after_settle": distal_pad_poses,
            "abort_reason": abort_reason, "proxy_lift_success": episode_success,
            "initial_active_joint_rad": initial_joint,
            "hand_root_z_m": hand_root_z,
            "collision_filter": collision_filter_report,
            "mpc_grasp_fraction": grasp_fraction,
            "contact_latched": contact_latched,
            "final_grasp_progress": grasp_progress, "final_lift_progress": lift_progress})
        # A contact impulse can corrupt PhysX state for subsequent trials.
        # Preserve this episode's failed MPC data and restart Isaac next run.
        if abort_reason:
            break

    summary["passed"] = any(ep["proxy_lift_success"] and not ep["abort_reason"]
                            for ep in summary["episodes"])
    replay_role = "student" if student_policy is not None else "teacher"
    replay_name = "student_physics_feedback.jsonl" if student_policy is not None else "mpc_physics_feedback.jsonl"
    summary["replay"] = write_replay(args.output / replay_name, transitions, role=replay_role)
    summary["thresholds"] = {"abort_force_n": args.abort_force_n, "proxy_lift_m": 0.02}
    summary["stage"] = "complete"
except BaseException as exc:
    import traceback
    summary["runtime_error"] = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()
finally:
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DH116_HAND_ONLY_MPC_FEEDBACK", json.dumps(summary), flush=True)
    app.close(wait_for_replicator=False, skip_cleanup=True)

if summary.get("runtime_error"):
    raise RuntimeError(summary["runtime_error"])
