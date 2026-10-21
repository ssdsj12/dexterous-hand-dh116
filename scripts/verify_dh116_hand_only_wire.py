"""IsaacLab smoke validation for the decoupled DH116 hand-only replay.

This scene intentionally has no Panda or approach controller.  The wire is a
rigid cylindrical proxy for contact plumbing; it is not a deformable Cosserat
validation.  A successful run only reports that replay actions, USD hands,
PhysX and contact sensors execute together.
"""
from __future__ import annotations

import argparse, json
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
try:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "assets/fixed_dual_panda_dh116_v1/manifest.json").read_text())
    sources = {(x["side"], x["kind"]): Path(x["path"]) for x in manifest["sources"]}
    policy = None
    if args.checkpoint:
        policy = json.loads(args.checkpoint.read_text())
        if policy.get("role") != "student" or policy.get("output_dim") != 12 or policy.get("arm_state_included") is not False:
            raise ValueError("checkpoint must be a 12D student model with no arm state")
        weights = np.asarray(policy["weights"], dtype=np.float32)
        if weights.shape != (int(policy["input_dim"]) + 1, 12) or policy["input_dim"] != 25:
            raise ValueError("unexpected student checkpoint dimensions")
        rows = [None] * args.max_actions
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
            init_state=ArticulationCfg.InitialStateCfg(pos=(x, 0.0, 0.24),
                joint_pos=dict(zip(JOINTS, map(float, expand(np.zeros(6) + 0.2))))),
            actuators={
                "active": ImplicitActuatorCfg(joint_names_expr=list(ACTIVE), stiffness=1.0,
                    damping=0.05, effort_limit_sim=0.1, velocity_limit_sim=0.3),
                "passive": ImplicitActuatorCfg(joint_names_expr=[n for n in JOINTS if n not in ACTIVE],
                    stiffness=0.0, damping=0.0, effort_limit_sim=0.0, velocity_limit_sim=0.3)},
        )
        hands.append(Articulation(cfg))
        sensors.append(ContactSensor(ContactSensorCfg(prim_path=f"/World/{side.title()}Hand/.*", update_period=0.0)))
    wire_cfg = RigidObjectCfg(prim_path="/World/WireProxy", spawn=sim_utils.CylinderCfg(
        radius=0.008, height=0.5, axis="X", activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.12)),
        # Place the wire just above the neutral hand envelope; closure can
        # approach it without starting in penetration.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args.wire_z)))
    wire = RigidObject(wire_cfg)
    wire_sensor = ContactSensor(ContactSensorCfg(prim_path="/World/WireProxy", update_period=0.0))
    sim.reset()
    for hand, sensor in zip(hands, sensors):
        hand.write_joint_state_to_sim(hand.data.default_joint_pos.clone(),
                                       torch.zeros_like(hand.data.default_joint_vel))
        hand.reset(); sensor.reset()
    wire.reset(); wire_sensor.reset()
    active_ids = [[hand.joint_names.index(n) for n in ACTIVE] for hand in hands]
    active_lower, active_upper = active_limits()
    contact_peak = [0.0, 0.0]
    command = [np.full(6, 0.2, dtype=np.float32), np.full(6, 0.2, dtype=np.float32)]
    wire_z_start = float(wire.data.root_pos_w[0, 2].item())
    previous_policy_action = np.full(12, 0.2, dtype=np.float32)
    for action_index, row in enumerate(rows):
        if policy is not None:
            observed = np.concatenate([
                hand.data.joint_pos[0, active_ids[i]].detach().cpu().numpy() for i, hand in enumerate(hands)
            ]).astype(np.float32)
            grasp_progress = min(1.0, (action_index + 1) / max(1, args.max_actions // 2))
            observation = np.concatenate((observed, previous_policy_action, [grasp_progress]))
            action = np.concatenate((observation, [1.0])) @ weights
            action = np.clip(action, active_lower.tolist() * 2, active_upper.tolist() * 2).astype(np.float32)
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
            wire.update(dt); wire_sensor.update(dt, force_recompute=True)
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
finally:
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DH116_HAND_ONLY_WIRE", json.dumps(result), flush=True)
    app.close(wait_for_replicator=False, skip_cleanup=True)
