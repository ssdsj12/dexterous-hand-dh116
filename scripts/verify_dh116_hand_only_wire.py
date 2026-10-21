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
parser.add_argument("--replay", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps-per-action", type=int, default=20)
parser.add_argument("--max-actions", type=int, default=48,
                    help="Maximum replay rows for a bounded Isaac smoke run")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("refusing to overwrite evidence")
args.output.mkdir(parents=True)
app = AppLauncher(args).app
result = {"passed": False, "physics_stepped": False, "arm_state_included": False,
          "task": "dh116_hand_only_grasp_lift", "wire_proxy": True}
try:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "assets/fixed_dual_panda_dh116_v1/manifest.json").read_text())
    sources = {(x["side"], x["kind"]): Path(x["path"]) for x in manifest["sources"]}
    rows = [json.loads(line) for line in args.replay.read_text().splitlines() if line.strip()]
    rows = rows[:args.max_actions]
    if not rows:
        raise ValueError("replay is empty")
    if any(len(row["action"]) != 12 for row in rows):
        raise ValueError("replay action must be 12D")
    dt = 1.0 / 240.0
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
                    fix_root_link=True, enabled_self_collisions=True,
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
        # Keep the proxy above the initial hand envelopes for a non-penetrating
        # integration smoke; contact qualification requires a separately
        # calibrated hand pose and is not inferred from this run.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.60)))
    wire = RigidObject(wire_cfg)
    wire_sensor = ContactSensor(ContactSensorCfg(prim_path="/World/WireProxy", update_period=0.0))
    sim.reset()
    for hand, sensor in zip(hands, sensors):
        hand.reset(); sensor.reset()
    wire.reset(); wire_sensor.reset()
    active_ids = [[hand.joint_names.index(n) for n in ACTIVE] for hand in hands]
    contact_peak = [0.0, 0.0]
    for row in rows:
        action = np.asarray(row["action"], dtype=np.float32)
        for _ in range(args.steps_per_action):
            for i, hand in enumerate(hands):
                q = expand(action[i * 6:(i + 1) * 6])
                # USD joint order is not assumed to equal the contract order.
                # Drive only the six active joints by name; passive followers
                # remain owned by the articulation's mimic configuration.
                target = torch.as_tensor(action[i * 6:(i + 1) * 6], device=args.device,
                                         dtype=torch.float32)[None]
                hand.set_joint_position_target(target, joint_ids=active_ids[i])
                hand.write_data_to_sim()
            sim.step(render=False)
            for i, (hand, sensor) in enumerate(zip(hands, sensors)):
                hand.update(dt); sensor.update(dt, force_recompute=True)
                force = float(torch.linalg.vector_norm(sensor.data.net_forces_w[0]).item())
                contact_peak[i] = max(contact_peak[i], force)
            wire.update(dt); wire_sensor.update(dt, force_recompute=True)
            result["physics_stepped"] = True
    wire_z = float(wire.data.root_pos_w[0, 2].item())
    stable = bool(np.isfinite(wire_z) and abs(wire_z) < 2.0 and
                  np.isfinite(contact_peak).all() and max(contact_peak) < 100.0)
    result.update({"passed": stable, "replay_rows": len(rows), "contact_peak_n": contact_peak,
                   "wire_z_m": wire_z, "numerically_stable": stable,
                   "contact_sensor_bodies": [s.body_names for s in sensors]})
finally:
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DH116_HAND_ONLY_WIRE", json.dumps(result), flush=True)
    app.close(wait_for_replicator=False, skip_cleanup=True)
