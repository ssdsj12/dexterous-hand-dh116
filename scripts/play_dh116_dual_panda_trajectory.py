"""Visible Isaac replay of the saved full dual-Panda DH116 wire trajectory.

This is a presentation replay of recorded joint and Cosserat states.  It does
not advance PhysX or re-qualify grasp/lift; the terminal frame and failure
reason are copied from the recorded run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument("--trajectory", type=Path)
source_group.add_argument("--pose", type=Path,
                          help="Pinned offline pose artifact for a static candidate preview")
parser.add_argument("--asset", type=Path,
                    default=Path("assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd"))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--stride", type=int, default=100,
                    help="Recorded physics rows between displayed frames")
parser.add_argument("--frame-time-s", type=float, default=0.03,
                    help="Wall-clock time between replay frames")
parser.add_argument("--hold-s", type=float, default=120.,
                    help="Keep the terminal failure frame visible")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("refusing to overwrite evidence")
if args.stride < 1 or not math.isfinite(args.frame_time_s) or args.frame_time_s <= 0:
    raise ValueError("stride and frame time must be positive")
if not math.isfinite(args.hold_s) or args.hold_s < 0:
    raise ValueError("hold time must be nonnegative")
STATIC_POSE = args.pose is not None


def collect_frames(path: Path, stride: int):
    frames = []
    last = None
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            last = row
            if int(row["step"]) % stride == 0:
                frames.append({
                    "step": int(row["step"]),
                    "stage": row["grasp_sequence"].get("stage"),
                    "sides": {
                        side: {
                            "joint_names": row["sides"][side]["joint_names"],
                            "q": row["sides"][side]["actual_joint_positions"],
                        }
                        for side in ("left", "right")
                    },
                    "rod_state": {
                        "rod_position": row["rod_state"]["rod_position"],
                        "rod_director": row["rod_state"]["rod_director"],
                    },
                })
    if last is None:
        raise ValueError("trajectory is empty")
    if not frames or frames[-1]["step"] != int(last["step"]):
        frames.append({
            "step": int(last["step"]),
            "stage": last["grasp_sequence"].get("stage"),
            "sides": {
                side: {
                    "joint_names": last["sides"][side]["joint_names"],
                    "q": last["sides"][side]["actual_joint_positions"],
                }
                for side in ("left", "right")
            },
            "rod_state": {
                "rod_position": last["rod_state"]["rod_position"],
                "rod_director": last["rod_state"]["rod_director"],
            },
        })
    return frames, last


def collect_pose_frame(path: Path):
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS, expand
    data = json.loads(path.read_text())
    # Candidate preview uses the same nominal tabletop wire as the offline
    # source screen.  No contact or dynamics evidence is inferred here.
    import numpy as np
    position = np.c_[np.full(41, .43), np.linspace(-.25, .25, 41), np.full(41, .308)].T
    director = np.repeat(np.eye(3, dtype=float)[:, :, None], 40, axis=2)
    sides = {}
    for side, q in data["states"].items():
        names = [f"{side}_panda_joint{i}" for i in range(1, 8)] + list(JOINTS)
        full_q = list(q[:7]) + list(expand(q[7:]))
        sides[side] = {"joint_names": names, "q": full_q}
    frame = {"step": 0, "stage": "candidate_pose", "sides": sides,
             "rod_state": {"rod_position": position.tolist(), "rod_director": director.tolist()}}
    return [frame], frame


if args.trajectory:
    frames, terminal = collect_frames(args.trajectory, args.stride)
else:
    frames, terminal = collect_pose_frame(args.pose)
args.output.mkdir(parents=True)
app = AppLauncher(args).app
terminal_stage = terminal.get("grasp_sequence", {}).get("stage", terminal.get("stage"))
result = {
    "replay_only": True,
    "physics_advanced": False,
    "source_trajectory": str(args.trajectory) if args.trajectory else None,
    "source_pose": str(args.pose) if args.pose else None,
    "source_terminal_step": int(terminal["step"]),
    "source_terminal_stage": terminal_stage,
    "source_failure_step": None,
    "source_failure_reason": None,
    "frames": len(frames),
}


def pump(seconds):
    """Keep Kit responsive while pacing a presentation frame."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and app.is_running():
        app.update()
        time.sleep(.001)


try:
    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    import omni.usd
    from pxr import Gf, UsdGeom, Vt
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
    from go2_pvcnn.control.dh116.no_load import reference
    from go2_pvcnn.control.dh116.preview_sync import (
        commit_static_pose,
        preview_simulation_cfg_kwargs,
    )
    from go2_pvcnn.control.dh116.visual_pose_sync import apply_static_visual_poses
    from go2_pvcnn.control.dh116.wire_mesh_publisher import WireMeshPublisher
    from go2_pvcnn.control.dh116.wire_table import TABLE_CENTER, TABLE_SIZE

    # Read the terminal metadata separately without retaining the 780 MB log.
    result["source_failure_step"] = (int(terminal.get("failure_step"))
                                     if terminal.get("failure_step") is not None else None)
    result["source_failure_reason"] = (
        terminal.get("error")
        or terminal.get("grasp_sequence", {}).get("reason")
        or ("offline candidate; no physics failure recorded" if args.pose
            else "recorded lift did not qualify")
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=1. / 60., device=args.device,
        **(preview_simulation_cfg_kwargs() if STATIC_POSE else {}),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    if not args.headless:
        sim.set_camera_view(eye=(1.35, -1.35, 1.05), target=(.36, 0., .32))
        sim_utils.DomeLightCfg(intensity=2000.).func('/World/DemoLight', sim_utils.DomeLightCfg(intensity=2000.))
    table = sim_utils.CuboidCfg(size=TABLE_SIZE, collision_props=sim_utils.CollisionPropertiesCfg())
    table.func('/World/Table', table, translation=TABLE_CENTER)
    stage = sim_utils.UsdFileCfg(usd_path=str(args.asset), activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True))
    stage.func('/World/Rig', stage)
    arms = {}
    arm_names = {}
    home = [0., -.6, 0., -2.2, 0., 1.8, .7]
    first = frames[0]
    for side in ("left", "right"):
        arm_names[side] = [f"{side}_panda_joint{i}" for i in range(1, 8)]
        q0 = dict(zip(first["sides"][side]["joint_names"], first["sides"][side]["q"]))
        init = {n: float(q0.get(n, 0.)) for n in arm_names[side]}
        init.update({n: float(q0.get(n, x)) for n, x in zip(JOINTS, expand(reference(0)))})
        # A pose preview is a kinematic inspection, not a dynamics run.  Any
        # nonzero drive here can pull the arm away from the reviewed offline
        # candidate while the UI is being kept alive.
        arm_stiffness = 0.0 if STATIC_POSE else 80.0
        arm_damping = 0.0 if STATIC_POSE else 4.0
        arm_effort = 0.0 if STATIC_POSE else 45.0
        hand_stiffness = 0.0 if STATIC_POSE else 10.0
        hand_damping = 0.0 if STATIC_POSE else .05
        hand_effort = 0.0 if STATIC_POSE else .15
        cfg = ArticulationCfg(
            prim_path=f"/World/Rig/{side}_arm", spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=init),
            actuators={
                "arm": ImplicitActuatorCfg(joint_names_expr=arm_names[side], stiffness=arm_stiffness, damping=arm_damping, effort_limit_sim=arm_effort, velocity_limit_sim=.15),
                "hand": ImplicitActuatorCfg(joint_names_expr=list(ACTIVE), stiffness=hand_stiffness, damping=hand_damping, effort_limit_sim=hand_effort, velocity_limit_sim=.3),
                "passive": ImplicitActuatorCfg(joint_names_expr=[n for n in JOINTS if n not in ACTIVE], stiffness=0., damping=0., effort_limit_sim=0., velocity_limit_sim=.3),
            },
        )
        arms[side] = Articulation(cfg)

    mesh = UsdGeom.Mesh.Define(omni.usd.get_context().get_stage(), '/World/CosseratWire')
    mesh.CreateSubdivisionSchemeAttr().Set('none')
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(1., .2, .035)])
    wire = WireMeshPublisher(mesh, .008)
    sim.reset()
    for arm in arms.values():
        arm.reset()
    if STATIC_POSE:
        # The pose is committed below with one controlled simulation tick;
        # after that tick the timeline is paused for inspection.
        sim.pause()

    def apply(frame):
        import torch
        for side, arm in arms.items():
            qmap = dict(zip(frame["sides"][side]["joint_names"], frame["sides"][side]["q"]))
            q = np.asarray([qmap[n] for n in arm.joint_names], dtype=np.float32)
            arm.write_joint_state_to_sim(torch.tensor(q[None], device=args.device),
                                         torch.zeros((1, len(q)), device=args.device))
        wire.publish({"rod_position": np.asarray(frame["rod_state"]["rod_position"], dtype=np.float64),
                      "rod_director": np.asarray(frame["rod_state"]["rod_director"], dtype=np.float64)})
        sim.physics_sim_view.update_articulations_kinematic()
        if STATIC_POSE and os.environ.get("DH116_DEBUG_VISUAL_SYNC"):
            for debug_side, debug_arm in arms.items():
                debug_q = debug_arm.data.joint_pos[0].detach().cpu().numpy()
                debug_map = dict(zip(debug_arm.joint_names, debug_q))
                target_map = dict(zip(frame["sides"][debug_side]["joint_names"], frame["sides"][debug_side]["q"]))
                debug_err = max(abs(float(debug_map[n]) - float(target_map[n])) for n in debug_arm.joint_names)
                print(f"DH116_VISUAL_Q side={debug_side} max_error={debug_err:.6g} bodies={len(debug_arm.body_names)}", flush=True)
        if STATIC_POSE:
            # Flush PhysX kinematics while playing; pausing first leaves the
            # USD renderable links at their default asset pose.
            commit_static_pose(sim)
            visual_updates = apply_static_visual_poses(omni.usd.get_context().get_stage(), arms)
            print(f"DH116_REPLAY visual_updates={visual_updates}", flush=True)
            sim.render()
        else:
            sim.render()

    for index, frame in enumerate(frames):
        apply(frame)
        print(f"DH116_REPLAY frame={index + 1}/{len(frames)} step={frame['step']} stage={frame['stage']}", flush=True)
        pump(args.frame_time_s)
    result["terminal_frame_stage"] = frames[-1]["stage"]
    apply(frames[-1])
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DH116_REPLAY_TERMINAL", json.dumps(result), flush=True)
    if args.hold_s:
        pump(args.hold_s)
except Exception as exc:
    result["error"] = repr(exc)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DH116_REPLAY_ERROR", json.dumps(result), flush=True)
    raise
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
