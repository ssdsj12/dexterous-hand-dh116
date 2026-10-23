"""Preview candidate3 with native Panda arms and separate DH116 USD hands."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pose", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--stage", choices=("static", "close", "hold", "lift"), default="static")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--hold-s", type=float, default=30.0)
parser.add_argument("--view", choices=("perspective", "top", "side"), default="perspective")
parser.add_argument("--capture-views", action="store_true",
                    help="capture perspective, top, and side Isaac viewport images")
parser.add_argument("--sequence", action="store_true",
                    help="run close, hold, and lift in one Isaac instance")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise ValueError(f"refusing to overwrite evidence: {args.output}")
if args.hold_s < 0:
    raise ValueError("--hold-s must be nonnegative")
if args.steps < 1:
    raise ValueError("--steps must be positive")
if args.sequence and args.stage != "close":
    raise ValueError("--sequence requires --stage close")

ROOT = Path(__file__).resolve().parents[1]
args.output.mkdir(parents=True)

from go2_pvcnn.control.dh116.native_scene import (  # noqa: E402
    NativeSceneSpec,
    candidate3_pose,
    default_native_scene_spec,
    apply_native_targets,
    spawn_native_scene,
    sync_native_hand_roots,
    wire_orientation_wxyz,
)
from go2_pvcnn.control.dh116.native_stage_controller import (  # noqa: E402
    NativeObservation,
    NativeStageController,
)
from go2_pvcnn.control.dh116.preview_sync import commit_static_pose  # noqa: E402
from go2_pvcnn.control.dh116.visual_pose_sync import apply_static_visual_poses  # noqa: E402
from go2_pvcnn.control.dh116.reference_view import (  # noqa: E402
    REFERENCE_TABLE_CENTER,
    REFERENCE_TABLE_SIZE,
    camera_preset,
    reference_wire_points,
)

pose = candidate3_pose(args.pose)
spec = default_native_scene_spec(ROOT)
pose_sha = hashlib.sha256(args.pose.read_bytes()).hexdigest()
result = {
    "asset_mode": "native_panda_plus_separate_dh116",
    "initial_pose_source": str(args.pose),
    "initial_pose_sha256": pose_sha,
    "stage_requested": args.stage,
    "stages": [],
    "physics_stepped": False,
    "pose_preview_only": args.stage == "static",
    "reference_layout": {
        "table_center": list(REFERENCE_TABLE_CENTER),
        "table_size": list(REFERENCE_TABLE_SIZE),
        "wire_points": reference_wire_points().T.tolist(),
        "camera": args.view,
    },
}

app = AppLauncher(args).app
try:
    import isaaclab.sim as sim_utils
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 240.0, device=args.device)
    )
    if not args.headless:
        view = camera_preset(args.view)
        sim.set_camera_view(eye=view.eye, target=view.target)
        sim_utils.DomeLightCfg(intensity=1800.0).func(
            "/World/SceneLight", sim_utils.DomeLightCfg(intensity=1800.0)
        )
    table_cfg = sim_utils.CuboidCfg(
        size=REFERENCE_TABLE_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    table_cfg.func("/World/Table", table_cfg, translation=REFERENCE_TABLE_CENTER)
    wire_cfg = sim_utils.CylinderCfg(
        radius=0.008,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.02)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    wire_points = reference_wire_points()
    wire_cfg.func(
        "/World/Wire",
        wire_cfg,
        # Candidate3 reference wire: vertical in top view, 8 mm radius,
        # resting just above the .30 m tabletop plane.
        translation=tuple(wire_points.mean(axis=1)),
        orientation=wire_orientation_wxyz(),
    )
    runtime = spawn_native_scene(sim, spec, pose, args.device)
    result.update(runtime.assert_visual_contract())
    # Record the actual native flange and DH116 base poses used for the visual
    # check; this makes the target-pose comparison auditable instead of only
    # reporting the requested joint vector.
    resolved_poses = {}
    contact_points = {
        "left": {
            "thumb": ("finger14_link", (0.00017948482980035264, -0.010452785059975968, -0.03394154097890579)),
            "index": ("finger23_link", (-0.0038150536494581076, -0.015150306258402396, 0.040506412058763396)),
        },
        "right": {
            "thumb": ("finger14_link", (-0.0008940861981015615, 0.03219328112399958, -0.015560792002874102)),
            "index": ("finger23_link", (0.004119789543680456, -0.01512276321265944, 0.03972204958180129)),
        },
    }
    def quat_to_matrix(q):
        w, x, y, z = [float(v) for v in q]
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    for side in ("left", "right"):
        arm = runtime.arms[side]
        hand = runtime.hands[side]
        # Refresh articulation caches after the final kinematic root write;
        # otherwise body_link_pose_w can still contain the pre-write frame.
        hand.update(1.0 / 240.0)
        ai = arm.body_names.index(spec.flange_link)
        hi = hand.body_names.index("base_link")
        static_override = runtime.static_body_poses.get(side, {})
        base_pose = np.asarray(
            static_override.get("base_link", hand.data.body_link_pose_w[0, hi].detach().cpu().numpy()),
            dtype=float,
        )
        resolved_poses[side] = {
            "panda_flange_pose_w": arm.data.body_link_pose_w[0, ai].detach().cpu().numpy().tolist(),
            "dh116_base_pose_w": base_pose.tolist(),
            "joint_names": list(hand.joint_names),
            "joint_pos": hand.data.joint_pos[0].detach().cpu().numpy().tolist(),
        }
        contact_points[side] = {
            digit: {
                "body": body,
                "point_w": (
                    np.asarray(static_override.get(body, hand.data.body_link_pose_w[0, hand.body_names.index(body)].detach().cpu().numpy()))[:3]
                    + quat_to_matrix(np.asarray(static_override.get(body, hand.data.body_link_pose_w[0, hand.body_names.index(body)].detach().cpu().numpy()))[3:7])
                    @ np.asarray(local, dtype=float)
                ).tolist(),
            }
            for digit, (body, local) in contact_points[side].items()
        }
    result["resolved_poses"] = resolved_poses
    result["contact_points"] = contact_points
    if args.stage == "static":
        # A paused Isaac timeline can retain authored default link transforms
        # even after PhysX accepts the commanded joints.  Flush one controlled
        # tick, then author the measured Panda and DH116 link poses into the
        # visible USD layer so the screenshot is the actual candidate pose.
        commit_static_pose(sim)
        # Panda link visuals are already driven by the native articulation;
        # the separately referenced DH116 links need the explicit USD flush.
        result["static_visual_updates"] = apply_static_visual_poses(
            runtime.stage,
            runtime.hands,
            runtime.spec.hand_paths,
            runtime.static_body_poses,
        )
        # The stage edits are picked up by the normal Kit update loop.  A
        # second explicit ``SimulationContext.render`` here races the 5.1
        # viewport delegate (and can close the GUI with ``'str' object has no
        # attribute 'capture'``), so leave redraw ownership to Kit.
    else:
        sim.step(render=not args.headless)
    result["physics_stepped"] = True
    if args.stage == "static":
        result["stages"].append("static")
        sim.pause()
    else:
        controller = NativeStageController("close" if args.sequence else args.stage)
        current = {side: pose[side].copy() for side in ("left", "right")}
        trajectory = []
        stage_captures = {}
        stage_steps = 0
        total_steps = args.steps * (3 if args.sequence else 1)
        sim.play()
        for step in range(total_steps):
            active_stage = controller.stage
            if active_stage == "close":
                fraction = min((stage_steps + 1) / args.steps, 1.0)
                for side in ("left", "right"):
                    current[side][7:] = pose[side][7:] + fraction * 0.03
            lift_displacement = min((stage_steps + 1) * controller.lift_step_m, 0.03)
            observation = NativeObservation(
                contact_established=active_stage != "close" or stage_steps == args.steps - 1,
                contact_duration_s=0.1 if active_stage == "close" and stage_steps == args.steps - 1 else 0.0,
                hold_duration_s=1.0 if active_stage == "hold" and stage_steps == args.steps - 1 else 0.0,
                slip_ok=True,
                lift_displacement_m=lift_displacement if active_stage == "lift" else 0.0,
                lift_duration_s=1.0 if active_stage == "lift" and stage_steps == args.steps - 1 else 0.0,
                whole_wire_clear=active_stage == "lift" and stage_steps == args.steps - 1,
                hand_active_targets={side: current[side][7:].copy() for side in ("left", "right")},
            )
            command = controller.step(observation)
            for side in ("left", "right"):
                current[side][7:] = command.hand_active_targets[side]
            apply_native_targets(runtime, current, args.device)
            lift_delta = max(command.panda_delta_z.values()) if lift_displacement < 0.03 else 0.0
            if lift_delta:
                runtime.apply_root_delta_z(lift_delta)
            sim.step(render=not args.headless)
            sync_native_hand_roots(runtime)
            trajectory.append({
                "step": step + 1,
                "observed_stage": active_stage,
                "stage": command.stage,
                "panda_delta_z": command.panda_delta_z,
                "hand_active_targets": {
                    side: current[side][7:].tolist() for side in ("left", "right")
                },
            })
            transition = controller.stage != active_stage or controller.failed or controller.completed
            if (transition or stage_steps == args.steps - 1) and not args.headless:
                stage_captures[active_stage] = str(args.output / f"{active_stage}_{args.view}.png")
                try:
                    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
                    view = camera_preset(args.view)
                    sim.set_camera_view(eye=view.eye, target=view.target)
                    sim.render()
                    capture_viewport_to_file(
                        get_active_viewport(), file_path=stage_captures[active_stage]
                    )
                    for _ in range(5):
                        app.update()
                except Exception as exc:
                    result.setdefault("screenshot_errors", {})[active_stage] = repr(exc)
            if transition and (not args.sequence or controller.failed or controller.completed):
                break
            stage_steps = 0 if controller.stage != active_stage else stage_steps + 1
        result["stages"].append(controller.stage)
        result["stage_evidence"] = controller.stage_evidence()
        result["physical_validation"] = False
        result["stage_captures"] = stage_captures
        (args.output / "trajectory.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in trajectory)
        )
        sim.pause()
    result["pose_contract"] = {
        side: {"arm_q": pose[side][:7].tolist(), "hand_active_q": pose[side][7:].tolist()}
        for side in ("left", "right")
    }
    if not args.headless:
        try:
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            names = ("perspective", "top", "side") if args.capture_views else (args.view,)
            view_captures = {}
            for name in names:
                view = camera_preset(name)
                sim.set_camera_view(eye=view.eye, target=view.target)
                sim.render()
                path = args.output / f"view_{name}.png"
                capture_viewport_to_file(get_active_viewport(), file_path=str(path))
                for _ in range(5):
                    app.update()
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"viewport capture produced no image: {path}")
                view_captures[name] = str(path)
            result["view_captures"] = view_captures
            result["screenshot"] = view_captures[args.view]
        except Exception as exc:
            result["screenshot_error"] = repr(exc)
    (args.output / "static_pose.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DH116_NATIVE_SCENE", json.dumps(result), flush=True)
    if not args.headless and args.hold_s:
        deadline = time.monotonic() + args.hold_s
        # Keep the GUI alive even while the simulation timeline is paused;
        # Isaac reports ``is_running`` false for a paused timeline in some
        # 5.1 builds although the Kit window is still usable.
        while time.monotonic() < deadline:
            app.update()
            time.sleep(0.01)
except Exception as exc:
    result["error"] = repr(exc)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DH116_NATIVE_SCENE_ERROR", json.dumps(result), flush=True)
    raise
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
