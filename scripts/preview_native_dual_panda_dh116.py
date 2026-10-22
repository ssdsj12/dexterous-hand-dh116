"""Preview candidate3 with native Panda arms and separate DH116 USD hands."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pose", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--stage", choices=("static", "close"), default="static")
parser.add_argument("--hold-s", type=float, default=30.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise ValueError(f"refusing to overwrite evidence: {args.output}")
if args.hold_s < 0:
    raise ValueError("--hold-s must be nonnegative")

ROOT = Path(__file__).resolve().parents[1]
args.output.mkdir(parents=True)

from go2_pvcnn.control.dh116.native_scene import (  # noqa: E402
    NativeSceneSpec,
    candidate3_pose,
    default_native_scene_spec,
    spawn_native_scene,
    wire_orientation_wxyz,
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
}

app = AppLauncher(args).app
try:
    import isaaclab.sim as sim_utils
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 240.0, device=args.device)
    )
    if not args.headless:
        sim.set_camera_view(eye=(1.2, -1.25, 0.95), target=(0.38, 0.0, 0.30))
        sim_utils.DomeLightCfg(intensity=1800.0).func(
            "/World/SceneLight", sim_utils.DomeLightCfg(intensity=1800.0)
        )
    table_cfg = sim_utils.CuboidCfg(
        size=(0.9, 0.9, 0.08),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    table_cfg.func("/World/Table", table_cfg, translation=(0.35, 0.0, 0.04))
    wire_cfg = sim_utils.CylinderCfg(
        radius=0.008,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.02)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    wire_cfg.func(
        "/World/Wire",
        wire_cfg,
        translation=(0.43, 0.0, 0.088),
        orientation=wire_orientation_wxyz(),
    )
    runtime = spawn_native_scene(sim, spec, pose, args.device)
    result.update(runtime.assert_visual_contract())
    result["stages"].append("static")
    sim.step(render=not args.headless)
    result["physics_stepped"] = True
    sim.pause()
    result["pose_contract"] = {
        side: {"arm_q": pose[side][:7].tolist(), "hand_active_q": pose[side][7:].tolist()}
        for side in ("left", "right")
    }
    (args.output / "static_pose.json").write_text(json.dumps(result, indent=2) + "\n")
    try:
        from omni.kit.viewport.utility import capture_viewport_to_file

        capture_viewport_to_file("/OmniverseKit_Persp", str(args.output / "static_pose.png"))
        result["screenshot"] = str(args.output / "static_pose.png")
    except Exception as exc:
        result["screenshot_error"] = repr(exc)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DH116_NATIVE_SCENE", json.dumps(result), flush=True)
    if not args.headless and args.hold_s:
        deadline = time.monotonic() + args.hold_s
        while app.is_running() and time.monotonic() < deadline:
            app.update()
            time.sleep(0.01)
except Exception as exc:
    result["error"] = repr(exc)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DH116_NATIVE_SCENE_ERROR", json.dumps(result), flush=True)
    raise
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
