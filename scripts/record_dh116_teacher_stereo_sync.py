#!/usr/bin/env python3
"""Bounded headless Teacher + fixed Student stereo timing recording."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from time import monotonic, monotonic_ns
import traceback
from types import SimpleNamespace

from scripts import diagnose_dh116_teacher_stereo_alignment as stereo
from scripts import diagnose_dh116_teacher_stat_cache_release as cleanup
from go2_pvcnn.control.dh116 import tabletop_teacher_stereo_alignment as alignment
from go2_pvcnn.control.dh116.tabletop_teacher_grasp_preview_runtime import MODES
from go2_pvcnn.control.dh116.tabletop_teacher_stereo_sync import ActionBoundary, sync_row, timing_summary

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path("/mnt/mydisk/xk/dh116")
REFERENCE_POLICY_HORIZON = 480
MODEL_CONFIG = "configs/dh116_execution/teacher_three_grasp_preview_v1.json"
VERSION = "dh116-teacher-stereo-sync-native-v1"
legacy = stereo.legacy


def freeze_plan(output, mode, max_steps):
    if mode not in MODES or type(max_steps) is not int or not 1 <= max_steps <= 500:
        raise ValueError("known gesture and bounded recording of at most 500 control steps required")
    plan = stereo.freeze_plan(ROOT, output, legacy.BINDINGS)
    cfg = json.loads((ROOT / MODEL_CONFIG).read_text())
    checkpoint = (ORIGINAL / cfg["checkpoints"][mode]).resolve(strict=True)
    checkpoint_sha = cfg["checkpoint_sha256"][mode]
    if legacy.sha(checkpoint) != checkpoint_sha or cfg["source_sha256"] != plan["teacher_source_sha256"]:
        raise ValueError("frozen inference Teacher checkpoint or source changed")
    plan["frozen_files"][str(checkpoint)] = checkpoint_sha
    plan["version"] = VERSION
    plan["checkpoint"] = str(checkpoint)
    plan["checkpoint_sha256"] = checkpoint_sha
    plan["teacher_stereo"]["capture_pairs"] = max_steps
    plan["budget"].update(mode=mode, control_steps=max_steps, teacher_queries=max_steps,
                          solver_steps=max_steps*8*12, max_total_sdk_steps=2+max_steps*8,
                          model_loads=1, policy_horizon=REFERENCE_POLICY_HORIZON, capture_wall_seconds=10, max_capture_renders=4*max_steps,
                          raw_rgb_upper_bound_bytes=2*640*480*3*max_steps, webrtc=False)
    plan["preserved_scope"] = "existing three-grasp preview seed and frozen Teacher, one selected mode per process"
    return plan


def check_plan(plan, sha):
    if plan.get("version") != VERSION or legacy.digest(plan) != sha:
        raise ValueError("exact bounded native synchronization fingerprint required")
    expected = freeze_plan(plan["output"], plan["budget"]["mode"], plan["budget"]["control_steps"])
    if plan != expected or Path(plan["output"]).exists():
        raise ValueError("frozen inputs changed or output directory already exists")


def _save_frame(output, receipt, packet, *, image_module, np):
    folder = output / f"frame-{receipt['identity']['policy_tick']:06d}"
    folder.mkdir()
    for side in ("left", "right"):
        rgb = packet[side+"_rgb"][0].permute(1, 2, 0).detach().cpu().numpy()
        path = folder / f"{side}.png"
        with path.open("xb") as f:
            image_module.fromarray(np.rint(rgb*255).astype(np.uint8)).save(f, format="PNG")
            f.flush()
            os.fsync(f.fileno())
        receipt["cameras"][side].update(rgb_file=path.name, rgb_sha256=legacy.sha(path))
    path = folder / "teacher_stereo_alignment.json"
    legacy.write_new(path, receipt)
    return str(path.relative_to(output)), legacy.sha(path)


def _record(app, plan, started, scope, rows, trace):
    import torch
    import numpy as np
    from PIL import Image
    from pxr import UsdGeom, UsdLux, Gf, Vt
    from go2_pvcnn.control.dh116.tabletop_teacher_grasp_preview_runtime import create_runtime
    from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
    from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
    from go2_pvcnn.control.dh116.tabletop_preview_warp_readback import inspect_rgb_readback_abi
    from go2_pvcnn.control.dh116.tabletop_stereo_rgb_bridge import _NativeRenderSetting
    from go2_pvcnn.control.dh116.tabletop_tiled_stereo_capture import TiledStereoCapture
    from go2_pvcnn.control.dh116.tabletop_tiled_stereo_native import TiledNativeStereoReader

    out, budget = Path(plan["output"]), plan["budget"]
    cal = StereoCalibration.from_json(ROOT / alignment.STUDENT_CAMERA)
    rt = sim = camera = reader = display = model = None
    boundary = ActionBoundary()
    try:
        args = SimpleNamespace(mode=budget["mode"], num_envs=1, updates=0, eval_only=True,
                               gpu=2, device="cuda:2", horizon=budget["policy_horizon"])
        rt = create_runtime(ROOT, args, app, plan["teacher_source_sha256"], step_observer=boundary)
        sim = scope["sim"] = rt.sim
        trace.watch("simulation", sim, owner="recording scope", release="owned SimulationContext cleanup")
        for side in ("left", "right"):
            hand = getattr(rt, side + "_hands")
            trace.watch(side + "_hand", hand, owner="runtime", release="runtime scope dropped")
            trace.watch(side + "_articulation_data", hand.data,
                        owner=side + " articulation", release="runtime scope dropped")
            trace.watch(side + "_root_physx_view", hand.root_physx_view,
                        owner=side + " articulation", release="runtime scope dropped", gpu_view=True)
        hand = None
        rt.reset_all()
        if sim.current_time_step_index != budget["sdk_setup_steps"]:
            raise RuntimeError("unexpected setup physics budget")
        legacy.validate_loaded_project_modules(ROOT, frozen_files=plan["frozen_files"])
        model = rt.actor_critic_factory(67, 24, init_std=.008, device=rt.device)
        raw_model = Path(plan["checkpoint"]).read_bytes()
        if hashlib.sha256(raw_model).hexdigest() != plan["checkpoint_sha256"]:
            raise ValueError("checkpoint changed before load")
        model.load_state_dict(torch.load(io.BytesIO(raw_model), map_location=rt.device,
                                         weights_only=False)["model"])
        model.eval()
        setting = _NativeRenderSetting()
        setting.set(False)
        if setting.get() is not False:
            raise RuntimeError("cannot fence physical steps during RGB render")
        display = ReadOnlyHandDisplay(sim.stage, [
            dict(env_id=0, hand=side, source_root=f"/World/envs/env_0/{side}Hand",
                 body_names=list(getattr(rt, side+"_hands").body_names))
            for side in ("left","right")])
        light = UsdLux.DomeLight.Define(sim.stage, "/World/DH116SyncLight")
        light.CreateIntensityAttr(900.)
        curve = UsdGeom.BasisCurves.Define(sim.stage, "/World/DH116SyncRod")
        curve.CreateTypeAttr().Set("linear")
        curve.CreateCurveVertexCountsAttr().Set([41])
        curve.CreateWidthsAttr().Set([2*float(rt.r_cfg["radius_m"])])
        curve.SetWidthsInterpolation("constant")
        curve.CreateDisplayColorAttr().Set([Gf.Vec3f(1., .35, .02)])
        origins = [rt.env_origins_np[0].tolist()]
        identity = None

        def snapshot():
            if (float(rt.rope.time) != identity["solver_time_s"]
                    or legacy.clock(sim) != {"sdk_step": identity["sdk_step"],
                                            "sdk_time_s": identity["sdk_timestamp_s"]}):
                raise RuntimeError("state advanced during stereo capture")
            return dict(identity, state_sha256=alignment.state_digest(alignment.runtime_state(rt)))

        reader = TiledNativeStereoReader(sim, origins, cal, rt.device)
        camera = TiledStereoCapture(sim, reader, cal, snapshot, render_setting=setting, origins=origins,
                                    max_render_attempts=plan["teacher_stereo"]["max_render_attempts"])
        trace.watch("fixed_stereo_reader", reader, owner="camera", release="owned camera close")
        warp_abi = inspect_rgb_readback_abi()
        legacy.write_new(out/"warp_rgb_abi.json", warp_abi)
        if warp_abi["error"] is not None:
            raise ValueError(warp_abi["error"])
        run = out.name
        with (out/"sync_log.jsonl").open("x") as ledger:
            window = monotonic()
            for tick in range(budget["control_steps"]):
                if monotonic()-window >= budget["capture_wall_seconds"]:
                    break
                legacy.continue_within_budget(started, "next synchronization frame", lambda: None)
                before = legacy.clock(sim)
                step, stamp = before["sdk_step"], before["sdk_time_s"]
                if step != 2+8*tick or step+8 > budget["max_total_sdk_steps"]:
                    raise RuntimeError("control step exceeds frozen SDK budget")
                state = alignment.runtime_state(rt)
                identity = dict(run_uuid=run, env_id=0, episode_uid=run+":0", reset_id=0, policy_tick=tick,
                                physics_tick_start=step, physics_tick_end=step, rod_step=step-2,
                                state_sha256=alignment.state_digest(state), simulation_timestamp=stamp,
                                calibration_sha256=cal.sha256, sdk_step=step, sdk_timestamp_s=stamp,
                                rod_tick=step-2, solver_time_s=float(rt.rope.time),
                                material_reference_uid=run+":rod0", rod_uid=run+":rod")
                pose_error = display.sync(
                    {(0, side):getattr(rt, side+"_hands").data.body_link_pose_w[0]
                     for side in ("left","right")}, read_clock=lambda:legacy.clock(sim))
                if pose_error["max_position_error_m"] > 1e-4 or pose_error["max_rotation_error_rad"] > 1e-3:
                    raise RuntimeError("rendered hands differ from actual link poses")
                curve.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(state["rod_nodes_world_m"],
                                                                            dtype=np.float32)))
                start_ns = monotonic_ns()
                packet = camera.capture()
                rgb_ns = monotonic_ns()
                if packet["source_scope"] != "native_tiled_reader" or packet["identity"] != identity:
                    raise RuntimeError("expected native pre-action RGB pair")
                if not sim.is_playing():
                    raise RuntimeError("timeline is paused before controller execution")
                boundary.begin()
                with torch.inference_mode():
                    obs = torch.as_tensor(state["teacher_obs67"], dtype=torch.float32, device=rt.device)[None]
                    action = model(obs)[0]
                    rt.step_env(action)
                row = sync_row(packet, boundary.issued, boundary.integrated,
                               capture_start_ns=start_ns, rgb_ready_ns=rgb_ns,
                               final_step=sim.current_time_step_index, actor_action=action[0].cpu().tolist(),
                               applied_action=boundary.applied[0].cpu().tolist())
                receipt = alignment.aligned_payload(state, identity, packet, cal, origins)
                receipt.update(source_scope=packet["source_scope"], phase="before_teacher_action",
                               privileged_source="frozen Teacher get_obs()", teacher_policy_queried=True,
                               env_origin_world_m=state["env_origin_world_m"],
                               teacher_reference_env_m=state["teacher_reference_env_m"],
                               teacher_wire_node_indices=state["teacher_wire_node_indices"])
                path, sha = _save_frame(out, receipt, packet, image_module=Image, np=np)
                row.update(frame_receipt=path, frame_receipt_sha256=sha, mode=budget["mode"],
                           checkpoint_sha256=plan["checkpoint_sha256"], render_passes=packet["render_passes"],
                           within_reference_policy_horizon=tick<budget["policy_horizon"])
                ledger.write(json.dumps(row, allow_nan=False)+"\n")
                ledger.flush()
                os.fsync(ledger.fileno())
                rows.append(row)
        if not rows:
            raise RuntimeError("recording window elapsed without a complete sample")
        scope["window"] = dict(elapsed_wall_s=monotonic()-window, completed_frames=len(rows),
            stop_reason="control_limit" if len(rows)==budget["control_steps"] else "capture_wall_limit")
    finally:
        try:
            if camera is not None:
                camera.close()
            elif reader is not None:
                reader.close()
        finally:
            try:
                if display is not None:
                    display.close()
            finally:
                if rt is not None:
                    rt.__dict__.clear()
                camera = reader = display = model = rt = sim = app = None


def execute_native(plan):
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    out = Path(plan["output"])
    if shutil.disk_usage(out.parent).free < max(256*1024**2, 2*plan["budget"]["raw_rgb_upper_bound_bytes"]):
        raise ValueError("not enough free disk space for bounded RGB evidence")
    with native_instance_lock():
        out.mkdir()
        legacy.write_new(out/"frozen_plan.json", plan)
        started = monotonic()
        app = trace = None
        errors, scope, rows = [], {}, []

        def save_error(phase, error):
            errors.append(dict(phase=phase, traceback=traceback.format_exc()))
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
            legacy.write_new(out/f"failure_{phase}.json", errors[-1])

        try:
            try:
                from isaaclab.app import AppLauncher
                import signal
                from go2_pvcnn.control.dh116.tabletop_preview_launcher import (
                    ManagedPreviewApp, preview_launcher_type)
                if any(name == "pxr" or name.startswith("pxr.") for name in sys.modules):
                    raise RuntimeError("USD loaded before owned AppLauncher")
                argv = sys.argv
                try:
                    sys.argv = argv[:1]
                    launcher = preview_launcher_type(AppLauncher)
                    app = ManagedPreviewApp(
                        launcher(headless=True, enable_cameras=True, device="cuda:2", livestream=0,
                                 active_gpu=2, physics_gpu=2, multi_gpu=False, fast_shutdown=False,
                                 experience=plan["experience"]), signal_api=signal)
                finally:
                    sys.argv = argv
                import torch
                torch.cuda.set_device(2)
                from scripts.dh116_pinch_startup_v3 import StartupTrace
                from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
                trace = StartupTrace(out/"ownership.jsonl")
                release_cache = cleanup.make_before_collect(plan)
                def release_at_current_boundary():
                    sim = scope.get("sim")
                    if sim is not None:
                        step = legacy.clock(sim)["sdk_step"]
                        if not 2 <= step <= plan["budget"]["max_total_sdk_steps"]:
                            raise ValueError("post-recording SDK clock outside frozen budget")
                        release_cache(sim, expected_sdk_step=step)

                run_owned_call(lambda owners: _record(app, plan, started, scope, rows, trace),
                               synchronize=lambda:torch.cuda.synchronize("cuda:2"), ownership=trace,
                               before_collect=release_at_current_boundary)
            except BaseException as error:
                save_error("record", error)

            try:
                summary = timing_summary(rows)
            except ValueError as error:
                save_error("timing", error)
                summary = dict(frames=len(rows), sequence_valid=False)
            legacy.write_new(out/"sync_log.json", dict(version=VERSION, mode=plan["budget"]["mode"],
                capture_complete=not errors, window=scope.get("window"), timings=summary, source="frozen native Teacher and Student cameras",
                data_file="sync_log.jsonl", teacher_admitted=False, process_exit_verified=False))
            failed_checks = [key for key in ("rgb_phys_under_2ms", "action_same_sdk_boundary",
                                             "action_sim_offset_constant") if summary.get(key) is False]
            if failed_checks:
                legacy.write_new(out/"failure_timing_targets.json", dict(failed_checks=failed_checks,
                    samples=len(rows), rgb_phys_outliers=[row["capture_id"] for row in rows
                        if abs(row["rgb_phys_delta_ms"]) >= 2.]))

        finally:
            if app is not None:
                sim = None
                from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import close_calibration_app
                try:
                    import isaaclab.sim as sim_utils
                    sim = scope.pop("sim", None) or sim_utils.SimulationContext.instance()
                    if sim is not None:
                        import carb.settings
                        import omni.timeline
                        from isaacsim.core.simulation_manager import SimulationManager
                        from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import (
                            clear_known_simulation_instance, simulation_source_contract)
                        setting = carb.settings.get_settings()
                        setting.set_bool("/app/player/playSimulations", False)
                        if setting.get("/app/player/playSimulations") is not False:
                            raise RuntimeError("physics freeze failed before STOP")
                        legacy.write_new(out/"known_simulation_cleanup.json",
                                         clear_known_simulation_instance(sim, contract=simulation_source_contract()))
                        sim = None
                        torch.cuda.synchronize("cuda:2")
                        omni.timeline.get_timeline_interface().stop()
                        app.update()
                        released = SimulationManager.get_physics_sim_view() is None
                        legacy.write_new(out/"stop.json", dict(physics_view_released=released))
                        if not released:
                            raise RuntimeError("SimulationManager still owns physics view after STOP")
                except BaseException as error:
                    save_error("stop", error)
                finally:
                    try:
                        owned = sim if sim is not None and type(sim).instance() is sim else None
                        sim = None
                        close_calibration_app(app, out, simulation=owned)
                        owned = None
                    except BaseException as error:
                        save_error("close", error)
                    else:
                        if trace is not None:
                            trace.checkpoint("after app close returned")
        if monotonic()-started > plan["budget"]["soft_wall_seconds"]:
            errors.append(dict(phase="wall", error="soft wall budget exceeded"))
        legacy.write_new(out/"result.json", dict(errors=errors, frames=len(rows),
                         process_exit_verified=False, teacher_admitted=False))
        return 1 if errors else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="pinch")
    parser.add_argument("--max-control-steps", type=int, default=500)
    parser.add_argument("--record-sync-data", action="store_true")
    parser.add_argument("--approved-frozen-sha256")
    args = parser.parse_args(argv)
    cleanup.validate_execution_context(ROOT)
    if not args.record_sync_data:
        plan = freeze_plan(args.output, args.mode, args.max_control_steps)
        legacy.write_new(args.plan, plan)
        print(json.dumps(dict(frozen_sha256=legacy.digest(plan), native_started=False, budget=plan["budget"])))
        return 0
    plan = json.loads(args.plan.read_text())
    check_plan(plan, args.approved_frozen_sha256)
    if (str(args.output.resolve()) != plan["output"] or args.mode != plan["budget"]["mode"]
            or args.max_control_steps != plan["budget"]["control_steps"]):
        raise ValueError("CLI and frozen recording identity differ")
    return execute_native(plan)


if __name__ == "__main__":
    raise SystemExit(main())
