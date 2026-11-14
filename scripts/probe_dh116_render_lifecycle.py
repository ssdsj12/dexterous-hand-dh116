#!/usr/bin/env python3
"""Independent F6/F7 camera ownership cases with bounded native captures."""

import argparse
from contextlib import ExitStack
import gc
import hashlib
from io import BytesIO
import json
from pathlib import Path
import traceback

import numpy as np

from go2_pvcnn.control.dh116.tabletop_stereo_capture_boundary import (
    audit_depth_pair, capture_at_boundary, lifecycle_status,
)
from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_vision_camera import paired_render_timestamp

if __package__:
    from .run_dh116_deformx_render_smoke import _write_json, app_launch_options
else:
    from run_dh116_deformx_render_smoke import _write_json, app_launch_options


ROOT = Path(__file__).resolve().parents[1]
CASES = ("V0", "V1", "V2", "V3", "V4")
FULL_PRODUCTS = {
    "rgb": "rgb",
    "radial_distance_m": "distance_to_camera",
    "optical_z_m": "distance_to_image_plane",
    "native_instance_id": "instance_id_segmentation_fast",
    "reference_time": "ReferenceTime",
    "isaac_times": "IsaacReadTimes",
}
MARKER = "/World/DH116LifecycleMarker"
STANDALONE_RESOLUTION = (96, 64)
PLANE_OPTICAL_Z_M = 1.98
DEPTH_AUDIT_TOLERANCE_M = .002


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _state_sha(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(f"{value.dtype.str}:{value.shape}:".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _clock(readers):
    reference = readers["reference_time"].get_data()
    times = readers["isaac_times"].get_data()
    if not isinstance(times, dict) or "simulationTime" not in times:
        raise ValueError("native render clock is not ready")
    timing = paired_render_timestamp(reference, times, legacy_time=times["simulationTime"])
    return tuple(timing[key] for key in (
        "swh_frame_number", "frame_id", "reference_time_denominator",
        "render_frame_number", "simulation_time",
    ))


def _attached(stack, rep, path, width, height, channels):
    product = rep.create.render_product(path, (width, height))
    stack.callback(product.destroy)
    annotators = {}
    for label, name in channels.items():
        params = {"colorize": False} if label == "native_instance_id" else None
        annotator = rep.AnnotatorRegistry.get_annotator(name, device="cpu", init_params=params)
        annotator.attach(product.path)
        stack.callback(annotator.detach, product.path)
        annotators[label] = annotator
    return annotators


def _rgb_image(value, width, height):
    rgb = np.asarray(value)
    if rgb.shape not in ((height, width, 3), (height, width, 4)) or rgb.dtype != np.uint8:
        raise ValueError("native RGB geometry or encoding differs")
    return rgb


def _scalar_image(value, width, height, name, dtypes):
    image = np.asarray(value)
    if image.dtype not in dtypes or image.shape not in ((height, width), (height, width, 1)):
        raise ValueError(f"{name} has wrong native geometry or dtype: {image.shape} {image.dtype}")
    return image if image.ndim == 2 else image[..., 0]


def _depth_valid(image):
    return np.isfinite(image) & (image > 0)


def _read_products(readers, width, height):
    read = lambda key: readers[key].get_data(device="cpu", do_array_copy=True)
    rgb = _rgb_image(read("rgb"), width, height)
    if "radial_distance_m" not in readers:
        return {"rgb": rgb}
    radial, optical = (_scalar_image(read(key), width, height, key, (np.float32,))
                       for key in ("radial_distance_m", "optical_z_m"))
    packet = read("native_instance_id")
    ids = _scalar_image(packet["data"], width, height, "native_instance_id", (np.int32, np.uint32))
    return {
        "rgb": rgb, "radial_distance_m": radial, "optical_z_m": optical,
        "radial_valid": _depth_valid(radial), "depth_valid": _depth_valid(optical),
        "native_instance_id": ids.view(np.uint32),
        "idToLabels": {str(key): value for key, value in packet["info"]["idToLabels"].items()},
    }


def _depth_diagnostics(products, intrinsics, plane):
    radial, optical = products["radial_distance_m"], products["optical_z_m"]
    diagnostics = audit_depth_pair(radial, optical, intrinsics)
    tolerance = DEPTH_AUDIT_TOLERANCE_M
    if diagnostics["p95_abs_error_m"] > tolerance:
        raise ValueError("native radial-to-optical Z audit exceeds 2 mm")
    if plane is not None and (
        diagnostics["center_p95_abs_error_m"] is None or diagnostics["edge_p95_abs_error_m"] is None
        or max(diagnostics["center_p95_abs_error_m"], diagnostics["edge_p95_abs_error_m"]) > tolerance
        or abs(float(np.median(optical[products["depth_valid"]])) - plane) > tolerance
    ):
        raise ValueError("known fronto-parallel target depth failed")
    return diagnostics


def _audit_saved_products(path, camera, width, height, plane):
    raw = path.read_bytes()
    if _sha(raw) != camera["capture_sha256"]:
        raise ValueError("capture NPZ digest mismatch")
    intrinsics = np.asarray(camera["camera_intrinsics"], dtype=np.float64)
    if _state_sha([intrinsics]) != camera["camera_intrinsics_sha256"]:
        raise ValueError("camera intrinsics digest mismatch")
    with np.load(BytesIO(raw), allow_pickle=False) as products:
        _rgb_image(products["rgb"], width, height)
        for name, dtypes in (("radial_distance_m", (np.float32,)), ("optical_z_m", (np.float32,)),
                             ("native_instance_id", (np.uint32,))):
            image = products[name]
            if image.ndim != 2:
                raise ValueError(f"capture artifact {name} must have canonical image axes")
            _scalar_image(image, width, height, name, dtypes)
        for depth, mask in (("radial_distance_m", "radial_valid"), ("optical_z_m", "depth_valid")):
            valid = products[mask]
            if valid.dtype != np.bool_ or not np.array_equal(valid, _depth_valid(products[depth])):
                raise ValueError(f"capture {mask} differs from its depth channel")
        _depth_diagnostics(products, intrinsics, plane)


def _audit_render_clock(record, sides):
    clocks = record["camera_render_clocks"]
    render_clock = record["render_clock"]
    state = record["state"]
    trace = record["physical_state_trace"]
    if (set(trace) != {"before_render", "before_read", "after_read"}
            or not {"physics_tick", "rod_step", "state_sha256", "simulation_time"}.issubset(state)
            or any(phase != state for phase in trace.values())):
        raise ValueError("capture boundary physical state changed across camera capture")
    if (set(clocks) != sides or len(render_clock) != 5 or type(render_clock[0]) is not int
            or type(render_clock[-1]) not in (float, int) or not np.isfinite(render_clock[-1])
            or not abs(render_clock[-1] - state["simulation_time"]) <= 1e-5):
        raise ValueError("capture boundary clocks do not identify the same rendered physical state")
    for camera in clocks.values():
        previous = camera["before_render"]
        if (len(previous) != len(render_clock) or type(previous[0]) is not int or previous[0] >= render_clock[0]
                or camera["before_read"] != render_clock or camera["after_read"] != render_clock):
            raise ValueError("capture boundary contains stale or different camera read clocks")

def _author_camera(stage, path, k, optical, width, height):
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateProjectionAttr("perspective")
    focal = 24.0
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(width * focal / k[0, 0])
    camera.CreateVerticalApertureAttr(height * focal / k[1, 1])
    camera.CreateHorizontalApertureOffsetAttr((k[0, 2] - width / 2) * focal / k[0, 0])
    camera.CreateVerticalApertureOffsetAttr(-(k[1, 2] - height / 2) * focal / k[1, 1])
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.))
    usd = optical @ np.diag([1., -1., -1., 1.])
    camera.AddTransformOp().Set(Gf.Matrix4d(*usd.T.reshape(-1).tolist()))
    fx = width * camera.GetFocalLengthAttr().Get() / camera.GetHorizontalApertureAttr().Get()
    fy = height * camera.GetFocalLengthAttr().Get() / camera.GetVerticalApertureAttr().Get()
    cx = width / 2 + fx * camera.GetHorizontalApertureOffsetAttr().Get() / focal
    cy = height / 2 - fy * camera.GetVerticalApertureOffsetAttr().Get() / focal
    measured = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    if not np.allclose(measured, k, atol=2e-5, rtol=2 * np.finfo(np.float32).eps):
        raise ValueError(f"pinhole USD lens readback differs: {path}")
    return measured


def _capture_pair(output, label, snapshot, render, cameras, matrices, width, height, *,
                  plane=None, warmup=True):
    if warmup:
        before_warmup = snapshot()
        for _ in range(8):
            render()
            if snapshot() != before_warmup:
                raise ValueError("capture boundary warmup advanced the frozen physical state")
            try:
                for reader in cameras.values():
                    _clock(reader)
            except ValueError:
                continue
            break
        else:
            raise RuntimeError("native camera clocks remained unavailable after 8 render-only warmups")

    result = capture_at_boundary(
        snapshot, render, lambda side: _clock(cameras[side]),
        {side: (lambda reader=reader: _read_products(reader, width, height))
         for side, reader in cameras.items()},
    )
    state = result["state"]
    if abs(result["render_clock"][-1] - state["simulation_time"]) > 1e-5:
        raise ValueError("capture boundary actual renderer time differs from the frozen simulation time")
    metrics = {}
    for side, products in result["captures"].items():
        k = matrices[side]
        diagnostics = _depth_diagnostics(products, k, plane)
        arrays = {key: value for key, value in products.items() if isinstance(value, np.ndarray)}
        path = output / f"{label}-{side}.npz"
        np.savez(path, **arrays)
        metrics[side] = {
            **diagnostics,
            "capture_sha256": _sha(path.read_bytes()),
            "idToLabels": products["idToLabels"],
            "camera_intrinsics": k.tolist(),
            "camera_intrinsics_sha256": _state_sha([k]),
            "depth_model": "pinhole_native_distance_to_camera_and_distance_to_image_plane",
        }
    _write_json(output / f"{label}-boundary.json", {
        "state": state, "render_clock": result["render_clock"], "cameras": metrics,
        "camera_render_clocks": result["camera_render_clocks"],
        "physical_state_trace": result["physical_state_trace"],
        "capture_synchronization_verified": True, "depth_semantics_verified": True,
    })
    return result


def _standalone_capture(app, context, case, output):
    import omni.replicator.core as rep
    import omni.timeline
    from pxr import Gf, UsdGeom, UsdLux

    context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("could not create the independent camera probe stage")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    width, height = STANDALONE_RESOLUTION
    k = np.array([[80., 0., width / 2], [0., 80., height / 2], [0., 0., 1.]])
    camera = "/World/ProbeCamera"
    k = _author_camera(stage, camera, k, np.diag([1., -1., -1., 1.]), width, height)
    target = UsdGeom.Cube.Define(stage, "/World/PlaneTarget")
    target.CreateSizeAttr(1.0)
    target.AddTranslateOp().Set(Gf.Vec3d(0., 0., -2.))
    target.AddScaleOp().Set(Gf.Vec3f(4., 3., .04))
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(500.)
    stage.GetRootLayer().Export(str(output / "probe_stage.usda"))
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    def snapshot():
        return {"physics_tick": None, "rod_step": None, "simulation_time": timeline.get_current_time(),
                "state_sha256": _sha(stage.GetRootLayer().ExportToString().encode()),
                "state_scope": "paused_stage_without_PhysX_or_RopeBatch"}

    channels = {"V1": {}, "V2": {"rgb": "rgb"}, "V3": FULL_PRODUCTS}[case]
    with ExitStack() as stack:
        reader = _attached(stack, rep, camera, width, height, channels)
        try:
            if case == "V1":
                app.update()
            elif case == "V2":
                rep.orchestrator.step(rt_subframes=1, pause_timeline=True, delta_time=0.)
                np.save(output / "rgb.npy", _read_products(reader, width, height)["rgb"])
            else:
                render = lambda: rep.orchestrator.step(rt_subframes=1, pause_timeline=True, delta_time=0.)
                _capture_pair(output, "plane", snapshot, render, {"single": reader},
                              {"single": k}, width, height, plane=PLANE_OPTICAL_Z_M)
        finally:
            if case != "V1":
                rep.orchestrator.stop()
                rep.orchestrator.wait_until_complete()
    app.update()


def standalone_launch_options(experience, no_sdk_default_stage):
    config, options = app_launch_options(experience)
    config["extra_args"] = [*config["extra_args"], "--enable", "isaacsim.core.nodes"]
    if no_sdk_default_stage:
        config["create_new_stage"] = False
    return config, options


def _standalone(case, experience, output, *, no_sdk_default_stage):
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock
    from isaacsim import SimulationApp

    config, options = standalone_launch_options(experience, no_sdk_default_stage)
    with native_instance_lock():
        app = SimulationApp(config, **options)
        try:
            import carb.settings
            import omni.usd

            if carb.settings.get_settings().get_as_bool("/app/fastShutdown"):
                raise RuntimeError("unexpected fast shutdown mode")
            if no_sdk_default_stage:
                effective = app.config.get("create_new_stage")
                stage_present = bool(omni.usd.get_context().get_stage())
                _write_json(output / "sdk_default_stage.json", {
                    "case": case, "requested_create_new_stage": False,
                    "effective_create_new_stage": effective, "sdk_default_stage_present": stage_present,
                    "before_probe_stage_creation": True,
                })
                if effective is not False or stage_present:
                    raise RuntimeError("SDK default stage suppression did not take effect")
            if case != "V0":
                _standalone_capture(app, omni.usd.get_context(), case, output)
        finally:
            import omni.usd

            if case != "V0":
                omni.usd.get_context().close_stage()
            gc.collect()
            app.close(wait_for_replicator=False)
            _write_json(output / "app_close_returned.json", {"app_close_returned": True,
                                                               "process_exit_verified": False})


def _task_capture(bridge, calibration, output):
    import carb.settings
    import omni.replicator.core as rep
    import omni.usd
    import torch
    from pxr import Gf, UsdGeom

    if carb.settings.get_settings().get_as_bool("/app/fastShutdown"):
        raise RuntimeError("unexpected fast shutdown mode")
    stage = omni.usd.get_context().get_stage()
    sim = bridge.world.sim
    world = bridge.device_world
    width, height = calibration.width, calibration.height
    matrices = {"left": calibration.K_left.numpy(), "right": calibration.K_right.numpy()}
    transforms = {"left": calibration.T_world_left.numpy().copy(),
                  "right": calibration.T_world_right.numpy().copy()}
    origin = world.origins[0].detach().cpu().numpy()
    marker = UsdGeom.Cube.Define(stage, MARKER)
    marker.CreateSizeAttr(.025)
    marker.CreateDisplayColorAttr([Gf.Vec3f(1., .1, .1)])
    position = marker.AddTranslateOp()
    for side, transform in transforms.items():
        transform[:3, 3] += origin
        matrices[side] = _author_camera(
            stage, f"/World/DH116Lifecycle_{side}", matrices[side], transform, width, height)

    def set_marker(x):
        transform = transforms["left"]
        location = transform[:3, :3] @ np.array([x, -.08, .40]) + transform[:3, 3]
        position.Set(Gf.Vec3d(*location.tolist()))

    def snapshot():
        arrays = [tensor.detach().cpu().numpy() for hand in world.hands
                  for tensor in (hand.data.joint_pos, hand.data.joint_vel, hand.data.root_pose_w)]
        arrays += [tensor.detach().cpu().numpy() for tensor in
                   (world.wire.data.root_pose_w, world.wire.data.root_vel_w)]
        physical_sha = _state_sha(arrays)
        identities = bridge.pre_action_identities(calibration.sha256)
        return {
            "physics_tick": int(world._physics_steps),
            "sdk_step": int(sim.current_time_step_index),
            "rod_step": None,
            "simulation_time": float(sim.current_time),
            "physics_state_sha256": physical_sha,
            "state_sha256": _state_sha([*arrays, np.asarray(position.Get())]),
            "episode_uid": identities[0]["episode_uid"],
            "reset_id": identities[0]["reset_id"],
            "calibration_sha256": calibration.sha256,
            "state_scope": "existing_rigid_proxy_with_nonphysical_visual_marker",
        }

    set_marker(-.02)
    with ExitStack() as stack:
        stack.callback(stage.RemovePrim, MARKER)
        for side in transforms:
            stack.callback(stage.RemovePrim, f"/World/DH116Lifecycle_{side}")
        cameras = {side: _attached(stack, rep, f"/World/DH116Lifecycle_{side}", width, height,
                                   FULL_PRODUCTS) for side in transforms}
        first = _capture_pair(output, "pre_action", snapshot, sim.render, cameras, matrices, width, height)
        (_, _, done, _) = bridge.step(torch.zeros(bridge.n, 24, device=bridge.device))
        if bool(done.any()):
            raise RuntimeError("task probe terminated before the second capture boundary")
        set_marker(.02)
        second = _capture_pair(output, "post_zero_action", snapshot, sim.render, cameras, matrices, width, height,
                               warmup=False)
    before, after = first["state"], second["state"]
    if (after["physics_tick"] <= before["physics_tick"] or after["sdk_step"] <= before["sdk_step"]
            or after["physics_state_sha256"] == before["physics_state_sha256"]
            or (after["episode_uid"], after["reset_id"]) != (before["episode_uid"], before["reset_id"])):
        raise ValueError("known dynamic rigid task interval did not advance consistently")
    displacement = {}
    for side in cameras:
        centres = []
        for capture in (first, second):
            images = capture["captures"][side]
            identifiers = [int(key) for key, value in images["idToLabels"].items()
                           if value == MARKER or isinstance(value, str) and value.startswith(MARKER + ".")]
            pixels = np.where(np.isin(images["native_instance_id"], identifiers))
            if not len(pixels[1]):
                raise ValueError(f"moving marker not observed in {side} native IDs")
            centres.append(float(pixels[1].mean()))
        displacement[side] = centres[1] - centres[0]
    if min(displacement.values()) <= 2.0:
        raise ValueError("both cameras did not observe the known positive-X marker displacement")
    _write_json(output / "task_capture_result.json", {
        "physical_tick_before": before["physics_tick"], "physical_tick_after": after["physics_tick"],
        "marker_displacement_px": displacement,
        "capture_synchronization_verified": True, "depth_semantics_verified": True,
        "task_kind": "existing_rigid_proxy", "flexible_rod_capture_verified": False,
    })
    return {"status": "bounded_rigid_task_camera_probe"}


def _run_task(output, args, calibration):
    from go2_pvcnn.control.dh116.tabletop_stereo_native import launch_native

    spec = json.loads(args.native_task.read_text())
    if spec.get("task_kind") != "rigid_proxy":
        raise ValueError("V4 reuses only the existing rigid-proxy native task")
    if any(getattr(calibration, side).abs().any() for side in ("distortion_left", "distortion_right")):
        raise ValueError("pinhole depth audit requires undistorted camera calibration")
    index, = [index for index, arg in enumerate(spec["arguments"]) if arg.startswith("--kit_args=")]
    spec["arguments"][index] += " --/app/fastShutdown=false"
    spec["arguments"].extend(("--experience", str(args.experience.resolve())))
    config = output / "derived-native-task.json"
    _write_json(config, spec)
    launch_native(lambda bridge: _task_capture(bridge, calibration, output),
                  native_arguments=config, output=output / "task", num_envs=2, enabled=True)
    _write_json(output / "app_close_returned.json", {"app_close_returned": True,
                                                       "process_exit_verified": False})


def audit_record(output, case, outer_exit):
    close_record = output / "app_close_returned.json"
    close_returned = (json.loads(close_record.read_text()).get("app_close_returned") is True
                      if close_record.is_file() else None)
    artifact_ok = synchronized = depth = False
    errors = []
    try:
        plan = json.loads((output / "plan.json").read_text())
        if plan["case"] != case:
            raise ValueError("capture plan case mismatch")
        if case == "V1":
            from pxr import Usd
            artifact_ok = bool(Usd.Stage.Open(str(output / "probe_stage.usda")))
        if case == "V2":
            rgb = np.load(output / "rgb.npy", allow_pickle=False)
            artifact_ok = rgb.dtype == np.uint8 and rgb.shape in ((64, 96, 3), (64, 96, 4))
        if case in ("V3", "V4"):
            names = ("plane",) if case == "V3" else ("pre_action", "post_zero_action")
            sides = {"single"} if case == "V3" else {"left", "right"}
            for label in names:
                record = json.loads((output / f"{label}-boundary.json").read_text())
                if set(record["cameras"]) != sides:
                    raise ValueError("capture boundary missing camera sides")
                if case == "V4" and record["state"]["calibration_sha256"] != plan["calibration_sha256"]:
                    raise ValueError("task capture calibration identity differs")
                _audit_render_clock(record, sides)
                width, height = STANDALONE_RESOLUTION if case == "V3" else plan["capture_resolution"]
                plane = PLANE_OPTICAL_Z_M if case == "V3" else None
                for side in sides:
                    _audit_saved_products(output / f"{label}-{side}.npz", record["cameras"][side],
                                          width, height, plane)
                if (record["capture_synchronization_verified"] is not True
                        or record["depth_semantics_verified"] is not True):
                    raise ValueError("capture boundary does not carry both source readback verdicts")
            artifact_ok = synchronized = depth = True
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))

    return {
        "case": case,
        **lifecycle_status(artifact_integrity=artifact_ok, synchronized=synchronized,
                           optical_depth=depth, app_close_returned=close_returned,
                           outer_exit_code=outer_exit),
        "dynamic_rigid_task_boundary_verified": case == "V4" and artifact_ok
                                                 and (output / "task_capture_result.json").is_file(),
        "artifact_audit_errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path,
                        default=ROOT / "configs/dh116_training/stereo_camera_front_224x96_v3.json")
    parser.add_argument("--native-task", type=Path,
                        default=ROOT / "configs/dh116_training/stereo_native_task_v1.json")
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--no-sdk-default-stage", action="store_true",
                        help="V0–V3 only: suppress SimulationApp's default empty stage")
    parser.add_argument("--audit-outer-exit", type=Path,
                        help="audit existing output against an independently recorded child exit code")
    args = parser.parse_args()
    if args.no_sdk_default_stage and args.case == "V4":
        parser.error("--no-sdk-default-stage does not change the V4 task bootstrap")
    if args.audit_outer_exit is not None:
        if args.execute_native:
            parser.error("--audit-outer-exit cannot be combined with --execute-native")
        status = audit_record(args.output, args.case, int(args.audit_outer_exit.read_text().strip()))
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    _, options = app_launch_options(args.experience)
    output = args.output.resolve()
    if output.exists() or args.output.is_symlink():
        raise FileExistsError(output)
    calibration = StereoCalibration.from_json(args.calibration)
    plan = {
        "case": args.case, "headless_experience": options["experience"],
        "calibration_sha256": calibration.sha256 if args.case == "V4" else None,
        "capture_resolution": [calibration.width, calibration.height] if args.case == "V4" else None,
        "task_config_sha256": _sha(args.native_task.read_bytes()) if args.case == "V4" else None,
        "step_budget": 1 if args.case == "V4" else 0,
        "capture_budget": 2 if args.case == "V4" else int(args.case != "V0"),
        "legacy_generator_used": False,
    }
    if args.no_sdk_default_stage:
        plan["requested_create_new_stage"] = False
    if not args.execute_native:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    output.mkdir(parents=True)
    _write_json(output / "plan.json", plan)
    try:
        if args.case == "V4":
            _run_task(output, args, calibration)
        else:
            _standalone(args.case, args.experience, output,
                        no_sdk_default_stage=args.no_sdk_default_stage)
    except Exception:
        _write_json(output / "failure.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
