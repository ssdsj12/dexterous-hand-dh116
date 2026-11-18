"""Teacher privileged labels registered to the fixed Student RGB pair."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .tabletop_front_depth_camera import project_points
from .tabletop_stereo_geometry import world_to_palms


STUDENT_CAMERA = "configs/dh116_training/stereo_camera_front_640x480_v3.json"


def state_digest(state):
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite(value, shape):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"finite data of shape {shape} required")
    return value


def _projection(nodes, k, world_from_camera, width, height):
    xyz = (nodes - world_from_camera[:3, 3]) @ world_from_camera[:3, :3]
    front = xyz[:, 2] > 0
    uv = np.zeros((len(nodes), 2), dtype=float)
    if front.any():
        uv[front], _ = project_points(nodes[front], k, world_from_camera)
    in_frame = front & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return dict(K=k.tolist(), T_world_from_optical=world_from_camera.tolist(),
                rod_pixels_uv=[row.tolist() if ok else None for row, ok in zip(uv, front)],
                rod_optical_z_m=xyz[:, 2].tolist(), rod_in_frame=in_frame.tolist())


def aligned_payload(state, expected_identity, pair, calibration, origins):
    """Register geometry from the same digest-bound Teacher observation."""
    timing = pair["render_timing"]
    expected_sha = calibration.sha256
    if (pair["identity"] != expected_identity
            or expected_identity["state_sha256"] != state_digest(state)
            or expected_identity["calibration_sha256"] != expected_sha
            or pair["calibration_sha256"] != expected_sha
            or timing["physical_step"] != expected_identity["sdk_step"]
            or timing["physical_step"] != expected_identity["physics_tick_end"]
            or timing["physical_timestamp_s"] != expected_identity["sdk_timestamp_s"]
            or timing["physical_timestamp_s"] != expected_identity["simulation_timestamp"]
            or timing["raw_left_clock"] != timing["raw_right_clock"]):
        raise ValueError("Teacher state and RGB identities differ")
    calibration.validate_readback(pair["K_left"], pair["K_right"],
                                  pair["T_world_left"], pair["T_world_right"], origins=origins)
    obs = _finite(state["teacher_obs67"], (67,))
    nodes = _finite(state["rod_nodes_world_m"], (41, 3))
    palms = _finite(state["palm_poses_world_xyz_wxyz"], (2, 7))
    local = world_to_palms(nodes[None], palms[None]).numpy()[0]
    cameras = {}
    for side in ("left", "right"):
        k = _finite(pair["K_" + side], (1, 3, 3))[0]
        pose = _finite(pair["T_world_" + side], (1, 4, 4))[0]
        cameras[side] = _projection(nodes, k, pose, calibration.width, calibration.height)
    return dict(schema="dh116-teacher-student-stereo-alignment-v1", teacher_obs67=obs.tolist(),
                rod_nodes_world_m=nodes.tolist(), palm_poses_world_xyz_wxyz=palms.tolist(),
                rod_nodes_palm_m={"left": local[0].tolist(), "right": local[1].tolist()},
                cameras=cameras, identity=expected_identity, render_timing=timing,
                calibration_sha256=expected_sha, resolution_wh=[calibration.width, calibration.height],
                palm_body_link="base_link", optical_axes="x right, y down, z forward",
                point_source="DeformX material nodes, privileged supervision only",
                occlusion_measured=False, teacher_policy_queried=False, native_qualification=False)


def runtime_state(runtime):
    """Read the frozen Teacher input plus its raw geometric provenance."""
    def array(tensor):
        return tensor.detach().cpu().numpy().copy()

    nodes = _finite(array(runtime.rope_pos_view), (1, 41, 3))[0]
    obs = _finite(array(runtime.get_obs()), (1, 67))[0]
    origin = _finite(array(runtime.env_origins), (1, 3))[0]
    reference = _finite(array(runtime.wire_ref_center), (3,))
    node_ids = array(runtime.wire_nodes_idx).tolist()
    if node_ids != [0, 10, 20, 30, 40]:
        raise ValueError("original Teacher wire-node selection differs")
    palms = np.stack([
        array(hand.data.body_link_pose_w)[0, index]
        for hand, index in ((runtime.left_hands, runtime.l_base_id), (runtime.right_hands, runtime.r_base_id))
    ])
    _finite(palms, (2, 7))
    decoded_positions = obs[24:30].reshape(2, 3) / 5 + origin + reference
    decoded_nodes = obs[48:63].reshape(5, 3) / 5 + origin + reference
    if (not np.allclose(decoded_positions, palms[:, :3], atol=2e-7, rtol=0)
            or not np.array_equal(obs[30:38].reshape(2, 4), palms[:, 3:])
            or not np.allclose(decoded_nodes, nodes[node_ids], atol=2e-7, rtol=0)):
        raise ValueError("raw geometry differs from frozen Teacher observation")
    return dict(teacher_obs67=obs.tolist(), rod_nodes_world_m=nodes.tolist(),
                palm_poses_world_xyz_wxyz=palms.tolist(), teacher_reference_env_m=reference.tolist(),
                teacher_wire_node_indices=node_ids, env_origin_world_m=origin.tolist())


def capture_initial_pair(runtime, plan, *, read_clock, write_receipt):
    """Capture one RGB pair at the existing frozen initialization boundary."""
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux, Vt
    from .tabletop_stereo_calibration import StereoCalibration
    from .tabletop_stereo_rgb_bridge import _NativeRenderSetting
    from .tabletop_tiled_stereo_capture import TiledStereoCapture
    from .tabletop_tiled_stereo_native import TiledNativeStereoReader

    root, output = Path(plan["project"]), Path(plan["output"])
    calibration = StereoCalibration.from_json(root / STUDENT_CAMERA)
    limits = plan["teacher_stereo"]
    if calibration.sha256 != limits["calibration_sha256"]:
        raise ValueError("Student calibration changed")
    sim = runtime.sim
    before = read_clock(sim)
    if before["sdk_step"] != 2 or float(runtime.rope.time) != 0:
        raise ValueError("one initialized Teacher endpoint before control is required")
    state = runtime_state(runtime)
    origins = [state["env_origin_world_m"]]
    identity = dict(run_uuid=output.name, env_id=0, episode_uid=output.name+":initialization", reset_id=0,
                    policy_tick=0, physics_tick_start=2, physics_tick_end=2, rod_step=0,
                    state_sha256=state_digest(state), simulation_timestamp=before["sdk_time_s"],
                    calibration_sha256=calibration.sha256, sdk_step=2, sdk_timestamp_s=before["sdk_time_s"],
                    rod_tick=0, solver_time_s=0., material_reference_uid=output.name+":initialized-rod",
                    rod_uid=output.name+":rod")

    def snapshot():
        if read_clock(sim) != before or float(runtime.rope.time) != 0:
            raise RuntimeError("Teacher clocks changed during RGB alignment")
        return dict(identity, state_sha256=state_digest(runtime_state(runtime)))

    # Visual-only curve authored from the very nodes used for the labels.
    curve = UsdGeom.BasisCurves.Define(sim.stage, "/World/DH116TeacherStereoRod")
    curve.CreateTypeAttr().Set("linear")
    curve.CreateCurveVertexCountsAttr().Set([41])
    curve.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(
        np.asarray(state["rod_nodes_world_m"], dtype=np.float32)))
    curve.CreateWidthsAttr().Set([2 * float(runtime.r_cfg["radius_m"])])
    curve.SetWidthsInterpolation("constant")
    curve.CreateDisplayColorAttr().Set([Gf.Vec3f(1., .35, .02)])
    light = UsdLux.DomeLight.Define(sim.stage, "/World/DH116TeacherStereoLight")
    light.CreateIntensityAttr(900.)
    setting = _NativeRenderSetting()
    original = setting.get()
    if type(original) is not bool:
        raise ValueError("readable physics playback setting required")
    reader = camera = None
    try:
        setting.set(False)
        if setting.get() is not False:
            raise RuntimeError("physics freeze failed before camera creation")
        reader = TiledNativeStereoReader(sim, origins, calibration, runtime.device)
        camera = TiledStereoCapture(sim, reader, calibration, snapshot, render_setting=setting,
                                    max_render_attempts=limits["max_render_attempts"], origins=origins)
        packet = camera.capture()
        result = aligned_payload(state, identity, packet, calibration, origins)
        result.update(source_scope=packet["source_scope"], phase="initialization_before_control",
                      privileged_source="original frozen GPU Teacher get_obs()", render_passes=packet["render_passes"])
        for side in ("left", "right"):
            filename = f"teacher_{side}_rgb.png"
            img = packet[side+"_rgb"][0].permute(1, 2, 0).detach().cpu().numpy()
            with (output / filename).open("xb") as stream:
                Image.fromarray(np.rint(img*255).astype(np.uint8)).save(stream, format="PNG")
                stream.flush()
                os.fsync(stream.fileno())
            result["cameras"][side]["rgb_file"] = filename
            result["cameras"][side]["rgb_sha256"] = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        write_receipt(output/"teacher_stereo_alignment.json", result)
    finally:
        try:
            if camera is not None:
                camera.close()
            elif reader is not None:
                reader.close()
        finally:
            setting.set(original)
            camera = reader = runtime = sim = None
    return result
