"""Read-only Student camera evidence at the measured Teacher policy boundary."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .tabletop_deformable_r0_replay import _digest
from .tabletop_stereo_geometry import world_to_palms
from .tabletop_stereo_validation import validate_images
from .tabletop_teacher_preaction_readback import _identity
from .tabletop_teacher_stereo_alignment import _finite, _projection, project_physical_palms
from .tabletop_teacher_stereo_sync import student_rgb_timing
from .tabletop_tiled_stereo_capture import _clock


SIDES = ("left", "right")


def capture_identity(pre, calibration_sha256, *, sdk_timestamp_s=None):
    raw = pre["common"]["identity"]
    clock = _identity(raw)
    step = clock["physics_step"]
    timestamp = clock["timestamp_s"]
    sdk_stamp = timestamp if sdk_timestamp_s is None else sdk_timestamp_s
    if (type(sdk_stamp) not in (int, float)
            or not math.isclose(sdk_stamp, timestamp, abs_tol=1e-6, rel_tol=0)):
        raise ValueError("actual SDK time differs from the Teacher rod clock")
    solver_time = raw["solver_time_s"]
    if raw["clock_valid"] is not True or type(solver_time) not in (float, int) or (
            not math.isfinite(solver_time) or solver_time < 0):
        raise ValueError("measured SDK and rod clock required")
    boundary = pre["rod_boundary"]
    rods = boundary["rod_snapshot"]
    if (boundary["run_uuid"] != clock["run_uuid"] or boundary["rod_uid"] != clock["rod_uid"]
            or boundary["rod_tick"] != clock["rod_tick"] or rods["rod_tick"] != clock["rod_tick"]
            or rods["env_ids"] != [clock["env_id"]] or rods["reset_ids"] != [clock["reset_id"]]
            or rods["episode_uids"] != [clock["episode_uid"]] or rods["rod_uids"] != [clock["rod_uid"]]
            or rods["physics_step"] != step or rods["timestamp_s"] != timestamp):
        raise ValueError("rod snapshot and Teacher pre-action state differ")
    material_uid = f"{clock['run_uuid']}:{clock['episode_uid']}:{clock['reset_id']}:{clock['rod_uid']}"
    return dict(run_uuid=clock["run_uuid"], env_id=clock["env_id"], episode_uid=clock["episode_uid"],
                reset_id=clock["reset_id"], rod_uid=clock["rod_uid"], policy_tick=clock["policy_step"],
                physics_tick_start=step, physics_tick_end=step, sdk_step=step,
                simulation_timestamp=timestamp, sdk_timestamp_s=sdk_stamp, rod_tick=clock["rod_tick"],
                rod_step=clock["rod_tick"], solver_time_s=solver_time, material_reference_uid=material_uid,
                state_sha256=_digest(pre), calibration_sha256=calibration_sha256)


def save_aligned_rgb_pair(pre, packet, calibration, output, *, origins=None, camera_profile="front"):
    from PIL import Image

    timing = packet["render_timing"]
    identity = capture_identity(pre, calibration.sha256, sdk_timestamp_s=timing["physical_timestamp_s"])
    left_clock = _clock(timing["raw_left_clock"])
    capture_id = timing["capture_id"]
    if (packet["identity"] != identity or packet["calibration_sha256"] != calibration.sha256
            or timing["physical_step"] != identity["sdk_step"]
            or timing["physical_timestamp_s"] != identity["sdk_timestamp_s"]
            or left_clock != _clock(timing["raw_right_clock"])
            or type(capture_id) is not int or capture_id < 1):
        raise ValueError("fixed Student RGB pair and Teacher clock/identity differ")
    validate_images(packet["left_rgb"], packet["right_rgb"], calibration)
    if packet["left_rgb"].shape[0] != 1:
        raise ValueError("single Teacher environment required")
    calibration.validate_readback(packet["K_left"], packet["K_right"],
                                  packet["T_world_left"], packet["T_world_right"], origins=origins)
    common = pre["common"]
    sources = common["source_packets"]
    by_side = {source["side"]: source for source in sources}
    if len(sources) != 2 or set(by_side) != set(SIDES) or any(
            _identity(source["identity"]) != _identity(common["identity"]) for source in sources):
        raise ValueError("measured palm poses must belong to the same Teacher boundary")
    palms = _finite([by_side[side]["body_link_pose_w"] for side in SIDES], (2, 7))
    nodes = _finite(pre["rod_boundary"]["rod_snapshot"]["node_positions"], (1, 41, 3))[0]
    local = world_to_palms(nodes[None], palms[None]).numpy()[0]
    values, valid = np.asarray(common["values"]), np.asarray(common["valid"])
    if (values.shape != (371,) or valid.shape != (371,) or valid.dtype != np.bool_
            or not np.isfinite(values[valid]).all()):
        raise ValueError("measured privileged ledger values and validity required")
    cameras = {}
    for side in SIDES:
        k = _finite(packet["K_" + side], (1, 3, 3))[0]
        transform = _finite(packet["T_world_" + side], (1, 4, 4))[0]
        cameras[side] = _projection(nodes, k, transform, calibration.width, calibration.height)
        cameras[side].update(project_physical_palms(
            palms, k, transform, calibration.width, calibration.height))

    folder = Path(output) / f"frame-{capture_id:06d}"
    folder.mkdir()
    for side in SIDES:
        rgb = packet[side+"_rgb"][0].permute(1, 2, 0).detach().cpu().numpy()
        path = folder / f"{side}.png"
        with path.open("xb") as stream:
            Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        cameras[side].update(rgb_file=path.name, rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    sync = student_rgb_timing(left_clock["simulation_time"], identity["sdk_timestamp_s"])
    hand_order_verified = all(camera["physical_left_before_right"] for camera in cameras.values())
    hand_order = dict(physical_hand_order_verified=hand_order_verified,
                      student_hand_order_usable=camera_profile != "hand_order" or hand_order_verified)
    receipt = dict(schema="dh116-native-teacher-student-stereo-v1", identity=identity,
                   camera_profile=camera_profile,
                   privileged_values=[float(v) if ok else None for v, ok in zip(values, valid)],
                   privileged_valid=valid.tolist(), cameras=cameras, render_timing=timing,
                   source_scope=packet["source_scope"], native_qualification=False,
                   rod_nodes_world_m=nodes.tolist(), palm_poses_world_xyz_wxyz=palms.tolist(),
                   rod_nodes_palm_m={side:local[i].tolist() for i,side in enumerate(SIDES)},
                   occlusion_measured=False, **sync, **hand_order)
    path = folder / "alignment.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(receipt_path=str(path.relative_to(output)),
                receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                state_sha256=identity["state_sha256"], capture_id=capture_id,
                calibration_sha256=calibration.sha256, **sync, **hand_order)
