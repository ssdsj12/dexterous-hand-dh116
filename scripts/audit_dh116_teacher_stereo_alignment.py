#!/usr/bin/env python3
"""Reproduce spatial and timing checks from Teacher stereo evidence without Isaac."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_geometry import world_to_palms
from go2_pvcnn.control.dh116.tabletop_teacher_stereo_alignment import STUDENT_CAMERA, _finite, _projection
from go2_pvcnn.control.dh116.tabletop_teacher_stereo_sync import student_rgb_timing
from go2_pvcnn.control.dh116.tabletop_tiled_stereo_capture import _clock


SIDES = ("left", "right")
EPISODE_KEYS = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid", "material_reference_uid")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values):
    return {key: float(value) for key, value in (
        ("mean", np.mean(values)), ("rmse", np.sqrt(np.mean(np.square(values)))),
        ("p95", np.percentile(values, 95)), ("max", np.max(values)))}


def _reprojection(labels, receipt_sha, cameras, projections, width, height):
    if labels["receipt_sha256"] != receipt_sha or labels["source"] != "manual_rgb_annotation":
        raise ValueError("manual RGB annotations must be bound to this receipt")
    errors = {side: [] for side in SIDES}
    seen = set()
    for row in labels["observations"]:
        side, node = row["side"], row["node_index"]
        if (side not in SIDES or type(node) is not int or not 0 <= node < 41
                or (side, node) in seen or row["rgb_sha256"] != cameras[side]["rgb_sha256"]):
            raise ValueError("invalid, duplicate or unbound RGB annotation")
        seen.add((side, node))
        pixel = _finite(row["pixel_uv"], (2,))
        projection = projections[side]
        if (not projection["rod_in_frame"][node] or not 0 <= pixel[0] < width
                or not 0 <= pixel[1] < height):
            raise ValueError("annotation and 3D landmark must be in frame")
        error = np.linalg.norm(pixel - projection["rod_pixels_uv"][node])
        errors[side].append((node, float(error)))
    result = {}
    for side, rows in errors.items():
        if not rows:
            result[side] = dict(samples=0, all_strictly_under_2px=None)
            continue
        values = np.asarray([error for _, error in rows])
        result[side] = dict(samples=len(rows), all_strictly_under_2px=bool((values < 2.0).all()),
                            points=[dict(node_index=node, error_px=error) for node, error in rows],
                            **{key+"_px": value for key, value in _summary(values).items()})
    return result


def audit_frame(path, calibration, labels=None):
    path = Path(path)
    raw = path.read_bytes()
    receipt_sha = hashlib.sha256(raw).hexdigest()
    receipt = json.loads(raw)
    if receipt["schema"] != "dh116-teacher-student-stereo-alignment-v1":
        raise ValueError("unsupported Teacher stereo receipt")
    identity, timing, cameras = (receipt[key] for key in ("identity", "render_timing", "cameras"))
    for key in ("env_id", "reset_id", "policy_tick", "physics_tick_end", "sdk_step"):
        if type(identity[key]) is not int or identity[key] < 0:
            raise ValueError(f"invalid integer identity: {key}")
    for key in ("run_uuid", "episode_uid", "rod_uid", "material_reference_uid"):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ValueError(f"invalid identity: {key}")
    width, height = calibration.width, calibration.height
    if (identity["env_id"] != 0 or receipt["resolution_wh"] != [width, height]
            or receipt["calibration_sha256"] != calibration.sha256
            or identity["calibration_sha256"] != calibration.sha256):
        raise ValueError("single env-0 receipt must match the Student calibration")
    timestamp = _finite(identity["sdk_timestamp_s"], ()).item()
    epoch = _clock(timing["raw_left_clock"])
    if (timestamp < 0 or timestamp != identity["simulation_timestamp"]
            or timestamp != timing["physical_timestamp_s"]
            or identity["sdk_step"] != timing["physical_step"]
            or identity["sdk_step"] != identity["physics_tick_end"]
            or _clock(timing["raw_right_clock"]) != epoch
            or type(timing["capture_id"]) is not int or timing["capture_id"] < 1):
        raise ValueError("Teacher and paired RGB clocks differ")

    _finite(receipt["teacher_obs67"], (67,))
    nodes = _finite(receipt["rod_nodes_world_m"], (41, 3))
    palms = _finite(receipt["palm_poses_world_xyz_wxyz"], (2, 7))
    local = world_to_palms(nodes[None], palms[None]).numpy()[0]
    intrinsics = [_finite(cameras[s]["K"], (3, 3))[None] for s in SIDES]
    transforms = [_finite(cameras[s]["T_world_from_optical"], (4, 4))[None] for s in SIDES]
    calibration.validate_readback(*intrinsics, *transforms, origins=[[0., 0., 0.]])
    projections = {}
    for index, side in enumerate(SIDES):
        camera = cameras[side]
        rgb = path.parent / camera["rgb_file"]
        if (rgb.name != camera["rgb_file"] or rgb.is_symlink()
                or file_sha(rgb) != camera["rgb_sha256"]):
            raise ValueError(f"{side} RGB file identity differs")
        with Image.open(rgb) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (width, height):
                raise ValueError(f"{side} RGB image format differs")
            image.load()
        projection = _projection(nodes, intrinsics[index][0], transforms[index][0], width, height)
        expected_pixels = projection["rod_pixels_uv"]
        stored_pixels = camera["rod_pixels_uv"]
        pixels_match = len(stored_pixels) == len(expected_pixels) and all(
            stored is None if expected is None else
            stored is not None and np.allclose(_finite(stored, (2,)), expected, atol=1e-7, rtol=0)
            for stored, expected in zip(stored_pixels, expected_pixels)
        )
        if (not pixels_match
                or camera["rod_in_frame"] != projection["rod_in_frame"]
                or not np.allclose(_finite(camera["rod_optical_z_m"], (41,)),
                                   projection["rod_optical_z_m"], atol=1e-9, rtol=0)
                or not np.allclose(_finite(receipt["rod_nodes_palm_m"][side], (41, 3)),
                                   local[index], atol=1e-7, rtol=0)):
            raise ValueError(f"{side} stored geometry differs from recomputed geometry")
        projections[side] = projection

    return dict(receipt_sha256=receipt_sha, identity=identity, capture_id=timing["capture_id"],
                renderer_clock=epoch, privileged_state_digest_verified=False,
                reprojection=None if labels is None else _reprojection(
                    labels, receipt_sha, cameras, projections, width, height))


def audit_sequence(frames):
    """Check one consecutive, ordered episode fragment under 60/480 Hz."""
    if not frames:
        raise ValueError("at least one Teacher stereo receipt is required")
    first = frames[0]
    episode = tuple(first["identity"][k] for k in EPISODE_KEYS)
    ids = set()
    timestamps = []
    for index, frame in enumerate(frames):
        current = frame["identity"]
        if tuple(current[k] for k in EPISODE_KEYS) != episode:
            raise ValueError("episode or reset identity changed")
        if frame["capture_id"] in ids:
            raise ValueError("repeated capture identity")
        ids.add(frame["capture_id"])
        if index == 0:
            continue
        previous = frames[index-1]
        prior = previous["identity"]
        if current["policy_tick"] - prior["policy_tick"] != 1:
            raise ValueError("non-consecutive policy tick")
        if frame["renderer_clock"]["render_frame_number"] <= previous["renderer_clock"]["render_frame_number"]:
            raise ValueError("non-increasing renderer epoch")
        dt = current["sdk_timestamp_s"] - prior["sdk_timestamp_s"]
        if current["sdk_step"] - prior["sdk_step"] != 8 or abs(dt - 1/60) > 1e-6:
            raise ValueError("non-consecutive 480 Hz physics boundary")
        timestamps.append(dt * 1000)
    return dict(frames=len(frames), intervals=len(timestamps),
                physical_delta_ms=_summary(timestamps) if timestamps else None,
                max_physical_jitter_ms=max((abs(dt-1000/60) for dt in timestamps), default=None),
                wall_capture_latency_ms=None, complete_episodes_verified=0, actions_verified=0)


def audit_reset_clock_origin(frames, reset_ledger, expected_sha):
    """Test a reset-relative origin against saved frames without qualifying them."""
    audit_sequence(frames)
    first = frames[0]["identity"]
    ledger = Path(reset_ledger)
    if (len(frames) < 2 or first["policy_tick"] != 0 or first["run_uuid"] != ledger.parent.name
            or ledger.name != "ownership.jsonl" or ledger.is_symlink()):
        raise ValueError("Reset clock audit requires consecutive frames from the start of the same run")
    data = ledger.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Reset ledger SHA differs")
    phases = ("episode_reset_before", "episode_reset_after")
    events = [json.loads(line) for line in data.splitlines()]
    resets = [event for event in events if event.get("phase") in phases]
    if tuple(event["phase"] for event in resets) != phases:
        raise ValueError("Reset ledger must record exactly one ordered reset boundary")
    anchor = resets[1]["sdk_clock"]
    if (resets[0]["sdk_clock"] != anchor or type(anchor["sdk_step"]) is not int
            or anchor["sdk_step"] != first["sdk_step"]
            or anchor["sdk_time_s"] != first["sdk_timestamp_s"]):
        raise ValueError("Reset ledger does not match the first physical boundary")
    offset = anchor["sdk_time_s"]
    rows = []
    for frame in frames:
        identity = frame["identity"]
        raw = frame["renderer_clock"]["simulation_time"]
        sdk = identity["sdk_timestamp_s"]
        solver = _finite(identity["solver_time_s"], ()).item()
        original = student_rgb_timing(raw, sdk)
        candidate = student_rgb_timing(raw+offset, sdk)
        rows.append(dict(receipt_sha256=frame["receipt_sha256"], capture_id=frame["capture_id"],
                         sdk_step=identity["sdk_step"], raw_rgb_s=raw, sdk_s=sdk, solver_s=solver,
                         raw_rgb_sdk_delta_ms=original["rgb_sdk_delta_ms"],
                         raw_within_2ms=original["student_sync_usable"],
                         candidate_rgb_sdk_delta_ms=candidate["rgb_sdk_delta_ms"],
                         candidate_within_2ms=candidate["student_sync_usable"],
                         rgb_solver_delta_ms=(raw-solver)*1000))
    return dict(reset_ledger_sha256=expected_sha, reset_sdk_step=anchor["sdk_step"], reset_sdk_time_s=offset,
                candidate_offset_ms=offset*1000, frames=rows, candidate_only=True,
                native_synchronization_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path, help="one episode fragment in capture order")
    parser.add_argument("--calibration", type=Path, default=Path(__file__).resolve().parents[1] / STUDENT_CAMERA)
    parser.add_argument("--annotations", type=Path, help="manual annotations for a single supplied receipt")
    parser.add_argument("--reset-ledger", type=Path, help="ownership.jsonl from the same sealed run")
    parser.add_argument("--reset-ledger-sha256", help="expected SHA256 of the reset ledger")
    args = parser.parse_args()
    if bool(args.reset_ledger) != bool(args.reset_ledger_sha256):
        parser.error("--reset-ledger and --reset-ledger-sha256 must be supplied together")
    if args.reset_ledger and any(path.resolve().parent.parent != args.reset_ledger.resolve().parent
                                 for path in args.receipts):
        parser.error("Reset ledger and frame receipts must share the native run directory")
    if args.annotations and len(args.receipts) != 1:
        parser.error("--annotations requires exactly one receipt")
    calibration = StereoCalibration.from_json(args.calibration)
    labels = json.loads(args.annotations.read_text()) if args.annotations else None
    frames = [audit_frame(path, calibration, labels) for path in args.receipts]
    result = dict(frames=frames, sequence=audit_sequence(frames))
    if args.reset_ledger:
        result["reset_clock_origin"] = audit_reset_clock_origin(frames, args.reset_ledger,
                                                               args.reset_ledger_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
