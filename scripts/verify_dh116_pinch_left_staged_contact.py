"""Compile the two-stage left Pinch approach after the measured right seating."""

import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT as WRIST

from scripts import screen_dh116_tripod_measured_precontact as measured
from scripts.verify_dh116_tripod_supported_reset import bilateral_schedule

REQUEST = measured.ROOT / "configs/dh116_tasks/flexible_pinch_left_staged_contact_v1.json"
CONTRACT = measured.ROOT / "docs/contracts/dh116_teacher_pinch_force_guided_index_seating_native_v1.json"
SDK = 848
WINDOWS = (118, 18)
PAUSES = (0, 16)
STATIONS = {"left.finger14_link": [35], "left.finger23_link": [35]}


def validate_left_wrist_workspace(projection, waypoints):
    # Use the measured world-space origin feeding the original wrist transport.
    projected_position = projection["projected_root_poses"][0, 0, :3]
    origin = projected_position - projection["palm_offsets_m"][0, 0]
    lower = origin + WRIST["palm_lower_m"]
    upper = origin + WRIST["palm_upper_m"]
    for index, waypoint in enumerate(waypoints, start=1):
        position = np.asarray(waypoint["pose_wxyz"][:3], dtype=float)
        if (position.shape != (3,) or not np.isfinite(position).all()
                or np.any(position < lower) or np.any(position > upper)):
            raise ValueError(f"left Pinch waypoint {index} outside original wrist workspace: "
                             f"position={position.tolist()}, lower={lower.tolist()}, upper={upper.tolist()}")


def load_left_route(receipt, config):
    src = json.loads(CONTRACT.read_bytes())
    outer = measured.sources.checked_json(receipt, src["outer_sha256"])
    raw = receipt.parent / "native"

    def read(path):
        return measured.sources.checked_json(path, outer["artifact_sha256"][path.name])

    boundary = raw/f"pinch_control_{SDK}.json"
    supports = raw/"pinch_support_scene.json"
    marker = raw/"app_close_returned.json"
    row = read(boundary)
    if (not src["native_natural_exit_verified"] or src["sdk_total"] != SDK
            or outer["forced_exit"] or outer["exceeded_budget"] or outer["exit_code"] != 0
            or outer["debugger_inferior_exit_code"] != 0 or read(marker)["app_close_returned"] is not True
            or measured.native.sha(boundary) != src["last_row_sha256"]
            or row["physics_step"] != SDK or row["rod"]["physics_step"] != SDK
            or row["sdk_hand_geometry"]["physics_step"] != SDK or row["rod"]["rod_tick"] != (SDK-2)*12
            or row["guard_failure"] is not None or read(supports)["supports"] != config["support_scene"]["supports"]):
        raise ValueError("staged left Pinch needs the exact native right seating handoff")
    active = {pair["mesh_name"] for pair in row["rod"]["rod_mesh_contact_pairs"]["pairs"]}
    if not {"right.finger14_link", "right.finger23_link"}.issubset(active) or any(n.startswith("left.") for n in active):
        raise ValueError("Pinch approach anchor requires two right contacts and no left contact")
    request = measured.sources.checked_json(REQUEST, "4d24edbb7d6f07c50bff13ef1ffbe571f06a9cb1dd5265d370701d1751e36753")
    if (request["version"] != "dh116-pinch-left-staged-contact-v1" or request["source_sdk"] != SDK
            or request["target_stations"] != STATIONS):
        raise ValueError("the immutable measured left Pinch path is required")
    decoded = measured.decode_owned(row)
    validate_left_wrist_workspace(decoded["wrist_projection"], request["left_waypoints"])
    commands = bilateral_schedule(config, decoded, request["left_waypoints"], WINDOWS, PAUSES)
    return commands, dict(boundary_sdk=SDK+8*WINDOWS[0], replay_sdk=SDK, stations=STATIONS,
                          midpoint_sdk=SDK+8*WINDOWS[0], midpoint_target=request["left_waypoints"][0],
                          archive=row), (CONTRACT, REQUEST, Path(__file__), receipt, marker, supports, boundary)
