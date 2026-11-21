"""Small, measured seating diagnostics on existing supported-contact replays."""

import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import MIN_HAND_LOAD_N
from scripts import screen_dh116_tripod_measured_precontact as measured

TRIPOD_REQUEST = measured.ROOT / "configs/dh116_tasks/flexible_tripod_right_force_seating_v1.json"
TRIPOD_CONTRACT = measured.ROOT / "docs/contracts/dh116_teacher_tripod_measured_left_retarget_native_v2.json"
TRIPOD_SDK = 1296
PAD_FOR_SLOT = {8: "index", 9: "middle"}
DOSE_THRESHOLD_N = .085


def validate_seating_schedule(schedule, pad_by_slot, *, max_travel_rad):
    schedule = np.asarray(schedule, dtype=float)
    if (not pad_by_slot or any(PAD_FOR_SLOT.get(slot) != pad for slot, pad in pad_by_slot.items())
            or schedule.ndim != 2 or schedule.shape[1] != 24 or not len(schedule)
            or not np.isfinite(schedule).all() or (schedule < 0.).any() or (schedule > 1.).any()
            or any(np.any(schedule[:, slot]) for slot in range(24) if slot not in pad_by_slot)):
        raise ValueError("screened positive finger-only seating commands are required")
    travel = schedule.sum(axis=0) * (ACTION_CONTRACT["joint_rate_rad_s"] / 60)
    if not 0. < max_travel_rad <= .003 or np.any(travel > max_travel_rad + 1e-12):
        raise ValueError("seating exceeds the screened finger travel budget")


def select_seating_command(planned, evidence, pad_by_slot):
    right = evidence["per_hand"]["right"]
    rows = right["per_substep"]
    if len(rows) != 12:
        raise ValueError("complete 12-substep current solver load required")
    combined = np.asarray([row["total_normal_load_n"] for row in rows], dtype=float)
    if not np.isfinite(combined).all() or (combined < 0.).any():
        raise ValueError("seating requires a valid total normal force")
    total_ready = bool((combined > MIN_HAND_LOAD_N).all())
    issued = np.asarray(planned, dtype=float).copy()
    minimums = {}
    for slot, pad in pad_by_slot.items():
        forces = np.asarray([row["per_pad"][pad]["normal_load_n"] for row in rows], dtype=float)
        if not np.isfinite(forces).all() or np.any(forces < 0.):
            raise ValueError("seating requires measured per-pad normal force")
        minimums[pad] = float(forces.min())
        if right["loaded"] or (total_ready and minimums[pad] >= DOSE_THRESHOLD_N):
            issued[slot] = 0.
    return issued, minimums


def load_tripod_seating(receipt, actions, *, start_sdk):
    expected = json.loads(TRIPOD_CONTRACT.read_bytes())
    if expected["force_followup_boundary"]["sdk"] != TRIPOD_SDK:
        raise ValueError("the measured Tripod seating checkpoint is required")
    rows, _, paths = measured.archive(receipt, contract=TRIPOD_CONTRACT, sdk=expected["actual"]["end_sdk"])
    if rows[-1]["physics_step"] <= TRIPOD_SDK:
        raise ValueError("completed retarget must cover the seating checkpoint")
    outer = measured.sources.checked_json(receipt, expected["actual"]["outer_sha256"])
    name = f"tripod_control_{TRIPOD_SDK}.json"
    checkpoint = receipt.parent / "native" / name
    row = measured.sources.checked_json(checkpoint, outer["artifact_sha256"][name])
    if (row["physics_step"] != TRIPOD_SDK or row["sdk_hand_geometry"]["physics_step"] != TRIPOD_SDK
            or row["rod"]["physics_step"] != TRIPOD_SDK or row["rod"]["rod_tick"] != (TRIPOD_SDK - 2) * 12
            or row["guard_failure"] is not None):
        raise ValueError("seating anchor must bind actual fingertip and rod data to the complete SDK step")

    request = json.loads(TRIPOD_REQUEST.read_bytes())
    if (request["version"] != "dh116-tripod-right-force-seating-v1" or request["control_boundary_sdk"] != TRIPOD_SDK
            or request["dose_windows"] != 10 or request["settle_windows"] != 18):
        raise ValueError("expected the frozen Tripod force seating contract")
    command = np.zeros(24)
    command[[8, 9]] = [request["planned_action_by_joint"][name] for name in ("finger21", "finger31")]
    appended = np.vstack((np.repeat(command[None], request["dose_windows"], axis=0),
                          np.zeros((request["settle_windows"], 24))))
    validate_seating_schedule(appended, PAD_FOR_SLOT, max_travel_rad=request["maximum_finger_travel_rad"])
    count = (TRIPOD_SDK - start_sdk) // 8
    if not 0 < count < len(actions) or start_sdk + 8 * count != TRIPOD_SDK:
        raise ValueError("seating requires the intact original replay through SDK 1296")
    return np.vstack((actions[:count], appended)), dict(sdk_step=TRIPOD_SDK, archive=row,
        pad_by_slot={str(slot): pad for slot, pad in PAD_FOR_SLOT.items()}), (
        *paths, Path(__file__), TRIPOD_REQUEST, checkpoint)
