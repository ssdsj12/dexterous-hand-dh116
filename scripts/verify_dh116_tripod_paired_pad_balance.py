"""One vector of measured, wrist-fixed Tripod seating commands."""
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts import screen_dh116_tripod_measured_precontact as measured

CONTRACT = measured.ROOT / "docs/contracts/dh116_teacher_tripod_left_index_stabilization_native_v1.json"
REQUEST = measured.ROOT / "configs/dh116_tasks/flexible_tripod_left_coordinated_pad_balance_v1.json"
SOURCE_SDK = 1704
JOINTS = ("finger11", "finger12", "finger21", "finger31")
SLOTS = tuple(ACTIVE.index(name) for name in JOINTS)
PADS = ("thumb", "index", "middle")
DOSE_WINDOWS, HOLD_WINDOWS = 12, 32


def select_coordinated_command(planned, evidence, *, side="left"):
    """Keep the screened joint ratio, or hold all four joints once fully loaded."""
    issued = np.asarray(planned, dtype=float)
    if (side != "left" or issued.shape != (24,) or not np.isfinite(issued).all()
            or np.any(issued < 0.) or np.any(issued > 1.)
            or any(issued[slot] for slot in range(24) if slot not in SLOTS)):
        raise ValueError("left Tripod coordinated seating requires only its four physical joint slots")
    if evidence.get("wrong_side_contacts") != [] or evidence.get("unexpected_hand_contacts") != []:
        raise ValueError("coordinated seating requires the original correct-side contact layout")
    hand = evidence["per_hand"]["left"]
    rows = hand["per_substep"]
    if len(rows) != 12:
        raise ValueError("complete 12-substep current solver load required")
    loads = np.asarray([[row["per_pad"][pad]["normal_load_n"] for pad in PADS] for row in rows])
    if not np.isfinite(loads).all() or np.any(loads < 0.):
        raise ValueError("coordinated seating requires measured per-pad normal force")
    totals = np.asarray([row["total_normal_load_n"] for row in rows], dtype=float)
    if (not np.isfinite(totals).all() or not np.allclose(totals, loads.sum(axis=1), atol=1e-10, rtol=1e-9)
            or type(hand["loaded"]) is not bool):
        raise ValueError("actual consistent hand total force required")
    if hand["loaded"] and (np.any(loads <= .05) or np.any(totals <= .3)
            or any(row.get("effective") is not True or row.get("opposition_cos") is None
                   or not -1. <= row["opposition_cos"] <= -.25 for row in rows)):
        raise ValueError("loaded flag must retain original force and opposition qualification")
    minimum = dict(zip(PADS, loads.min(axis=0).tolist(), strict=True))
    return (np.zeros_like(issued) if hand["loaded"] else issued.copy()), minimum


def load_coordinated_seating(receipt, actions, *, start_sdk):
    source = json.loads(CONTRACT.read_bytes())["actual"]
    if source["end_sdk"] != SOURCE_SDK or source["longest_consecutive_bilateral_sdk_steps"] != 1:
        raise ValueError("requires the measured fingertip phase-drift capture")
    rows, _, lineage = measured.archive(receipt, contract=CONTRACT, sdk=SOURCE_SDK)
    anchor = rows[-1]
    if (anchor["physics_step"] != SOURCE_SDK or anchor["rod"]["physics_step"] != SOURCE_SDK
            or anchor["sdk_hand_geometry"]["physics_step"] != SOURCE_SDK
            or anchor["rod"]["rod_tick"] != (SOURCE_SDK-2)*12 or anchor["guard_failure"] is not None):
        raise ValueError("coordinated seating requires actual same-clock hand and rod states")
    req = measured.sources.checked_json(REQUEST, "33265e8803a0904cc1366e5cff62ebfc2f5cc8281749ddce24d00ae3bfc7c063")
    if (req["version"] != "dh116-tripod-left-coordinated-pad-balance-v1" or req["source_sdk"] != SOURCE_SDK
            or (req["dose_windows"],req["hold_windows"]) != (DOSE_WINDOWS,HOLD_WINDOWS)
            or set(req["travel_rad"]) != set(JOINTS)):
        raise ValueError("frozen four-joint Tripod dose required")
    travel = np.array([req["travel_rad"][name] for name in JOINTS])
    if not np.isfinite(travel).all() or np.any(travel <= 0.) or np.any(travel > .0013):
        raise ValueError("coordinated command exceeds its offline screened travel")
    if start_sdk + 8*len(actions) != SOURCE_SDK:
        raise ValueError("the original right and left seating rollout must replay through the exact source")
    continued = np.zeros((DOSE_WINDOWS+HOLD_WINDOWS,24))
    continued[:DOSE_WINDOWS, SLOTS] = travel / (DOSE_WINDOWS*ACTION_CONTRACT["joint_rate_rad_s"]/60)
    stage = dict(sdk_step=SOURCE_SDK, stop_sdk=SOURCE_SDK+8*(DOSE_WINDOWS+HOLD_WINDOWS),
                 archive=anchor, side="left", variant="left_coordinated_seating_v1",
                 command_semantics="increment_on_existing_backend_targets_not_measured_q_goal",
                 geometry_evidence_scope="archived_q_kinematic_reachability_not_predicted_native_trajectory")
    return np.vstack((actions,continued)),stage,(*lineage,REQUEST,Path(__file__))
