#!/usr/bin/env python3
"""Audit staged left Tripod closure while holding the measured right grasp target."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import screen_dh116_tripod_measured_precontact as measured
from scripts.screen_dh116_tripod_right_first_contact import MAX_PAD_SDF_PENETRATION_M, STATIONS as RIGHT_STATIONS

CONTRACT = measured.ROOT / "docs/contracts/dh116_teacher_tripod_right_first_contact_native_v1.json"
REQUEST = measured.ROOT / "configs/dh116_tasks/flexible_tripod_bilateral_first_contact_v1.json"
SDK = 752
WINDOWS = (24, 24)
PAUSES = (8, 12)
STATIONS = {"left.finger14_link": (34, 35, 36, 37), "left.finger23_link": (34, 35, 36, 37),
            "left.finger33_link": (32, 33, 34, 35)}


def read_request():
    request = json.loads(REQUEST.read_bytes())
    if (request["version"] != "dh116-tripod-bilateral-first-contact-v1" or len(request["left_waypoints"]) != 2
            or request["target_stations"] != {key: list(v) for key, v in STATIONS.items()}):
        raise ValueError("the exact left-side two-waypoint continuation is required")
    return request


def route_passes(samples):
    if set(samples) != set(measured.SIDES):
        return False
    for side, route in samples.items():
        if len(route) != len(WINDOWS) * measured.SAMPLES:
            return False
        station = STATIONS if side == "left" else RIGHT_STATIONS
        for row in route:
            margins = np.asarray([row["noncontact_rod_clearance_m"], row["support_clearance_m"],
                                  row["table_clearance_m"]], dtype=float)
            gaps = np.asarray(row["pad_gaps_m"], dtype=float)
            if (row["shape_count"] != 22 or not np.isfinite(margins).all()
                    or margins[0] <= measured.ROD_RESERVE_M or margins[1] <= 0. or margins[2] < measured.CLEARANCE_M
                    or gaps.shape != (3,) or not np.isfinite(gaps).all()
                    or (gaps < -MAX_PAD_SDF_PENETRATION_M).any()
                    or any(node not in station[f"{side}.{name}"]
                           for name, node in zip(measured.PADS, row["node_ids"], strict=True))):
                return False
    end = samples["left"][-1]
    return bool(max(end["pad_gaps_m"]) < .0001 and max(end["opposition"]) < -.25)


def screen(receipt):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task

    raw, supports, files = measured.archive(receipt, contract=CONTRACT, sdk=SDK)
    request = read_request()
    prior = [measured.decode_owned(row) for row in raw]
    nodes = np.stack([row["rod"]["node_positions"][0] for row in prior])
    if not any(pair["mesh_name"].startswith("right.") for pair in prior[-1]["rod"]["rod_mesh_contact_pairs"]["pairs"]):
        raise ValueError("a measured right Tripod contact boundary is needed")
    tube = measured.archived_tube(nodes)
    boxes = [(np.asarray(box["center_m"]) - np.asarray(box["size_m"]) / 2,
              np.asarray(box["center_m"]) + np.asarray(box["size_m"]) / 2) for box in supports]
    table = load_flexible_grasp_task(measured.ROOT / "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json", measured.ROOT)["scene"]
    table_top = table["table_center_m"][2] + .5 * table["table_size_m"][2]
    state = prior[-1]
    right = state["action_readback"]["post"]["hands"]["right"]
    xyz, quat = state["sdk_hand_geometry"]["poses"]["right.base_link"]
    pose = np.r_[xyz, np.asarray(quat, dtype=float)[[3, 0, 1, 2]] / np.linalg.norm(np.asarray(quat,dtype=float))]
    mate = dict(pose_wxyz=pose.tolist(), active_q_rad=right["q"][0,right["active_indices"]].tolist())
    goals = dict(left=request["left_waypoints"], right=[mate,mate])
    grids = scene.load_sdf_arrays(scene.prepare_scene(measured.ROOT))
    samples = {side: measured.measure_side(side, state, nodes[-1], tube, boxes, grids, goals[side], table_top,
               contact_pads=measured.PADS, max_delta_m=.004, max_delta_rad=.08)
               for side in measured.SIDES}
    return dict(version=request["version"], request_sha256=measured.native.sha(REQUEST),
                outer_sha256=measured.native.sha(receipt), last_row_sha256=measured.native.sha(files[-2]),
                before_sdk=SDK, samples=samples, all_passed=route_passes(samples),
                grasp_qualified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = screen(args.receipt)
    measured.native.write_new(args.output, result)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
