#!/usr/bin/env python3
"""Screen right Pinch first contact from its measured supported approach end."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import screen_dh116_tripod_measured_precontact as measured
from scripts.screen_dh116_tripod_right_first_contact import route_passes

CONTRACT = measured.ROOT / "docs/contracts/dh116_teacher_pinch_supported_guarded_approach_native_v1.json"
REQUEST = measured.ROOT / "configs/dh116_tasks/flexible_pinch_right_first_contact_v1.json"
SDK = 432
DRIVE_WINDOWS = 24
COAST_WINDOWS = 4
STATIONS = {"right.finger14_link": (2, 3, 4), "right.finger23_link": (1, 2, 3)}
PADS = ("finger14_link", "finger23_link")
MAX_PAD_SDF_PENETRATION_M = .00025


def read_request():
    request = json.loads(REQUEST.read_bytes())
    if (request["version"] != "dh116-pinch-right-first-contact-v1"
            or set(request["goals"]) != set(measured.SIDES)
            or request["target_stations"] != {key: list(v) for key, v in STATIONS.items()}):
        raise ValueError("the independent right Pinch grasp station is required")
    return request


def screen(receipt):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task

    rows, supports, artifacts = measured.archive(receipt, contract=CONTRACT, sdk=SDK, task="pinch")
    request = read_request()
    rows = [measured.decode_owned(row) for row in rows]
    nodes = np.stack([row["rod"]["node_positions"][0] for row in rows])
    tube = measured.archived_tube(nodes)
    boxes = [(np.asarray(box["center_m"]) - np.asarray(box["size_m"]) / 2,
              np.asarray(box["center_m"]) + np.asarray(box["size_m"]) / 2) for box in supports]
    table = load_flexible_grasp_task(measured.ROOT / "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json", measured.ROOT)["scene"]
    table_top = table["table_center_m"][2] + .5 * table["table_size_m"][2]
    grids = scene.load_sdf_arrays(scene.prepare_scene(measured.ROOT))
    samples = {side: measured.measure_side(side, rows[-1], nodes[-1], tube, boxes, grids,
               request["goals"][side], table_top,
               pads=PADS, contact_pads=PADS if side == "right" else ()) for side in measured.SIDES}
    return dict(version=request["version"], request_sha256=measured.native.sha(REQUEST),
                outer_sha256=measured.native.sha(receipt), last_row_sha256=measured.native.sha(artifacts[-2]),
                before_sdk=SDK, samples=samples, all_passed=route_passes(samples, pads=PADS, stations=STATIONS),
                native_contact_verified=False, grasp_qualified=False, teacher_admission=False)


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
