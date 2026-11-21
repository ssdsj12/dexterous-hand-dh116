#!/usr/bin/env python3
"""Screen bilateral Pinch approach using archived own-hand poses and a settled free rod."""

import argparse
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from scripts import screen_dh116_tripod_measured_precontact as measured

CONTRACT = measured.ROOT / "docs/contracts/dh116_teacher_pinch_supported_reset_native_v1.json"
REQUEST = measured.ROOT / "configs/dh116_tasks/flexible_pinch_supported_approach_v1.json"
SDK = 240
WINDOWS = 24
PADS = ("finger14_link", "finger23_link")


def archive(receipt):
    expected = json.loads(CONTRACT.read_bytes())["actual"]
    outer = measured.sources.checked_json(receipt, expected["outer_sha256"])
    raw = receipt.parent / "native"
    close_path, support_path = raw / "app_close_returned.json", raw / "pinch_support_scene.json"
    def load(path):
        return measured.sources.checked_json(path, outer["artifact_sha256"][path.name])

    close, scene = load(close_path), load(support_path)
    if (outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or outer["forced_exit"] or outer["exceeded_budget"]
            or close["app_close_returned"] is not True
            or scene["no_hidden_rod_pins"] is not True or scene["authored_rod_force"] is not False):
        raise ValueError("naturally closed original supported Pinch Reset required")
    paths = [raw / f"pinch_exchange_{tick}.json" for tick in range(SDK - 8, SDK + 1)]
    rows = []
    for tick, path in enumerate(paths, SDK - 8):
        row = load(path)
        if (row["rod"]["physics_step"] != tick or row["rod"]["rod_tick"] != (tick - 2) * 12
                or row["preceding_reaction_sdk_step"] != tick - 1
                or not row["rod"]["clock_valid"] or row["rod"]["contact_valid"] is not True
                or {pair["mesh_name"] for pair in row["rod"]["rod_mesh_contact_pairs"]["pairs"]}
                != {"support_0", "support_1"}):
            raise ValueError("Pinch Reset window must contain only both supports at known SDK ticks")
        rows.append(row)
    if measured.native.sha(paths[-1]) != expected["final_exchange_sha256"]:
        raise ValueError("Pinch settled end-state reference differs")
    return rows, scene["supports"], [CONTRACT, receipt, close_path, support_path, *paths]


def settled_control_state(row, config):
    hands = {}
    for side in measured.SIDES:
        native_hand = row["native_joint_state"][side]
        names = native_hand["names"]
        if any(names.count(name) != 1 for name in ACTIVE):
            raise ValueError("original own-hand active names required")
        indices = [names.index(name) for name in ACTIVE]
        hands[side] = dict(joint_names=names, active_indices=indices,
                           q=native_hand["position"], backend_targets=native_hand["target"])
    return dict(action_readback=dict(post=dict(hands=hands)),
                sdk_hand_geometry=dict(physics_step=row["rod"]["physics_step"], poses=row["hand_link_poses"]),
                wrist_projection=dict(projected_root_poses=np.asarray(
                    [[config["hands"][side]["pose_wxyz"] for side in measured.SIDES]], dtype=float),
                    projected_angles=np.zeros((1, 2, 3))))


def screen(receipt):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task
    from scripts.verify_dh116_pinch_supported_reset import REQUEST as RESET_REQUEST

    rows, supports, files = archive(receipt)
    request = json.loads(REQUEST.read_bytes())
    if request["version"] != "dh116-pinch-supported-approach-v1" or set(request["goals"]) != set(measured.SIDES):
        raise ValueError("separate left and right measured Pinch targets required")
    rows = [decode_owned(row) for row in rows]
    trajectory = np.stack([row["rod"]["node_positions"][0] for row in rows])
    tube = measured.archived_tube(trajectory)
    boxes = [(np.asarray(box["center_m"]) - np.asarray(box["size_m"]) / 2,
              np.asarray(box["center_m"]) + np.asarray(box["size_m"]) / 2) for box in supports]
    task = load_flexible_grasp_task(measured.ROOT / "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json", measured.ROOT)
    table = task["scene"]
    table_top = table["table_center_m"][2] + .5 * table["table_size_m"][2]
    state = settled_control_state(rows[-1], json.loads(RESET_REQUEST.read_bytes())["config"])
    grids = scene.load_sdf_arrays(scene.prepare_scene(measured.ROOT))
    samples = {side: measured.measure_side(side, state, trajectory[-1], tube, boxes, grids,
                 request["goals"][side], table_top, pads=PADS, max_delta_m=.004, max_delta_rad=.06)
               for side in measured.SIDES}
    return dict(version=request["version"], request_sha256=measured.native.sha(REQUEST),
                outer_sha256=measured.native.sha(receipt), last_row_sha256=measured.native.sha(files[-1]),
                before_sdk=SDK, route_sample_count=measured.SAMPLES, samples=samples,
                all_passed=all(measured.check_route(values, pad_count=len(PADS)) for values in samples.values()),
                native_contact_verified=False, teacher_admission=False)


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
