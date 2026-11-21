"""Screen the left Tripod continuation against the measured deflected rod."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import screen_dh116_tripod_bilateral_first_contact as prior

measured = prior.measured
CONTRACT = measured.ROOT / "docs/contracts/dh116_teacher_tripod_bilateral_distal_native_v1.json"
REQUEST = measured.ROOT / "configs/dh116_tasks/flexible_tripod_bilateral_retarget_v2.json"
SDK = 1136
GUARD_SDK = 1178
DRIVE_WINDOWS = 24
SETTLE_WINDOWS = 16
SCREEN_RESERVE_M = .0008


def diagnostic_boundary_valid(outer, failure, close):
    return (outer["exit_code"] == outer["debugger_inferior_exit_code"] == 1
            and outer["debugger_inferior_signal"] is None and outer["debugger_status_verified"]
            and not outer["forced_exit"] and not outer["exceeded_budget"]
            and close["app_close_returned"] is True and failure["guarded_stop_sdk_step"] == GUARD_SDK
            and "live hand source envelope lost its staged rod clearance" in failure["failure"]
            and failure["grasp_qualified"] is False and failure["teacher_admission"] is False
            and failure["ppo_updates"] == 0)


def completed_rows(receipt):
    expected = json.loads(CONTRACT.read_bytes())["actual"]
    outer = measured.sources.checked_json(receipt, expected["outer_sha256"])
    raw = receipt.parent / "native"
    paths = []

    def read(name):
        path = raw / name
        paths.append(path)
        return measured.sources.checked_json(path, outer["artifact_sha256"][name])

    close = read("app_close_returned.json")
    failure = read("tripod_failure.json")
    supports = read("tripod_support_scene.json")["supports"]
    if not diagnostic_boundary_valid(outer, failure, close):
        raise ValueError("expected the exact Tripod geometry stop and normal SDK closure")
    rows = {}
    for tick in (*range(SDK - 8, SDK + 1), GUARD_SDK - 8, GUARD_SDK - 1):
        row = read(f"tripod_control_{tick}.json")
        if (row["physics_step"] != tick or row["sdk_hand_geometry"]["physics_step"] != tick
                or row["rod"]["physics_step"] != tick or row["rod"]["rod_tick"] != (tick - 2) * 12
                or row["guard_failure"] is not None):
            raise ValueError("retarget source requires fully measured hand and rod at the same step")
        rows[tick] = row
    contacts = rows[SDK]["rod"]["rod_mesh_contact_pairs"]["pairs"]
    names = {pair["mesh_name"] for pair in contacts}
    if not set(prior.RIGHT_STATIONS).issubset(names) or any(name.startswith("left.") for name in names):
        raise ValueError("retarget checkpoint requires live right Tripod and no left contact yet")
    return rows, supports, (CONTRACT, receipt, *paths)


def route_passes(rows):
    if len(rows) != measured.SAMPLES:
        return False
    for row in rows:
        margins = np.asarray([row["noncontact_rod_clearance_m"], row["support_clearance_m"],
                              row["table_clearance_m"]], dtype=float)
        gaps = np.asarray(row["pad_gaps_m"], dtype=float)
        stations = tuple(prior.STATIONS[f"left.{name}"] for name in measured.PADS)
        if (row["shape_count"] != 22 or not np.isfinite(margins).all()
                or margins[0] < SCREEN_RESERVE_M or margins[1] <= 0 or margins[2] < measured.CLEARANCE_M
                or gaps.shape != (3,) or not np.isfinite(gaps).all()
                or not np.all(gaps >= -prior.MAX_PAD_SDF_PENETRATION_M)
                or any(node not in allowed for node, allowed in zip(row["node_ids"], stations, strict=True))):
            return False
    last = rows[-1]
    return bool(max(last["pad_gaps_m"]) < .0001 and max(last["opposition"]) < -.25)


def screen(receipt):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task

    raw, supports, _ = completed_rows(receipt)
    request = read_request()
    rows = {tick: measured.decode_owned(row) for tick, row in raw.items()}
    snapshots = np.stack([row["rod"]["node_positions"][0] for row in rows.values()])
    boxes = [(np.asarray(box["center_m"]) - np.asarray(box["size_m"]) / 2,
              np.asarray(box["center_m"]) + np.asarray(box["size_m"]) / 2) for box in supports]
    task = load_flexible_grasp_task(measured.ROOT / "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json", measured.ROOT)
    table = task["scene"]
    top = table["table_center_m"][2] + .5 * table["table_size_m"][2]
    grids = scene.load_sdf_arrays(scene.prepare_scene(measured.ROOT))
    samples = measured.measure_side("left", rows[SDK], rows[SDK]["rod"]["node_positions"][0],
                                   measured.archived_tube(snapshots), boxes, grids, request["left_goal"],
                                   top, contact_pads=measured.PADS, max_delta_m=.004, max_delta_rad=.08)
    if not route_passes(samples):
        raise ValueError("Tripod retarget failed sampled rod margin, target stations, or opposition")
    return dict(version=request["version"], receipt_sha256=measured.native.sha(receipt),
                request_sha256=measured.native.sha(REQUEST), sdk_step=SDK,
                archived_row_sha256=measured.native.sha(receipt.parent/"native"/f"tripod_control_{SDK}.json"),
                samples=samples, grasp_qualified=False)


def read_request():
    request = json.loads(REQUEST.read_bytes())
    if request["version"] != "dh116-tripod-bilateral-retarget-v2":
        raise ValueError("expected the frozen Tripod retarget request")
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    measured.native.write_new(args.output, screen(args.receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
