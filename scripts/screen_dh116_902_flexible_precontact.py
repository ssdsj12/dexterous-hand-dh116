#!/usr/bin/env python3
"""Screen a guarded hands-clear approach from each sealed 902 flexible Reset."""

import argparse
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from scripts.screen_dh116_pinch_supported_approach import settled_control_state
from scripts import screen_dh116_tripod_measured_precontact as measured
from scripts import verify_dh116_902_flexible_reset as reset
from scripts.verify_dh116_tripod_supported_reset import waypoint_controls

RECEIPTS = reset.ROOT / "docs/contracts/dh116_teacher_902_flexible_distal_reset_v1.json"
GOALS = reset.ROOT / "configs/dh116_tasks/flexible_902_distal_precontact_v1.json"
SIDES = ("left", "right")
WINDOWS = 24
SDK_END = reset.SDK_END + 8 * WINDOWS


def material_side_matches(samples, side):
    return bool(samples) and all(row["node_ids"] and all(
        (node > 20 if side == "left" else node < 20) for node in row["node_ids"])
        for row in samples)


def load_settled(receipt, grasp, config):
    actual = json.loads(RECEIPTS.read_bytes())["actual"][grasp]
    outer = measured.sources.checked_json(receipt, actual["outer_sha256"])
    native_dir = receipt.parent / "native"
    marker = native_dir / "app_close_returned.json"
    recorded = outer["artifact_sha256"]
    close = measured.sources.checked_json(marker, recorded[marker.name])
    if (outer["exit_code"] != 0 or outer["debugger_inferior_exit_code"] != 0
            or outer["forced_exit"] or outer["exceeded_budget"] or close["app_close_returned"] is not True):
        raise ValueError("sealed same-mode settled rod and natural exit required")
    support_path = native_dir / f"{grasp}_support_scene.json"
    support = measured.sources.checked_json(support_path, recorded[support_path.name])
    if support["supports"] != config["support_scene"]["supports"] or not support["no_hidden_rod_pins"]:
        raise ValueError("both measured supports and free rod must match the Reset")
    row_paths = tuple(native_dir / f"{grasp}_exchange_{tick}.json" for tick in range(reset.SDK_END-8, reset.SDK_END+1))
    if recorded[row_paths[-1].name] != actual["last_row_sha256"]:
        raise ValueError("wrong terminal settled row")
    rows = []
    for tick, path in enumerate(row_paths, reset.SDK_END - 8):
        row = measured.sources.checked_json(path, recorded[path.name])
        rod = row["rod"]
        if rod["physics_step"] != tick or rod["rod_tick"] != 12 * (tick - 2):
            raise ValueError("native settled geometry and rod clocks differ")
        rows.append(row)
    return rows, (RECEIPTS, receipt, marker, support_path, *row_paths)


def screen(receipt, grasp):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene

    reference, config, _ = reset.load_candidate(grasp)
    raw, _ = load_settled(receipt, grasp, config)
    request = json.loads(GOALS.read_bytes())
    if request["version"] != "dh116-902-distal-precontact-v1":
        raise ValueError("unknown 902 precontact request")
    goals = request["goals"][grasp]
    if set(goals) != set(SIDES):
        raise ValueError("independent measured waypoint for each physical hand required")
    rows = [decode_owned(row) for row in raw]
    nodes = np.stack([row["rod"]["node_positions"][0] for row in rows])
    tube = measured.archived_tube(nodes)
    supports = config["support_scene"]["supports"]
    boxes = [(np.asarray(b["center_m"]) - np.asarray(b["size_m"]) / 2,
              np.asarray(b["center_m"]) + np.asarray(b["size_m"]) / 2) for b in supports]
    grids = scene.load_sdf_arrays(scene.prepare_scene(reset.ROOT))
    state = settled_control_state(rows[-1], config)
    pads = reset.PADS[:2] if grasp == "pinch" else reset.PADS
    table = reference["table"]
    table_top = table["center_m"][2] + table["size_m"][2] / 2
    measurements = {side: measured.measure_side(side, state, nodes[-1], tube, boxes, grids,
        goals[side], table_top, pads=pads, max_delta_m=.004, max_delta_rad=.06) for side in SIDES}
    waypoint_controls(config, state, goals, windows=WINDOWS)
    passed = all(measured.check_route(measurements[side], pad_count=len(pads))
                 and material_side_matches(measurements[side], side) for side in SIDES)
    return dict(version=request["version"], grasp=grasp, goal_sha256=measured.native.digest(goals),
                outer_sha256=reset.sha(receipt), before_sdk=reset.SDK_END,
                all_passed=passed, samples=measurements, native_grasp_verified=False)


def bounded_prefix(actions, windows):
    """Shorten the original command sequence without speeding up its target motion."""
    actions = np.asarray(actions, dtype=float)
    if (type(windows) is not int or not 1 <= windows <= WINDOWS
            or actions.shape != (WINDOWS, 24) or not np.isfinite(actions).all()):
        raise ValueError("a finite original 24-window approach and bounded integer prefix required")
    return actions[:windows].copy()


def load_approach(receipt, proof_path, grasp, config, *, windows=WINDOWS):
    raw, files = load_settled(receipt, grasp, config)
    proof = json.loads(proof_path.read_bytes())
    request = json.loads(GOALS.read_bytes())
    expected = request["screen_sha256"][grasp]
    count = 2 if grasp == "pinch" else 3
    if (expected != reset.sha(proof_path) or proof["all_passed"] is not True or proof["grasp"] != grasp
            or proof["goal_sha256"] != measured.native.digest(request["goals"][grasp])
            or proof["outer_sha256"] != reset.sha(receipt) or proof["before_sdk"] != reset.SDK_END
            or any(not measured.check_route(proof["samples"][side], pad_count=count)
                   or not material_side_matches(proof["samples"][side], side) for side in SIDES)):
        raise ValueError("passing sealed 22-envelope waypoint proof required")
    state = settled_control_state(decode_owned(raw[-1]), config)
    actions = waypoint_controls(config, state, request["goals"][grasp], windows=WINDOWS)
    return raw[-1], bounded_prefix(actions, windows), (*files, GOALS, proof_path, Path(__file__), Path(measured.__file__))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp", choices=("pinch", "tripod"), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    proof = screen(args.receipt, args.grasp)
    measured.native.write_new(args.output, proof)
    return int(not proof["all_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
