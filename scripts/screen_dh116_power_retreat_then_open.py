#!/usr/bin/env python3
"""Screen a retreat before bilateral Power opening against the settled physical rod."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts import screen_dh116_tripod_measured_precontact as measured

CONTRACT = measured.ROOT / "docs/contracts/dh116_teacher_power_proximal_binding_native_v1.json"
REQUEST = measured.ROOT / "configs/dh116_tasks/flexible_power_retreat_then_open_v1.json"
SDK = 240
SAMPLES = 65
WINDOWS = (32, 48)


def load_archive(receipt):
    from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference

    contract = json.loads(CONTRACT.read_bytes())
    expected = contract["compared_with_15_mesh_baseline"]["native_outer_sha"]
    raw = receipt.parent / "native"
    evidence = measured.sources.checked_json(receipt, expected)
    refs = [load_parked_reference(receipt, expected, tick) for tick in range(SDK - 8, SDK + 1)]
    row_paths = [raw / f"power_exchange_{tick}.json" for tick in range(SDK - 8, SDK + 1)]
    row = measured.sources.checked_json(row_paths[-1], evidence["artifact_sha256"][row_paths[-1].name])
    return refs, row, [CONTRACT, receipt, *row_paths]


def certify_intervals(values, motion):
    values = np.asarray(values, dtype=float)
    motion = np.asarray(motion, dtype=float)
    if (values.ndim != 3 or values.shape[:2] != (SAMPLES, 3)
            or motion.shape != (values.shape[2],) or not np.isfinite(values).all()
            or not np.isfinite(motion).all() or np.any(motion < 0.)):
        raise ValueError("complete rod/support/table samples and per-shape travel bound required")
    return np.min(values - motion[None, None] / (2 * (SAMPLES - 1)), axis=(0, 2))


def screen(receipt):
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, active_limits
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import load_flexible_grasp_task
    from scripts.audit_dh116_pinch_initial_pose import load_geometry, path_displacement_bound
    from scripts.certify_dh116_tripod_distal_route import support_clearance_lower
    from scripts.prepare_dh116_pinch_closure_margin import fk_poses

    request = json.loads(REQUEST.read_bytes())
    if (request["version"] != "dh116-power-retreat-then-open-v1"
            or set(request["hands"]) != set(measured.SIDES)):
        raise ValueError("independent bilateral Power retreat and open request required")
    refs, row, files = load_archive(receipt)
    row = measured.decode_owned(row)
    tube = measured.archived_tube(np.stack([ref.node_positions for ref in refs]))
    boxes = [(np.asarray(b["center_m"]) - np.asarray(b["size_m"]) / 2,
              np.asarray(b["center_m"]) + np.asarray(b["size_m"]) / 2) for b in refs[-1].support_boxes]
    cfg = load_flexible_grasp_task(measured.ROOT / "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json", measured.ROOT)
    table = cfg["scene"]
    top = table["table_center_m"][2] + .5 * table["table_size_m"][2]
    result = {}
    for side in measured.SIDES:
        hand = row["native_joint_state"][side]
        ids = [hand["names"].index(key) for key in ACTIVE]
        q = hand["position"][0, ids]
        native_poses = {k.split(".", 1)[1]: measured.as_matrix(*v)
                        for k, v in row["hand_link_poses"].items() if k.startswith(side + ".")}
        p = native_poses["base_link"]
        initial = np.r_[p[:3, 3], Rotation.from_matrix(p[:3, :3]).as_quat()[[3, 0, 1, 2]]]
        model = load_geometry(measured.ROOT, side=side)
        modeled = fk_poses(model, initial, q)
        corrections = {name: np.linalg.solve(modeled[name], pose) for name, pose in native_poses.items()}
        local_model = dict(model)
        for group in ("source_colliders", "cooked_pads"):
            marked = {}
            for name, item in model[group].items():
                correction = corrections.get(name, np.eye(4))
                local = item["vertices"] @ correction[:3, :3].T + correction[:3, 3]
                marked[name] = {**item, "vertices": local}
            local_model[group] = marked
        shapes = [(group, name, item["vertices"]) for group in ("source_colliders", "cooked_pads")
                  for name, item in local_model[group].items()]
        specification = request["hands"][side]
        retreat = initial.copy()
        retreat[0] += specification["retreat_world_x_m"]
        open_q = q.copy()
        open_q[1:] = np.maximum(active_limits()[0][1:] + 1e-5, q[1:] - specification["opening_rad"])
        stage_results = []
        for p0, p1, q0, q1 in ((initial, retreat, q, q), (retreat, retreat, q, open_q)):
            delta = q1 - q0
            displacement = float(np.linalg.norm(p1[:3] - p0[:3]))
            bound = {group: path_displacement_bound(local_model, q0, delta, cooked=group == "cooked_pads")
                     for group in ("source_colliders", "cooked_pads")}
            motion = [displacement + bound[group][name] for group, name, _ in shapes]
            samples = []
            for t in np.linspace(0., 1., SAMPLES):
                pose = (1. - t) * p0 + t * p1
                pose[3:] = p0[3:]
                poses = fk_poses(local_model, pose, q0 + t * delta)
                world = [verts @ poses[name][:3, :3].T + poses[name][:3, 3] for _, name, verts in shapes]
                samples.append([list(measured.separation(v, *tube) for v in world),
                    [min(support_clearance_lower(v, a, b) for a, b in boxes) for v in world],
                    [float(v[:, 2].min() - top) for v in world]])
            certified = certify_intervals(samples, motion)
            stage_results.append(dict(certified_rod_m=float(certified[0]), certified_support_m=float(certified[1]),
                certified_table_m=float(certified[2]), maximum_adjacent_shape_travel_m=max(motion)/(SAMPLES-1)))
        result[side] = dict(shape_count=len(shapes), retreat_pose_wxyz=retreat.tolist(),
                            start_active_rad=q.tolist(), opened_active_rad=open_q.tolist(), stages=stage_results)
    passed = all(m["certified_rod_m"] > measured.ROD_RESERVE_M and m["certified_support_m"] > 0
                 and m["certified_table_m"] >= measured.CLEARANCE_M
                 for v in result.values() for m in v["stages"])
    return dict(version=request["version"], outer_sha256=measured.native.sha(receipt),
                last_row_sha256=measured.native.sha(files[-1]), request_sha256=measured.native.sha(REQUEST),
                before_sdk=SDK, number_of_path_samples=SAMPLES, hands=result,
                all_passed=passed, grasp_qualified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = screen(args.receipt)
    measured.native.write_new(args.output, result)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
