#!/usr/bin/env python3
"""Check a two-hand Tripod reference trajectory on its original material polyline."""
import argparse
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_teacher_seed_initialization import seed_arrays
from scripts.audit_dh116_pinch_initial_pose import (
    DISTAL_PAD_LINKS, checked_json, load_geometry, path_displacement_bound,
)
from scripts.audit_dh116_tripod_fixed_root_path import certify_path
from scripts.audit_dh116_tripod_sdk_path import compile_actions, runtime_reference
from scripts.audit_dh116_wire_contact_witness import audit_hand, endpoint_gap, shapes_world
from scripts.prepare_dh116_wire_pose_candidate import HANDS, _pad_gap_interval


def compile_bilateral_actions(endpoints, *, physics_steps):
    if set(endpoints) != set(HANDS):
        raise ValueError("both independently sourced hand paths required")
    plans = {side: compile_actions(*endpoints[side], side=side, physics_steps=physics_steps,
                                   mode="intrinsic_linear") for side in HANDS}
    return sum(np.asarray(plan["actions"]) for plan in plans.values()), plans


def audit_reference(artifact, project):
    seed = artifact["trial_seed"]
    nodes, joints, roots, _ = seed_arrays(seed)
    ids = [JOINTS.index(key) for key in ACTIVE]
    delta = np.array([0., .01, .01, .01, 0., 0.])
    geometry = {side: load_geometry(project, side=side) for side in HANDS}
    endpoints = {}
    for index, side in enumerate(HANDS):
        if geometry[side]["source_sha256"] != artifact["hands"][side]["source_sha256"]:
            raise ValueError("original per-hand geometry sources differ")
        end = roots[index].copy()
        if side == "left":
            end[[0, 2]] += [-.00002, .00001]
        start = end.copy()
        start[2] += .0007
        q = joints[index, ids]
        endpoints[side] = (start, end, q-delta, q)

    actions, plans = compile_bilateral_actions(endpoints, physics_steps=160)
    hands = {}
    for side in HANDS:
        g = geometry[side]
        p0, p1, q0, q1 = endpoints[side]
        layout = g["reference"]["reference_layout"]
        top = layout["table_center"][2] + layout["table_size"][2] / 2
        radius = g["old_candidate"]["rod"]["radius_m"]
        projection = runtime_reference(plans[side], g)["rows"]
        positions = np.array([row["pose_wxyz"] for row in projection])
        angles = np.array([row["q_rad"] for row in projection])
        count = len(projection) - 1
        translation = count * np.linalg.norm(np.diff(positions[:, :3], axis=0), axis=1).max()
        increments = count * np.abs(np.diff(angles, axis=0)).max(0)
        # The FK travel routine expands linear mimic deltas. Use the signed
        # monotone nominal q change and scale coefficients by its max rate
        # ratio to cover binary32 rounding of each successive knot.
        nominal = q1 - q0
        if np.any(increments[nominal == 0.] > 0.):
            raise ValueError("original target replay moved an unrequested finger")
        ratio = float(max(1., (increments[nominal != 0.] / np.abs(nominal[nominal != 0.])).max()))
        travel = {kind+":"+name: translation + value*ratio
                  for kind in ("source", "cooked")
                  for name, value in path_displacement_bound(g, q0, nominal, cooked=kind=="cooked").items()}
        if not np.allclose(positions[:, 3:], p1[3:], atol=1e-12, rtol=0.):
            raise ValueError("fixed-orientation Tripod replay required")
        sign = 1. if side == "left" else -1.

        def query(t):
            index = min(int(t*count), count-1)
            fraction = t*count-index
            pose = positions[index].copy()
            pose[:3] = (1.-fraction)*pose[:3] + fraction*positions[index+1, :3]
            active = (1.-fraction)*angles[index] + fraction*angles[index+1]
            rows = {}
            for key, hull in shapes_world(g, pose, active).items():
                rows["table:"+key] = float(hull[:, 2].min()-top)
                rows["side:"+key] = float((sign*hull[:, 1]).min())
                gap = _pad_gap_interval(hull, nodes, radius)["lower_m"]
                if gap < .00011:
                    gap = endpoint_gap(hull, nodes, radius)["lower_m"]
                rows["rod:"+key] = gap
            return rows

        bounds = {prefix+key: value for prefix in ("table:", "side:", "rod:")
                  for key, value in travel.items()}
        thresholds = {key: CLEARANCE_M if key.startswith("table:") else 0. for key in bounds}
        proof = certify_path(query, bounds, thresholds)
        final = audit_hand(shapes_world(g, positions[-1], angles[-1]), nodes, radius,
                           layout, DISTAL_PAD_LINKS[:3])
        changes = [row for row in projection[1:] if row["projection_reasons"]]
        side_margin = min((value for interval in proof["certified_intervals"]
                           for key, value in interval["margin_lower_m"].items() if key.startswith("side:")),
                          default=None)
        hands[side] = dict(proof=proof, terminal=final, projection_changes=len(changes),
                           side_plane_clearance_m=side_margin,
                           executed_target_end_pose_wxyz=positions[-1].tolist(),
                           executed_target_end_q_rad=angles[-1].tolist())
    passed = all(item["proof"]["proved"] and item["projection_changes"] == 0
                 and item["side_plane_clearance_m"] > 0.
                 and item["terminal"]["geometric_approach_passed"]
                 and item["terminal"]["direction_gate_passed"] for item in hands.values())
    return dict(hands=hands, action24_reference=actions.tolist(), planned_sdk_steps=160,
                continuous_reference_path_proved=passed, actual_isaac_steps=0,
                self_collision_verified=False, native_contact_verified=False, teacher_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--endpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = checked_json(args.endpoint, args.endpoint_sha256)
    result = audit_reference(artifact, Path(__file__).resolve().parents[1])
    result["input_sha256"] = args.endpoint_sha256
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["continuous_reference_path_proved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
