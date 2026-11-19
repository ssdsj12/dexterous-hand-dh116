#!/usr/bin/env python3
"""Check static convex witnesses on one saved two-hand wire candidate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_polyline_closest_support import refine
from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import NUMERIC_GUARD
from go2_pvcnn.control.dh116.tabletop_teacher_seed_initialization import seed_arrays
from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, box_relation, load_geometry, world_vertices
from scripts.prepare_dh116_wire_pose_candidate import HANDS, _material_tangents, _pad_gap_interval
from scripts.solve_dh116_tripod_joint_root_direction import COARSE

PAD_COUNTS = {"tripod": 3, "power": 5}


def _unambiguous_direction(segments, best, feasible_gap, radius):
    direction = best["new_normal"]
    if direction is None:
        return None
    unit = np.asarray(direction) / np.linalg.norm(direction)
    for row in segments:
        if row["lower_m"] > feasible_gap:
            continue
        proposal = row["proposal"]
        if proposal is None or proposal["new_normal"] is None:
            return None
        other = np.asarray(proposal["new_normal"])
        other /= np.linalg.norm(other)
        if (proposal["feasible_distance_upper_m"] - radius - row["lower_m"] > 4 * NUMERIC_GUARD
                or float(other @ unit) < .99):
            return None
    return unit.tolist()


def endpoint_gap(hull, nodes, radius):
    initial = _pad_gap_interval(hull, nodes, radius)
    proof = refine(hull, nodes, max_segments=4)  # Validates complete geometry and every proposed point.
    proposals = [(row["segment"], row["proposal"]) for row in proof["segments"]
                 if row["proposal"] is not None and "feasible_distance_upper_m" in row["proposal"]]
    lower = max(initial["lower_m"], proof["lower_m"])
    upper = initial["upper_m"]
    witness = None
    if proposals:
        index, best = min(proposals, key=lambda item: item[1]["feasible_distance_upper_m"])
        feasible = best["feasible_distance_upper_m"] - radius
        upper = min(upper, feasible)
        # A residual inside the rod capsule or an unresolved distance cannot establish opposition.
        tight_clear = lower >= 0. and 0. <= feasible - lower <= 4 * NUMERIC_GUARD
        witness = dict(segment=index, fraction=best["segment_fraction"], hull_point_m=best["point_world_m"],
            rod_axis_point_m=best["segment_point_world_m"], feasible_upper_m=feasible,
            direction=_unambiguous_direction(proof["segments"], best, feasible, radius) if tight_clear else None)
    if lower > upper:
        raise ValueError("closest-point interval contradicts the conservative bounds")
    status = "clear" if lower >= 0. else "intersection_proven" if upper < 0. else "unresolved"
    return dict(lower_m=lower, upper_m=upper, vertex_upper_m=initial["upper_m"],
        original_lower_m=initial["lower_m"], status=status, witness=witness,
        actual_contact_measurement=False)


def audit_hand(shapes, nodes, radius, layout, pads, *, include_palm=False):
    table = {key: box_relation(vertices, layout["table_center"], layout["table_size"])
             for key, vertices in shapes.items()}
    required = ["cooked:" + pad for pad in pads]
    if include_palm:
        required.append("source:base_link")
    gaps = {}
    for key, vertices in shapes.items():
        interval = _pad_gap_interval(vertices, nodes, radius)
        gaps[key] = endpoint_gap(vertices, nodes, radius) if key in required or interval["lower_m"] < 0. else dict(
            interval, status="clear", vertex_upper_m=interval["upper_m"],
            original_lower_m=interval["lower_m"], witness=None, actual_contact_measurement=False)
    clear = all(row["status"] == "clear" for row in gaps.values())
    table_passed = all(not row["intersects"] and row["top_clearance_m"] >= CLEARANCE_M for row in table.values())
    near = all(gaps[key]["upper_m"] <= COARSE for key in required)

    def direction(key):
        witness = gaps[key]["witness"]
        return None if witness is None else witness["direction"]

    thumb = direction(required[0])
    dots = {}
    for key in required[1:len(pads)]:
        other = direction(key)
        dots[key] = None if thumb is None or other is None else float(np.dot(thumb, other))
    return dict(pad_source_ids=required, gaps=gaps, witness_direction_dots=dots,
        witness_direction_scope="rod_axis_to_convex_point_static_proxy",
        direction_gate_passed=all(value is not None and value < -.2 for value in dots.values()),
        minimum_table_clearance_m=min(row["top_clearance_m"] for row in table.values()),
        table_clearance_passed=table_passed, complete_convex_clearance_passed=clear, required_approach_passed=near,
        geometric_approach_passed=table_passed and clear and near)


def shapes_world(geometry, pose, q):
    return {kind + ":" + link: vertices for kind, locked in (("source", False), ("cooked", True))
            for link, vertices in world_vertices(geometry, pose, q, cooked=locked).items()}


def audit_seed(seed, *, style, project):
    nodes, full, roots, _ = seed_arrays(seed)
    _material_tangents(nodes)
    active_ids = [JOINTS.index(name) for name in ACTIVE]
    pads = DISTAL_PAD_LINKS[:PAD_COUNTS[style]]
    hands = {}
    extents = []
    for index, side in enumerate(HANDS):
        geometry = load_geometry(project, side=side)
        shapes = shapes_world(geometry, roots[index], full[index, active_ids])
        row = audit_hand(shapes, nodes, geometry["old_candidate"]["rod"]["radius_m"],
                         geometry["reference"]["reference_layout"], pads, include_palm=style == "power")
        row["source_sha256"] = geometry["source_sha256"]
        hands[side] = row
        ys = np.concatenate([vertices[:, 1] for vertices in shapes.values()])
        extents.append((float(ys.min()), float(ys.max())))
    separation = max(extents[0][0] - extents[1][1], extents[1][0] - extents[0][1])
    approach_passed = separation > 0. and all(row["geometric_approach_passed"] for row in hands.values())
    tripod_passed = style == "tripod" and approach_passed and all(row["direction_gate_passed"] for row in hands.values())
    return dict(schema="dh116-wire-endpoint-witness-v1", grasp=style, hands=hands, bilateral_y_clearance_m=separation,
        static_approach_passed=approach_passed, static_tripod_endpoint_passed=tripod_passed,
        actual_contact_measurement=False, continuous_path_verified=False, self_collision_verified=False,
        palm_contact_verified=False, native_contact_verified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--grasp", required=True, choices=PAD_COUNTS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    raw = args.seed.read_bytes()
    result = audit_seed(json.loads(raw), style=args.grasp, project=args.project)
    result["input_seed_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    passed = result["static_tripod_endpoint_passed"] if args.grasp == "tripod" else result["static_approach_passed"]
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
