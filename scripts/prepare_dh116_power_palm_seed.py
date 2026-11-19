#!/usr/bin/env python3
"""Construct an independent palm-referenced Power starting pose on CPU."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_teacher_seed_initialization import seed_arrays
from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, box_relation, load_geometry
from scripts.audit_dh116_wire_contact_witness import audit_seed, shapes_world
from scripts.prepare_dh116_wire_pose_candidate import HANDS, _material_tangents, _pad_gap_interval, _station_frame
from scripts.solve_dh116_tripod_joint_root_direction import COARSE

ROLL_DEGREES = (0, -45, 45, -90, 90, -135, 135, 180)


def seat_palm(hull, finger_centroid, station, tangent, axis_offset):
    """Translate a convex palm support point to a chosen material cross-section."""
    palm = np.asarray(hull, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    fingers = np.asarray(finger_centroid, dtype=np.float64)
    station = np.asarray(station, dtype=np.float64)
    if (palm.ndim != 2 or palm.shape[1] != 3 or len(palm) < 4
            or tangent.shape != (3,) or fingers.shape != (3,) or station.shape != (3,)
            or not np.isfinite(palm).all() or not np.isfinite(station).all()
            or not np.isfinite(axis_offset) or axis_offset <= 0.
            or not np.isclose(np.linalg.norm(tangent), 1., atol=1e-10, rtol=0.)):
        raise ValueError("original finite palm hull and unit material tangent required")
    delta = fingers - palm.mean(0)
    transverse = delta - tangent * np.dot(delta, tangent)
    length = np.linalg.norm(transverse)
    if not np.isfinite(length) or length <= 1e-8:
        raise ValueError("palm has no transverse direction toward the finger group")
    direction = transverse / length
    support = palm @ direction
    face = palm[support >= support.max() - 1e-10]
    point = face.mean(0)
    return station - point - axis_offset * direction, direction


def support_collisions(shapes, supports):
    return [dict(source=key, support_index=index)
            for key, vertices in shapes.items()
            for index, support in enumerate(supports)
            if box_relation(vertices, support["center_m"], support["size_m"])["intersects"]]


def _search_palm(geometry, pose, q, nodes, station, tangent, supports):
    if geometry["fk"].root.rsplit("/", 1)[-1] != "base_link":
        raise ValueError("Power palm support requires the original base_link")
    layout = geometry["reference"]["reference_layout"]
    table_top = layout["table_center"][2] + layout["table_size"][2] / 2
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    reference = Rotation.from_quat(np.roll(pose[3:], -1))
    pads = ["cooked:" + k for k in DISTAL_PAD_LINKS]
    trials = []
    for degrees in ROLL_DEGREES:
        root = pose.copy()
        rotation = Rotation.from_rotvec(np.deg2rad(degrees) * tangent) * reference
        root[3:] = np.roll(rotation.as_quat(), 1)
        shapes = shapes_world(geometry, root, q)
        palm = shapes["source:base_link"]
        fingers = np.array([shapes[key].mean(0) for key in pads])
        shift, direction = seat_palm(palm, fingers.mean(0), station, tangent, radius + COARSE / 2)
        lowest = min(vertices[:, 2].min() for vertices in shapes.values())
        rise = max(0., table_top + CLEARANCE_M + 1e-8 - (lowest + shift[2]))
        shift[2] += rise
        root[:3] += shift
        placed = {key: vertices + shift for key, vertices in shapes.items()}
        gaps = {key: _pad_gap_interval(vertices, nodes, radius) for key, vertices in placed.items()}
        support_hits = support_collisions(placed, supports)
        palm_gap = gaps["source:base_link"]
        sides = (fingers + shift - station) @ direction
        negatives = [max(0., -row["lower_m"]) for row in gaps.values()]
        approach_excess = [max(0., gaps[key]["upper_m"] - COARSE) for key in ("source:base_link", *pads)]
        rank = (len(support_hits) + sum(value > 0. for value in negatives), sum(negatives),
                max(approach_excess), sum(approach_excess), -float(sides.min()))
        trials.append(dict(root=root.tolist(), roll_degrees=degrees, table_rise_m=float(rise),
            translation_m=shift.tolist(), palm_toward_fingers=direction.tolist(),
            finger_side_projection_m=dict(zip(DISTAL_PAD_LINKS, sides.tolist())),
            initial_palm_clearance_m=palm_gap, support_intersections=support_hits, rank=rank))
    best = min(trials, key=lambda row: row["rank"])
    return best, len(trials)


def prepare_palm_seed(seed, *, stations, project, supports=()):
    nodes, joints, roots, _ = seed_arrays(seed)
    tangent = _material_tangents(nodes)
    if (not isinstance(stations, dict) or set(stations) != set(HANDS)
            or any(type(i) is not int or not 0 < i < len(nodes) - 1 for i in stations.values())
            or stations["left"] == stations["right"]):
        raise ValueError("two distinct interior material stations required")
    active_ids = [JOINTS.index(key) for key in ACTIVE]
    trial = deepcopy(seed)
    search = {}
    for idx, side in enumerate(HANDS):
        node = stations[side]
        frame = _station_frame(tangent, node)
        selected, count = _search_palm(load_geometry(project, side=side), roots[idx],
            joints[idx, active_ids], nodes, nodes[node], frame[:, 0], supports)
        trial["roots"][idx] = selected["root"]
        search[side] = dict(selected, node_id=node, candidates_evaluated=count)
    seed_arrays(trial)
    evidence = audit_seed(trial, style="power", project=project)
    opposed_sides = all(all(item > 0. for item in row["finger_side_projection_m"].values())
                        for row in search.values())
    supports_clear = all(not row["support_intersections"] for row in search.values())
    return dict(schema="dh116-power-palm-referenced-seed-v1", trial_seed=trial, search=search, evidence=evidence,
        support_boxes_audited=len(supports), discrete_support_clearance_passed=supports_clear,
        geometric_wrap_passed=opposed_sides and supports_clear and evidence["static_approach_passed"],
        locked_palm_cooked_available=False, palm_patch_identity_proven=False,
        native_contact_verified=False, palm_contact_verified=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed", type=Path)
    source.add_argument("--end-supported", action="store_true",
                        help="use the existing independent Power seed and support boxes")
    parser.add_argument("--left-node", required=True, type=int)
    parser.add_argument("--right-node", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    supports = ()
    if args.end_supported:
        from go2_pvcnn.control.dh116.tabletop_teacher_supported_runtime import supported_seed_plan

        plan, _ = supported_seed_plan(args.project)
        supported = plan["seeds_by_style"]["power"]
        seed = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in supported.items()}
        raw = json.dumps(seed, sort_keys=True, allow_nan=False).encode()
        supports = plan["scene"]["supports"]
    else:
        raw = args.seed.read_bytes()
    result = prepare_palm_seed(json.loads(raw), stations=dict(left=args.left_node, right=args.right_node),
                               project=args.project, supports=supports)
    result["input_seed_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["geometric_wrap_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
