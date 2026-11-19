#!/usr/bin/env python3
"""Place a DH116 pregrasp relative to two fixed material nodes of a target rod."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.tabletop_teacher_seed_initialization import seed_arrays


HANDS = ("left", "right")
GRASPS = ("pinch", "tripod", "power")
NODE_COUNT = 41
MIN_SEGMENT_M = 1e-9
MIN_FRAME_DIRECTION_NORM = 1e-3


def _material_tangents(nodes):
    segments = np.diff(nodes, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths < MIN_SEGMENT_M):
        raise ValueError("wire centerline must have finite nonzero segments")
    return segments / lengths[:, None]


def _station_frame(tangents, node_id):
    # Increasing material IDs orient the tangent independently of row storage.
    direction = tangents[node_id - 1] + tangents[node_id]
    length = np.linalg.norm(direction)
    if length < MIN_FRAME_DIRECTION_NORM:
        raise ValueError(f"adjacent segments folded back at material node {node_id}")
    direction /= length
    lateral = np.cross([0., 0., 1.], direction)
    length = np.linalg.norm(lateral)
    if length < MIN_FRAME_DIRECTION_NORM:
        raise ValueError(f"gravity cannot define a transverse frame at material node {node_id}")
    lateral /= length
    return np.column_stack((direction, lateral, np.cross(direction, lateral)))


def retarget_seed(reference, target, stations):
    """Return an initializer-compatible candidate without changing the rod or fingers.

    Reference rows are in DH116's canonical material order. The target carries
    its actual material IDs. Each hand keeps its reference rigid transform
    relative to its own gravity-aligned local centerline frame.
    """
    nodes, joints, roots, _ = seed_arrays(reference)
    if (not isinstance(stations, dict) or set(stations) != set(HANDS)
            or any(type(i) is not int or not 0 < i < NODE_COUNT - 1 for i in stations.values())
            or stations["left"] == stations["right"]):
        raise ValueError("left/right stations must name distinct interior material nodes")
    if not isinstance(target, dict):
        raise ValueError("target wire packet required")
    target_ids = target.get("material_node_ids")
    if (not isinstance(target_ids, list) or len(target_ids) != NODE_COUNT
            or any(type(i) is not int for i in target_ids)
            or set(target_ids) != set(range(NODE_COUNT))):
        raise ValueError("target material node IDs must be the exact set 0 through 40")
    target_nodes = np.asarray(target.get("nodes_w_m"), dtype=float)
    if target_nodes.shape != (NODE_COUNT, 3) or not np.isfinite(target_nodes).all():
        raise ValueError("target must have 41 finite world positions in metres")
    target_nodes = target_nodes[np.argsort(target_ids)]

    source_tangents = _material_tangents(nodes)
    target_tangents = _material_tangents(target_nodes)
    placed = []
    for hand, pose in zip(HANDS, roots):
        node_id = stations[hand]
        source_frame = _station_frame(source_tangents, node_id)
        target_frame = _station_frame(target_tangents, node_id)
        rotation = Rotation.from_matrix(target_frame @ source_frame.T)
        root = target_nodes[node_id] + rotation.apply(pose[:3] - nodes[node_id])
        quaternion = (rotation * Rotation.from_quat(np.roll(pose[3:], -1))).as_quat()
        placed.append(np.r_[root, np.roll(quaternion, 1)].tolist())
    result = dict(nodes=target_nodes.tolist(), joints=joints.tolist(), roots=placed)
    seed_arrays(result)
    return result


def audit_table_clearance(seed, *, project):
    """Screen actual candidate FK against the fixed scene table using CPU geometry."""
    from scripts.audit_dh116_pinch_initial_pose import box_relation, load_geometry, world_vertices
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS
    from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import CLEARANCE_M

    _, joints, roots, _ = seed_arrays(seed)
    active = [JOINTS.index(name) for name in ACTIVE]
    hands = {}
    for side, root, q in zip(HANDS, roots, joints):
        geometry = load_geometry(project, side=side)
        layout = geometry["reference"]["reference_layout"]

        def inspect(*, cooked):
            shapes = world_vertices(geometry, root, q[active], cooked=cooked)
            return {name: box_relation(hull, layout["table_center"], layout["table_size"])
                    for name, hull in shapes.items()}

        source = inspect(cooked=False)
        pads = inspect(cooked=True)
        source_hits = sorted(name for name, row in source.items() if row["intersects"])
        cooked_hits = sorted(name for name, row in pads.items() if row["intersects"])
        minimum = min(row["top_clearance_m"] for row in (*source.values(), *pads.values()))
        hands[side] = dict(source=source, locked_cooked_pads=pads, source_intersections=source_hits,
            cooked_pad_intersections=cooked_hits, minimum_table_top_clearance_m=minimum,
            passed=not source_hits and not cooked_hits and minimum >= CLEARANCE_M,
            source_sha256=geometry["source_sha256"])
    return dict(scope="static_world_table_reference", minimum_required_clearance_m=CLEARANCE_M,
        hands=hands, passed=all(hand["passed"] for hand in hands.values()),
        live_cooked_geometry_verified=False, grasp_qualified=False)



def _pad_gap_interval(hull, nodes, radius):
    from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import (
        NUMERIC_GUARD, RADIUS, certify,
    )

    if radius != RADIUS:
        raise ValueError("wire radius differs from the existing support certificate")
    # The support planes bound the distance below. The test of real hull
    # vertices against continuous segments supplies an achievable upper bound.
    lower = certify(hull, nodes)["segment_lower_m"]
    edge = np.diff(nodes, axis=0)
    offset = hull[:, None, :] - nodes[None, :-1, :]
    t = np.clip(np.einsum("vsi,si->vs", offset, edge) / np.einsum("si,si->s", edge, edge), 0., 1.)
    upper = float(np.linalg.norm(offset - t[:, :, None] * edge, axis=-1).min() - radius + NUMERIC_GUARD)
    return dict(lower_m=lower, upper_m=upper)


def propose_vertical_repair(seed, *, grasp, project, initial_table):
    """Try only a world-Z root shift while retaining the original seed."""
    from copy import deepcopy

    from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, load_geometry, world_vertices
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS

    nodes, joints, roots, _ = seed_arrays(seed)
    _material_tangents(nodes)
    active = [JOINTS.index(name) for name in ACTIVE]
    required = DISTAL_PAD_LINKS[:{"pinch": 2, "tripod": 3, "power": 5}[grasp]]
    trial = deepcopy(seed)
    hands = {}
    for i, side in enumerate(HANDS):
        margin = initial_table["minimum_required_clearance_m"] - initial_table["hands"][side]["minimum_table_top_clearance_m"]
        # A 10 nm computational cushion covers rounding at the existing floor.
        rise = float(margin + 1e-8) if margin > 0. else 0.
        trial["roots"][i][2] += rise
        pad_checks = {}
        if rise:
            geometry = load_geometry(project, side=side)
            cooked = world_vertices(geometry, roots[i], joints[i, active], cooked=True)
            radius = geometry["old_candidate"]["rod"]["radius_m"]
            for pad in required:
                hull = cooked[pad]
                before = _pad_gap_interval(hull, nodes, radius)
                after = _pad_gap_interval(hull + [0., 0., rise], nodes, radius)
                if after["lower_m"] > before["upper_m"]:
                    comparison = "proven_farther"
                elif after["upper_m"] <= before["lower_m"]:
                    comparison = "nonworsening"
                else:
                    comparison = "unresolved"
                pad_checks[pad] = dict(before_m=before, after_m=after, comparison=comparison)
        hands[side] = dict(rise_m=rise, required_pad_gaps=pad_checks, preserve_pad_reach_proven=all(
            check["comparison"] == "nonworsening" and check["after_m"]["lower_m"] >= 0.
            for check in pad_checks.values()))
    changed = any(row["rise_m"] > 0. for row in hands.values())
    table_after = audit_table_clearance(trial, project=project) if changed else initial_table
    passed = table_after["passed"] and all(row["preserve_pad_reach_proven"] for row in hands.values())
    status = "static_repair_proven" if passed else "blocked"
    if not changed and passed:
        status = "unchanged"
    return dict(status=status, scope="static_table_and_required_pad_geometry",
                trial_seed=trial, table_after=table_after, hands=hands,
                static_repair_proven=bool(changed and passed),
                actual_contact_measurement=False, teacher_admission=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="DH116 candidate seed in canonical material, joint and hand order")
    parser.add_argument("--target", type=Path, required=True,
                        help="target material_node_ids and nodes_w_m JSON")
    parser.add_argument("--output", type=Path, required=True, help="new CPU candidate JSON file")
    parser.add_argument("--grasp", choices=GRASPS, required=True)
    parser.add_argument("--left-node", type=int, required=True)
    parser.add_argument("--right-node", type=int, required=True)
    parser.add_argument("--audit-table-clearance", action="store_true",
                        help="screen both candidate hands against the fixed table before using the seed")
    parser.add_argument("--propose-table-repair", action="store_true",
                        help="test a separate vertical root correction and preserve its geometry evidence")
    parser.add_argument("--joint-pregrasp-search", action="store_true",
                        help="search a separate bounded Tripod or Power wrist/finger candidate")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    if args.joint_pregrasp_search and args.grasp == "pinch":
        parser.error("--joint-pregrasp-search requires tripod or power")
    if args.joint_pregrasp_search and args.propose_table_repair:
        parser.error("select one trial algorithm")

    reference = args.reference.read_bytes()
    target = args.target.read_bytes()
    stations = dict(left=args.left_node, right=args.right_node)
    seed = retarget_seed(json.loads(reference), json.loads(target), stations)
    report = dict(schema="dh116-wire-pose-candidate-v1", grasp=args.grasp,
                  material_node_ids_by_hand=stations,
                  input_sha256=dict(reference=sha256(reference).hexdigest(), target=sha256(target).hexdigest()),
                  seed=seed, collision_qualified=False, native_contact_verified=False, teacher_admission=False)
    if args.audit_table_clearance or args.propose_table_repair:
        report["static_table_clearance"] = audit_table_clearance(seed, project=args.project)
    if args.propose_table_repair:
        report["vertical_repair"] = propose_vertical_repair(
            seed, grasp=args.grasp, project=args.project, initial_table=report["static_table_clearance"])
    if args.joint_pregrasp_search:
        from scripts.solve_dh116_wire_pregrasp import search_pregrasp

        report["joint_pregrasp_search"] = search_pregrasp(seed, style=args.grasp, project=args.project)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
    if args.joint_pregrasp_search:
        return 0 if report["joint_pregrasp_search"]["geometry_candidate_passed"] else 2
    if args.propose_table_repair:
        return 2 if report["vertical_repair"]["status"] == "blocked" else 0
    return 0 if report.get("static_table_clearance", {}).get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
