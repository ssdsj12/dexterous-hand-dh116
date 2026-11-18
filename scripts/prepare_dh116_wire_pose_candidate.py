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
    args = parser.parse_args(argv)

    reference = args.reference.read_bytes()
    target = args.target.read_bytes()
    stations = dict(left=args.left_node, right=args.right_node)
    seed = retarget_seed(json.loads(reference), json.loads(target), stations)
    report = dict(schema="dh116-wire-pose-candidate-v1", grasp=args.grasp,
                  material_node_ids_by_hand=stations,
                  input_sha256=dict(reference=sha256(reference).hexdigest(), target=sha256(target).hexdigest()),
                  seed=seed, collision_qualified=False, native_contact_verified=False, teacher_admission=False)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
