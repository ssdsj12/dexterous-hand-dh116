"""Reference layouts for the arm-free DH116 tabletop scene."""
from __future__ import annotations

from typing import Any

import numpy as np

from .linear_coupling import ACTIVE


_CANDIDATE3_ROOTS = {
    "left": np.asarray([0.38154750224737377, 0.2373296462947355, 0.3612084291104437,
                        -0.00524514, 0.99977878, -0.00643838, -0.01932409], dtype=np.float32),
    "right": np.asarray([0.379352202466893, -0.237214928805469, 0.36181091453662617,
                         0.00606007, 0.99966999, 0.00472094, -0.02451318], dtype=np.float32),
}


def candidate3_hand_only_layout(payload: dict[str, Any]) -> dict[str, Any]:
    """Map candidate3 joint states to hand-only roots; discard Panda joints."""
    if payload.get("schema") != "dh116-static-pose-v1":
        raise ValueError("candidate3 pose schema mismatch")
    states = payload.get("states")
    if not isinstance(states, dict) or set(states) != {"left", "right"}:
        raise ValueError("candidate3 pose must contain left and right states")
    active: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        values = np.asarray(states[side], dtype=np.float32)
        if values.shape != (13,) or not np.isfinite(values).all():
            raise ValueError(f"candidate3 {side} state must contain 13 finite joint values")
        active[side] = values[7:].copy()
    return {
        "hand_roots": {side: pose.copy() for side, pose in _CANDIDATE3_ROOTS.items()},
        "active_joints": active,
        "wire_center": (0.43, 0.0, 0.308),
        "wire_axis": "Y",
        "wire_length_m": 0.5,
        "wire_radius_m": 0.008,
        "table_center": (0.60, 0.0, 0.25),
        "table_size": (0.80, 1.20, 0.10),
    }


def assess_table_clearance(body_heights_by_side: dict[str, Any], table_top_z_m: float,
                           minimum_clearance_m: float = 0.01) -> dict[str, Any]:
    """Gate a tabletop rollout using measured world-Z for every hand body."""
    if not np.isfinite(table_top_z_m) or not np.isfinite(minimum_clearance_m):
        raise ValueError("table height and minimum clearance must be finite")
    if minimum_clearance_m < 0 or not body_heights_by_side:
        raise ValueError("minimum clearance must be non-negative and body heights non-empty")
    minima: dict[str, float] = {}
    for side, heights in body_heights_by_side.items():
        values = np.asarray(heights, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all() or np.any(np.abs(values) > 10.0):
            raise ValueError(f"{side} body heights must be non-empty, finite, and within scene bounds")
        minima[side] = float(values.min())
    minimum_body_z = min(minima.values())
    clearance = minimum_body_z - float(table_top_z_m)
    passed = clearance >= minimum_clearance_m
    return {
        "passed": passed,
        "table_top_z_m": float(table_top_z_m),
        "minimum_clearance_required_m": float(minimum_clearance_m),
        "minimum_body_z_m": minimum_body_z,
        "minimum_clearance_m": clearance,
        "minimum_body_z_by_side_m": minima,
        "failure_reason": None if passed else (
            "tabletop_penetration" if clearance < 0 else "tabletop_clearance_below_minimum"),
    }


def native_pose_hand_only_layout(payload: dict[str, Any],
                                 roll_degrees: dict[str, float] | None = None) -> dict[str, Any]:
    """Extract measured DH116 world roots/joints from a native pose summary."""
    poses = payload.get("resolved_poses")
    if not isinstance(poses, dict) or set(poses) != {"left", "right"}:
        raise ValueError("native pose summary must contain both resolved hand poses")
    roots: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        row = poses[side]
        root = np.asarray(row.get("dh116_base_pose_w"), dtype=np.float64)
        names = row.get("joint_names")
        positions = np.asarray(row.get("joint_pos"), dtype=np.float64)
        if root.shape != (7,) or not np.isfinite(root).all():
            raise ValueError(f"native {side} hand root must contain seven finite pose values")
        if not isinstance(names, list) or positions.shape != (len(names),) or not np.isfinite(positions).all():
            raise ValueError(f"native {side} joint names/positions are invalid")
        if len(names) != len(set(names)) or not set(ACTIVE).issubset(names):
            raise ValueError(f"native {side} joints do not contain unique active DH116 names")
        roots[side] = root.astype(np.float32)
        by_name = dict(zip(names, positions.tolist()))
        active[side] = np.asarray([by_name[name] for name in ACTIVE], dtype=np.float32)
    if roll_degrees is not None:
        contact_payload = payload.get("contact_points")
        if not isinstance(contact_payload, dict):
            raise ValueError("native summary is missing world contact points for grasp-axis roll")
        contacts = {
            side: np.asarray([contact_payload[side][digit]["point_w"]
                              for digit in ("thumb", "index")], dtype=np.float64)
            for side in ("left", "right")
        }
        roots = rotate_roots_about_grasp_axes(roots, contacts, roll_degrees)
    return {
        "hand_roots": roots,
        "active_joints": active,
        "wire_center": (0.43, 0.0, 0.308),
        "wire_axis": "Y",
        "wire_length_m": 0.5,
        "wire_radius_m": 0.008,
        "table_center": (0.60, 0.0, 0.25),
        "table_size": (0.80, 1.20, 0.10),
    }


def _quat_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                       [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                       [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        return np.asarray([0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                           (matrix[0, 2] - matrix[2, 0]) / scale,
                           (matrix[1, 0] - matrix[0, 1]) / scale])
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        return np.asarray([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                           (matrix[0, 1] + matrix[1, 0]) / scale,
                           (matrix[0, 2] + matrix[2, 0]) / scale])
    if index == 1:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        return np.asarray([(matrix[0, 2] - matrix[2, 0]) / scale,
                           (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                           (matrix[1, 2] + matrix[2, 1]) / scale])
    scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
    return np.asarray([(matrix[1, 0] - matrix[0, 1]) / scale,
                       (matrix[0, 2] + matrix[2, 0]) / scale,
                       (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])


def rotate_roots_about_grasp_axes(roots: dict[str, Any], contacts: dict[str, Any],
                                  roll_degrees: dict[str, float]) -> dict[str, np.ndarray]:
    """Roll each hand around its two fixed world pad targets."""
    rotated: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        root = np.asarray(roots[side], dtype=np.float64)
        pads = np.asarray(contacts[side], dtype=np.float64)
        angle = np.deg2rad(float(roll_degrees[side]))
        if root.shape != (7,) or pads.shape != (2, 3) or not np.isfinite(root).all() or not np.isfinite(pads).all() or not np.isfinite(angle):
            raise ValueError(f"invalid {side} hand roll input")
        axis = pads[1] - pads[0]
        axis /= np.linalg.norm(axis)
        skew = np.asarray([[0.0, -axis[2], axis[1]],
                           [axis[2], 0.0, -axis[0]],
                           [-axis[1], axis[0], 0.0]])
        roll = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
        old_rotation = _quat_to_matrix_wxyz(root[3:])
        new_rotation = roll @ old_rotation
        local_pads = (pads - root[:3]) @ old_rotation
        new_translation = pads.mean(axis=0) - local_pads.mean(axis=0) @ new_rotation.T
        quaternion = _matrix_to_quat_wxyz(new_rotation)
        quaternion /= np.linalg.norm(quaternion)
        rotated[side] = np.concatenate((new_translation, quaternion))
    return rotated


__all__ = ["assess_table_clearance", "candidate3_hand_only_layout",
           "native_pose_hand_only_layout", "rotate_roots_about_grasp_axes"]
