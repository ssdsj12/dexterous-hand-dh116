"""Bounded command and geometry helpers for native single-pad flexible contact."""

import numpy as np


DT = 1 / 480
SUBSTEPS = 12
STEPS = 240
SETUP_STEPS = 2
MEASURED_STEPS = STEPS - SETUP_STEPS
ZERO_STEPS = 48
MOVE_STEPS = 160
VELOCITY_LIMIT = .3
RADIUS_M = .005


def command_at_step(start, endpoint, *, step, side):
    if side not in ("left", "right") or not 1 <= step <= STEPS:
        raise ValueError("expected valid hand and bounded step")
    offset = np.subtract(endpoint, start)
    speed = offset / (MOVE_STEPS * DT)
    if (offset.shape != (6,) or not np.isfinite(speed).all() or np.any(np.abs(speed) > VELOCITY_LIMIT)
            or np.any(np.delete(offset, 1) != 0)):
        raise ValueError("expected safe thumb velocity with other joints unchanged")
    move_ticks = min(max(step - ZERO_STEPS, 0), MOVE_STEPS)
    action = np.zeros(24)
    if ZERO_STEPS < step <= ZERO_STEPS + MOVE_STEPS:
        action[(0 if side == "left" else 6) + 1] = speed[1] / VELOCITY_LIMIT
    return start + offset * (move_ticks / MOVE_STEPS), action


def _points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def min_grid_distance(grid, world_nodes, link_to_world):
    values, inside = grid.sample(_points(np.linalg.inv(link_to_world), world_nodes))
    if not np.isfinite(values[inside]).all():
        raise ValueError("nonfinite SDF inside locked support")
    # Values outside the padded lattice do not participate in backend contact.
    return float(min(.012, values[inside].min())) if inside.any() else .012


def plan_thumb_approach(grid, vertices, triangles, q0, pad_pose, limits, observed):
    """Choose one precontact rod and a small signed thumb motion from real hull geometry."""
    faces = vertices[triangles]
    centers = faces.mean(axis=1)
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    magnitudes = np.linalg.norm(normals, axis=1)
    valid = magnitudes > 1e-10
    centers, normals = centers[valid], normals[valid] / magnitudes[valid, None]
    flip = np.sum(normals * (centers - vertices.mean(axis=0)), axis=1) < 0
    normals[flip] *= -1
    correction = observed @ np.linalg.inv(pad_pose(q0))
    candidates = []
    for sign in (1, -1):
        endpoint = q0.copy()
        endpoint[1] += sign * .09
        if np.any(endpoint < limits[:, 0]) or np.any(endpoint > limits[:, 1]):
            continue
        poses = [correction @ pad_pose(q0 + (endpoint-q0)*f) for f in (.25, .5, .75, 1.)]
        for center, normal in zip(centers, normals):
            tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
            tangent /= np.linalg.norm(tangent)
            for overlap in (.0008, .0013, .0018):
                local = center + normal * (RADIUS_M - overlap)
                nodes = _points(poses[-1], local + np.linspace(-.25, .25, 21)[:, None] * tangent)
                initial = min_grid_distance(grid, nodes, observed)
                if initial <= RADIUS_M + .00035:
                    continue
                distances = [min_grid_distance(grid, nodes, pose) for pose in poses]
                if not .0025 < min(distances) <= distances[-1] <= .0048:
                    continue
                # Favor early establishment and finite margin without large endpoint penetration.
                candidates.append((abs(distances[2] - .0048), -initial, endpoint, nodes, initial, distances))
    if not candidates:
        raise ValueError("no safe nonpenetrating rod placement with the bounded thumb approach")
    _, _, endpoint, nodes, initial, distances = min(candidates, key=lambda row: row[:2])
    return endpoint, nodes, {"initial_min_signed_distance_m": initial,
                             "predicted_min_signed_distances_m": distances,
                             "relative_fk_alignment_max_delta": float(np.abs(correction-np.eye(4)).max())}


def schedule_summary(rows):
    previous = np.zeros(6)
    contact = []
    for index, row in enumerate(rows):
        if (row["phase_step"] != index + 1 or row["rod_substep"] != (index + 1) * SUBSTEPS
                or index and (row["sdk_step"] != rows[index-1]["sdk_step"] + 1
                              or abs(row["sdk_time_s"] - rows[index-1]["sdk_time_s"] - DT) > 2e-7)):
            raise ValueError("non-contiguous SDK or rod clock")
        if not np.allclose(row["applied_wrench_com_w"], previous, atol=3e-5, rtol=2e-5):
            raise ValueError("one-exchange applied wrench does not match the previous sample")
        previous = np.asarray(row["rod_on_hand_com_w"])
        contact.append(float(np.linalg.norm(previous[:3])) > 1e-5)
    zero = not any(contact[:ZERO_STEPS])
    residual = max((row["pair_force_residual_n"] for row in rows), default=None)
    failures = [row["failure"] for row in rows if row["failure"]]
    later_contacts = sum(contact[ZERO_STEPS:])
    return {
        "physics_rows": len(rows), "zero_contact_window_passed": zero,
        "contact_steps_after_zero": later_contacts, "max_pair_force_residual_n": residual,
        "failure_reasons": failures,
        "sample_validation_passed": (len(rows) == MEASURED_STEPS and zero and later_contacts >= 3
                                     and residual is not None and residual <= 3e-4 and not failures),
    }
