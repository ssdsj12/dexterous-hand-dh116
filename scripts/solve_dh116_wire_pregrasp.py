"""Bounded CPU wrist and finger search for the original 41-node seed."""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, box_relation, load_geometry, world_vertices
from scripts.prepare_dh116_wire_pose_candidate import HANDS, _material_tangents, _pad_gap_interval
from scripts.solve_dh116_tripod_joint_root_direction import COARSE
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_teacher_seed_initialization import seed_arrays

# The existing style-specific offline translation and joint/rotation scales.
TRANSLATION_M = {"tripod": .005, "power": .015}
JOINT_COUNT = {"tripod": 4, "power": 6}
LEVELS = (1., .5, .25, .125, .0625)
MAX_EVALUATIONS = 1 + 2 * (6 + max(JOINT_COUNT.values())) * len(LEVELS)


def _closest_vertex(hull, nodes, edges, squared_lengths):
    offsets = hull[:, None, :] - nodes[None, :-1, :]
    t = np.clip(np.einsum("vsi,si->vs", offsets, edges) / squared_lengths, 0., 1.)
    vectors = offsets - t[..., None] * edges
    squared = np.einsum("vsi,vsi->vs", vectors, vectors)
    vertex, segment = np.unravel_index(squared.argmin(), squared.shape)
    distance = float(np.sqrt(squared[vertex, segment]))
    direction = vectors[vertex, segment] / distance if distance > 1e-12 else None
    return distance, direction


def _coordinate_search(evaluate, lower, upper):
    """Fixed ordered steps with at most two evaluations per dimension and level."""
    at = np.zeros_like(lower)
    initial = best = evaluate(at)
    evaluations = 1
    for level in LEVELS:
        for axis in range(len(at)):
            origin = at.copy()
            for direction in (-1, 1):
                trial = origin.copy()
                trial[axis] = np.clip(trial[axis] + direction * level, lower[axis], upper[axis])
                if trial[axis] == origin[axis]:
                    continue
                row = evaluate(trial)
                evaluations += 1
                if row["rank"] < best["rank"]:
                    best, at = row, trial
    return initial, best, evaluations


def _search_hand(geometry, pose, q, nodes, style):
    count = JOINT_COUNT[style]
    pads = DISTAL_PAD_LINKS[:3 if style == "tripod" else 5]
    layout = geometry["reference"]["reference_layout"]
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    table_top = layout["table_center"][2] + layout["table_size"][2] / 2
    limits_min, limits_max = active_limits()
    upper = np.r_[np.ones(6), np.minimum(1., (limits_max[:count] - q[:count]) / .1)]
    lower = np.r_[-1., -1., 0., -1., -1., -1., np.maximum(-1., (limits_min[:count] - q[:count]) / .1)]
    edges = np.diff(nodes, axis=0)
    squared_lengths = np.einsum("si,si->s", edges, edges)
    scale = TRANSLATION_M[style]
    rotation = Rotation.from_quat(np.roll(pose[3:], -1))

    def evaluate(x):
        root = pose.copy()
        root[:2] += scale * x[:2]
        root[3:] = np.roll((Rotation.from_rotvec(.1 * x[3:6]) * rotation).as_quat(), 1)
        active = q.copy()
        active[:count] = np.clip(q[:count] + .1 * x[6:], limits_min[:count], limits_max[:count])
        source, cooked = (world_vertices(geometry, root, active, cooked=locked) for locked in (False, True))
        shapes = {**{"source:" + k: v for k, v in source.items()},
                  **{"cooked:" + k: v for k, v in cooked.items()}}
        min_z = min(v[:, 2].min() for v in shapes.values())
        required_rise = max(0., table_top + CLEARANCE_M + 1e-8 - min_z)
        rise = required_rise + scale * x[2]
        root[2] += rise
        shapes = {n: v + [0., 0., rise] for n, v in shapes.items()}
        distances = {}
        directions = {}
        for name, hull in shapes.items():
            distance, direction = _closest_vertex(hull, nodes, edges, squared_lengths)
            distances[name] = distance - radius
            directions[name] = direction
        values = [distances["cooked:" + pad] for pad in pads]
        thumb = directions["cooked:" + pads[0]]
        dots = {}
        for pad in pads[1:]:
            partner = directions["cooked:" + pad]
            dots[pad] = float(thumb @ partner) if thumb is not None and partner is not None else None
        error = sum(((gap - COARSE / 2) * 1000.) ** 2 for gap in values)
        error += 4. * sum((max(0., -gap) * 1000.) ** 2 for n, gap in distances.items()
                          if n.split(":")[1] not in pads)
        error += 4. * sum((1.2 if dot is None else max(0., dot + .2)) ** 2 for dot in dots.values())
        return dict(root=root, active=active, shapes=shapes, rank=float(error), dots=dots)

    initial, best, used = _coordinate_search(evaluate, lower, upper)
    table = {name: box_relation(v, layout["table_center"], layout["table_size"])
             for name, v in best["shapes"].items()}
    pad_geometry = {pad: _pad_gap_interval(best["shapes"]["cooked:" + pad], nodes, radius) for pad in pads}
    nonrequired = {name: _pad_gap_interval(v, nodes, radius)["lower_m"]
                   for name, v in best["shapes"].items() if name.split(":")[1] not in pads}
    table_passed = all(row["top_clearance_m"] >= CLEARANCE_M and not row["intersects"] for row in table.values())
    pad_passed = all(row["lower_m"] >= 0. and row["upper_m"] <= COARSE for row in pad_geometry.values())
    directions_passed = all(dot is not None and dot < -.2 for dot in best["dots"].values())
    clear_nonrequired = min(nonrequired.values()) >= 0.
    y_bounds = [min(float(v[:, 1].min()) for v in best["shapes"].values()),
                max(float(v[:, 1].max()) for v in best["shapes"].values())]
    return best["root"], expand(best["active"]), dict(
        active_search_joints=list(ACTIVE[:count]), evaluations=used, objective=best["rank"],
        initial_objective=initial["rank"], candidate_root_z_change_m=float(best["root"][2] - pose[2]),
        root_xy_search_limit_m=scale, required_pad_max_gap_m=COARSE, table_clearance_passed=table_passed,
        minimum_table_clearance_m=min(row["top_clearance_m"] for row in table.values()),
        pad_geometry=pad_geometry, pad_gap_gate_passed=pad_passed, vertex_direction_dots=best["dots"],
        vertex_direction_gate_passed=directions_passed, nonrequired_rod_lower_m=nonrequired,
        nonrequired_rod_clear=clear_nonrequired, hand_y_bounds_m=y_bounds,
        endpoint_geometry_passed=table_passed and pad_passed and directions_passed and clear_nonrequired,
        source_sha256=geometry["source_sha256"])


def search_pregrasp(seed, *, style, project):
    if style not in JOINT_COUNT:
        raise ValueError("joint pregrasp search requires Tripod or Power")
    nodes, full, roots, _ = seed_arrays(seed)
    _material_tangents(nodes)
    active_indices = [JOINTS.index(name) for name in ACTIVE]
    trial = deepcopy(seed)
    hands = {}
    for i, side in enumerate(HANDS):
        root, joints, hands[side] = _search_hand(
            load_geometry(project, side=side), roots[i], full[i, active_indices], nodes, style)
        trial["roots"][i], trial["joints"][i] = root.tolist(), joints.tolist()
    seed_arrays(trial)
    left, right = (hands[side]["hand_y_bounds_m"] for side in HANDS)
    separation = max(left[0] - right[1], right[0] - left[1])
    return dict(trial_seed=trial, hands=hands, max_evaluations_per_hand=MAX_EVALUATIONS,
        scope="static_joint_wrist_search_on_input_polyline", bilateral_y_clearance_m=separation,
        geometry_candidate_passed=separation > 0. and all(row["endpoint_geometry_passed"] for row in hands.values()),
        self_collision_verified=False, continuous_path_verified=False, palm_contact_verified=False,
        native_contact_verified=False, teacher_admission=False)
