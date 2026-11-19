"""Bounded CPU wrist and finger search for the original 41-node seed."""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, box_relation, load_geometry, world_vertices
from scripts.prepare_dh116_wire_pose_candidate import HANDS, _material_tangents, _pad_gap_interval
from scripts.solve_dh116_tripod_joint_root_direction import COARSE
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import NUMERIC_GUARD
from go2_pvcnn.control.dh116.tabletop_teacher_seed_initialization import seed_arrays

# The existing style-specific offline translation and joint/rotation scales.
TRANSLATION_M = {"tripod": .005, "power": .015}
JOINT_COUNT = {"tripod": 4, "power": 6}
WRIST_ROTATION_RAD = .1
TRIPOD_COUPLED_LEVELS = (.0625, .03125, .015625, .0078125)
# XYZ translation, XYZ rotation, then the four style-specific active joints.
TRIPOD_COUPLED_AXES = ((0, 1), (0, 2), (1, 2), (0, 4), (1, 5), (2, 4), (2, 5),
    (0, 6), (1, 6), (2, 6), (0, 7), (1, 7), (2, 7),
    (0, 8), (1, 8), (2, 8), (0, 9), (1, 9), (2, 9),
    (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9),
    (3, 6), (3, 7), (4, 6), (4, 7), (5, 6), (5, 7))
MAX_COUPLED_EVALUATIONS = 4 * len(TRIPOD_COUPLED_LEVELS) * len(TRIPOD_COUPLED_AXES)
LEVELS = (1., .5, .25, .125, .0625)
REFINEMENT_LEVELS = LEVELS + (.03125, .015625)
MAX_DIMENSIONS = 6 + max(JOINT_COUNT.values())
MAX_EVALUATIONS = 1 + 2 * MAX_DIMENSIONS * len(LEVELS)
MAX_REFINEMENT_EVALUATIONS = 1 + 2 * MAX_DIMENSIONS * len(REFINEMENT_LEVELS)


def _closest_vertex(hull, nodes, edges, squared_lengths):
    offsets = hull[:, None, :] - nodes[None, :-1, :]
    t = np.clip(np.einsum("vsi,si->vs", offsets, edges) / squared_lengths, 0., 1.)
    vectors = offsets - t[..., None] * edges
    squared = np.einsum("vsi,vsi->vs", vectors, vectors)
    vertex, segment = np.unravel_index(squared.argmin(), squared.shape)
    distance = float(np.sqrt(squared[vertex, segment]))
    direction = vectors[vertex, segment] / distance if distance > 1e-12 else None
    return distance, direction


def _coordinate_search(evaluate, lower, upper, *, start=None, levels=LEVELS):
    """Fixed ordered steps with at most two evaluations per dimension and level."""
    at = np.zeros_like(lower) if start is None else start.copy()
    initial = best = evaluate(at)
    evaluations = 1
    for level in levels:
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
    return initial, best, evaluations, at


def _tripod_coupled_search(evaluate, coordinates, best, lower, upper):
    evaluations = 0
    for stride in TRIPOD_COUPLED_LEVELS:
        for i, j in TRIPOD_COUPLED_AXES:
            axes = [i, j]
            for direction in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                target = np.clip(coordinates[axes] + stride * np.array(direction), lower[axes], upper[axes])
                if np.any(target == coordinates[axes]):
                    continue
                trial = coordinates.copy()
                trial[axes] = target
                row = evaluate(trial)
                evaluations += 1
                if row["rank"] < best["rank"]:
                    best, coordinates = row, trial
    return best, evaluations, coordinates


def _palm_vertical_reach(hull, root_xyz, nodes, radius):
    """One-way table clearance correction cannot lower the rigid palm."""
    lever = np.linalg.norm(hull - root_xyz, axis=1).max()
    angle = np.sqrt(3.) * WRIST_ROTATION_RAD
    displacement = float(2. * lever * np.sin(angle / 2.))
    lower_gap = float(hull[:, 2].min() - nodes[:, 2].max() - radius - displacement - NUMERIC_GUARD)
    return dict(vertical_gap_lower_m=lower_gap, max_rotation_displacement_m=displacement,
        can_rule_out_coarse_approach=lower_gap > COARSE)


def _constraint_violations(row, pads, nodes, radius):
    segment_min = np.minimum(nodes[:-1], nodes[1:])
    segment_max = np.maximum(nodes[:-1], nodes[1:])
    violations = {}
    for name, hull in row["shapes"].items():
        source, link = name.split(":")
        required = source == "cooked" and link in pads
        axis_separation = np.maximum(hull.min(0) - segment_max, segment_min - hull.max(0))
        if not required and np.all(axis_separation.max(1) > radius + NUMERIC_GUARD + 1e-10):
            # These are the same axis supports already included in certify().
            violations[name + ":nonpenetration"] = 0.
            continue
        gap = _pad_gap_interval(hull, nodes, radius)
        violations[name + ":nonpenetration"] = max(0., -gap["lower_m"] * 1000.)
        if required:
            violations[name + ":approach"] = max(0., (gap["upper_m"] - COARSE) * 1000.)
    for pad, dot in row["dots"].items():
        # Strict dot < -.2 mirrors the existing geometric direction gate.
        violations[pad + ":opposition"] = 1.2 if dot is None else (
            0. if dot < -.2 else max(1e-12, dot + .2))
    return violations


def _refine_hand(evaluate, start, lower, upper, pads, nodes, radius, *, coupled):
    """Retain the first search result unless certified endpoint violations improve."""
    baseline = _constraint_violations(start, pads, nodes, radius)
    preserve = [key for key, value in baseline.items() if value == 0.]

    def evaluate_constraints(x):
        row = evaluate(x)
        violations = _constraint_violations(row, pads, nodes, radius)
        total = sum(value ** 2 for value in violations.values())
        penetrations = sum(value > 0. for key, value in violations.items() if key.endswith(":nonpenetration"))
        rank = (any(violations[key] > 0. for key in preserve), penetrations, total)
        return dict(row=row, rank=rank, violations=violations)

    initial, best, used, coordinates = _coordinate_search(
        evaluate_constraints, lower, upper, start=start["coordinates"], levels=REFINEMENT_LEVELS)
    detail = dict(evaluations=used, baseline_violations=baseline, baseline_total_violation=initial["rank"][-1])
    if coupled:
        before = best["violations"]
        preserve = [key for key, value in before.items() if value == 0.]
        best, evaluations, coordinates = _tripod_coupled_search(
            evaluate_constraints, coordinates, best, lower, upper)
        detail["coupled"] = dict(evaluations=evaluations, max_evaluations=MAX_COUPLED_EVALUATIONS,
            before_violations=before, selected_coordinates=coordinates.tolist(),
            lower=lower.tolist(), upper=upper.tolist())
    detail.update(selected_violations=best["violations"], selected_total_violation=best["rank"][-1])
    return best["row"], detail


def _y_bounds(shapes):
    return [min(float(v[:, 1].min()) for v in shapes.values()),
            max(float(v[:, 1].max()) for v in shapes.values())]


def _bilateral_clearance(bounds):
    left, right = bounds
    return max(left[0] - right[1], right[0] - left[1])


def _required_source_clearance(shapes, pads, nodes, radius):
    return {name: _pad_gap_interval(shapes[name], nodes, radius)["lower_m"]
            for name in ("source:" + pad for pad in pads)}


def _search_hand(geometry, pose, q, nodes, style, *, refine, coupled):
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
        root[3:] = np.roll((Rotation.from_rotvec(WRIST_ROTATION_RAD * x[3:6]) * rotation).as_quat(), 1)
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
                          if n not in {"cooked:" + pad for pad in pads})
        error += 4. * sum((1.2 if dot is None else max(0., dot + .2)) ** 2 for dot in dots.values())
        return dict(root=root, active=active, shapes=shapes, rank=float(error), dots=dots)

    initial, baseline, used, coordinates = _coordinate_search(evaluate, lower, upper)
    best = baseline
    refinement = None
    if refine:
        best, refinement = _refine_hand(
            evaluate, dict(baseline, coordinates=coordinates), lower, upper, pads, nodes, radius, coupled=coupled)
    palm_reach = None
    if refine and style == "power":
        if geometry["fk"].root.rsplit("/", 1)[-1] != "base_link":
            raise ValueError("palm reach certificate requires base_link as the FK root")
        palm = world_vertices(geometry, pose, q)["base_link"]
        palm_reach = _palm_vertical_reach(palm, pose[:3], nodes, radius)

    def describe(best, refinement):
        table = {name: box_relation(v, layout["table_center"], layout["table_size"])
                 for name, v in best["shapes"].items()}
        pad_geometry = {pad: _pad_gap_interval(best["shapes"]["cooked:" + pad], nodes, radius) for pad in pads}
        nonrequired = {name: _pad_gap_interval(v, nodes, radius)["lower_m"]
                       for name, v in best["shapes"].items() if name.split(":")[1] not in pads}
        table_passed = all(row["top_clearance_m"] >= CLEARANCE_M and not row["intersects"] for row in table.values())
        pad_passed = all(row["lower_m"] >= 0. and row["upper_m"] <= COARSE for row in pad_geometry.values())
        directions_passed = all(dot is not None and dot < -.2 for dot in best["dots"].values())
        clear_nonrequired = min(nonrequired.values()) >= 0.
        source_pads = _required_source_clearance(best["shapes"], pads, nodes, radius)
        source_pads_clear = min(source_pads.values()) >= 0.
        y_bounds = _y_bounds(best["shapes"])
        evidence = dict(
            active_search_joints=list(ACTIVE[:count]), evaluations=used, objective=best["rank"],
            initial_objective=initial["rank"], candidate_root_z_change_m=float(best["root"][2] - pose[2]),
            root_xy_search_limit_m=scale, required_pad_max_gap_m=COARSE, table_clearance_passed=table_passed,
            minimum_table_clearance_m=min(row["top_clearance_m"] for row in table.values()),
            pad_geometry=pad_geometry, pad_gap_gate_passed=pad_passed, vertex_direction_dots=best["dots"],
            vertex_direction_gate_passed=directions_passed, nonrequired_rod_lower_m=nonrequired,
            nonrequired_rod_clear=clear_nonrequired, hand_y_bounds_m=y_bounds,
            required_source_rod_lower_m=source_pads, required_source_rod_clear=source_pads_clear,
            endpoint_geometry_passed=table_passed and pad_passed and directions_passed
                                     and clear_nonrequired and source_pads_clear,
            source_sha256=geometry["source_sha256"])
        if refinement is not None:
            evidence["refinement"] = refinement
            if style == "tripod":
                from scripts.audit_dh116_wire_contact_witness import audit_hand

                witness = audit_hand(best["shapes"], nodes, radius, layout, pads)
                certified = witness["geometric_approach_passed"] and witness["direction_gate_passed"]
                evidence.update(search_proxy_endpoint_passed=evidence["endpoint_geometry_passed"],
                    endpoint_geometry_passed=certified,
                    endpoint_witness={key: witness[key] for key in (
                        "gaps", "witness_direction_dots", "direction_gate_passed", "geometric_approach_passed")})
            if style == "power":
                link = "base_link"
                evidence["palm_geometry"] = dict(link=link,
                    source=geometry["source_colliders"][link]["kind"],
                    rod_gap=_pad_gap_interval(best["shapes"]["source:" + link], nodes, radius),
                    locked_cooked_available=link in geometry["cooked_pads"], native_reaction_measured=False,
                    local_vertical_reach=palm_reach)
        return evidence

    evidence = describe(best, refinement)
    baseline_evidence = None
    if refinement is not None:
        retained = dict(refinement, selected_total_violation=refinement["baseline_total_violation"],
                        selected_violations=refinement["baseline_violations"])
        baseline_evidence = describe(baseline, retained)
        if baseline_evidence["endpoint_geometry_passed"] and not evidence["endpoint_geometry_passed"]:
            return baseline, baseline_evidence, baseline, baseline_evidence
    return best, evidence, baseline, baseline_evidence


def search_pregrasp(seed, *, style, project, refine=False, coupled=False):
    if style not in JOINT_COUNT:
        raise ValueError("joint pregrasp search requires Tripod or Power")
    if coupled and (style != "tripod" or not refine):
        raise ValueError("coupled search requires Tripod refinement")
    nodes, full, roots, _ = seed_arrays(seed)
    _material_tangents(nodes)
    active_indices = [JOINTS.index(name) for name in ACTIVE]
    trial = deepcopy(seed)
    hands = {}
    baseline_seed = deepcopy(seed) if refine else None
    baseline_hands = {}
    for i, side in enumerate(HANDS):
        selected, hands[side], baseline, baseline_hands[side] = _search_hand(
            load_geometry(project, side=side), roots[i], full[i, active_indices], nodes, style, refine=refine, coupled=coupled)
        trial["roots"][i] = selected["root"].tolist()
        trial["joints"][i] = expand(selected["active"]).tolist()
        if refine:
            baseline_seed["roots"][i] = baseline["root"].tolist()
            baseline_seed["joints"][i] = expand(baseline["active"]).tolist()
    seed_arrays(trial)
    separation = _bilateral_clearance([hands[side]["hand_y_bounds_m"] for side in HANDS])
    result = dict(trial_seed=trial, hands=hands, max_evaluations_per_hand=MAX_EVALUATIONS,
        scope="static_joint_wrist_search_on_input_polyline", bilateral_y_clearance_m=separation,
        geometry_candidate_passed=separation > 0. and all(row["endpoint_geometry_passed"] for row in hands.values()),
        self_collision_verified=False, continuous_path_verified=False, palm_contact_verified=False,
        native_contact_verified=False, teacher_admission=False)
    if refine:
        baseline_y_gap = _bilateral_clearance([baseline_hands[side]["hand_y_bounds_m"] for side in HANDS])
        result["refinement"] = dict(baseline_seed=baseline_seed,
            baseline_bilateral_y_clearance_m=baseline_y_gap, max_evaluations_per_hand=MAX_REFINEMENT_EVALUATIONS)
        if baseline_y_gap > 0. and separation <= 0.:
            # Preserve separation even if the independently optimized hands each improve.
            result["refinement"]["rejection"] = "refined_hand_y_ranges_overlap"
            result.update(trial_seed=baseline_seed, hands=baseline_hands, bilateral_y_clearance_m=baseline_y_gap,
                          geometry_candidate_passed=all(row["endpoint_geometry_passed"] for row in baseline_hands.values()))
    return result
