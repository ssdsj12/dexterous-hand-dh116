#!/usr/bin/env python3
"""Construct an independent palm-referenced Power starting pose on CPU."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.clearance import separation_lower_bound
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, active_limits, expand
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_teacher_seed_initialization import seed_arrays
from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, box_relation, load_geometry
from scripts.audit_dh116_wire_contact_witness import audit_hand, audit_seed, endpoint_gap, shapes_world
from scripts.prepare_dh116_wire_pose_candidate import HANDS, _material_tangents, _pad_gap_interval, _station_frame
from scripts.solve_dh116_tripod_joint_root_direction import COARSE

ROLL_DEGREES = (0, -45, 45, -90, 90, -135, 135, 180)
FLEXION_TRIALS_RAD = (0., -.10, .10)
SEATING_BACKOFF_M = (0., .004)
APPROACH_STEP_M = .0005
APPROACH_STEPS = 40


def palm_axis_frames(material_frame):
    """Both axial directions preserve a proper right-handed rotation."""
    frame = np.asarray(material_frame, dtype=float)
    if (frame.shape != (3, 3) or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-10, rtol=0.)
            or np.linalg.det(frame) <= 0.):
        raise ValueError("right-handed material frame required")
    return ((1, frame), (-1, frame * [-1., 1., -1.]))


def align_palm(palm, fingers, target_frame):
    """Map this hand's index-to-little span onto the wire, without a reflection."""
    palm, fingers, frame = (np.asarray(value, dtype=float) for value in (palm, fingers, target_frame))
    if (fingers.shape != (5, 3) or palm.ndim != 2 or palm.shape[1] != 3 or not len(palm)
            or frame.shape != (3, 3) or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-10, rtol=0.)
            or np.linalg.det(frame) <= 0. or not np.isfinite(palm).all() or not np.isfinite(fingers).all()):
        raise ValueError("finite same-hand geometry and right-handed material frame required")
    span = fingers[4] - fingers[1]
    width = np.linalg.norm(span)
    if width <= 1e-8:
        raise ValueError("degenerate Power finger span")
    span /= width
    inward = fingers[1:].mean(0) - palm.mean(0)
    inward -= span * np.dot(inward, span)
    depth = np.linalg.norm(inward)
    if depth <= 1e-8:
        raise ValueError("degenerate Power palm axis")
    inward /= depth
    basis = np.column_stack((span, inward, np.cross(span, inward)))
    return Rotation.from_matrix(frame @ basis.T)


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


def support_clearance(shapes, supports):
    hits = []
    minimum_gap = None
    for name, vertices in shapes.items():
        for index, support in enumerate(supports):
            result = box_relation(vertices, support["center_m"], support["size_m"])
            gap = result["max_separating_axis_gap_m"]
            minimum_gap = gap if minimum_gap is None else min(minimum_gap, gap)
            if result["intersects"]:
                hits.append(dict(source=name, support_index=index))
    return hits, minimum_gap


def support_collisions(shapes, supports):
    return support_clearance(shapes, supports)[0]


def audit_fixed_preshape_approach(shapes, nodes, radius, direction, supports):
    """Certify the swept convex envelopes of a horizontal translation at fixed q."""
    axis = np.asarray(direction, dtype=float).copy()
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("finite palm direction required")
    axis[2] = 0.
    length = np.linalg.norm(axis)
    if length <= 1e-8:
        return dict(reason="horizontal_direction_unavailable", last_accepted_translation_m=None,
                    native_contact_verified=False)
    axis /= length
    accepted = None
    rejected = None
    for index in range(APPROACH_STEPS + 1):
        distance = index * APPROACH_STEP_M
        shift = distance * axis
        # At fixed joints and rotation, convex hull(V U (V + shift)) is
        # precisely the continuous translational sweep of each convex hull.
        swept = {key: np.vstack((vertices, vertices + shift)) for key, vertices in shapes.items()}
        hits, support_gap = support_clearance(swept, supports)
        uncertain = [key for key, hull in swept.items() if _pad_gap_interval(hull, nodes, radius)["lower_m"] < 0.]
        evidence = dict(translation_m=distance, support_intersections=hits,
                        minimum_support_separating_axis_gap_m=support_gap, rod_uncertified_shapes=uncertain)
        if hits or (support_gap is not None and support_gap < CLEARANCE_M) or uncertain:
            rejected = evidence
            break
        accepted = evidence
    required = ("source:base_link", *("cooked:" + key for key in DISTAL_PAD_LINKS))
    witnesses = {}
    if accepted is not None:
        shift = accepted["translation_m"] * axis
        witnesses = {key: endpoint_gap(shapes[key] + shift, nodes, radius) for key in required if key in shapes}
    return dict(last_accepted_translation_m=None if accepted is None else accepted["translation_m"],
                last_accepted=accepted, first_rejected=rejected, endpoint_witnesses=witnesses,
                table_clearance_unchanged=True, native_contact_verified=False)


def search_closure_joints(seed, lower, upper, evaluate):
    """Start with coordinated closure, then vary each flexion independently."""
    def score(q):
        result = evaluate(q)
        return None if result is None else dict(result, active_q=q.copy())

    best = score(seed)
    count = 1
    for thumb in (.25, .5, .75):
        for fingers in (.25, .5, .75):
            q = seed.copy()
            q[1:] = (lower + (upper - lower) * np.r_[0., thumb, np.full(4, fingers)])[1:]
            result = score(q)
            count += 1
            if result is not None and (best is None or result["rank"] < best["rank"]):
                best = result

    for joint in range(1, len(seed)):
        center = seed.copy() if best is None else best["active_q"].copy()
        for fraction in (0., .25, .5, .75, 1.):
            q = center.copy()
            q[joint] = lower[joint] + fraction * (upper[joint] - lower[joint])
            if q[joint] == center[joint]:
                continue
            result = score(q)
            count += 1
            if result is not None and (best is None or result["rank"] < best["rank"]):
                best = result
    return best, count


def thumb_clearance(shapes):
    """Conservative separating-axis certificates between thumb and other digits."""
    thumb = [v for key, v in shapes.items() if key.split(":")[1].startswith("finger1")]
    fingers = [v for key, v in shapes.items() if key.split(":")[1].startswith(
        ("finger2", "finger3", "finger4", "finger5"))]
    if not thumb or not fingers:
        raise ValueError("original thumb and other finger hulls required")
    bounds = {id(hull): (hull.min(0), hull.max(0)) for hull in thumb + fingers}
    normals = {}
    gaps = []
    for a in thumb:
        lo_a, hi_a = bounds[id(a)]
        for b in fingers:
            lo_b, hi_b = bounds[id(b)]
            gap = float(np.maximum(lo_a - hi_b, lo_b - hi_a).max())
            if gap <= 0.:
                # AABB overlap alone is unresolved. Try source hull face axes
                # and retain only a certified strictly positive separation.
                for hull in (a, b):
                    if id(hull) not in normals:
                        normals[id(hull)] = ConvexHull(hull).equations[:, :3]
                gap = separation_lower_bound(a, b, np.vstack((normals[id(a)], normals[id(b)])))
            gaps.append(gap)
    return min(gaps)


def audit_closure_positions(geometry, row, nodes, supports):
    """Screen endpoint geometry with real FK; do not certify a motion or load."""
    distance = row["fixed_preshape_approach"]["last_accepted_translation_m"]
    if distance is None:
        return []
    axis = np.asarray(row["palm_toward_fingers"]) * [1., 1., 0.]
    axis /= np.linalg.norm(axis)
    layout = geometry["reference"]["reference_layout"]
    table_top = layout["table_center"][2] + layout["table_size"][2] / 2
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    required = ("source:base_link", *("cooked:" + pad for pad in DISTAL_PAD_LINKS))
    results = []
    for fraction in (0., .5, 1.):
        root = np.asarray(row["root"]).copy()
        root[:3] += distance * fraction * axis
        rejected = dict(table=0, support=0, rod=0, thumb_clearance=0)

        def evaluate(q):
            expand(q)
            hulls = shapes_world(geometry, root, q)
            clearance = min(float(vertices[:, 2].min()) for vertices in hulls.values()) - table_top
            if clearance < CLEARANCE_M:
                rejected["table"] += 1
                return None
            hits, support_gap = support_clearance(hulls, supports)
            if hits or (support_gap is not None and support_gap < CLEARANCE_M):
                rejected["support"] += 1
                return None
            gaps = {key: _pad_gap_interval(vertices, nodes, radius) for key, vertices in hulls.items()}
            if any(gap["lower_m"] < 0. for gap in gaps.values()):
                rejected["rod"] += 1
                return None
            thumb_gap = thumb_clearance(hulls)
            if thumb_gap <= 0.:
                rejected["thumb_clearance"] += 1
                return None
            uppers = [gaps[key]["upper_m"] for key in required]
            return dict(rank=(max(uppers), sum(uppers)), hulls=hulls,
                        minimum_support_separating_axis_gap_m=support_gap,
                        thumb_separating_axis_gap_m=thumb_gap)

        best, count = search_closure_joints(np.asarray(row["active_q"]), *active_limits(), evaluate)
        result = dict(wrist_fraction=fraction, evaluations=count, rejections=rejected,
                      root=root.tolist(), endpoint_geometry_passed=False)
        if best is not None:
            hulls = best.pop("hulls")
            witness = audit_hand(hulls, nodes, radius, layout, DISTAL_PAD_LINKS, include_palm=True)
            directions = {key: (witness["gaps"][key]["witness"] or {}).get("direction")
                          for key in required}
            palm, thumb = (directions[key] for key in required[:2])
            fingers = [directions[key] for key in required[2:]]
            palm_dots = [None if palm is None or finger is None else float(np.dot(palm, finger))
                         for finger in fingers]
            thumb_dots = list(witness["witness_direction_dots"].values())
            y = np.concatenate([v[:, 1] for v in hulls.values()])
            endpoint = (witness["geometric_approach_passed"]
                        and all(dot is not None and dot < 0. for dot in palm_dots)
                        and any(dot is not None and dot < -.2 for dot in thumb_dots))
            result.update(best, active_q=best["active_q"].tolist(), exact=witness, palm_finger_direction_dots=palm_dots,
                          hand_y_bounds_m=[float(y.min()), float(y.max())], endpoint_geometry_passed=endpoint)
        results.append(result)
    return results


def _search_palm(geometry, pose, q, nodes, station, frame, supports, *, aligned_preshape):
    if geometry["fk"].root.rsplit("/", 1)[-1] != "base_link":
        raise ValueError("Power palm support requires the original base_link")
    layout = geometry["reference"]["reference_layout"]
    table_top = layout["table_center"][2] + layout["table_size"][2] / 2
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    tangent = frame[:, 0]
    reference = Rotation.from_quat(np.roll(pose[3:], -1))
    pads = ["cooked:" + k for k in DISTAL_PAD_LINKS]
    trials = []
    minimum, maximum = active_limits()
    flexions = FLEXION_TRIALS_RAD if aligned_preshape else (0.,)
    for flexion in flexions:
        # Change the five flexion commands, holding thumb abduction fixed.
        active = np.clip(q + np.r_[0., np.full(5, flexion)], minimum, maximum)
        expand(active)
        frames = palm_axis_frames(frame) if aligned_preshape else ((None, frame),)
        measured = shapes_world(geometry, pose, active) if aligned_preshape else None
        for span_sign, target in frames:
            align = (align_palm(measured["source:base_link"], [measured[key].mean(0) for key in pads], target)
                     if aligned_preshape else Rotation.identity())
            for degrees in ROLL_DEGREES:
                angle = Rotation.from_rotvec(np.deg2rad(degrees) * tangent)
                rotated = pose.copy()
                rotated[3:] = np.roll((angle * align * reference).as_quat(), 1)
                shapes = shapes_world(geometry, rotated, active)
                palm = shapes["source:base_link"]
                fingers = np.array([shapes[key].mean(0) for key in pads])
                lowest = min(vertices[:, 2].min() for vertices in shapes.values())
                for backoff in (SEATING_BACKOFF_M if aligned_preshape else (0.,)):
                    root = rotated.copy()
                    finger_group = fingers[1:] if aligned_preshape else fingers
                    shift, direction = seat_palm(palm, finger_group.mean(0), station, tangent,
                                                 radius + COARSE / 2 + backoff)
                    rise = max(0., table_top + CLEARANCE_M + 1e-8 - (lowest + shift[2]))
                    shift[2] += rise
                    root[:3] += shift
                    placed = {key: vertices + shift for key, vertices in shapes.items()}
                    gaps = {key: _pad_gap_interval(vertices, nodes, radius) for key, vertices in placed.items()}
                    support_hits, support_gap = support_clearance(placed, supports)
                    sides = (fingers + shift - station) @ direction
                    overlap = [max(0., -row["lower_m"]) for row in gaps.values()]
                    excess = [max(0., gaps[key]["upper_m"] - COARSE) for key in ("source:base_link", *pads)]
                    support_too_close = aligned_preshape and support_gap is not None and support_gap < CLEARANCE_M
                    rank = (int(support_too_close) + len(support_hits) + sum(value > 0. for value in overlap), sum(overlap),
                            max(excess), sum(excess), -float(sides.min()))
                    local_span = fingers[4] - fingers[1]
                    trials.append(dict(root=root.tolist(), active_q=active.tolist(), roll_degrees=degrees,
                        requested_flexion_delta_rad=flexion, seating_backoff_m=backoff,
                        span_sign=span_sign, span_tangent_cosine=float(local_span @ tangent / np.linalg.norm(local_span)),
                        table_rise_m=float(rise), translation_m=shift.tolist(), palm_toward_fingers=direction.tolist(),
                        finger_side_projection_m=dict(zip(DISTAL_PAD_LINKS, sides.tolist())),
                        initial_palm_clearance_m=gaps["source:base_link"], support_intersections=support_hits,
                        minimum_support_separating_axis_gap_m=support_gap, rank=rank))
    return min(trials, key=lambda row: row["rank"]), len(trials)


def prepare_palm_seed(seed, *, stations, project, supports=(), aligned_preshape=False,
                      audit_approach=False, audit_closure=False):
    if audit_closure and not audit_approach:
        raise ValueError("closure audit requires a fixed-preshape approach audit")
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
        geometry = load_geometry(project, side=side)
        selected, count = _search_palm(geometry, roots[idx],
            joints[idx, active_ids], nodes, nodes[node], frame, supports, aligned_preshape=aligned_preshape)
        trial["roots"][idx] = selected["root"]
        if aligned_preshape:
            trial["joints"][idx] = expand(selected["active_q"]).tolist()
        search[side] = dict(selected, node_id=node, candidates_evaluated=count)
    seed_arrays(trial)
    evidence = audit_seed(trial, style="power", project=project)
    if audit_approach:
        for side, row in search.items():
            if not evidence["hands"][side]["table_clearance_passed"]:
                row["fixed_preshape_approach"] = dict(reason="table_clearance_not_qualified",
                    last_accepted_translation_m=None, native_contact_verified=False)
                continue
            geometry = load_geometry(project, side=side)
            hulls = shapes_world(geometry, row["root"], row["active_q"])
            row["fixed_preshape_approach"] = audit_fixed_preshape_approach(
                hulls, nodes, geometry["old_candidate"]["rod"]["radius_m"], row["palm_toward_fingers"], supports)
            if audit_closure:
                row["closure_endpoints"] = audit_closure_positions(geometry, row, nodes, supports)
    opposed_sides = all(all(item > 0. for item in row["finger_side_projection_m"].values())
                        for row in search.values())
    supports_clear = all(
        not row["support_intersections"] and (not aligned_preshape or not supports
            or row["minimum_support_separating_axis_gap_m"] >= CLEARANCE_M)
        for row in search.values())
    result = dict(schema="dh116-power-palm-referenced-seed-v1", trial_seed=trial, search=search, evidence=evidence,
        support_boxes_audited=len(supports), discrete_support_clearance_passed=supports_clear,
        aligned_preshape=aligned_preshape,
        geometric_wrap_passed=opposed_sides and supports_clear and evidence["static_approach_passed"],
        locked_palm_cooked_available=False, palm_patch_identity_proven=False,
        native_contact_verified=False, palm_contact_verified=False, teacher_admission=False)
    if audit_closure:
        pairs = []
        for left in search["left"].get("closure_endpoints", []):
            for right in search["right"].get("closure_endpoints", []):
                if left["endpoint_geometry_passed"] and right["endpoint_geometry_passed"]:
                    a, b = left["hand_y_bounds_m"], right["hand_y_bounds_m"]
                    gap = max(a[0] - b[1], b[0] - a[1])
                    if gap > 0.:
                        pairs.append(dict(left=left["wrist_fraction"], right=right["wrist_fraction"],
                                          bilateral_y_clearance_m=gap))
        result["closure_screen"] = dict(pairs=pairs, continuous_path_verified=False,
                                        native_contact_verified=False, teacher_admission=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed", type=Path)
    source.add_argument("--end-supported", action="store_true",
                        help="use the existing independent Power seed and support boxes")
    parser.add_argument("--aligned-preshape", action="store_true",
                        help="align each hand from its FK finger span and try bounded flexion offsets")
    parser.add_argument("--audit-approach", action="store_true",
                        help="audit a bounded fixed-joint horizontal sweep; does not move the candidate")
    parser.add_argument("--audit-closure", action="store_true",
                        help="screen independent joint endpoints at three certified wrist positions")
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
                               project=args.project, supports=supports, aligned_preshape=args.aligned_preshape,
                               audit_approach=args.audit_approach, audit_closure=args.audit_closure)
    result["input_seed_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if args.audit_closure:
        return 0 if result["closure_screen"]["pairs"] else 2
    return 0 if result["geometric_wrap_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
