#!/usr/bin/env python3
"""Audit bounded object-local Power wrist offsets on the same supported rod."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
from go2_pvcnn.control.dh116.tabletop_teacher_seed_initialization import seed_arrays
from go2_pvcnn.control.dh116.tabletop_teacher_supported_runtime import supported_seed_plan
from scripts.audit_dh116_pinch_initial_pose import DISTAL_PAD_LINKS, checked_json, load_geometry, path_displacement_bound
from scripts.audit_dh116_wire_contact_witness import audit_hand, endpoint_gap, shapes_world
from scripts.prepare_dh116_power_palm_seed import (
    active_limits, audit_closure_at_root, support_clearance, thumb_clearance,
)
from scripts.prepare_dh116_wire_pose_candidate import HANDS, _material_tangents, _pad_gap_interval, _station_frame

ANGLES_DEG = (0., -30., -15., 15., 30.)
LATERAL_M = (0., -.004, .004, -.008, .008)
BACKOFF_M = (0., .004)
SHORTLIST = 10


def material_wrist_seed(root, station, tangent, inward, *, angle_deg, lateral_m, backoff_m):
    axis = np.cross(tangent, inward)
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length <= 1e-8:
        raise ValueError("nondegenerate material cross-section required")
    spin = Rotation.from_rotvec(np.deg2rad(angle_deg) * tangent)
    offset = root[:3] - station + lateral_m * axis / length - backoff_m * inward
    orientation = spin * Rotation.from_quat(np.roll(root[3:], -1))
    return np.r_[station + spin.apply(offset), np.roll(orientation.as_quat(), 1)]


def palm_approach_samples(measure):
    """Find near-palm endpoints in a bounded single-axis sweep, including between grid points."""
    target = .00025
    rows = [dict(inward_m=mm/1000, **measure(mm/1000)) for mm in range(0, 25, 2)]
    refinements = []
    for first, last in zip(rows, rows[1:]):
        if not first["palm_gap_m"] > target >= last["palm_gap_m"]:
            continue
        lo, hi = first["inward_m"], last["inward_m"]
        for _ in range(6):
            distance = (lo + hi) / 2
            candidate = dict(inward_m=distance, **measure(distance))
            refinements.append(candidate)
            if candidate["palm_gap_m"] >= target:
                lo = distance
            else:
                hi = distance
    return sorted([*rows, *refinements], key=lambda candidate: candidate["inward_m"])


def rotate_root_about_witness(root, witness, axis, degrees):
    """Rotate a wxyz wrist pose about a point and axis expressed in World."""
    axis = np.asarray(axis, dtype=float)
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length <= 1e-8:
        raise ValueError("nondegenerate world rotation axis required")
    spin = Rotation.from_rotvec(np.deg2rad(degrees) * (axis / length))
    orientation = spin * Rotation.from_quat(np.roll(root[3:], -1))
    return np.r_[witness + spin.apply(root[:3] - witness), np.roll(orientation.as_quat(), 1)]


def transport_power_station(root, inward, nodes, tangents, *, source, target):
    if not (0 < source < len(nodes) - 1 and 0 < target < len(nodes) - 1):
        raise ValueError("interior material stations required")
    source_frame = _station_frame(tangents, source)
    target_frame = _station_frame(tangents, target)
    change = Rotation.from_matrix(target_frame @ source_frame.T)
    orientation = change * Rotation.from_quat(np.roll(root[3:], -1))
    return (np.r_[nodes[target] + change.apply(root[:3] - nodes[source]),
                  np.roll(orientation.as_quat(), 1)], change.apply(inward))


def seated_geometry(hulls, nodes, supports, top, radius):
    table = min(float(hull[:, 2].min() - top) for hull in hulls.values())
    hits, support = support_clearance(hulls, supports)
    rod = {}
    for key, hull in hulls.items():
        gap = _pad_gap_interval(hull, nodes, radius)["lower_m"]
        rod[key] = endpoint_gap(hull, nodes, radius)["lower_m"] if gap < .00011 else gap
    limiting = min(rod, key=rod.get)
    thumb_gap = thumb_clearance(hulls)
    return dict(table_gap_m=table, support_gap_m=support, support_hits=hits, rod_gap_m=rod[limiting],
                rod_limiting_hull=limiting, thumb_gap_m=thumb_gap,
                eligible=(table >= CLEARANCE_M and not hits and support >= CLEARANCE_M
                          and rod[limiting] >= 0. and thumb_gap > 0.))


def screen_palm_endpoint(g, root, inward, nodes, supports, top, radius, *, reclose=True):
    open_q, _ = active_limits()
    original = shapes_world(g, root, open_q)

    def measure(distance):
        gap = endpoint_gap(original["source:base_link"] + distance * inward, nodes, radius)
        return dict(palm_gap_m=gap["upper_m"], palm_lower_m=gap["lower_m"])

    samples = palm_approach_samples(measure)
    for row in samples:
        row["eligible"] = False
        if row["palm_lower_m"] < 0. or row["palm_gap_m"] > .0005:
            continue
        hulls = {key: hull + row["inward_m"] * inward for key, hull in original.items()}
        row.update(seated_geometry(hulls, nodes, supports, top, radius))
    eligible = [row for row in samples if row["eligible"]]
    closure = None
    if eligible and reclose:
        wrist = root.copy()
        wrist[:3] += min(eligible, key=lambda row: row["palm_gap_m"])["inward_m"] * inward
        closure = audit_closure_at_root(g, wrist, open_q, nodes, supports, include_thumb_abduction=True)
    return dict(open_q_rad=open_q.tolist(), samples=samples, eligible_endpoints=len(eligible),
                closure_reaudit=closure)


def screen_seated_pockets(g, root, inward, nodes, tangents, station, supports, top, radius, *, side):
    setup = screen_palm_endpoint(g, root, inward, nodes, supports, top, radius, reclose=False)
    q = np.asarray(setup["open_q_rad"])
    near = [row for row in setup["samples"] if 0. <= row["palm_lower_m"] <= row["palm_gap_m"] <= .0005]
    candidates = []
    pivot = None
    if side == "left":
        if not near:
            raise ValueError("left source has no near-palm geometric witness")
        advance = min(near, key=lambda row: abs(row["palm_gap_m"] - .0002))["inward_m"]
        for offset in (0, -1, -2, -4):
            target = station + offset
            candidate, direction = transport_power_station(root, inward, nodes, tangents,
                                                            source=station, target=target)
            candidate[:3] += advance * direction
            candidates.append((dict(station=target, mode="station"), candidate))
    else:
        eligible = [row for row in near if row["eligible"]]
        if not eligible:
            raise ValueError("right source has no clear near-palm geometric witness")
        advance = min(eligible, key=lambda row: row["palm_gap_m"])["inward_m"]
        seated = root.copy()
        seated[:3] += advance * inward
        palm = endpoint_gap(shapes_world(g, seated, q)["source:base_link"], nodes, radius)
        if palm["witness"] is None:
            raise ValueError("right palm witness is missing")
        pivot = np.asarray(palm["witness"]["hull_point_m"])
        tangent = _station_frame(tangents, station)[:, 0]
        candidates.append((dict(station=station, mode="reference"), seated))
        for label, axis in (("rod_tangent", tangent), ("transverse", np.cross(tangent, inward))):
            for degree in (-10., -5., 5., 10.):
                candidate = rotate_root_about_witness(seated, pivot, axis, degree)
                candidates.append((dict(station=station, mode=label, degrees=degree), candidate))

    rows = []
    for labels, candidate in candidates:
        hulls = shapes_world(g, candidate, q)
        palm = endpoint_gap(hulls["source:base_link"], nodes, radius)
        bounds = seated_geometry(hulls, nodes, supports, top, radius)
        eligible = bounds.pop("eligible") and 0. <= palm["lower_m"] <= palm["upper_m"] <= .0005
        row = dict(labels, root=candidate.tolist(), **bounds, palm_lower_m=palm["lower_m"],
                   palm_gap_m=palm["upper_m"], eligible=eligible,
                   open_y_bounds_m=[min(float(v[:, 1].min()) for v in hulls.values()),
                                    max(float(v[:, 1].max()) for v in hulls.values())])
        if eligible:
            row["closure"] = audit_closure_at_root(g, candidate, q, nodes, supports,
                                                   include_thumb_abduction=True, refine_uncertain_rod=True)
        rows.append(row)
    return dict(source_station=station, inward_advance_m=advance, open_q_rad=q.tolist(),
                pivot_world_m=None if pivot is None else pivot.tolist(), candidates=rows)


def audit_reach(artifact, *, project, resolve_closure=False, palm_approach_only=False, seated_pockets=False):
    nodes, *_ = seed_arrays(artifact["trial_seed"])
    plan, _ = supported_seed_plan(project)
    supported = plan["seeds_by_style"]["power"]
    if not np.array_equal(nodes, np.asarray(supported["nodes"], dtype=nodes.dtype)):
        raise ValueError("same end-supported Power rod required")
    supports = plan["scene"]["supports"]
    tangents = _material_tangents(nodes)
    required = ("source:base_link", *("cooked:"+name for name in DISTAL_PAD_LINKS))
    hands = {}
    for side in HANDS:
        g = load_geometry(project, side=side)
        row = artifact["search"][side]
        prior = next(x for x in row["closure_endpoints"] if x["wrist_fraction"] == 0.)
        root, q = np.asarray(prior["root"]), np.asarray(prior["active_q"])
        station = row["node_id"]
        inward = np.asarray(row["palm_toward_fingers"])
        layout = g["reference"]["reference_layout"]
        top = layout["table_center"][2] + layout["table_size"][2] / 2
        radius = g["old_candidate"]["rod"]["radius_m"]
        if seated_pockets:
            hands[side] = screen_seated_pockets(g, root, inward, nodes, tangents, station, supports, top,
                                                radius, side=side)
            continue
        if palm_approach_only:
            hands[side] = screen_palm_endpoint(g, root, inward, nodes, supports, top, radius)
            continue
        candidates = []
        for angle, lateral, backoff in itertools.product(ANGLES_DEG, LATERAL_M, BACKOFF_M):
            wrist = material_wrist_seed(root, nodes[station], tangents[station], inward,
                                       angle_deg=angle, lateral_m=lateral, backoff_m=backoff)
            shapes = shapes_world(g, wrist, q)
            rise = max(0., top+CLEARANCE_M+1e-8-min(v[:, 2].min() for v in shapes.values()))
            shapes = {key: hull + [0., 0., rise] for key, hull in shapes.items()}
            wrist[2] += rise
            gaps = {key: _pad_gap_interval(hull, nodes, radius) for key, hull in shapes.items()}
            if min(value["lower_m"] for value in gaps.values()) < 0.:
                continue
            bound = max(gaps[key]["upper_m"] for key in required)
            candidates.append(dict(root=wrist.tolist(), table_correction_m=rise, angle_deg=angle,
                                   lateral_m=lateral, backoff_m=backoff, max_conservative_upper_m=bound,
                                   shapes=shapes))
        checked = []
        for candidate in sorted(candidates, key=lambda row: row["max_conservative_upper_m"])[:SHORTLIST]:
            shapes = candidate.pop("shapes")
            hits, support_gap = support_clearance(shapes, supports)
            if hits or support_gap < CLEARANCE_M or thumb_clearance(shapes) <= 0.:
                continue
            witness = audit_hand(shapes, nodes, radius, layout, DISTAL_PAD_LINKS, include_palm=True)
            if not witness["table_clearance_passed"] or not witness["complete_convex_clearance_passed"]:
                continue
            candidate["terminal"] = witness
            candidate["max_exact_upper_m"] = max(witness["gaps"][key]["upper_m"] for key in required)
            checked.append(candidate)
        checked.sort(key=lambda candidate: candidate["max_exact_upper_m"])
        baseline = max(prior["exact"]["gaps"][key]["upper_m"] for key in required)
        hands[side] = dict(node_id=station, candidates=len(ANGLES_DEG)*len(LATERAL_M)*len(BACKOFF_M),
                           rod_clear_candidates=len(candidates), unchanged_q_rad=q.tolist(),
                           baseline_max_exact_upper_m=baseline,
                           checked=checked)
        if resolve_closure:
            trials = [("reference", root)]
            for candidate in checked:
                wrist = np.asarray(candidate["root"])
                rotation_delta = (Rotation.from_quat(np.roll(wrist[3:], -1))
                                  * Rotation.from_quat(np.roll(root[3:], -1)).inv())
                if np.linalg.norm(wrist[:3] - root[:3]) > 1e-4 or rotation_delta.magnitude() > 1e-3:
                    trials.append(("material_wrist", wrist))
                    break
            hands[side]["closure_reaudit"] = [
                dict(source=label, **audit_closure_at_root(
                    g, wrist, q, nodes, supports, include_thumb_abduction=True))
                for label, wrist in trials
            ]
    extra = {}
    if seated_pockets:
        left, right = ([row for row in hands[side]["candidates"] if "closure" in row]
                       for side in HANDS)
        extra["open_bilateral_y_separation_m"] = [
            dict(left_station=l["station"], right_mode=h["mode"], right_degrees=h.get("degrees"),
                 gap_m=l["open_y_bounds_m"][0] - h["open_y_bounds_m"][1])
            for l in left for h in right
        ]
    return dict(hands=hands, support_boxes=len(supports), actual_isaac_steps=0,
                continuous_path_verified=False, locked_palm_cooked_available=False,
                native_contact_verified=False, teacher_admission=False, **extra)


def summarize_middle_reach(samples, *, point_travel_m):
    """Bound the continuous minimum gap using the largest nearest-knot displacement."""
    if not samples or not np.isfinite(point_travel_m) or point_travel_m < 0.:
        raise ValueError("nonempty sweep and finite point travel reserve required")
    lower = min(row["gap_lower_m"] for row in samples) - point_travel_m
    best = min(samples, key=lambda row: row["gap_upper_m"])
    best_clear = min((row["gap_upper_m"] for row in samples if row["clear"]), default=None)
    return dict(continuous_minimum_lower_m=lower, best_sample_upper_m=best["gap_upper_m"],
                best_clear_sample_upper_m=best_clear, fixed_wrist_reach_excluded=bool(lower > .0005))


def middle_reach(artifact, seated, *, project):
    nodes, *_ = seed_arrays(artifact["trial_seed"])
    plan, _ = supported_seed_plan(project)
    if not np.array_equal(nodes, np.asarray(plan["seeds_by_style"]["power"]["nodes"], dtype=nodes.dtype)):
        raise ValueError("same end-supported Power rod required")
    supports = plan["scene"]["supports"]
    lower, upper = active_limits()
    index = ACTIVE.index("finger31")
    sweep = np.linspace(lower[index], upper[index], 65)
    hands = {}
    for side in HANDS:
        geometry = load_geometry(project, side=side)
        layout = geometry["reference"]["reference_layout"]
        top = layout["table_center"][2] + layout["table_size"][2] / 2
        radius = geometry["old_candidate"]["rod"]["radius_m"]
        prior = seated["hands"][side]
        q = np.asarray(prior["open_q_rad"]).copy()
        if not np.array_equal(q, lower):
            raise ValueError("original open Power joint seed required")
        delta = np.zeros_like(lower)
        delta[index] = upper[index] - lower[index]
        pad = "cooked:finger33_link"
        reserve = path_displacement_bound(geometry, q, delta, cooked=True)["finger33_link"] / (2*(len(sweep)-1))
        hands[side] = []
        for candidate in prior["candidates"]:
            if not candidate["eligible"]:
                continue
            root = np.asarray(candidate["root"])
            samples = []
            for angle in sweep:
                q[index] = angle
                digit = {key: shape for key, shape in shapes_world(geometry, root, q).items()
                         if key.split(":")[1].startswith("finger3")}
                gap = _pad_gap_interval(digit[pad], nodes, radius)
                table = min(float(vertices[:, 2].min() - top) for vertices in digit.values())
                hits, support = support_clearance(digit, supports)
                rod_ok = True
                for vertices in digit.values():
                    g = _pad_gap_interval(vertices, nodes, radius)
                    if g["lower_m"] < 0.:
                        g = endpoint_gap(vertices, nodes, radius) if g["upper_m"] >= 0. else g
                    rod_ok &= g["lower_m"] >= 0.
                    if not rod_ok:
                        break
                samples.append(dict(angle_rad=float(angle), gap_lower_m=gap["lower_m"], gap_upper_m=gap["upper_m"],
                                     table_gap_m=table, support_gap_m=support, rod_clear=bool(rod_ok),
                                     clear=bool(rod_ok and table >= CLEARANCE_M and not hits and support >= CLEARANCE_M)))
            bounds = summarize_middle_reach(samples, point_travel_m=reserve)
            q[index] = min(samples, key=lambda row: row["gap_upper_m"])["angle_rad"]
            tight_upper = endpoint_gap(shapes_world(geometry, root, q)[pad], nodes, radius)["upper_m"]
            hands[side].append(dict(station=candidate["station"], mode=candidate["mode"],
                                    degrees=candidate.get("degrees"), continuous_bounds=bounds,
                                    best_coarse_sample_tight_upper_m=tight_upper,
                                    point_travel_m=reserve, samples=samples))
        if not hands[side]:
            raise ValueError("seated evidence contains no eligible wrist for " + side)
    return dict(hands=hands, scoped_digit="finger31 → finger33_link", native_contact_verified=False,
                teacher_admission=False, actual_isaac_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-evidence", type=Path, required=True)
    parser.add_argument("--closure-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resolve-closure", action="store_true",
                        help="re-audit all six active joints at the reference and best distinct wrist")
    mode.add_argument("--palm-approach-only", action="store_true",
                      help="screen a single inward axis with the hand open before attempting closure")
    mode.add_argument("--screen-seated-pockets", action="store_true",
                      help="change the left material station and rotate the right wrist about its palm witness")
    mode.add_argument("--screen-middle-reach", action="store_true")
    parser.add_argument("--seated-evidence", type=Path)
    parser.add_argument("--seated-sha256")
    args = parser.parse_args()
    artifact = checked_json(args.closure_evidence, args.closure_sha256)
    project = Path(__file__).resolve().parents[1]
    if args.screen_middle_reach:
        if not args.seated_evidence or not args.seated_sha256:
            parser.error("middle reach requires a pinned seated-pocket result")
        seated = checked_json(args.seated_evidence, args.seated_sha256)
        if seated["input_sha256"] != args.closure_sha256:
            raise ValueError("seated evidence binds a different closure source")
        result = middle_reach(artifact, seated, project=project)
        result["seated_sha256"] = args.seated_sha256
    else:
        if args.seated_evidence or args.seated_sha256:
            parser.error("seated evidence applies only to middle reach")
        result = audit_reach(artifact, project=project, resolve_closure=args.resolve_closure,
                             palm_approach_only=args.palm_approach_only, seated_pockets=args.screen_seated_pockets)
    result["input_sha256"] = args.closure_sha256
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
