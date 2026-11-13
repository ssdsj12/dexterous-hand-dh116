#!/usr/bin/env python3
"""Bounded independent Tripod/Power geometry preparation; no native or training.

All candidate generation is a fixed finite grid. The reused convex-QP routine
measures distances only; it never changes a pose, parameter or training weight.
Old seeds, physics, limits and negative archive qualification remain untouched.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.audit_dh116_pinch_initial_pose import (
    EVIDENCE, checked_json, convex_patch_closest, digest, finite_array, load_geometry,
    path_displacement_bound, pose_matrix, settling_gap, world_vertices,
)
from scripts.audit_dh116_flexible_power_pose import load_power_archive
from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, active_limits, expand

VERSION = "dh116-flexible-style-pose-preparation-v1"
TRIPOD_SHA = "c36e8a06c9ff227f7cfb32477dd83de3c4ab425092e1397894445d47a5f1053b"
POWER_SHA = "e5e040251c218dc90d198721b316b0a2e0bb4fec4c60b1f373de518ba8146c39"
POWER_ARCHIVE = EVIDENCE/"teacher-wrap-diverse-search-20261112-Ey7e0j/probe.json"
AUDIT_SHA = "15ede98e092a911470bf858f6d37e2a01007f05e3dbdb8ab2178ed95f6712db8"
MARGIN = .0019
CONSTRUCTION_CLEARANCE = .002
DURATION_S = 160/480
MAX_ACTIVE_DELTA = .3*DURATION_S
CLOSURE_SAMPLES = 9
PATH_INTERVALS = 160
REACH_TOLERANCE = .0005
ROLL_GRID = (-20., 0., 20.)
PITCH_GRID = (-30., -15., 0., 15., 30.)
CLOSURE_BIAS_GRID = (0., .05, .10)
X_SHIFT_GRID = (-.004, -.002, 0., .002, .004)


def load_tripod_seed(path):
    seed = checked_json(Path(path), TRIPOD_SHA)
    if seed["diagnostics"]["joint_order"] != list(JOINTS):
        raise ValueError("Tripod canonical joint order differs")
    hands = {}
    for i, side in enumerate(("left", "right")):
        full = finite_array(seed["joint_positions"][i], (11,))
        q = np.array([full[JOINTS.index(name)] for name in ACTIVE])
        if not np.allclose(expand(q), full, atol=1e-10, rtol=0):
            raise ValueError("Tripod seed violates original mimic mapping")
        pose = finite_array(seed["roots"][i], (7,))
        pose_matrix(pose)
        hands[side] = {"base_pose_wxyz": pose.tolist(), "active_q_rad": q.tolist()}
    return {"source_sha256": TRIPOD_SHA, "source_contract": seed["source_contract"], "hands": hands}


def rotate_seed_pose(pose, pivot, roll_deg, pitch_deg):
    pose, pivot = finite_array(pose, (7,)), finite_array(pivot, (3,))
    if not np.isfinite([roll_deg, pitch_deg]).all():
        raise ValueError("finite rotation required")
    delta = Rotation.from_euler("y", pitch_deg, degrees=True)*Rotation.from_euler("x", roll_deg, degrees=True)
    matrix = pose_matrix(pose)
    return np.r_[pivot+delta.apply(pose[:3]-pivot),
                 np.roll((delta*Rotation.from_matrix(matrix[:3, :3])).as_quat(), 1)].tolist()


def close_samples(q, channel):
    q = finite_array(q, (6,))
    expand(q)
    if type(channel) is not int or not 0 <= channel < 6:
        raise ValueError("canonical active channel required")
    upper = active_limits()[1]
    amount = min(MAX_ACTIVE_DELTA, upper[channel]-q[channel])
    return np.linspace(0, amount, CLOSURE_SAMPLES)


def _all_world(geometry, pose, q):
    return (world_vertices(geometry, pose, q),
            world_vertices(geometry, pose, q, cooked=True))


def clearance_corrected_pose(geometry, pose, q):
    """Translate the independent seed vertically using all 17 source + 5 cooked."""
    top = sum((geometry["reference"]["reference_layout"]["table_center"][2],
               geometry["reference"]["reference_layout"]["table_size"][2]/2))
    source, cooked = _all_world(geometry, pose, q)
    lowest = min(v[:, 2].min()-top for v in [*source.values(), *cooked.values()])
    shift = CONSTRUCTION_CLEARANCE-lowest
    adjusted = finite_array(pose, (7,)).copy()
    adjusted[2] += shift
    return {"base_pose_wxyz": adjusted.tolist(), "vertical_shift_m": float(shift),
            "original_minimum_clearance_m": float(lowest),
            "minimum_clearance_m": float(lowest+shift)}


def tripod_candidates(geometry, seed):
    side = geometry["side"]
    pivot = [.43, .2 if side == "left" else -.2, .308]
    for roll, pitch, bias in itertools.product(ROLL_GRID, PITCH_GRID, CLOSURE_BIAS_GRID):
        q = np.asarray(seed["active_q_rad"], float).copy()
        q[1:4] += bias
        expand(q)
        pose = rotate_seed_pose(seed["base_pose_wxyz"], pivot, roll, pitch)
        construction = clearance_corrected_pose(geometry, pose, q)
        for xshift in X_SHIFT_GRID:
            pose = np.asarray(construction["base_pose_wxyz"]).copy()
            pose[0] += xshift
            yield {"candidate_id": f"tripod-{side}-r{roll:g}-p{pitch:g}-b{bias:g}-x{xshift:g}",
                "independent_style": "tripod", "side": side,
                "independent_seed_sha256": TRIPOD_SHA,
                "base_pose_wxyz": pose.tolist(), "active_joint_names": list(ACTIVE),
                "active_q_rad": q.tolist(), "construction": {
                    **construction, "roll_deg": roll, "pitch_deg": pitch,
                    "closure_bias_rad": bias, "world_x_shift_m": xshift}}


def _settled_closest(vertices, layout, radius):
    nodes = np.array(layout["wire_points"], float)
    nodes[:, 2] = layout["table_center"][2]+layout["table_size"][2]/2+radius
    closest = convex_patch_closest(vertices, nodes[0], [nodes[-1]-nodes[0]])
    return {**closest, "capsule_surface_gap_m": closest["distance_m"]-radius}


def measure_closure_grid(geometry, candidate, channels):
    """Select from nine measured angles independently for each requested digit."""
    pose, q = candidate["base_pose_wxyz"], np.asarray(candidate["active_q_rad"])
    layout = geometry["reference"]["reference_layout"]
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    required = [DISTAL_PAD_LINKS[channel-1] for channel in channels]
    cooked = world_vertices(geometry, pose, q, cooked=True)
    initial = {name: settling_gap(v, layout, radius)["capsule_surface_gap_m"]
               for name, v in cooked.items()}
    chosen, curves, delta = {}, {}, np.zeros(6)
    for channel, name in zip(channels, required):
        samples = []
        for amount in close_samples(q, channel):
            moved = q.copy()
            moved[channel] += amount
            v = world_vertices(geometry, pose, moved, cooked=True)[name]
            record = _settled_closest(v, layout, radius)
            samples.append({"active_delta_rad": float(amount), **record})
        # The grid is evaluated in full, including nonmonotone distance curves.
        selected = min(samples, key=lambda row: (abs(row["capsule_surface_gap_m"]),
                                                 row["active_delta_rad"]))
        curves[name] = samples
        chosen[name] = selected
        delta[channel] = selected["active_delta_rad"]
    end_source, end_cooked = _all_world(geometry, pose, q+delta)
    top = layout["table_center"][2]+layout["table_size"][2]/2
    end_clear = min(v[:, 2].min()-top for v in [*end_source.values(), *end_cooked.values()])
    dots = {}
    thumb = chosen[required[0]]["geometric_axis_to_hull_direction"]
    for name in required[1:]:
        direction = chosen[name]["geometric_axis_to_hull_direction"]
        dots[name] = float(np.dot(thumb, direction)) if thumb is not None and direction is not None else None
    maximum_gap = max(abs(row["capsule_surface_gap_m"]) for row in chosen.values())
    blockers = []
    if min(initial.values()) < 0:
        blockers.append("initial_cooked_pad_intersects_rod_settling_envelope")
    if end_clear < MARGIN:
        blockers.append("endpoint_table_clearance_below_margin")
    if maximum_gap > REACH_TOLERANCE:
        blockers.append("required_cooked_pad_gap_exceeds_0p5mm")
    if any(dot is None or dot >= -.2 for dot in dots.values()):
        blockers.append("required_geometric_directions_not_opposed")
    return {**candidate, "active_delta_rad": delta.tolist(),
        "command_duration_s": DURATION_S, "active_velocity_rad_s": (delta/DURATION_S).tolist(),
        "required_pads": required, "initial_cooked_settling_gap_m": initial,
        "closure_grid": curves, "selected_cooked_endpoint": chosen,
        "max_required_endpoint_abs_gap_m": maximum_gap,
        "geometric_thumb_partner_dot": dots, "endpoint_all_shape_table_clearance_m": float(end_clear),
        "coarse_blockers": blockers, "coarse_geometry_passed": not blockers,
        "native_ready": False, "teacher_admitted": False}


def verify_candidate_path(geometry, candidate):
    """Independent source/cooked path and unregistered-link rod clearance gate."""
    pose, q = candidate["base_pose_wxyz"], np.asarray(candidate["active_q_rad"])
    delta = np.asarray(candidate["active_delta_rad"])
    expand(q+delta)
    if np.any(delta < 0) or np.max(delta) > MAX_ACTIVE_DELTA+1e-12:
        raise ValueError("path exceeds unchanged 0.3 rad/s active velocity limit")
    layout = geometry["reference"]["reference_layout"]
    radius = geometry["old_candidate"]["rod"]["radius_m"]
    top = layout["table_center"][2]+layout["table_size"][2]/2
    required = set(candidate["required_pads"])
    initial_source, initial_cooked = _all_world(geometry, pose, q)
    initial_gaps = {"source": {n: settling_gap(v, layout, radius)["capsule_surface_gap_m"]
                              for n, v in initial_source.items()},
                    "cooked": {n: settling_gap(v, layout, radius)["capsule_surface_gap_m"]
                              for n, v in initial_cooked.items()}}
    bounds = {"source": path_displacement_bound(geometry, q, delta),
              "cooked": path_displacement_bound(geometry, q, delta, cooked=True)}
    table_min = {"source": {}, "cooked": {}}
    other_min = {"source": {}, "cooked": {}}
    y_bounds = [float("inf"), -float("inf")]
    for fraction in np.linspace(0, 1, PATH_INTERVALS+1):
        sets = dict(zip(("source", "cooked"), _all_world(geometry, pose, q+fraction*delta)))
        for kind, shapes in sets.items():
            for name, v in shapes.items():
                margin = bounds[kind][name]/(2*PATH_INTERVALS)
                y_bounds[0] = min(y_bounds[0], float(v[:, 1].min())-margin)
                y_bounds[1] = max(y_bounds[1], float(v[:, 1].max())+margin)
                table_min[kind][name] = min(table_min[kind].get(name, float("inf")),
                                             float(v[:, 2].min()-top))
                if name not in required:
                    gap = settling_gap(v, layout, radius)["capsule_surface_gap_m"]
                    other_min[kind][name] = min(other_min[kind].get(name, float("inf")), gap)
    table_lower = {kind: {n: value-bounds[kind][n]/(2*PATH_INTERVALS)
                        for n, value in values.items()} for kind, values in table_min.items()}
    other_lower = {kind: {n: value-bounds[kind][n]/(2*PATH_INTERVALS)-2e-6
                        for n, value in values.items()} for kind, values in other_min.items()}
    initial_min_table = min(v[:, 2].min()-top for v in [*initial_source.values(), *initial_cooked.values()])
    reset_clear = (initial_min_table >= MARGIN and
                   min(value for rows in initial_gaps.values() for value in rows.values()) >= 0)
    blockers = list(candidate["coarse_blockers"])
    if min(value for rows in initial_gaps.values() for value in rows.values()) < 0:
        blockers.append("initial_source_or_cooked_rod_overlap")
    if min(value for rows in table_lower.values() for value in rows.values()) < MARGIN:
        blockers.append("continuous_source_or_cooked_table_clearance")
    if min(value for rows in other_lower.values() for value in rows.values()) < 0:
        blockers.append("non_required_collider_rod_swept_envelope_overlap")
    return {"candidate_id": candidate["candidate_id"],
        "sample_count": PATH_INTERVALS+1, "initial_rod_gaps_m": initial_gaps,
        "initial_all_shape_table_clearance_m": float(initial_min_table),
        "reset_geometry_passed": bool(reset_clear),
        "reset_is_native_verified": False,
        "continuous_table_clearance_lower_bound_m": table_lower,
        "continuous_nonrequired_rod_gap_lower_bound_m": other_lower,
        "continuous_hand_y_bounds_m": y_bounds, "geometry_preparation_passed": not blockers,
        "blockers": blockers, "effective_opposed_contact": None,
        "native_ready": False, "teacher_admitted": False}


def _rank(row):
    return (len(row["coarse_blockers"]), row["max_required_endpoint_abs_gap_m"],
            sum(abs(v["capsule_surface_gap_m"]) for v in row["selected_cooked_endpoint"].values()),
            row["candidate_id"])


def select_prepared_candidate(detailed):
    if not detailed:
        raise ValueError("at least one measured candidate required")
    return min(detailed, key=lambda item: (
        not item["path"]["geometry_preparation_passed"],
        not item["path"]["reset_geometry_passed"],
        len(item["path"]["blockers"]),
        item["candidate"]["max_required_endpoint_abs_gap_m"],
        item["candidate"]["candidate_id"]))


def prepare_tripod(project, *, progress=None):
    seed = load_tripod_seed(project/"configs/dh116_training/stereo_tripod_reset_20261105_v1.json")
    sides = {}
    for side in ("left", "right"):
        g = load_geometry(project, side=side)
        if (seed["source_contract"]["source_sha256"][side] != g["usd"]["sha256"]
                or seed["source_contract"]["pose_sha256"] != g["source_sha256"]["pose_reference"]):
            raise ValueError("Tripod source identity differs")
        rows = []
        for i, candidate in enumerate(tripod_candidates(g, seed["hands"][side]), 1):
            rows.append(measure_closure_grid(g, candidate, (1, 2, 3)))
            if progress and i % 25 == 0:
                progress({"style": "tripod", "side": side, "measured_candidates": i, "cap": 225})
        ranked = sorted(rows, key=_rank)
        detailed = []
        for row in ranked[:3]:
            detail = verify_candidate_path(g, row)
            detailed.append({"candidate": row, "path": detail})
            if detail["geometry_preparation_passed"]:
                break
        selected = select_prepared_candidate(detailed)
        sides[side] = {"source_sha256": g["source_sha256"],
            "candidate_count": len(rows), "distance_samples_per_required_digit": CLOSURE_SAMPLES,
            "coarse_geometry_pass_count": sum(r["coarse_geometry_passed"] for r in rows),
            "selected": selected, "detailed_candidates": detailed,
            "all_candidate_summaries": [{k: v for k, v in row.items() if k != "closure_grid"} for row in rows]}
    left = sides["left"]["selected"]["path"]; right = sides["right"]["selected"]["path"]
    y_gap = left["continuous_hand_y_bounds_m"][0]-right["continuous_hand_y_bounds_m"][1]
    passed = bool(left["geometry_preparation_passed"] and right["geometry_preparation_passed"] and y_gap > 0)
    return {"independent_style": "tripod", "seed_sha256": TRIPOD_SHA,
        "enumeration": {"roll_deg": ROLL_GRID, "pitch_deg": PITCH_GRID,
            "closure_bias_rad": CLOSURE_BIAS_GRID, "world_x_shift_m": X_SHIFT_GRID,
            "maximum_candidates_per_side": 225, "maximum_detailed_paths_per_side": 3,
            "optimizer_used_for_pose_search": False},
        "sides": sides, "bilateral_y_separation_lower_bound_m": y_gap,
        "bilateral_static_reset_geometry_passed": bool(
            left["reset_geometry_passed"] and right["reset_geometry_passed"] and y_gap > 0),
        "bilateral_geometry_preparation_passed": passed,
        "native_ready": False, "teacher_admitted": False}


def fixed_root_gap_lower_bounds(gaps, travel_bounds):
    if set(gaps) != set(travel_bounds) or not gaps:
        raise ValueError("named required-pad distances and displacement bounds must agree")
    lower = {name: float(gap-travel_bounds[name]-2e-6) for name, gap in gaps.items()}
    return {"gap_lower_bound_m": lower,
        "all_required_pass_necessary_reach_bound": all(value <= REACH_TOLERANCE for value in lower.values()),
        "meaning": "necessary_condition_only_for_fixed_root_monotone_positive_closure",
        "native_ready": False, "teacher_admitted": False}


def prepare_power(project, archive=POWER_ARCHIVE):
    sides = {}
    for side, rows in load_power_archive(archive, POWER_SHA).items():
        g = load_geometry(project, side=side)
        layout = g["reference"]["reference_layout"]; radius = g["old_candidate"]["rod"]["radius_m"]
        records = []
        for row in rows:
            q = np.asarray(row["active_angles"])
            expand(q)
            shifted = clearance_corrected_pose(g, row["wrist_pose_w"], q)
            for mode, pose in (("archived_root", row["wrist_pose_w"]),
                               ("minimum_table_clearance_root", shifted["base_pose_wxyz"])):
                delta = np.minimum(MAX_ACTIVE_DELTA, active_limits()[1]-q)
                bounds = path_displacement_bound(g, q, delta, cooked=True)
                cooked = world_vertices(g, pose, q, cooked=True)
                gaps = {n: _settled_closest(v, layout, radius)["capsule_surface_gap_m"]
                        for n, v in cooked.items()}
                lower = fixed_root_gap_lower_bounds(gaps, bounds)
                source, cooked = _all_world(g, pose, q)
                top = layout["table_center"][2]+layout["table_size"][2]/2
                initial_rod_gaps = {kind: {n: settling_gap(v, layout, radius)["capsule_surface_gap_m"]
                    for n, v in shapes.items()} for kind, shapes in (("source", source), ("cooked", cooked))}
                records.append({"archive_start_index": row["start_index"], "mode": mode,
                    "base_pose_wxyz": pose, "active_joint_names": list(ACTIVE),
                    "active_q_rad": q.tolist(), "maximum_positive_closure_rad": delta.tolist(),
                    "closure_duration_s": DURATION_S, "vertical_adjustment_m": 0. if mode == "archived_root" else shifted["vertical_shift_m"],
                    "initial_all_shape_table_clearance_m": float(min(v[:,2].min()-top for v in [*source.values(), *cooked.values()])),
                    "initial_rod_settling_gaps_m": initial_rod_gaps,
                    "initial_cooked_settled_rod_gaps_m": gaps,
                    "maximum_cooked_point_travel_bound_m": bounds,
                    **lower, "archived_wrap_qualified": False})
        probes = []
        possible = [r for r in records if r["all_required_pass_necessary_reach_bound"]]
        for row in possible[:4]:
            entry = {"archive_start_index": row["archive_start_index"], "mode": row["mode"]}
            if min(v for shapes in row["initial_rod_settling_gaps_m"].values()
                   for v in shapes.values()) < 0:
                probes.append({**entry, "grid_performed": False,
                    "blocked_reasons": ["initial_source_or_cooked_rod_overlap"]})
                continue
            candidate = {"candidate_id": f'power-{side}-{row["archive_start_index"]}-{row["mode"]}',
                "independent_style": "power", "side": side, "independent_seed_sha256": POWER_SHA,
                "base_pose_wxyz": row["base_pose_wxyz"], "active_joint_names": list(ACTIVE),
                "active_q_rad": row["active_q_rad"]}
            measured = measure_closure_grid(g, candidate, (1, 2, 3, 4, 5))
            entry.update(grid_performed=True, candidate=measured)
            if measured["coarse_geometry_passed"]:
                entry["path"] = verify_candidate_path(g, measured)
            probes.append(entry)
        sides[side] = {"source_sha256": g["source_sha256"], "records": records,
            "fixed_root_rows_with_necessary_all_pad_reach": len(possible),
            "maximum_fixed_root_closure_grid_probes": 4, "closure_grid_probes": probes}
    return {"independent_style": "power", "archive_sha256": POWER_SHA,
        "maximum_records_per_side": 32, "sides": sides,
        "new_wrist_rotation_search_performed": False,
        "blockers": ["No archived Power qualification is promoted.",
            "Fixed-root positive joint closure and minimum-table vertical placement are only a bounded necessary-condition screen.",
            "Other wrist rotations, translated approach paths and live contact remain unverified."],
        "native_ready": False, "teacher_admitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=("tripod", "power", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if digest(project/"scripts/audit_dh116_pinch_initial_pose.py") != AUDIT_SHA:
        raise ValueError("locked reusable geometry audit SHA256 differs")
    result = {"version": VERSION, "scope": "independent_bounded_CPU_geometry_only",
        "preparation_source_sha256": digest(Path(__file__)), "geometry_audit_sha256": AUDIT_SHA,
        "native_execution_performed": False, "physics_steps": 0, "PPO_updates": 0,
        "material_task_NA_unchanged": True, "native_recipe_created": False}
    if args.style in ("tripod", "both"):
        result["tripod"] = prepare_tripod(project, progress=lambda row: print(json.dumps(row), flush=True))
    if args.style in ("power", "both"):
        result["power"] = prepare_power(project)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output), "native_ready": False}))


if __name__ == "__main__":
    main()
